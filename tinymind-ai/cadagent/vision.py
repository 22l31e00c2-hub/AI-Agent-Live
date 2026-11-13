"""Local image inspection: what is in this picture, without a pretrained model.

Nothing here is a neural network. The report is measurement - format, size,
colour, ink coverage, edge density, EXIF - plus line-level OCR with confidence
when the tesseract binary is installed. That makes it useful for QA of drawings
and scans (is this sheet blank? is it linework or a photo? which EPSG-able
metadata does the photo carry?) and honest about what it cannot do (understand
semantics).

Pillow is optional: without it, amu explains how to install it instead of
failing with an ImportError.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ocr

try:                                   # Pillow is optional
    from PIL import Image, ImageFilter, ImageStat
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except Exception:                      # pragma: no cover - import guard
    Image = None
    ImageFilter = None
    ImageStat = None
    TAGS = {}
    PIL_AVAILABLE = False

IMAGE_EXTS = "png|jpe?g|tif{1,2}|bmp|webp|gif"

IMAGE_PATH_RE = re.compile(
    r"(?<![\w])([A-Za-z0-9_.\-:\\/]*[A-Za-z0-9_.\-]\.(?:"
    + IMAGE_EXTS
    + r"))(?![\w])",
    re.I,
)

GPS_TAG = 34853


class VisionError(Exception):
    """Any reason an image could not be inspected."""


@dataclass
class ImageReport:
    path: str
    format: str
    mode: str
    width: int
    height: int
    mean_brightness: float
    contrast: float
    ink_ratio: float
    edge_ratio: float
    verdict: str
    has_alpha: bool = False
    content_box: Optional[Tuple[int, int, int, int]] = None
    margins: Dict[str, float] = field(default_factory=dict)
    colours: List[Tuple[str, float]] = field(default_factory=list)
    exif: Dict[str, str] = field(default_factory=dict)
    ocr_text: str = ""
    ocr_lines: int = 0
    ocr_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000.0


def find_image_paths(text: str) -> List[str]:
    found: List[str] = []
    for match in IMAGE_PATH_RE.finditer(text or ""):
        path = match.group(1).strip().strip("\"'").rstrip(".,;:)]")
        if path and path not in found:
            found.append(path)
    return found


def _colours(handle, limit: int = 5) -> List[Tuple[str, float]]:
    small = handle.convert("RGB").resize((64, 64))
    counts = small.getcolors(maxcolors=64 * 64) or []
    counts.sort(reverse=True)
    total = float(sum(count for count, _ in counts)) or 1.0
    return [
        ("#%02x%02x%02x" % rgb, count / total)
        for count, rgb in counts[:limit]
    ]


def _exif(handle) -> Dict[str, str]:
    try:
        data = handle.getexif()
    except Exception:
        return {}
    if not data:
        return {}

    summary: Dict[str, str] = {}
    for tag_id in list(data.keys())[:20]:
        name = TAGS.get(tag_id, str(tag_id))
        try:
            value = data.get(tag_id)
        except Exception:
            continue
        summary[str(name)] = str(value)[:80]
    return summary


def inspect(path: str, run_ocr: bool = True) -> ImageReport:
    """Measure an image and describe what kind of content it holds."""
    if not PIL_AVAILABLE:
        raise VisionError(
            "Pillow is not installed, so I cannot read images. "
            "Run: pip install pillow"
        )
    if not os.path.isfile(path):
        raise VisionError("file not found: %s" % path)

    try:
        with Image.open(path) as handle:
            handle.load()
            fmt = (handle.format or "unknown").upper()
            mode = handle.mode
            width, height = handle.size
            has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in handle.info

            gray = handle.convert("L")
            stats = ImageStat.Stat(gray)
            mean = float(stats.mean[0])
            contrast = float(stats.stddev[0])

            histogram = gray.histogram()
            total = float(sum(histogram)) or 1.0
            ink = sum(histogram[:100]) / total

            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_histogram = edges.histogram()
            edge_total = float(sum(edge_histogram)) or 1.0
            edge_ratio = sum(edge_histogram[45:]) / edge_total

            colours = _colours(handle)
            exif = _exif(handle)

            mask = gray.point(lambda value: 255 if value < 200 else 0)
            box = mask.getbbox()
    except VisionError:
        raise
    except Exception as exc:
        raise VisionError("could not decode %s: %s" % (path, exc))

    margins: Dict[str, float] = {}
    if box:
        left, top, right, bottom = box
        margins = {
            "left": left / float(width),
            "top": top / float(height),
            "right": (width - right) / float(width),
            "bottom": (height - bottom) / float(height),
        }

    notes: List[str] = []
    if ink < 0.005 and contrast < 6:
        verdict = "visually blank - almost no dark pixels and no variation"
        notes.append("blank or near-blank image, nothing to read")
    elif ink <= 0.30 and edge_ratio > 0.005:
        verdict = "thin, line-like content - typical of a drawing, plan, scan or diagram"
        notes.append(
            "mostly background with thin marks: text and linework read well, "
            "photos do not look like this"
        )
    elif ink > 0.60:
        verdict = "dark, low-key image - inverted colours or a night capture"
    else:
        verdict = "continuous-tone content - looks like a photo or a render"

    if GPS_TAG in exif or "GPSInfo" in exif:
        notes.append(
            "carries GPS metadata - usable for georeferencing in QGIS "
            "(see the georeferencer plugin)"
        )

    if box and min(margins.values()) < 0.005:
        notes.append(
            "content runs to the image edge - the sheet may be cropped, or the "
            "plot window was tighter than the page"
        )

    text = ""
    line_count = 0
    confidence = 0.0
    if run_ocr:
        result = ocr.read(path)
        text = result.text
        line_count = len(result.lines)
        confidence = result.confidence
        if result.reason.startswith("no text"):
            notes.append("OCR: no text found in this image")
        elif result.reason:
            notes.append("OCR unavailable: %s" % result.reason)

    return ImageReport(
        path=path,
        format=fmt,
        mode=mode,
        width=width,
        height=height,
        mean_brightness=mean,
        contrast=contrast,
        ink_ratio=ink,
        edge_ratio=edge_ratio,
        verdict=verdict,
        has_alpha=has_alpha,
        content_box=box,
        margins=margins,
        colours=colours,
        exif=exif,
        ocr_text=text,
        ocr_lines=line_count,
        ocr_confidence=confidence,
        notes=notes,
    )


def describe(info: ImageReport) -> str:
    """Render an ImageReport as the text amu prints."""
    lines = [
        "amu vision | %s" % os.path.basename(info.path),
        "file     : %s" % info.path,
        "format   : %s %dx%d (%.2f MP), mode %s%s" % (
            info.format, info.width, info.height, info.megapixels,
            info.mode, " with alpha" if info.has_alpha else "",
        ),
        "content  : %s" % info.verdict,
        "ink      : %.2f%% dark pixels, edge density %.2f%%" % (
            info.ink_ratio * 100.0, info.edge_ratio * 100.0,
        ),
        "contrast : mean %.1f, std %.1f (0-255 greyscale)" % (
            info.mean_brightness, info.contrast,
        ),
    ]

    if info.content_box:
        box = info.content_box
        lines.append(
            "ink box  : x %d-%d, y %d-%d; margins L %.1f%% R %.1f%% T %.1f%% B %.1f%%"
            % (
                box[0], box[2], box[1], box[3],
                info.margins.get("left", 0.0) * 100.0,
                info.margins.get("right", 0.0) * 100.0,
                info.margins.get("top", 0.0) * 100.0,
                info.margins.get("bottom", 0.0) * 100.0,
            )
        )

    if info.colours:
        lines.append(
            "colours  : %s"
            % ", ".join("%s %.0f%%" % (code, share * 100.0) for code, share in info.colours)
        )

    if info.exif:
        lines.append(
            "exif     : %s"
            % ", ".join("%s=%s" % (key, value) for key, value in info.exif.items())
        )

    if info.ocr_text:
        preview = re.sub(r"\s+", " ", info.ocr_text)[:600]
        lines.append(
            "ocr      : %d lines, %d characters, mean confidence %.0f%%"
            % (info.ocr_lines, len(info.ocr_text), info.ocr_confidence)
        )
        lines.append("text     : %s" % preview)
        if info.ocr_confidence and info.ocr_confidence < 60.0:
            lines.append(
                "ocr note : low confidence; check rotation, contrast or try "
                "AMU_OCR_PSM=6"
            )
    elif "OCR unavailable" in " ".join(info.notes):
        lines.append("ocr      : not available")

    if info.notes:
        lines.append("notes    :")
        for note in info.notes:
            lines.append("    - %s" % note)

    lines.append("")
    lines.append(
        "limits   : measurements only. No pretrained vision model, so I describe "
        "structure and colour, not meaning."
    )
    return "\n".join(lines)
