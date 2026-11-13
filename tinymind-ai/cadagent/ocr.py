"""OCR for amu: readable text from scans and drawings, through tesseract.

This is a tool path, not a neural capability of the model: the network writes
text, the agent reads pixels. When pytesseract or the tesseract binary is
missing, the caller gets a reason instead of an exception, so an image report
still carries every measurement it could make.

What this adds over a plain image_to_string call:

* line-level output with a mean confidence per line, parsed from tesseract's
  dictionary output, so a bad scan is visible as a low number instead of
  silently wrong text;
* preprocessing for CAD sheets: grayscale, autocontrast, and a 2x upscale of
  images smaller than 1000 pixels on their longest side;
* language and page-segmentation mode from AMU_OCR_LANG / AMU_OCR_PSM, or
  passed directly by the caller.

Everything here is optional: without Pillow, pytesseract or the binary the
result simply carries a reason.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:                                   # Pillow is optional
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:                      # pragma: no cover - import guard
    Image = None
    ImageOps = None
    PIL_AVAILABLE = False

try:                                   # pytesseract is optional
    import pytesseract
    PY_AVAILABLE = True
except Exception:                      # pragma: no cover - import guard
    pytesseract = None
    PY_AVAILABLE = False

TESSERACT_BINARY = "tesseract"
OCR_TIMEOUT = 60


@dataclass
class OcrLine:
    text: str
    confidence: float


@dataclass
class OcrResult:
    text: str = ""
    lines: List[OcrLine] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


def available() -> bool:
    """True when both the Python binding and the tesseract binary exist."""
    return PIL_AVAILABLE and PY_AVAILABLE and shutil.which(TESSERACT_BINARY) is not None


def _column(data: Dict[str, List], name: str, default, total: int) -> List:
    values = data.get(name)
    if isinstance(values, list) and len(values) == total:
        return values
    return [default] * total


def parse_data(data: Dict[str, List]) -> OcrResult:
    """Turn tesseract's dictionary output into lines with confidences.

    Pure function: the tests feed it a hand-written dictionary, so parsing is
    checked without installing tesseract.
    """
    if not data:
        return OcrResult()

    words = data.get("text") or []
    total = len(words)

    pages = _column(data, "page_num", 1, total)
    blocks = _column(data, "block_num", 0, total)
    paragraphs = _column(data, "par_num", 0, total)
    line_numbers = _column(data, "line_num", 0, total)
    scores = _column(data, "conf", -1, total)

    grouped: Dict[tuple, List[tuple]] = {}

    for index, word in enumerate(words):
        word = str(word or "").strip()
        if not word:
            continue

        try:
            confidence = float(scores[index])
        except (TypeError, ValueError):
            confidence = -1.0

        key = (pages[index], blocks[index], paragraphs[index], line_numbers[index])
        grouped.setdefault(key, []).append((word, confidence))

    lines: List[OcrLine] = []
    for key in sorted(grouped):
        entries = grouped[key]
        measured = [score for _, score in entries if score >= 0]
        lines.append(OcrLine(
            text=" ".join(word for word, _ in entries),
            confidence=(sum(measured) / len(measured)) if measured else 0.0,
        ))

    text = "\n".join(line.text for line in lines)

    measured = [line.confidence for line in lines if line.confidence > 0]
    confidence = (sum(measured) / len(measured)) if measured else 0.0

    return OcrResult(text=text, lines=lines, confidence=confidence)


def prepare(image):
    """Grayscale, autocontrast and upscale a small image before OCR."""
    if ImageOps is None:
        return image

    prepared = image.convert("L")

    width, height = prepared.size
    if max(width, height) < 1000:
        prepared = prepared.resize((width * 2, height * 2))

    return ImageOps.autocontrast(prepared)


def _config(psm) -> str:
    if psm is None:
        psm = os.environ.get("AMU_OCR_PSM")
    if psm is None or str(psm).strip() == "":
        return ""
    return "--psm %s" % str(psm).strip()


def _language(lang) -> Optional[str]:
    if lang is not None:
        return lang
    value = os.environ.get("AMU_OCR_LANG")
    return value or None


def read(path: str, lang: Optional[str] = None, psm=None) -> OcrResult:
    """OCR one image. Never raises: a failure comes back as `reason`."""
    if not PIL_AVAILABLE:
        return OcrResult(reason="install pillow to read images")

    if not PY_AVAILABLE:
        return OcrResult(
            reason="install pytesseract and the tesseract binary to read text"
        )

    if shutil.which(TESSERACT_BINARY) is None:
        return OcrResult(reason="the tesseract binary is not on PATH")

    config = _config(psm)
    language = _language(lang)

    try:
        with Image.open(path) as handle:
            image = prepare(handle)

        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT,
        )
        result = parse_data(data)

        if result.text:
            return result

        text = pytesseract.image_to_string(
            image,
            lang=language,
            config=config,
            timeout=OCR_TIMEOUT,
        ).strip()

        if text:
            lines = [
                OcrLine(text=line.strip(), confidence=0.0)
                for line in text.splitlines() if line.strip()
            ]
            return OcrResult(text=text, lines=lines, confidence=0.0)

        return OcrResult(reason="no text found")

    except Exception as exc:
        return OcrResult(reason="OCR failed: %s" % exc)
