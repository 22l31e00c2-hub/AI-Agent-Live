"""Dependency-free slot extraction from a natural-language request.

This is deliberately simple and explainable: regular expressions over the
request text. No model is involved, so the agent never silently invents a
file path or a coordinate system.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

EXTS = (
    "shp|gpkg|geojson|csv|dxf|dwg|rvt|tif|tiff|kml|gml|tab|sqlite"
    "|qgz|qgs|lsp|scr|txt|json"
)

QUOTED_PATH_RE = re.compile(r'"([^"\n]*\.(?:' + EXTS + r'))"', re.I)
BARE_PATH_RE = re.compile(
    r'(?<![\w])([A-Za-z0-9_.\-:]+(?:[\\/][A-Za-z0-9_.\- ]+)*\.(?:'
    + EXTS
    + r'))(?![\w])',
    re.I,
)

EPSG_RE = re.compile(r"(?:epsg|srid|esri)\s*[:=#\s]*\s*(\d{4,6})", re.I)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

LAYER_RE = re.compile(
    r"\blayers?\b[\s:=-]*(?:called|named|for|with)?\s*([^.;()\[\]]+)", re.I
)
LAYER_CUTS = (
    " with ", " using ", " in ", " at ", " on ", " as ", " from ", " to ",
    " colour", " color", " that ", " which ",
)

KV_RE = re.compile(
    r"\b(out|output|target|dest|destination|source|input|layer|layers|epsg|crs"
    r"|category|parameter|value|prefix|height|distance|folder|color|colour"
    r"|find|replace|pattern|block|scale)"
    r"\s*=\s*(\"[^\"]*\"|[^\s,;]+)",
    re.I,
)

COLOR_WORDS = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "cyan": 4,
    "blue": 5,
    "magenta": 6,
    "white": 7,
}

LAYER_NOISE = {
    "the", "a", "an", "new", "layer", "layers", "called", "named", "all",
    "each", "my", "this", "same", "color", "colour", "colour:", "linetype",
}


def _clean_path(raw: str) -> str:
    return raw.strip().strip('"\'').rstrip(".,;:)")


def find_paths(text: str) -> List[Tuple[str, int, int]]:
    """Return (path, start, end) for every file path mentioned."""
    found: List[Tuple[str, int, int]] = []
    seen_spans: List[Tuple[int, int]] = []

    for m in QUOTED_PATH_RE.finditer(text):
        found.append((_clean_path(m.group(1)), m.start(), m.end()))
        seen_spans.append(m.span())

    for m in BARE_PATH_RE.finditer(text):
        start, end = m.span()
        if any(a <= start < b or a < end <= b for a, b in seen_spans):
            continue
        found.append((_clean_path(m.group(1)), start, end))

    found.sort(key=lambda item: item[1])
    return found


def find_epsg(text: str) -> Optional[int]:
    m = EPSG_RE.search(text)
    return int(m.group(1)) if m else None


def _mask(text: str, spans: List[Tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


def find_numbers(text: str, masked_spans: List[Tuple[int, int]]) -> List[float]:
    """All standalone numbers, ignoring ones inside paths / EPSG codes."""
    stripped = _mask(text, masked_spans)
    values = []
    for m in NUM_RE.finditer(stripped):
        try:
            values.append(float(m.group(0)))
        except ValueError:
            continue
    return values


def find_quoted(text: str) -> List[str]:
    return [m.group(1) for m in re.finditer(r'"([^"\n]+)"', text)]


def guess_layers(text: str) -> List[str]:
    m = LAYER_RE.search(text)
    if not m:
        return []

    chunk = " " + m.group(1).strip()
    lowered = chunk.lower()
    for cut in LAYER_CUTS:
        pos = lowered.find(cut)
        if pos > 0:
            chunk = chunk[:pos]
            lowered = chunk.lower()

    items = re.split(r",|;|\band\b|&|\+|\|", chunk)
    layers: List[str] = []
    for item in items:
        name = item.strip().strip('"\'').strip()
        name = re.sub(r"\s+", "_", name)
        if not name or len(name) > 40:
            continue
        if name.lower().strip("_") in LAYER_NOISE:
            continue
        if name not in layers:
            layers.append(name)
    return layers


def find_kv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in KV_RE.finditer(text):
        key = m.group(1).lower()
        value = m.group(2).strip().strip('"')
        if key and value and key not in out:
            out[key] = value
    return out


def color_from_text(text: str) -> Optional[int]:
    lowered = text.lower()
    for word, code in COLOR_WORDS.items():
        if re.search(r"\b" + word + r"\b", lowered):
            return code
    return None


def scan(text: str) -> Dict[str, Any]:
    """Read everything we can out of a request, once."""
    paths = find_paths(text)
    epsg = find_epsg(text)
    spans = [(start, end) for _, start, end in paths]

    epsg_match = EPSG_RE.search(text)
    if epsg_match:
        spans.append(epsg_match.span())

    return {
        "text": text,
        "lower": text.lower(),
        "paths": [p for p, _, _ in paths],
        "epsg": epsg,
        "numbers": find_numbers(text, spans),
        "quoted": find_quoted(text),
        "kv": find_kv(text),
        "layers": guess_layers(text),
        "color": color_from_text(text),
    }
