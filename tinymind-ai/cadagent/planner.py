"""Turn a request into a concrete plan: which recipe, with which parameters.

Routing is keyword/trigger scoring over the registry. It is deterministic and
inspectable: you can always ask the agent why it picked a recipe (``--explain``
prints the winning score and the runners-up).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .extract import scan
from .types import Param, Recipe

DOMAIN_LEXICON: Dict[str, List[str]] = {
    "revit": [
        "revit", "rvt", "bim", "family", "families", "workset", "schedule",
        "sheet", "sheets", "pyrevit", "dynamo", "ifc", "level", "levels",
        "wall", "walls", "parameter", "parameters", "titleblock", "view",
        "transaction", "document",
    ],
    "autocad": [
        "autocad", "acad", "dwg", "lisp", "autolisp", "layer", "layers",
        "polyline", "pline", "hatch", "block", "xref", "purge", "audit",
        "plot", "ctb", "script", "scr", "attribute", "linetype",
    ],
    "map3d": [
        "map 3d", "map3d", "autocad map", "fdo", "topology", "mapimport",
        "mapexport", "mapcsassign", "adesetcrdrsys", "coordinate system",
        "parcel fabric", "gis data",
    ],
    "qgis": [
        "qgis", "qgz", "pyqgis", "geopackage", "gpkg", "shapefile", "shp",
        "raster", "reproject", "buffer", "clip", "dissolve", "spatial join",
        "processing", "epsg", "crs", "ogr", "feature",
    ],
}

DOMAIN_ALIASES = {
    "revit": "revit",
    "rvt": "revit",
    "autocad": "autocad",
    "acad": "autocad",
    "dwg": "autocad",
    "map3d": "map3d",
    "map": "map3d",
    "map 3d": "map3d",
    "autocad map 3d": "map3d",
    "qgis": "qgis",
    "gis": "qgis",
}

MIN_SCORE = 4.0


def coerce_param(param: Param, value: Any) -> Any:
    """Turn extracted text / CLI strings into the type the recipe expects."""
    if isinstance(value, (list, tuple)) and param.kind not in ("numbers", "layers"):
        value = value[0] if value else None

    if param.kind == "numbers":
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return [float(part) for part in re.split(r"[,\s]+", str(value)) if part]

    if param.kind == "number":
        return float(value)

    if param.kind == "epsg":
        digits = re.sub(r"[^0-9]", "", str(value))
        if not digits:
            raise ValueError("no EPSG code in %r" % (value,))
        return int(digits)

    if param.kind == "layers":
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in re.split(r"[,;]", str(value)) if part.strip()]

    if param.kind == "flag":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    return str(value)


@dataclass
class Plan:
    request: str
    domain: Optional[str]
    recipe: Optional[Recipe]
    params: Dict[str, Any] = field(default_factory=dict)
    missing: List[Param] = field(default_factory=list)
    score: float = 0.0
    suggestions: List[Recipe] = field(default_factory=list)
    read: Dict[str, Any] = field(default_factory=dict)


class Planner:
    def __init__(self, recipes: Sequence[Recipe]):
        self.recipes = list(recipes)
        self.by_id = {r.id: r for r in self.recipes}

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------
    @staticmethod
    def detect_domain(lowered: str) -> Optional[str]:
        best: Optional[str] = None
        best_score = 0
        for domain, words in DOMAIN_LEXICON.items():
            score = sum(1 for w in words if w in lowered)
            if score > best_score:
                best, best_score = domain, score
        return best

    @staticmethod
    def score_recipe(lowered: str, recipe: Recipe) -> float:
        score = 0.0
        for trigger in recipe.triggers:
            if trigger in lowered:
                # multi-word triggers are more specific than single words
                score += 4.0 + (len(trigger.split()) - 1)
        for word in DOMAIN_LEXICON.get(recipe.domain, []):
            if word in lowered:
                score += 1.0
        return score

    def rank(self, request: str, domain: Optional[str] = None):
        lowered = request.lower()
        candidates = [
            r for r in self.recipes if domain is None or r.domain == domain
        ]
        ranked = sorted(
            ((self.score_recipe(lowered, r), r) for r in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_path(name: str, paths: List[str]) -> Optional[str]:
        if not paths:
            return None
        lowered = name.lower()
        if "csv" in lowered:
            for path in paths:
                if path.lower().endswith((".csv", ".txt")):
                    return path
        if any(word in lowered for word in ("out", "dest", "target", "export")):
            if len(paths) > 1:
                return paths[1]
            return None
        return paths[0]

    def fill_params(
        self,
        recipe: Recipe,
        data: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        overrides = {k.lower(): v for k, v in (overrides or {}).items()}
        kv = data["kv"]
        values: Dict[str, Any] = {}
        missing: List[Param] = []

        for param in recipe.params:
            name = param.name
            value = None

            if name in overrides:
                value = overrides[name]
            elif name in kv:
                value = kv[name]
            elif param.kind == "epsg":
                value = data["epsg"]
            elif param.kind == "path":
                value = self._pick_path(name, data["paths"])
            elif param.kind == "csv":
                value = self._pick_path("csv", data["paths"])
            elif param.kind == "layers":
                value = data["layers"] or None
            elif param.kind == "numbers":
                value = data["numbers"] or None
            elif param.kind == "number":
                if name in ("color", "colour"):
                    value = data["color"]
                if value is None and data["numbers"]:
                    value = data["numbers"][0]
            elif param.kind == "text" and name in ("layer", "layer_name") and data["layers"]:
                value = data["layers"][0]
            elif param.kind == "flag":
                value = None

            if value is not None:
                try:
                    value = coerce_param(param, value)
                except (TypeError, ValueError):
                    value = None

            if value is None:
                value = param.default

            if value is None and param.required:
                missing.append(param)
            elif value is not None:
                values[name] = value

        return values, missing

    # ------------------------------------------------------------------
    def plan(
        self,
        request: str,
        domain: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        data = scan(request)
        lowered = data["lower"]

        if domain:
            domain = DOMAIN_ALIASES.get(domain.lower().strip(), domain.lower())
        else:
            domain = self.detect_domain(lowered)

        ranked = self.rank(request, domain)
        if not ranked or ranked[0][0] < MIN_SCORE:
            return Plan(
                request=request,
                domain=domain,
                recipe=None,
                suggestions=[r for score, r in ranked[:3] if score > 0],
                read=data,
            )

        score, recipe = ranked[0]
        params, missing = self.fill_params(recipe, data, overrides)

        return Plan(
            request=request,
            domain=recipe.domain,
            recipe=recipe,
            params=params,
            missing=missing,
            score=score,
            suggestions=[r for s, r in ranked[1:4] if s >= MIN_SCORE],
            read=data,
        )
