"""QGIS recipes.

Each recipe ships two things: a standalone PyQGIS script (runs from a normal
Python interpreter with QGIS installed) and the equivalent ``qgis_process``
command, which is the only thing the agent will actually execute for you -
with explicit consent.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..types import Artifact, ExecuteSpec, Param, Recipe, ToolResult

QGIS_PREAMBLE = '''import os

from qgis.core import QgsApplication

prefix = os.environ.get("QGIS_PREFIX_PATH")
if prefix:
    QgsApplication.setPrefixPath(prefix, True)

app = QgsApplication([], False)
app.initQgis()

from qgis.analysis import QgsNativeAlgorithms
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

import processing
'''


def _derive(source: str, suffix: str, ext: str) -> str:
    base = os.path.splitext(os.path.basename(source))[0]
    return os.path.join("out", base + suffix + ext)


def _quote(value: str) -> str:
    return '"%s"' % value if (" " in value or not value) else value


# ----------------------------------------------------------------------
# 1. reproject
# ----------------------------------------------------------------------
REPROJECT = '''# amu: reproject a vector layer
# Run with QGIS installed: python reproject_layer.py
__PREAMBLE__
SOURCE = r"__SOURCE__"
OUTPUT = r"__OUTPUT__"
TARGET_CRS = "EPSG:__EPSG__"

try:
    result = processing.run("native:reprojectlayer", {
        "INPUT": SOURCE,
        "TARGET_CRS": TARGET_CRS,
        "OUTPUT": OUTPUT,
    })
    print("Wrote:", result["OUTPUT"])
finally:
    app.exitQgis()
'''


def build_reproject(p: Dict[str, Any]) -> ToolResult:
    source = str(p["source"])
    epsg = int(p["epsg"])
    out = str(p.get("out") or _derive(source, "_" + str(epsg), ".gpkg"))

    script = (
        REPROJECT.replace("__PREAMBLE__", QGIS_PREAMBLE)
        .replace("__SOURCE__", source)
        .replace("__OUTPUT__", out)
        .replace("__EPSG__", str(epsg))
    )

    argv = [
        "qgis_process", "run", "native:reprojectlayer", "--",
        "INPUT=%s" % _quote(source),
        "TARGET_CRS=EPSG:%d" % epsg,
        "OUTPUT=%s" % _quote(out),
    ]

    return ToolResult(
        title="Reproject %s to EPSG:%d" % (os.path.basename(source), epsg),
        artifacts=[Artifact("reproject_layer.py", "python", script)],
        commands=[" ".join(argv)],
        notes=[
            "The output keeps the same geometry type and attributes.",
            "QGIS writes a GeoPackage for .gpkg paths and a Shapefile for .shp.",
            "Add more target codes with --param out=... to keep several copies.",
        ],
        execute=ExecuteSpec(argv=argv, description="QGIS reproject layer"),
    )


# ----------------------------------------------------------------------
# 2. buffer + clip
# ----------------------------------------------------------------------
BUFFER_CLIP = '''# amu: buffer a layer and clip it to a boundary
# Run with QGIS installed: python buffer_clip.py
__PREAMBLE__
SOURCE = r"__SOURCE__"
OVERLAY = r"__OVERLAY__"
DISTANCE = __DISTANCE__
BUFFERED = r"__BUFFERED__"
OUTPUT = r"__OUTPUT__"

try:
    buffered = processing.run("native:buffer", {
        "INPUT": SOURCE,
        "DISTANCE": DISTANCE,
        "SEGMENTS": 5,
        "END_CAP_STYLE": 0,
        "JOIN_STYLE": 0,
        "MITER_LIMIT": 2,
        "DISSOLVE": False,
        "OUTPUT": BUFFERED,
    })
    print("Buffered:", buffered["OUTPUT"])

    clipped = processing.run("native:clip", {
        "INPUT": buffered["OUTPUT"],
        "OVERLAY": OVERLAY,
        "OUTPUT": OUTPUT,
    })
    print("Wrote:", clipped["OUTPUT"])
finally:
    app.exitQgis()
'''


def build_buffer_clip(p: Dict[str, Any]) -> ToolResult:
    source = str(p["source"])
    overlay = str(p["overlay"])
    distance = float(p.get("distance") or 10)
    buffered = _derive(source, "_buffer", ".gpkg")
    out = _derive(source, "_buffer_clip", ".gpkg")

    script = (
        BUFFER_CLIP.replace("__PREAMBLE__", QGIS_PREAMBLE)
        .replace("__SOURCE__", source)
        .replace("__OVERLAY__", overlay)
        .replace("__DISTANCE__", "%.3f" % distance)
        .replace("__BUFFERED__", buffered)
        .replace("__OUTPUT__", out)
    )

    buffer_argv = [
        "qgis_process", "run", "native:buffer", "--",
        "INPUT=%s" % _quote(source),
        "DISTANCE=%g" % distance,
        "SEGMENTS=5",
        "OUTPUT=%s" % _quote(buffered),
    ]
    clip_argv = [
        "qgis_process", "run", "native:clip", "--",
        "INPUT=%s" % _quote(buffered),
        "OVERLAY=%s" % _quote(overlay),
        "OUTPUT=%s" % _quote(out),
    ]

    return ToolResult(
        title="Buffer %.3g then clip to %s" % (distance, os.path.basename(overlay)),
        artifacts=[Artifact("buffer_clip.py", "python", script)],
        commands=[" ".join(buffer_argv) + " && " + " ".join(clip_argv)],
        notes=[
            "DISTANCE is in the CRS units of the source layer: metres for a "
            "projected CRS, degrees for EPSG:4326. Reproject first if unsure.",
            "The intermediate buffer is kept so you can inspect it.",
            "SEGMENTS=5 is five segments per quarter circle; raise it for "
            "smoother curves.",
        ],
        execute=ExecuteSpec(argv=buffer_argv, description="QGIS buffer"),
    )


# ----------------------------------------------------------------------
# 3. spatial count
# ----------------------------------------------------------------------
SPATIAL_COUNT = '''# amu: count points inside each polygon
# Run with QGIS installed: python spatial_count.py
import csv

__PREAMBLE__
from qgis.core import QgsSpatialIndex, QgsVectorLayer

POINTS = r"__POINTS__"
POLYGONS = r"__POLYGONS__"
OUTPUT_CSV = r"__OUTPUT__"

try:
    points = QgsVectorLayer(POINTS, "points", "ogr")
    polygons = QgsVectorLayer(POLYGONS, "polygons", "ogr")

    if not points.isValid():
        raise SystemExit("Cannot open points layer: " + POINTS)
    if not polygons.isValid():
        raise SystemExit("Cannot open polygon layer: " + POLYGONS)

    if points.crs() != polygons.crs():
        print("WARNING: different CRS:", points.crs().authid(),
              "vs", polygons.crs().authid())
        print("         counts will be wrong - reproject first.")

    index = QgsSpatialIndex(polygons.getFeatures())
    counts = {}
    unmatched = 0

    for feature in points.getFeatures():
        geometry = feature.geometry()
        matched = None
        for polygon_id in index.intersects(geometry.boundingBox()):
            polygon = polygons.getFeature(polygon_id)
            if polygon.geometry().contains(geometry):
                matched = polygon_id
                break
        if matched is None:
            unmatched += 1
        else:
            counts[matched] = counts.get(matched, 0) + 1

    field_names = polygons.fields().names()
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["polygon_fid", "point_count"] + field_names)
        for feature in polygons.getFeatures():
            writer.writerow([feature.id(), counts.get(feature.id(), 0)]
                            + [feature[name] for name in field_names])

    print("Wrote:", OUTPUT_CSV)
    print("Points inside no polygon:", unmatched)
finally:
    app.exitQgis()
'''


def build_spatial_count(p: Dict[str, Any]) -> ToolResult:
    points = str(p["points"])
    polygons = str(p["polygons"])
    out = str(p.get("out") or _derive(polygons, "_counts", ".csv"))

    script = (
        SPATIAL_COUNT.replace("__PREAMBLE__", QGIS_PREAMBLE)
        .replace("__POINTS__", points)
        .replace("__POLYGONS__", polygons)
        .replace("__OUTPUT__", out)
    )

    return ToolResult(
        title="Count points per polygon",
        artifacts=[Artifact("spatial_count.py", "python", script)],
        steps=[
            "Run spatial_count.py with QGIS installed.",
            "Open the resulting CSV in QGIS or a spreadsheet.",
        ],
        notes=[
            "Uses 'contains', so points exactly on a boundary count as outside.",
            "Both layers must share one CRS; the script warns when they do not.",
            "This recipe has no qgis_process equivalent, so the agent will not "
            "execute it - it only writes the script.",
        ],
    )


# ----------------------------------------------------------------------
# 4. layer report
# ----------------------------------------------------------------------
LAYER_REPORT = '''# amu: describe a vector layer
# Run with QGIS installed: python layer_report.py
import os

from qgis.core import QgsApplication, QgsVectorLayer

prefix = os.environ.get("QGIS_PREFIX_PATH")
if prefix:
    QgsApplication.setPrefixPath(prefix, True)
app = QgsApplication([], False)
app.initQgis()

SOURCE = r"__SOURCE__"

try:
    layer = QgsVectorLayer(SOURCE, os.path.basename(SOURCE), "ogr")
    if not layer.isValid():
        raise SystemExit("Cannot open: " + SOURCE)

    print("Layer        :", layer.name())
    print("CRS          :", layer.crs().authid(), "-", layer.crs().description())
    print("Geometry type:", layer.geometryType(), "(0 point, 1 line, 2 polygon)")
    print("Feature count:", layer.featureCount())
    print("Fields       :")
    for field in layer.fields():
        print("   -", field.name(), "|", field.typeName())

    extent = layer.extent()
    print("Extent       : %.3f %.3f -> %.3f %.3f" % (
        extent.xMinimum(), extent.yMinimum(),
        extent.xMaximum(), extent.yMaximum()))
finally:
    app.exitQgis()
'''


def build_layer_report(p: Dict[str, Any]) -> ToolResult:
    source = str(p["source"])
    script = LAYER_REPORT.replace("__SOURCE__", source)

    argv = [
        "gdalinfo", "-so", source,
    ]

    return ToolResult(
        title="Report %s" % os.path.basename(source),
        artifacts=[Artifact("layer_report.py", "python", script)],
        commands=[" ".join(argv) + "   # quick alternative, no QGIS needed"],
        notes=[
            "The PyQGIS script starts its own QGIS application instance.",
            "Set QGIS_PREFIX_PATH if QGIS is installed in a non-default location.",
            "Feature count can be -1 for some OGR sources; the field list is "
            "always reliable.",
        ],
        execute=ExecuteSpec(argv=argv, description="gdalinfo summary"),
    )


RECIPES = [
    Recipe(
        id="qgis.reproject",
        domain="qgis",
        title="Reproject a layer",
        summary="Change a vector layer's CRS to another EPSG code.",
        triggers=[
            "reproject", "reprojection", "change crs", "to epsg", "assign crs",
            "convert crs", "change coordinate system", "transform to",
        ],
        params=[
            Param("source", "Which vector layer?", kind="path"),
            Param("epsg", "Target EPSG code?", kind="epsg"),
            Param("out", "Output path?", kind="path", required=False),
        ],
        mutates=False,
        build=build_reproject,
        needs=["qgis"],
    ),
    Recipe(
        id="qgis.buffer_clip",
        domain="qgis",
        title="Buffer and clip",
        summary="Buffer a layer by a distance and clip it to a boundary layer.",
        triggers=[
            "buffer", "buffer and clip", "buffer then clip", "clip to",
            "corridor", "clip layer", "distance around",
        ],
        params=[
            Param("source", "Layer to buffer?", kind="path"),
            Param("overlay", "Boundary layer to clip with?", kind="path"),
            Param("distance", "Buffer distance in CRS units?", kind="number",
                  required=False, default=10),
            Param("out", "Output path?", kind="path", required=False),
        ],
        mutates=False,
        build=build_buffer_clip,
        needs=["qgis"],
    ),
    Recipe(
        id="qgis.spatial_count",
        domain="qgis",
        title="Count points per polygon",
        summary="Spatial count of point features inside each polygon, to CSV.",
        triggers=[
            "count points", "points per polygon", "how many points",
            "spatial count", "points in polygon", "count features in",
        ],
        params=[
            Param("points", "Points layer?", kind="path"),
            Param("polygons", "Polygons layer?", kind="path"),
            Param("out", "Output CSV path?", kind="path", required=False),
        ],
        mutates=False,
        build=build_spatial_count,
        needs=["qgis"],
    ),
    Recipe(
        id="qgis.layer_report",
        domain="qgis",
        title="Describe a layer",
        summary="CRS, geometry type, feature count, fields and extent.",
        triggers=[
            "describe layer", "layer report", "inspect layer", "layer info",
            "what crs", "which crs", "how many features", "layer summary",
            "check the layer",
        ],
        params=[
            Param("source", "Which layer?", kind="path"),
        ],
        mutates=False,
        build=build_layer_report,
        needs=["qgis"],
    ),
]
