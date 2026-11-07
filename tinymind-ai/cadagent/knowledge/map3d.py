"""AutoCAD Map 3D recipes.

Map 3D's GIS commands (MAPIMPORT, MAPCSASSIGN, MAPEXPORT) are dialog-driven in
most releases, so a blind command-line script is unsafe - the wrong answer to a
prompt can modify the drawing. The agent therefore:

* generates a GDAL (ogr2ogr) command that does the data conversion reliably and
  identically on every machine, and
* gives you the exact Map 3D dialog steps to finish the job in the drawing.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..types import Artifact, ExecuteSpec, Param, Recipe, ToolResult


def _derive(source: str, suffix: str, ext: str) -> str:
    base = os.path.splitext(os.path.basename(source))[0]
    return os.path.join("out", base + suffix + ext)


# ----------------------------------------------------------------------
# 1. GIS -> DWG (shapefile to DXF, then Map 3D)
# ----------------------------------------------------------------------
def build_gis_to_dwg(p: Dict[str, Any]) -> ToolResult:
    source = str(p["source"])
    epsg = p.get("epsg")
    out = str(p.get("out") or _derive(source, "", ".dxf"))

    argv = ["ogr2ogr", "-f", "DXF", out, source]
    if epsg:
        argv += ["-t_srs", "EPSG:%d" % int(epsg)]
    command = " ".join('"%s"' % part if " " in part else part for part in argv)

    steps = [
        "Run the ogr2ogr command (or the generated .bat/.sh file).",
        "Open the DXF in Map 3D / AutoCAD (it arrives as blocks + layers).",
    ]
    if epsg:
        steps += [
            "Run MAPCSASSIGN in the drawing and assign EPSG:%d." % int(epsg),
            "Confirm with a known coordinate: MAPCSASSIGN > List shows the code.",
        ]
    else:
        steps += [
            "Before assigning anything: decide the source coordinate system.",
            "Run MAPCSASSIGN and pick the same EPSG code the data uses.",
        ]

    notes = [
        "ogr2ogr -f DXF writes AutoCAD blocks; attributes become block attributes.",
        "No reprojection happens without -t_srs, so coordinates stay as stored.",
        "Map 3D's MAPIMPORT dialog can also read the SHP directly if you prefer "
        "FDO; the DXF route keeps the drawing editable as plain AutoCAD objects.",
    ]

    artifacts = [
        Artifact(
            "gis_to_dwg.sh",
            "bash",
            "#!/bin/sh\n# amu: vector data -> DXF for AutoCAD / Map 3D\n"
            + command + "\n",
        ),
        Artifact(
            "gis_to_dwg.bat",
            "text",
            "@echo off\r\nREM amu: vector data -> DXF for AutoCAD / Map 3D\r\n"
            + command + "\r\n",
        ),
    ]

    return ToolResult(
        title="Import %s into a drawing" % os.path.basename(source),
        artifacts=artifacts,
        commands=[command],
        steps=steps,
        notes=notes,
        execute=ExecuteSpec(argv=argv, description="convert to DXF"),
    )


# ----------------------------------------------------------------------
# 2. DWG/DXF -> GIS
# ----------------------------------------------------------------------
def build_dwg_to_gis(p: Dict[str, Any]) -> ToolResult:
    source = str(p["source"])
    epsg = p.get("epsg")
    out = str(p.get("out") or _derive(source, "", ".gpkg"))

    argv = ["ogr2ogr", "-f", "GPKG", out, source]
    if epsg:
        argv += ["-a_srs", "EPSG:%d" % int(epsg)]
    command = " ".join('"%s"' % part if " " in part else part for part in argv)

    return ToolResult(
        title="Export %s to GIS" % os.path.basename(source),
        artifacts=[
            Artifact(
                "dwg_to_gis.sh",
                "bash",
                "#!/bin/sh\n# amu: drawing -> geopackage for QGIS / GIS\n"
                + command + "\n",
            ),
            Artifact(
                "dwg_to_gis.bat",
                "text",
                "@echo off\r\nREM amu: drawing -> geopackage for QGIS / GIS\r\n"
                + command + "\r\n",
            ),
        ],
        commands=[command],
        steps=[
            "Save the drawing as DXF (DXF is what GDAL reads; DWG needs a "
            "licensed ODA reader).",
            "Run the ogr2ogr command to write a GeoPackage.",
            "Open the .gpkg in QGIS and check the extent against the drawing.",
        ],
        notes=[
            "-a_srs only labels the data; it does not transform coordinates.",
            "Only model-space entities become features; choose one entity type "
            "per export for clean attribute tables.",
            "Map 3D alternative: MAPEXPORT from the drawing, selection-based.",
        ],
        execute=ExecuteSpec(argv=argv, description="convert DXF to GeoPackage"),
    )


# ----------------------------------------------------------------------
# 3. CRS / units audit
# ----------------------------------------------------------------------
CRS_AUDIT_LSP = ''';; amu: drawing units and extents audit
;; APPLOAD this file, then run: TMK-UNITS
(defun c:TMK-UNITS ( / )
  (princ (strcat "\\nINSUNITS    = " (itoa (getvar "INSUNITS"))
                 "   (6 = metres, 4 = millimetres)"))
  (princ (strcat "\\nMEASUREMENT = " (itoa (getvar "MEASUREMENT"))
                 "   (1 = metric, 0 = imperial)"))
  (princ (strcat "\\nLUNITS      = " (itoa (getvar "LUNITS"))))
  (princ (strcat "\\nEXTMIN      = " (vl-princ-to-string (getvar "EXTMIN"))))
  (princ (strcat "\\nEXTMAX      = " (vl-princ-to-string (getvar "EXTMAX"))))
  (princ "\\n-- Map 3D: run MAPCSASSIGN to read the assigned coordinate system --")
  (princ)
)
'''


def build_crs_audit(p: Dict[str, Any]) -> ToolResult:
    return ToolResult(
        title="Drawing units / CRS audit",
        artifacts=[Artifact("crs_audit.lsp", "lisp", CRS_AUDIT_LSP)],
        commands=[
            "APPLOAD crs_audit.lsp",
            "Type TMK-UNITS",
            "Map 3D: run MAPCSASSIGN and use List to see the assigned CRS",
        ],
        steps=[
            "Check INSUNITS first: 6 = metres, 4 = millimetres.",
            "Compare the extents with the expected project area to spot a "
            "factor-of-1000 unit mistake.",
            "Only then assign a coordinate system with MAPCSASSIGN.",
        ],
        notes=[
            "A drawing can hold correct geometry with no CRS assigned at all; "
            "Map 3D cannot reproject or export geographically until it is set.",
            "Never assign a CRS to 'fix' units - scale the geometry instead.",
        ],
    )


RECIPES = [
    Recipe(
        id="map3d.gis_to_dwg",
        domain="map3d",
        title="Bring GIS data into a drawing",
        summary="Shapefile/GeoPackage to DXF with GDAL, plus Map 3D steps.",
        triggers=[
            "shapefile into autocad", "import shapefile", "shp to dwg",
            "gis to cad", "shapefile to dxf", "map 3d import",
            "mapimport", "bring gis data into", "load shapefile",
            "shapefile into map 3d",
        ],
        params=[
            Param("source", "Which vector file (SHP, GeoPackage, GeoJSON)?",
                  kind="path"),
            Param("out", "Output DXF path?", kind="path", required=False),
            Param("epsg", "Target EPSG code?", kind="epsg", required=False),
        ],
        mutates=False,
        build=build_gis_to_dwg,
        needs=["gdal"],
    ),
    Recipe(
        id="map3d.dwg_to_gis",
        domain="map3d",
        title="Export drawing objects to GIS",
        summary="DXF to GeoPackage with GDAL, plus MAPEXPORT alternative.",
        triggers=[
            "dwg to gis", "export to shapefile", "mapexport", "dxf to shp",
            "drawing to shapefile", "cad to gis", "export drawing to gis",
            "geopackage from drawing",
        ],
        params=[
            Param("source", "Which DXF/DWG file?", kind="path"),
            Param("out", "Output GeoPackage path?", kind="path", required=False),
            Param("epsg", "EPSG code that describes the drawing?", kind="epsg",
                  required=False),
        ],
        mutates=False,
        build=build_dwg_to_gis,
        needs=["gdal"],
    ),
    Recipe(
        id="map3d.crs_audit",
        domain="map3d",
        title="Audit drawing units and CRS",
        summary="Report INSUNITS, extents and the Map 3D coordinate system steps.",
        triggers=[
            "coordinate system", "check crs", "crs audit", "drawing units",
            "mapcsassign", "units check", "check the units", "wrong scale",
            "scale is wrong", "epsg of drawing",
        ],
        params=[],
        mutates=False,
        build=build_crs_audit,
        needs=["windows", "autocad"],
    ),
]
