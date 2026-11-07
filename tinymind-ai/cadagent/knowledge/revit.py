"""Revit recipes.

Scripts are written for pyRevit / RevitPythonShell, which run IronPython inside
a running Revit session and expose the Revit API through ``__revit__``. They
cannot run on Linux: Revit is Windows-only. The agent therefore generates the
script and the run instructions; it only executes if you have pyRevit's CLI
available on Windows.
"""

from __future__ import annotations

from typing import Any, Dict

from ..types import Artifact, Param, Recipe, ToolResult

RUN_HINT = (
    "Revit is Windows-only: run inside the model via pyRevit / RevitPythonShell, "
    "or unattended with `pyrevit run <script.py> <model.rvt>`."
)


# ----------------------------------------------------------------------
# 1. create levels
# ----------------------------------------------------------------------
LEVELS = '''# amu: create levels
# pyRevit / RevitPythonShell script. Elevations below are in METRES.
from Autodesk.Revit.DB import FilteredElementCollector, Level, Transaction

doc = __revit__.ActiveUIDocument.Document

LEVELS = [
__TABLE__
]

t = Transaction(doc, "amu: create levels")
t.Start()

existing = set(level.Name for level in FilteredElementCollector(doc).OfClass(Level))
created = 0
skipped = 0

for elevation_m, name in LEVELS:
    if name in existing:
        print("skip (exists):", name)
        skipped += 1
        continue
    level = Level.Create(doc, elevation_m / 0.3048)   # feet are Revit internal units
    level.Name = name
    created += 1
    print("created:", name, "at", round(elevation_m, 3), "m")

t.Commit()
print("created", created, "levels, skipped", skipped)
'''


def build_levels(p: Dict[str, Any]) -> ToolResult:
    elevations = [float(e) for e in p["elevations"]]
    prefix = str(p.get("prefix") or "Level")

    rows = []
    for index, elevation in enumerate(elevations, start=1):
        rows.append('    (%.3f, "%s %d"),' % (elevation, prefix, index))

    script = LEVELS.replace("__TABLE__", "\n".join(rows))
    return ToolResult(
        title="Create %d levels in Revit" % len(elevations),
        artifacts=[Artifact("create_levels.py", "python", script)],
        steps=[
            "Open the target model in Revit (or use `pyrevit run create_levels.py Model.rvt`).",
            "pyRevit > script: run create_levels.py.",
            "Check the Revit status bar / output for created and skipped levels.",
        ],
        notes=[
            "Existing levels with the same name are skipped, not renamed.",
            "Level.Create takes feet; the script converts your metres automatically.",
            RUN_HINT,
        ],
    )


# ----------------------------------------------------------------------
# 2. model health report
# ----------------------------------------------------------------------
MODEL_HEALTH = '''# amu: Revit model health report (read-only)
import os

from Autodesk.Revit.DB import *

doc = __revit__.ActiveUIDocument.Document

print("=" * 60)
print("Document   :", doc.Title)
print("Path       :", doc.PathName)
print("Workshared :", doc.IsWorkshared)
if doc.PathName and os.path.exists(doc.PathName):
    print("File size  : %.1f MB" % (os.path.getsize(doc.PathName) / 1048576.0))

warnings = list(doc.GetWarnings())
print("Warnings   :", len(warnings))
for index, warning in enumerate(warnings[:20]):
    print("  -", warning.GetDescriptionText())
if len(warnings) > 20:
    print("  ... %d more" % (len(warnings) - 20))

print("-" * 60)
CATEGORIES = [
    (BuiltInCategory.OST_Walls, "walls"),
    (BuiltInCategory.OST_Doors, "doors"),
    (BuiltInCategory.OST_Windows, "windows"),
    (BuiltInCategory.OST_Floors, "floors"),
    (BuiltInCategory.OST_Rooms, "rooms"),
]
for category, label in CATEGORIES:
    count = (FilteredElementCollector(doc)
             .OfCategory(category)
             .WhereElementIsNotElementType()
             .GetElementCount())
    print("%-8s %d" % (label, count))

links = FilteredElementCollector(doc).OfClass(RevitLinkInstance).ToElements()
print("links   ", len(links))
for link in links:
    print("  -", link.Name)
print("=" * 60)
'''


def build_model_health(p: Dict[str, Any]) -> ToolResult:
    return ToolResult(
        title="Revit model health report",
        artifacts=[Artifact("model_health.py", "python", MODEL_HEALTH)],
        steps=[
            "Open the model, then run model_health.py with pyRevit.",
            "Copy the console output into your QA notes.",
        ],
        notes=[
            "Read-only: no transaction is started, nothing in the model changes.",
            "Warnings are the fastest signal for model quality; 0 is the target.",
            RUN_HINT,
        ],
    )


# ----------------------------------------------------------------------
# 3. export all sheets to one PDF
# ----------------------------------------------------------------------
SHEETS_PDF = '''# amu: export every sheet to a single PDF (Revit 2022+)
from Autodesk.Revit.DB import *
from System.Collections.Generic import List

doc = __revit__.ActiveUIDocument.Document

OUTDIR = r"__OUTDIR__"
BASENAME = "__BASENAME__"

sheets = sorted(
    FilteredElementCollector(doc).OfClass(ViewSheet).ToElements(),
    key=lambda sheet: sheet.SheetNumber,
)

if not sheets:
    print("No sheets in this model.")
else:
    ids = List[ElementId]([sheet.Id for sheet in sheets])
    options = PDFExportOptions()
    options.Combine = True
    options.FileName = BASENAME

    ok = doc.Export(OUTDIR, ids, options)
    print("sheets:", len(sheets))
    print("export ok:", ok)
    print("folder:", OUTDIR)
'''


def build_sheets_pdf(p: Dict[str, Any]) -> ToolResult:
    outdir = str(p.get("outdir") or "sheets_pdf")
    basename = str(p.get("basename") or "sheets")

    script = SHEETS_PDF.replace("__OUTDIR__", outdir).replace(
        "__BASENAME__", basename
    )
    return ToolResult(
        title="Export all Revit sheets to PDF",
        artifacts=[Artifact("export_sheets_pdf.py", "python", script)],
        steps=[
            "Open the model and run export_sheets_pdf.py.",
            "PDF lands in %s/%s.pdf" % (outdir, basename),
        ],
        notes=[
            "Requires Revit 2022 or newer (PDFExportOptions).",
            "Sheets are sorted by sheet number before export.",
            "OUTDIR must be an absolute path or writable from the model's folder.",
            RUN_HINT,
        ],
    )


# ----------------------------------------------------------------------
# 4. bulk parameter edit
# ----------------------------------------------------------------------
BULK_PARAM = '''# amu: set one parameter on every element of a category
from Autodesk.Revit.DB import *

CATEGORY_NAME = "__CATEGORY__"   # e.g. OST_Walls, OST_Doors, OST_Rooms
PARAMETER = "__PARAM__"
VALUE = """__VALUE__"""

doc = __revit__.ActiveUIDocument.Document
category = getattr(BuiltInCategory, CATEGORY_NAME)

t = Transaction(doc, "amu: set parameter")
t.Start()

changed = 0
skipped = 0
for element in (FilteredElementCollector(doc)
                .OfCategory(category)
                .WhereElementIsNotElementType()):
    param = element.LookupParameter(PARAMETER)
    if param is None or param.IsReadOnly:
        skipped += 1
        continue

    if param.StorageType == StorageType.String:
        param.Set(VALUE)
    elif param.StorageType == StorageType.Integer:
        param.Set(int(VALUE))
    elif param.StorageType == StorageType.Double:
        param.Set(float(VALUE))
    else:
        skipped += 1
        continue
    changed += 1

t.Commit()
print("category:", CATEGORY_NAME, "changed:", changed, "skipped:", skipped)
'''


def build_bulk_param(p: Dict[str, Any]) -> ToolResult:
    category = str(p["category"]).strip()
    if not category.startswith("OST_"):
        category = "OST_" + category.lstrip("_")

    value = str(p["value"]).replace('"', "'")
    script = (
        BULK_PARAM.replace("__CATEGORY__", category)
        .replace("__PARAM__", str(p["parameter"]))
        .replace("__VALUE__", value)
    )

    return ToolResult(
        title="Set %s = %r on all %s" % (p["parameter"], value, category),
        artifacts=[Artifact("set_parameter.py", "python", script)],
        steps=[
            "Open the model and run set_parameter.py.",
            "Check the changed/skipped counts; skipped usually means the "
            "parameter is missing or read-only on that element.",
        ],
        notes=[
            "One transaction wraps the whole edit, so it can be undone in one step.",
            "Type parameters are not touched (WhereElementIsNotElementType).",
            "Unit-aware parameters (length, area) expect Revit internal units.",
            RUN_HINT,
        ],
    )


# ----------------------------------------------------------------------
# 5. walls from selected lines
# ----------------------------------------------------------------------
WALLS_FROM_LINES = '''# amu: create walls from the selected lines / curves
from Autodesk.Revit.DB import *

HEIGHT_MM = __HEIGHT__

uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

levels = sorted(
    FilteredElementCollector(doc).OfClass(Level).ToElements(),
    key=lambda level: level.Elevation,
)
if not levels:
    raise Exception("No levels in this model - create a level first.")

base_level = levels[0]
wall_type = FilteredElementCollector(doc).OfClass(WallType).FirstElement()

curves = []
for element_id in uidoc.Selection.GetElementIds():
    element = doc.GetElement(element_id)
    if element is None:
        continue
    location = getattr(element, "Location", None)
    if isinstance(location, LocationCurve):
        curves.append(location.Curve)
    elif isinstance(element, ModelLine):
        curves.append(element.GeometryCurve)

if not curves:
    print("Nothing selected. Select lines (or model lines) and run again.")
else:
    t = Transaction(doc, "amu: walls from lines")
    t.Start()
    created = 0
    for curve in curves:
        Wall.Create(doc, curve, wall_type.Id, base_level.Id,
                    HEIGHT_MM / 304.8, 0.0, False, True)
        created += 1
    t.Commit()
    print("created", created, "walls on", base_level.Name,
          "| height", HEIGHT_MM, "mm")
'''


def build_walls_from_lines(p: Dict[str, Any]) -> ToolResult:
    height = float(p.get("height") or 3000)
    script = WALLS_FROM_LINES.replace("__HEIGHT__", "%.1f" % height)

    return ToolResult(
        title="Create walls from selected lines (%.0f mm)" % height,
        artifacts=[Artifact("walls_from_lines.py", "python", script)],
        steps=[
            "Open the model and select the lines / model lines you want as wall axes.",
            "Run walls_from_lines.py.",
            "Walls are hosted on the lowest level and use the project's first wall type.",
        ],
        notes=[
            "Uses the first WallType in the project; change `wall_type` for another.",
            "Structural flag is True and the base offset is 0.",
            RUN_HINT,
        ],
    )


RECIPES = [
    Recipe(
        id="revit.levels",
        domain="revit",
        title="Create levels in Revit",
        summary="Create one level per elevation (metres) with a name prefix.",
        triggers=[
            "create level", "create levels", "add levels", "add level",
            "levels at", "level at", "new levels", "revit level",
        ],
        params=[
            Param("elevations", "Which elevations in metres (e.g. 0, 3.5, 7)?",
                  kind="numbers"),
            Param("prefix", "Name prefix for the levels?", kind="text",
                  required=False, default="Level"),
        ],
        mutates=True,
        build=build_levels,
        needs=["windows", "revit", "pyrevit"],
    ),
    Recipe(
        id="revit.model_health",
        domain="revit",
        title="Revit model health report",
        summary="Warnings, category counts, links and file size, read-only.",
        triggers=[
            "model health", "health check", "warnings", "audit the model",
            "model audit", "check the model", "bim health", "model report",
        ],
        params=[],
        mutates=False,
        build=build_model_health,
        needs=["windows", "revit"],
    ),
    Recipe(
        id="revit.sheets_to_pdf",
        domain="revit",
        title="Export Revit sheets to PDF",
        summary="Batch print every sheet into one combined PDF.",
        triggers=[
            "export sheets", "sheets to pdf", "pdf from revit", "print sheets",
            "batch plot revit", "sheet pdf",
        ],
        params=[
            Param("outdir", "Output folder for the PDF?", kind="path",
                  required=False, default="sheets_pdf"),
            Param("basename", "PDF file name (without .pdf)?", kind="text",
                  required=False, default="sheets"),
        ],
        mutates=False,
        build=build_sheets_pdf,
        needs=["windows", "revit"],
    ),
    Recipe(
        id="revit.bulk_parameter",
        domain="revit",
        title="Bulk-set a Revit parameter",
        summary="Write one parameter value on every element of a category.",
        triggers=[
            "bulk parameter", "set parameter", "set a parameter",
            "parameter on all", "mark parameter", "fill parameter",
            "update parameter",
        ],
        params=[
            Param("category", "Which category (e.g. OST_Walls)?", kind="text"),
            Param("parameter", "Exact parameter name?", kind="text"),
            Param("value", "New value?", kind="text"),
        ],
        mutates=True,
        build=build_bulk_param,
        needs=["windows", "revit"],
    ),
    Recipe(
        id="revit.walls_from_lines",
        domain="revit",
        title="Walls from selected lines",
        summary="Turn selected lines into walls on the lowest level.",
        triggers=[
            "walls from lines", "wall from line", "create walls",
            "draw walls", "make walls", "lines to walls",
        ],
        params=[
            Param("height", "Wall height in millimetres?", kind="number",
                  required=False, default=3000),
        ],
        mutates=True,
        build=build_walls_from_lines,
        needs=["windows", "revit"],
    ),
]
