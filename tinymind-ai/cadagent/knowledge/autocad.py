"""AutoCAD recipes: ``.scr`` command scripts and generated AutoLISP programs.

Every LISP file here is composed with :mod:`cadagent.lisp` (escaped literals,
shared helpers, ``defun``/``program`` builders) and self-checked for balanced
parentheses and strings before it is written out.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Tuple

from ..lisp import (
    check,
    defun,
    lisp_string,
    number_literal,
    path_literal,
    program,
    quoted_pairs,
)
from ..types import Artifact, Param, Recipe, ToolResult

LISP_NOTE = "LISP self-check"


def _lisp_artifact(filename: str, text: str) -> Tuple[Artifact, List[str]]:
    return Artifact(filename, "lisp", text), check(text)


def _with_check(notes: List[str], problems: List[str]) -> List[str]:
    if problems:
        notes.append("%s failed: %s" % (LISP_NOTE, "; ".join(problems)))
    return notes


# ----------------------------------------------------------------------
# 1. layer setup  (.scr + AutoLISP)
# ----------------------------------------------------------------------
def _layer_scr(layers: List[str], color: int) -> str:
    lines = ["; amu layer setup", "; AutoCAD: command SCRIPT -> pick this file"]
    for name in layers:
        lines += [
            "-LAYER",
            "M", name,
            "C", str(color), name,
            "",
        ]
    return "\n".join(lines) + "\n"


def _layer_lisp(layers: List[str], color: int) -> str:
    specs = quoted_pairs([(name, color) for name in layers])

    body = [
        "(setq specs",
        specs,
        ")",
        "",
        "(foreach spec specs",
        "  (setq name (car spec))",
        "  (setq color (cadr spec))",
        '  (command "_.-LAYER" "_M" name "_C" (itoa color) name "")',
        ")",
        "",
        r'(princ (strcat "\namu: " (itoa (length specs)) " layers ready."))',
    ]

    form = defun(
        "TMK-LAYERS",
        locals_=["specs", "spec", "name", "color"],
        body=body,
        doc="Create the requested layers, or reset their colour if they exist.",
        command=True,
    )
    return program("layer setup", [form], command="TMK-LAYERS")


def build_layer_setup(p: Dict[str, Any]) -> ToolResult:
    layers = [str(name).strip() for name in p["layers"] if str(name).strip()]
    color = int(p.get("color") or 7)

    text = _layer_lisp(layers, color)
    artifact, problems = _lisp_artifact("layers.lsp", text)

    notes = _with_check(
        [
            "Layers are created if missing and set to ACI colour %d." % color,
            "Set a linetype afterwards with -LINETYPE if you need dashed lines; "
            "the linetype must already be loaded.",
            "Both files are safe to re-run.",
        ],
        problems,
    )

    return ToolResult(
        title="Create %d layers" % len(layers),
        artifacts=[Artifact("layers.scr", "text", _layer_scr(layers, color)), artifact],
        commands=[
            "AutoCAD: type SCRIPT, choose layers.scr  (fastest, no loading)",
            "AutoCAD: APPLOAD layers.lsp, then type TMK-LAYERS",
        ],
        notes=notes,
    )


# ----------------------------------------------------------------------
# 2. polyline from CSV
# ----------------------------------------------------------------------
def _polyline_lisp(path: str, layer: str, skip_header: bool) -> str:
    layer_line = '(command "_.-LAYER" "_M" %s "")' % lisp_string(layer)

    body = [
        "(setq path %s)" % path_literal(path),
        "(setq skip %s)" % number_literal(1 if skip_header else 0),
        '(setq handle (open path "r"))',
        "",
        "(if (null handle)",
        r'  (princ (strcat "\nCannot open " path))',
        "  (progn",
        "    " + layer_line,
        '    (command "_.PLINE")',
        "    (setq points 0)",
        "    (while (setq line (read-line handle))",
        "      (if (> skip 0)",
        "        (setq skip (1- skip))",
        "        (progn",
        '          (setq parts (tm:split line ","))',
        "          (if (and (>= (length parts) 2) (> (strlen (car parts)) 0))",
        "            (progn",
        "              (command (list (atof (nth 0 parts)) (atof (nth 1 parts))))",
        "              (setq points (1+ points))",
        "            )",
        "          )",
        "        )",
        "      )",
        "    )",
        '    (command "")',
        "    (close handle)",
        r'    (princ (strcat "\namu: polyline with " (itoa points) " vertices."))',
        "  )",
        ")",
    ]

    form = defun(
        "TMK-PLINE",
        locals_=["path", "handle", "line", "parts", "skip", "points"],
        body=body,
        doc="Draw a polyline through the X,Y rows of a CSV file.",
        command=True,
    )
    return program(
        "polyline from a CSV of points", [form], helpers=["split"], command="TMK-PLINE"
    )


def build_polyline_from_csv(p: Dict[str, Any]) -> ToolResult:
    path = str(p["csv_path"])
    layer = str(p.get("layer") or "TMK_PLINE").strip()
    skip = bool(p.get("has_header"))

    text = _polyline_lisp(path, layer, skip)
    artifact, problems = _lisp_artifact("polyline_from_csv.lsp", text)

    notes = _with_check(
        [
            "Decimal separator must be a dot, not a comma.",
            "Uses an absolute path on your machine: keep the file where you put it.",
        ],
        problems,
    )

    return ToolResult(
        title="Polyline from %s" % os.path.basename(path),
        artifacts=[artifact],
        commands=[
            "APPLOAD polyline_from_csv.lsp",
            "Type TMK-PLINE inside the target drawing",
        ],
        steps=[
            "CSV format: one point per line, X first, Y second, optional Z.",
            "The first line is %s." % (
                "skipped as a header" if skip else "read as coordinates"),
            "Current drawing units are used; no scaling is applied.",
        ],
        notes=notes,
    )


# ----------------------------------------------------------------------
# 3. hatch closed polylines
# ----------------------------------------------------------------------
def _hatch_lisp(layer: str, pattern: str, scale: float) -> str:
    body = [
        "(setq layer %s)" % lisp_string(layer),
        "(setq pattern %s)" % lisp_string(pattern),
        "(setq scale %s)" % number_literal(scale),
        "(setq ms (vla-get-ModelSpace (vla-get-ActiveDocument (vlax-get-acad-object))))",
        "",
        '(if (= layer "*")',
        r'  (setq ss (ssget "_X" (list (cons 0 "LWPOLYLINE"))))',
        r'  (setq ss (ssget "_X" (list (cons 0 "LWPOLYLINE") (cons 8 layer))))',
        ")",
        "",
        "(if (null ss)",
        r'  (princ (strcat "\nNo LWPOLYLINE found on layer: " layer))',
        "  (progn",
        r'    (command "_.UNDO" "_BE")',
        "    (setq index 0)",
        "    (setq skipped 0)",
        "    (while (< index (sslength ss))",
        "      (setq ename (ssname ss index))",
        "      (setq data (entget ename))",
        "      (if (= 0 (logand 1 (cdr (assoc 70 data))))",
        "        (setq skipped (1+ skipped))",
        "        (progn",
        "          (setq hatch (vla-AddHatch ms 1 pattern :vlax-true))",
        r"          (setq loop (vlax-make-safearray vlax-vbObject '(0 . 0)))",
        "          (vlax-safearray-put-element loop 0 (vlax-ename->vla-object ename))",
        "          (vla-AppendOuterLoop hatch loop)",
        "          (vla-put-PatternScale hatch scale)",
        r'          (if (/= layer "*") (vla-put-Layer hatch layer))',
        "          (vla-Evaluate hatch)",
        "        )",
        "      )",
        "      (setq index (1+ index))",
        "    )",
        r'    (command "_.UNDO" "_E")',
        r'    (princ (strcat "\nHatched " (itoa (- index skipped))',
        r'                   " closed polylines, skipped " (itoa skipped) " open ones."))',
        "  )",
        ")",
    ]

    form = defun(
        "TMK-HATCH",
        locals_=["layer", "pattern", "scale", "ms", "ss", "index", "ename",
                 "data", "hatch", "loop", "skipped"],
        body=body,
        doc="Hatch every closed LWPOLYLINE on a layer (ActiveX, one undo step).",
        command=True,
    )
    return program("hatch closed polylines", [form], command="TMK-HATCH")


def build_hatch_closed(p: Dict[str, Any]) -> ToolResult:
    layer = str(p.get("layer") or "*").strip()
    pattern = str(p.get("pattern") or "ANSI31").strip()
    scale = float(p.get("scale") or 1.0)

    text = _hatch_lisp(layer, pattern, scale)
    artifact, problems = _lisp_artifact("hatch_closed.lsp", text)

    notes = _with_check(
        [
            "Only closed LWPOLYLINE objects are hatched; open ones are counted and skipped.",
            "The closed flag is tested with (logand 1 (cdr (assoc 70 ...))), so "
            "polylines with extra flags (for example 129) are still caught.",
            "Model space only: hatches are created in the current drawing.",
            "Everything runs inside one UNDO group.",
        ],
        problems,
    )

    return ToolResult(
        title="Hatch closed polylines on %s" % layer,
        artifacts=[artifact],
        commands=["APPLOAD hatch_closed.lsp", "Type TMK-HATCH"],
        steps=[
            "Check the pattern name exists in your AutoCAD (ANSI31, SOLID, AR-CONC...).",
            "Run TMK-HATCH; open polylines are skipped and reported.",
        ],
        notes=notes,
    )


# ----------------------------------------------------------------------
# 4. find and replace text
# ----------------------------------------------------------------------
def _text_replace_lisp(find: str, replace: str) -> str:
    body = [
        "(setq find %s)" % lisp_string(find),
        "(setq repl %s)" % lisp_string(replace),
        r'(setq ss (ssget "_X" (list (cons 0 "TEXT,MTEXT"))))',
        "",
        "(if (null ss)",
        r'  (princ "\nNo TEXT or MTEXT found.")',
        "  (progn",
        r'    (command "_.UNDO" "_BE")',
        "    (setq index 0)",
        "    (setq changed 0)",
        "    (while (< index (sslength ss))",
        "      (setq ename (ssname ss index))",
        "      (setq obj (vlax-ename->vla-object ename))",
        "      (setq old (vla-get-TextString obj))",
        "      (setq new (tm:replace-all old find repl))",
        r'      (if (/= old new) (progn',
        "        (vla-put-TextString obj new)",
        "        (setq changed (1+ changed))",
        "      ))",
        "      (setq index (1+ index))",
        "    )",
        r'    (command "_.UNDO" "_E")',
        r'    (princ (strcat "\nChecked " (itoa index) " text objects, changed "',
        r'                   (itoa changed) "."))',
        "  )",
        ")",
    ]

    form = defun(
        "TMK-TREPLACE",
        locals_=["find", "repl", "ss", "index", "ename", "obj", "old", "new", "changed"],
        body=body,
        doc="Replace every occurrence of a string in TEXT and MTEXT objects.",
        command=True,
    )
    return program("find and replace text", [form], helpers=["replace-all"],
                   command="TMK-TREPLACE")


def build_text_replace(p: Dict[str, Any]) -> ToolResult:
    find = str(p["find"])
    replace = str(p["replace"])

    text = _text_replace_lisp(find, replace)
    artifact, problems = _lisp_artifact("text_replace.lsp", text)

    notes = _with_check(
        [
            "Applies to every TEXT and MTEXT object in model and paper space.",
            "The whole run is one UNDO group, so a single U undoes it.",
            "Matching is literal and case sensitive; it does not use wildcards.",
            "Attribute definitions inside blocks are not changed (use BATTMAN "
            "for those).",
        ],
        problems,
    )

    return ToolResult(
        title="Replace %r with %r" % (find, replace),
        artifacts=[artifact],
        commands=["APPLOAD text_replace.lsp", "Type TMK-TREPLACE"],
        steps=[
            "Save the drawing first: this edits every text object in the file.",
            "Run TMK-TREPLACE and read the checked/changed counts.",
        ],
        notes=notes,
    )


# ----------------------------------------------------------------------
# 5. block attributes to CSV
# ----------------------------------------------------------------------
def _attributes_lisp(out: str) -> str:
    body = [
        "(setq path %s)" % path_literal(out),
        r'(setq ss (ssget "_X" (list (cons 0 "INSERT") (cons 66 1))))',
        "",
        "(if (null ss)",
        r'  (princ "\nNo attributed blocks found.")',
        "  (progn",
        '    (setq handle (open path "w"))',
        "    (if (null handle)",
        r'      (princ (strcat "\nCannot write " path))',
        "      (progn",
        r'        (write-line "block,x,y,z,attributes" handle)',
        "        (setq index 0)",
        "        (while (< index (sslength ss))",
        "          (setq ename (ssname ss index))",
        "          (setq obj (vlax-ename->vla-object ename))",
        "          (setq point (vlax-safearray->list",
        "                        (vlax-variant-value (vla-get-InsertionPoint obj))))",
        "          (setq line (strcat (tm:csv (vla-get-Name obj))",
        '                             "," (rtos (car point) 2 3)',
        '                             "," (rtos (cadr point) 2 3)',
        '                             "," (rtos (caddr point) 2 3)))',
        "          (foreach item",
        "            (vlax-safearray->list (vlax-variant-value (vla-GetAttributes obj)))",
        r'            (setq line (strcat line "," (tm:csv (vla-get-TagString item))',
        r'                               "=" (tm:csv (vla-get-TextString item))))',
        "          )",
        "          (write-line line handle)",
        "          (setq index (1+ index))",
        "        )",
        "        (close handle)",
        r'        (princ (strcat "\nWrote " (itoa index) " blocks to " path))',
        "      )",
        "    )",
        "  )",
        ")",
    ]

    form = defun(
        "TMK-ATT2CSV",
        locals_=["path", "ss", "index", "ename", "obj", "point", "line",
                 "handle", "item"],
        body=body,
        doc="Write block name, insertion point and attributes to a CSV file.",
        command=True,
    )
    return program("block attributes to CSV", [form],
                   helpers=["replace-all", "csv"], command="TMK-ATT2CSV")


def build_attributes_csv(p: Dict[str, Any]) -> ToolResult:
    out = str(p.get("out") or os.path.join("out", "attributes.csv"))

    text = _attributes_lisp(out)
    artifact, problems = _lisp_artifact("attributes_to_csv.lsp", text)

    notes = _with_check(
        [
            "Only INSERT objects that carry attributes (group 66 = 1) are exported.",
            "Commas inside a value become semicolons; quotes become apostrophes, "
            "so the CSV stays parseable.",
            "Uses RTOS with 3 decimals, independent of the current LUPREC.",
            "UTF-8 is not guaranteed: AutoCAD writes the drawing's ANSI code page.",
        ],
        problems,
    )

    return ToolResult(
        title="Export block attributes to CSV",
        artifacts=[artifact],
        commands=["APPLOAD attributes_to_csv.lsp", "Type TMK-ATT2CSV"],
        notes=notes,
    )


# ----------------------------------------------------------------------
# 6. insert blocks from CSV
# ----------------------------------------------------------------------
def _blocks_from_csv_lisp(path: str, block: str, scale: float) -> str:
    body = [
        "(setq path %s)" % path_literal(path),
        "(setq block %s)" % lisp_string(block),
        "(setq scale %s)" % number_literal(scale),
        "",
        r'(if (null (tblsearch "BLOCK" block))',
        r'  (princ (strcat "\nBlock not found in this drawing: " block))',
        "  (progn",
        '    (setq handle (open path "r"))',
        "    (if (null handle)",
        r'      (princ (strcat "\nCannot open " path))',
        "      (progn",
        "        (setq count 0)",
        r'        (command "_.UNDO" "_BE")',
        "        (while (setq line (read-line handle))",
        '          (setq parts (tm:split line ","))',
        "          (if (and (>= (length parts) 2) (> (strlen (car parts)) 0))",
        "            (progn",
        "              (setq point (list (atof (nth 0 parts)) (atof (nth 1 parts)) 0.0))",
        '              (command "_.-INSERT" block point scale scale 0.0)',
        "              (setq count (1+ count))",
        "            )",
        "          )",
        "        )",
        "        (close handle)",
        r'        (command "_.UNDO" "_E")',
        r'        (princ (strcat "\nInserted " (itoa count) " blocks of " block "."))',
        "      )",
        "    )",
        "  )",
        ")",
    ]

    form = defun(
        "TMK-BLOCKCSV",
        locals_=["path", "block", "scale", "handle", "line", "parts", "point", "count"],
        body=body,
        doc="Insert one block at every X,Y row of a CSV file.",
        command=True,
    )
    return program("insert blocks from CSV", [form], helpers=["split"],
                   command="TMK-BLOCKCSV")


def build_blocks_from_csv(p: Dict[str, Any]) -> ToolResult:
    path = str(p["csv_path"])
    block = str(p["block"]).strip()
    scale = float(p.get("scale") or 1.0)

    text = _blocks_from_csv_lisp(path, block, scale)
    artifact, problems = _lisp_artifact("blocks_from_csv.lsp", text)

    notes = _with_check(
        [
            "The block must already exist in the drawing; the script checks with "
            "tblsearch and stops with a message instead of feeding answers to a "
            "repeating -INSERT prompt.",
            "Rows with fewer than two columns are skipped.",
            "Insertion point Z is always 0; extend the list for 3D points.",
            "Everything runs inside one UNDO group.",
        ],
        problems,
    )

    return ToolResult(
        title="Insert %r at CSV points" % block,
        artifacts=[artifact],
        commands=["APPLOAD blocks_from_csv.lsp", "Type TMK-BLOCKCSV"],
        steps=[
            "CSV format: X first, Y second, one point per line.",
            "Run TMK-BLOCKCSV and read the inserted count.",
        ],
        notes=notes,
    )


# ----------------------------------------------------------------------
# 7. batch purge / audit across a folder (.scr only)
# ----------------------------------------------------------------------
def _batch_scr(files: List[str], audit: bool) -> str:
    lines = [
        "; amu batch clean: OPEN -> AUDIT -> PURGE -> QSAVE -> CLOSE",
        "; AutoCAD: SCRIPT -> this file. Every drawing in the list is modified.",
        "_.FILEDIA",
        "0",
    ]
    if not files:
        lines.append("; No .dwg files were found when this script was generated.")
        lines.append("; Regenerate it, or paste your file paths below the FILEDIA block.")

    for path in files:
        lines += ['_.OPEN', '"%s"' % path.replace("\\", "/")]
        if audit:
            lines += ["_.AUDIT", "Y"]
        lines += [
            "-PURGE",
            "A",
            "*",
            "N",
            "_.QSAVE",
            "_.CLOSE",
        ]

    lines += ["_.FILEDIA", "1", "; done"]
    return "\n".join(lines) + "\n"


def build_batch_purge(p: Dict[str, Any]) -> ToolResult:
    folder = str(p["folder"])
    pattern = str(p.get("pattern") or "*.dwg")
    audit = bool(p.get("audit", True))

    files: List[str] = []
    if os.path.isdir(folder):
        files = sorted(glob.glob(os.path.join(folder, pattern)))

    script = _batch_scr(files, audit)
    return ToolResult(
        title="Batch clean %d drawings" % len(files),
        artifacts=[Artifact("batch_clean.scr", "text", script)],
        commands=["AutoCAD: type SCRIPT, choose batch_clean.scr"],
        steps=[
            "Close every drawing first: the script opens and closes each file.",
            "Run SCRIPT once and let it finish; each file is saved with QSAVE.",
            "Verify a few files before running it on a whole project folder.",
        ],
        notes=[
            "Make a copy of the folder first: PURGE and AUDIT modify the DWGs.",
            "If your AutoCAD asks an extra -PURGE prompt (nested items), add one "
            "more line to the script after 'N'.",
            "Scripts use FILEDIA 0; the script restores FILEDIA 1 at the end.",
        ],
    )


RECIPES = [
    Recipe(
        id="autocad.layer_setup",
        domain="autocad",
        title="Create AutoCAD layers",
        summary="Generate .scr and .lsp files that create layers with colours.",
        triggers=[
            "create layers", "create a layer", "new layers", "layer setup",
            "setup layers", "set up layers", "layers for", "make layers",
            "add layers", "colour of layer",
        ],
        params=[
            Param("layers", "Which layer names (comma separated)?", kind="layers"),
            Param("color", "ACI colour number (1 red, 2 yellow, 3 green, 4 cyan, "
                           "5 blue, 6 magenta, 7 white)?", kind="number",
                  required=False, default=7),
        ],
        mutates=True,
        build=build_layer_setup,
        needs=["windows", "autocad"],
    ),
    Recipe(
        id="autocad.polyline_from_csv",
        domain="autocad",
        title="Polyline from CSV coordinates",
        summary="AutoLISP that draws a polyline from X,Y pairs in a CSV.",
        triggers=[
            "polyline from csv", "points from csv", "csv to polyline",
            "draw polyline", "polyline from text", "coordinates from csv",
            "xyz to polyline", "import points",
        ],
        params=[
            Param("csv_path", "Path to the CSV file?", kind="csv"),
            Param("layer", "Layer for the polyline?", kind="text",
                  required=False, default="TMK_PLINE"),
            Param("has_header", "Does the CSV have a header line?", kind="flag",
                  required=False, default=False),
        ],
        mutates=True,
        build=build_polyline_from_csv,
        needs=["windows", "autocad"],
    ),
    Recipe(
        id="autocad.hatch_closed_polylines",
        domain="autocad",
        title="Hatch closed polylines",
        summary="Hatch every closed LWPOLYLINE on a layer, one undo group.",
        triggers=[
            "hatch", "hatch polylines", "hatch closed", "fill polygons",
            "add hatch", "hatch layer",
        ],
        params=[
            Param("layer", "Layer with the closed polylines ('*' for all)?",
                  kind="text", required=False, default="*"),
            Param("pattern", "Hatch pattern name?", kind="text",
                  required=False, default="ANSI31"),
            Param("scale", "Hatch pattern scale?", kind="number",
                  required=False, default=1.0),
        ],
        mutates=True,
        build=build_hatch_closed,
        needs=["windows", "autocad"],
    ),
    Recipe(
        id="autocad.text_find_replace",
        domain="autocad",
        title="Find and replace text",
        summary="AutoLISP that replaces a string in every TEXT and MTEXT object.",
        triggers=[
            "find and replace", "replace text", "find replace", "change text",
            "rename text", "update text", "search and replace",
        ],
        params=[
            Param("find", "Text to find?", kind="text"),
            Param("replace", "Replacement text?", kind="text"),
        ],
        mutates=True,
        build=build_text_replace,
        needs=["windows", "autocad"],
    ),
    Recipe(
        id="autocad.attributes_to_csv",
        domain="autocad",
        title="Block attributes to CSV",
        summary="AutoLISP that exports block name, point and attributes to CSV.",
        triggers=[
            "block attributes", "attributes to csv", "export attributes",
            "extract attributes", "attribute export", "attribute report",
            "export block data",
        ],
        params=[
            Param("out", "Output CSV path?", kind="path", required=False),
        ],
        mutates=False,
        build=build_attributes_csv,
        needs=["windows", "autocad"],
    ),
    Recipe(
        id="autocad.blocks_from_csv",
        domain="autocad",
        title="Insert blocks from CSV",
        summary="AutoLISP that inserts one block at every X,Y row of a CSV.",
        triggers=[
            "insert blocks", "blocks from csv", "insert block", "place blocks",
            "block from csv", "blocks at points",
        ],
        params=[
            Param("csv_path", "CSV with X,Y columns?", kind="csv"),
            Param("block", "Block name to insert?", kind="text"),
            Param("scale", "Insertion scale?", kind="number",
                  required=False, default=1.0),
        ],
        mutates=True,
        build=build_blocks_from_csv,
        needs=["windows", "autocad"],
    ),
    Recipe(
        id="autocad.batch_purge",
        domain="autocad",
        title="Batch audit and purge drawings",
        summary="Script that opens every DWG in a folder, audits, purges and saves.",
        triggers=[
            "batch purge", "purge all drawings", "audit all drawings",
            "clean up drawings", "batch audit", "purge and audit",
            "purge dwg", "clean drawings",
        ],
        params=[
            Param("folder", "Folder containing the DWG files?", kind="path"),
            Param("pattern", "File pattern?", kind="text",
                  required=False, default="*.dwg"),
            Param("audit", "Also run AUDIT before PURGE?", kind="flag",
                  required=False, default=True),
        ],
        mutates=True,
        build=build_batch_purge,
        needs=["windows", "autocad"],
    ),
]
