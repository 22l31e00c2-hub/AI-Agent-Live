# amu

A from-scratch neural network **and** a local CAD/GIS agent, in one project.

No pretrained weights. No external AI API. No paid service.

```
tinymind-ai/
├── model.py            # the from-scratch miniature GPT (Version 1)
├── train.py            # trains it on data/input.txt
├── generate.py         # samples text from the trained weights
├── data/input.txt      # training text (CAD/GIS flavoured)
├── checkpoints/        # created by train.py
│
├── cadagent/           # the agent for Revit / AutoCAD / Map 3D / QGIS
│   ├── types.py        # Param, Artifact, ToolResult, Recipe, ExecuteSpec
│   ├── extract.py      # reads paths, EPSG codes, numbers, layer names from a request
│   ├── lisp.py         # AutoLISP builders, escaping and a syntax checker
│   ├── social.py       # rule-based greetings, identity, capabilities
│   ├── web.py          # web reading: guarded fetch, summary, best-effort search
│   ├── vision.py       # image inspection (Pillow) and optional OCR
│   ├── planner.py      # routes a request to a recipe and fills its parameters
│   ├── registry.py     # aggregates all recipes
│   ├── runner.py       # capability probe + allowlisted execution
│   ├── core.py         # the agent loop, artifact writing, session log
│   ├── cli.py          # python -m cadagent
│   └── knowledge/      # revit.py, autocad.py, map3d.py, qgis.py
│
├── tests/test_agent.py # unit tests, standard library only
└── requirements.txt
```

## 1. The model (Version 1)

```bash
pip install -r requirements.txt
python train.py                 # 5000 steps, CPU is fine
python generate.py "LAYER " 200
```

`AMU_STEPS=50 python train.py` trains for a few seconds, for a smoke test.

This is a character-level GPT prototype: it learns which character tends to
follow which, nothing more. It is the foundation, not a chatbot.

## 2. The agent

```bash
python -m cadagent --list-recipes          # what it can do
python -m cadagent --doctor                # what this machine can execute
python -m cadagent "create layers roads, parcels, contours in red"
python -m cadagent "create levels at 0, 3.5, 7, 10.5" --domain revit
python -m cadagent "reproject parcels.shp to EPSG:25832"
python -m cadagent "import shapefile into autocad map 3d" --param out=site.dxf
python -m cadagent "look up https://docs.qgis.org/"
python -m cadagent "search the web for qgis native buffer"
python -m cadagent --search "qgis native buffer"
python -m cadagent "describe site_plan.png"
python -m cadagent --fetch https://example.com
python -m cadagent --image site_plan.png
python -m cadagent                          # interactive mode
```

Each request produces: the recipe that matched, the inputs it read, the steps to
follow, any command line, and the **scripts written into `ws/out/<timestamp>-<recipe>/`**.

Greetings, "who are you" and "what can you do" are handled by a small rule-based
responder (`cadagent/social.py`), not by the network. Recipe routing always runs
first, so "hey amu, create layers roads" is still a task.

### What it actually generates

| Domain | Deliverable | Runs where |
| --- | --- | --- |
| Revit | pyRevit / RevitPythonShell IronPython scripts (levels, walls, parameters, sheets→PDF, model health) | Windows + Revit |
| AutoCAD | `.scr` scripts and generated AutoLISP (layers, polyline from CSV, hatch, text find/replace, attributes→CSV, blocks from CSV, batch audit+purge) | Windows + AutoCAD |
| AutoCAD Map 3D | `ogr2ogr` commands for SHP/GeoPackage↔DXF, plus the exact MAPCSASSIGN / MAPIMPORT / MAPEXPORT dialog steps, plus a units+CRS audit LISP | GDAL anywhere; dialogs in Map 3D |
| QGIS | standalone PyQGIS scripts and `qgis_process` commands (reproject, buffer+clip, spatial count, layer report) | QGIS installed |
| Web | readable text + keyword summary of one page, cached under `ws/cache/web`, plus best-effort search | any machine with network |
| Images | size, colour, ink coverage, edge density, EXIF, optional OCR | any machine with Pillow |

### AutoLISP generation

`cadagent/lisp.py` is a generator, not a folder of copy-pasted templates:

* `lisp_string`, `path_literal` and `number_literal` escape every value that
  lands in a program (quotes, backslashes, newlines, NaN), so a layer name or a
  Windows path cannot break the file.
* `defun()`, `program()` and `quoted_pairs()` emit balanced forms with
  consistent naming: `hatch` becomes `c:TMK-HATCH`, and commands end with the
  conventional silent `(princ)`.
* `HELPERS` holds reusable functions (`tm:split`, `tm:replace-all`, `tm:csv`)
  that recipes compose instead of copying.
* `check()` scans a finished program for unbalanced parentheses and unterminated
  strings, and every recipe runs it on its own output before the file is
  written. The test suite runs it over every generated `.lsp` artifact.

Commands you get: `TMK-LAYERS`, `TMK-PLINE`, `TMK-HATCH`, `TMK-TREPLACE`,
`TMK-ATT2CSV`, `TMK-BLOCKCSV`, and `TMK-UNITS` (Map 3D units/CRS audit).

### Web and vision

**Web** (`cadagent/web.py`) is one HTTP GET through the standard library: no
API key, no paid service, no headless browser. It strips scripts, styles and
tags into readable text, then keeps the sentences that overlap your question and
caches the page under `ws/cache/web` so a repeat question works offline. A
cached copy counts as current for six hours; after that it is fetched again, and
if the network is down the older copy is still served but labelled stale, with
the reason. `--fresh` ignores the cache entirely. A wrong `charset` header does
not produce mojibake: the decoding with the fewest replacement characters wins. It
refuses `file://`, private ranges, `localhost` and link-local addresses, so a
crafted request cannot make it read your own network. `search the web for ...`
tries DuckDuckGo (two no-JavaScript endpoints) and then the Wikipedia API -
best effort, since none of them needs a key; `look up <url>` is the reliable
path.

**Vision** (`cadagent/vision.py`) measures an image with Pillow: format, size,
mode, dominant colours, ink coverage, edge density, EXIF (including a GPS note
that matters for georeferencing), plus OCR when the `tesseract` binary is
installed. The verdict distinguishes blank sheets, thin linework (drawings,
plans, scans) and continuous-tone photos - which is what you need for drawing
QA. There is no pretrained vision model, so it describes structure, not meaning.
It also reports the bounding box of the drawn content and the four margins, and
flags work that runs to the image edge - the usual sign of a cropped sheet or a
plot window tighter than the paper.

Image paths and URLs are picked up from the sentence itself, so `describe
plan.png` and `look up https://...` work without extra flags. CAD tasks always
win: a request that matches a recipe is never rerouted to the web.

### Safety rules the code enforces

* Generating a script is always allowed. Writing files happens on every request.
* **Execution** requires `--execute`, only allowlisted binaries
  (`ogr2ogr`, `qgis_process`, `gdalinfo`), never a shell string, never `shell=True`.
* Execution that changes your data (`recipe.mutates`) additionally requires `--yes`.
  Revit/AutoCAD scripts are always generation-only: they need the real
  application, so the agent tells you how to run them instead of pretending.
* Every session is appended to `ws/sessions.jsonl`.

### Honest limits

* The planner is **deterministic keyword/trigger scoring**, not a language model.
  It is explainable (each answer prints the match score) and it never invents a
  path or an EPSG code: unknown slots come back as questions.
* There is no chat model. Greetings and self-description come from a small
  rule-based responder; anything else has to be a task a recipe covers, and you
  get an explicit "I do not have a recipe for that" when it is not.
* Web reading is one HTTP GET: no JavaScript, no login, no search API key. The
  search fallback tries DuckDuckGo and then Wikipedia, and can be rate limited,
  blocked, or return nothing parseable.
* Vision is measurement, not semantics. It reports size, colour, ink coverage,
  edge density and EXIF, and OCR only if `tesseract` is installed. With no
  pretrained model it cannot tell you that a plan shows a school.
* Revit and AutoCAD cannot be driven from this Linux workspace. Their recipes
  were written against the documented APIs (Revit API via `__revit__`, AutoCAD
  `-LAYER`, `-PURGE`, AutoLISP) but were **not** executed against a live
  installation here.
* Map 3D's GIS commands are dialog-driven in most releases, so the agent does
  the data conversion with GDAL and leaves the dialogs to you. Blindly feeding
  answers to a dialog prompt sequence can corrupt a drawing.
* Sub-word tokenization, instruction tuning and ONNX export (Versions 2–5 of the
  original plan) are not in this milestone.

## 3. Adding a capability

One file, one list. Every recipe is a `Recipe` with triggers, params, a
`mutates` flag and a `build()` that returns a `ToolResult`:

```python
from ..types import Artifact, Param, Recipe, ToolResult

from ..lisp import check, defun, lisp_string, program

def build_offset(p):
    body = [
        '(setq ss (ssget "_X" (list (cons 8 %s))))' % lisp_string(p["layer"]),
        '(princ "\nSelected.")',
    ]
    form = defun("TMK-OFFSET", body=body, command=True)
    text = program("offset polylines", [form], command="TMK-OFFSET")
    assert not check(text), "generated LISP must be balanced"
    return ToolResult(
        title="Offset polylines",
        artifacts=[Artifact("offset.lsp", "lisp", text)],
        notes=["Review before running."],
    )

RECIPES = [
    Recipe(
        id="autocad.offset_polylines",
        domain="autocad",
        title="Offset polylines",
        summary="Offset every polyline on a layer.",
        triggers=["offset", "offset polylines"],
        params=[Param("layer", "Which layer?", kind="text")],
        mutates=True,
        build=build_offset,
        needs=["windows", "autocad"],
    ),
]
```

Drop it in `cadagent/knowledge/autocad.py`, add the triggers, and it shows up in
`--list-recipes` and in routing immediately.

## 4. Tests

```bash
cd tinymind-ai
python -m unittest discover -s tests -v
```

The suite covers routing for all four applications, slot extraction (EPSG,
paths, layer names, elevations), missing-parameter handling, the guarantee that
every recipe builds a non-empty artifact, the execution allowlist, the
"mutating work needs `--yes`" rule, and the session log. It also runs the
AutoLISP builders (escaping, naming, parenthesis checking) and asserts that
every generated `.lsp` file is structurally well formed.
