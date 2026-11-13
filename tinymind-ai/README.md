# amu

A from-scratch neural network **and** a local CAD/GIS agent, in one project.

No pretrained weights. No external AI API. No paid service.

```
tinymind-ai/
├── model.py             # the from-scratch miniature GPT
├── train.py             # trains it on data/input.txt, writes a JSON record
├── generate.py          # samples text from the trained weights
├── report.py            # report card: config, seed, scores, speed, sample
├── run_tests.py         # runs every test, fails if anything was skipped
├── data/build_corpus.py # deterministic generator for the training corpus
├── data/input.txt       # generated training text, 120K characters
├── checkpoints/         # created by train.py: weights plus .json records
│
├── cadagent/           # the agent for Revit / AutoCAD / Map 3D / QGIS
│   ├── types.py        # Param, Artifact, ToolResult, Recipe, ExecuteSpec
│   ├── extract.py      # reads paths, EPSG codes, numbers, layer names from a request
│   ├── lisp.py         # AutoLISP builders, escaping and a syntax checker
│   ├── social.py       # rule-based greetings, identity, capabilities
│   ├── web.py          # web reading: guarded fetch, summary, best-effort search
│   ├── ocr.py          # tesseract OCR: lines, confidence, preprocessing
│   ├── vision.py       # image inspection (Pillow) plus OCR when available
│   ├── planner.py      # routes a request to a recipe and fills its parameters
│   ├── registry.py     # aggregates all recipes
│   ├── runner.py       # capability probe + allowlisted execution
│   ├── core.py         # the agent loop, artifact writing, session log
│   ├── cli.py          # python -m cadagent
│   └── knowledge/      # revit.py, autocad.py, map3d.py, qgis.py
│
├── tests/test_agent.py  # agent tests, standard library only
├── tests/test_model.py  # network tests (need torch)
├── tests/test_corpus.py # corpus size, ASCII, exact reproducibility
├── tests/test_report.py # report-card formatting
├── tests/test_ocr.py    # OCR line grouping, confidence, failure paths
└── requirements.txt
```

## 1. The model

```bash
pip install -r requirements.txt
python data/build_corpus.py     # regenerate data/input.txt (deterministic)
python train.py                 # warmup + cosine, keeps the best validation weights
python generate.py "AutoCAD " 200
python generate.py --top-k 5 "from qgis.core import QgsProject" 300
python generate.py --top-k 5 "(defun c:TMK-" 300
python generate.py --bench      # parameters and characters per second
python report.py                # one-command report card for the checkpoint
```

`AMU_STEPS=50 python train.py` trains for a few seconds, for a smoke test.
`train.py` also reads `AMU_LR`, `AMU_WARMUP`, `AMU_EVAL_EVERY`, `AMU_PATIENCE`
and `AMU_SEED` from the environment, so a run is reproducible.

This is a character-level GPT prototype: it learns which character tends to
follow which. It is the foundation, not a chatbot, and fluent output below
means it has learned the corpus's phrasing, not that it understands CAD.

### The corpus

`data/input.txt` is about 180,000 characters of plain ASCII about Revit,
AutoCAD, AutoCAD Map 3D and QGIS, plus the GIS, automation, OCR and delivery
notes those tasks depend on. It is not downloaded: `data/build_corpus.py`
generates it from seeded sentence templates and complete code samples (pyRevit,
PyQGIS, AutoLISP, SCR, `qgis_process`, `ogr2ogr`), filled with real layer names,
EPSG codes, formats and commands, so the corpus can be grown, re-seeded and
audited like code.

```bash
python data/build_corpus.py                          # rewrite data/input.txt
python data/build_corpus.py --target 250000 --out /tmp/big.txt
```

`tests/test_corpus.py` asserts that the checked-in file is exactly what the
generator produces, is plain ASCII, covers all four applications, and is large
enough to train on. Growing the text - adding templates or raising `--target` -
is the honest lever for quality; adding parameters to a 180 KB corpus is not.

### The report card

Every checkpoint is saved with a JSON record beside it
(`checkpoints/amu.json`, `checkpoints/amu-final.json`) that holds the model
shape, the corpus checksum, the seed, the schedule, the losses and the training
time. One command prints it and measures generation speed:

```bash
python report.py                # best-validation checkpoint, with a sample
python report.py --final        # last state of the run
python report.py --no-bench     # skip loading torch, print the record only
```

### The architecture, and what came from Ling-3.1-flash

| Piece | Here | Why |
| --- | --- | --- |
| Normalisation | pre-norm **RMSNorm** | one scale vector per norm, no mean subtraction, stable at any depth |
| Positions | **RoPE** | rotary position applied to queries and keys; no position parameters at all |
| Attention | **grouped-query** (4 query heads, 2 key/value heads) | smaller key/value projections and cache, query capacity unchanged |
| Feed-forward | **SwiGLU** at ~8/3 width | gated activation inside the old 4x parameter budget |
| Embeddings | **tied** input and output matrix | fewer parameters, one shared meaning per token |
| Decoding | **KV cache** | each new character attends to cached keys instead of re-running the window |
| Training | warmup, cosine decay to 10%, AdamW beta2 0.95, weight decay 0.1 on matrices, grad clip 1.0 | the recipe current LLMs use at every scale |

The reference point is
[Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash): 560B
parameters with 25B active, 63 KDA (gated linear attention) layers and 9
gated-MLA layers in a 7:1 ratio, 512 routed experts plus one shared expert
(8 active), 64 heads, hidden size 4096, a 1M-token context, and a native
thinking mode with tool calling.

Six ideas transfer down to 737K parameters and are used above: an RMSNorm-only
pre-norm block, RoPE, grouped key/value heads, SwiGLU, tied embeddings, and the
warmup + cosine schedule with beta2 0.95 and decoupled weight decay.

Three do not, and pretending otherwise would be dishonest:

* **Mixture of experts.** Routing 512 experts needs far more tokens than the
  180 KB corpus here. Every expert would see the same handful of characters
  and the router would learn noise.
* **KDA / linear attention.** Its win is context length: a constant-size state
  instead of a growing cache. At `block_size = 128` there is nothing to save,
  and it gives up exact recall - the one thing a character model needs.
* **MLA and a 1M context.** Compressed latent attention is engineering for
  serving, not for learning `-LAYER` commands from a 180 KB corpus.

Measured on this machine (2 CPU threads, the 179,672-character corpus,
81-character vocabulary): **736,512 parameters**, 4.4 training steps/s, and
about 715 generated characters/s. Validation loss fell from 4.458 at step 0 to
**0.1660 (0.240 bits/char) at step 3600**; the last state of the 4,000-step run
scored 0.1700, so the curve flattened instead of collapsing.

With this corpus the model writes its language instead of character soup. One
sample, from the best checkpoint, prompt `AutoCAD `:

```text
AutoCAD drawing basics

A block is a reusable group of objects, and one edit to the block updates every insertion. A drawing is the container for everything you draw in AutoCAD, and it is saved as a DWG file.
```

It also writes small programs when the prompt looks like the start of a file and
the sampling is tightened with `--top-k`. Prompt
`from qgis.core import QgsProject`, top-k 5:

```python
from qgis.core import QgsProject


def layer_report(project):
    for layer in project.mapLayers().values():
        print(layer.name(), layer.featureCount())


layer_report(QgsProject.instance())
```

That is memorised corpus code, not reasoning: change the prompt and it drifts,
and a pyRevit prompt puts `transaction.Commit()` inside the loop. Treat any
generated program as a draft to read before running it. A four-character prompt
such as `def ` gets prose, because four characters do not identify the code
sections; a file opening does.

So the honest result is a 737K-parameter character model that has learned this
corpus well. It produces new sentences in the corpus's style and stops there;
it does not know what a shapefile is. The ceiling is still the data, which is
why the corpus is generated and tested rather than a fixed blob.

`train.py` keeps the best-validation weights in `checkpoints/amu.pt` and the
last state in `checkpoints/amu-final.pt`:

```bash
python generate.py "AutoCAD " 200          # best validation: 0.240 bits/char
python generate.py --final "AutoCAD " 200  # last step of the same run
python generate.py --top-k 5 "from qgis.core import QgsProject" 300
python report.py                           # config, seed, score, speed, sample
```

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
| Images | size, colour, ink coverage, edge density, EXIF, plus line-level OCR with confidence when tesseract is installed | any machine with Pillow |

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
best effort, since none of them needs a key - and then **opens the top result
itself**, so you get a summary and `page.txt` / `summary.md` instead of a list
of links. If the first page will not load it tries the next one and says which
were skipped; the link list always stays in the answer. `look up <url>` is
still the reliable path when you already know the page.

**Vision** (`cadagent/vision.py`) measures an image with Pillow: format, size,
mode, dominant colours, ink coverage, edge density, EXIF (including a GPS note
that matters for georeferencing), plus OCR when the `tesseract` binary is
installed. The verdict distinguishes blank sheets, thin linework (drawings,
plans, scans) and continuous-tone photos - which is what you need for drawing
QA. There is no pretrained vision model, so it describes structure, not meaning.
It also reports the bounding box of the drawn content and the four margins, and
flags work that runs to the image edge - the usual sign of a cropped sheet or a
plot window tighter than the paper.

**OCR** (`cadagent/ocr.py`) goes through tesseract and returns line-level
results instead of one blob: every line carries its mean confidence, so a
rotated or noisy scan shows up as a low number rather than quiet nonsense. It
preprocesses before reading - grayscale, autocontrast, and a 2x upscale when the
image is smaller than 1000 pixels - and takes `AMU_OCR_LANG=deu` and
`AMU_OCR_PSM=6` from the environment. If tesseract is missing, the image report
says so and keeps every measurement. This is a tool path, not a model
capability: the network never sees the pixels.

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
  edge density and EXIF, and OCR only if `tesseract` is installed; the neural
  network never reads pixels. With no pretrained model it cannot tell you that
  a plan shows a school.
* Revit and AutoCAD cannot be driven from this Linux workspace. Their recipes
  were written against the documented APIs (Revit API via `__revit__`, AutoCAD
  `-LAYER`, `-PURGE`, AutoLISP) but were **not** executed against a live
  installation here.
* Map 3D's GIS commands are dialog-driven in most releases, so the agent does
  the data conversion with GDAL and leaves the dialogs to you. Blindly feeding
  answers to a dialog prompt sequence can corrupt a drawing.
* The corpus is generated from templates, so a very low validation loss partly
  measures learned template grammar. It is reproducible and inspectable, not
  broad world knowledge.
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
python run_tests.py -v
```

`run_tests.py` runs the whole suite under the project virtualenv (it re-executes
itself there when the current interpreter cannot import torch) and **fails the
run if any test was skipped**, so a partial suite can never look like a pass.
`.venv/bin/python -m unittest discover -s tests` is the same suite without that
guard.

The suite covers routing for all four applications, slot extraction (EPSG,
paths, layer names, elevations), missing-parameter handling, the guarantee that
every recipe builds a non-empty artifact, the execution allowlist, the
"mutating work needs `--yes`" rule, and the session log. It also runs the
AutoLISP builders (escaping, naming, parenthesis checking) and asserts that
every generated `.lsp` file is structurally well formed. Web tests run against
injected transports, so the suite never touches the network.

`tests/test_model.py` checks the network itself: the causal mask (a later
character cannot change an earlier prediction), cached decoding matching a full
forward pass, grouped-query shapes, and generation running past `block_size`.
`tests/test_corpus.py`, `tests/test_report.py` and `tests/test_ocr.py` cover the
corpus generator, the report card and OCR parsing, and need only the standard
library. The full suite is 91 tests, all executed, none skipped.
