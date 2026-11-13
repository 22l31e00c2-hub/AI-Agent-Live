"""Build data/input.txt: the corpus amu learns from.

The corpus is generated rather than downloaded: this file is the source of the
training text, so the data can be grown, re-seeded and audited like code.

The material is plain ASCII English about the four applications amu's agent
drives - Revit, AutoCAD, AutoCAD Map 3D and QGIS - plus the GIS, automation
and quality-control notes those tasks depend on. Sentence templates are filled
with real layer names, EPSG codes, formats and commands, and shuffled with a
seeded generator, so every run with the same seed produces the same bytes.

Run:
    python data/build_corpus.py
    python data/build_corpus.py --target 250000 --seed 7 --out /tmp/big.txt
"""

import argparse
import random

DEFAULT_SEED = 2026
DEFAULT_TARGET = 180_000

LAYERS = [
    "ROADS", "PARCELS", "CONTOURS", "BUILDINGS", "WATER", "SEWER",
    "TREES", "PIPES", "SITE", "GRID", "TEXT", "DIMENSIONS", "EASEMENTS",
    "STREETLIGHTS",
]

COLORS = ["red", "yellow", "green", "cyan", "white", "blue"]

FORMATS = ["SHP", "DXF", "GeoPackage", "GeoJSON", "CSV", "KML"]

# EPSG code, name, unit, region
SYSTEMS = [
    (25832, "ETRS89 / UTM zone 32N", "metres", "northern Europe"),
    (25833, "ETRS89 / UTM zone 33N", "metres", "eastern Europe"),
    (28992, "Amersfoort / RD New", "metres", "the Netherlands"),
    (27700, "OSGB36 / British National Grid", "metres", "Great Britain"),
    (3857, "WGS 84 / Pseudo-Mercator", "metres", "web maps"),
    (4326, "WGS 84", "degrees", "GPS positions"),
    (32633, "WGS 84 / UTM zone 33N", "metres", "the eastern hemisphere"),
    (2154, "RGF93 / Lambert-93", "metres", "France"),
]

COMMANDS = [
    "-LAYER", "AUDIT", "PURGE", "ZOOM", "MAPIMPORT", "MAPEXPORT",
    "MAPCSASSIGN", "MAPCLEAN", "-BLOCK", "-INSERT", "-HATCH", "-PURGE",
    "-UNITS", "ADEQUERY",
]

REASONS = [
    "they print at different scales",
    "they belong to different trades",
    "one is existing and one is proposed",
    "a query should be able to select one of them",
    "the plot style treats them differently",
]

INTRO = """amu corpus

amu is a small language model trained from scratch on this text.
amu reads one character at a time and predicts the next character.
amu has no pretrained weights and no external service.
The model is a character-level transformer, and the tokenizer is the set of characters in this file.
The agent around amu is deterministic: it matches a request to a recipe and fills in the parameters.
The model and the agent are separate, and the model does not route requests.
Every sentence here is about drawing, mapping, building models or the scripts that change them.

"""

PROSE = [
    ("AutoCAD drawing basics", [
        "A drawing is the container for everything you draw in AutoCAD, and it is saved as a DWG file.",
        "Model space holds the geometry, and paper space holds the layouts that print it.",
        "A layout holds a title block and one or more viewports, and each viewport shows model space at a chosen scale.",
        "The command UNITS sets the units of the drawing, and the command INSUNITS records them for other applications.",
        "A polyline is one object with many segments, and PLINE draws it segment by segment.",
        "A block is a reusable group of objects, and one edit to the block updates every insertion.",
        "An xref links another drawing into the current drawing instead of copying its objects.",
        "The command AUDIT checks a drawing for errors, and the command PURGE removes unused blocks, layers and linetypes.",
        "A script with the SCR extension replays command line input, one line for every prompt.",
        "A drawing that mixes metres and millimetres will not measure correctly, so agree on units first.",
        "Layer 0 is special: objects on layer 0 take the properties of the block that contains them.",
        "The command ZOOM with the option Extents fits the drawing to the screen, which is useful after an import.",
        "An object has a colour, a layer, a linetype and a lineweight, and each of them can be set directly or by layer.",
        "A viewport can freeze a layer for one layout without freezing it in the model.",
        "A template drawing with the right layers and styles removes that work from every new drawing.",
    ]),

    ("Layers and scripts", [
        "The {layer} layer and the {layer2} layer stay separate, because {reason}.",
        "Create the {layer} layer before drawing, so that every object lands in the right place.",
        "The command -LAYER changes layer state from the command line, which is what a script needs.",
        "A script sets the current layer, draws, and restores the previous layer before it ends.",
        "Colour, linetype and lineweight belong to the layer, not to the object, when objects use BYLAYER.",
        "The layer {layer} is drawn in {color} so that the plot reads cleanly.",
        "The command LAYERP opens the layer properties manager, and -LAYER works without a dialog.",
        "A layer filter groups layers by name, and a saved filter makes a large layer list usable.",
        "Freeze a layer when its objects should not be computed, and turn it off when they should only be hidden.",
        "Lock a layer when you want to see it but not edit it, because a locked object cannot be selected for editing.",
        "A script that creates layers twice fails on the second run unless it checks whether the layer exists.",
        "The command -PURGE with the option All removes the unused layers that a script leaves behind.",
        "Name layers in a fixed pattern so that a filter, a query or an export can find them later.",
        "A layer with no objects is not free to keep if it also carries a plot style that nobody understands.",
    ]),

    ("AutoLISP", [
        "AutoLISP is the scripting language built into AutoCAD, and a program is stored in a LSP file.",
        "The function defun creates a command, and a command whose name starts with c: can be typed at the prompt.",
        "The function setq assigns a value, and the function list builds a list.",
        "Strings in AutoLISP are escaped so that a quote, a backslash or a newline cannot break the program.",
        'The function ssget builds a selection set, and the filter (cons 8 "ROADS") selects one layer.',
        "The function command calls an AutoCAD command, and it expects the same answers that you would type.",
        "A program ends with (princ), which prints nothing and returns quietly to the prompt.",
        "A command that begins with c: runs like a built-in command, and a function without c: must be called from another program.",
        "Wrap file paths with the function findfile, because a path with spaces or a backslash is easy to break.",
        "A LISP file that is loaded with APPLOAD stays in memory until the drawing closes.",
        "Check the parentheses before you load a program, because one missing parenthesis stops the whole file.",
        "Read the command line prompts before you script a command, because every option changes the sequence of answers.",
        "A selection set can be iterated with ssname and sslength, and a loop over it edits one entity at a time.",
        "An entity name is valid until the entity is deleted, so finish reading it before you erase it.",
    ]),

    ("AutoCAD Map 3D", [
        "AutoCAD Map 3D adds geographic data to AutoCAD, and the drawing stays the centre of the work.",
        "A coordinate system maps drawing coordinates to positions on the earth, and an EPSG code names that system.",
        "EPSG {code} is {name}, measured in {unit}, and it is used in {region}.",
        "The command MAPCSASSIGN assigns a coordinate system to the drawing and warns when one is already assigned.",
        "The command MAPIMPORT brings a shapefile into the drawing, and the command MAPEXPORT writes drawing objects back out.",
        "A shapefile is a set of files, and the DBF file holds the attributes that the SHP file cannot.",
        "Object data lives inside the drawing, and a feature data object connection lives outside it.",
        "The command MAPCLEAN closes small gaps, removes duplicates and fixes overshoots.",
        "A topology records how lines connect and where polygons share a boundary.",
        "Do not mix drawing units, because a line drawn in millimetres will not line up with a parcel measured in metres.",
        "Check the coordinate system before an export, because the export writes the coordinates it is given.",
        "The command ADEQUERY filters drawing objects before an export, so a large drawing can be split by attribute.",
        "A drawing that arrives without a coordinate system is a sketch, not a map.",
        "Map 3D keeps the original drawing objects, so an import can be reversed more easily than a conversion.",
    ]),

    ("Coordinates and projections", [
        "A geographic coordinate system uses latitude and longitude, and a projected coordinate system uses metres on a flat surface.",
        "A projection always distorts something, so choose one that keeps the measurement you care about.",
        "EPSG {code} is {name}, and a coordinate in it is given in {unit}.",
        "A datum shift can move features by metres, so never assume that two systems agree.",
        "Reprojection changes coordinates, not the shapes on the ground.",
        "Store coordinates in the system you will analyse in, and reproject at the edge of the work.",
        "Web Mercator is convenient for tiled maps and wrong for area measurements.",
        "A false easting keeps coordinates positive, and a false northing does the same for the other axis.",
        "The zone number in a UTM system changes every six degrees of longitude.",
        "A coordinate without a system is a number without a meaning.",
        "The command ogr2ogr can reproject while it converts, with the option -t_srs.",
        "A PRJ file stores the coordinate system of a shapefile, and a missing PRJ means an unknown system.",
        "A transformation between two datums can need a grid file, and without it the result is only approximate.",
        "Check the extent after a reprojection, because a wrong zone moves the whole dataset.",
    ]),

    ("QGIS", [
        "QGIS is a free geographic information system that runs on Windows, macOS and Linux.",
        "A QGIS project is saved as a QGZ file, and a GeoPackage stores many layers in one SQLite file.",
        "The OGR provider reads a shapefile, and the same provider writes a GeoPackage layer.",
        "The processing toolbox runs buffer, clip, dissolve, intersection and spatial join.",
        "The command qgis_process runs those algorithms without a window, which makes it usable from a script.",
        "PyQGIS exposes layers, features, geometry and the processing registry to Python.",
        "Reproject a layer when its coordinate reference system does not match the project.",
        "A buffer builds a polygon at a fixed distance around each feature, and a negative distance shrinks it.",
        "A spatial join copies attributes from the layer you join onto the layer you join to.",
        "The field calculator evaluates an expression for every feature and writes the result into a field.",
        "A print layout holds a map, a legend, a scale bar and a title, and it exports to PDF.",
        "Save the style with the layer, because a colour ramp lost on reload is work you repeat.",
        "The layer panel shows the drawing order, and the layer on top hides everything below it.",
        "A temporary scratch layer disappears when the project closes, so copy it before you close.",
        "A geometry check finds invalid polygons, and the fix geometry tool repairs them.",
    ]),

    ("PyQGIS and qgis_process", [
        "The class QgsProject holds the open project, and the method mapLayers returns its layers.",
        "A vector layer is opened with QgsVectorLayer, and the second argument names the provider.",
        "The method addMapLayer adds a layer to the project, and the method removeMapLayer takes it away.",
        "A feature is read with getFeatures, and each feature carries a geometry and a set of attributes.",
        "The processing module runs an algorithm with run, and the parameters are passed as a dictionary.",
        "An algorithm name looks like native:buffer, and the prefix names the provider that supplies it.",
        "A feedback object carries progress and cancellation, and a long job should check it.",
        "The method commitChanges saves edits to the data source, and the method rollBack undoes them.",
        "A memory layer keeps features in memory, which is useful for a result that does not need saving.",
        "The class QgsCoordinateTransform changes coordinates between two reference systems.",
        "A print layout is exported with the class QgsLayoutExporter, and the format is set on the export settings.",
        "A script that runs inside QGIS has a running application, and a script that runs outside it must create one.",
        "The distance parameter of a buffer is in the units of the layer, not in the units of the project.",
        "A field that does not exist is created with a field definition and an edit buffer, and the change is committed after the loop.",
    ]),

    ("Revit and building data", [
        "Revit stores a building model in an RVT file and organises it by categories, families and types.",
        "A level is a horizontal reference plane, and every wall is hosted by one.",
        "A family is a parametric component, and a family instance is one placement of it.",
        "A type holds the dimensions, and an instance holds the position.",
        "A schedule reads parameters from elements and lists them in a table.",
        "A sheet is a printable page that holds views, and sheets export to PDF in a batch.",
        "A transaction groups changes so that they are saved or undone together.",
        "The Revit API is reached with Python through pyRevit, and every change must be inside a transaction.",
        "Read the model before you write to it, because a script that guesses will overwrite work.",
        "Shared parameters travel between projects, and project parameters stay in one model.",
        "A workset splits a model so that several people can work in it at the same time.",
        "Purge unused families before delivery, because an RVT file carries everything it ever held.",
        "The document class gives access to the model, and the active view is not always the view you want.",
        "A filter collects elements by category and parameter, and a collector without a filter returns the whole model.",
        "An element id is stable inside one model, and it is not a reliable key between models.",
        "A warning in Revit is not an error, but a batch of warnings usually hides one.",
    ]),

    ("Data exchange", [
        "Every format loses something, so decide what matters before you convert.",
        "A SHP file stores geometry and attributes, and it loses styling, labels and relationships.",
        "A DXF file stores drawing geometry and layers, and it is the safest exchange format for AutoCAD.",
        "A GeoPackage stores many layers, long field names and spatial indexes in one file.",
        "A CSV file stores attributes without geometry, so a coordinate pair must be carried in two fields.",
        "A coordinate pair in a CSV needs a declared order, because latitude first and longitude first look alike.",
        "Run a count before and after a conversion, and compare the totals.",
        "Check the extent after a reprojection, because a wrong zone moves the whole dataset.",
        "A field name longer than ten characters is truncated by a shapefile, and the truncation can collide.",
        "A date field is text in a shapefile and a real date in a GeoPackage, so check it after a conversion.",
        "Keep the source file read only, and write every result into a new folder.",
        "A checksum of the source makes a repeatable conversion verifiable.",
        "A DXF file with a different unit setting can arrive ten times too large or ten times too small.",
        "An export that keeps object data is not the same as an export that keeps plain attributes, so say which one the receiver needs.",
    ]),

    ("Automation safety", [
        "Generate the script, read it, then run it, because a script repeats every mistake you make very fast.",
        "Test a script on a copy of the drawing, never on the only copy.",
        "A script that deletes or overwrites needs a confirmation flag before it runs.",
        "Log every change with the file, the recipe and the time, so a bad run can be traced.",
        "Keep the source data read only, and write results into a new folder.",
        "Never build a command by joining strings from a request, because a stray quote changes the command.",
        "Pin the tool version, because an algorithm that changes behaviour changes your result.",
        "Run the audit before and after, and compare the counts.",
        "A dry run prints the steps without touching data, and it costs seconds.",
        "Back up the drawing before a batch, and keep the backup until the batch is checked.",
        "An allowlist of binaries is safer than a blocklist, because a blocklist always misses one.",
        "A path from a request is data, not code, and it should be quoted as data.",
        "A confirmation prompt is not a security boundary when the same flag can be passed by a script.",
        "Keep the working folder small, because a folder with ten thousand files slows every tool.",
    ]),

    ("Troubleshooting", [
        "If a layer is missing, check that the name matches exactly, because a script does not guess.",
        "If a script stops halfway, run it line by line and find the first prompt that differs.",
        "If a shapefile will not open, check that the SHX and DBF files sit next to the SHP file.",
        "If features land in the wrong place, check the coordinate system before you move anything.",
        "If a polygon has a hole, check the ring order and the closing point.",
        "If a buffer looks wrong, check the units of the distance.",
        "If a join returns nothing, check the field type, because text and numbers do not match.",
        "If an export is empty, check the selection and the layer visibility.",
        "If a PDF is blank, check the plot window and the page setup.",
        "If the model is slow, check the number of open drawings and the size of the cache.",
        "If the result differs between runs, check the random seed and the tool version.",
        "If a file is locked, check for a second application that still has it open.",
        "If a command is unknown, check the application version, because commands move between releases.",
        "If a dialog appears in a script, the script is waiting for an answer that will never come.",
    ]),

    ("OCR and scanned drawings", [
        "The OCR pass reads the title block and returns the project name and the sheet number.",
        "Tesseract reads a clean scan well and a photograph badly, so deskew and threshold first.",
        "The page segmentation mode tells tesseract how to divide the page into blocks.",
        "A confidence below sixty usually means the scan is rotated, noisy or too small.",
        "Upscale a small scan before OCR, because tesseract expects around three hundred dots per inch.",
        "OCR returns text, not geometry: a detected label still needs a check against the drawing.",
        "A rotated scan is deskewed before OCR, otherwise every line comes back as fragments.",
        "The OCR output is saved as text next to the image, so the next step does not repeat the work.",
        "Languages are named by their traineddata file, for example eng or deu.",
        "A title block is the easiest text to read, because the letters are large and the contrast is high.",
        "Handwriting is not what tesseract is built for, and the confidence shows it.",
        "If OCR finds nothing, check that the image is not blank and that the text is not lighter than the background.",
        "A table comes back as loose words, so the columns have to be rebuilt from their coordinates.",
        "The OCR text is a hint for the operator, not a source of coordinates.",
        "A page segmentation mode of six treats the page as one uniform block of text.",
        "A confidence of ninety or more is usually correct, and a confidence of thirty is usually wrong.",
    ]),

    ("Writing scripts", [
        "A generated program starts with its imports, reads the document, and changes it inside one transaction.",
        "Keep every function short, because a long function is hard to review before it runs.",
        "Print what the program did, so the operator can see the result without opening the file.",
        "Write the output path into the script, and never overwrite the input file.",
        "A script that runs twice should produce the same result, or it should refuse the second run.",
        "Quote every path and every layer name, because a space or a quote breaks an unquoted command.",
        "One recipe writes one artifact, so a failed run leaves one file to delete.",
        "The header of a script names the drawing, the date and the recipe that produced it.",
        "A comment above a block explains why the block exists, not what each line does.",
        "Test the program on a copy of the drawing, then read its output before running it on the real model.",
        "An error handler rolls the transaction back and prints the message, so a failure leaves no partial work.",
        "Keep the command line and the script in the same folder, so the two do not drift apart.",
        "A Python script for QGIS needs a QgsApplication when it runs outside the desktop.",
        "The script prints a summary line with the counts of created, changed and skipped objects.",
    ]),

    ("Glossary", [
        "A vertex is a point where two segments of a line meet.",
        "A segment is the straight piece between two vertices.",
        "A ring is a closed line that bounds a polygon.",
        "A polygon is a closed area bounded by one outer ring and any number of inner rings.",
        "A multipolygon holds several polygons that belong to one feature.",
        "An attribute is a value attached to a feature, and a field is the column that holds it.",
        "A feature is one row of geometry and attributes in a vector layer.",
        "A raster cell is one square of a grid, and its value is the measurement stored there.",
        "A nodata value marks a cell with no measurement, and it is not the same as zero.",
        "A spatial index is a lookup structure that makes a location query fast.",
        "A topology is a set of rules about how features connect and share edges.",
        "A tolerance is the distance below which two points are treated as one.",
        "A sliver is a long thin polygon created by an imperfect overlay.",
        "An overlay combines two layers and keeps the parts that intersect.",
        "A dissolve merges features that share a value into one feature.",
        "A centroid is the centre of mass of a geometry, and it can fall outside a concave polygon.",
        "A bounding box is the smallest rectangle that contains a geometry.",
        "A viewport is a window on a layout that shows model space.",
    ]),

    ("Delivery notes", [
        "Deliver the drawing, the source data, the script and a short note that says what changed.",
        "Name the output with the date and the recipe, so a later run does not overwrite an earlier one.",
        "Keep a copy of the inputs next to the outputs when the job is small.",
        "Record the coordinate system in the note, because a later reader cannot recover it from the geometry alone.",
        "Record the tool version in the note, because a result without a version cannot be repeated.",
        "A drawing that has been audited, purged and saved is ready to send.",
        "A layer list is part of the deliverable, because the next person needs to know what each layer means.",
        "A plot style is part of the deliverable, because a drawing that prints black is not the drawing you approved.",
        "A PDF is for review, and the DWG is for editing, so send both when both are needed.",
        "A zip file with a flat folder is easier to open than a deep tree of folders.",
        "Remove the temporary files before you zip, and check the archive by opening it.",
        "A delivery note is short: what it is, what changed, what is missing.",
    ]),
]

DIALOGUES = [
    ("create layers roads, parcels, contours in red",
     "Recipe autocad.create_layers matched. Wrote layers.scr with three -LAYER calls and a layer list note."),
    ("create levels at 0, 3.5, 7, 10.5",
     "Recipe revit.create_levels matched. Wrote create_levels.py, which opens one transaction and creates four levels."),
    ("reproject parcels.shp to EPSG:25832",
     "Recipe qgis.reproject matched. Wrote reproject.py and the qgis_process command with -t_srs EPSG:25832."),
    ("import shapefile into autocad map 3d",
     "Recipe map3d.import_shapefile matched. Wrote the ogr2ogr conversion and the MAPCSASSIGN and MAPIMPORT steps."),
    ("buffer roads by 25 metres and clip to parcels",
     "Recipe qgis.buffer_clip matched. Wrote the PyQGIS script with a 25 metre buffer and a clip to the parcel layer."),
    ("export attributes of blocks to csv",
     "Recipe autocad.attributes_to_csv matched. Wrote att2csv.lsp, which writes one row per block reference."),
    ("audit and purge this drawing",
     "Recipe autocad.audit_purge matched. Wrote audit_purge.scr, and the note says to run it on a copy first."),
    ("search the web for qgis native buffer",
     "The search returned a QGIS documentation page, and the agent read the top result before answering."),
    ("describe site_plan.png",
     "The image is 2400 by 1700 pixels, with 4.2 percent ink coverage and content that reaches the right edge."),
    ("what can you do",
     "amu writes scripts for Revit, AutoCAD, AutoCAD Map 3D and QGIS, reads web pages, and measures images."),
]

CODE = [
    ("create levels with a function in pyRevit", '''from Autodesk.Revit.DB import Level, Transaction


def create_levels(doc, elevations):
    transaction = Transaction(doc, "Create levels")
    transaction.Start()
    try:
        for elevation in elevations:
            Level.Create(doc, elevation)
        transaction.Commit()
    except Exception:
        transaction.RollBack()
        raise


doc = __revit__.ActiveUIDocument.Document
create_levels(doc, [0.0, 3.5, 7.0, 10.5])
'''),

    ("create levels in pyRevit", '''from Autodesk.Revit.DB import Level, Transaction

doc = __revit__.ActiveUIDocument.Document
elevations = [0.0, 3.5, 7.0, 10.5]

transaction = Transaction(doc, "Create levels")
transaction.Start()
try:
    for elevation in elevations:
        Level.Create(doc, elevation)
    transaction.Commit()
except Exception:
    transaction.RollBack()
    raise
'''),

    ("read wall marks in pyRevit", '''from Autodesk.Revit.DB import BuiltInCategory, FilteredElementCollector

doc = __revit__.ActiveUIDocument.Document
walls = FilteredElementCollector(doc).OfCategory(
    BuiltInCategory.OST_Walls
).WhereElementIsNotElementType()

for wall in walls:
    mark = wall.LookupParameter("Mark")
    if mark is not None:
        print(wall.Id, mark.AsString())
'''),

    ("buffer and clip in PyQGIS", '''from qgis.core import QgsProject, QgsVectorLayer
import processing

roads = QgsVectorLayer("roads.shp", "roads", "ogr")
parcels = QgsVectorLayer("parcels.shp", "parcels", "ogr")

buffered = processing.run("native:buffer", {
    "INPUT": roads,
    "DISTANCE": 25.0,
    "SEGMENTS": 8,
    "END_CAP_STYLE": 0,
    "JOIN_STYLE": 0,
    "DISSOLVE": False,
    "OUTPUT": "memory:buffer",
})["OUTPUT"]

clipped = processing.run("native:clip", {
    "INPUT": buffered,
    "OVERLAY": parcels,
    "OUTPUT": "buffer_clip.gpkg",
})["OUTPUT"]

QgsProject.instance().addMapLayer(clipped)
'''),

    ("report layers with a function in PyQGIS", '''from qgis.core import QgsProject


def layer_report(project):
    for layer in project.mapLayers().values():
        print(layer.name(), layer.featureCount())


layer_report(QgsProject.instance())
'''),

    ("reproject with qgis_process", '''qgis_process run native:reprojectlayer --INPUT=parcels.shp --TARGET_CRS=EPSG:25832 --OUTPUT=parcels_25832.gpkg'''),

    ("convert a shapefile with ogr2ogr", '''ogr2ogr -f GPKG parcels.gpkg parcels.shp -t_srs EPSG:25832 -nln parcels
ogr2ogr -f DXF roads.dxf roads.gpkg -nln roads'''),

    ("create layers in AutoLISP", '''(defun c:TMK-LAYERS ( / layers )
  (setq layers '(
    ("ROADS" 1)
    ("PARCELS" 2)
    ("CONTOURS" 3)
  ))
  (foreach item layers
    (if (not (tblsearch "LAYER" (car item)))
      (command "-LAYER" "New" (car item) "Color" (cadr item) (car item) "")
    )
  )
  (princ)
)'''),

    ("export block handles in AutoLISP", '''(defun c:TMK-ATT2CSV ( / ss file index entity handle block )
  (setq ss (ssget "_X" '((0 . "INSERT"))))
  (setq file (open "C:/work/attributes.csv" "w"))
  (write-line "handle,block" file)
  (setq index 0)
  (while (< index (sslength ss))
    (setq entity (ssname ss index))
    (setq handle (cdr (assoc 5 (entget entity))))
    (setq block (cdr (assoc 2 (entget entity))))
    (write-line (strcat handle "," block) file)
    (setq index (1+ index))
  )
  (close file)
  (princ)
)'''),

    ("count objects on a layer in AutoLISP", '''(defun c:TMK-COUNT ( / ss )
  (setq ss (ssget "_X" '((8 . "ROADS"))))
  (if ss
    (princ (strcat "Selected " (itoa (sslength ss)) " objects."))
    (princ "No objects found on ROADS.")
  )
  (princ)
)'''),

    ("create layers in a script file", '''-LAYER
New
ROADS,PARCELS,CONTOURS

-COLOR
RED
ROADS

ZOOM
Extents
AUDIT
Y
-PURGE
All
'''),

    ("list layers in a GeoPackage", '''from osgeo import ogr

source = ogr.Open("site.gpkg")
for index in range(source.GetLayerCount()):
    layer = source.GetLayerByIndex(index)
    print(layer.GetName(), layer.GetFeatureCount())
'''),

    ("read a title block with pytesseract", '''from PIL import Image
import pytesseract

image = Image.open("title_block.png").convert("L")
text = pytesseract.image_to_string(image, config="--psm 6")
print(text)'''),
]


def _slots(rng):
    """Every value a template may use, drawn from the lists above."""
    code, name, unit, region = rng.choice(SYSTEMS)
    first = rng.choice(LAYERS)
    second = rng.choice([layer for layer in LAYERS if layer != first])

    return {
        "layer": first,
        "layer2": second,
        "color": rng.choice(COLORS),
        "fmt": rng.choice(FORMATS),
        "fmt2": rng.choice(FORMATS),
        "code": code,
        "name": name,
        "unit": unit,
        "region": region,
        "reason": rng.choice(REASONS),
        "cmd": rng.choice(COMMANDS),
    }


def _paragraph(rng, templates, sentences):
    chosen = rng.sample(templates, k=min(sentences, len(templates)))
    return " ".join(template.format(**_slots(rng)) for template in chosen)


def _dialogue_block(rng, count):
    pairs = rng.sample(DIALOGUES, k=min(count, len(DIALOGUES)))
    lines = ["Example sessions"]
    for request, answer in pairs:
        lines.append("User: %s" % request)
        lines.append("amu: %s" % answer)
    return "\n".join(lines)


def build(seed=DEFAULT_SEED, target=DEFAULT_TARGET):
    """Return the corpus text: deterministic for a given seed and size."""
    rng = random.Random(seed)
    blocks = [INTRO]
    length = len(INTRO)
    code_queue = []

    while length < target:
        order = list(PROSE)
        rng.shuffle(order)

        for heading, templates in order:
            parts = [heading]
            for _ in range(rng.randint(2, 3)):
                parts.append(_paragraph(rng, templates, rng.randint(3, 5)))

            block = "\n\n".join(parts) + "\n\n"
            blocks.append(block)
            length += len(block)

            if length >= target:
                break

        dialogue = _dialogue_block(rng, 5) + "\n\n"
        blocks.append(dialogue)
        length += len(dialogue)

        for _ in range(2):
            if not code_queue:
                code_queue = list(CODE)
                rng.shuffle(code_queue)

            heading, source = code_queue.pop()
            block = "Code: %s\n\n%s\n\n" % (heading, source.strip())
            blocks.append(block)
            length += len(block)

    text = "".join(blocks)

    if len(text) > target:
        cut = text.rfind("\n\n", 0, target)
        text = text[:cut if cut > 0 else target]

    return text.strip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="approximate corpus size in characters")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="generator seed; the same seed gives the same text")
    parser.add_argument("--out", default="data/input.txt",
                        help="where to write the corpus")
    args = parser.parse_args(argv)

    text = build(seed=args.seed, target=args.target)

    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    print(
        "wrote %s: %d characters, %d lines, %d distinct characters"
        % (args.out, len(text), text.count("\n"), len(set(text)))
    )


if __name__ == "__main__":
    main()
