"""Rule-based replies for greetings, identity and capabilities.

Important: this is *not* amu's neural network. ``model.py`` is a character-level
GPT prototype that learns letter statistics from a text file, so it cannot hold
a conversation. These replies are deterministic strings, so the CLI can answer
"who are you" honestly instead of pretending the network did it.

Recipe routing always runs first, so a greeting that also carries a task
("hey amu, create layers roads") still becomes a task.
"""

from __future__ import annotations

import re
from typing import List, Optional

CREATOR = "Yuva"

GREETING_RE = re.compile(
    r"\b(hey|hi|hello|yo|good morning|good afternoon|good evening)\b", re.I
)
NAME_RE = re.compile(
    r"\b(?:i am|i'm|iam|my name is|this is)\s+([A-Za-z][A-Za-z\-']{1,30})", re.I
)
THANKS_RE = re.compile(r"\b(thanks|thank you|cheers)\b", re.I)
HELP_RE = re.compile(r"\bhelp\b", re.I)

IDENTITY_KEYS = (
    "who are you", "what are you", "your name", "are you an ai",
    "are you a chatbot", "are you human", "are you real", "what is amu",
    "who is amu",
)
CREATOR_KEYS = (
    "who made you", "who built you", "who created you", "your creator",
    "creator of you", "who wrote you", "who trained you", "your author",
)
HELP_KEYS = (
    "what can you do", "commands", "capabilities", "what do you do",
    "how do i use",
)

DOMAIN_LABEL = {
    "revit": "Revit",
    "autocad": "AutoCAD",
    "map3d": "AutoCAD Map 3D",
    "qgis": "QGIS",
}


def _name(request: str) -> Optional[str]:
    match = NAME_RE.search(request)
    if not match:
        return None
    name = match.group(1).strip("-'")
    if not name:
        return None
    return name[:1].upper() + name[1:]


def _capabilities(recipes: List) -> str:
    counts = {}
    for recipe in recipes:
        counts[recipe.domain] = counts.get(recipe.domain, 0) + 1

    parts = [
        "%s (%d)" % (DOMAIN_LABEL.get(domain, domain), counts[domain])
        for domain in ("revit", "autocad", "map3d", "qgis")
        if counts.get(domain)
    ]

    return (
        "I can plan work for %s - %d recipes in total. I read a sentence, fill "
        "in the parameters, and write the scripts (pyRevit, AutoLISP, .scr, "
        "PyQGIS, ogr2ogr). Try `python -m cadagent --list-recipes` for the full "
        "list, or ask something like \"buffer the roads layer 25 and clip it to "
        "the site boundary\"." % (", ".join(parts), len(recipes))
    )


def respond(request: str, recipes: List) -> Optional[str]:
    """Reply to a social/identity request. Returns None for real tasks."""
    text = request.lower()
    name = _name(request)
    who = (" Nice to meet you, %s." % name) if name else ""

    if any(key in text for key in CREATOR_KEYS):
        return (
            "Yuva created me - I am amu: the architecture, the training loop, "
            "the vocabulary and this agent are all local to this project.%s "
            "I run on your machine - no pretrained weights, no external API, "
            "no paid service." % who
        )

    if any(key in text for key in IDENTITY_KEYS):
        return (
            "I am amu. There are two parts to me: a small GPT-style language "
            "model written from scratch (model.py, trained by you with "
            "train.py) and a deterministic agent for Revit, AutoCAD, Map 3D and "
            "QGIS (cadagent). To be exact about this conversation: these replies "
            "are rule-based, not model output - the network is a character-level "
            "prototype."
        )

    if any(key in text for key in HELP_KEYS) or HELP_RE.search(text):
        return _capabilities(recipes)

    if THANKS_RE.search(text):
        return "Anytime.%s Give me the next task when you are ready." % who

    if GREETING_RE.search(text):
        return (
            "Hey!%s I am amu, your local CAD/GIS agent for Revit, AutoCAD, "
            "Map 3D and QGIS. Tell me a task, for example \"create layers "
            "roads, parcels in red\" or \"reproject parcels.shp to "
            "EPSG:25832\"." % who
        )

    return None
