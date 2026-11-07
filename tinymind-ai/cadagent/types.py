"""Shared datatypes for the amu CAD/GIS agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Param:
    """One piece of information a recipe needs from the user."""

    name: str
    question: str
    kind: str = "text"      # text | number | numbers | epsg | path | csv | layers | flag
    required: bool = True
    default: Any = None


@dataclass
class Artifact:
    """A generated file (script, command sheet, report)."""

    filename: str
    language: str
    content: str


@dataclass
class ExecuteSpec:
    """A command the agent is allowed to run *only* with explicit consent.

    ``argv`` is a list, never a shell string, and ``argv[0]`` must be on the
    runner allowlist (ogr2ogr, qgis_process, gdalinfo).
    """

    argv: List[str]
    description: str = ""


@dataclass
class ToolResult:
    title: str
    artifacts: List[Artifact] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    execute: Optional[ExecuteSpec] = None


@dataclass
class Recipe:
    """A reusable CAD/GIS task: how to recognise it and how to do it."""

    id: str
    domain: str              # revit | autocad | map3d | qgis
    title: str
    summary: str
    triggers: List[str]      # lowercase phrases that select this recipe
    params: List[Param]
    mutates: bool            # True => changes a drawing/model/project in place
    build: Callable[[Dict[str, Any]], ToolResult]
    needs: List[str] = field(default_factory=list)
