"""amu CAD/GIS agent.

A local agent for Revit, AutoCAD, AutoCAD Map 3D and QGIS. It routes a request
to a recipe, fills the recipe's parameters from the request, and writes the
scripts that do the work. Execution is separate, allowlisted and opt-in.
"""

from .core import Agent, Report
from .planner import Plan, Planner
from .registry import all_recipes, by_domain, get_recipe
from .types import Artifact, ExecuteSpec, Param, Recipe, ToolResult

__all__ = [
    "Agent",
    "Artifact",
    "ExecuteSpec",
    "Param",
    "Plan",
    "Planner",
    "Recipe",
    "Report",
    "ToolResult",
    "all_recipes",
    "by_domain",
    "get_recipe",
]

__version__ = "0.1.0"
