"""Recipe registry: every capability the agent knows about."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .knowledge import autocad, map3d, qgis, revit
from .types import Recipe

_MODULES = (revit, autocad, map3d, qgis)

_RECIPES: Dict[str, Recipe] = {}


def _load() -> Dict[str, Recipe]:
    if _RECIPES:
        return _RECIPES

    for module in _MODULES:
        for recipe in module.RECIPES:
            if recipe.id in _RECIPES:
                raise ValueError("Duplicate recipe id: %s" % recipe.id)
            _RECIPES[recipe.id] = recipe
    return _RECIPES


def all_recipes() -> List[Recipe]:
    return list(_load().values())


def get_recipe(recipe_id: str) -> Recipe:
    try:
        return _load()[recipe_id]
    except KeyError:
        raise KeyError(
            "Unknown recipe %r. Try `--list-recipes`." % recipe_id
        )


def by_domain(domain: str) -> List[Recipe]:
    return [r for r in all_recipes() if r.domain == domain]


def recipe_ids() -> Iterable[str]:
    return sorted(_load())
