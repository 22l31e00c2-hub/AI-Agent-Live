"""Command line interface for the amu CAD/GIS agent."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

from .core import Agent
from .registry import all_recipes


def _parse_overrides(pairs: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit("--param expects name=value, got %r" % pair)
        name, value = pair.split("=", 1)
        overrides[name.strip()] = value.strip()
    return overrides


def _print_recipes() -> None:
    print("%-28s %-8s %-8s %s" % ("id", "domain", "mutates", "title"))
    print("-" * 100)
    for recipe in all_recipes():
        print(
            "%-28s %-8s %-8s %s"
            % (recipe.id, recipe.domain, "yes" if recipe.mutates else "no", recipe.title)
        )


def _interactive(agent: Agent, domain: Optional[str]) -> int:
    print("amu CAD/GIS agent - describe a task, or type 'exit'.")
    print("Applications: Revit, AutoCAD, AutoCAD Map 3D, QGIS.")
    print("I also read web pages ('look up https://...') and images ('describe plan.png').")

    while True:
        try:
            request = input("\ntask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if request.lower() in ("exit", "quit"):
            return 0
        if not request:
            continue

        report = agent.answer(request, domain=domain)
        print()
        print(report.text)

        if report.plan.missing:
            overrides: Dict[str, str] = {}
            for param in report.plan.missing:
                try:
                    answer = input("    %s " % param.question).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if answer:
                    overrides[param.name] = answer

            if overrides:
                report = agent.answer(request, domain=domain, overrides=overrides)
                print()
                print(report.text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cadagent",
        description=(
            "Local CAD/GIS agent for Revit, AutoCAD, AutoCAD Map 3D and QGIS. "
            "It plans the work and writes the scripts; running them is opt-in. "
            "It can also read web pages and inspect images."
        ),
    )
    parser.add_argument("request", nargs="*", help="what you want to do")
    parser.add_argument(
        "--domain",
        choices=["revit", "autocad", "map3d", "qgis"],
        help="force a target application instead of guessing",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fill a recipe parameter (repeatable)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="also run the recipe's command (allowlisted tools only)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm execution that changes your data",
    )
    parser.add_argument("--fetch", metavar="URL", help="read a web page and summarise it")
    parser.add_argument(
        "--search", metavar="QUERY", help="search the web and list the results"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore cached pages and fetch them again",
    )
    parser.add_argument(
        "--image", metavar="PATH", help="inspect an image (needs Pillow)"
    )
    parser.add_argument(
        "--workspace",
        default="ws",
        help="folder for generated files, cache and the session log (default: ws)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--list-recipes", action="store_true", help="show capabilities")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="show which tools this machine can actually execute",
    )
    args = parser.parse_args(argv)

    if args.list_recipes:
        _print_recipes()
        return 0

    agent = Agent(
        workspace=args.workspace,
        execute=args.execute,
        assume_yes=args.yes,
    )

    if args.doctor:
        print(agent.doctor())
        return 0

    overrides = _parse_overrides(args.param)

    if args.fresh:
        agent.cache_ttl = 0

    if args.fetch:
        request = args.fetch
        report = agent.web_report(args.fetch)
    elif args.search:
        request = args.search
        report = agent.search_report(args.search, query=args.search)
    elif args.image:
        request = args.image
        report = agent.vision_report(args.image)
    elif args.request:
        request = " ".join(args.request).strip()
        report = agent.answer(request, domain=args.domain, overrides=overrides)
    else:
        return _interactive(agent, args.domain)

    if args.json:
        payload = {
            "request": request,
            "capability": report.capability,
            "domain": report.plan.domain,
            "recipe": report.plan.recipe.id if report.plan.recipe else None,
            "score": report.plan.score,
            "params": report.plan.params,
            "missing": [param.name for param in report.plan.missing],
            "files": [agent.rel_path(path) for path in report.files],
            "executed": bool(report.outcome and report.outcome.ran),
            "execution": None,
            "error": report.error,
            "text": report.text,
        }
        if report.outcome is not None:
            payload["execution"] = {
                "ran": report.outcome.ran,
                "reason": report.outcome.reason,
                "returncode": report.outcome.returncode,
                "stdout": report.outcome.stdout,
                "stderr": report.outcome.stderr,
            }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(report.text)

    return 0 if not report.error else 1


if __name__ == "__main__":
    sys.exit(main())
