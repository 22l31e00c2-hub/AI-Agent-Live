"""The agent loop: understand -> plan -> generate -> (guarded) execute -> record.

Order of resolution is deliberate:

1. an explicit "search the web" request,
2. a CAD/GIS recipe (a real task always wins),
3. an image path (vision),
4. a URL (web reading),
5. social/identity replies,
6. the honest "I do not have a recipe for that".
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import vision, web
from .planner import Plan, Planner
from .registry import all_recipes
from .runner import Outcome, capabilities, run_spec
from .social import respond as social_respond
from .types import Artifact, ToolResult

DOMAIN_LABEL = {
    "revit": "Revit",
    "autocad": "AutoCAD",
    "map3d": "AutoCAD Map 3D",
    "qgis": "QGIS",
}


@dataclass
class Report:
    """One agent response: what was planned, written and executed."""

    plan: Plan
    result: Optional[ToolResult] = None
    files: List[Path] = field(default_factory=list)
    outcome: Optional[Outcome] = None
    execution_note: str = ""
    error: str = ""
    text: str = ""
    capability: str = "recipe"     # recipe | web | vision | social | none


class Agent:
    def __init__(
        self,
        workspace: str = "ws",
        execute: bool = False,
        assume_yes: bool = False,
    ):
        self.planner = Planner(all_recipes())
        self.workspace = Path(workspace)
        self.out_root = self.workspace / "out"
        self.log_path = self.workspace / "sessions.jsonl"
        self.cache_dir = self.workspace / "cache" / "web"
        self.cache_ttl = web.CACHE_TTL_SECONDS
        self.execute = execute
        self.assume_yes = assume_yes

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def plan(
        self,
        request: str,
        domain: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        return self.planner.plan(request, domain=domain, overrides=overrides)

    def answer(
        self,
        request: str,
        domain: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Report:
        if web.is_search_request(request):
            return self.search_report(request)

        plan = self.plan(request, domain=domain, overrides=overrides)

        if plan.recipe is not None:
            if plan.missing:
                report = Report(plan=plan, error="missing parameters")
                report.text = self._render_missing(plan)
                self._record(report)
                return report
            return self._recipe_report(plan)

        image_paths = vision.find_image_paths(request)
        if image_paths:
            return self.vision_report(image_paths[0], question=request)

        urls = web.find_urls(request)
        if urls:
            return self.web_report(urls[0], question=request)

        reply = social_respond(request, self.planner.recipes)
        if reply:
            report = Report(plan=plan, capability="social")
            report.text = reply
            self._record(report)
            return report

        report = Report(plan=plan, error="no matching recipe", capability="none")
        report.text = self._render_no_match(plan)
        self._record(report)
        return report

    def history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows[-limit:]

    def doctor(self) -> str:
        lines = ["amu agent: what this machine can execute"]
        for key, value in capabilities().items():
            if isinstance(value, bool):
                shown = "installed" if value else "not installed"
            else:
                shown = value or "not installed"
            lines.append("    %-14s %s" % (key, shown))
        lines += [
            "",
            "    Script generation always works; execution needs the tools above.",
            "    Web reading uses the Python standard library (urllib) - no API key.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # recipes
    # ------------------------------------------------------------------
    def _recipe_report(self, plan: Plan) -> Report:
        try:
            result = plan.recipe.build(plan.params)
        except Exception as exc:   # a recipe bug must not kill the session
            report = Report(plan=plan, error="%s: %s" % (type(exc).__name__, exc))
            report.text = "Could not build %s: %s" % (plan.recipe.id, report.error)
            self._record(report)
            return report

        files = self._write_artifacts(plan.recipe.id, result.artifacts)
        outcome, note = self._maybe_execute(plan, result)

        report = Report(
            plan=plan,
            result=result,
            files=files,
            outcome=outcome,
            execution_note=note,
        )
        report.text = self._render(report)
        self._record(report)
        return report

    # ------------------------------------------------------------------
    # web
    # ------------------------------------------------------------------
    def web_report(
        self,
        url: str,
        question: Optional[str] = None,
        transport=None,
        max_age: Optional[float] = None,
    ) -> Report:
        plan = Plan(request=question or url, domain=None, recipe=None)
        report = Report(plan=plan, capability="web")

        try:
            data = web.lookup(
                url,
                question=question,
                transport=transport,
                cache_dir=self.cache_dir,
                max_age=self.cache_ttl if max_age is None else max_age,
            )
        except web.WebError as exc:
            report.error = str(exc)
            report.text = "Web lookup failed: %s" % exc
            self._record(report)
            return report

        page = data.page
        lines = [
            "amu web | %s" % (page.title or "(no title)"),
            "url     : %s" % page.url,
            "status  : %s  %s  %d characters extracted" % (
                page.status,
                page.content_type or "unknown type",
                len(page.text),
            ),
            "cache   : %s" % (web.cache_note(page) or "live fetch"),
            "",
        ]

        if data.sentences:
            lines.append("summary :")
            for sentence in data.sentences:
                lines.append("    - %s" % sentence)
        else:
            lines.append("summary : the page had no readable sentences")

        lines += [
            "",
            "limits  : one HTTP GET, no JavaScript, no login. Anything critical "
            "should be confirmed at the source.",
        ]

        artifacts = [
            Artifact("page.txt", "text", page.text[:200_000]),
            Artifact(
                "summary.md",
                "markdown",
                "# %s\n\n%s\n\nSource: %s\n" % (
                    page.title or page.url,
                    "\n".join("- " + sentence for sentence in data.sentences)
                    or "(no readable sentences)",
                    page.url,
                ),
            ),
        ]
        report.files = self._write_artifacts("web-lookup", artifacts)
        report.text = "\n".join(lines)
        self._record(report)
        return report

    def search_report(
        self,
        request: str,
        transport=None,
        query: Optional[str] = None,
        read_top: bool = True,
    ) -> Report:
        plan = Plan(request=request, domain=None, recipe=None)
        report = Report(plan=plan, capability="web")
        term = query or web.search_query(request)

        try:
            outcome = web.search(term, transport=transport)
        except web.WebError as exc:
            report.error = str(exc)
            report.text = (
                "Web search failed: %s\n"
                "I try DuckDuckGo (two endpoints) and Wikipedia, all without an "
                "API key, so any of them can be rate limited or blocked. If you "
                "already know the page, use: look up https://..." % exc
            )
            self._record(report)
            return report

        results = outcome.results
        if not results:
            report.error = "no results"
            report.text = (
                "No results for %r. Use 'look up <url>' with a page you know."
                % term
            )
            self._record(report)
            return report

        # A list of links is not an answer, so read the best result too. The
        # first one that loads wins; the rest stay as links either way.
        read_block: List[str] = []
        read_notes: List[str] = []
        read_url: Optional[str] = None
        artifacts: List[Artifact] = []

        if read_top:
            for title, url in results[:2]:
                try:
                    data = web.lookup(
                        url,
                        question=term,
                        transport=transport,
                        cache_dir=self.cache_dir,
                        max_age=self.cache_ttl,
                    )
                except web.WebError as exc:
                    read_notes.append(
                        "skipped : %s (%s)" % (url, exc)
                    )
                    continue

                page = data.page
                read_url = url
                read_block = [
                    "read    : %s" % (page.title or title),
                    "url     : %s" % url,
                    "cache   : %s" % (web.cache_note(page) or "live fetch"),
                ]
                for sentence in data.sentences[:4]:
                    read_block.append("    - %s" % sentence)
                if not data.sentences:
                    read_block.append("    (the page had no readable sentences)")

                artifacts = [
                    Artifact("page.txt", "text", page.text[:200_000]),
                    Artifact(
                        "summary.md",
                        "markdown",
                        "# %s\n\n%s\n\nSource: %s\n" % (
                            page.title or url,
                            "\n".join(
                                "- " + sentence for sentence in data.sentences
                            )
                            or "(no readable sentences)",
                            url,
                        ),
                    ),
                ]
                break

        lines = ["amu web search | %s (%s)" % (term, outcome.provider), ""]
        lines += read_block
        lines += read_notes
        if read_block:
            lines += ["", "results :"]
        elif read_top:
            lines += [
                "read    : none of the top results could be opened - "
                "links only",
                "",
                "results :",
            ]
        else:
            lines += ["results :"]

        for index, (title, url) in enumerate(results[:8], start=1):
            marker = "  (read)" if url == read_url else ""
            lines.append("  %d. %s%s" % (index, title, marker))
            lines.append("     %s" % url)

        lines += ["", "Ask for another page: look up <url>"]

        if artifacts:
            report.files = self._write_artifacts("web-search", artifacts)

        report.text = "\n".join(lines)
        self._record(report)
        return report

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    def vision_report(self, path: str, question: Optional[str] = None) -> Report:
        plan = Plan(request=question or path, domain=None, recipe=None)
        report = Report(plan=plan, capability="vision")

        try:
            info = vision.inspect(path)
        except vision.VisionError as exc:
            report.error = str(exc)
            report.text = "I cannot read that image: %s" % exc
            self._record(report)
            return report

        report.text = vision.describe(info)
        report.files = self._write_artifacts(
            "vision",
            [Artifact("image_report.txt", "text", report.text + "\n")],
        )
        self._record(report)
        return report

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def rel_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except (ValueError, OSError):
            return str(path)

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:48] or "run"

    def _write_artifacts(self, slug: str, artifacts: List[Artifact]) -> List[Path]:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = self.out_root / ("%s-%s" % (stamp, self._slug(slug)))
        folder.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for artifact in artifacts:
            path = folder / artifact.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(artifact.content, encoding="utf-8")
            if artifact.filename.endswith(".sh"):
                path.chmod(0o755)
            written.append(path)
        return written

    def _maybe_execute(self, plan: Plan, result: ToolResult):
        if result.execute is None:
            return None, "no executable step for this recipe"
        if not self.execute:
            return None, "dry run - pass --execute to run it"
        if plan.recipe.mutates and not self.assume_yes:
            return None, (
                "%s changes your data; re-run with --execute --yes to allow it"
                % plan.recipe.id
            )
        return run_spec(result.execute), ""

    def _record(self, report: Report) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)

        params = {}
        for key, value in report.plan.params.items():
            params[key] = value if isinstance(
                value, (str, int, float, bool, list)
            ) else str(value)

        entry = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "request": report.plan.request,
            "capability": report.capability,
            "domain": report.plan.domain,
            "recipe": report.plan.recipe.id if report.plan.recipe else None,
            "params": params,
            "files": [self.rel_path(path) for path in report.files],
            "executed": bool(report.outcome and report.outcome.ran),
            "error": report.error,
            "text": report.text,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _render(self, report: Report) -> str:
        plan = report.plan
        result = report.result

        lines = [
            "amu agent | %s | %s"
            % (DOMAIN_LABEL.get(plan.domain, plan.domain), plan.recipe.title),
            "request : %s" % plan.request,
            "recipe  : %s (match score %.1f)" % (plan.recipe.id, plan.score),
        ]

        if plan.params:
            lines.append("inputs  :")
            for key in sorted(plan.params):
                lines.append("    %-14s %s" % (key, plan.params[key]))

        if result.steps:
            lines.append("steps   :")
            for index, step in enumerate(result.steps, start=1):
                lines.append("    %d. %s" % (index, step))

        if result.commands:
            lines.append("commands:")
            for command in result.commands:
                lines.append("    $ %s" % command)

        if report.files:
            lines.append("files   :")
            for path in report.files:
                lines.append("    %s" % self.rel_path(path))

        if report.outcome is not None:
            outcome = report.outcome
            if outcome.ran:
                status = "ok" if outcome.returncode == 0 else "exit %s" % outcome.returncode
                lines.append("executed: %s -> %s" % (result.execute.argv[0], status))
                if outcome.stdout:
                    lines.append("    " + outcome.stdout[:2000].replace("\n", "\n    "))
                if outcome.stderr:
                    lines.append(
                        "    stderr: " + outcome.stderr[:1000].replace("\n", "\n    ")
                    )
            else:
                lines.append("executed: no - %s" % outcome.reason)
        elif report.execution_note:
            lines.append("executed: no - %s" % report.execution_note)

        if result.notes:
            lines.append("notes   :")
            for note in result.notes:
                lines.append("    - %s" % note)

        return "\n".join(lines)

    def _render_missing(self, plan: Plan) -> str:
        lines = [
            "I can do that with %s, but I need more information." % plan.recipe.id
        ]
        for param in plan.missing:
            lines.append("    %s: %s" % (param.name, param.question))
        lines.append("Give it with --param name=value, or use the interactive mode.")
        return "\n".join(lines)

    def _render_no_match(self, plan: Plan) -> str:
        lines = ["I do not have a recipe for that request."]
        if plan.suggestions:
            lines.append("Closest recipes I know:")
            for recipe in plan.suggestions:
                lines.append("    %-26s %s" % (recipe.id, recipe.title))
        else:
            lines.append("Recipes I know:")
            for recipe in all_recipes()[:8]:
                lines.append("    %-26s %s" % (recipe.id, recipe.title))
            lines.append(
                "Run --list-recipes for the full list. I also read web pages "
                "('look up https://...'), search the web ('search the web for "
                "...') and inspect images ('describe plan.png')."
            )
        return "\n".join(lines)
