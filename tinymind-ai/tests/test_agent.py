"""Tests for the cadagent package.

Standard library only: run with

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadagent import lisp, vision, web
from cadagent.core import Agent
from cadagent.registry import all_recipes
from cadagent.runner import ALLOWED_TOOLS

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    PIL_AVAILABLE = False

DUMMY = {
    "text": "sample",
    "number": 1.0,
    "numbers": [0.0, 3.0, 6.0],
    "epsg": 25832,
    "path": "input.shp",
    "csv": "points.csv",
    "layers": ["ROADS", "PARCELS"],
    "flag": False,
}


class RegistryTests(unittest.TestCase):
    def test_ids_unique_and_domains_known(self):
        recipes = all_recipes()
        ids = [recipe.id for recipe in recipes]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(recipes), 12)

        for recipe in recipes:
            self.assertIn(recipe.domain, {"revit", "autocad", "map3d", "qgis"})
            self.assertTrue(recipe.triggers, recipe.id)
            self.assertTrue(recipe.summary, recipe.id)

    def test_every_recipe_builds(self):
        """Every recipe must produce a non-empty script from dummy inputs."""
        for recipe in all_recipes():
            params = {param.name: DUMMY[param.kind] for param in recipe.params}
            result = recipe.build(params)

            self.assertTrue(result.artifacts, "%s produced no files" % recipe.id)
            for artifact in result.artifacts:
                self.assertTrue(artifact.content.strip(), recipe.id)
                self.assertFalse(artifact.filename.startswith("/"), recipe.id)

            if result.execute is not None:
                self.assertIsInstance(result.execute.argv, list)
                self.assertIn(result.execute.argv[0], ALLOWED_TOOLS)
                for part in result.execute.argv:
                    self.assertNotIn(";", part)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent = Agent(workspace=os.path.join(self.tmp.name, "ws"))

    def plan(self, request, **kwargs):
        return self.agent.plan(request, **kwargs)

    def test_revit_levels(self):
        plan = self.plan("create levels at 0, 3.5, 7")
        self.assertEqual(plan.recipe.id, "revit.levels")
        self.assertEqual(plan.params["elevations"], [0.0, 3.5, 7.0])
        self.assertEqual(plan.params["prefix"], "Level")
        self.assertEqual(plan.missing, [])

    def test_autocad_layers(self):
        plan = self.plan("create layers roads, parcels, contours in red")
        self.assertEqual(plan.recipe.id, "autocad.layer_setup")
        self.assertEqual(plan.params["layers"], ["roads", "parcels", "contours"])
        self.assertEqual(plan.params["color"], 1)

    def test_qgis_reproject(self):
        plan = self.plan("reproject parcels.shp to EPSG:25832")
        self.assertEqual(plan.recipe.id, "qgis.reproject")
        self.assertEqual(plan.params["source"], "parcels.shp")
        self.assertEqual(plan.params["epsg"], 25832)

    def test_map3d_wording_prefers_map3d(self):
        plan = self.plan("import shapefile into autocad map 3d")
        self.assertEqual(plan.recipe.domain, "map3d")

    def test_qgis_wording_prefers_qgis(self):
        plan = self.plan("buffer the roads layer and clip it to the site boundary")
        self.assertEqual(plan.recipe.id, "qgis.buffer_clip")

    def test_domain_hint(self):
        plan = self.plan("check the layer parcels.shp", domain="qgis")
        self.assertEqual(plan.recipe.domain, "qgis")

    def test_unknown_request(self):
        plan = self.plan("write me a poem about concrete")
        self.assertIsNone(plan.recipe)

    def test_missing_parameter_reported(self):
        plan = self.plan("create layers")
        self.assertEqual(plan.recipe.id, "autocad.layer_setup")
        self.assertEqual([param.name for param in plan.missing], ["layers"])

    def test_override_fills_missing_parameter(self):
        plan = self.plan("create layers", overrides={"layers": "A,B"})
        self.assertEqual(plan.params["layers"], ["A", "B"])
        self.assertEqual(plan.missing, [])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = os.path.join(self.tmp.name, "ws")
        self.agent = Agent(workspace=self.workspace)

    def test_files_written_and_logged(self):
        report = self.agent.answer("create layers roads, parcels, contours in red")

        self.assertEqual(len(report.files), 2)
        for path in report.files:
            self.assertTrue(Path(path).exists(), path)

        scr = [p for p in report.files if p.name == "layers.scr"][0]
        text = scr.read_text(encoding="utf-8")
        self.assertIn("-LAYER", text)
        self.assertIn("roads", text)

        log = Path(self.workspace) / "sessions.jsonl"
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["recipe"], "autocad.layer_setup")
        self.assertFalse(rows[-1]["executed"])

    def test_rendered_report_has_steps_and_files(self):
        report = self.agent.answer("create levels at 0, 3")
        self.assertIn("steps", report.text)
        self.assertIn("create_levels.py", report.text)
        self.assertIn("executed: no", report.text)

    def test_missing_parameters_answer(self):
        report = self.agent.answer("create layers")
        self.assertTrue(report.error)
        self.assertIn("--param", report.text)

    def test_dry_run_never_executes(self):
        report = self.agent.answer("reproject parcels.shp to EPSG:25832")
        self.assertIsNone(report.outcome)
        self.assertIn("--execute", report.execution_note)
        self.assertTrue(report.files)

    def test_mutating_execution_needs_confirmation(self):
        from cadagent.planner import Plan
        from cadagent.types import ExecuteSpec, Recipe, ToolResult

        recipe = Recipe(
            id="test.mutating",
            domain="qgis",
            title="mutating test recipe",
            summary="not part of the registry",
            triggers=["x"],
            params=[],
            mutates=True,
            build=lambda params: ToolResult(title="t"),
        )
        plan = Plan(request="x", domain="qgis", recipe=recipe, score=99.0)
        result = ToolResult(
            title="t",
            execute=ExecuteSpec(argv=["gdalinfo", "-so", "input.shp"]),
        )

        agent = Agent(workspace=self.workspace, execute=True, assume_yes=False)
        outcome, note = agent._maybe_execute(plan, result)
        self.assertIsNone(outcome)
        self.assertIn("--yes", note)

        # with --yes the gate opens; the tool is still allowlisted and probed
        agent.assume_yes = True
        outcome, note = agent._maybe_execute(plan, result)
        self.assertIsNotNone(outcome)
        self.assertFalse(outcome.ran)
        self.assertIn("not installed", outcome.reason)

    def test_script_only_recipe_never_executes(self):
        agent = Agent(workspace=self.workspace, execute=True, assume_yes=True)
        report = agent.answer("create levels at 0, 3")
        self.assertIsNone(report.outcome)
        self.assertIn("no executable step", report.execution_note)
        self.assertTrue(report.files)

    def test_history_records_sessions(self):
        self.agent.answer("create levels at 0, 3")
        self.agent.answer("model health report")
        history = self.agent.history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["recipe"], "revit.levels")


class LispBuilderTests(unittest.TestCase):
    def test_string_escaping(self):
        self.assertEqual(lisp.lisp_string('a"b\\c'), '"a\\"b\\\\c"')
        self.assertEqual(lisp.lisp_string("line1\nline2"), '"line1\\nline2"')

    def test_path_literal_uses_forward_slashes(self):
        self.assertEqual(
            lisp.path_literal("C:\\proj\\points.csv"), '"C:/proj/points.csv"'
        )

    def test_number_literal(self):
        self.assertEqual(lisp.number_literal(1.0), "1")
        self.assertEqual(lisp.number_literal(2.5), "2.5")
        self.assertEqual(lisp.number_literal("3"), "3")
        with self.assertRaises(ValueError):
            lisp.number_literal("abc")
        with self.assertRaises(ValueError):
            lisp.number_literal(float("nan"))

    def test_command_symbol(self):
        self.assertEqual(lisp.command_symbol("hatch"), "c:TMK-HATCH")
        self.assertEqual(lisp.command_symbol("TMK-HATCH"), "c:TMK-HATCH")
        self.assertEqual(lisp.command_symbol("c:tmk-hatch"), "c:TMK-HATCH")
        with self.assertRaises(ValueError):
            lisp.command_symbol("///")

    def test_defun_is_balanced_and_named(self):
        text = lisp.defun(
            "demo", locals_=["value"], body=["(setq value 1)"], command=True
        )
        self.assertEqual(lisp.check(text), [])
        self.assertIn("(defun c:TMK-DEMO ( / value)", text)
        self.assertTrue(text.rstrip().endswith("(princ)\n)"))

    def test_check_detects_problems(self):
        self.assertTrue(lisp.check("(defun x ( )"))
        self.assertTrue(lisp.check('(princ "abc)'))
        self.assertTrue(lisp.check("(princ 1))"))
        self.assertEqual(lisp.check(';; (unbalanced in a comment\n(princ "ok")'), [])
        self.assertEqual(lisp.check("(princ 1) ; trailing comment )"), [])


class LispArtifactTests(unittest.TestCase):
    def test_every_generated_lisp_is_well_formed(self):
        count = 0
        for recipe in all_recipes():
            params = {param.name: DUMMY[param.kind] for param in recipe.params}
            result = recipe.build(params)
            for artifact in result.artifacts:
                if not artifact.filename.endswith(".lsp"):
                    continue
                count += 1
                self.assertEqual(
                    lisp.check(artifact.content),
                    [],
                    "%s/%s" % (recipe.id, artifact.filename),
                )
        self.assertGreaterEqual(count, 6)


class AutocadLispRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent = Agent(workspace=os.path.join(self.tmp.name, "ws"))

    def test_hatch_reads_layer_name(self):
        plan = self.agent.plan("hatch the closed polylines on layer PARCELS")
        self.assertEqual(plan.recipe.id, "autocad.hatch_closed_polylines")
        self.assertEqual(plan.params["layer"], "PARCELS")
        self.assertEqual(plan.params["pattern"], "ANSI31")
        self.assertEqual(plan.missing, [])

    def test_find_replace_reads_key_value_pairs(self):
        plan = self.agent.plan("find and replace text find=DRAFT replace=ISSUED")
        self.assertEqual(plan.recipe.id, "autocad.text_find_replace")
        self.assertEqual(plan.params["find"], "DRAFT")
        self.assertEqual(plan.params["replace"], "ISSUED")

    def test_attributes_to_csv(self):
        plan = self.agent.plan("export block attributes to csv")
        self.assertEqual(plan.recipe.id, "autocad.attributes_to_csv")

    def test_blocks_from_csv(self):
        plan = self.agent.plan(
            "insert blocks from csv points.csv", overrides={"block": "TREE"}
        )
        self.assertEqual(plan.recipe.id, "autocad.blocks_from_csv")
        self.assertEqual(plan.params["csv_path"], "points.csv")
        self.assertEqual(plan.params["block"], "TREE")

    def test_hatch_file_written_and_well_formed(self):
        report = self.agent.answer("hatch closed polylines on layer PARCELS")
        names = [path.name for path in report.files]
        self.assertIn("hatch_closed.lsp", names)

        text = [p for p in report.files if p.name == "hatch_closed.lsp"][0].read_text(
            encoding="utf-8"
        )
        self.assertIn("(defun c:TMK-HATCH", text)
        self.assertIn('"PARCELS"', text)
        self.assertEqual(lisp.check(text), [])


class SocialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent = Agent(workspace=os.path.join(self.tmp.name, "ws"))

    def test_greeting_and_creator(self):
        report = self.agent.answer("Hey amu, iam Yuva creator of you")
        self.assertFalse(report.error)
        self.assertIn("Yuva", report.text)
        self.assertIn("amu", report.text)
        self.assertEqual(report.files, [])

    def test_identity_is_honest_about_rules(self):
        report = self.agent.answer("who are you?")
        self.assertFalse(report.error)
        self.assertIn("from scratch", report.text.lower())
        self.assertIn("rule-based", report.text.lower())

    def test_capabilities_lists_domains(self):
        report = self.agent.answer("what can you do")
        self.assertIn("Revit", report.text)
        self.assertIn("QGIS", report.text)
        self.assertRegex(report.text, r"\d+ recipes")

    def test_greeting_does_not_swallow_a_task(self):
        report = self.agent.answer("hey amu, create layers roads, parcels in red")
        self.assertEqual(report.plan.recipe.id, "autocad.layer_setup")
        self.assertTrue(report.files)

    def test_unknown_task_still_says_so(self):
        report = self.agent.answer("write me a poem about concrete")
        self.assertIn("do not have a recipe", report.text)
        self.assertTrue(report.error)


class WebTests(unittest.TestCase):
    def test_readable_text_strips_markup(self):
        markup = (
            "<html><head><title>Docs</title><style>.x{color:red}</style></head>"
            "<body><h1>Buffer</h1><p>Distance is in CRS units &amp; metres.</p>"
            "<script>var secret = 1;</script></body></html>"
        )
        text = web.readable_text(markup)
        self.assertIn("Distance is in CRS units & metres.", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("<p>", text)
        self.assertNotIn("color:red", text)

    def test_title_and_urls(self):
        self.assertEqual(web.title_of("<title>QGIS docs</title>"), "QGIS docs")
        self.assertEqual(
            web.find_urls("see https://docs.qgis.org/x.html, and http://a.io/b"),
            ["https://docs.qgis.org/x.html", "http://a.io/b"],
        )

    def test_summarize_prefers_relevant_sentence(self):
        text = (
            "This page explains many unrelated things about styling layers and "
            "printing maps at large scale. The buffer distance is measured in the "
            "units of the layer coordinate reference system. Buffer distance "
            "defaults to ten metres inside the processing toolbox."
        )
        best = web.summarize(text, "buffer distance")
        self.assertTrue(best)
        self.assertIn("buffer distance", best[0].lower())

    def test_guard_rejects_unsafe_targets(self):
        transport = lambda url, timeout: (200, {}, b"<p>ok</p>")
        for bad in (
            "file:///etc/passwd",
            "ftp://example.com/x",
            "http://127.0.0.1/x",
            "http://10.1.2.3/x",
            "http://[::1]/x",
            "http://localhost/x",
        ):
            with self.assertRaises(web.WebError, msg=bad):
                web.fetch(bad, transport=transport)

    def test_fetch_with_injected_transport(self):
        body = (
            b"<html><head><title>Docs</title></head><body>"
            b"<p>The buffer distance is measured in layer CRS units.</p>"
            b"</body></html>"
        )

        def transport(url, timeout):
            return 200, {"Content-Type": "text/html; charset=utf-8"}, body

        page = web.fetch("https://example.com/docs", transport=transport)
        self.assertEqual(page.status, 200)
        self.assertEqual(page.title, "Docs")
        self.assertIn("buffer distance", page.text)

    def test_lookup_summarizes_the_page(self):
        body = (
            b"<html><body><p>Unrelated paragraph about trees and arctic weather "
            b"conditions in winter.</p><p>The buffer distance is measured in the "
            b"units of the layer CRS.</p></body></html>"
        )

        def transport(url, timeout):
            return 200, {"Content-Type": "text/html"}, body

        result = web.lookup(
            "https://example.com/buffer", question="buffer distance", transport=transport
        )
        self.assertTrue(result.sentences)
        self.assertIn("buffer distance", " ".join(result.sentences).lower())

    def test_parse_search_results_unwraps_redirects(self):
        markup = (
            '<a rel="nofollow" class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.qgis.org%2Fbuffer.html">'
            "QGIS buffer docs</a>"
        )
        self.assertEqual(
            web.parse_search_results(markup),
            [("QGIS buffer docs", "https://docs.qgis.org/buffer.html")],
        )

    def test_search_falls_back_through_providers(self):
        lite = (
            b'<a rel="nofollow" href="https://docs.qgis.org/buffer.html" '
            b"class='result-link'>QGIS buffer docs</a>"
        )
        wiki = json.dumps(
            {
                "query": {
                    "search": [
                        {
                            "title": "Geographic information system",
                            "snippet": "A <b>GIS</b> handles spatial data.",
                        }
                    ]
                }
            }
        ).encode()

        def transport(url, timeout):
            if "lite.duckduckgo.com" in url:
                return 200, {"Content-Type": "text/html"}, lite
            if "wikipedia.org" in url:
                return 200, {"Content-Type": "application/json"}, wiki
            return 200, {"Content-Type": "text/html"}, b"<html>blocked</html>"

        outcome = web.search("qgis buffer", transport=transport)
        self.assertEqual(outcome.provider, "duckduckgo-lite")
        self.assertEqual(outcome.results[0][1], "https://docs.qgis.org/buffer.html")

        def wiki_only(url, timeout):
            if "wikipedia.org" in url:
                return 200, {"Content-Type": "application/json"}, wiki
            return 200, {"Content-Type": "text/html"}, b"<html>blocked</html>"

        outcome = web.search("gis", transport=wiki_only)
        self.assertEqual(outcome.provider, "wikipedia")
        self.assertIn("Geographic information system", outcome.results[0][0])
        self.assertTrue(
            outcome.results[0][1].startswith("https://en.wikipedia.org/wiki/")
        )

    def test_search_reports_every_provider_failure(self):
        def blocked(url, timeout):
            return 200, {"Content-Type": "text/html"}, b"<html>nope</html>"

        with self.assertRaises(web.WebError) as caught:
            web.search("qgis", transport=blocked)
        message = str(caught.exception)
        self.assertIn("duckduckgo-html", message)
        self.assertIn("wikipedia", message)

    def test_cache_expires_and_serves_a_stale_copy_offline(self):
        with tempfile.TemporaryDirectory() as tmp:

            def online(url, timeout):
                return (
                    200,
                    {"Content-Type": "text/html"},
                    b"<title>T</title><p>" + b"first body version " * 4 + b"</p>",
                )

            first = web.fetch("https://example.com/a", transport=online, cache_dir=tmp)
            self.assertFalse(first.cached)

            reused = web.fetch(
                "https://example.com/a",
                transport=lambda url, timeout: self.fail("the cache was ignored"),
                cache_dir=tmp,
            )
            self.assertTrue(reused.cached)
            self.assertEqual(reused.text, first.text)
            self.assertIn("cached", web.cache_note(reused))

            def offline(url, timeout):
                raise web.WebError("no route to host")

            stale = web.fetch(
                "https://example.com/a", transport=offline, cache_dir=tmp, max_age=0
            )
            self.assertTrue(stale.stale)
            self.assertIn("no route", stale.note)
            self.assertIn("stale", web.cache_note(stale))

            def updated(url, timeout):
                return (
                    200,
                    {"Content-Type": "text/html"},
                    b"<title>T</title><p>" + b"second body version " * 4 + b"</p>",
                )

            fresh = web.fetch(
                "https://example.com/a", transport=updated, cache_dir=tmp, max_age=0
            )
            self.assertFalse(fresh.cached)
            self.assertIn("second body", fresh.text)

    def test_wrong_charset_is_recovered(self):
        def transport(url, timeout):
            return (
                200,
                {"Content-Type": "text/html; charset=utf-8"},
                "\u00c9chelle 1:500 du plan".encode("latin-1"),
            )

        page = web.fetch("https://example.com/x", transport=transport)
        self.assertIn("\u00c9chelle", page.text)
        self.assertNotIn("\ufffd", page.text)


class VisionTests(unittest.TestCase):
    def test_find_image_paths(self):
        self.assertEqual(
            vision.find_image_paths("describe site_plan.png or parcels.shp, maybe a.JPG"),
            ["site_plan.png", "a.JPG"],
        )

    @unittest.skipUnless(Image is not None, "Pillow is not installed")
    def test_blank_image_is_reported_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blank.png")
            Image.new("RGB", (200, 120), "white").save(path)
            info = vision.inspect(path)
        self.assertEqual((info.width, info.height), (200, 120))
        self.assertEqual(info.format, "PNG")
        self.assertIn("blank", info.verdict.lower())

    @unittest.skipUnless(Image is not None, "Pillow is not installed")
    def test_linework_image_is_reported_as_linework(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan.png")
            image = Image.new("RGB", (240, 160), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 200, 120), outline="black", width=2)
            draw.line((20, 140, 220, 140), fill="black", width=2)
            image.save(path)
            info = vision.inspect(path, run_ocr=False)
        self.assertGreater(info.ink_ratio, 0.0)
        self.assertLess(info.ink_ratio, 0.25)
        self.assertIn("line", info.verdict.lower())

    @unittest.skipUnless(Image is not None, "Pillow is not installed")
    def test_content_box_and_margins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "framed.png")
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).rectangle((40, 20, 159, 79), outline="black", width=2)
            image.save(path)
            info = vision.inspect(path, run_ocr=False)
        left, top, right, bottom = info.content_box
        self.assertLessEqual(abs(left - 40), 2)
        self.assertLessEqual(abs(top - 20), 2)
        self.assertLessEqual(abs(right - 160), 2)
        self.assertLessEqual(abs(bottom - 80), 2)
        self.assertIn("margins", vision.describe(info).lower())

    @unittest.skipUnless(Image is not None, "Pillow is not installed")
    def test_edge_to_edge_content_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clipped.png")
            image = Image.new("RGB", (100, 100), "white")
            ImageDraw.Draw(image).line((0, 50, 99, 50), fill="black")
            image.save(path)
            info = vision.inspect(path, run_ocr=False)
        self.assertTrue(any("edge" in note for note in info.notes))

    def test_missing_pillow_is_reported(self):
        with mock.patch.object(vision, "PIL_AVAILABLE", False):
            with self.assertRaises(vision.VisionError) as caught:
                vision.inspect("anything.png")
        self.assertIn("pillow", str(caught.exception).lower())


class CapabilityRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent = Agent(workspace=os.path.join(self.tmp.name, "ws"))

    @unittest.skipUnless(Image is not None, "Pillow is not installed")
    def test_answer_inspects_an_image(self):
        path = os.path.join(self.tmp.name, "site.png")
        Image.new("RGB", (120, 80), "white").save(path)
        report = self.agent.answer("describe %s" % path)
        self.assertFalse(report.error)
        self.assertEqual(report.capability, "vision")
        self.assertIn("amu vision", report.text)
        self.assertIn("PNG", report.text)
        self.assertEqual([path.name for path in report.files], ["image_report.txt"])

    def test_answer_routes_a_url(self):
        page = web.Page(
            url="https://example.com/docs",
            status=200,
            content_type="text/html",
            text="The buffer distance is measured in layer CRS units.",
            title="Docs",
        )
        with mock.patch(
            "cadagent.core.web.lookup",
            return_value=web.Lookup(page=page, sentences=[page.text]),
        ):
            report = self.agent.answer("look up https://example.com/docs")
        self.assertFalse(report.error)
        self.assertEqual(report.capability, "web")
        self.assertIn("example.com/docs", report.text)
        self.assertIn("page.txt", [path.name for path in report.files])

    def test_explicit_search_intent_wins_over_recipe(self):
        page = web.Page(
            url="https://docs.qgis.org",
            status=200,
            content_type="text/html",
            text="Buffer distance is in layer units.",
            title="QGIS docs",
        )
        with mock.patch(
            "cadagent.core.web.search",
            return_value=web.SearchOutcome(
                query="qgis buffer",
                provider="wikipedia",
                results=[("QGIS docs", "https://docs.qgis.org")],
            ),
        ), mock.patch(
            "cadagent.core.web.lookup",
            return_value=web.Lookup(
                page=page, sentences=["Buffer distance is in layer units."]
            ),
        ):
            report = self.agent.answer("search the web for qgis buffer")
        self.assertEqual(report.capability, "web")
        self.assertIn("QGIS docs", report.text)
        self.assertIn("https://docs.qgis.org", report.text)
        self.assertIn("layer units", report.text)

    def test_search_query_override_is_used(self):
        outcome = web.SearchOutcome(query="EPSG:25832", provider="wikipedia")
        with mock.patch("cadagent.core.web.search", return_value=outcome) as mocked:
            self.agent.search_report("EPSG:25832", query="EPSG:25832")
        mocked.assert_called_once_with("EPSG:25832", transport=None)

    def test_search_reads_the_top_result_automatically(self):
        markup = (
            b"<html><body>"
            b'<a class="result__a" href="https://docs.qgis.org/buffer.html">'
            b"QGIS buffer algorithm</a>"
            b'<a class="result__a" href="https://example.org/second">Second</a>'
            b"</body></html>"
        )
        article = (
            b"<html><head><title>QGIS buffer algorithm</title></head><body><p>"
            b"The native buffer algorithm grows every geometry by a distance in "
            b"layer units, so project the layer before you run it."
            b"</p></body></html>"
        )

        def transport(url, timeout):
            if "buffer.html" in url:
                return 200, {"Content-Type": "text/html; charset=utf-8"}, article
            return 200, {"Content-Type": "text/html; charset=utf-8"}, markup

        report = self.agent.search_report("qgis native buffer", transport=transport)

        self.assertFalse(report.error)
        self.assertIn("read    : QGIS buffer algorithm", report.text)
        self.assertIn("native buffer algorithm", report.text)
        self.assertIn("page.txt", [path.name for path in report.files])

    def test_search_keeps_the_link_list_when_the_top_page_fails(self):
        markup = (
            b"<html><body>"
            b'<a class="result__a" href="https://example.org/blocked">'
            b"Blocked page</a></body></html>"
        )

        def transport(url, timeout):
            if "blocked" in url:
                raise web.WebError("HTTP 403")
            return 200, {"Content-Type": "text/html"}, markup

        report = self.agent.search_report("qgis buffer", transport=transport)

        self.assertFalse(report.error)
        self.assertIn("skipped : https://example.org/blocked", report.text)
        self.assertIn("https://example.org/blocked", report.text)

    def test_cad_task_still_wins(self):
        report = self.agent.answer("hatch closed polylines on layer PARCELS")
        self.assertEqual(report.capability, "recipe")
        self.assertEqual(report.plan.recipe.id, "autocad.hatch_closed_polylines")

    def test_unknown_request_is_still_honest(self):
        report = self.agent.answer("write me a poem about concrete")
        self.assertIn("do not have a recipe", report.text)


if __name__ == "__main__":
    unittest.main()
