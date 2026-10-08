"""Meaningful regression checks for record coverage and publication safety."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import figure_ops as ops
import make_figures as figures


class BuildContracts(unittest.TestCase):
    def test_failed_redraw_preserves_current_publication(self):
        directory = figures.OUT / "joint_bge_m3"
        current = json.loads((directory / "figures_manifest.json").read_text(encoding="utf-8"))
        if current.get("schema_version") != ops.SCHEMA_VERSION:
            self.skipTest("requires a v2 release")
        fingerprint, _ = ops.input_fingerprint(figures.ROOT, directory)
        code_hashes = {name: ops.sha256(figures.ROOT / name) for name in ("make_figures.py", "figure_ops.py")}
        if fingerprint != current["data_fingerprint"] or code_hashes != current["code_sha256"]:
            self.skipTest("requires a release matching current code")
        before = {name: ops.sha256(directory / name) for name in ("figure_gallery.html", "figures_manifest.json", "figures/fig01_topic_overview.png")}
        status = directory / "figure_build_status.json"
        old_status = status.read_bytes() if status.exists() else None
        try:
            with patch.object(sys, "argv", ["make_figures.py", "--only", "1"]):
                with patch.dict(figures.FUNCTIONS, {1: lambda _: (_ for _ in ()).throw(RuntimeError("injected rendering failure"))}):
                    with self.assertRaisesRegex(RuntimeError, "injected rendering failure"):
                        figures.main()
            self.assertEqual(before, {name: ops.sha256(directory / name) for name in before})
            self.assertFalse(list(directory.glob(".figure_stage_*")))
            self.assertFalse((directory / ".figure_build.lock").exists())
        finally:
            if old_status is not None:
                status.write_bytes(old_status)

    def test_selection_rejects_invalid_and_never_falls_back_to_all(self):
        for value in ("16", "0", "1,x", "1,", "-1", " "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ops.parse_selection(value)
        self.assertEqual(ops.parse_selection("1, 4,1,9"), [1, 4, 9])

    def test_addresses_remove_author_semicolons_and_handle_us_states(self):
        pairs = figures.extract_all_addresses("[Smith, A; Li, B] Univ California, Davis, CA 95616 USA; [Li, B] Zhejiang Univ, Hangzhou 310058, Peoples R China")
        self.assertEqual(pairs, [("美国", "Univ California"), ("中国", "Zhejiang Univ")])
        self.assertEqual(figures.extract_all_addresses("Org, London, England; Org2, Edinburgh, Scotland"), [("英国", "Org"), ("英国", "Org2")])
        self.assertEqual(figures.extract_country_institution("Org, Dept, Unknown City"), ("", "Org"))
        self.assertEqual(figures.extract_country_institution(""), ("", ""))
        self.assertEqual(figures.extract_all_addresses("Org, CA 12345 USA; Org, CA 12345 USA"), [("美国", "Org")])

    def test_lock_rejects_concurrent_writer_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(RuntimeError):
                with ops.BuildLock(directory):
                    with ops.BuildLock(directory):
                        pass
            self.assertFalse((directory / ".figure_build.lock").exists())
            with ops.BuildLock(directory):
                self.assertTrue((directory / ".figure_build.lock").is_file())

    def test_publish_rolls_back_already_replaced_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            destination, stage = base / "release", base / "stage"
            destination.mkdir()
            stage.mkdir()
            for directory, prefix in ((destination, "old"), (stage, "new")):
                (directory / "a.txt").write_text(prefix + " a")
                (directory / "figures_manifest.json").write_text(prefix + " manifest")
            replace = ops.os.replace
            def fail_commit(source, target):
                if Path(source) == stage / "figures_manifest.json":
                    raise OSError("injected commit failure")
                return replace(source, target)
            with patch.object(ops.os, "replace", side_effect=fail_commit), self.assertRaises(OSError):
                ops.publish(stage, destination, ["a.txt", "figures_manifest.json"])
            self.assertEqual((destination / "a.txt").read_text(), "old a")
            self.assertEqual((destination / "figures_manifest.json").read_text(), "old manifest")

    def test_validate_detects_content_tampering(self):
        directory = figures.OUT / "joint_bge_m3"
        path = directory / "figures_manifest.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("schema_version") != ops.SCHEMA_VERSION:
            self.skipTest("requires a v2 release")
        manifest = dict(current)
        manifest["artifacts"] = [dict(item) for item in current["artifacts"]]
        manifest["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "校验失败"):
            ops.validate_release(directory, manifest)

    def test_document_plot_contains_each_record_once(self):
        context = figures.Ctx()
        captured = {}
        def inspect_plot(plot, name, height=None):
            ids = [row[0] for trace in plot.data for row in trace.customdata]
            captured["ids"] = ids
            captured["traces"] = len(plot.data)
        with patch.object(figures, "save_figure"), patch.object(figures, "write_interactive", side_effect=inspect_plot):
            figures.fig05_document_map(context)
        self.assertEqual(len(captured["ids"]), len(context.df))
        self.assertEqual(set(captured["ids"]), set(context.df.record_id))
        self.assertEqual(captured["traces"], len(context.ids) + 1)
        figures.plt.close("all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
