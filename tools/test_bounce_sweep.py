"""Checks that the offline bounce-sweep analysis actually measures what a session records.

The sweep exists to turn one gpu.csv plus a list of sampled frame ranges into a bounce cost curve.
The two ways that goes quietly wrong are both in this file's scope: a segment that includes frames
from the neighbouring arm (changing a setting takes effect between two csv rows, so the boundary is
a frame number, not a row index), and a marginal cost computed across a gap in the sweep (a missing
arm must break the chain, not silently bridge it). The gpu.csv shape here mirrors RtGpuTiming:
`frame,<name>Ms,...`, monotonic frame numbers with holes where a frame was not timed.

Run: python -m unittest tools.test_bounce_sweep
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools import bounce_sweep as sweep

HEADER = ["frame", "skyLutMs", "tracePrimaryMs", "traceIndirectMs", "dlssMs"]


def write_csv(rows):
    """rows: list of (frame, primary, indirect); skyLut/dlss padded constant."""
    lines = [",".join(HEADER)]
    for frame, primary, indirect in rows:
        lines.append(f"{frame},0.500,{primary:.4f},{indirect:.4f},5.000")
    return "\n".join(lines) + "\n"


class PercentileTest(unittest.TestCase):
    def test_exact_positions(self):
        self.assertEqual(sweep.percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(sweep.percentile([1.0, 2.0, 3.0, 4.0], 0.0), 1.0)
        self.assertEqual(sweep.percentile([1.0, 2.0, 3.0, 4.0], 1.0), 4.0)

    def test_interpolates_between_ranks(self):
        # p95 of 20 samples sits between the 18th and 19th sorted values.
        values = sorted(float(i) for i in range(1, 21))
        self.assertAlmostEqual(sweep.percentile(values, 0.95), 19.05)

    def test_single_sample(self):
        self.assertEqual(sweep.percentile([7.0], 0.95), 7.0)

    def test_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            sweep.percentile([], 0.5)


class ReadGpuCsvTest(unittest.TestCase):
    def test_parses_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(10, 1.0, 6.0), (11, 1.1, 6.1)]), encoding="utf-8")
            header, rows = sweep.read_gpu_csv(path)
        self.assertEqual(header, HEADER)
        self.assertEqual(rows[10], [0.5, 1.0, 6.0, 5.0])
        self.assertEqual(len(rows), 2)

    def test_frame_gaps_are_preserved_not_renumbered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(10, 1.0, 6.0), (13, 1.1, 6.1)]), encoding="utf-8")
            _, rows = sweep.read_gpu_csv(path)
        self.assertEqual(sorted(rows), [10, 13])

    def test_short_row_is_skipped(self):
        text = write_csv([(10, 1.0, 6.0)]) + "11,1.1\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(text, encoding="utf-8")
            _, rows = sweep.read_gpu_csv(path)
        self.assertEqual(sorted(rows), [10])

    def test_duplicate_frame_raises(self):
        text = write_csv([(10, 1.0, 6.0), (10, 2.0, 7.0)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                sweep.read_gpu_csv(path)

    def test_missing_header_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text("1,2,3,4,5\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sweep.read_gpu_csv(path)


class ArmStatsTest(unittest.TestCase):
    def setUp(self):
        self.rows = {frame: [0.5, float(frame) / 10, 6.0 + frame / 100, 5.0]
                     for frame in range(100, 200)}

    def header(self):
        return HEADER

    def test_segment_is_by_frame_number(self):
        arm = {"label": "B4", "bounces": 4, "frame_first": 150, "frame_last": 159}
        stats = sweep.arm_stats(self.header(), self.rows, arm)
        self.assertEqual(stats["frames"], 10)
        # tracePrimary median over frames 150..159 = (15.0+15.9)/2
        self.assertAlmostEqual(stats["tracePrimaryMs"]["median"], 15.45)

    def test_skip_drops_leading_frames_by_number(self):
        arm = {"label": "B4", "bounces": 4, "frame_first": 150, "frame_last": 159}
        stats = sweep.arm_stats(self.header(), self.rows, arm, skip=5)
        self.assertEqual(stats["frames"], 5)
        # frames 155..159 -> (15.5+15.9)/2
        self.assertAlmostEqual(stats["tracePrimaryMs"]["median"], 15.7)

    def test_empty_segment_reports_none(self):
        arm = {"label": "B9", "bounces": 8, "frame_first": 500, "frame_last": 600}
        self.assertIsNone(sweep.arm_stats(self.header(), self.rows, arm))

    def test_total_is_the_row_sum(self):
        arm = {"label": "B4", "bounces": 4, "frame_first": 100, "frame_last": 100}
        stats = sweep.arm_stats(self.header(), self.rows, arm)
        self.assertAlmostEqual(stats["total"]["median"], 0.5 + 10.0 + 7.0 + 5.0)

    def test_perf_snapshot_fields_pass_through(self):
        arm = {"label": "B4", "bounces": 4, "frame_first": 100, "frame_last": 100,
               "fps": 61.5, "fps_1pct_low": 50.0, "limit_reason": "NONE"}
        stats = sweep.arm_stats(self.header(), self.rows, arm)
        self.assertEqual(stats["fps"], 61.5)
        self.assertEqual(stats["fps_1pct_low"], 50.0)
        self.assertEqual(stats["limit_reason"], "NONE")
        self.assertNotIn("frame_ms_p95", stats)


class MarginalTest(unittest.TestCase):
    @staticmethod
    def stat(bounces, primary, indirect, total):
        return {"bounces": bounces,
                "tracePrimaryMs": {"median": primary},
                "traceIndirectMs": {"median": indirect},
                "total": {"median": total}}

    def test_chain_over_sorted_levels(self):
        stats = {2: self.stat(2, 1.0, 6.0, 12.0),
                 3: self.stat(3, 1.2, 8.0, 14.0),
                 4: self.stat(4, 1.5, 10.0, 17.0)}
        steps = sweep.marginal_costs(stats)
        self.assertEqual([(s["from"], s["to"]) for s in steps], [(2, 3), (3, 4)])
        self.assertAlmostEqual(steps[1]["d_traceIndirectMs"], 2.0)
        self.assertAlmostEqual(steps[1]["d_tracePrimaryMs"], 0.3)
        self.assertAlmostEqual(steps[1]["d_total"], 3.0)

    def test_gap_in_levels_is_not_bridged(self):
        # 2 -> 8 with no arms between must produce nothing: an unmeasured middle
        # level would otherwise smear several steps of cost into one number.
        stats = {2: self.stat(2, 1.0, 6.0, 12.0),
                 8: self.stat(8, 2.0, 20.0, 28.0)}
        self.assertEqual(sweep.marginal_costs(stats), [])

    def test_failed_middle_arm_breaks_the_chain(self):
        # A level that was swept but yielded no rows breaks the chain exactly like
        # a level that was never swept.
        stats = {2: self.stat(2, 1.0, 6.0, 12.0),
                 3: None,
                 4: self.stat(4, 1.5, 10.0, 17.0)}
        self.assertEqual(sweep.marginal_costs(stats), [])

    def test_none_entries_dropped(self):
        stats = {2: self.stat(2, 1.0, 6.0, 12.0),
                 3: None,
                 4: self.stat(4, 1.5, 10.0, 17.0),
                 5: self.stat(5, 1.8, 12.0, 20.0)}
        steps = sweep.marginal_costs(stats)
        self.assertEqual([(s["from"], s["to"]) for s in steps], [(4, 5)])


class DriftTest(unittest.TestCase):
    @staticmethod
    def stat(primary, indirect):
        return {"tracePrimaryMs": {"median": primary},
                "traceIndirectMs": {"median": indirect}}

    def test_absolute_difference_on_focus_columns(self):
        d = sweep.reference_drift(self.stat(1.50, 10.0), self.stat(1.44, 10.10))
        self.assertAlmostEqual(d["tracePrimaryMs"], 0.06)
        self.assertAlmostEqual(d["traceIndirectMs"], 0.10)

    def test_missing_side_is_none(self):
        self.assertIsNone(sweep.reference_drift(None, self.stat(1.0, 6.0)))
        self.assertIsNone(sweep.reference_drift(self.stat(1.0, 6.0), None))


class AnalyseTest(unittest.TestCase):
    def build_doc(self, csv_path):
        return {
            "gpu_csv": str(csv_path),
            "scene": "test",
            "arms": [
                {"label": "B2", "scene": "opaque", "bounces": 2, "role": "sweep",
                 "frame_first": 100, "frame_last": 109},
                {"label": "B3", "scene": "opaque", "bounces": 3, "role": "sweep",
                 "frame_first": 120, "frame_last": 129},
                {"label": "B4", "scene": "opaque", "bounces": 4, "role": "sweep",
                 "frame_first": 150, "frame_last": 159},
                {"label": "R4a", "scene": "opaque", "bounces": 4, "role": "ref",
                 "frame_first": 130, "frame_last": 139},
                {"label": "B2", "scene": "glass", "bounces": 2, "role": "sweep",
                 "frame_first": 200, "frame_last": 209},
            ],
        }

    def test_segments_stay_within_their_ranges(self):
        # Frame 149 sits between the ref arm and the sweep arm: neither may see it.
        rows = {frame: [0.5, 1.0, 6.0 + frame / 1000, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(f, 1.0, 6.0 + f / 1000) for f in sorted(rows)]),
                            encoding="utf-8")
            report = sweep.analyse(self.build_doc(path))
        by_scene = report["scenes"]
        self.assertEqual([a["frames"] for a in by_scene["opaque"]["arms"]], [10, 10, 10, 10])
        self.assertEqual(by_scene["glass"]["arms"][0]["frames"], 10)

    def test_marginal_and_missing_are_reported_per_scene(self):
        rows = {frame: [0.5, 1.0, 6.0, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(f, 1.0, 6.0) for f in sorted(rows)]), encoding="utf-8")
            report = sweep.analyse(self.build_doc(path))
        opaque = report["scenes"]["opaque"]
        self.assertEqual([(s["from"], s["to"]) for s in opaque["marginal"]],
                         [(2, 3), (3, 4)])
        self.assertEqual(opaque["marginal"][0]["d_traceIndirectMs"], 0.0)
        # glass has only bounces=2 and it has data: a single level has no steps,
        # and no promised level went missing.
        glass = report["scenes"]["glass"]
        self.assertEqual(glass["marginal"], [])
        self.assertEqual(glass["missing_bounces"], [])

    def test_ref_arm_carries_drift(self):
        rows = {frame: [0.5, 1.0, 6.0, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(f, 1.0, 6.0) for f in sorted(rows)]), encoding="utf-8")
            report = sweep.analyse(self.build_doc(path))
        ref = next(a for a in report["scenes"]["opaque"]["arms"] if a["label"] == "R4a")
        self.assertIn("drift_vs_sweep4", ref)
        self.assertEqual(ref["drift_vs_sweep4"]["traceIndirectMs"], 0.0)

    def test_table_renders_marginal_lines(self):
        rows = {frame: [0.5, 1.0, 6.0, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(f, 1.0, 6.0) for f in sorted(rows)]), encoding="utf-8")
            report = sweep.analyse(self.build_doc(path))
        table = sweep.format_table(report)
        self.assertIn("== opaque ==", table)
        self.assertIn("marginal 3->4", table)

    def test_missing_bounces_names_the_gap(self):
        # A sweep arm whose frame range has no rows must be named, not silently
        # dropped from the curve.
        rows = {frame: [0.5, 1.0, 6.0, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(f, 1.0, 6.0) for f in sorted(rows)]), encoding="utf-8")
            doc = self.build_doc(path)
            glass = next(a for a in doc["arms"] if a["scene"] == "glass")
            glass["frame_first"] = 900
            glass["frame_last"] = 909
            table = sweep.format_table(sweep.analyse(doc))
        self.assertIn("MISSING sweep arms for bounces [2]", table)

    def test_ref_arm_represents_its_level_in_the_chain(self):
        # The interleaved reference (default 4) has no sweep arm of its own; the
        # marginal chain must still cross 3->4->5 through the ref arms' pooled
        # median instead of breaking there.
        rows = {frame: [0.5, 1.0, float(frame) / 10, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(write_csv([(f, 1.0, f / 10) for f in sorted(rows)]), encoding="utf-8")
            doc = self.build_doc(path)
            doc["arms"] = [a for a in doc["arms"] if a["scene"] == "opaque"
                           and not (a["bounces"] == 4 and a["role"] == "sweep")]
            doc["arms"] += [
                {"label": "B5", "scene": "opaque", "bounces": 5, "role": "sweep",
                 "frame_first": 160, "frame_last": 169},
            ]
            report = sweep.analyse(doc)
        opaque = report["scenes"]["opaque"]
        self.assertEqual([(s["from"], s["to"]) for s in opaque["marginal"]],
                         [(2, 3), (3, 4), (4, 5)])
        step_4 = next(s for s in opaque["marginal"] if s["to"] == 4)
        # ref arm frames 130..139 -> traceIndirect median 13.45; sweep arm B3
        # frames 120..129 -> 12.45
        self.assertAlmostEqual(step_4["d_traceIndirectMs"], 13.45 - 12.45, places=3)

    def test_out_json_is_written(self):
        # The CLI contract: --arms doc + --gpu override, --out writes the report
        # beside whatever cwd the caller has.
        rows = {frame: [0.5, 1.0, 6.0, 5.0] for frame in range(100, 220)}
        with tempfile.TemporaryDirectory() as tmp:
            gpu = Path(tmp) / "gpu.csv"
            gpu.write_text(write_csv([(f, 1.0, 6.0) for f in sorted(rows)]), encoding="utf-8")
            arms_path = Path(tmp) / "arms.json"
            doc = self.build_doc(gpu)
            del doc["gpu_csv"]
            arms_path.write_text(json.dumps(doc), encoding="utf-8")
            out = Path(tmp) / "sub" / "curve.json"
            rc = sweep.main(["--gpu", str(gpu), "--arms", str(arms_path), "--out", str(out)])
            self.assertEqual(rc, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(report["gpu_csv"], str(gpu))
        self.assertIn("scenes", report)


if __name__ == "__main__":
    unittest.main()
