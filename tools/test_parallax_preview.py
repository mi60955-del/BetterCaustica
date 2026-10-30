"""Checks that the offline parallax preview still describes the march it mirrors.

The preview is only useful while it keeps computing what
`shaders/pipelines/world/closest_hit.rchit.slang` computes: the constants are the whole tuning
surface, so a value changed on one side and not the other silently turns the preview's verdict
into fiction. These tests pin the constants to the shader source, pin the tool's defaults to
`CausticaConfig`'s, and exercise the parts of the port that can be checked without a GPU
(sampler linearity, layer budgeting, wall gating, shadow directionality). The fuller behavioural
suite lives in the tool itself: `python tools/parallax_preview.py --check`.
"""

import re
import unittest
from pathlib import Path

import numpy as np

from tools import parallax_preview as pp

ROOT = Path(__file__).resolve().parent.parent
HIT_SHADER = ROOT / "shaders" / "pipelines" / "world" / "closest_hit.rchit.slang"
WORLD_COMMON = ROOT / "shaders" / "pipelines" / "world" / "world_common.slang"
CONFIG = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "CausticaConfig.java"
COMPOSITE = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "rt" / "RtComposite.java"

DECLARATION = re.compile(r"(?:public\s+)?static\s+const\s+(?:float|int|uint) ([A-Z0-9_]+) = ([^;]+);")


def shader_constants(path: Path) -> dict[str, float]:
    """Every `static const` declared as a plain numeric literal; expressions are skipped."""
    found = {}
    for name, value in DECLARATION.findall(path.read_text(encoding="utf-8")):
        literal = value.strip().rstrip("uf").strip()
        try:
            found[name] = float(int(literal, 16)) if literal.lower().startswith("0x") else float(literal)
        except ValueError:
            continue
    return found


def config_float(source: str, toml_key: str) -> tuple[float, float, float]:
    """The (default, min, max) of a clampedFloat setting, read out of CausticaConfig."""
    m = re.search(r'clampedFloat\("[^"]+", "' + re.escape(toml_key) + r'",\s*'
                  r"(-?[\d.]+)f?,\s*(-?[\d.]+)f?,\s*(-?[\d.]+)f?", source)
    assert m, f"clampedFloat for {toml_key} not found"
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def config_int(source: str, toml_key: str) -> tuple[int, int, int]:
    """The (default, min, max) of a clampedInt setting, read out of CausticaConfig."""
    m = re.search(r'clampedInt\("[^"]+", "' + re.escape(toml_key) + r'",\s*'
                  r"(-?\d+),\s*(-?\d+),\s*(-?\d+)", source)
    assert m, f"clampedInt for {toml_key} not found"
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


class ShaderConstantTest(unittest.TestCase):
    """The preview's SHADER_ constants are the shader's constants, not a copy that drifted."""

    def test_march_constants_match_shader(self):
        constants = shader_constants(HIT_SHADER)
        mirrored = {
            "PARALLAX_MIN_LAYERS": pp.SHADER_MIN_LAYERS,
            "PARALLAX_MAX_LAYERS": pp.SHADER_MAX_LAYERS,
            "PARALLAX_MIN_VIEW_Z": pp.SHADER_MIN_VIEW_Z,
            "PARALLAX_MAX_LOD": pp.SHADER_MAX_LOD,
            "PARALLAX_WALL_GRADIENT": pp.SHADER_WALL_GRADIENT,
            "PARALLAX_SMOOTH_MIP_BIAS": pp.SHADER_SMOOTH_MIP_BIAS,
            "PARALLAX_SMOOTH_WALL": pp.SHADER_SMOOTH_WALL,
            "PARALLAX_SHADOW_STEPS": pp.SHADER_SHADOW_STEPS,
            "PARALLAX_SHADOW_MIN_ELEVATION": pp.SHADER_SHADOW_MIN_ELEVATION,
            "PARALLAX_SHADOW_SOFTNESS": pp.SHADER_SHADOW_SOFTNESS,
            "PARALLAX_SHADOW_BIAS": pp.SHADER_SHADOW_BIAS,
            "NORMAL_AO_STRENGTH": pp.NORMAL_AO_STRENGTH,
        }
        for name, ours in mirrored.items():
            self.assertIn(name, constants, f"{name} disappeared from the hit shader")
            self.assertEqual(constants[name], ours, f"{name} drifted between shader and preview")

    def test_style_constants_match_shader(self):
        constants = shader_constants(WORLD_COMMON)
        self.assertEqual(constants["PARALLAX_STYLE_SMOOTH"], pp.STYLE_SMOOTH)
        self.assertEqual(constants["PARALLAX_STYLE_SHARP"], pp.STYLE_SHARP)
        sharp = pp.resolve_style(pp.STYLE_SHARP)
        smooth = pp.resolve_style(pp.STYLE_SMOOTH)
        self.assertEqual(sharp["mip_bias"], 0.0)
        self.assertEqual(sharp["wall_scale"], 1.0)
        self.assertEqual(smooth["mip_bias"], pp.SHADER_SMOOTH_MIP_BIAS)
        self.assertEqual(smooth["wall_scale"], pp.SHADER_SMOOTH_WALL)


class ConfigDefaultTest(unittest.TestCase):
    """The preview's defaults are the settings the game actually ships with."""

    def setUp(self):
        self.source = CONFIG.read_text(encoding="utf-8")

    def test_parallax_depth_default(self):
        default, lo, hi = config_float(self.source, "composite.parallax-depth")
        self.assertEqual(pp.DEFAULT_DEPTH_SCALE, default)
        self.assertGreaterEqual(pp.DEFAULT_DEPTH_SCALE, lo)
        self.assertLessEqual(pp.DEFAULT_DEPTH_SCALE, hi)

    def test_parallax_layers_default(self):
        default, lo, hi = config_int(self.source, "materials.parallax-layers")
        self.assertEqual(pp.DEFAULT_LAYERS_CAP, default)
        self.assertGreaterEqual(pp.DEFAULT_LAYERS_CAP, lo)
        self.assertLessEqual(pp.DEFAULT_LAYERS_CAP, hi)
        self.assertGreaterEqual(pp.SHADER_MIN_LAYERS, lo)
        self.assertLessEqual(pp.SHADER_MAX_LAYERS, hi)

    def test_style_and_shadow_defaults(self):
        self.assertIn('clampedInt("caustica.rt.materials.parallaxStyle", "materials.parallax-style",\n'
                      "                            Materials.PARALLAX_STYLE_SHARP", self.source)
        self.assertIn('bool("caustica.rt.materials.parallaxShadow", "materials.parallax-shadow", true)',
                      self.source)

    def test_depth_rides_the_push_constants(self):
        push = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("CausticaConfig.Rt.Composite.PARALLAX_DEPTH.value()", push)
        self.assertIn("CausticaConfig.Rt.Materials.PARALLAX_LAYERS.value()", push)
        self.assertIn("CausticaConfig.Rt.Materials.PARALLAX_STYLE.value()", push)


class SamplerTest(unittest.TestCase):
    """The trilinear sampler has to reproduce the fields the march scores against."""

    def test_bilinear_at_lod0_hits_texel_centres(self):
        tex = np.arange(16, dtype=np.float32).reshape(4, 4)
        mips = pp.make_mips(tex)
        centres = [(i + 0.5) / 4 for i in range(4)]
        uv = np.stack(np.meshgrid(centres, centres), axis=-1)  # uv[...,0]=x across columns
        got = pp.sample(mips, np.ascontiguousarray(uv), 0.0)
        np.testing.assert_allclose(got, tex, atol=1e-6)

    def test_linear_field_survives_mips_and_trilinear(self):
        """A field linear in x must sample exactly at any lod: box averages and bilinear of a
        linear ramp are the same ramp, which is what makes the analytic answer exact. The claim
        holds only half a mip texel inside the borders, where clamped addressing takes over."""
        x = np.linspace(0.0, 1.0, 16, endpoint=False)[None, :] + 1.0 / 32
        field = np.repeat(1.0 - x, 16, axis=0)
        mips = pp.make_mips(field)
        u = np.linspace(0.15, 0.85, 48)
        uv = np.stack(np.meshgrid(u, u), axis=-1)
        for lod in (0.0, 0.75, 2.0):
            got = pp.sample(mips, np.ascontiguousarray(uv), lod)
            np.testing.assert_allclose(got, 1.0 - uv[..., 0], atol=1e-5)

    def test_sample_wraps_addressing(self):
        """A sample pulled past the border reads the opposite edge — the tiled continuation the
        neighbouring face shows — which is the wrapSpriteUv semantics the hit shader marches with."""
        tex = np.tile(np.linspace(0.0, 1.0, 16)[None, :], (16, 1))
        mips = pp.make_mips(tex)
        uv = np.array([[[-0.02, 0.5], [1.02, 0.5]]], np.float32)
        got = pp.sample(mips, uv, 0.0)
        self.assertGreater(float(got[0, 0]), 0.9)   # x=-0.02 wraps to ~0.98: the far end of the ramp
        self.assertLess(float(got[0, 1]), 0.1)      # x=1.02 wraps to ~0.02: the near end


class MarchBehaviourTest(unittest.TestCase):
    """Invariants of the ported walk that do not need a full check-suite run."""

    def test_layer_budget_is_the_quality_cap(self):
        """The DDA's column budget is the quality setting itself; the walk never exceeds it."""
        uv0, vts, valid, lod = pp.camera_rays("tilt45", 96)
        mips = pp.make_mips(np.tile(np.linspace(1.0, 0.0, 16)[None, :], (16, 1)))
        res = pp.march(mips, uv0, vts, lod, layers_cap=64)
        sel = valid & res["crossed"]
        self.assertTrue((res["steps_used"][sel] <= res["layers"][sel]).all())
        self.assertTrue((res["layers"][sel] == 64).all())

    def test_displaced_uv_stays_on_the_sprite(self):
        for material in pp.MATERIALS:
            c = pp.run_case(material, "grazing", res_px=128)
            self.assertGreaterEqual(float(c["res"]["uv"][c["valid"]].min()), 0.0)
            self.assertLessEqual(float(c["res"]["uv"][c["valid"]].max()), 1.0)

    def test_wall_is_axis_aligned_and_needs_a_step(self):
        """DDA walls face along the crossed boundary's own axis, and only where a height step
        cleared the wall ramp — never on a flat landing or an untouched pixel."""
        c = pp.run_case("stair4", "tilt45", res_px=128)
        w = c["res"]["wall_amount"] > 0.0
        w &= c["valid"]
        self.assertTrue(w.any())
        wd = c["res"]["wall_dir"][w]
        norms = np.linalg.norm(wd, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1.0e-6)          # unit
        self.assertTrue(np.all((np.abs(wd[:, 0]) < 0.5) | (np.abs(wd[:, 1]) < 0.5)))  # one axis
        self.assertTrue((c["res"]["crossed"][w]).all())
        flat = c["valid"] & ~w
        land = flat & c["res"]["crossed"]
        # flat landings (the ray settled on a column's ceiling) carry no wall
        self.assertTrue(np.all(np.linalg.norm(c["res"]["wall_dir"][land], axis=-1) == 0.0))

    def test_shadow_needs_a_lit_elevation(self):
        uv0, vts, valid, lod = pp.camera_rays("tilt30", 96)
        mips = pp.make_mips(pp.MATERIALS["groove"]()["alpha"])
        below = pp.march(mips, uv0, vts, lod, light_ts=pp.sun_tangent(3.0, 0.0))
        self.assertEqual(float(below["shadow"].min()), 1.0)
        lit = pp.march(mips, uv0, vts, lod, light_ts=pp.sun_tangent(40.0, 0.0))
        self.assertLess(float(lit["shadow"][valid].min()), 1.0)

    def test_case_metrics_shape(self):
        c = pp.run_case("stair4", "tilt30", res_px=128)
        m = pp.case_metrics(c)
        for key in ("displaced_fraction", "mean_steps", "max_steps", "mean_offset_texels",
                    "wall_fraction", "min_shadow"):
            self.assertIn(key, m)
        self.assertTrue(0.0 <= m["displaced_fraction"] <= 1.0)

    def test_tool_checks_pass(self):
        """The agent-facing verdicts: all ten must hold against this exact port."""
        for record in pp.run_checks():
            self.assertTrue(record["pass"], f"{record['name']}: {record['detail']}")


if __name__ == "__main__":
    unittest.main()
