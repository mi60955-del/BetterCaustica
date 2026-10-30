"""Checks that the offline sun-shaft preview still describes the code it mirrors.

The preview exists to test one claim: the shaft term is not a scattering integral over a medium, yet it
costs what the volumetric tiers cost. Both halves of that are easy to lose touch with. The constants are
the tuning surface — a value changed in `world/volume.slang` or `CausticaConfig.java` without the preview
moving turns its verdict into fiction — so they are pinned here, both sides parsed from source. The
behaviour tests then exercise the claim itself: the control with V forced to one must return exactly
zero, the term must still light the path with the fog density at zero, the shadowed air must stay dark,
and the spend must stay per-pixel constant at eight probes while the effect itself varies by an order of
magnitude. The gating tests read the raygen sources and the build file, because the cost class of the
term depends on where it is compiled: like the fog, it lives behind `CAUSTICA_VOLUME_PATH`.

Run: python -m unittest tools.test_shaft_preview
"""

import re
import unittest
from pathlib import Path

import numpy as np

from tools import shaft_preview as shaft

ROOT = Path(__file__).resolve().parent.parent
VOLUME = ROOT / "shaders" / "pipelines" / "world" / "volume.slang"
PRIMARY = ROOT / "shaders" / "pipelines" / "world" / "primary.rgen.slang"
INDIRECT = ROOT / "shaders" / "pipelines" / "world" / "indirect.rgen.slang"
COMPOSITE = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "rt" / "RtComposite.java"
CONFIG = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "CausticaConfig.java"
BUILD = ROOT / "build.gradle"


DECLARATION = re.compile(r"static const (float|int|uint) ([A-Z0-9_]+) = ([^;]+);")
CONFIG_FLOAT = re.compile(
    r'clampedFloat\("caustica\.rt\.volume\.[A-Za-z]+",\s*"volume\.([a-z-]+)",\s*'
    r'(-?[0-9.]+)f?,')
CONFIG_INT = re.compile(
    r'clampedInt\("caustica\.rt\.volume\.[A-Za-z]+",\s*"volume\.([a-z-]+)",\s*(\d+)')
CONFIG_BOOL = re.compile(
    r'bool\("caustica\.rt\.volume\.[A-Za-z]+",\s*"volume\.([a-z-]+)",\s*(true|false)\)')


def shader_constants() -> dict[str, float]:
    """Every `static const` numeric the world volume shader declares."""
    found: dict[str, float] = {}
    text = VOLUME.read_text(encoding="utf-8")
    for kind, name, value in DECLARATION.findall(text):
        literal = value.strip()
        if kind == "float" and literal.endswith("f"):
            literal = literal[:-1]
        if kind in ("int", "uint") and literal.lower().startswith("0x"):
            found[name] = float(int(literal, 16))
        else:
            found[name] = float(literal)
    return found


def config_defaults() -> tuple[dict[str, float], dict[str, bool]]:
    """The `Rt.Volume` slider defaults, keyed by their user-facing name (`shaft-density`, ...)."""
    text = CONFIG.read_text(encoding="utf-8")
    floats = {name: float(value) for name, value in CONFIG_FLOAT.findall(text)}
    floats.update({name: float(value) for name, value in CONFIG_INT.findall(text)})
    bools = {name: value == "true" for name, value in CONFIG_BOOL.findall(text)}
    return floats, bools


def render_small(**overrides):
    """One 48x27 frame of the room scene; small enough that the whole suite stays a second."""
    scene = shaft.build_air_scene()
    sun = np.array([0.25, 0.55, 0.80], dtype=shaft.F32)
    sun = (sun / np.linalg.norm(sun)).astype(shaft.F32)
    camera = ((0.0, 1.7, -10.0), (0.0, 1.5, 6.0), 55.0)
    options = dict(fog_on=False, shafts_on=True)
    options.update(overrides)
    return scene, shaft.render(scene, 48, 27, options.pop("med"), sun, camera, **options)


class ShaderConstantTest(unittest.TestCase):
    """The preview's SHADER_ constants are the shader's constants, not a copy that drifted."""

    def test_preview_constants_match_shader_source(self):
        constants = shader_constants()
        mirrored = {
            "SHAFT_STRATA": shaft.SHADER_SHAFT_STRATA,
            "SHAFT_PHASE_ENERGY": shaft.SHADER_SHAFT_PHASE_ENERGY,
            "WATER_SHAFT_STRUCTURE_FLOOR": shaft.SHADER_WATER_SHAFT_STRUCTURE_FLOOR,
            "FOG_SUN_BANDS": shaft.SHADER_FOG_SUN_BANDS,
            "FOG_MARCH_PROBE_STRATA": shaft.SHADER_FOG_MARCH_PROBE_STRATA,
            "CLOUD_EXTINCTION": shaft.SHADER_CLOUD_EXTINCTION,
        }
        for name, value in mirrored.items():
            with self.subTest(constant=name):
                self.assertIn(name, constants)
                self.assertAlmostEqual(float(value), constants[name], places=6)

    def test_phase_energy_and_frame_walk_are_the_source_literals(self):
        volume = VOLUME.read_text(encoding="utf-8")
        self.assertIn("hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7", volume)
        self.assertIn("hg(dot(rd, celestialLight.dir), g) * SHAFT_PHASE_ENERGY", volume)
        self.assertAlmostEqual(float(shaft.SHADER_PHASE_ENERGY), 0.7, places=6)
        # The golden-ratio frame walk appears in the march, spray, fog and shaft jitters; the preview
        # mirrors it as one constant.
        self.assertGreaterEqual(volume.count("0.6180339887"), 4)
        self.assertAlmostEqual(float(shaft.SHADER_JITTER_STEP), 0.6180339887, places=9)

    def test_budget_is_the_two_numbers_the_claim_turns_on(self):
        self.assertEqual(shaft.SHADER_SHAFT_STRATA, 8)
        self.assertEqual(shaft.SHADER_FOG_SUN_BANDS, 4)
        self.assertEqual(shaft.SHADER_SHAFT_STRATA, 2 * shaft.SHADER_FOG_SUN_BANDS)


class ConfigDefaultTest(unittest.TestCase):
    """The preview's defaults are the shipped defaults; the arms are compared at those settings."""

    def test_shaft_defaults_match_the_config(self):
        floats, bools = config_defaults()
        self.assertTrue(bools.get("shafts"))
        self.assertAlmostEqual(shaft.DEFAULT_SHAFT_DENSITY, floats["shaft-density"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_SHAFT_ANISOTROPY, floats["shaft-anisotropy"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_SHAFT_DISTANCE, floats["shaft-distance"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_SHAFT_INTENSITY, floats["shaft-intensity"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_WATER_SHAFT_DENSITY,
                               floats["water-shaft-density"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_WATER_SHAFT_CEILING,
                               floats["water-shaft-ceiling"], places=6)

    def test_fog_defaults_match_the_config(self):
        floats, bools = config_defaults()
        self.assertAlmostEqual(shaft.DEFAULT_FOG_DENSITY, floats["fog-density"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_FOG_HEIGHT, floats["fog-height"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_FOG_ALBEDO, floats["fog-albedo"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_FOG_ANISOTROPY, floats["fog-anisotropy"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_FOG_AMBIENT, floats["fog-ambient"], places=6)
        self.assertAlmostEqual(shaft.DEFAULT_FOG_SHADOW_FLOOR, floats["fog-shadow-floor"], places=6)
        self.assertEqual(shaft.DEFAULT_FOG_SHADOW, bools["fog-shadow"])
        self.assertEqual(shaft.DEFAULT_FOG_MARCH_STEPS, floats["fog-march-steps"])


class DepthWeightTest(unittest.TestCase):
    """The stratum boundaries are the shader's: equal in-scatter weight, not equal distance."""

    def test_boundaries_run_from_zero_to_the_reach(self):
        sigma, reach = shaft.F32(0.01), shaft.F32(192.0)
        self.assertEqual(float(shaft.shaft_depth_at_weight(sigma, reach, 0)), 0.0)
        self.assertAlmostEqual(float(shaft.shaft_depth_at_weight(sigma, reach, 8)), 192.0, places=3)
        depths = [float(shaft.shaft_depth_at_weight(sigma, reach, k)) for k in range(9)]
        self.assertEqual(depths, sorted(depths))

    def test_each_stratum_carries_the_same_share_of_the_scatter(self):
        sigma, reach = shaft.F32(0.02), shaft.F32(160.0)
        total = 1.0 - float(np.exp(-float(sigma) * float(reach)))
        for k in range(1, 8):
            depth = float(shaft.shaft_depth_at_weight(sigma, reach, k))
            transmittance = float(np.exp(-float(sigma) * depth))
            self.assertAlmostEqual(1.0 - transmittance, k * total / 8.0, places=5)


class EffectTest(unittest.TestCase):
    """What the term is made of: shadow structure, weighted by a medium, not a scattering integral."""

    def setUp(self):
        self.scene = shaft.build_air_scene()
        sun = np.array([0.25, 0.55, 0.80], dtype=shaft.F32)
        self.sun = (sun / np.linalg.norm(sun)).astype(shaft.F32)
        self.camera = ((0.0, 1.7, -10.0), (0.0, 1.5, 6.0), 55.0)

    def test_a_uniformly_lit_path_contributes_exactly_nothing(self):
        """The control: with V forced to one, `structure = 1 - mean(V)` is zero, so the term is zero.

        A volumetric in-scatter would still light the path there; that difference is the point.
        """
        image = shaft.render(self.scene, 48, 27, shaft.Medium(fog_density=0.0), self.sun,
                             self.camera, fog_on=False, shafts_on=True, control_visible=True)
        self.assertEqual(float(np.max(image["color"])), float(np.max(image["base"])))
        self.assertLess(float(np.max(np.abs(image["shaft"]))), 1e-9)

    def test_with_the_fog_at_zero_the_path_still_carries_beams(self):
        quiet = shaft.Medium(fog_density=0.0)
        plain = shaft.render(self.scene, 48, 27, quiet, self.sun, self.camera,
                             fog_on=False, shafts_on=False)
        beams = shaft.render(self.scene, 48, 27, quiet, self.sun, self.camera,
                             fog_on=False, shafts_on=True)
        delta = beams["color"] - plain["color"]
        self.assertGreater(float(np.max(delta)), 1e-4)
        self.assertGreaterEqual(float(np.min(delta)), 0.0)

    def test_fully_shadowed_air_stays_dark(self):
        """`structure` saturates at one where nothing sees the sun, but the sun term is then zero."""
        scene = shaft.Scene([(((-64.0, 0.0, 6.0), (64.0, 400.0, 6.6)))],
                            [(0.4, 0.4, 0.4)], ground_y=0.0, ground_albedo=(0.3, 0.3, 0.3))
        image = shaft.render(scene, 24, 14, shaft.Medium(fog_density=0.0), self.sun, self.camera,
                             fog_on=False, shafts_on=True)
        hit = image["hit"]
        self.assertTrue(bool(np.any(hit)))
        self.assertLess(float(np.max(np.abs(image["shaft"][hit]))), 1e-6)
        self.assertGreater(float(np.max(image["structure"][hit])), 0.99)

    def test_underwater_structure_keeps_its_floor(self):
        """The water keeps a floor on the gate, so a fully lit water column still shows its beams."""
        scene = shaft.Scene([((-9.0, 0.0, 20.0), (9.0, 0.2, 20.6))], [(0.4, 0.4, 0.4)],
                            ground_y=0.0, ground_albedo=(0.3, 0.3, 0.3), water_y=8.0)
        camera = ((0.0, 2.0, -6.0), (0.0, 3.0, 6.0), 60.0)
        image = shaft.render(scene, 32, 18, shaft.Medium(fog_density=0.0), self.sun, camera,
                             fog_on=False, shafts_on=True)
        hit = image["hit"]
        self.assertTrue(bool(np.any(hit)))
        floor = float(shaft.SHADER_WATER_SHAFT_STRUCTURE_FLOOR)
        self.assertGreaterEqual(float(np.min(image["structure"][hit])), floor - 1e-6)
        self.assertGreater(float(np.max(image["shaft"][hit])), 1e-4)


class CostTest(unittest.TestCase):
    """The other half of the claim: the term charges the volumetric tier's probe budget."""

    def test_budget_table_is_read_from_the_shader_constants(self):
        budget = dict((name, tuple(spend)) for name, *spend in shaft.probe_budget())
        self.assertEqual(budget["analytic fog"], (shaft.SHADER_FOG_SUN_BANDS,) * 2)
        self.assertEqual(budget["sun shafts"], (shaft.SHADER_SHAFT_STRATA,) * 2)
        self.assertEqual(budget["marched fog"],
                         (min(shaft.SHADER_FOG_MARCH_PROBE_STRATA,
                              shaft.DEFAULT_FOG_MARCH_STEPS),) * 2)
        self.assertAlmostEqual(budget["sun shafts"][0] / budget["analytic fog"][0], 2.0, places=6)

    def test_measured_shaft_spend_is_eight_rays_per_pixel_whatever_the_medium(self):
        for reach, density in ((16.0, 0.002), (512.0, 0.05), (192.0, 0.010)):
            with self.subTest(reach=reach, density=density):
                med = shaft.Medium(fog_density=0.0, shaft_distance=reach, shaft_density=density)
                scene, image = render_small(med=med)
                hits = int(np.count_nonzero(image["hit"]))
                self.assertGreater(hits, 0)
                self.assertEqual(image["logs"]["shaft"].shadow_rays, 8 * hits)
                self.assertEqual(image["logs"]["shaft"].cloud_field_evals, 8 * hits)

    def test_measured_fog_count_is_four_per_pixel(self):
        med = shaft.Medium(fog_density=0.02)
        scene, image = render_small(med=med, fog_on=True, shafts_on=False)
        hits = int(np.count_nonzero(image["hit"]))
        sky = 48 * 27 - hits
        self.assertEqual(image["logs"]["fog"].shadow_rays, 4 * hits)
        # The sky pixels take the unshadowed branch: no probes, one cloud evaluation at the camera.
        self.assertEqual(image["logs"]["fog"].cloud_field_evals, 4 * hits + sky)

    def test_control_arm_traces_no_rays(self):
        """Forcing V to one must skip the trace entirely, or the control would cost what it refutes."""
        scene, image = render_small(med=shaft.Medium(fog_density=0.0), control_visible=True)
        self.assertEqual(image["logs"]["shaft"].shadow_rays, 0)

    def test_shaft_error_epsilon_skips_the_probes(self):
        """A segment below the weight epsilon casts nothing, which is the one thing that raises the count."""
        med = shaft.Medium(fog_density=0.0, shaft_density=1e-7)
        scene, image = render_small(med=med)
        self.assertEqual(image["logs"]["shaft"].shadow_rays, 0)

    def test_shafts_cost_twice_the_fog_in_the_same_frame(self):
        med = shaft.Medium(fog_density=0.02)
        scene, frame = render_small(med=med, fog_on=True, shafts_on=True)
        hits = int(np.count_nonzero(frame["hit"]))
        self.assertEqual(frame["logs"]["fog"].shadow_rays, 4 * hits)
        self.assertEqual(frame["logs"]["shaft"].shadow_rays, 8 * hits)


class GatingTest(unittest.TestCase):
    """The compiled-out path: shafts are stripped with the volumetrics, not next to them."""

    def test_the_audit_finds_no_violation_in_the_tree(self):
        checks = shaft.gating_audit()
        self.assertTrue(checks)
        failures = [check for check, ok, _ in checks if ok is False]
        self.assertEqual(failures, [])

    def test_the_shaft_and_fog_calls_share_the_volume_path_gate(self):
        for path in (PRIMARY, INDIRECT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("#define CAUSTICA_VOLUME_PATH", text)
            self.assertIn("#ifndef CAUSTICA_NO_VOLUME_PATH", text)
        primary = PRIMARY.read_text(encoding="utf-8")
        # The debug image guard-clauses on the compile switch instead of testing it at each call.
        self.assertIn("if (!CAUSTICA_VOLUME_PATH) {", primary)
        self.assertIn("if (CAUSTICA_VOLUME_PATH && fogOn) {", primary)
        self.assertIn("if (CAUSTICA_VOLUME_PATH && shaftOn) {", primary)
        self.assertIn("if (CAUSTICA_VOLUME_PATH && medium.current.water && shaftOn) {", primary)
        self.assertEqual(primary.count("shaftInScatter("), 2)
        self.assertEqual(primary.count("fogSegmentInScatter("), 2)

    def test_the_volume_free_raygens_are_built_with_the_define(self):
        gradle = BUILD.read_text(encoding="utf-8")
        self.assertEqual(gradle.count("-DCAUSTICA_NO_VOLUME_PATH"), 3)
        for twin in ("primary_lean", "indirect_lean", "indirect_lean_ser"):
            self.assertIn(f"world/{twin}.rgen.spv", gradle)

    def test_the_shaft_layer_selects_the_volume_path_raygen(self):
        """`leanVolumePath` requires the shafts off, so a shafts-on frame never runs the volume-free twin."""
        composite = COMPOSITE.read_text(encoding="utf-8")
        match = re.search(r"boolean leanVolumePath\s*=(.*?);", composite, re.S)
        self.assertIsNotNone(match)
        self.assertIn("!shaftOn", match.group(1))
        self.assertIn("boolean shaftOn = CausticaConfig.Rt.Volume.SHAFT_ENABLED.value()", composite)
        self.assertIn("WORLD_FLAG_SHAFTS", composite)
        self.assertNotIn("RAYGEN_SHAFT_VISIBILITY", composite)
        self.assertNotIn("shaftVisibility", composite)


if __name__ == "__main__":
    unittest.main()
