import io
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools import bake_scotopic_basis as bake

REPO_ROOT = Path(__file__).resolve().parent.parent
DISPLAY_SHADER = REPO_ROOT / "shaders" / "pipelines" / "display" / "main.comp.slang"

# The display pass stores its basis as literals, so it is checked against a fresh bake rather than
# trusted. Full moon is about EV -3 and a moonless night EV -6 on the renderer's EV100 scale.
NIGHT_EV = -3.0
SCOTOPIC_LIMIT_EV = -6.0
SCOTOPIC_LIMIT_LUMA_EV = 5.35  # EV100 of the CIE photopic limit (5 cd/m2), rounded up off the edge


def shader_constants() -> dict:
    text = DISPLAY_SHADER.read_text(encoding="utf-8")
    values: dict = {}
    for name in ("PURKINJE_NEUTRAL_RATIO", "MESOPIC_EV_INTERCEPT", "MESOPIC_EV_SLOPE"):
        match = re.search(rf"static const float {name} = (-?[\d.eE+-]+);", text)
        if match is None:
            raise AssertionError(f"{name} is not defined in {DISPLAY_SHADER.name}")
        values[name] = float(match.group(1))
    for name in ("ACESCG_LUMA", "ACESCG_SCOTOPIC_LUMA"):
        match = re.search(rf"static const float3 {name} = float3\(([^)]*)\);", text)
        if match is None:
            raise AssertionError(f"{name} is not defined in {DISPLAY_SHADER.name}")
        values[name] = np.array([float(value) for value in match.group(1).split(",")], dtype=np.float64)
    return values


class ScotopicBasisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.wavelength, cls.cmf, cls.v_scotopic = bake.load_cie_tables()
            cls.baked = bake.bake(cls.wavelength, cls.cmf, cls.v_scotopic)

    def test_vendored_cie_tables_still_match_their_published_checksums(self):
        with redirect_stdout(io.StringIO()):
            bake.verify_checksums()

    def test_display_pass_constants_match_a_fresh_bake(self):
        constants = shader_constants()
        np.testing.assert_allclose(constants["ACESCG_LUMA"], self.baked.photopic_luma, atol=1e-8)
        np.testing.assert_allclose(constants["ACESCG_SCOTOPIC_LUMA"], self.baked.scotopic_luma,
                                   atol=1e-8)
        self.assertAlmostEqual(constants["PURKINJE_NEUTRAL_RATIO"], self.baked.neutral_ratio, places=8)
        self.assertAlmostEqual(constants["MESOPIC_EV_INTERCEPT"], self.baked.ev_intercept, places=8)
        self.assertAlmostEqual(constants["MESOPIC_EV_SLOPE"], self.baked.ev_slope, places=8)

    def test_photopic_row_of_the_derived_matrix_is_the_meters_own_luma(self):
        # ACESCG_LUMA is what RtExposure meters and what AgX grades with, so the matrix the scotopic
        # row is composed from has to be the same colour space the renderer stores radiance in.
        np.testing.assert_allclose(self.baked.photopic_luma, bake.EXPECTED_PHOTOPIC_LUMA, atol=1e-8)
        self.assertAlmostEqual(float(self.baked.photopic_luma.sum()), 1.0, places=7)

    def test_ev_form_of_the_mesopic_weight_matches_the_luminance_form(self):
        # The display pass stores the CIE rule as intercept + slope * EV100; this checks that
        # restatement against log10 of luminance, which is how the standard writes it.
        for luminance in (0.005, 0.015625, 0.125, 1.0, 5.0):
            ev_scene = float(np.log2(8.0 * luminance))
            stated = float(np.clip(bake.MESOPIC_A + bake.MESOPIC_B * np.log10(luminance), 0.0, 1.0))
            self.assertAlmostEqual(bake.photopic_weight(ev_scene), stated, places=12)
        self.assertEqual(bake.photopic_weight(15.0), 1.0)
        self.assertEqual(bake.photopic_weight(SCOTOPIC_LIMIT_EV), 0.0)

    def test_daylight_pixels_are_left_exactly_alone(self):
        # At or above the CIE photopic limit the weight clamps to 1, and the gain formula is then
        # identically 1 for every colour: a sunlit scene cannot be re-tinted by this effect.
        for rgb in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.ones(3)):
            for ev in (15.0, 8.0, SCOTOPIC_LIMIT_LUMA_EV):
                self.assertEqual(self.baked.pixel_gain(rgb, ev, strength=1.0), 1.0)
                self.assertEqual(self.baked.pixel_gain(rgb, ev, strength=2.0), 1.0)

    def test_strength_zero_is_identity_at_any_adaptation_level(self):
        for rgb in (np.array([1.0, 0.0, 0.0]), np.array([0.2, 0.6, 0.1]), np.ones(3)):
            self.assertEqual(self.baked.pixel_gain(rgb, SCOTOPIC_LIMIT_EV, strength=0.0), 1.0)

    def test_neutral_keeps_its_brightness_so_the_exposure_level_does_not_drift(self):
        for ev in (15.0, 0.0, NIGHT_EV, SCOTOPIC_LIMIT_EV):
            for strength in (0.0, 1.0, 2.0):
                self.assertAlmostEqual(self.baked.pixel_gain(np.ones(3), ev, strength), 1.0, places=12)

    def test_the_shift_collapses_red_and_holds_blue_green(self):
        red = np.array([1.0, 0.0, 0.0])
        green = np.array([0.0, 1.0, 0.0])
        blue = np.array([0.0, 0.0, 1.0])
        # Deep red is where rods have almost no sensitivity, so its brightness falls as the eye
        # adapts, and it keeps falling as the adaptation level drops further.
        self.assertLess(self.baked.pixel_gain(red, NIGHT_EV), 0.35)
        self.assertLess(self.baked.pixel_gain(red, SCOTOPIC_LIMIT_EV),
                        self.baked.pixel_gain(red, NIGHT_EV))
        self.assertEqual(self.baked.pixel_gain(red, SCOTOPIC_LIMIT_EV), 0.0)
        # Green sits near the rod peak, so it is the colour the shift barely touches, and blue is
        # the one that wins.
        self.assertAlmostEqual(self.baked.pixel_gain(green, SCOTOPIC_LIMIT_EV), 1.12, delta=0.05)
        self.assertGreater(self.baked.pixel_gain(blue, SCOTOPIC_LIMIT_EV),
                           self.baked.pixel_gain(green, SCOTOPIC_LIMIT_EV))
        self.assertLess(self.baked.pixel_gain(red, SCOTOPIC_LIMIT_EV),
                        self.baked.pixel_gain(green, SCOTOPIC_LIMIT_EV))

    def test_saturated_red_would_get_a_negative_rod_response_without_the_clamp(self):
        # The fit is colourimetric, not spectral: it extrapolates deep red below zero, which is why
        # the display pass clamps it. Assert the clamp is load-bearing rather than defensive.
        self.assertLess(float(self.baked.scotopic_luma @ np.array([1.0, 0.0, 0.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
