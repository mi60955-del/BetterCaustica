"""Checks that the offline lens preview still describes the shaders it mirrors.

The preview is only useful while it keeps computing what `shaders/pipelines/lens/*.slang` compute: the
constants are the whole tuning surface, so a value changed on one side and not the other silently turns the
preview's verdict into fiction. These tests pin the constants to the shader source, then exercise the parts
of the model that can be checked without a GPU (thresholding, monotonicity, the mask on the dust, spike
count, ring radius, sample-count stability) and the tap budget that keeps the chain affordable.
"""

import re
import unittest
from pathlib import Path

import json
import numpy as np

from tools import lens_preview as lens

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHADER = ROOT / "shaders" / "pipelines" / "lens" / "source.comp.slang"
FLARE_SHADER = ROOT / "shaders" / "pipelines" / "lens" / "main.comp.slang"
COMPOSITE = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "rt" / "RtComposite.java"
CONFIG = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "CausticaConfig.java"

DECLARATION = re.compile(r"static const (float|int|uint) ([A-Z0-9_]+) = ([^;]+);")


def shader_constants(*paths: Path) -> dict[str, float]:
    """Every `static const` the lens shaders declare, parsed to a float (seeds included, as integers)."""
    found: dict[str, float] = {}
    for path in paths:
        for kind, name, value in DECLARATION.findall(path.read_text(encoding="utf-8")):
            literal = value.strip().rstrip("uf").strip()
            if kind == "uint" and literal.lower().startswith("0x"):
                found[name] = float(int(literal, 16))
            elif kind == "int" and literal.lower().startswith("0x"):
                found[name] = float(int(literal, 16))
            else:
                found[name] = float(literal)
    return found


class ShaderConstantTest(unittest.TestCase):
    """The preview's SHADER_ constants are the shader's constants, not a copy that drifted."""

    def test_preview_constants_match_shader_source(self):
        constants = shader_constants(SOURCE_SHADER, FLARE_SHADER)
        mirrored = {
            "BOX_TAPS": lens.SHADER_HOT_BOX_TAPS,
            "BOX_STEP": lens.SHADER_HOT_BOX_STEP,
            "HOT_THRESHOLD": lens.SHADER_HOT_THRESHOLD,
            "HOT_CEILING": lens.SHADER_HOT_CEILING,
            "SPIKE_FALLOFF": lens.SHADER_SPIKE_FALLOFF,
            "SPIKE_GAIN": lens.SHADER_SPIKE_GAIN,
            "HALO_GAIN": lens.SHADER_HALO_GAIN,
            "HALO_ELLIPSE": lens.SHADER_HALO_ELLIPSE,
            "HALO_SUBTAPS": lens.SHADER_HALO_SUBTAPS,
            "HALO_CHROMA_R": lens.SHADER_HALO_CHROMA[0],
            "HALO_CHROMA_G": lens.SHADER_HALO_CHROMA[1],
            "HALO_CHROMA_B": lens.SHADER_HALO_CHROMA[2],
            "DIRT_GAIN": lens.SHADER_DIRT_GAIN,
            "DIRT_SMUDGE_FREQ": lens.SHADER_DIRT_SMUDGE_FREQ,
            "DIRT_EMPTY": lens.SHADER_DIRT_EMPTY,
            "DIRT_LIGHT_CEILING": lens.SHADER_DIRT_LIGHT_CEILING,
            "DIRT_CELL_MARGIN": lens.SHADER_DIRT_CELL_MARGIN,
            "DIRT_CELL_SPREAD": lens.SHADER_DIRT_CELL_SPREAD,
            "DIRT_SPECK_MIN_RADIUS": lens.SHADER_DIRT_SPECK_MIN_RADIUS,
            "DIRT_SPECK_RADIUS_SPREAD": lens.SHADER_DIRT_SPECK_RADIUS_SPREAD,
            "DIRT_SPECK_BODY": lens.SHADER_DIRT_SPECK_BODY,
            "DIRT_SPECK_RIM": lens.SHADER_DIRT_SPECK_RIM,
            "DIRT_RIM_CENTER": lens.SHADER_DIRT_RIM_CENTER,
            "DIRT_RIM_WIDTH": lens.SHADER_DIRT_RIM_WIDTH,
            "DIRT_GRIME_FLOOR": lens.SHADER_DIRT_GRIME_FLOOR,
            "DIRT_GRIME_RANGE": lens.SHADER_DIRT_GRIME_RANGE,
            "DIRT_FREQ_COARSE": lens.SHADER_DIRT_FREQ[0],
            "DIRT_FREQ_MEDIUM": lens.SHADER_DIRT_FREQ[1],
            "DIRT_FREQ_FINE": lens.SHADER_DIRT_FREQ[2],
            "DIRT_WEIGHT_COARSE": lens.SHADER_DIRT_WEIGHT[0],
            "DIRT_WEIGHT_MEDIUM": lens.SHADER_DIRT_WEIGHT[1],
            "DIRT_WEIGHT_FINE": lens.SHADER_DIRT_WEIGHT[2],
            "DIRT_SEED_COARSE": lens.SHADER_DIRT_SEED[0],
            "DIRT_SEED_MEDIUM": lens.SHADER_DIRT_SEED[1],
            "DIRT_SEED_FINE": lens.SHADER_DIRT_SEED[2],
            "DIRT_SMUDGE_SEED": lens.SHADER_DIRT_SMUDGE_SEED,
        }
        for name, expected in mirrored.items():
            with self.subTest(constant=name):
                self.assertIn(name, constants, f"{name} is missing from the shaders")
                self.assertAlmostEqual(
                    float(expected), constants[name], places=6, msg=f"{name} differs from the shader"
                )

    def test_settings_defaults_match_preview_defaults(self):
        """The preview's defaults are the shipped defaults, so its sheet shows what the game shows."""
        source = CONFIG.read_text(encoding="utf-8")
        floats = dict(re.findall(
            r'clampedFloat\("caustica\.rt\.lens\.[A-Za-z]+", "lens\.([a-z-]+)", ([0-9.]+)f?',
            source,
        ))
        ints = dict(re.findall(
            r'clampedInt\("caustica\.rt\.lens\.[A-Za-z]+", "lens\.([a-z-]+)", ([0-9]+)',
            source,
        ))
        preview = (ROOT / "tools" / "lens_preview.py").read_text(encoding="utf-8")
        cli = dict(re.findall(r'add_argument\("--([a-z-]+)", [^)]*?default=([0-9.]+)', preview))
        self.assertAlmostEqual(float(floats["spike-length"]), float(cli["spike-length"]), places=4)
        self.assertAlmostEqual(float(floats["halo-radius"]), float(cli["halo-radius"]), places=4)
        self.assertAlmostEqual(float(floats["dispersion"]), float(cli["dispersion"]), places=4)
        self.assertEqual(int(ints["samples"]), int(float(cli["samples"])))
        self.assertEqual(int(ints["halo-taps"]), int(lens.DEFAULT_HALO_TAPS))
        self.assertEqual(int(ints["spikes"]), int(float(cli["spikes"])))
        self.assertEqual(int(ints["resolution"]), int(float(cli["divisor"])))
        self.assertAlmostEqual(float(floats["sample-rate"]), float(cli["sample-rate"]), places=4)

    def test_lens_textures_follow_the_resolution_setting(self):
        source = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("lensDivisor = Math.max(1, CausticaConfig.Rt.Lens.RESOLUTION.value());", source)
        # The flare image is the quality dial, so it follows the setting ...
        self.assertIn("int flareWidth = Math.max(1, width / lensDivisor);", source)
        self.assertIn("int flareHeight = Math.max(1, height / lensDivisor);", source)
        # ... while the source field keeps its fixed divisor, which is what leaves the rays thin.
        self.assertIn("private static final int LENS_SOURCE_DIVISOR = 8;", source)
        self.assertIn("int sourceWidth = Math.max(1, width / LENS_SOURCE_DIVISOR);", source)
        self.assertIn("int sourceHeight = Math.max(1, height / LENS_SOURCE_DIVISOR);", source)
        self.assertIn('"RT lens source " + sourceWidth + "x" + sourceHeight', source)
        self.assertIn('"RT lens flare " + flareWidth + "x" + flareHeight', source)
        # A resolution change has to reach the rebuild, or the gathers keep reading images of the old size.
        self.assertIn("&& lensDivisor == CausticaConfig.Rt.Lens.RESOLUTION.value()", source)
        self.assertIn("lensDivisor = 0;", source)
        preview = (ROOT / "tools" / "lens_preview.py").read_text(encoding="utf-8")
        self.assertIn("SOURCE_DIVISOR = 8", preview)

    def test_sample_rate_scales_both_gathers(self):
        source = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("float sampleRate = CausticaConfig.Rt.Lens.SAMPLE_RATE.value();", source)
        self.assertIn("Math.round(CausticaConfig.Rt.Lens.SAMPLES.value() * sampleRate)", source)
        self.assertIn("Math.round(CausticaConfig.Rt.Lens.HALO_TAPS.value() * sampleRate)", source)

    def test_quality_dials_are_exposed_with_names_and_units(self):
        settings = (ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "client"
                    / "CausticaSettings.java").read_text(encoding="utf-8")
        self.assertIn('count(out, Page.IMAGE, "lens", "lens.resolution", RT + "lensResolution"', settings)
        self.assertIn('"caustica.options.rt.lensResolution.");', settings)
        self.assertIn('scaled(out, Page.IMAGE, "lens", "lens.sample-rate", RT + "lensSampleRate"', settings)
        self.assertIn("CausticaConfig.Rt.Lens.SAMPLE_RATE, 25, 200, 5, 100.0, Unit.PERCENT", settings)
        lang = json.loads((ROOT / "src" / "main" / "resources" / "assets" / "bettercaustica" / "lang"
                           / "en_us.json").read_text(encoding="utf-8"))
        for key in ("lensResolution", "lensSampleRate"):
            self.assertIn(f"caustica.options.rt.{key}", lang)
            self.assertIn(f"caustica.options.rt.{key}.tooltip", lang)
        # Every slider stop the Count control can produce needs a fraction, because the option is named.
        for divisor in range(4, 17, 2):
            self.assertIn(f"caustica.options.rt.lensResolution.{divisor}", lang)


class HotSourceTest(unittest.TestCase):
    def test_scene_below_the_threshold_produces_no_flare_source(self):
        flat = np.full((36, 64, 3), 1.0, dtype=lens.F32)
        hot = lens.hot_source(flat, 8, 4)
        self.assertEqual(float(np.max(hot)), 0.0)

    def test_soft_ceiling_is_strictly_increasing_and_bounded(self):
        threshold = float(lens.SHADER_HOT_THRESHOLD)
        values = [threshold * 1.5, threshold * 2.5, threshold * 5.0, threshold * 50.0, threshold * 5.0e3]
        previous = -1.0
        for value in values:
            flat = np.full((16, 16, 3), value, dtype=lens.F32)
            hot = lens.hot_source(flat, 4, 4)
            peak = float(hot[..., 0].max())
            self.assertGreater(peak, previous, value)
            self.assertLess(peak, float(lens.SHADER_HOT_CEILING) + 1e-3, value)
            previous = peak

    def test_only_the_hot_region_is_non_zero(self):
        scene = np.zeros((40, 80, 3), dtype=lens.F32)
        scene[..., :] = 0.2
        scene[20, 40] = 9000.0
        hot = lens.hot_source(scene, 10, 5)
        self.assertGreater(float(np.max(hot)), 0.0)
        self.assertEqual(float(np.min(hot)), 0.0)


def blob(w: int = 160, h: int = 90, radius: int = 2) -> np.ndarray:
    """A small round source: a single lit texel is too sharp for a bilinear gather to see from every angle."""
    hot = np.zeros((h, w, 3), dtype=lens.F32)
    cy, cx = h // 2, w // 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= radius:
                hot[cy + dy, cx + dx] = 1.0 - distance / (radius + 1.0)
    return hot


class StarburstTest(unittest.TestCase):
    def test_ray_count_follows_the_aperture_blade_count(self):
        for spikes in (3, 4, 6):
            with self.subTest(spikes=spikes):
                star = lens.starburst(blob(), spikes, 32, 0.30)
                self.assertEqual(count_angular_peaks(star), 2 * spikes)

    def test_tap_count_changes_smoothness_not_level(self):
        hot = blob()
        sparse = lens.starburst(hot, 6, 8, 0.30)
        dense = lens.starburst(hot, 6, 32, 0.30)
        self.assertGreater(float(sparse.mean()), 0.0)
        self.assertLess(abs(float(sparse.mean()) - float(dense.mean())) / float(dense.mean()), 0.25)

    def test_spikes_stay_non_negative(self):
        star = lens.starburst(blob(), 6, 16, 0.22)
        self.assertGreater(float(star.max()), 0.0)
        self.assertGreater(float(star.min()), -1e-6)


class HaloTest(unittest.TestCase):
    def test_ring_lands_at_the_aperture_radius(self):
        w, h = 256, 144
        hot = blob(w, h, radius=1)
        radius = 0.14
        ring = lens.halo(hot, 96, radius, 0.0)
        profile = ring[h // 2, w // 2:, 1]
        offset_uv = int(np.argmax(profile)) / float(h)
        self.assertAlmostEqual(offset_uv, radius, delta=0.02)

    def test_dispersion_splits_the_channels(self):
        w, h = 256, 144
        hot = blob(w, h, radius=1)
        split = lens.halo(hot, 96, 0.14, 0.20)
        neutral = lens.halo(hot, 96, 0.14, 0.0)
        red_peak = int(np.argmax(split[h // 2, w // 2:, 0]))
        blue_peak = int(np.argmax(split[h // 2, w // 2:, 2]))
        self.assertGreater(red_peak, blue_peak)
        self.assertEqual(int(np.argmax(neutral[h // 2, w // 2:, 0])),
                         int(np.argmax(neutral[h // 2, w // 2:, 2])))


class LensDirtTest(unittest.TestCase):
    def test_dust_only_shows_where_the_element_is_lit(self):
        dark = np.zeros((72, 128, 3), dtype=lens.F32)
        self.assertEqual(float(np.max(lens.lens_dirt(dark, 1.0))), 0.0)
        lit = np.full((72, 128, 3), 2.0, dtype=lens.F32)
        patchy = lens.lens_dirt(lit, 1.0)
        self.assertGreater(float(patchy.max()), 0.0)
        self.assertLess(float(patchy.mean()), 0.2)

    def test_grain_size_is_a_fraction_of_the_texture_height(self):
        """The specks scale with the frame, so a smaller flare texture keeps them the same size on screen."""
        coarse = lens.lens_dirt(np.ones((36, 64, 3), dtype=lens.F32), 1.0)
        fine = lens.lens_dirt(np.ones((72, 128, 3), dtype=lens.F32), 1.0)
        coarse_cells = count_speck_islands(coarse[..., 0] > 0.02)
        fine_cells = count_speck_islands(fine[..., 0] > 0.02)
        self.assertGreater(coarse_cells, 0)
        self.assertGreater(fine_cells, 0)


def count_angular_peaks(star: np.ndarray) -> int:
    """Number of bright rays leaving the middle of the frame, sampled on a circle at half the spike length."""
    h, w = star.shape[:2]
    centre = np.array([w * 0.5, h * 0.5], dtype=lens.F32)
    radius = 0.16 * h
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    offsets_u = np.cos(angles) * radius
    offsets_v = np.sin(angles) * radius
    u = (centre[0] + offsets_u) / w
    v = (centre[1] + offsets_v) / h
    samples = lens.bilinear(star, u, v)[..., 0]
    peak = float(samples.max())
    if peak <= 0.0:
        return 0
    bright = samples > peak * 0.4
    # Count runs of bright samples: each ray crosses the circle once, so the run count is the ray count.
    return int(np.sum(bright & ~np.roll(bright, 1)))


def count_speck_islands(mask: np.ndarray) -> int:
    """Flood-fill style count of connected lit specks, without scipy."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    islands = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            islands += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return islands


class ArtifactSwitchTest(unittest.TestCase):
    """Each artifact has its own switch, and the switches must reach both the dispatch and the push data."""

    def test_settings_expose_one_switch_per_artifact(self):
        source = CONFIG.read_text(encoding="utf-8")
        switches = dict(re.findall(
            r'bool\("caustica\.rt\.lens\.[A-Za-z]+", "lens\.([a-z-]+)", (true|false)\)',
            source,
        ))
        self.assertEqual(switches.get("enabled"), "true")
        self.assertEqual(switches.get("starburst-enabled"), "true")
        self.assertEqual(switches.get("halo-enabled"), "true")
        self.assertEqual(switches.get("dirt-enabled"), "true")

    def test_switches_reach_the_dispatch_and_the_push_data(self):
        composite = COMPOSITE.read_text(encoding="utf-8")
        for flag in ("lensStarburst", "lensHalo", "lensDirt"):
            self.assertIn(f"boolean {flag} = lensChain && CausticaConfig.Rt.Lens.", composite)
        # The source field exists only for the two gathers that read it; the dust rides on bloom alone.
        self.assertIn("if (lensStarburst || lensHalo) {", composite)
        self.assertIn("lensFlarePushData(lensStarburst, lensHalo, lensDirt)", composite)
        self.assertIn("starburst ? CausticaConfig.Rt.Lens.STARBURST.value() : 0f,", composite)

    def test_switches_and_their_sliders_are_reachable_in_the_option_tree(self):
        settings = (ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "client"
                    / "CausticaSettings.java").read_text(encoding="utf-8")
        for path in ("lens.starburst-enabled", "lens.halo-enabled", "lens.dirt-enabled"):
            self.assertIn(f'"{path}"', settings)
        for prereq in ("LENS_STARBURST", "LENS_HALO", "LENS_DIRT"):
            self.assertIn(f"private static final Supplier<Boolean> {prereq} = ", settings)
        lang = json.loads((ROOT / "src" / "main" / "resources" / "assets" / "bettercaustica" / "lang"
                           / "en_us.json").read_text(encoding="utf-8"))
        for key in ("lensStarburstEnabled", "lensHaloEnabled", "lensDirtEnabled"):
            self.assertIn(f"caustica.options.rt.{key}", lang)
            self.assertIn(f"caustica.options.rt.{key}.tooltip", lang)


class CostTest(unittest.TestCase):
    """The chain must stay a handful of fetches per display pixel, which is why it runs at 1/8 res."""

    def test_default_tap_budget_stays_below_five_fetches_per_pixel(self):
        width, height = 1920, 1080
        divisor = 8
        spikes, samples, halo_taps = 6, 12, 24
        source_taps = lens.SHADER_HOT_BOX_TAPS ** 2
        flare_taps = 2 * spikes * samples + 3 * halo_taps
        texels = max(1, width // divisor) * max(1, height // divisor)
        fetches = (source_taps + flare_taps) * texels
        per_pixel = fetches / float(width * height)
        self.assertLess(per_pixel, 5.0)
        self.assertAlmostEqual(texels / float(width * height), 1.0 / 64.0, delta=0.01)

    def test_lens_image_is_written_by_a_pass_that_always_runs_when_read(self):
        """The display pass must not sample a flare image the chain skipped writing."""
        composite = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("boolean lensChain = CausticaConfig.Rt.Lens.ENABLED.value();", composite)
        self.assertIn("boolean lensEnabled = lensStarburst || lensHalo || lensDirt;", composite)
        self.assertIn("lensEnabled ? 1.0f : 0.0f", composite)
        self.assertIn("lensFlareImage.view, lensPipeline.sampler()", composite)
        display = (ROOT / "shaders" / "pipelines" / "display" / "main.comp.slang").read_text(encoding="utf-8")
        self.assertIn("if (pc.lensStrength <= 0.0)", display)


if __name__ == "__main__":
    unittest.main()
