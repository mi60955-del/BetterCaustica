"""Checks that the offline depth-of-field preview still describes the lens the shaders trace.

The preview is only evidence while it computes what `primary.rgen.slang`, `math.slang` and
`RtDepthOfField.java` compute, so its constants are parsed out of those sources and compared, and the
properties the feature rests on are exercised without a GPU: the focal plane must stay exactly where the
pinhole put it, the disc must measure the analytic circle of confusion, the far field must settle at
aperture over focus whatever the focal distance is, a zero radius must reproduce the old ray exactly, and
the aperture walk must cover its disk evenly and differently per pixel.

    python -m unittest tools.test_dof_preview
"""

import math
import re
import unittest
from pathlib import Path

import numpy as np

from tools import dof_preview as dof

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = ROOT / "shaders" / "pipelines" / "world" / "primary.rgen.slang"
MATH = ROOT / "shaders" / "pipelines" / "world" / "math.slang"
WORLD_COMMON = ROOT / "shaders" / "pipelines" / "world" / "world_common.slang"
GUIDES = ROOT / "shaders" / "pipelines" / "world" / "guides.slang"
BINDINGS = ROOT / "shaders" / "pipelines" / "world" / "bindings.slang"
COMPOSITE = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "rt" / \
    "RtComposite.java"
DEPTH_OF_FIELD = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "rt" / \
    "RtDepthOfField.java"
CONFIG = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "CausticaConfig.java"
SETTINGS = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "client" / \
    "CausticaSettings.java"
FOCUS_PASS = ROOT / "shaders" / "pipelines" / "dof_focus" / "main.comp.slang"
BINDING_TABLE = ROOT / "buildSrc" / "src" / "main" / "groovy" / "dev" / "comfyfluffy" / "caustica" / \
    "build" / "GenerateRtBindings.groovy"
FOCUS_PROBE = ROOT / "src" / "main" / "java" / "dev" / "comfyfluffy" / "caustica" / "rt" / "pipeline" / \
    "RtFocusProbe.java"


def group(pattern, path, label):
    found = re.search(pattern, path.read_text(encoding="utf-8"), re.S)
    assert found, f"{label} is no longer where the preview expects it ({path.name})"
    return found


class MirrorsItsSources(unittest.TestCase):
    def test_lens_lattice_and_wrap_match_math_slang(self):
        steps = group(r"LENS_LATTICE_STEPS = float2\(([^)]*)\)", MATH, "LENS_LATTICE_STEPS")
        shader = tuple(float(value.strip()) for value in steps.group(1).split(","))
        self.assertEqual(dof.LENS_LATTICE_STEPS, shader)
        wrap = group(r"float\(frameIndex & (\d+)u\)", MATH, "the frame-index wrap")
        self.assertEqual(int(wrap.group(1)), dof.LENS_LATTICE_INDEX_MASK)

    def test_pixel_seed_matches_the_raygen(self):
        seed = group(r"uint pixelSeed = dispatchIndex\.x \* (\d+)u \+ dispatchIndex\.y \* (\d+)u"
                     r" \+ (\d+)u", PRIMARY, "the pixel seed mix")
        self.assertEqual(tuple(int(value) for value in seed.groups()), dof.LENS_PIXEL_SEED)

    def test_raygen_reads_radius_from_x_and_focus_from_y(self):
        source = PRIMARY.read_text(encoding="utf-8")
        self.assertIn("if (worldPush.dof.x > 0.0 && worldPush.dof.y > 0.0)", source,
                      "the lens must still gate on a zero radius")
        self.assertIn("sampleLensDisk(worldPush.frameIndex, pixelSeed) * radius", source,
                      "the disk point must be scaled by the ceiling-tightened radius, which x seeds")
        self.assertIn("worldPush.dof.y / max(dot(pinholeDir, axis)", source,
                      "y must still be the focus distance the focal point is placed at")

    def test_push_block_is_appended_after_glint(self):
        # Java fills WorldPush positionally from the reflected layout, so the lens block has to stay the
        # last member: inserting it in the middle would silently shift every lane after it.
        body = re.search(r"public struct WorldPush \{(.*?)\n\};",
                         WORLD_COMMON.read_text(encoding="utf-8"), re.S).group(1)
        fields = re.findall(r"public\s+\S+\s+(\w+);", body)
        self.assertEqual(fields[-1], "dof")
        self.assertEqual(fields[-2], "glintParams")

    def test_host_constants_match_the_preview(self):
        far = group(r"FAR_BLOCKS = ([\d.]+)f", DEPTH_OF_FIELD, "the focus meter's far stop")
        self.assertAlmostEqual(float(far.group(1)), dof.FOCUS_PROBE_BLOCKS, places=6)
        near = group(r"NEAR_BLOCK = ([\d.]+)f", FOCUS_PROBE, "the focus meter's near clamp")
        self.assertAlmostEqual(float(near.group(1)), dof.FOCUS_NEAR_BLOCK, places=6)

    def test_config_defaults_match_the_preview(self):
        source = CONFIG.read_text("utf-8")
        aperture = group(r'"caustica\.rt\.dof\.aperture", "dof\.aperture", ([\d.]+)f',
                         CONFIG, "the default lens radius")
        focus = group(r'"caustica\.rt\.dof\.focusBlocks", "dof\.focus-distance", ([\d.]+)f',
                      CONFIG, "the default focus distance")
        self.assertAlmostEqual(float(aperture.group(1)), dof.DEFAULT_APERTURE_BLOCKS, places=6)
        cap = group(r'"caustica\.rt\.dof\.maxBlurPixels", "dof\.max-blur", ([\d.]+)f',
                    CONFIG, "the default blur cap")
        self.assertAlmostEqual(float(cap.group(1)), dof.DEFAULT_MAX_BLUR_PIXELS, places=6)
        self.assertIn('bool("caustica.rt.dof.blurCap", "dof.blur-cap", true)', source,
                      "the cap ships on: an uncapped lens is the option, not the default")
        self.assertIn('bool("caustica.rt.dof.perPixelCap", "dof.per-pixel-cap", true)', source,
                      "the per-pixel half of the cap ships on too")
        self.assertIn('bool("caustica.rt.dof.pinholeGuides", "dof.pinhole-guides", true)', source,
                      "the guides ship held on the pinhole ray: RR smearing the lens is the option")
        self.assertAlmostEqual(float(focus.group(1)), dof.DEFAULT_FOCUS_BLOCKS, places=6)
        rack = group(r'"caustica\.rt\.dof\.rackSeconds", "dof\.rack-time", ([\d.]+)f',
                     CONFIG, "the default focus time")
        self.assertAlmostEqual(float(rack.group(1)), dof.FOCUS_RACK_SECONDS, places=6)

    def test_the_host_clamps_the_dial_by_the_horizon_ceiling(self):
        # The preview's lens_radius_blocks is these lines and nothing else: the dialled radius, clamped by
        # a ceiling that grows with the focus because the horizon's blur is radius over focus.
        source = DEPTH_OF_FIELD.read_text(encoding="utf-8")
        self.assertIn("Math.min(aperture, cap * focusBlocks)", source)
        self.assertIn("blurCapPixels() * tangentPerDisplayPixel", source)
        self.assertIn("BLUR_CAP.value()", source, "the switch has to reach the pushed cap")
        self.assertIn("CausticaConfig.Rt.DepthOfField.PER_PIXEL_CAP.value() ? cap : 0.0f", source,
                      "the per-pixel half needs its own lane, or the switch cannot retire it alone")
        self.assertIn("CausticaConfig.Rt.DepthOfField.PINHOLE_GUIDES.value() ? 1.0f : 0.0f", source,
                      "the guide probe is a push lane, not a recompile")
        self.assertIn("CausticaConfig.Rt.DepthOfField.RACK_SECONDS.value()", source,
                      "the focus time is a dial, not a constant")
        self.assertNotIn("Math.atan", source,
                         "the ceiling is a tangent-space span, and routing it through atan is the "
                         "small-angle approximation wearing an exact formula's clothes")

    def test_settings_surface_points_at_the_same_settings(self):
        settings = SETTINGS.read_text(encoding="utf-8")
        self.assertIn('"dof.enabled", RT + "dof"', settings)
        self.assertIn('"dof.auto-focus", RT + "dofAutoFocus"', settings)
        self.assertIn('"dof.focus-distance", RT + "dofFocusDistance"', settings)
        self.assertIn('"dof.aperture", RT + "dofAperture"', settings)
        self.assertIn('"dof.blur-cap", RT + "dofBlurCap"', settings)
        self.assertIn('"dof.max-blur", RT + "dofMaxBlur"', settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.BLUR_CAP", settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.MAX_BLUR_PIXELS", settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.APERTURE", settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.FOCUS_BLOCKS", settings)
        self.assertIn('"dof.rack-time", RT + "dofFocusTime"', settings)
        self.assertIn('"dof.per-pixel-cap", RT + "dofPerPixelCap"', settings)
        self.assertIn('"dof.pinhole-guides", RT + "dofPinholeGuides"', settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.RACK_SECONDS", settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.PER_PIXEL_CAP", settings)
        self.assertIn("CausticaConfig.Rt.DepthOfField.PINHOLE_GUIDES", settings)


class FocusMeter(unittest.TestCase):
    """The pass that decides where the plane sits, and the fact that the host stopped deciding it on the
    block state. A block-shape raycast answers one voxel face, so it cannot see an entity, a painting, a
    water face or the recessed side of a carved block -- which is the whole reason the meter reads the
    depth the trace published instead."""

    def test_the_host_no_longer_raycast_the_block_state(self):
        source = DEPTH_OF_FIELD.read_text(encoding="utf-8")
        self.assertNotIn("ClipContext", source,
                         "the CPU block probe is the thing the traced-depth meter replaced")
        self.assertNotIn("Minecraft.getInstance", source,
                         "the lens host takes its focus as an argument now, so it stays testable")

    def test_the_meter_averages_dioptric_distance(self):
        # One surface half a block out and nine at fifty: the arithmetic mean answers 45.05, which puts the
        # plane in front of the near surface and behind nothing. The dioptric mean answers 4.59.
        depths = [0.5] + [50.0] * 9
        metered = dof.meter_focus(depths)
        self.assertAlmostEqual(metered, 10.0 / (1.0 / 0.5 + 9.0 / 50.0), places=9)
        self.assertLess(metered, sum(depths) / len(depths) / 5.0)

    def test_the_meter_ignores_texels_no_trace_wrote(self):
        self.assertEqual(dof.meter_focus([0.0, -1.0, float("nan"), 4.0]), 4.0,
                         "an unwritten texel must not move the plane, and a valid one must")
        self.assertEqual(dof.meter_focus([0.0, 0.0, float("nan")]), 0.0,
                         "a window with nothing in it is the host's no-measurement signal")

    def test_the_meter_settles_at_the_far_stop_for_sky(self):
        # A sky texel is the guide point the miss shader placed down the ray, not an unwritten texel, so it
        # is a real (if enormous) range: the meter clamps it to the far stop rather than dropping it.
        self.assertAlmostEqual(dof.meter_focus([1.0e6] * 64), dof.FOCUS_PROBE_BLOCKS, places=6)
        half_sky = dof.meter_focus([1.0e6] * 32 + [2.0] * 32)
        self.assertTrue(2.0 < half_sky < 6.0,
                        f"half a window of sky should pull the plane out a little, not out to the sky: "
                        f"{half_sky}")
        self.assertEqual(dof.meter_focus([1.0e12]), 0.0,
                         "a range past anything the trace can place is allocator garbage, not a surface")

    def test_the_pass_is_wired_into_the_frame(self):
        self.assertIn('prefix: "DOF_FOCUS"', BINDING_TABLE.read_text(encoding="utf-8"),
                      "a compute pass whose bindings are not in the generated table cannot be bound")
        composite = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("focusProbe.record(cmd, stack, focusDepthImages[focusDepthWrite].view,", composite,
                      "the meter reads the image this frame's primary pass wrote, not last frame's")
        self.assertIn("focusProbe.beginFrame(graphicsUseWaiter);", composite,
                      "the ring has to advance before the push constants are written")
        self.assertIn("focusProbe.markReadbackUse(graphicsUse);", composite,
                      "and a slot is only safe to read once the submission that filled it completed")
        probe = FOCUS_PROBE.read_text(encoding="utf-8")
        self.assertIn("VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT", probe,
                      "the meter's answer leaves the GPU through a buffer copy")
        self.assertIn("stack.longs(gpu.descriptorSet())", probe)


class LensGeometry(unittest.TestCase):
    HEIGHT = 24
    WIDTH = 32
    TAN_HALF_HEIGHT = math.tan(dof.DEFAULT_FOV_Y_RADIANS / 2.0)

    def setUp(self):
        self.tan_half_width = self.TAN_HALF_HEIGHT * self.WIDTH / self.HEIGHT
        self.ndc_x, self.ndc_y = dof.frame_screen(self.HEIGHT, self.WIDTH)
        self.seeds = dof.pixel_seeds(self.WIDTH, self.HEIGHT)

    def at_depth(self, origin, direction, depth):
        """Where a batch of rays crosses the plane `depth` blocks past the lens plane."""
        z = dof.PROJECTION_Z_NEAR + depth
        travel = (z - origin[..., 2]) / direction[..., 2]
        return origin + np.expand_dims(travel, -1) * direction

    def test_zero_radius_reproduces_the_pinhole_ray(self):
        near, pinhole = dof.pinhole_rays(self.ndc_x, self.ndc_y, self.tan_half_width,
                                         self.TAN_HALF_HEIGHT)
        disk = dof.sample_lens_disk(3, self.seeds)
        origin, direction = dof.lens_rays(self.ndc_x, self.ndc_y, self.tan_half_width,
                                          self.TAN_HALF_HEIGHT, disk, 0.0, 8.0)
        self.assertEqual(float(np.max(np.abs(origin - near))), 0.0)
        self.assertLess(float(np.max(np.abs(direction - pinhole))), 1.0e-12)

    def test_focal_plane_lands_where_the_pinhole_put_it(self):
        near, pinhole = dof.pinhole_rays(self.ndc_x, self.ndc_y, self.tan_half_width,
                                         self.TAN_HALF_HEIGHT)
        focus = 8.0
        expected = self.at_depth(near, pinhole, focus)
        for frame in range(48):
            disk = dof.sample_lens_disk(frame, self.seeds)
            origin, direction = dof.lens_rays(self.ndc_x, self.ndc_y, self.tan_half_width,
                                              self.TAN_HALF_HEIGHT, disk, 0.05, focus)
            landed = self.at_depth(origin, direction, focus)
            self.assertLess(float(np.max(np.abs(landed - expected))), 1.0e-9,
                            "the focal plane drifted off the point the pinhole ray crosses it at")

    def test_disc_is_the_analytic_circle_of_confusion(self):
        radius, focus = 0.05, 8.0
        disk = np.broadcast_to(np.array([1.0, 0.0]), (self.HEIGHT, self.WIDTH, 2))
        near, pinhole = dof.pinhole_rays(self.ndc_x, self.ndc_y, self.tan_half_width,
                                         self.TAN_HALF_HEIGHT)
        origin, direction = dof.lens_rays(self.ndc_x, self.ndc_y, self.tan_half_width,
                                          self.TAN_HALF_HEIGHT, disk, radius, focus)
        for depth in (1.0, 2.0, 4.0, 12.0, 64.0, 1024.0):
            pinhole_at_depth = self.at_depth(near, pinhole, depth)
            lens_at_depth = self.at_depth(origin, direction, depth)
            measured = np.abs(lens_at_depth[..., 0] - pinhole_at_depth[..., 0])
            self.assertLess(float(np.max(np.abs(measured - dof.circle_of_confusion(depth, radius,
                                                                                   focus)))),
                            1.0e-9, f"depth {depth}: the disc is not the analytic one")

    def test_far_field_settles_at_radius_over_focus(self):
        radius, focus = 0.05, 8.0
        angle = dof.circle_of_confusion(1.0e9, radius, focus) / 1.0e9
        self.assertAlmostEqual(angle, radius / focus, places=8)

    def test_blur_grows_monotonically_away_from_the_focal_plane(self):
        radius, focus = 0.05, 8.0
        # The *angle* is what the eye reads: radius * |1/depth - 1/focus|. It opens toward the camera and
        # saturates at radius / focus toward the horizon, and the world-space disc does the opposite,
        # because the same angle covers more blocks the further out the surface is.
        def angle(depth):
            return dof.circle_of_confusion(depth, radius, focus) / depth

        nearer = [angle(depth) for depth in (7.0, 4.0, 2.0, 1.0)]
        self.assertTrue(all(a < b for a, b in zip(nearer, nearer[1:])),
                        f"a surface closer than the focal plane must spread over more radians: {nearer}")
        far = [angle(depth) for depth in (9.0, 12.0, 24.0, 96.0, 1.0e5)]
        self.assertTrue(all(a < b for a, b in zip(far, far[1:])),
                        f"the far field must settle, not keep opening: {far}")
        self.assertLess(far[-1], radius / focus * 1.0001)
        self.assertEqual(dof.circle_of_confusion(focus, radius, focus), 0.0)

    def test_the_horizon_stays_inside_its_ceiling_at_any_focus(self):
        # The dial sets the lens; only the ceiling is expressed in pixels, and only the horizon can hit it.
        # Under the ceiling the horizon is exactly radius over focus, so the round trip has to say that
        # too. Both have to hold at any resolution and field of view, since the ceiling is counted in
        # pixels of the finished frame.
        for height in (720, 1080, 2160):
            for fov in (dof.DEFAULT_FOV_Y_RADIANS, math.radians(110.0)):
                for focus in (0.5, 1.0, 8.0, 64.0, 400.0):
                    radius = dof.lens_radius_blocks(dof.DEFAULT_APERTURE_BLOCKS, focus, height, fov)
                    horizon = dof.disc_pixels(1.0e12, radius, focus, height, fov)
                    self.assertLessEqual(horizon, dof.DEFAULT_MAX_BLUR_PIXELS + 1.0e-6,
                                         f"focus {focus}, height {height}, fov {fov}")
                    self.assertAlmostEqual(
                        horizon, min(dof.DEFAULT_APERTURE_BLOCKS / focus
                                     / dof.tangent_per_display_pixel(height, fov),
                                     dof.DEFAULT_MAX_BLUR_PIXELS), places=6,
                        msg=f"focus {focus}, height {height}, fov {fov}")

    def test_the_ceiling_only_binds_on_a_near_focus(self):
        # A lens dialled wider than the ceiling allows is a lens the frame cannot resolve anyway, so the
        # dial keeps control at any ordinary focus and gives up only close to the eye.
        height, fov = dof.DEFAULT_DISPLAY_HEIGHT, dof.DEFAULT_FOV_Y_RADIANS
        self.assertEqual(dof.lens_radius_blocks(0.04, 8.0, height, fov), 0.04)
        self.assertLess(dof.lens_radius_blocks(0.04, 0.6, height, fov), 0.04)

    def test_the_raygen_applies_the_ceiling_against_its_own_pixel(self):
        source = PRIMARY.read_text(encoding="utf-8")
        self.assertIn("clamp(focusDepthPrev[pix], 0.0, worldPush.dof.y)", source,
                      "the read-back depth must be clamped before it can reach the ray")
        self.assertIn("worldPush.dof.z * seen * worldPush.dof.y / behind", source,
                      "the per-pixel bound must stay radius <= ceiling * s * f / (f - s)")
        self.assertIn("if (worldPush.dof.z > 0.0 && behind > 1.0e-3)", source,
                      "a zero cap must switch the clamp off, not close the lens")
        guides = GUIDES.read_text(encoding="utf-8")
        self.assertIn("float focusDepth = (worldPush.flags & WORLD_FLAG_DOF_REFLECTION) != 0u && imageDepth > 0.0\n"
                      "                ? imageDepth : curClip.w;", guides,
                      "the guide pass must publish the pinhole point's own axial distance, and the "
                      "reflected image's only where the switch asks for it")
        self.assertIn("if (worldPush.dof.x > 0.0) {\n        float focusDepth", guides,
                      "the depth the autofocus meter reads has to be published whenever the lens is open, "
                      "not only while the per-pixel ceiling is on")

    def test_the_guides_stay_on_the_pinhole_ray(self):
        # The whole point of the probe: DLSS-RR classifies and reprojects with the guides, and a lens
        # sample moves them off this pixel.
        source = PRIMARY.read_text(encoding="utf-8")
        self.assertIn("if (lensSampled && worldPush.dof.w > 0.0) {", source)
        self.assertIn("queue, splitRecord, pix, false, true, probePrefix, probeNext);", source,
                      "the probe must run guides-only, after the radiance record is queued")
        self.assertIn("writeGuides(pix, pinholeDir, size, rayConeSpread)", source,
                      "the guides must be resolved against the pinhole direction")

    def test_the_depth_history_is_a_ping_pong_pair(self):
        bindings = BINDINGS.read_text(encoding="utf-8")
        self.assertIn("public RWTexture2D<float> focusDepthPrev;", bindings)
        self.assertIn("public RWTexture2D<float> focusDepthCur;", bindings)
        composite = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("worldPipeline.setFocusDepthImages(focusDepthImages[focusDepthWrite ^ 1].view,",
                      composite, "the pair must be bound prev-then-cur")
        self.assertIn("focusDepthWrite ^= 1;", composite,
                      "and advance one step per traced frame, like the reservoir pair")

    def test_a_cap_switched_off_leaves_the_dial_in_charge(self):
        # The shader gates on the cap lane being positive, so "off" has to mean the dialled lens at every
        # depth -- a zero cap that clamped instead would turn the camera into a pinhole.
        height, fov = dof.DEFAULT_DISPLAY_HEIGHT, dof.DEFAULT_FOV_Y_RADIANS
        for focus in (0.6, 8.0, 64.0):
            for depth in (0.05, 0.6, 4.0, 1000.0):
                self.assertEqual(
                    dof.lens_radius_blocks(dof.DEFAULT_APERTURE_BLOCKS, focus, height, fov,
                                           depth=depth, cap_pixels=0.0),
                    dof.DEFAULT_APERTURE_BLOCKS)

    def test_a_wider_cap_blurs_more_and_never_less(self):
        height, fov = dof.DEFAULT_DISPLAY_HEIGHT, dof.DEFAULT_FOV_Y_RADIANS
        widths = [dof.lens_radius_blocks(0.04, 1.0, height, fov, depth=0.6, cap_pixels=cap)
                  for cap in (2.0, 8.0, 14.0, 48.0)]
        self.assertTrue(all(a < b for a, b in zip(widths, widths[1:])),
                        f"the cap dial did not open the lens monotonically: {widths}")

    def test_the_per_pixel_switch_leaves_the_horizon_bound(self):
        # Cap Each Pixel off is the same ceiling applied to the one disc the focus distance sets on its
        # own: the horizon keeps its bound, every nearer surface gets the dialled lens back.
        height, fov = dof.DEFAULT_DISPLAY_HEIGHT, dof.DEFAULT_FOV_Y_RADIANS
        for focus in (0.6, 8.0, 64.0):
            horizon = dof.lens_radius_blocks(0.04, focus, height, fov)
            for depth in (0.05, 0.6, 4.0, 1000.0):
                self.assertEqual(
                    dof.lens_radius_blocks(0.04, focus, height, fov, depth=depth, per_pixel=False),
                    horizon)
        self.assertLess(
            dof.lens_radius_blocks(0.04, 8.0, height, fov, depth=0.6),
            dof.lens_radius_blocks(0.04, 8.0, height, fov, depth=0.6, per_pixel=False),
            "switching the per-pixel cap off has to open the near field, not just relabel it")

    def test_one_pixel_is_its_tangent_span(self):
        base = dof.tangent_per_display_pixel(1080, dof.DEFAULT_FOV_Y_RADIANS)
        self.assertAlmostEqual(base, 2.0 * math.tan(dof.DEFAULT_FOV_Y_RADIANS / 2.0) / 1080.0, places=12)
        self.assertAlmostEqual(dof.tangent_per_display_pixel(2160, dof.DEFAULT_FOV_Y_RADIANS),
                               0.5 * base, places=12)


class MirrorFocus(unittest.TestCase):
    """What the lens actually focuses on through a plane mirror, which is the geometry `dof.reflection-focus`
    hands the meter.

    A flat reflector images a point at |image - eye| = (eye to mirror) + (mirror to point), and the eye sees
    that image down the very ray it is looking along — so the reflected bundle gathers on the object exactly
    when the pushed focus is that image's axial distance, which is the interface's own scaled by how much
    further the path went. The premise of the feature is the other half of this: pushing the mirror's own
    distance leaves the reflected image out of focus by a disc wide enough to see.
    """

    MIRROR_Z = 6.0        # a wall mirror six blocks out
    OBJECT_BLOCKS = 8.0   # what it reflects, eight blocks along the mirror ray
    RADIUS = 0.10         # the aperture's widest setting, so the disc is at its largest
    EYE_HEIGHT = 1.6      # the player's eye, for the water-incidence check below
    TAN_HALF_HEIGHT = math.tan(dof.DEFAULT_FOV_Y_RADIANS / 2.0)

    def setUp(self):
        self.tan_half_width = self.TAN_HALF_HEIGHT * 16.0 / 9.0
        self.seeds = dof.pixel_seeds(16, 9)

    def mirror_hit(self, origin, direction):
        """Where a batch of rays crosses the mirror plane, and the direction each leaves it. The plane faces
        the camera, so reflecting about it flips the z of the direction and nothing else."""
        travel = (self.MIRROR_Z - origin[..., 2]) / direction[..., 2]
        hit = origin + np.expand_dims(travel, -1) * direction
        return hit, direction * np.array([1.0, 1.0, -1.0])

    def image_depth(self, ndc_x, ndc_y):
        """The axial distance `resolveSpecularGuides` writes for this pixel, plus the pinhole's hit on the
        mirror and its direction."""
        near, pinhole = dof.pinhole_rays(ndc_x, ndc_y, self.tan_half_width, self.TAN_HALF_HEIGHT)
        hit, _ = self.mirror_hit(near, pinhole)
        reach = np.linalg.norm(hit, axis=-1)
        return hit[..., 2] * (1.0 + self.OBJECT_BLOCKS / reach), hit, pinhole

    def miss_blocks(self, ndc_x, ndc_y, focus):
        """How far the aperture rays pass from the object point after the mirror: zero is a point image,
        which is what "in focus" means for the thing the reflection shows."""
        _, hit, pinhole = self.image_depth(ndc_x, ndc_y)
        mirrored = pinhole * np.array([1.0, 1.0, -1.0])
        target = hit + np.expand_dims(self.OBJECT_BLOCKS, -1) * mirrored
        disk = dof.sample_lens_disk(0, self.seeds)
        origin, direction = dof.lens_rays(ndc_x, ndc_y, self.tan_half_width, self.TAN_HALF_HEIGHT,
                                          disk, self.RADIUS, focus)
        hit_lens, reflected = self.mirror_hit(origin, direction)
        reflected /= np.linalg.norm(reflected, axis=-1, keepdims=True)
        offset = target - hit_lens
        along = np.sum(offset * reflected, axis=-1, keepdims=True)
        return float(np.max(np.linalg.norm(offset - along * reflected, axis=-1)))

    def test_focusing_the_image_gathers_the_reflection_anywhere_in_the_frame(self):
        for ndc_x, ndc_y in ((0.0, 0.0), (0.3, -0.2), (-0.6, 0.45)):
            depth, _, _ = self.image_depth(np.float64(ndc_x), np.float64(ndc_y))
            # The published number is eye-relative, like the depth it extends, while the lens measures its
            # focus from the near plane — the same 5 cm of slack the whole depth image already carries.
            self.assertLess(self.miss_blocks(ndc_x, ndc_y, float(depth) - dof.PROJECTION_Z_NEAR), 1.0e-9,
                            f"pixel ({ndc_x}, {ndc_y}): the image depth does not gather the mirror")
            self.assertLess(self.miss_blocks(ndc_x, ndc_y, float(depth)), 1.0e-3,
                            "the near-plane slack has to stay under a millimetre")

    def test_focusing_the_mirror_leaves_the_reflection_a_visible_disc(self):
        depth, hit, _ = self.image_depth(np.float64(0.0), np.float64(0.0))
        miss = self.miss_blocks(0.0, 0.0, float(hit[..., 2]))
        # The disc of the reflected image is the mirror's own focus throwing the image depth off:
        # aperture * |1 - depth / focus|. In pixels of a 1080-tall frame that is what a player sees.
        disc = dof.disc_pixels(float(depth), self.RADIUS, float(hit[..., 2]), 1080,
                               dof.DEFAULT_FOV_Y_RADIANS)
        self.assertGreater(miss, 1.0e-2, "measuring the mirror instead of its image barely moves the rays")
        self.assertGreater(disc, 4.0, "the two answers have to be several pixels apart to be worth a switch")

    def test_the_shader_reports_that_same_distance(self):
        source = GUIDES.read_text(encoding="utf-8")
        self.assertIn("imageDepth = min(curClip.w * (1.0 + reflectedRange / reach), REFLECT_IMAGE_MAX)",
                      source, "the meter's reflected distance must stay the image's axial distance")
        self.assertIn("gv_reflectShare >= REFLECT_IMAGE_SHARE", source,
                      "the image may only be reported where the mirror lobe is what the pixel shows")

    def test_the_depth_image_only_carries_images_while_the_meter_reads_it(self):
        # The same image is the per-pixel blur ceiling's history. Handing a manual focus's ceiling a
        # reflected distance would stop it narrowing the lens on the pixels nearest the plane.
        composite = COMPOSITE.read_text(encoding="utf-8")
        self.assertIn("if (usesFocusMeter() && CausticaConfig.Rt.DepthOfField.REFLECTION_FOCUS.value())",
                      composite, "the flag must ride the meter, not the lens")

    def test_a_water_face_is_a_reflection_at_a_grazing_look_and_not_at_a_steep_one(self):
        # The share gate in physical terms, using water's index, the dial's damping and the shader's line:
        # the incidence of water `d` blocks out from an eye 1.6 blocks up is atan(d / 1.6) from the normal.
        shader_share = float(group(r"REFLECT_IMAGE_SHARE = ([\d.]+);", GUIDES,
                                   "the share a reflection has to carry the pixel at").group(1))
        dial = float(group(r'"composite\.water-reflection",\s*([\d.]+)f', CONFIG,
                           "the water reflection dial").group(1))
        r0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2

        def share(distance):
            cos_incidence = self.EYE_HEIGHT / math.hypot(distance, self.EYE_HEIGHT)
            return (r0 + (1.0 - r0) * (1.0 - cos_incidence) ** 5) * dial

        self.assertGreater(share(20.0), shader_share, "the far half of a lake has to clear the gate")
        self.assertLess(share(2.0), shader_share, "the water at the player's feet must not — it shows the bed")


class ApertureWalk(unittest.TestCase):
    def setUp(self):
        self.seeds = dof.pixel_seeds(64, 48)

    def test_points_stay_inside_the_unit_disk(self):
        for frame in range(0, 1024, 13):
            disk = dof.sample_lens_disk(frame, self.seeds)
            self.assertLessEqual(float(np.max(np.linalg.norm(disk, axis=-1))), 1.0 + 1.0e-6)

    def test_walk_covers_the_disk_instead_of_clumping(self):
        seed = np.full((1, 1), int(self.seeds[24, 24]), dtype=np.uint32)
        frames = np.arange(1024, dtype=np.uint32).reshape(-1, 1, 1)
        points = dof.sample_lens_disk(frames, np.broadcast_to(seed, (1024, 1, 1))).reshape(-1, 2)
        self.assertEqual(points.shape, (1024, 2))
        quadrant = (points[:, 0] > 0).astype(int) + 2 * (points[:, 1] > 0).astype(int)
        counts = np.bincount(quadrant, minlength=4)
        self.assertLess(float(counts.max() / counts.min()), 1.6, f"quadrant counts {counts}")
        # Area-uniform by construction: the mean squared radius of a uniform disk is 1/2.
        self.assertAlmostEqual(float(np.mean(np.sum(points * points, axis=-1))), 0.5, delta=0.02)

    def test_pixels_do_not_share_a_phase(self):
        frame = dof.sample_lens_disk(5, self.seeds)
        self.assertGreater(float(np.std(np.diff(frame[:, :, 0], axis=1))), 0.3)
        self.assertGreater(float(np.std(np.diff(frame[:, :, 1], axis=0))), 0.3)

    def test_consecutive_frames_move_the_point(self):
        first = dof.sample_lens_disk(100, self.seeds)
        second = dof.sample_lens_disk(101, self.seeds)
        self.assertGreater(float(np.mean(np.linalg.norm(second - first, axis=-1))), 0.05)


class FocusRack(unittest.TestCase):
    def test_approaches_without_overshooting(self):
        current = 8.0
        seen = []
        for _ in range(60):
            current = dof.rack_focus(current, 2.0, 1.0 / 60.0)
            seen.append(current)
        self.assertTrue(all(a >= b for a, b in zip(seen, seen[1:])), "the rack overshot or oscillated")
        self.assertLess(abs(seen[-1] - 2.0), 0.05)

    def test_a_longer_focus_time_eases_the_plane_in(self):
        # The dial is the glide's time constant: one constant covers 63% of the distance, and a longer
        # one must arrive later at every step.
        one_tau = dof.rack_focus(8.0, 2.0, 0.2, 0.2)
        self.assertAlmostEqual(one_tau, 8.0 - 6.0 * (1.0 - math.exp(-1.0)), places=9)
        quick = dof.rack_focus(8.0, 2.0, 1.0 / 60.0, 0.05)
        slow = dof.rack_focus(8.0, 2.0, 1.0 / 60.0, 0.8)
        self.assertTrue(2.0 < quick < slow, f"the focus time dial inverted: {quick} vs {slow}")

    def test_is_frame_rate_independent(self):
        by_frame = 8.0
        for _ in range(120):
            by_frame = dof.rack_focus(by_frame, 2.0, 1.0 / 120.0)
        by_half = 8.0
        for _ in range(60):
            by_half = dof.rack_focus(by_half, 2.0, 1.0 / 60.0)
        self.assertAlmostEqual(by_frame, by_half, places=9)


class RenderedScene(unittest.TestCase):
    HEIGHT = 135
    WIDTH = 240

    def render_chart(self, depth, radius, focus, samples=40):
        scene = dof.Scene(charts=((depth, 0.0, 0.30),), aspect=self.WIDTH / self.HEIGHT)
        return dof.render(radius, focus, samples, scene, self.HEIGHT, self.WIDTH)

    def test_pinhole_ignores_the_focus(self):
        reference = self.render_chart(8.0, 0.0, 8.0, samples=1)
        for focus in (2.0, 8.0, 40.0):
            self.assertLess(float(np.max(np.abs(
                self.render_chart(8.0, 0.0, focus, samples=5) - reference))), 1.0e-12,
                f"a pinhole camera changed with the focus dial at {focus} blocks")

    def test_the_focal_plane_keeps_its_detail_and_the_near_chart_loses_it(self):
        radius = dof.lens_radius_blocks(dof.DEFAULT_APERTURE_BLOCKS, dof.DEFAULT_FOCUS_BLOCKS,
                                        dof.DEFAULT_DISPLAY_HEIGHT, dof.DEFAULT_FOV_Y_RADIANS)
        sharp_focus = dof.patch_contrast(self.render_chart(8.0, 0.0, 8.0, samples=1), 0.0, 0.30)
        lens_focus = dof.patch_contrast(self.render_chart(8.0, radius, 8.0), 0.0, 0.30)
        sharp_near = dof.patch_contrast(self.render_chart(2.0, 0.0, 8.0, samples=1), 0.0, 0.30)
        lens_near = dof.patch_contrast(self.render_chart(2.0, radius, 8.0), 0.0, 0.30)
        self.assertGreater(lens_focus, 0.85 * sharp_focus,
                           f"the chart at the focal plane lost detail: {sharp_focus:.3f} -> "
                           f"{lens_focus:.3f}")
        self.assertLess(lens_near, 0.85 * sharp_near,
                        f"the near chart kept its detail: {sharp_near:.3f} -> {lens_near:.3f}")
        self.assertLess(lens_near, lens_focus,
                        "the near chart must be softer than the chart on the focal plane")

    def test_focusing_the_near_chart_makes_it_sharp_again(self):
        radius = dof.lens_radius_blocks(dof.DEFAULT_APERTURE_BLOCKS, 2.0, dof.DEFAULT_DISPLAY_HEIGHT,
                                        dof.DEFAULT_FOV_Y_RADIANS)
        sharp = dof.patch_contrast(self.render_chart(2.0, 0.0, 2.0, samples=1), 0.0, 0.30)
        lens = dof.patch_contrast(self.render_chart(2.0, radius, 2.0), 0.0, 0.30)
        self.assertGreater(lens, 0.85 * sharp,
                           f"the chart on the focal plane lost detail: {sharp:.3f} -> {lens:.3f}")


if __name__ == "__main__":
    unittest.main()