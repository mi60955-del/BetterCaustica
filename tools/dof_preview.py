"""Offline preview of the traced camera's lens (depth of field).

`shaders/pipelines/world/primary.rgen.slang` builds its primary ray by moving the pixel's pinhole ray
sideways inside the sensor plane and re-aiming it at the plane the lens is focused on. This module ports
that construction, the aperture sampler in `math.slang` and the host's blur-budget arithmetic in
`RtDepthOfField.java` into numpy, and renders a synthetic scene through it: a checked ground plane running
away from the camera, and charts of bright dots standing at chosen depths.

It is a geometric model, not a pixel-exact copy: no textures, no fog, no reflection, no denoiser, and the
sheet averages its aperture points over samples where the game converges them over frames. What it does
reproduce is where each sample lands, which is the only thing a lens changes.

    python tools/dof_preview.py --width 640 --height 360
    python -m unittest tools.test_dof_preview

Every axial distance here — the focus the lens is aimed at and a chart's depth — is measured from the lens
plane, as the raygen measures it: the lens sits on the near plane rather than at the eye, because view
bobbing is baked into the projection and the near plane is the plane the unprojection can name.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

# --- mirrored constants ------------------------------------------------------------------------
# Each value is pinned to its declaration in the shaders or in Java by tools/test_dof_preview.py.
LENS_LATTICE_STEPS = (0.7548776662, 0.5698402910)  # math.slang LENS_LATTICE_STEPS
LENS_LATTICE_INDEX_MASK = 1023  # math.slang sampleLensDisk's frame-index wrap
LENS_PIXEL_SEED = (1973, 9277, 26699)  # primary.rgen's pixel seed, before the frame index
PROJECTION_Z_NEAR = 0.05  # blocks; the preview's own near plane, which is where its lens sits
EYE_HEIGHT = 1.6  # blocks; the preview camera's height over the ground plane
FOCUS_PROBE_BLOCKS = 64.0  # RtDepthOfField.FAR_BLOCKS, the meter's far stop
FOCUS_NEAR_BLOCK = 0.05  # RtFocusProbe.NEAR_BLOCK, the closest range a tap may answer
FOCUS_RACK_SECONDS = 0.2  # the dof.rack-time default
DEFAULT_APERTURE_BLOCKS = 0.04  # CausticaConfig.Rt.DepthOfField.APERTURE: a 4 cm lens
DEFAULT_MAX_BLUR_PIXELS = 14.0  # CausticaConfig.Rt.DepthOfField.MAX_BLUR_PIXELS
DEFAULT_FOCUS_BLOCKS = 8.0  # CausticaConfig.Rt.DepthOfField.FOCUS_BLOCKS
DEFAULT_DISPLAY_HEIGHT = 1080  # the height RtComposite reads the ceiling off
DEFAULT_FOV_Y_RADIANS = math.radians(70.0)

_UINT32 = np.uint32


def pcg(state):
    """The shader's PCG hash, on uint32 wraps."""
    state = _UINT32(state * np.uint32(747796405) + np.uint32(2891336453))
    roll = state >> np.uint32((state >> np.uint32(28)) + np.uint32(4))
    w = _UINT32(np.bitwise_xor(roll, state) * np.uint32(277803737))
    return _UINT32(np.bitwise_right_shift(w, np.uint32(22)) ^ w)


def rndf_batch(values):
    """`rndf` over an array of streams: the high 24 bits, exactly as the shader converts them."""
    return np.asarray(_UINT32(pcg(values) >> np.uint32(8)) * (1.0 / 16777216.0), dtype=np.float64)


def pixel_seeds(width: int, height: int) -> np.ndarray:
    """The primary raygen's per-pixel seed, before the frame index is folded into the path stream."""
    x = np.arange(width, dtype=np.uint32)[None, :] * np.uint32(LENS_PIXEL_SEED[0])
    y = np.arange(height, dtype=np.uint32)[:, None] * np.uint32(LENS_PIXEL_SEED[1])
    return _UINT32(x + y + np.uint32(LENS_PIXEL_SEED[2]))


def sample_lens_disk(frame_index: int, seeds) -> np.ndarray:
    """Port of `math.slang.sampleLensDisk` over an array of pixel seeds: points inside the unit disk."""
    phase_seed = pcg(np.asarray(seeds, dtype=_UINT32))
    phase_x = rndf_batch(phase_seed)
    phase_y = rndf_batch(phase_seed)
    # Accepts a scalar frame index or an array of them, so a test can walk a pixel's whole run at once.
    walk = np.asarray(np.bitwise_and(np.asarray(frame_index, dtype=_UINT32),
                                     _UINT32(LENS_LATTICE_INDEX_MASK)), dtype=np.float64)
    u = np.mod(walk * LENS_LATTICE_STEPS[0] + phase_x, 1.0)
    v = np.mod(walk * LENS_LATTICE_STEPS[1] + phase_y, 1.0)
    radius = np.sqrt(u)
    phi = 2.0 * math.pi * v
    return np.stack([radius * np.cos(phi), radius * np.sin(phi)], axis=-1)


# --- the host's arithmetic (RtDepthOfField) ----------------------------------------------------


def tangent_per_display_pixel(display_height: int, fov_y_radians: float) -> float:
    """Port of `RtDepthOfField.tangentPerDisplayPixel`: the share of the frame's tangent span one display
    pixel covers, which is the unit a surface's blur offset is measured in (offset over depth)."""
    return 2.0 * math.tan(fov_y_radians / 2.0) / display_height


def blur_ceiling(display_height, fov_y_radians, cap_pixels=DEFAULT_MAX_BLUR_PIXELS):
    """Port of the host's cap, in shares of the frame's tangent span. Zero is a cap switched off, and it
    has to disable the clamp rather than close the lens -- which is why every caller tests it instead of
    multiplying it into a bound."""
    if cap_pixels <= 0.0:
        return 0.0
    return cap_pixels * tangent_per_display_pixel(display_height, fov_y_radians)


def lens_radius_blocks(aperture, focus_blocks, display_height, fov_y_radians, depth=None,
                       cap_pixels=DEFAULT_MAX_BLUR_PIXELS, per_pixel=True):
    """Port of the two clamps, host and raygen. Without `depth` only the horizon bound applies, which is
    all the host can see; with it the answer is one radius per pixel, narrowed by the distance that pixel
    showed. A surface at depth s spreads over radius * (1/s - 1/f) in tangent space while it sits in front
    of the focal plane, so the bound on the radius is ceiling * s * f / (f - s); behind the plane the
    spread falls back on radius / f, which is the horizon bound. `per_pixel=False` is the Cap Per Pixel
    switch off: the horizon bound stays and no pixel gets its own."""
    ceiling = blur_ceiling(display_height, fov_y_radians, cap_pixels)
    if ceiling == 0.0:
        # An uncapped lens is the dial, at every depth.
        return aperture if depth is None else np.full(np.shape(depth), float(aperture))
    radius = min(aperture, ceiling * focus_blocks)
    if depth is None or not per_pixel:
        return radius if depth is None else np.full(np.shape(depth), float(radius))
    seen = np.clip(np.asarray(depth, dtype=np.float64), 0.0, focus_blocks)
    behind = np.maximum(focus_blocks - seen, 1.0e-3)
    return np.minimum(radius, ceiling * seen * focus_blocks / behind)


def meter_focus(depths, near_block=FOCUS_NEAR_BLOCK, far_block=FOCUS_PROBE_BLOCKS) -> float:
    """Port of `pipelines/dof_focus`: the harmonic mean of a window's distances, which is its mean dioptric
    distance inverted -- 1/range is the quantity a lens focuses against, so averaging ranges would answer a
    near surface surrounded by a field as if the field were the subject. A range that is not positive is a
    texel no trace wrote and is left out of the mean; a window with nothing valid in it answers 0, which is
    the host's 'no measurement yet'."""
    values = np.asarray(depths, dtype=np.float64).ravel()
    valid = (values > 0.0) & (values < 1.0e7)
    taps = int(np.count_nonzero(valid))
    if taps == 0:
        return 0.0
    diopters = float(np.sum(1.0 / np.clip(values[valid], near_block, far_block)))
    return taps / diopters if diopters > 0.0 else 0.0


def rack_focus(current: float, target: float, frame_seconds: float,
               rack_seconds: float = FOCUS_RACK_SECONDS) -> float:
    """Port of `RtDepthOfField.rackFocus`: exponential approach, so the plane covers the same fraction of
    its remaining distance per second whatever the frame rate is. `rack_seconds` is the Focus Time dial."""
    approach = 1.0 - math.exp(-max(0.0, frame_seconds) / rack_seconds)
    return current + (target - current) * approach


# --- the lens (primary.rgen) -------------------------------------------------------------------


def pinhole_rays(ndc_x, ndc_y, tan_half_width, tan_half_height):
    """The pixel's pinhole ray as the raygen reconstructs it: a point on the near plane and a direction
    toward the far plane. The eye sits at the origin here; `Scene` adds its height when it traces."""
    near = np.stack([ndc_x * tan_half_width * PROJECTION_Z_NEAR,
                     ndc_y * tan_half_height * PROJECTION_Z_NEAR,
                     np.full_like(ndc_x, PROJECTION_Z_NEAR)], axis=-1)
    far = np.stack([ndc_x * tan_half_width * 1000.0,
                    ndc_y * tan_half_height * 1000.0,
                    np.full_like(ndc_x, 1000.0)], axis=-1)
    direction = far - near
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return near, direction


def lens_rays(ndc_x, ndc_y, tan_half_width, tan_half_height, disk, radius, focus):
    """Port of the raygen's lens branch. `disk` holds unit-disk points, and each becomes a ray that starts
    at that point of the aperture and crosses the focal plane where the pinhole ray does."""
    near, pinhole = pinhole_rays(ndc_x, ndc_y, tan_half_width, tan_half_height)
    # An axis-aligned camera's sensor frame is the world's own axes. The shader derives the same three
    # vectors from the inverse projection and orthogonalises the unprojected x axis against the axis it
    # measured with, which for a symmetric frustum is exactly this.
    axis = np.array([0.0, 0.0, 1.0])
    # `radius` is the dialled lens, or one ceiling-tightened radius per pixel.
    wide = np.asarray(radius)[..., None] if np.ndim(radius) else radius
    lateral = disk[..., 0:1] * wide * np.array([1.0, 0.0, 0.0]) \
        + disk[..., 1:2] * wide * np.array([0.0, 1.0, 0.0])
    origin = near + lateral
    along_axis = np.clip(np.einsum("...i,i->...", pinhole, axis), 1.0e-4, None)
    focal_point = near + pinhole * (focus / along_axis)[..., None]
    direction = focal_point - origin
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return origin, direction


def circle_of_confusion(depth: float, radius: float, focus: float) -> float:
    """The analytic disc the construction throws on a surface `depth` blocks past the lens plane:
    `radius * |1 - depth / focus|` blocks measured there, which is that over depth in tangent space. It
    closes on the focal plane and the far field settles at radius / focus."""
    return radius * abs(1.0 - depth / focus)


def disc_pixels(depth: float, radius: float, focus: float, frame_height: int,
                fov_y_radians: float) -> float:
    """The same disc in pixels of the finished frame, which is the unit a player judges a blur by. The
    offset is already a tangent-space ratio, so it converts on the frame's own vertical span."""
    return circle_of_confusion(depth, radius, focus) / depth \
        / (2.0 * math.tan(fov_y_radians / 2.0)) * frame_height


# --- the synthetic scene -----------------------------------------------------------------------


class Scene:
    """A checked ground plane at y = 0 and charts of bright dots standing at chosen depths.

    Each chart is described on the screen (a depth, an ndc centre and an ndc half-size) and its world
    rectangle is derived from that, so every chart fills the same part of the frame whatever its depth.
    The dot period scales with depth for the same reason: a fixed world period is sub-pixel on a far chart
    and would read as flat texture whether the lens was sharp there or not. A constant on-screen period
    also turns a chart's contrast into a pass band of the lens at that one spatial frequency, which is
    what makes the sheet's numbers comparable between depths.
    """

    def __init__(self, charts=((0.6, -0.55, 0.22), (8.0, 0.0, 0.22), (40.0, 0.55, 0.22)),
                 fov_y_radians=DEFAULT_FOV_Y_RADIANS, aspect=16 / 9):
        self.tan_half_height = math.tan(fov_y_radians / 2.0)
        self.tan_half_width = self.tan_half_height * aspect
        self.charts = tuple(charts)

    def chart_rect(self, depth, ndc_x, ndc_half):
        """The chart's world rectangle: x0, x1, y0, y1 at `PROJECTION_Z_NEAR + depth`."""
        z = PROJECTION_Z_NEAR + depth
        half = ndc_half * z * self.tan_half_width
        centre = ndc_x * z * self.tan_half_width
        y_half = ndc_half * z * self.tan_half_height
        return centre - half, centre + half, EYE_HEIGHT - y_half, EYE_HEIGHT + y_half

    def chart_period(self, depth):
        """A dot period of about 5% of the depth, i.e. a constant span of screen: 1.2% of the frame's
        height per dot for the preview's default field of view."""
        return 0.05 * depth

    def hit(self, origin, direction):
        """Nearest intersection per ray, in camera space with the eye raised to its height. Returns the
        travel and a kind per ray: 0 the ground, 1..n the charts in depth order, and an infinite travel
        where the ray reaches nothing."""
        origin = origin + np.array([0.0, EYE_HEIGHT, 0.0])
        t = np.full(origin.shape[:-1] + (len(self.charts) + 1,), np.inf)
        downward = direction[..., 1] < -1.0e-6
        ground = np.where(downward, -origin[..., 1] / np.where(downward, direction[..., 1], 1.0), np.inf)
        t[..., 0] = np.where(np.isfinite(ground) & (ground > 1.0e-3), ground, np.inf)
        for index, (depth, ndc_x, ndc_half) in enumerate(self.charts, start=1):
            x0, x1, y0, y1 = self.chart_rect(depth, ndc_x, ndc_half)
            forward = direction[..., 2] > 1.0e-6
            z = PROJECTION_Z_NEAR + depth
            plane = np.where(forward, (z - origin[..., 2]) / np.where(forward, direction[..., 2], 1.0),
                             np.inf)
            point = origin + np.expand_dims(plane, -1) * direction
            inside = (point[..., 0] > x0) & (point[..., 0] < x1)                 & (point[..., 1] > y0) & (point[..., 1] < y1)
            t[..., index] = np.where(np.isfinite(plane) & (plane > 1.0e-3) & inside, plane, np.inf)
        kind = np.argmin(t, axis=-1)
        travel = np.take_along_axis(t, kind[..., None], -1).squeeze(-1)
        return origin, travel, kind

    def trace(self, origin, direction):
        """The hit points, with the kind that selected them."""
        origin, travel, kind = self.hit(origin, direction)
        point = origin + np.expand_dims(np.where(np.isfinite(travel), travel, 0.0), -1) * direction
        return point, kind

    def axial_depth(self, origin, direction):
        """How far down the axis each ray's surface sits past the lens plane. A ray that reaches nothing
        reports the far field, which is what the guide point's own distance does in the game: the horizon
        bound then applies to it rather than a per-pixel one."""
        origin, travel, _ = self.hit(origin, direction)
        reached = np.isfinite(travel)
        point = origin + np.expand_dims(np.where(reached, travel, 0.0), -1) * direction
        return np.where(reached, np.maximum(point[..., 2] - PROJECTION_Z_NEAR, 0.0), 1.0e6)

    def color(self, point, kind):
        """A chart is a grid of bright dots on a dark field, which is what a blur disc reads as: a dot of
        the wrong size is a disc of the wrong size. The ground is a checkerboard of one-block squares, so
        the same blur reads there as a softened edge."""
        u = point[..., 0]
        chart = kind > 0
        # A chart is measured across its own face; the ground is measured in plan, so its checker runs in
        # x and z rather than collapsing into stripes at a constant height.
        v = np.where(chart, point[..., 1], point[..., 2])
        bright = np.zeros(kind.shape, dtype=bool)
        for index, (depth, _, _) in enumerate(self.charts, start=1):
            period = self.chart_period(depth)
            du = u / period - np.floor(u / period) - 0.5
            dv = v / period - np.floor(v / period) - 0.5
            bright = np.where(kind == index, du * du + dv * dv < 0.10, bright)
        check = (np.floor(u) + np.floor(v)) % 2 == 0
        bright = np.where(chart, bright, check)
        tint = np.where(chart[..., None], np.array([0.92, 0.88, 0.80]), np.array([0.60, 0.57, 0.52]))
        shade = np.where(chart[..., None], np.array([0.04, 0.04, 0.05]), np.array([0.13, 0.12, 0.11]))
        return tint * bright[..., None] + shade * (1.0 - bright[..., None])


def frame_screen(height: int, width: int):
    """The frame's pixel-centre ndc coordinates."""
    ys = np.arange(height, dtype=np.float64)
    xs = np.arange(width, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return grid_x / width * 2.0 - 1.0 + 1.0 / width, grid_y / height * 2.0 - 1.0 + 1.0 / height


def render(aperture, focus, samples, scene: Scene, height=270, width=480,
           fov_y_radians=DEFAULT_FOV_Y_RADIANS, display_height=DEFAULT_DISPLAY_HEIGHT,
           cap_pixels=DEFAULT_MAX_BLUR_PIXELS, per_pixel=True):
    """Average `samples` aperture points per pixel, each one narrowed by the ceiling against the depth
    this pixel's pinhole ray reaches. That depth is last frame's in the game and this frame's here: the
    preview can afford the honest number, and the two agree anywhere the camera is not moving.

    The ceiling is counted in pixels of the *game's* frame, so `display_height` is the game's height
    while the panel is drawn at `height`; the discs therefore have the game's angular size, drawn on a
    smaller canvas.
    """
    tan_half_height = math.tan(fov_y_radians / 2.0)
    tan_half_width = tan_half_height * width / height
    ndc_x, ndc_y = frame_screen(height, width)
    seeds = pixel_seeds(width, height)
    image = np.zeros((height, width, 3), dtype=np.float64)
    if aperture <= 0.0:
        radius = 0.0
    else:
        near, pinhole = pinhole_rays(ndc_x, ndc_y, tan_half_width, tan_half_height)
        radius = lens_radius_blocks(aperture, focus, display_height, fov_y_radians,
                                    depth=scene.axial_depth(near, pinhole), cap_pixels=cap_pixels,
                                    per_pixel=per_pixel)
    pinhole_only = float(np.max(radius)) == 0.0
    for frame in range(samples):
        if pinhole_only:
            origin, direction = pinhole_rays(ndc_x, ndc_y, tan_half_width, tan_half_height)
        else:
            origin, direction = lens_rays(ndc_x, ndc_y, tan_half_width, tan_half_height,
                                          sample_lens_disk(frame, seeds), radius, focus)
        point, kind = scene.trace(origin, direction)
        image += scene.color(point, kind)
    return image / max(1, samples)


def tonemap(linear):
    """A display curve just good enough to look at; the preview is about geometry, not grading."""
    clipped = np.clip(linear, 0.0, None)
    return np.clip(np.power(clipped / (1.0 + clipped), 1 / 2.2), 0.0, 1.0)


def patch_contrast(image, ndc_x, ndc_half):
    """The spread of luminance inside one chart's part of the frame. Blur flattens it: dots that have
    spread into each other measure less contrast than the same chart seen by a pinhole, which is the
    number the sheet is read with."""
    height, width = image.shape[:2]
    inside = 0.8 * ndc_half  # stay off the chart's own edge, where the ground behind it shows through
    columns = slice(int((ndc_x - inside + 1.0) * 0.5 * width),
                    int((ndc_x + inside + 1.0) * 0.5 * width))
    rows = slice(int((1.0 - inside) * 0.5 * height), int((1.0 + inside) * 0.5 * height))
    return float(np.std(tonemap(image[rows, columns]).mean(axis=-1)))


def panel(title, image, width):
    array = (tonemap(image) * 255.0).astype(np.uint8)
    band = np.full((20, array.shape[1], 3), 18, dtype=np.uint8)
    out = Image.fromarray(np.concatenate([band, array], axis=0))
    ImageDraw.Draw(out).text((6, 5), title, fill=(225, 225, 225))
    if out.width != width:
        out = out.resize((width, round(out.height * width / out.width)))
    return out


def stack(panels, columns):
    rows = math.ceil(len(panels) / columns)
    tile_width, tile_height = panels[0].width, panels[0].height
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), (10, 10, 10))
    for index, tile in enumerate(panels):
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Render the traced lens against a synthetic scene.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--display-height", type=int, default=DEFAULT_DISPLAY_HEIGHT,
                        help="the height the horizon ceiling is counted in, as RtComposite passes it")
    parser.add_argument("--focus", type=float, default=DEFAULT_FOCUS_BLOCKS)
    parser.add_argument("--aperture", type=float, default=DEFAULT_APERTURE_BLOCKS * 100.0,
                        help="the lens radius in centimetres, as the Aperture dial shows it")
    parser.add_argument("--fov-y", type=float, default=math.degrees(DEFAULT_FOV_Y_RADIANS))
    parser.add_argument("--cap", type=float, default=DEFAULT_MAX_BLUR_PIXELS,
                        help="the Max Blur Radius dial; 0 switches the cap off")
    parser.add_argument("--no-pixel-cap", action="store_true",
                        help="switch Cap Per Pixel off: the horizon bound alone, no depth read-back")
    parser.add_argument("--out", default=os.path.join("tools", "dof-preview-out"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fov = math.radians(args.fov_y)
    scene = Scene(fov_y_radians=fov, aspect=args.width / args.height)
    depths = tuple(chart[0] for chart in scene.charts)
    aperture = args.aperture / 100.0
    per_pixel = not args.no_pixel_cap

    cases = [
        ("pinhole: aperture 0, the camera this feature replaced",
         0.0, args.focus, args.samples, args.cap, True),
        (f"aperture {aperture * 100:.0f} cm, cap {args.cap:.0f} px, focus {args.focus:.1f} blocks",
         aperture, args.focus, args.samples, args.cap, True),
        (f"aperture {aperture * 300:.0f} cm, cap {args.cap:.0f} px",
         aperture * 3, args.focus, args.samples, args.cap, True),
        ("cap off: the same lens, discs unbounded (the physically correct camera)",
         aperture, args.focus, args.samples, 0.0, True),
        ("cap per pixel off: the horizon bound alone, so the foreground runs away",
         aperture, args.focus, args.samples, args.cap, False),
        (f"one frame, aperture {aperture * 100:.0f} cm: what the denoiser starts from",
         aperture, args.focus, 1, args.cap, True),
    ]
    images = [render(dial, focus, samples, scene, args.height, args.width, fov,
                     display_height=args.display_height, cap_pixels=cap, per_pixel=pixel)
              for _, dial, focus, samples, cap, pixel in cases]
    sheet = stack([panel(title, image, args.width)
                   for (title, _, _, _, _, _), image in zip(cases, images)], 3)
    path = os.path.join(args.out, "dof-sheet.png")
    sheet.save(path)

    print(f"wrote {path} ({sheet.width}x{sheet.height}); "
          f"{args.width}x{args.height} per panel, {args.samples} aperture samples per pixel")
    ceiling = blur_ceiling(args.display_height, fov, args.cap)
    bound = f"{ceiling * args.focus * 100:.2f} cm" if ceiling else "none (cap off)"
    print(f"lens: dialled {args.aperture:.0f} cm, horizon bound {bound} at focus {args.focus:.1f} "
          f"blocks, cap {args.cap:.0f} px")
    print("chart depth   disc (cm)   disc (px)   bounded (px)   contrast pinhole -> lens")
    for depth, ndc_x, ndc_half in scene.charts:
        radius = lens_radius_blocks(aperture, args.focus, args.display_height, fov,
                                    cap_pixels=args.cap)
        bounded = lens_radius_blocks(aperture, args.focus, args.display_height, fov, depth=depth,
                                     cap_pixels=args.cap, per_pixel=per_pixel)
        before = patch_contrast(images[0], ndc_x, ndc_half)
        after = patch_contrast(images[1], ndc_x, ndc_half)
        print(f"{depth:8.1f} {circle_of_confusion(depth, radius, args.focus) * 100:12.1f}"
              f"{disc_pixels(depth, radius, args.focus, args.display_height, fov):12.1f}"
              f"{disc_pixels(depth, bounded, args.focus, args.display_height, fov):14.1f}"
              f"{before:14.3f} ->{after:7.3f}")
    print("the fourth panel is the cap switched off, which is the mosaic the cap exists to prevent; the "
          "fifth is the cap with Cap Per Pixel off, where only the horizon is held and the foreground "
          "keeps its runaway discs. The last is one aperture point per pixel, what the denoiser starts "
          "from.")


if __name__ == "__main__":
    main()
