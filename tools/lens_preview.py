"""Offline preview and cost model of the lens pass (shaders/pipelines/lens/*.slang).

A numpy port of the two dispatches that conjure the three lens artifacts, driven by a synthetic HDR
frame instead of the traced image, so their shape and strength can be iterated in seconds rather than
one game round-trip per parameter change:

  hot source  A 16-display-pixel box downsample of the exposed rt image, thresholded and softly
              compressed into "how badly is this part of the frame overloading the lens". Everything
              below the threshold (the sky, lit surfaces) is exactly zero, so flares only come from
              sources far above white. Sampled 8x8 per output texel so a sun smaller than one texel of
              the source texture still lands in it instead of flickering as the camera moves.
  Starburst   Diffraction spikes from the aperture blades: a ray integral through the hot field,
              `SPIKE_SAMPLES` taps along each of `spikes` directions.
  Halos       Aperture rings at `haloRadius` around every hot core, with the three channels gathered at
              slightly different radii so the ring splits into a chromatic rim.
  Lens dirt   Out-of-focus dust on the front element, lit by the scene's own bloom glow. Procedural: a
              three-octave lattice of soft-rimmed specks, so no texture is sampled and the pattern is
              identical on every machine.

The tool prints the per-pixel tap budget of each stage because that is what the pass costs: the
algorithm is a fixed number of texture fetches per output texel, and both textures are an eighth of the
display size in each axis (1/64 the pixels), so the whole thing is two small dispatches.

Constants prefixed with `SHADER_` must match the shaders; test_lens_preview.py parses both and fails
when they drift. Values are float32 throughout, matching the shader's arithmetic closely enough to
judge the look; this is a shape/energy preview, not a bit-exact reference.

Usage:
    python tools/lens_preview.py                       # 960x540 sheet into tools/lens-preview-out/
    python tools/lens_preview.py --width 1920 --height 1080 --out some/dir
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32

# --- mirrored from shaders/pipelines/lens/source.comp.slang -------------------------------------------
SHADER_HOT_BOX_TAPS = 4              # per axis, so 16 bilinear fetches per hot texel
SHADER_HOT_BOX_STEP = F32(1.0)       # display pixels between box taps; 4 * 1 = a 4 px footprint
SHADER_HOT_THRESHOLD = F32(2.5)      # exposed ACEScg luma removed before the soft ceiling
SHADER_HOT_CEILING = F32(50.0)       # soft ceiling: v / (1 + v / ceiling), the bloom prefilter's cap

# Divisor on the display size for the source field, fixed rather than following the resolution dial: its
# coarseness is what leaves the rays thin, so it is a look constant. RtComposite allocates the source image
# at this divisor whatever the flare image's own resolution is.
SOURCE_DIVISOR = 8

# --- mirrored from shaders/pipelines/lens/main.comp.slang ----------------------------------------------
SHADER_SPIKE_FALLOFF = F32(3.0)      # exp(-falloff * f) along a spike, f = distance / length
SHADER_SPIKE_GAIN = F32(0.012)
DEFAULT_HALO_TAPS = 24               # the mod's lens.halo-taps default; 8-48 in the settings
SHADER_HALO_SUBTAPS = 2              # fetches per tap, so 24 taps cover the ring at ~2 texels apart
SHADER_HALO_ELLIPSE = F32(0.90)      # vertical ring radius as a fraction of the horizontal one
SHADER_HALO_CHROMA = np.array([1.0, 0.0, -1.0], dtype=F32)  # multipliers on `dispersion`
HALO_PHASE = (0.0, 0.37, 0.71)       # per-channel tap rotation, so the three ring gathers interleave
SHADER_HALO_GAIN = F32(0.04)
SHADER_DIRT_FREQ = (F32(6.0), F32(13.0), F32(27.0))  # DIRT_FREQ_COARSE/MEDIUM/FINE, cells per height
SHADER_DIRT_WEIGHT = (F32(1.0), F32(0.7), F32(0.45))   # DIRT_WEIGHT_COARSE/MEDIUM/FINE
SHADER_DIRT_SEED = (0x51ED, 0x6A4D, 0x8321)            # DIRT_SEED_COARSE/MEDIUM/FINE
SHADER_DIRT_SMUDGE_FREQ = F32(2.6)   # cells per texture height for the low-frequency grime
SHADER_DIRT_EMPTY = F32(0.42)        # hash above this leaves a cell without a speck
SHADER_DIRT_LIGHT_CEILING = F32(8.0) # soft ceiling on the light, so a sun is not an opaque white blot
SHADER_DIRT_GAIN = F32(0.42)
SHADER_DIRT_CELL_MARGIN = F32(0.20)          # a speck's centre stays inside the middle of its cell
SHADER_DIRT_CELL_SPREAD = F32(0.60)
SHADER_DIRT_SPECK_MIN_RADIUS = F32(0.11)
SHADER_DIRT_SPECK_RADIUS_SPREAD = F32(0.17)
SHADER_DIRT_SPECK_BODY = F32(0.70)           # dim body plus brighter rim, like a defocused grain
SHADER_DIRT_SPECK_RIM = F32(0.30)
SHADER_DIRT_RIM_CENTER = F32(0.82)
SHADER_DIRT_RIM_WIDTH = F32(0.06)
SHADER_DIRT_GRIME_FLOOR = F32(0.55)          # how grimy the element is at its cleanest
SHADER_DIRT_GRIME_RANGE = F32(0.45)
SHADER_DIRT_SMUDGE_SEED = 0x2A17

ACESCG_LUMA = np.array([0.2722287168, 0.6740817658, 0.0536895174], dtype=F32)


# --- sampling ---------------------------------------------------------------------------------------
def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Clamp-to-edge bilinear fetch, the sampler mode every lens input uses."""
    h, w = img.shape[:2]
    x = np.clip(u, 0.0, 1.0) * F32(w - 1)
    y = np.clip(v, 0.0, 1.0) * F32(h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def pixel_grid(w: int, h: int):
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    return (xx + F32(0.5)) / F32(w), (yy + F32(0.5)) / F32(h)


def luminance(img: np.ndarray) -> np.ndarray:
    return np.einsum("...c,c->...", img, ACESCG_LUMA).astype(F32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian, edge-clamped; the preview's stand-in for kernel-shaped PyTorch-free blurs."""
    if sigma <= 0.0:
        return img
    radius = max(1, int(round(sigma * 2.5)))
    taps = np.arange(-radius, radius + 1, dtype=F32)
    kernel = np.exp(-(taps * taps) / F32(2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    pad = ((radius, radius), (radius, radius)) + (((0, 0),) if img.ndim == 3 else ())
    padded = np.pad(img, pad, mode="edge")
    out = img
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for i, k in enumerate(kernel):
            if axis == 0:
                acc += padded[i : i + out.shape[0], radius : radius + out.shape[1]] * k
            else:
                acc += padded[radius : radius + out.shape[0], i : i + out.shape[1]] * k
        out = acc
    return out


# --- synthetic frame --------------------------------------------------------------------------------
def soft_disk(u, v, cx, cy, radius, aspect):
    """Antialiased disk in screen-square coordinates (x divided by aspect, y as is)."""
    dx = (u - F32(cx)) * F32(aspect)
    dy = v - F32(cy)
    d = np.sqrt(dx * dx + dy * dy) / F32(radius)
    return np.clip((F32(1.0) - d) / F32(0.35), 0.0, 1.0)


def synthetic_scene(w: int, h: int) -> np.ndarray:
    """An HDR frame with a sun, two practical lights, lit windows and a mid-grey card.

    Deliberately extreme: a sun of 9000 in scene units is what makes lens artifacts appear at all, and
    the card is the pass-through reference that must stay unchanged.
    """
    rng = np.random.default_rng(7)
    u, v = pixel_grid(w, h)
    aspect = F32(w) / F32(h)

    sky = np.stack(
        [F32(0.55) - F32(0.18) * v, F32(0.62) - F32(0.20) * v, F32(0.78) - F32(0.22) * v], axis=-1
    ).astype(F32)
    ground = np.stack([F32(0.16), F32(0.13), F32(0.10)], axis=-1).astype(F32) * (
        F32(1.0) - F32(0.5) * (v - F32(0.62))
    )[..., None]
    scene = np.where((v > F32(0.62))[..., None], ground, sky).astype(F32)

    card = (u > F32(0.63)) & (u < F32(0.83)) & (v > F32(0.30)) & (v < F32(0.58))
    scene[card] = F32(0.20)

    sun = soft_disk(u, v, 0.30, 0.26, 0.010, aspect)
    scene += sun[..., None] * np.array([9000.0, 8600.0, 7800.0], dtype=F32)

    for cx, cy, rad, val, tint in (
        (0.52, 0.35, 0.006, 320.0, (1.0, 0.72, 0.42)),
        (0.44, 0.47, 0.004, 90.0, (0.55, 0.72, 1.0)),
    ):
        scene += soft_disk(u, v, cx, cy, rad, aspect)[..., None] * np.array(tint, dtype=F32) * F32(val)

    for cx, cy in ((0.70, 0.36), (0.74, 0.36), (0.70, 0.44), (0.78, 0.44), (0.86, 0.40)):
        win = (np.abs(u - F32(cx)) < F32(0.012)) & (np.abs(v - F32(cy)) < F32(0.020))
        scene[win] = np.array([38.0, 26.0, 14.0], dtype=F32)

    scene += rng.normal(0.0, 0.004, scene.shape).astype(F32)
    return np.maximum(scene, F32(0.0)).astype(F32)


def bloom_glow(scene: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    """Stand-in for the exposed bloom level 0: bright pass, then the pyramid's wide skirt.

    The real pass thresholds in exposed scene-linear ACEScg and sums one tent-blurred band per level, so
    energy decays like a real lens skirt rather than a single Gaussian. Three scales reproduce that shape
    well enough to judge how much light the dust gets, and to see why the glow itself is the wrong
    source for spikes: it is far wider than any tap spacing the lens pass could afford.
    """
    lum = luminance(scene)
    knee = F32(0.5)
    soft = np.clip(lum - F32(threshold) + knee, 0.0, 2.0 * knee)
    soft = soft * soft / (4.0 * knee)
    contribution = np.maximum(lum - F32(threshold), soft)
    bright = scene * (contribution / np.maximum(lum, F32(1e-6)))[..., None]
    glow = (
        gaussian_blur(bright, 2.0) * F32(0.55)
        + gaussian_blur(bright, 9.0) * F32(0.30)
        + gaussian_blur(bright, 40.0) * F32(0.12)
    )
    return np.maximum(glow, F32(0.0)).astype(F32)


def resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resize to exactly w x h (single fetch per output texel)."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    uu, vv = np.meshgrid(
        (np.arange(w, dtype=F32) + F32(0.5)) / F32(w),
        (np.arange(h, dtype=F32) + F32(0.5)) / F32(h),
    )
    return bilinear(img, uu, vv)


def hot_source(scene: np.ndarray, w: int, h: int) -> np.ndarray:
    """The hot-source dispatch: box downsample of the exposed image, threshold, soft ceiling.

    The box is 8x8 bilinear taps two display pixels apart, i.e. a 16-pixel footprint per output texel,
    so a sun smaller than one texel still contributes its whole energy. Here that footprint is
    approximated by a sigma-4 gaussian plus an area downsample, which has the same energy response.

    The soft ceiling is `v / (1 + v / ceiling)`: strictly increasing in v, so a brighter source still
    produces a brighter spike, but a sun millions of times over white cannot dwarf a lamp by the same
    factor. Without it, an exposed sun would make every flare an opaque white star.
    """
    average = resize_to(gaussian_blur(scene, math.sqrt((SHADER_HOT_BOX_TAPS * SHADER_HOT_BOX_STEP) ** 2 / 12.0)), w, h)
    lum = luminance(average)
    above = np.maximum(lum - SHADER_HOT_THRESHOLD, F32(0.0))
    scale = above / np.maximum(lum, F32(1e-6))
    value = average * scale[..., None]
    return (value / (F32(1.0) + value / SHADER_HOT_CEILING)).astype(F32)


# --- the three stages -------------------------------------------------------------------------------
def texel_dither(w: int, h: int, seed: int) -> np.ndarray:
    """Per-texel tap phase in [0,1): the interleave that keeps a sparse gather from beading."""
    ii, jj = np.meshgrid(np.arange(w, dtype=np.int64), np.arange(h, dtype=np.int64))
    return hash2(ii, jj, np.full_like(ii, np.uint32(seed)))


def starburst(hot: np.ndarray, spikes: int, samples: int, length: float,
              shape: tuple[int, int] | None = None) -> np.ndarray:
    """Ray integral through the hot field: `samples` taps along each of `spikes` directions.

    Taps march outward at a uniform spacing, so the source's own width bridges the gaps between them, and
    carry an exponential envelope so a spike fades towards its tip instead of staying a flat wedge. Each
    texel slides its tap chain by its own random phase, which interleaves neighbouring texels: the display
    pass then interpolates them into a continuous ray instead of a dotted chain. Normalising by the weight
    sum keeps the overall energy independent of the sample count, so the quality setting trades smoothness
    for cost and nothing else.
    """
    h, w = shape if shape is not None else hot.shape[:2]
    u, v = pixel_grid(w, h)
    aspect = F32(w) / F32(h)
    dither = texel_dither(w, h, 0x1F35)
    steps = np.arange(samples, dtype=F32)[None, None, :]
    fractions = (steps + dither[..., None]) / F32(samples)
    weights = np.exp(-float(SHADER_SPIKE_FALLOFF) * fractions)
    weight_sum = np.maximum(weights.sum(axis=-1), F32(1e-6))
    acc = np.zeros((h, w, 3), dtype=F32)
    for k in range(spikes):
        angle = math.pi * k / spikes
        dx = F32(math.cos(angle)) / aspect
        dy = F32(math.sin(angle))
        for i in range(samples):
            f = fractions[..., i]
            du = dx * f * F32(length)
            dv = dy * f * F32(length)
            # Both signs of the offset: a gather only carries light from the side it reads, so a single
            # direction would light one half of a spike line and leave a one-armed star.
            acc += (bilinear(hot, u + du, v + dv) + bilinear(hot, u - du, v - dv)) * weights[..., i][..., None]
    return (acc / weight_sum[..., None] * F32(spikes) * SHADER_SPIKE_GAIN).astype(F32)


def halo(hot: np.ndarray, taps: int, radius: float, dispersion: float,
         shape: tuple[int, int] | None = None) -> np.ndarray:
    """Aperture ring of `radius` around each hot core, one gather per channel at its own radius.

    Gathering at radius `radius * (1 + dispersion * chroma)` is the ring a defocused aperture makes, with
    lateral chromatic aberration splitting it: red outside, blue inside. The tap phases rotate per texel, so
    a budget of a few dozen taps resolves into a continuous ring rather than a string of beads.
    """
    h, w = shape if shape is not None else hot.shape[:2]
    u, v = pixel_grid(w, h)
    aspect = F32(w) / F32(h)
    dither = texel_dither(w, h, 0x2B71)
    out = np.zeros((h, w, 3), dtype=F32)
    for channel in range(3):
        r = F32(radius) * (F32(1.0) + SHADER_HALO_CHROMA[channel] * F32(dispersion))
        phase = dither + F32(HALO_PHASE[channel])
        acc = np.zeros((h, w), dtype=F32)
        for i in range(taps * SHADER_HALO_SUBTAPS):
            angle = F32(2.0 * math.pi) * (F32(i) + phase) / F32(taps * SHADER_HALO_SUBTAPS)
            acc += bilinear(
                hot[..., channel],
                u + np.cos(angle) / aspect * r,
                v + np.sin(angle) * r * SHADER_HALO_ELLIPSE,
            )
        out[..., channel] = acc / F32(taps * SHADER_HALO_SUBTAPS)
    return (out * SHADER_HALO_GAIN).astype(F32)


def hash2(ix: np.ndarray, iy: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Dave Hoskins-style integer hash of a lattice point, as a float in [0,1)."""
    x = (ix.astype(np.uint32) * np.uint32(0x9E3779B1)) ^ (iy.astype(np.uint32) * np.uint32(0x85EBCA77))
    x ^= seed.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return (x >> np.uint32(8)).astype(F32) * F32(1.0 / 16777216.0)


def value_noise(pu: np.ndarray, pv: np.ndarray, frequency: F32, seed: int) -> np.ndarray:
    """Smooth lattice noise in [0,1): bilinear blend of hashed corners, i.e. the smudges between specks."""
    qu = pu * frequency
    qv = pv * frequency
    x0 = np.floor(qu)
    y0 = np.floor(qv)
    fx = qu - x0
    fy = qv - y0
    fx = fx * fx * (F32(3.0) - F32(2.0) * fx)
    fy = fy * fy * (F32(3.0) - F32(2.0) * fy)
    seed_arr = np.full_like(x0, np.uint32(seed))
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    v00 = hash2(x0, y0, seed_arr)
    v10 = hash2(x0 + 1, y0, seed_arr)
    v01 = hash2(x0, y0 + 1, seed_arr)
    v11 = hash2(x0 + 1, y0 + 1, seed_arr)
    top = v00 + (v10 - v00) * fx
    bottom = v01 + (v11 - v01) * fx
    return (top + (bottom - top) * fy).astype(F32)


def dust_layer(pu: np.ndarray, pv: np.ndarray, frequency: F32, seed: int) -> np.ndarray:
    """One octave of out-of-focus specks: a jittered lattice of soft disks, some cells left empty.

    Four lattice cells are tested per texel (the 2x2 block containing it); each cell owns at most one
    speck whose position, radius and brightness come from its hash, so the pattern is stable and needs no
    texture. Specks are soft disks with a faint rim, which is what a defocused dust grain looks like: a
    broad dim body with a brighter edge where the grain refracts, never a hard circle.
    """
    qu = pu * frequency
    qv = pv * frequency
    base_u = np.floor(qu)
    base_v = np.floor(qv)
    seed_arr = np.full_like(base_u, np.uint32(seed)).astype(np.uint32)
    ii = base_u.astype(np.int64)
    jj = base_v.astype(np.int64)
    field = np.zeros_like(qu, dtype=F32)
    for du in (0, 1):
        for dv in (0, 1):
            cu = ii + du
            cv = jj + dv
            hu = hash2(cu, cv, seed_arr)
            hv = hash2(cv, cu, seed_arr + np.uint32(0x68BC21EB))
            hr = hash2(cu + 0x01234567, cv + 0x07654321, seed_arr + np.uint32(0x02E5BE93))
            hk = hash2(cv + 0x0BADF00D, cu + 0x00C0FFEE, seed_arr + np.uint32(0x5F356495))
            center_u = cu.astype(F32) + SHADER_DIRT_CELL_MARGIN + SHADER_DIRT_CELL_SPREAD * hu
            center_v = cv.astype(F32) + SHADER_DIRT_CELL_MARGIN + SHADER_DIRT_CELL_SPREAD * hv
            radius = SHADER_DIRT_SPECK_MIN_RADIUS + SHADER_DIRT_SPECK_RADIUS_SPREAD * hr
            bright = np.clip((hk - SHADER_DIRT_EMPTY) / (F32(1.0) - SHADER_DIRT_EMPTY), 0.0, 1.0)
            dist = np.sqrt((qu - center_u) ** 2 + (qv - center_v) ** 2) / radius
            body = np.clip(F32(1.0) - dist, 0.0, 1.0)
            body = body * body * (F32(3.0) - F32(2.0) * body)
            rim = np.exp(-((dist - SHADER_DIRT_RIM_CENTER) ** 2) / SHADER_DIRT_RIM_WIDTH)
            field += bright * (body * SHADER_DIRT_SPECK_BODY + rim * SHADER_DIRT_SPECK_RIM)
    return np.clip(field, 0.0, 1.0).astype(F32)


def lens_dirt(glow: np.ndarray, scale: float) -> np.ndarray:
    """Dust field times the light falling on it: dirt only shows where the front element is lit.

    Frequencies are per texture height, not per texel, so the grain keeps its size in the frame at every
    resolution. The light is soft-ceilinged: without that, a 9000-unit sun turns the grain into opaque
    white blobs instead of a faint veil, because the glow under it is three orders of magnitude above 1.
    """
    h, w = glow.shape[:2]
    u, v = pixel_grid(w, h)
    aspect = F32(w) / F32(h)
    pu = u * aspect
    pv = v
    field = np.zeros((h, w), dtype=F32)
    for frequency, weight, seed in zip(SHADER_DIRT_FREQ, SHADER_DIRT_WEIGHT, SHADER_DIRT_SEED):
        field += dust_layer(pu, pv, frequency * F32(scale), seed) * weight
    grime = value_noise(pu, pv, SHADER_DIRT_SMUDGE_FREQ * F32(scale), SHADER_DIRT_SMUDGE_SEED)
    field = np.clip(field, 0.0, 1.0) * (SHADER_DIRT_GRIME_FLOOR + SHADER_DIRT_GRIME_RANGE * grime)
    light = glow / (F32(1.0) + glow / SHADER_DIRT_LIGHT_CEILING)
    return (light * field[..., None] * SHADER_DIRT_GAIN).astype(F32)


# --- preview plumbing -------------------------------------------------------------------------------
def tonemap_for_preview(hdr: np.ndarray) -> np.ndarray:
    """ACES-approx filmic curve plus sRGB encode: what the display pass would show, near enough."""
    x = np.maximum(hdr, F32(0.0))
    a, b, c, d, e = F32(2.51), F32(0.03), F32(2.43), F32(0.59), F32(0.14)
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    srgb = np.where(
        mapped <= F32(0.0031308), mapped * F32(12.92), F32(1.055) * mapped ** (F32(1.0) / F32(2.4)) - F32(0.055)
    )
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def gray_panel(img: np.ndarray, w: int, h: int) -> np.ndarray:
    lum = resize_to(luminance(img)[..., None], w, h)
    lum = np.repeat(lum, 3, axis=-1)
    peak = float(lum.max())
    scaled = lum / peak if peak > 1e-6 else lum
    return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(img)


def sheet(panels: list[np.ndarray], columns: int) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    tile_h, tile_w = panels[0].shape[:2]
    out = Image.new("RGB", (tile_w * columns, tile_h * rows), (0, 0, 0))
    for index, panel in enumerate(panels):
        out.paste(Image.fromarray(panel), ((index % columns) * tile_w, (index // columns) * tile_h))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=960, help="display width of the synthetic frame")
    parser.add_argument("--height", type=int, default=540, help="display height of the synthetic frame")
    parser.add_argument("--out", default="tools/lens-preview-out", help="output directory")
    parser.add_argument("--starburst", type=float, default=1.0)
    parser.add_argument("--spikes", type=int, default=6)
    parser.add_argument("--spike-length", type=float, default=0.14, help="fraction of the screen height")
    parser.add_argument("--samples", type=int, default=12, help="taps per spike (lens.samples)")
    parser.add_argument("--halo", type=float, default=1.0)
    parser.add_argument("--halo-radius", type=float, default=0.14, help="fraction of the screen height")
    parser.add_argument("--halo-taps", type=int, default=DEFAULT_HALO_TAPS, help="taps per ring, per channel")
    parser.add_argument("--dispersion", type=float, default=0.06)
    parser.add_argument("--dirt", type=float, default=1.0)
    parser.add_argument("--divisor", type=int, default=8, help="lens texture size = display size / divisor")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="multiplier on both tap counts (lens.sample-rate)")
    parser.add_argument("--dirt-scale", type=float, default=1.0)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    scene = synthetic_scene(args.width, args.height)
    glow = bloom_glow(scene)
    t1 = time.perf_counter()

    # The flare image is the display size divided by the divisor; the source field keeps its own eighth,
    # because its coarseness is what leaves the rays thin instead of turning each source disc into a fan.
    fw, fh = max(1, args.width // args.divisor), max(1, args.height // args.divisor)
    sw, sh = max(1, args.width // SOURCE_DIVISOR), max(1, args.height // SOURCE_DIVISOR)
    hot = hot_source(scene, sw, sh)
    glow_small = resize_to(glow, fw, fh)

    # The sample rate scales both gather densities, mirroring the clamps in RtComposite.lensFlarePushData.
    samples = max(1, round(args.samples * args.sample_rate))
    halo_taps = max(8, round(args.halo_taps * args.sample_rate))

    star = starburst(hot, args.spikes, samples, args.spike_length, shape=(fh, fw))
    t2 = time.perf_counter()
    ring = halo(hot, halo_taps, args.halo_radius, args.dispersion, shape=(fh, fw))
    t3 = time.perf_counter()
    dirt = lens_dirt(glow_small, args.dirt_scale)
    t4 = time.perf_counter()

    flare = star * F32(args.starburst) + ring * F32(args.halo) + dirt * F32(args.dirt)
    # The display pass reads this texture at full resolution; upsample the same way so the sheet shows
    # the softness the game will actually present.
    flare_full = resize_to(flare, args.width, args.height)

    # Each spike tap is read on both sides of the ray, so one tap is two fetches.
    taps_star = 2 * args.spikes * samples
    taps_halo = 3 * halo_taps * SHADER_HALO_SUBTAPS
    taps_source = SHADER_HOT_BOX_TAPS * SHADER_HOT_BOX_TAPS
    source_texels = sw * sh
    texels = fw * fh
    print(f"frame {args.width}x{args.height} -> source field {sw}x{sh}, flare image {fw}x{fh}")
    print(f"  hot source: {taps_source} taps/texel  {taps_source * source_texels / 1e6:.2f} M fetches")
    print(f"  starburst : {taps_star} taps/texel  {taps_star * texels / 1e6:.2f} M fetches")
    print(f"  halo      : {taps_halo} taps/texel  {taps_halo * texels / 1e6:.2f} M fetches")
    print(f"  dirt      : 1 glow fetch + {4 * len(SHADER_DIRT_FREQ)} procedural lattice reads per texel")
    total = taps_source * source_texels + (taps_star + taps_halo) * texels
    print(f"  total     : {total / 1e6:.2f} M texture fetches, "
          f"{total / float(args.width * args.height):.2f} per display pixel")
    print(f"  numpy timings: source {t1 - t0:.2f}s, star {t2 - t1:.2f}s, halo {t3 - t2:.2f}s, dirt {t4 - t3:.2f}s")

    panels = [
        label(tonemap_for_preview(scene), "input (ACES approx)"),
        label(gray_panel(hot, args.width, args.height), "hot source (normalised)"),
        label(tonemap_for_preview(flare_full), "flare total"),
        label(tonemap_for_preview(scene + flare_full), "composited (scene + flare)"),
        label(tonemap_for_preview(resize_to(star, args.width, args.height) * args.starburst), "starburst"),
        label(tonemap_for_preview(resize_to(ring, args.width, args.height) * args.halo), "halos"),
        label(tonemap_for_preview(resize_to(dirt, args.width, args.height) * args.dirt), "lens dirt"),
        label(tonemap_for_preview(resize_to(glow_small, args.width, args.height)), "bloom glow (light source)"),
    ]
    if not args.no_write:
        path = os.path.join(args.out, "lens-sheet.png")
        sheet(panels, 2).save(path)
        Image.fromarray(tonemap_for_preview(scene + flare_full)).save(
            os.path.join(args.out, "lens-composited.png")
        )
        print(f"wrote {path} and lens-composited.png")


if __name__ == "__main__":
    main()
