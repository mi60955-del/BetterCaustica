"""Offline verifier for the parallax self-shadow (parallaxShadow in closest_hit.rchit.slang).

Ports the hit shader's parallax march (sampleParallax DDA) and its self-shadow walk
(parallaxShadow) line for line to Python, feeds them the real LabPBR height field from a
_n texture, and measures where the walk still produces a visible darkening.

The GPU sampler for the material atlas is NEAREST + REPEAT (RtComposite.atlasSampler), so
every field read below is a nearest-texel fetch: the staircase the march walks on is the
same staircase the shadow walk sees.

Run:
    py -3.13 tools/verify_parallax_shadow.py [--out build/parallax-shadow-verify]

Outputs a sweep table over face-space sun elevation, plus per-elevation shadow-map renders
and a span_scale A/B (1.0 = the shader as written).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image

# ---- config mirror (E:\Minecraft\26.2\config\bettercaustica.toml, [materials]/[composite]) ----
PARALLAX_DEPTH = 0.27       # composite.parallax-depth -> pc.parallaxDepth -> reach scale
PARALLAX_LAYERS = 8         # materials.parallax-layers -> pc.parallaxLayers (sharp style, lod 0)
PARALLAX_SHADOW_STEPS = 10
PARALLAX_SHADOW_MIN_ELEVATION = 0.10
PARALLAX_SHADOW_SOFTNESS = 0.15
PARALLAX_SHADOW_BIAS = 0.02
PARALLAX_MIN_VIEW_Z = 0.2


def load_field(path: str) -> np.ndarray:
    """LabPBR height field: field = 1 - alpha/255, field 0 = flush, 1 = quarter block deep."""
    im = np.array(Image.open(path).convert("RGBA"), dtype=np.float64)
    return 1.0 - im[:, :, 3] / 255.0


def nearest(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-texel REPEAT fetch in sprite-local [0,1)^2, matching the atlas sampler."""
    h, w = field.shape
    x = np.floor(uv[..., 0] * w).astype(int) % w
    y = np.floor(uv[..., 1] * h).astype(int) % h
    return field[y, x]


def column_depth(field: np.ndarray, texel: np.ndarray) -> np.ndarray:
    """The march's per-column fetch: texel centres, wrapped (closest_hit.rchit.slang:282)."""
    grid = np.array([field.shape[1], field.shape[0]], dtype=np.float64)
    uv = (texel + 0.5) / grid
    return nearest(field, uv)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_parallax(field: np.ndarray, local_uv: np.ndarray, view_slope_xy, view_vz: float,
                    reach: float, span_scale: float = 1.0, fix_wall: bool = True):
    """Port of sampleParallax (sharp style, lod 0, pc.parallaxLayers layers).

    fix_wall=False reproduces the shader as written: on a wall hit hit_depth keeps the
    START column's depth, while the shaded UV is the wall column's centre — the walk then
    starts buried under its own wall and darkens wall bands at every sun elevation.
    fix_wall=True starts the walk at the wall's ceiling (the intent the shader comment
    states: "climbs from the ceiling, the top of that face").
    Returns (displaced_uv, hit_depth, landed, moved, wall_step)."""
    grid = np.array([field.shape[1], field.shape[0]], dtype=np.float64)
    slope = np.asarray(view_slope_xy, dtype=np.float64) / max(abs(view_vz), PARALLAX_MIN_VIEW_Z)
    texel_vel = -slope * reach * grid * span_scale
    entry = local_uv * grid
    texel = np.floor(entry)
    f = column_depth(field, texel)

    shape = local_uv.shape[:-1]
    t = np.zeros(shape)
    hit_depth = f.copy()
    prev_field = f.copy()
    wall_step = np.zeros(shape)          # |field - prevField| where a wall was struck
    move_last = np.zeros((shape[0], shape[1], 2))
    live = f > 0.0                       # flush entry: the ray starts on the surface
    landed = np.zeros(shape, dtype=bool)
    wall = np.zeros(shape, dtype=bool)

    sdir = np.sign(texel_vel)
    ardir = 1.0 / np.maximum(np.abs(texel_vel), 1e-9)
    frac = entry - texel
    boundary = (sdir * (0.5 - frac) + 0.5) * ardir

    for _ in range(PARALLAX_LAYERS):
        # wall hit: the ray entered this column already below its ceiling (field <= t)
        w = live & (f <= t)
        wall |= w
        wall_step = np.where(w, np.abs(f - prev_field), wall_step)
        hit_depth = np.where(w & fix_wall, f, hit_depth)
        landed |= w
        live &= ~w
        # flat landing: the ceiling arrives while the ray is still over this column
        b = np.minimum(boundary[..., 0], boundary[..., 1])
        flat = live & (f <= b)
        hit_depth = np.where(flat, f, hit_depth)
        landed |= flat
        live &= ~flat
        if not live.any():
            break
        # step across the nearer boundary
        step = live
        x_first = boundary[..., 0] <= boundary[..., 1]
        mx = np.where(x_first, sdir[..., 0], 0.0)
        my = np.where(x_first, 0.0, sdir[..., 1])
        boundary[..., 0] = np.where(step & x_first, boundary[..., 0] + ardir[..., 0], boundary[..., 0])
        boundary[..., 1] = np.where(step & ~x_first, boundary[..., 1] + ardir[..., 1], boundary[..., 1])
        t = np.where(step, b, t)
        move_last = np.where(step[..., None], np.stack([mx, my], axis=-1), move_last)
        texel = np.where(step[..., None], texel + np.stack([mx, my], axis=-1), texel)
        prev_field = np.where(step, f, prev_field)
        f = np.where(step, column_depth(field, texel), f)

    moved = landed & (np.abs(move_last).sum(axis=-1) > 0)
    grid_uv = (texel + 0.5) / grid
    displaced = grid_uv - np.floor(grid_uv)
    displaced = np.where(moved[..., None], displaced, local_uv)
    depth_out = np.where(landed, hit_depth, 0.0)
    return displaced, depth_out, landed, moved, wall_step


def parallax_shadow(field: np.ndarray, uv: np.ndarray, hit_depth: np.ndarray,
                    elev: float, azimuth: float, reach: float,
                    stretch: float | None = None, steps: int = PARALLAX_SHADOW_STEPS) -> np.ndarray:
    """Port of parallaxShadow (closest_hit.rchit.slang:248).

    stretch=None  -> the shader as written: lateral speed = cot * reach (UV per unit climb).
    stretch=S     -> the decoupled walk: one unit of stored depth sweeps S texels laterally
                     at a 45-degree light, independent of the march's physical reach."""
    elev_sin = math.sin(elev)
    active = hit_depth > 1.0e-4
    if elev_sin <= PARALLAX_SHADOW_MIN_ELEVATION or not active.any():
        return np.ones(uv.shape[:-1])
    ct = math.cos(elev) / elev_sin
    lat = np.array([math.cos(azimuth), math.sin(azimuth)]) * ct
    if stretch is None:
        lat = lat * reach
    else:
        lat = lat * (stretch / field.shape[0])
    step_depth = hit_depth / steps
    shadow = np.ones(uv.shape[:-1])
    for i in range(1, steps + 1):
        climbed = step_depth * i
        suv = uv + lat[None, None, :] * climbed[..., None]
        suv = suv - np.floor(suv)
        f = nearest(field, suv)
        blocked = (hit_depth - climbed - f - PARALLAX_SHADOW_BIAS) / PARALLAX_SHADOW_SOFTNESS
        shadow = np.minimum(shadow, 1.0 - np.clip(blocked, 0.0, 1.0))
    ramp = smoothstep(PARALLAX_SHADOW_MIN_ELEVATION, PARALLAX_SHADOW_MIN_ELEVATION * 2.0, elev_sin)
    return np.where(active, 1.0 + (shadow - 1.0) * ramp, 1.0)


def render(field: np.ndarray, view_slope_xy, view_vz: float, elev: float, azimuth: float,
           reach: float, res: int, stretch: float | None = None, steps: int = PARALLAX_SHADOW_STEPS,
           face_on_shadow: bool = False, fix_wall: bool = True):
    us = (np.arange(res) + 0.5) / res
    uu, vv = np.meshgrid(us, us)
    local = np.stack([uu, vv], axis=-1)
    displaced, depth, landed, moved, _ = sample_parallax(field, local, view_slope_xy, view_vz,
                                                         reach, fix_wall=fix_wall)
    # Where the march never left its starting column the shader returns the flat mapping;
    # the decoupled-shadow variant keeps that UV contract but still lets the column's own
    # depth cast a contact shadow (face-on groove bottoms).
    uv = displaced if face_on_shadow else np.where(moved[..., None], displaced, local)
    depth = depth if face_on_shadow else np.where(moved, depth, 0.0)
    shadow = parallax_shadow(field, uv, depth, elev, azimuth, reach, stretch, steps)
    return shadow, depth, moved


def sweep(field: np.ndarray, args, view_slope_xy, view_vz: float, azimuth: float,
          stretch: float | None, steps: int, face_on_shadow: bool, elevations,
          tag_suffix: str = "", fix_wall: bool = True) -> None:
    """stretch=None reproduces the written walk (rides the march's reach)."""
    label = "reach-coupled (as written)" if stretch is None else f"stretch={stretch:g} texels"
    print(f"--- {label}  steps={steps}  face-on shadow={face_on_shadow} ---")
    header = (f"{'elev':>5} {'displaced':>9} {'shadowed':>9} {'dark>10%':>9} {'dark>50%':>9} "
              f"{'min':>6} {'mean':>6} {'walk span px':>12}")
    print(header)
    for elev_deg in elevations:
        elev = math.radians(elev_deg)
        shadow, depth, moved = render(field, view_slope_xy, view_vz, elev, azimuth,
                                      args.reach, args.res, stretch, steps, face_on_shadow,
                                      fix_wall)
        dark = 1.0 - shadow
        shadowed = depth > 1.0e-4
        dmax = depth.max()
        per_unit = (args.reach if stretch is None else stretch / field.shape[0]) \
            * field.shape[0]
        span = (math.cos(elev) / math.sin(elev)) * per_unit * (dmax if dmax > 0 else 0)
        print(f"{elev_deg:>4d}° {moved.mean():>9.1%} {shadowed.mean():>9.1%} "
              f"{(dark > 0.1).mean():>9.2%} {(dark > 0.5).mean():>9.2%} "
              f"{shadow.min():>6.3f} {shadow.mean():>6.3f} {span:>12.2f}")
        tag = f"elev{elev_deg:02d}"
        tag += "_cur" if stretch is None else f"_st{stretch:g}"
        if steps != PARALLAX_SHADOW_STEPS:
            tag += f"_n{steps}"
        tag += tag_suffix
        Image.fromarray((np.clip(shadow, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"shadow_{tag}.png"))
        if dmax > 0:
            Image.fromarray((depth / dmax * 255).astype(np.uint8)).save(
                os.path.join(args.out, f"hitdepth_{tag}.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texture", default=os.path.join(os.path.dirname(__file__), "sandstone_n.png"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "build",
                                                  "parallax-shadow-verify"))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--reach", type=float, default=PARALLAX_DEPTH)
    args = ap.parse_args()

    field = load_field(args.texture)
    os.makedirs(args.out, exist_ok=True)
    Image.fromarray(((1.0 - field) * 255).astype(np.uint8)).save(
        os.path.join(args.out, "alpha_input.png"))
    Image.fromarray((field * 255).astype(np.uint8)).save(
        os.path.join(args.out, "height_field.png"))

    print(f"height field {field.shape[1]}x{field.shape[0]}  depth min/max/mean: "
          f"{field.min():.3f}/{field.max():.3f}/{field.mean():.3f}")
    print(f"reach = {args.reach}  layers = {PARALLAX_LAYERS}")
    print()

    # A vertical face seen 45° obliquely along +u; the sun rakes across the strata (azimuth 0).
    oblique = ((1.0, 0.0), 1.0)
    azimuth = 0.0
    elevations = (75, 60, 45, 30, 20)

    sweep(field, args, *oblique, azimuth, None, PARALLAX_SHADOW_STEPS, False, elevations,
          fix_wall=False)
    print()
    for stretch in (8.0, 12.0, 16.0, 24.0):
        sweep(field, args, *oblique, azimuth, stretch, PARALLAX_SHADOW_STEPS, True, elevations)
        print()
    # Step-count sensitivity at the shipping candidate (16): the span at hitDepth = 1 is a
    # full sprite width, so 16 steps keep the per-step stride under one texel.
    sweep(field, args, *oblique, azimuth, 16.0, 16, True, elevations)
    print()
    # Face-on view (the common case in a screenshot): the march never moves, so the written
    # shader returns shadow = 1 for every pixel. The decoupled variant shades the column's
    # own depth instead.
    face_on = ((0.0, 0.0), 1.0)
    sweep(field, args, *face_on, azimuth, None, PARALLAX_SHADOW_STEPS, False, (45, 20))
    sweep(field, args, *face_on, azimuth, 16.0, PARALLAX_SHADOW_STEPS, True, (45, 20),
          tag_suffix="_faceon")
    print()
    print("shadow maps written to", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
