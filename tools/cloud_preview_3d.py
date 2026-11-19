"""3D offscreen raymarch of the low deck — the shape jury.

A numpy raymarcher that walks the SAME density field cloud_preview.py mirrors (which mirrors
volume.slang), integrated the same way the shader's volumetric tier does: per-pixel slab span,
fixed step split, front-to-back transmittance, Beer-powder lighting stand-in. What the two 2D
previews cannot show — whether a body is stretched by the deck thickness or truncated by the lid,
whether tops dome or cut flat, whether walls stand vertical — this shows directly, from any camera.

Exists because the flat-lid problem was invisible in plan and cross views and obvious from a
flying camera. Constants and field math live in cloud_preview (import, don't fork); the variant
modes below are candidate SHAPE changes, validated here before they touch the shader.

Modes (pure body-scale A/B, one deployed field math under all of them):
  s5000   — the deployed 5000/700 tiles: compact bodies.
  s8000   — 8000/1120: the same shape machinery with the noise enlarged ~1.6x.
  s12000  — 12000/1680: bodies ~2.4x the deployed width.
The renders decide; the winner goes into volume.slang's layer tables.

Run:
    ./.venv/Scripts/python.exe tools/cloud_preview_3d.py [--w 380 --h 220]
Output: tools/cloud-preview-out/3d_<view>.png, one strip of variant tiles per view.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

import cloud_preview as cp

RES = 380
HEIGHT_RANGE = 0.55
CROWN_MIN = 0.25
CROWN_POWER = 0.8
CROWN_REF = 0.35
FINE_CURL = 0.06

MODES = ["s5000", "s8000", "s12000"]
VIEWS = ["above", "side", "below"]

# The modes are a pure body-scale A/B: the deployed field math (cloud_preview.cumulus) is one, and
# each mode only re-patches the low deck's tiles. 5000/700 is the deployed table; 8000 and 12000
# exist because "enlarge the noise" is a look dial with no analytic answer — the renders decide.
SCALES = {
    "s5000": (5000.0, 700.0),
    "s8000": (8000.0, 1120.0),
    "s12000": (12000.0, 1680.0),
}

SUN = None  # set in main: normalized (x, y, z)
DIRECT = 6.0
SKY_AMB = 1.35
ALBEDO = np.array([0.86, 0.87, 0.90], np.float32)


# ---- field -----------------------------------------------------------------------------------------

def _frac(v):
    return v - np.floor(v)


def _grad(px, py):
    """Mirror of cloudGradient: one 2D unit gradient per lattice point, scale-free."""
    p3x = _frac(px * 0.1031)
    p3y = _frac(py * 0.1030)
    p3z = _frac(px * 0.0973)
    dot = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33)
    p3x += dot
    p3y += dot
    p3z += dot
    hx = _frac((p3x + p3y) * p3z)
    hy = _frac((p3x + p3z) * p3y)
    nx = hx - 0.5 + 1.0e-4
    ny = hy - 0.5 + 1.0e-4
    n = np.sqrt(nx * nx + ny * ny)
    return nx / n, ny / n


def _curl(px, py):
    """Mirror of cloudCurl: divergence-free swirl of the gradient field, magnitude O(1)."""
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    dux = 30.0 * fx * fx * (fx * (fx - 2.0) + 1.0)
    duy = 30.0 * fy * fy * (fy * (fy - 2.0) + 1.0)
    g00x, g00y = _grad(ix, iy)
    g10x, g10y = _grad(ix + 1.0, iy)
    g01x, g01y = _grad(ix, iy + 1.0)
    g11x, g11y = _grad(ix + 1.0, iy + 1.0)
    v00 = g00x * fx + g00y * fy
    v10 = g10x * (fx - 1.0) + g10y * fy
    v01 = g01x * fx + g01y * (fy - 1.0)
    v11 = g11x * (fx - 1.0) + g11y * (fy - 1.0)
    vx = (v10 - v00) * (1.0 - uy) + (v11 - v01) * uy
    vy = (v01 - v00) * (1.0 - ux) + (v11 - v10) * ux
    return vx * dux, vy * duy


def density(x, z, y, cov, patch, mode, eroded=True):
    """Low-deck density 0..1 at one tile scale: the deployed math, re-patched tiles per mode."""
    saved = (cp.FIELD_TILE, cp.DETAIL_TILE)
    cp.FIELD_TILE, cp.DETAIL_TILE = SCALES.get(mode, SCALES["s5000"])
    try:
        return cp.cumulus(x, z, y, cov, patch, eroded=eroded)
    finally:
        cp.FIELD_TILE, cp.DETAIL_TILE = saved


# ---- march -----------------------------------------------------------------------------------------

def _hg(cos_theta, g):
    g2 = g * g
    return (1.0 - g2) / (4.0 * math.pi * (1.0 + g2 - 2.0 * g * cos_theta) ** 1.5)


def _phase(cos_theta, powder):
    a = _hg(cos_theta, 0.80) * (1 - powder) + _hg(cos_theta, 0.55) * powder
    b = _hg(cos_theta, 0.35) * (1 - powder) + _hg(cos_theta, 0.10) * powder
    return (a * (1 - powder) + b * powder) + powder * 2.0 * _hg(-cos_theta, 0.25)


def render(cam, pitch, yaw, cov, patch, mode, w, h, steps=96, fov_deg=70.0):
    sun = SUN
    fov = math.radians(fov_deg)
    cp_, sp_ = math.cos(pitch), math.sin(pitch)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    fwd = np.array([cp_ * cy_, sp_, cp_ * sy_], np.float32)
    right = np.array([sy_, 0.0, -cy_], np.float32)
    up = np.cross(right, fwd)
    u = ((np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0) * math.tan(fov / 2)
    v = ((np.arange(h, dtype=np.float32) + 0.5) / h * 2.0 - 1.0) * math.tan(fov / 2)
    uu = np.repeat(u[None, :], h, axis=0)
    vv = np.repeat(v[:, None], w, axis=1)
    aspect = w / h
    rd = fwd[None, None, :] + right[None, None, :] * uu[..., None] \
        + up[None, None, :] * vv[..., None] * aspect
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.array(cam, np.float32)
    dx, dy, dz = rd[..., 0], rd[..., 1], rd[..., 2]

    base, top = cp.BASE - 2.0, cp.TOP + 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (base - ro[1]) / np.where(np.abs(dy) < 1.0e-5, 1.0e-5, dy)
        tb = (top - ro[1]) / np.where(np.abs(dy) < 1.0e-5, 1.0e-5, dy)
    t0 = np.clip(np.minimum(ta, tb), 0.0, None)
    t1 = np.maximum(np.minimum(np.maximum(ta, tb), 40000.0), 0.0)
    span = t1 - t0
    hit = span > 1.0

    tallness = cp.TALLNESS
    dt = np.where(hit, span / steps, 0.0).astype(np.float32)
    t_start = (t0 + 0.5 * dt).astype(np.float32)
    px = ro[0] + dx * t_start
    py = ro[1] + dy * t_start
    pz = ro[2] + dz * t_start

    L = np.zeros((h, w, 3), np.float32)
    T = np.ones((h, w), np.float32)
    cos_theta = dx * sun[0] + dy * sun[1] + dz * sun[2]

    for _ in range(steps):
        active = hit & (T > 0.01)
        if not active.any():
            break
        dens = np.where(active, density(px, pz, py, cov, patch, mode), 0.0)
        faded = dens * np.clip((py - cp.BASE) / tallness * 4.0, 0.0, 1.0) ** 2
        tau_cell = faded * cp.LOW_DENSITY * dt
        # Analytic sun leg stand-in: the column of cloud above the sample toward the deck's top.
        # Free, and it keeps crowns bright and bases dark, which is all the shape jury needs.
        tau_sun = dens * np.maximum(cp.TOP - py, 0.0) * cp.LOW_DENSITY / max(sun[1], 0.3)
        sun_mask = 1.0 - np.exp(-tau_sun / 8.0)
        beer = 6.28 * np.exp(-5.0 * sun_mask)
        powder = np.minimum(np.exp(-4.0 * np.exp(-4.0 * dens)) * 3.5, 1.0)
        height01 = np.clip((py - cp.BASE) / tallness, 0.0, 1.0)
        amb = np.exp2(-5.0 * np.clip(dens, 0, 1))
        w_occl = height01 * height01 * 0.45
        occl = (1.0 - w_occl) + amb * w_occl
        amb_occlusion = 0.15 + 0.85 * occl
        rad = ALBEDO[None, None, :] * (
            beer[..., None] * DIRECT * _phase(cos_theta, powder)[..., None]
            + SKY_AMB * amb_occlusion[..., None])
        absorbed = (1.0 - np.exp(-tau_cell)) * np.where(active, T, 0.0)
        L += (T * absorbed)[..., None] * rad
        T *= 1.0 - absorbed
        px += dx * dt
        py += dy * dt
        pz += dz * dt

    # Sky: simple gradient; below-horizon a flat haze so cloud bases read against it.
    sky = np.empty((h, w, 3), np.float32)
    horizon = np.clip(dy, -1.0, 1.0)
    sky[..., 0] = 0.36 + 0.26 * np.clip(horizon, 0, 1)
    sky[..., 1] = 0.52 + 0.24 * np.clip(horizon, 0, 1)
    sky[..., 2] = 0.82 - 0.10 * horizon
    below = horizon < 0.0
    sky[below] = np.array([0.55, 0.62, 0.72], np.float32) * (1.0 + 0.4 * horizon[below])[:, None]
    out = L + T[..., None] * sky
    out = 1.0 - np.exp(-out * 0.9)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def sheet(items, out, cols, w, h, pad=6, bar=20):
    rows = (len(items) + cols - 1) // cols
    img = Image.new("RGB", (cols * w + (cols + 1) * pad,
                            rows * (h + bar) + (rows + 1) * pad), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    for i, (label, arr) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + bar + pad)
        img.paste(Image.fromarray(arr), (x, y + bar))
        draw.text((x + 2, y + 4), label, fill=(230, 230, 230))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print("wrote", out, img.size)


PRESETS = {
    # cam (x, y, z), pitch, yaw — above is the flying-camera situation, side is the ground one.
    "above": dict(cam=(0.0, cp.TOP + 300.0, 0.0), pitch=-0.52, yaw=0.4),
    "side": dict(cam=(0.0, cp.BASE + 0.5 * cp.TALLNESS, 1600.0), pitch=-0.02, yaw=0.0),
    "below": dict(cam=(0.0, 96.0, 60.0), pitch=0.34, yaw=0.15),
}


def main():
    global SUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="tools/cloud-preview-out")
    ap.add_argument("--w", type=int, default=RES)
    ap.add_argument("--h", type=int, default=220)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--patch", type=float, default=0.5)
    args = ap.parse_args()
    SUN = tuple(np.array([0.45, 0.62, -0.64], np.float32)
                / np.linalg.norm([0.45, 0.62, -0.64]))

    views = [v for v in args.views.split(",") if v]
    modes = [m for m in args.modes.split(",") if m]
    for view in views:
        p = PRESETS[view]
        tiles = []
        for mode in modes:
            label = f"{view} / {mode} / cov {args.cov:.1f}"
            arr = render(p["cam"], p["pitch"], p["yaw"], args.cov, args.patch, mode,
                         args.w, args.h, steps=args.steps)
            tiles.append((label, arr))
        sheet(tiles, os.path.join(args.outdir, f"3d_{view}.png"), len(tiles),
              args.w, args.h)


if __name__ == "__main__":
    main()
