"""Offline preview of volume.slang's procedural cloud stack.

A numpy port of RtCloudNoise.generate (the 512-square tileable octave atlas) and of the low cumulus
deck's shape field: the biased coverage threshold, the flat base / domed crown profile and the
fourth-power erosion. Rendered as a plan view, a vertical cross-section and a perspective view
looking up into the deck.

Exists so a change to the shape constants can be judged in seconds instead of a game round-trip. It
is NOT pixel-exact: Python's rounding differs, the lighting is a stand-in (the real one reads the sky
LUT and the day cycle), and only the LOW deck is drawn. What it reproduces faithfully is the field's
GEOMETRY — where the bodies stand, how solid they are and how their rims are cut.

The constants below mirror volume.slang and RtCloudNoise; keep them in step or the preview lies.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

RES = 512
PERIODS = [4, 8, 16, 32, 64, 128]
SEEDS = [101, 211, 307, 401]
GAIN = 1.45

MASK_FINE = 1.30
MASK_LARGE = 0.60
MASK_BIAS = 0.10
MASK_PATCH = 0.35
SHAPE_SOFTNESS = 0.20
BASE_ROUND = 26.0
TOP_POWER = 2.2
EROSION = 0.55
EROSION_BASE = 0.75
EROSION_CELL = 12.0
EROSION_FINE = 0.35
SHEAR = 26.0
# Tiling suppression and the height shear: the warp breaks the atlas wrap, the scroll makes a body
# billow instead of extruding one horizontal blob from base to lid. Values mirror volume.slang.
FIELD_SCROLL = (1.5, -1.1)
WARP_TILE = 0.27
WARP_AMOUNT = 0.18
INNER_TEXTURE = 0.28
FINE_CURL = 0.10
# Per-column crown height: a column's own top comes from the mask read at one fixed reference
# height, so a body domes with its own shape instead of ending on the deck's flat lid.
CROWN_REF = 0.35
HEIGHT_RANGE = 0.55
CROWN_MIN = 0.25
CROWN_POWER = 0.8

FIELD_TILE = 5000.0
DETAIL_TILE = 700.0
BASE = 312.0
TALLNESS = 232.0 * 0.56
TOP = BASE + TALLNESS
LOW_DENSITY = 0.50


def hash01(x, y, seed):
    h = (x * 374761393 + y * 668265263 + seed * 2147483) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h >> 8) / 16777216.0


def lattice(period, seed):
    g = np.empty((period, period), np.float32)
    for y in range(period):
        for x in range(period):
            g[y, x] = hash01(x, y, seed)
    return g


def generate_atlas():
    """Mirror of RtCloudNoise.generate: one independent octave stack per channel."""
    total = np.zeros((RES, RES, 4), np.float32)
    norm, amp = 0.0, 1.0
    for oct_i, period in enumerate(PERIODS):
        step = period / RES
        coords = (np.arange(RES, dtype=np.float32) + 0.5) * step
        i0 = np.floor(coords).astype(np.int32)
        t = coords - i0
        t = t * t * (3.0 - 2.0 * t)
        i1 = (i0 + 1) % period
        i0 %= period
        tx, ty = t[None, :], t[:, None]
        for ch in range(4):
            g = lattice(period, SEEDS[ch] + oct_i * 977)
            a = g[np.ix_(i0, i0)]
            b = g[np.ix_(i0, i1)]
            c = g[np.ix_(i1, i0)]
            d = g[np.ix_(i1, i1)]
            total[:, :, ch] += amp * (
                (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)
        norm += amp
        amp *= 0.5
    return np.clip(0.5 + (total / norm - 0.5) * GAIN, 0.0, 1.0)


NOISE = generate_atlas()


def sample(u, v, ch):
    """Wrap-addressed bilinear fetch; mip levels are ignored in the preview."""
    x = (u % 1.0) * RES
    y = (v % 1.0) * RES
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    x1, y1 = (x0 + 1) % RES, (y0 + 1) % RES
    x0 %= RES
    y0 %= RES
    c = NOISE[:, :, ch]
    a = c[y0, x0] * (1 - fx) + c[y0, x1] * fx
    b = c[y1, x0] * (1 - fx) + c[y1, x1] * fx
    return a * (1 - fy) + b * fy


def cell_noise(px, py, pz):
    a = sample(px / 512.0, (pz + py * 193.0) / 512.0, 1)
    b = sample(px * 1.7 / 512.0, (pz * 1.7 + py * 193.0 + py * 11.0) / 512.0, 2)
    return a * 0.6 + b * 0.4


def coverage_mask(big, fine, cov, patchiness):
    swing = MASK_PATCH * (np.clip(patchiness, 0.0, 1.0) * 2.0 - 1.0)
    large = MASK_LARGE * (1.0 + swing)
    fine_w = MASK_FINE * (1.0 - swing)
    mask = large * big + fine_w * fine + MASK_BIAS
    q = np.clip((cov - mask) / np.sqrt(max(cov, 1e-3)) / SHAPE_SOFTNESS, 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q)


def field_warp(px, pz, tile):
    """Mirror of cloudFieldWarp: one fetch at an incommensurate fraction of the field period."""
    u = px / (tile * WARP_TILE)
    v = pz / (tile * WARP_TILE)
    return (sample(u, v, 0) - 0.5) * WARP_AMOUNT, (sample(u, v, 1) - 0.5) * WARP_AMOUNT


def field_address(px, pz, above, tile, warp):
    """Mirror of cloudFieldAddress: wind (0 here), the height shear, and the warp scaled to tile."""
    sx = px + above * FIELD_SCROLL[0] / tile + warp[0] * tile
    sz = pz + above * FIELD_SCROLL[1] / tile + warp[1] * tile
    return sx, sz


def cloud_curl(px, pz):
    """Mirror of cloudCurl: divergence-free swirl of a gradient hash, magnitude O(1)."""
    def grad(ax, az):
        p3x = (ax * 0.1031) % 1.0
        p3y = (az * 0.1030) % 1.0
        p3z = (ax * 0.0973) % 1.0
        dot = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33)
        p3x, p3y, p3z = p3x + dot, p3y + dot, p3z + dot
        hx = ((p3x + p3y) * p3z) % 1.0
        hy = ((p3x + p3z) * p3y) % 1.0
        nx, ny = hx - 0.5 + 1.0e-4, hy - 0.5 + 1.0e-4
        n = np.sqrt(nx * nx + ny * ny)
        return nx / n, ny / n

    ix, iz = np.floor(px), np.floor(pz)
    fx, fz = px - ix, pz - iz
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    dux = 30.0 * fx * fx * (fx * (fx - 2.0) + 1.0)
    duz = 30.0 * fz * fz * (fz * (fz - 2.0) + 1.0)
    g00 = grad(ix, iz)
    g10 = grad(ix + 1.0, iz)
    g01 = grad(ix, iz + 1.0)
    g11 = grad(ix + 1.0, iz + 1.0)
    v00 = g00[0] * fx + g00[1] * fz
    v10 = g10[0] * (fx - 1.0) + g10[1] * fz
    v01 = g01[0] * fx + g01[1] * (fz - 1.0)
    v11 = g11[0] * (fx - 1.0) + g11[1] * (fz - 1.0)
    vx = (v10 - v00) * (1 - uz) + (v11 - v01) * uz
    vz = (v01 - v00) * (1 - ux) + (v11 - v10) * ux
    return vx * dux, vz * duz


def cumulus(x, z, y, cov, patchiness, eroded=True):
    """Mirror of cloudCumulusDensity for the low deck, at any scalar/broadcast y."""
    below = y - BASE
    bw = field_warp(x, z, FIELD_TILE)
    bx, bz = field_address(x, z, below, FIELD_TILE, bw)
    big = sample(bx / FIELD_TILE, bz / FIELD_TILE, 2)
    fw = field_warp(x, z, DETAIL_TILE)
    fx, fz = field_address(x, z, below, DETAIL_TILE, fw)
    cu, cv = cloud_curl(fx / DETAIL_TILE, fz / DETAIL_TILE)
    fx = fx + cu * FINE_CURL * DETAIL_TILE
    fz = fz + cv * FINE_CURL * DETAIL_TILE
    fine = 1.0 - sample(fx / DETAIL_TILE, fz / DETAIL_TILE, 0)
    inner = sample(fx / DETAIL_TILE, fz / DETAIL_TILE, 3)
    shape = coverage_mask(big, fine, cov, patchiness)
    # Per-column crown: the column's own top comes from the mask read at one fixed reference
    # height, so the height field never moves with the sample's y and a body domes with its own
    # shape instead of every column ending on the deck's one horizontal lid.
    swing = MASK_PATCH * (np.clip(patchiness, 0.0, 1.0) * 2.0 - 1.0)
    rx, rz = field_address(x, z, TALLNESS * CROWN_REF, DETAIL_TILE, fw)
    ref_fine = 1.0 - sample(rx / DETAIL_TILE, rz / DETAIL_TILE, 0)
    hmask = MASK_LARGE * (1.0 + swing) * big + MASK_FINE * (1.0 - swing) * ref_fine + MASK_BIAS
    hexc = np.clip((cov - hmask) / np.sqrt(max(cov, 1.0e-3)) / HEIGHT_RANGE, 0.0, 1.0)
    col_tall = np.maximum(TALLNESS * (CROWN_MIN + (1.0 - CROWN_MIN) * hexc ** CROWN_POWER), 1.0)
    above = BASE + col_tall - y
    shape = np.minimum(np.minimum(shape, np.clip(above, 0, 1)), np.clip(below, 0, 1))
    bottom = 1.0 - (1.0 - np.clip(below / BASE_ROUND, 0, 1)) ** 5
    top = 1.0 - (1.0 - np.clip(above / col_tall, 0, 1)) ** TOP_POWER
    shape = np.maximum(shape - 1.0 + top * bottom, 0.0)
    shape = shape * (1.0 - INNER_TEXTURE * (1.0 - inner))
    if not eroded:
        return np.maximum(shape - EROSION * 0.5, 0.0)
    lean = (np.maximum(below - 20.0, 0.0) / max(TALLNESS * 0.2, 1.0)) ** 1.5 * SHEAR
    qx = x + lean
    falloff = 1.0 - np.clip(above / np.maximum(col_tall * 0.75, 1.0), 0, 1)
    e1 = 1.0 - cell_noise(qx / EROSION_CELL, y / EROSION_CELL, z / EROSION_CELL)
    e2 = cell_noise(qx / (EROSION_CELL * 0.55), y / (EROSION_CELL * 0.55),
                    z / (EROSION_CELL * 0.55))
    e3 = 1.0 - cell_noise(qx / (EROSION_CELL * 0.30), y / (EROSION_CELL * 0.30),
                          z / (EROSION_CELL * 0.30))
    om = 1.0 - shape
    er = e1 * np.sqrt(om) + np.abs(e2 - falloff) * EROSION_BASE * om * om * (1.0 - falloff * 0.5)
    er = er + e3 * EROSION_FINE * om * (1.0 - falloff * 0.5)
    return np.maximum(shape - er ** 4 * EROSION, 0.0)


def render_plan(cov, patchiness, width, height, window=6000.0):
    xs = np.linspace(-window / 2, window / 2, width, dtype=np.float32)[None, :]
    zs = np.linspace(-window * 2 / 3, window * 2 / 3, height, dtype=np.float32)[:, None]
    d = cumulus(xs, zs, BASE + 0.5 * TALLNESS, cov, patchiness)
    return (np.clip(d, 0, 1) * 255).astype(np.uint8)


def render_persp(cov, patchiness, width, height):
    """Looking up from y=64 through a wide cone; lighting is a stand-in gradient."""
    cam_y = 64.0
    fov = math.radians(70.0)
    u = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
    v = (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0 - 1.0
    dx = u[None, :] * math.tan(fov / 2) * (width / height)
    dy = np.maximum(0.55 - v[:, None] * 0.45, 0.02)
    dz = -np.sqrt(np.maximum(1.0 - dx * dx, 0.05))
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    mid = BASE + 0.5 * TALLNESS
    t = (mid - cam_y) / dy
    d = cumulus(dx * t, dz * t, mid, cov, patchiness)
    tau = d * LOW_DENSITY * TALLNESS / np.maximum(np.abs(dy), 0.04)
    fade = 1.0 - np.clip((t - 9830.0) / (16384.0 - 9830.0), 0.0, 1.0)
    opacity = (1.0 - np.exp(-tau)) * fade
    sky = np.stack([
        0.42 + 0.30 * (1.0 - dy) + 0.10 * np.maximum(dx, 0),
        0.58 + 0.24 * (1.0 - dy) + 0.05 * np.maximum(dx, 0),
        0.86 - 0.12 * dy], axis=-1)
    cloud = np.stack([0.93 + 0.07 * d, 0.94 + 0.06 * d, 0.97 + 0.03 * d], axis=-1)
    out = cloud * opacity[:, :, None] + sky * (1.0 - opacity[:, :, None])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def render_cross(cov, patchiness, width, height):
    xs = np.linspace(-3000, 3000, width, dtype=np.float32)[None, :]
    ys = np.linspace(BASE - 20, TOP + 40, height, dtype=np.float32)[:, None]
    d = cumulus(xs, np.zeros_like(xs), ys, cov, patchiness)
    return (np.clip(d, 0, 1) * 255).astype(np.uint8)


VARIANTS = [
    ("cov 40% (scattered)", dict(cov=0.4, patchiness=0.5)),
    ("cov 70%", dict(cov=0.7, patchiness=0.5)),
    ("cov 100% (authored)", dict(cov=1.0, patchiness=0.5)),
    ("cov 140% (overcast)", dict(cov=1.4, patchiness=0.5)),
    ("cov 100%, patch 0%", dict(cov=1.0, patchiness=0.0)),
    ("cov 100%, patch 100%", dict(cov=1.0, patchiness=1.0)),
]


def sheet(items, out, cols, width, height, mode, pad=6, bar=22):
    tiles = []
    for label, p in items:
        if mode == "persp":
            arr = render_persp(width=width, height=height, **p)
        elif mode == "cross":
            arr = render_cross(width=width, height=height, **p)
        else:
            arr = render_plan(width=width, height=height, **p)
        tiles.append((label, arr))
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * width + (cols + 1) * pad,
                            rows * (height + bar) + (rows + 1) * pad), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    for i, (label, arr) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (width + pad)
        y = pad + r * (height + bar + pad)
        img.paste(Image.fromarray(arr), (x, y + bar))
        draw.text((x + 2, y + 5), label, fill=(230, 230, 230))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print("wrote", out, img.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="tools/cloud-preview-out")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=300)
    args = ap.parse_args()
    sheet(VARIANTS, os.path.join(args.outdir, "sheet_persp.png"), 3, args.w, args.h, "persp")
    sheet(VARIANTS, os.path.join(args.outdir, "sheet_plan.png"), 3, args.w, args.h, "plan")
    sheet(VARIANTS, os.path.join(args.outdir, "sheet_cross.png"), 3, args.w, args.h, "cross")


if __name__ == "__main__":
    main()
