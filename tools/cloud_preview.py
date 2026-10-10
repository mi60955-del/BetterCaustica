"""Offline preview of volume.slang's planar-cloud coverage field.

A line-by-line numpy port of the shader's noise stack (cloudHash12 / cloudValueNoise / cloudFbm /
the wind-frame coverage field with its warp, mid-scale carve, cross-wind shred and macro patch
gate), rendered as a perspective view looking up at the pushed cloud plane. Exists so cloud-shape
tuning can be iterated in seconds instead of a game round-trip per parameter change.

The projection mirrors planarCloudSkyInScatter: t = (planeY - camY) / d.y, tau = cov * density *
CLOUD_EXTINCTION * thickness / max(|d.y|, 0.04), opacity = 1 - exp(-tau), and the blend is
cloudL + (1 - opacity) * sky.

Not a pixel-exact port: Python runs float64/float32 at its own rounding and the lighting is a
stand-in (the real one reads the sky LUT and the day cycle). What it reproduces faithfully is the
GEOMETRY of the field: streak anisotropy, patch clustering, scale mix.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32

CLOUD_EXTINCTION = F32(0.22)   # volume.slang
CLOUD_G = F32(0.4)


def frac(x):
    return x - np.floor(x)


def hash12(px, py):
    """Dave Hoskins' hash12, the shader's cloudHash12."""
    x = frac(px * F32(0.1031))
    y = frac(py * F32(0.1031))
    z = x                      # shader builds float3(p.x, p.y, p.x)
    d = x * (y + F32(33.33)) + y * (z + F32(33.33)) + z * (x + F32(33.33))
    x = x + d
    y = y + d
    z = z + d
    return frac((x + y) * z)


def value_noise(px, py):
    ix = np.floor(px)
    iy = np.floor(py)
    fx = px - ix
    fy = py - iy
    ux = fx * fx * (F32(3.0) - F32(2.0) * fx)
    uy = fy * fy * (F32(3.0) - F32(2.0) * fy)
    a = hash12(ix, iy)
    b = hash12(ix + F32(1.0), iy)
    c = hash12(ix, iy + F32(1.0))
    d = hash12(ix + F32(1.0), iy + F32(1.0))
    return (a + (b - a) * ux) * (F32(1.0) - uy) + (c + (d - c) * ux) * uy


def fbm(px, py, octaves=4):
    s = np.zeros_like(px, dtype=F32)
    amp = F32(0.5)
    norm = F32(0.0)
    for _ in range(octaves):
        s = s + amp * value_noise(px, py)
        norm = norm + amp
        # shader: p = p * 2.0 + float2(31.0, 17.0) — the offset is constant per octave
        px = px * F32(2.0) + F32(31.0)
        py = py * F32(2.0) + F32(17.0)
        amp = amp * F32(0.5)
    return s / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def coverage(px, py, c, patchiness, along=48.0, cross=16.0, warp_freq=0.35, warp_amp=1.6,
             mid_carve=0.12, shred=0.18, macro_along=2048.0, macro_cross=1024.0,
             macro_octaves=4, wind_deg=0.0, dir_jitter=0.0, dir_cell=900.0,
             cross_lo=None, cross_hi=None, warp2=0.0, warp2_freq=0.9, octaves=4,
             width_floor=0.0, thresh_k=1.0, width_base=0.0, width_k=0.30):
    """The shader's cloudCoverageField over world XZ (anchor and drift are zero here).

    dir_jitter / dir_cell / cross_lo / cross_hi / warp2 are preview-only probes for structure
    ideas not yet in the shader (per-region wind direction, macro-modulated streak width, a
    second warp octave) so a candidate can be judged before it is implemented.
    """
    a = math.radians(wind_deg)
    dx, dy = math.cos(a), math.sin(a)
    if dir_jitter > 0.0:
        # Wind direction wanders slowly around the pushed azimuth (radians of jitter). px/py are
        # arrays here, so the trig must stay vectorised.
        j = (value_noise(px / F32(dir_cell) + F32(3.0), py / F32(dir_cell) + F32(7.0)) - F32(0.5))
        a = a + j * F32(math.radians(dir_jitter) * 2.0)
        dx, dy = np.cos(a), np.sin(a)
    perx, pery = -dy, dx

    # The streak width can breathe with the macro field: round-ish clusters where the macro is
    # low, long streets where it is high. Probed here before it goes into the shader.
    if cross_lo is not None:
        mgu0 = (px * F32(dx) + py * F32(dy)) / F32(macro_along)
        mgv0 = (px * F32(perx) + py * F32(pery)) / F32(macro_cross)
        m0 = fbm(mgu0, mgv0, macro_octaves)
        cross_eff = F32(cross_lo) + (F32(cross_hi) - F32(cross_lo)) * m0
    else:
        cross_eff = F32(cross)

    u = (px * F32(dx) + py * F32(dy)) / F32(along)
    v = (px * F32(perx) + py * F32(pery)) / cross_eff

    wq = F32(warp_freq)
    wu = value_noise(u * wq + F32(17.0), v * wq + F32(17.0)) - F32(0.5)
    wv = value_noise(u * wq + F32(91.0), v * wq + F32(91.0)) - F32(0.5)
    u = u + wu * F32(warp_amp)
    v = v + wv * F32(warp_amp)
    if warp2 > 0.0:
        w2q = F32(warp2_freq)
        u = u + (value_noise(u * w2q + F32(53.0), v * w2q + F32(53.0)) - F32(0.5)) * F32(warp2)
        v = v + (value_noise(u * w2q + F32(157.0), v * w2q + F32(157.0)) - F32(0.5)) * F32(warp2)

    n = (fbm(u, v, octaves)
         - F32(mid_carve) * value_noise(u * F32(0.25) + F32(7.0), v * F32(0.25) + F32(7.0))
         - F32(shred) * value_noise(u + F32(31.0), v * F32(6.0) + F32(31.0)))

    # Threshold/width response: the shader's rule is threshold 1-c with width 0.30c. thresh_k < 1
    # or a width floor/base widens the usable slider range so mid coverages stay interesting.
    width = max(F32(width_base) + F32(width_k) * F32(c), F32(width_floor))
    cov = np.clip((n - (F32(1.0) - F32(c) * F32(thresh_k))) / max(width, F32(1.0e-3)), 0.0, 1.0)

    mgu = (px * F32(dx) + py * F32(dy)) / F32(macro_along)
    mgv = (px * F32(perx) + py * F32(pery)) / F32(macro_cross)
    macro = fbm(mgu, mgv, macro_octaves)
    gate = smoothstep(F32(0.28), F32(0.78), macro)
    gate = gate + (F32(1.0) - gate) * np.clip((F32(c) - F32(0.7)) / F32(0.3), 0.0, 1.0)
    return cov * (F32(1.0) - np.clip(F32(patchiness), 0.0, 1.0)
                  + np.clip(F32(patchiness), 0.0, 1.0) * gate)


def hg(cos_theta, g):
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / np.maximum(denom * np.sqrt(np.maximum(denom, 1e-6)), 1e-6)


def render(params, width=480, height=300, cam_y=70.0, plane_y=192.0, density=1.0, thickness=12.0,
           pitch_deg=42.0, fov_deg=60.0, sun_elev_deg=28.0, sun_azim_deg=35.0, world_x=0.0,
           world_z=0.0):
    """Perspective view from below the cloud plane; exact sheet geometry, stand-in lighting."""
    xs = (np.arange(width, dtype=F32) + F32(0.5)) / F32(width)
    ys = (np.arange(height, dtype=F32) + F32(0.5)) / F32(height)
    ndc_x = (xs * F32(2.0) - F32(1.0)) * math.tan(math.radians(fov_deg) * 0.5)
    ndc_y = (F32(1.0) - ys * F32(2.0)) * math.tan(math.radians(fov_deg) * 0.5) * (height / width)
    ndc_x, ndc_y = np.meshgrid(ndc_x, ndc_y)

    p0 = math.radians(pitch_deg)
    fy = math.sin(p0)
    fz = math.cos(p0)
    d_x = ndc_x
    d_y = ndc_y * math.cos(p0) + fy
    d_z = -ndc_y * math.sin(p0) + fz
    norm = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
    d_x, d_y, d_z = d_x / norm, d_y / norm, d_z / norm

    plane_hit = d_y > F32(1.0e-3)
    safe_dy = np.where(plane_hit, d_y, F32(1.0))
    t = (F32(plane_y) - F32(cam_y)) / safe_dy
    hit_x = F32(world_x) + d_x * t
    hit_z = F32(world_z) + d_z * t

    cov = coverage(hit_x, hit_z, params["coverage"], params["patchiness"],
                   along=params["along"], cross=params["cross"],
                   warp_freq=params["warp_freq"], warp_amp=params["warp_amp"],
                   mid_carve=params["mid_carve"], shred=params["shred"],
                   macro_along=params["macro_along"], macro_cross=params["macro_cross"],
                   macro_octaves=params["macro_octaves"], wind_deg=params["wind_deg"],
                   dir_jitter=params.get("dir_jitter", 0.0),
                   dir_cell=params.get("dir_cell", 900.0),
                   cross_lo=params.get("cross_lo"), cross_hi=params.get("cross_hi"),
                   warp2=params.get("warp2", 0.0), warp2_freq=params.get("warp2_freq", 0.9),
                   octaves=params.get("octaves", 4), width_floor=params.get("width_floor", 0.0),
                   thresh_k=params.get("thresh_k", 1.0), width_base=params.get("width_base", 0.0),
                   width_k=params.get("width_k", 0.30))
    cov = np.where(plane_hit, cov, F32(0.0))

    tau = cov * F32(density) * CLOUD_EXTINCTION * F32(thickness) / np.maximum(np.abs(d_y), F32(0.04))
    opacity = F32(1.0) - np.exp(-tau)
    return composite(cov, opacity, d_x, d_y, d_z, sun_elev_deg, sun_azim_deg)


def render_plan(params, width=480, height=300, span_x=6000.0, span_z=3750.0, world_x=0.0,
                world_z=0.0, density=1.0, thickness=12.0, sun_elev_deg=50.0, sun_azim_deg=35.0):
    """Top-down view of the coverage field over a span_x x span_z block window (structure read)."""
    xs = F32(world_x) + (np.arange(width, dtype=F32) + F32(0.5)) / F32(width) * F32(span_x)
    zs = F32(world_z) + (np.arange(height, dtype=F32) + F32(0.5)) / F32(height) * F32(span_z)
    px, pz = np.meshgrid(xs, zs)
    cov = coverage(px, pz, params["coverage"], params["patchiness"],
                   along=params["along"], cross=params["cross"],
                   warp_freq=params["warp_freq"], warp_amp=params["warp_amp"],
                   mid_carve=params["mid_carve"], shred=params["shred"],
                   macro_along=params["macro_along"], macro_cross=params["macro_cross"],
                   macro_octaves=params["macro_octaves"], wind_deg=params["wind_deg"],
                   dir_jitter=params.get("dir_jitter", 0.0),
                   dir_cell=params.get("dir_cell", 900.0),
                   cross_lo=params.get("cross_lo"), cross_hi=params.get("cross_hi"),
                   warp2=params.get("warp2", 0.0), warp2_freq=params.get("warp2_freq", 0.9),
                   octaves=params.get("octaves", 4), width_floor=params.get("width_floor", 0.0),
                   thresh_k=params.get("thresh_k", 1.0), width_base=params.get("width_base", 0.0),
                   width_k=params.get("width_k", 0.30))
    tau = cov * F32(density) * CLOUD_EXTINCTION * F32(thickness)   # straight-down slant = 1
    opacity = F32(1.0) - np.exp(-tau)
    d_x = np.full_like(cov, math.cos(math.radians(sun_azim_deg)) * math.cos(math.radians(sun_elev_deg)))
    d_y = np.full_like(cov, math.sin(math.radians(sun_elev_deg)))
    d_z = np.full_like(cov, math.sin(math.radians(sun_azim_deg)) * math.cos(math.radians(sun_elev_deg)))
    return composite(cov, opacity, d_x, d_y, d_z, sun_elev_deg, sun_azim_deg)


def composite(cov, opacity, d_x, d_y, d_z, sun_elev_deg, sun_azim_deg):
    """Shared shading: HG sun on the slab, self-shadow, a horizon-bright stand-in sky."""
    sa = math.radians(sun_azim_deg)
    se = math.radians(sun_elev_deg)
    s_dx = math.cos(se) * math.cos(sa)
    s_dy = math.sin(se)
    s_dz = math.cos(se) * math.sin(sa)
    cos_theta = np.clip(d_x * s_dx + d_y * s_dy + d_z * s_dz, -1.0, 1.0)
    phase = hg(cos_theta, float(CLOUD_G)) * F32(0.7)
    tau_self = cov * CLOUD_EXTINCTION * F32(12.0) / F32(max(math.sin(se), 0.05))
    sun = phase * F32(1.15) * np.exp(-tau_self)
    ambient = F32(0.30)
    lit = sun + ambient
    cloud_l = np.dstack([lit * F32(1.00), lit * F32(0.99), lit * F32(0.97)])

    # Sky: dark blue zenith to bright horizon haze.
    horiz = np.clip(F32(1.0) - np.abs(d_y) * F32(2.2), 0.0, 1.0)
    sky = (np.dstack([np.full_like(cov, F32(0.16)), np.full_like(cov, F32(0.28)),
                      np.full_like(cov, F32(0.52))]) * (1.0 - horiz)[..., None]
           + np.dstack([np.full_like(cov, F32(0.72)), np.full_like(cov, F32(0.78)),
                        np.full_like(cov, F32(0.80))]) * horiz[..., None])

    rgb = cloud_l * opacity[..., None] + sky * (F32(1.0) - opacity[..., None])
    rgb = rgb / (F32(1.0) + rgb)          # Reinhard, just to keep the preview readable
    img = (np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)
    return img


# Shipped shader values (volume.slang) — keep this mirror exact or the preview lies.
BASE = dict(coverage=0.5, patchiness=0.7, along=96.0, cross=16.0, cross_lo=16.0, cross_hi=40.0,
            warp_freq=0.35, warp_amp=1.1, mid_carve=0.07, shred=0.12, octaves=3,
            macro_along=2048.0, macro_cross=1024.0, macro_octaves=4, wind_deg=0.0,
            dir_jitter=9.74, dir_cell=900.0, thresh_k=1.15, width_base=0.10, width_k=0.20)

def variant(**kw):
    p = dict(BASE)
    p.update(kw)
    return p


# Round-3 candidate base: bigger cells, 3 octaves, mild structure jitter, gentler response.
CAND = dict(octaves=3, cross=32.0, along=96.0, dir_jitter=10.0, cross_lo=16.0, cross_hi=40.0,
            warp_amp=1.1, mid_carve=0.07, shred=0.12,
            thresh_k=1.15, width_base=0.10, width_k=0.20)

SHEET_PERSP = [
    ("shipped @cov0.20 (sparse)", variant(coverage=0.20)),
    ("shipped @cov0.35", variant(coverage=0.35)),
    ("shipped @cov0.50", variant(coverage=0.50)),
    ("shipped @cov0.65", variant(coverage=0.65)),
    ("shipped @cov0.80", variant(coverage=0.80)),
    ("@cov0.50, patchiness 0%", variant(coverage=0.50, patchiness=0.0)),
    ("@cov0.50, patchiness 100%", variant(coverage=0.50, patchiness=1.0)),
    ("@cov0.50, dissolve 3000 (near)", variant(coverage=0.50)),
]

SHEET_PLAN = [
    ("plan shipped @cov0.20", variant(coverage=0.20)),
    ("plan shipped @cov0.35", variant(coverage=0.35)),
    ("plan shipped @cov0.50", variant(coverage=0.50)),
    ("plan shipped @cov0.65", variant(coverage=0.65)),
]

def sheet(items, out, cols, w, h, mode, pad=6, bar=22):
    tiles = [(label, render(params, width=w, height=h) if mode == "persp"
              else render_plan(params, width=w, height=h)) for label, params in items]
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + bar) + (rows + 1) * pad),
                    (24, 24, 28))
    draw = ImageDraw.Draw(img)
    for i, (label, arr) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + bar + pad)
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
    sheet(SHEET_PERSP, os.path.join(args.outdir, "sheet_persp.png"), 4, args.w, args.h, "persp")
    sheet(SHEET_PLAN, os.path.join(args.outdir, "sheet_plan.png"), 2, args.w, args.h, "plan")


if __name__ == "__main__":
    main()
