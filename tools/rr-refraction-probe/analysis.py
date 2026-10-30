#!/usr/bin/env python3
"""Analysis half of the probe: image helpers, exact flow, metrics, demos, report.

Imported by probe.py; kept separate so the geometry stays readable.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EPS = 1e-9


# --------------------------------------------------------------------------- #
# exact image flow: invert the pixel -> surface map
# --------------------------------------------------------------------------- #
def forward_face(uv, pose, res, sc):
    from probe import trace, ray_dirs
    return trace(ray_dirs(pose, uv, res, sc.fov_y), pose, sc)["P_face"]


def solve_uv(W, pose, res, sc, iters=14):
    """Pixel in `pose` whose refracted ray lands on the 3D point W."""
    from probe import trace, ray_dirs, pinhole_project
    uv = pinhole_project(W, pose, res, sc.fov_y)
    err = np.full(W.shape[0], np.inf)
    for _ in range(iters):
        P0 = trace(ray_dirs(pose, uv, res, sc.fov_y), pose, sc)["P_face"]
        e = P0[:, :2] - W[:, :2]
        err = np.sqrt(np.sum(e * e, axis=1))
        if not np.any(err > 2e-7):
            break
        h = 0.5
        Pu = trace(ray_dirs(pose, uv + np.array([h, 0.0]), res, sc.fov_y), pose, sc)["P_face"][:, :2]
        Pv = trace(ray_dirs(pose, uv + np.array([0.0, h]), res, sc.fov_y), pose, sc)["P_face"][:, :2]
        Ju = (Pu - P0[:, :2]) / h
        Jv = (Pv - P0[:, :2]) / h
        det = Ju[:, 0] * Jv[:, 1] - Ju[:, 1] * Jv[:, 0]
        det = np.where(np.abs(det) < 1e-13, 1e-13, det)
        du = -(Jv[:, 1] * e[:, 0] - Jv[:, 0] * e[:, 1]) / det
        dv = -(-Ju[:, 1] * e[:, 0] + Ju[:, 0] * e[:, 1]) / det
        uv = uv + np.stack([np.clip(du, -30.0, 30.0), np.clip(dv, -30.0, 30.0)], 1)
    return uv, err


def rot_mat_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def project_dir(d, pose, res, fov_y):
    """Pinhole projection of a world *direction* (no translation)."""
    from probe import camera_basis
    f, r, u = camera_basis(pose)
    ty = math.tan(fov_y * 0.5)
    W, H = res
    tx = ty * (W / H)
    z = np.maximum(d @ f, 1e-9)
    return np.stack([((d @ r) / z / tx + 1.0) * 0.5 * W,
                     (1.0 - (d @ u) / z / ty) * 0.5 * H], 1)


def rotation_flow(uv0, pose0, pose1, res, sc):
    """Pure-rotation flow of a pinhole camera.

    The world direction of the refracted path's entry ray is fixed by the
    surface point, so a camera rotation alone just re-projects that same world
    direction through the new basis.  The flow is therefore depth independent
    and exactly what any depth-based motion vector produces.
    """
    from probe import ray_dirs
    d = ray_dirs(pose0, uv0, res, sc.fov_y)
    return project_dir(d, pose1, res, sc.fov_y) - uv0


def sample_bilinear(img, uv):
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0] - 0.5, 0.0, W - 1.001)
    y = np.clip(uv[:, 1] - 0.5, 0.0, H - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def to_u8(img):
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def font(sz=13):
    try:
        return ImageFont.load_default(size=sz)
    except Exception:
        return ImageFont.load_default()


def label(img_u8, text, sz=15):
    im = Image.fromarray(np.ascontiguousarray(img_u8))
    d = ImageDraw.Draw(im)
    f = font(sz)
    bb = d.textbbox((0, 0), text, font=f)
    d.rectangle([0, 0, bb[2] + 10, bb[3] + 8], fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def save(img_u8, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    Image.fromarray(np.ascontiguousarray(img_u8)).save(p)
    return p


def contact_sheet(tiles, cols, pad=8, bg=(18, 18, 18)):
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), np.uint8)
    sheet[:, :] = bg
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = t
    return sheet


# --------------------------------------------------------------------------- #
# shadow measurement
# --------------------------------------------------------------------------- #
def interior_mask(gmk, margin=3):
    m = gmk > 0.995
    er = m.copy()
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            er &= np.roll(np.roll(m, dy, 0), dx, 1)
    return er


def edge_crossings(cov, axis="v"):
    """Sub-pixel 0.5 crossings, per column (axis='v') or per row (axis='h').

    Returns (first, last) crossing coordinate for every line; NaN when absent.
    """
    a = cov if axis == "v" else cov.T
    n_line = a.shape[1]
    first = np.full(n_line, np.nan)
    last = np.full(n_line, np.nan)
    for j in range(n_line):
        c = a[:, j]
        idx = np.nonzero(c >= 0.5)[0]
        if idx.size == 0:
            continue
        i0, i1 = idx[0], idx[-1]
        if i0 > 0:
            c0, c1 = c[i0 - 1], c[i0]
            if c1 - c0 > 1e-6:
                first[j] = (i0 - 1) + (0.5 - c0) / (c1 - c0)
        if i1 < c.shape[0] - 1:
            c0, c1 = c[i1], c[i1 + 1]
            if c0 - c1 > 1e-6:
                last[j] = i1 + (c0 - 0.5) / (c0 - c1)
    return first, last


def shadow_window(cov, gmk, pad=45):
    m = cov > 0.5
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    H, W = cov.shape
    y0 = max(0, ys.min() - pad)
    y1 = min(H, ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(W, xs.max() + 1 + pad)
    inter = interior_mask(gmk, 2)
    # shrink until the whole window is interior glass
    for _ in range(8):
        if inter[y0:y1, x0:x1].all():
            break
        y0 = min(y0 + 6, H - 2)
        y1 = max(y1 - 6, y0 + 1)
        x0 = min(x0 + 6, W - 2)
        x1 = max(x1 - 6, x0 + 1)
    if not inter[y0:y1, x0:x1].all():
        return None
    return (slice(y0, y1), slice(x0, x1))


def shadow_stats(cov, win):
    sub = cov[win]
    m = (sub > 0.5).astype(np.float64)
    tot = m.sum()
    if tot <= 0:
        return None
    ys, xs = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    cy = (m * ys).sum() / tot + win[0].start
    cx = (m * xs).sum() / tot + win[1].start
    top, bot = edge_crossings(sub, "v")
    lef, rig = edge_crossings(sub, "h")
    ok_v = np.isfinite(top) & np.isfinite(bot)
    ok_h = np.isfinite(lef) & np.isfinite(rig)
    return dict(
        area=float(tot),
        cx=float(cx), cy=float(cy),
        top=top + win[0].start, bot=bot + win[0].start,
        lef=lef + win[1].start, rig=rig + win[1].start,
        height=float(np.nanmedian((bot - top)[ok_v])) if ok_v.any() else float("nan"),
        width=float(np.nanmedian((rig - lef)[ok_h])) if ok_h.any() else float("nan"),
    )


def compare_shadow(ref, cur):
    """Piecewise differences between two shadow_stats."""
    out = {}
    out["centroid_px"] = math.hypot(cur["cx"] - ref["cx"], cur["cy"] - ref["cy"])
    out["dcx"] = cur["cx"] - ref["cx"]
    out["dcy"] = cur["cy"] - ref["cy"]
    out["area_pct"] = 100.0 * (cur["area"] - ref["area"]) / max(ref["area"], 1.0)
    out["d_width"] = cur["width"] - ref["width"]
    out["d_height"] = cur["height"] - ref["height"]
    for key in ("top", "bot", "lef", "rig"):
        a, b = ref[key], cur[key]
        ok = np.isfinite(a) & np.isfinite(b)
        out["rms_" + key] = float(np.sqrt(np.mean((b[ok] - a[ok]) ** 2))) if ok.any() else float("nan")
        out["max_" + key] = float(np.max(np.abs(b[ok] - a[ok]))) if ok.any() else float("nan")
    return out
