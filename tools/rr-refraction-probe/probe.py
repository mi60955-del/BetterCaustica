#!/usr/bin/env python3
"""
Glass-refraction shadow probe  (standalone; no engine, no GPU, no mod code)

Scene: oblique camera -> glass -> textured block face touching the glass back
        -> one hard cast shadow.

Why a convex-polyhedron glass and not a flat pane: a parallel slab does not
bend the ray, it only offsets it.  That makes every straight-line guide look
almost correct and hides guide errors.  A wedge (bevel / lens / non-parallel
faces) does bend the ray, and that is where a denoiser's straight-line
assumptions can go badly wrong.  `--wedge` tilts the back face so both regimes
can be measured with the same code.

Per pixel the probe computes:
  * the exact surface point seen through the glass (Snell at both interfaces)
  * where that point appears in the next frame (Newton inversion of the
    pixel->surface map) -> the only correct history-reprojection target
  * where each straight-line guide would reproject to (depth from glass front,
    glass back, unbent hit, hit-distance virtual point)
  * the resulting shadow-edge error in pixels, statically and under motion

Everything is exact geometry: no denoiser, no ML, no heuristics.
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
EPS = 1e-9


# --------------------------------------------------------------------------- #
# vector helpers
# --------------------------------------------------------------------------- #
def unit(v):
    n = np.sqrt(np.sum(v * v, axis=1, keepdims=True))
    return v / np.maximum(n, 1e-12)


def dot(a, b):
    return np.sum(a * b, axis=1)


def reflect(I, N):
    return unit(I - 2.0 * dot(I, N)[:, None] * N)


def refract(I, N, eta):
    """N oriented against I (dot(I,N) < 0); eta = n_incident / n_transmitted."""
    cosi = -dot(I, N)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    T = eta * I + (eta * cosi - np.sqrt(np.maximum(k, 0.0)))[:, None] * N
    return unit(T), k > 0.0


def plane_hit(o, d, n, p):
    n = np.asarray(n, np.float64).reshape(3)
    p = np.asarray(p, np.float64).reshape(3)
    den = d @ n
    return ((p - o) @ n) / np.where(np.abs(den) < EPS, EPS, den)


def poly_enter_exit(o, d, normals, offsets):
    """Convex polyhedron as half-spaces n.x <= b. Returns enter/exit t and normals."""
    M = normals.shape[0]
    den = d @ normals.T                                   # (N,M)
    num = offsets[None, :] - o @ normals.T
    t = num / np.where(np.abs(den) < 1e-12, 1e-12, den)
    t_in = np.where(den < -1e-12, t, -np.inf)
    t_out = np.where(den > 1e-12, t, np.inf)
    i_in = np.argmax(t_in, axis=1)
    i_out = np.argmin(t_out, axis=1)
    rows = np.arange(d.shape[0])
    t_enter = t_in[rows, i_in]
    t_exit = t_out[rows, i_out]
    n_enter = normals[i_in]
    n_exit = normals[i_out]
    valid = np.isfinite(t_enter) & np.isfinite(t_exit) & (t_exit > t_enter)
    return valid, t_enter, t_exit, n_enter, n_exit


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
class Scene:
    def __init__(self, gap=0.0, thickness=0.5, ior=1.52, wedge_deg=0.0, gx=1.15, gy=0.90):
        self.fov_y = math.radians(40.0)

        # glass: convex polyhedron, front face z = z0, back face tilted by wedge
        self.z0 = 2.0
        self.thickness = thickness
        self.gx, self.gy = gx, gy
        self.ior = ior

        beta = math.radians(wedge_deg)
        self.beta = beta
        n_back = np.array([math.sin(beta), 0.0, math.cos(beta)])
        p_back = np.array([0.0, 0.0, self.z0 + thickness])
        self.n_back = n_back
        self.p_back = p_back
        off_back = float(n_back @ p_back)

        self.glass_n = np.array([
            [0.0, 0.0, -1.0],      # front  (-z <= -z0)
            n_back,                # back
            [-1.0, 0.0, 0.0],      # left
            [1.0, 0.0, 0.0],       # right
            [0.0, -1.0, 0.0],      # bottom
            [0.0, 1.0, 0.0],       # top
        ])
        self.glass_b = np.array([-self.z0, off_back, gx, gx, gy, gy])

        # the block face touching the glass back, optionally set back by `gap`
        self.n_face = -n_back
        self.p_face = p_back + n_back * gap
        self.face_gap = gap

        # gobo casting the shadow onto the face
        self.gobo_min = np.array([-1.35, 0.45, 0.95])
        self.gobo_max = np.array([-0.45, 1.35, 1.35])
        self.light_dir = unit(np.array([[0.45, -0.45, 0.77]]))[0]

        # brick texture on the face plane, with a block seam at x = seam_x
        up = np.array([0.0, 1.0, 0.0])
        self.e1 = unit(np.cross(self.n_face[None, :], up[None, :]))[0]
        self.e2 = np.cross(self.e1, self.n_face)
        self.brick_w, self.brick_h, self.mortar = 0.42, 0.21, 0.022
        self.brick_base = np.array([0.66, 0.63, 0.58])
        self.mortar_col = np.array([0.38, 0.38, 0.37])
        self.seam_x = 0.52
        self.stone_base = np.array([0.30, 0.33, 0.38])

        self.lit = 0.95
        self.unlit = 0.10
        self.glass_tint = np.array([0.90, 0.96, 0.94])
        self.fresnel0 = 0.035
        self.sky_zen = np.array([0.10, 0.20, 0.42])
        self.sky_hor = np.array([0.42, 0.50, 0.60])

    def face_point_at(self, P):
        q = P - self.p_face[None, :]
        return q @ self.e1, q @ self.e2


class Pose:
    def __init__(self, pos, yaw, pitch):
        self.pos = np.asarray(pos, np.float64)
        self.yaw = float(yaw)
        self.pitch = float(pitch)


def camera_basis(pose):
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    f = np.array([sy * cp, sp, cy * cp])
    r = np.array([cy, 0.0, -sy])
    u = np.cross(f, r)
    return f, r, u


def ray_dirs(pose, uv, res, fov_y):
    W, H = res
    ty = math.tan(fov_y * 0.5)
    tx = ty * (W / H)
    x = (uv[:, 0] / W * 2.0 - 1.0) * tx
    y = (1.0 - uv[:, 1] / H * 2.0) * ty
    f, r, u = camera_basis(pose)
    return unit(f[None, :] + x[:, None] * r[None, :] + y[:, None] * u[None, :])


def pinhole_project(P, pose, res, fov_y):
    W, H = res
    ty = math.tan(fov_y * 0.5)
    tx = ty * (W / H)
    f, r, u = camera_basis(pose)
    q = P - pose.pos[None, :]
    z = q @ f
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([((q @ r) / z / tx + 1.0) * 0.5 * W,
                     (1.0 - (q @ u) / z / ty) * 0.5 * H], axis=1)


def pixel_centers(res):
    W, H = res
    gy, gx = np.mgrid[0:H, 0:W]
    return np.stack([gx.ravel() + 0.5, gy.ravel() + 0.5], axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# tracing
# --------------------------------------------------------------------------- #
def trace(dirs, pose, sc):
    N = dirs.shape[0]
    cam = pose.pos
    t_face = plane_hit(cam, dirs, sc.n_face, sc.p_face)
    P_straight = cam[None, :] + dirs * t_face[:, None]

    o = np.broadcast_to(cam[None, :], dirs.shape)
    valid, t_enter, t_exit, n_in, n_out = poly_enter_exit(o, dirs, sc.glass_n, sc.glass_b)
    glass = valid & (t_enter > 1e-4) & (t_face > t_enter)

    P_face = P_straight.copy()
    P_front = np.full((N, 3), np.nan)
    P_back = np.full((N, 3), np.nan)
    P_pdir = np.full((N, 3), np.nan)
    path_len = np.zeros(N)
    n_glass = np.zeros((N, 3))
    tir = np.zeros(N, bool)

    idx = np.nonzero(glass)[0]
    if idx.size:
        D = dirs[idx]
        E = cam[None, :] + D * t_enter[idx, None]
        T1, ok1 = refract(D, n_in[idx], 1.0 / sc.ior)

        okx, tnx, tfx, ninx, noutx = poly_enter_exit(
            E, T1, sc.glass_n, sc.glass_b)
        tfx = np.where(np.isfinite(tfx), tfx, 0.0)
        X = E + T1 * tfx[:, None]
        T2, ok2 = refract(T1, -noutx, sc.ior / 1.0)

        t2 = plane_hit(X, T2, sc.n_face, sc.p_face)
        Pf = X + T2 * t2[:, None]
        okpath = ok1 & ok2 & okx & (t2 > -1e-6)

        P_face[idx] = np.where(okpath[:, None], Pf, P_straight[idx])
        P_front[idx] = E
        P_back[idx] = X
        P_pdir[idx] = cam[None, :] + D * (t_enter[idx] + tfx + np.maximum(t2, 0.0))[:, None]
        path_len[idx] = t_enter[idx] + tfx + np.maximum(t2, 0.0)
        n_glass[idx] = n_in[idx]
        tir[idx] = ~okpath

    return dict(D=dirs, glass=glass, P_face=P_face, P_straight=P_straight,
                P_front=P_front, P_back=P_back, P_pdir=P_pdir,
                path_len=path_len, n_glass=n_glass, tir=tir)


def is_shadowed(P, sc):
    o = np.broadcast_to(P, (P.shape[0], 3))
    d = np.broadcast_to(-sc.light_dir[None, :], (P.shape[0], 3))
    valid, t1, t2, _, _ = poly_enter_exit(o, d,
                                          np.array([[1.0, 0, 0], [-1.0, 0, 0],
                                                    [0, 1.0, 0], [0, -1.0, 0],
                                                    [0, 0, 1.0], [0, 0, -1.0]]),
                                          np.array([sc.gobo_max[0], -sc.gobo_min[0],
                                                    sc.gobo_max[1], -sc.gobo_min[1],
                                                    sc.gobo_max[2], -sc.gobo_min[2]]))
    return valid & (t1 > 1e-5)


def face_albedo(P, sc):
    x, y = sc.face_point_at(P)
    bw, bh, m = sc.brick_w, sc.brick_h, sc.mortar
    stone = x > sc.seam_x
    row = np.floor(y / bh)
    off = np.mod(row, 2.0) * 0.5 * bw
    u = np.mod(x + off, bw)
    v = np.mod(y, bh)
    mort = (u < m) | (v < m)
    cid = (np.floor((x + off) / bw) * 7919.0 + row * 104729.0).astype(np.int64)
    h = np.mod(cid * 2654435761, 1000).astype(np.float64) / 1000.0
    brick = sc.brick_base[None, :] * (0.86 + 0.24 * h)[:, None]
    block = sc.stone_base[None, :] * (0.90 + 0.20 * h)[:, None]
    base = np.where(stone[:, None], block, brick)
    joint = np.where(stone[:, None], sc.stone_base[None, :] * 0.55,
                     sc.mortar_col[None, :])
    return np.where(mort[:, None], joint, base)


def sky_color(D, sc):
    t = np.clip(D[:, 1] * 1.6 + 0.45, 0.0, 1.0)[:, None]
    return sc.sky_zen[None, :] * t + sc.sky_hor[None, :] * (1.0 - t)


GUIDES = {
    "truth":       ("P_face",     "surface actually seen through the glass"),
    "glass_front": ("P_front",    "guide from the glass front face (primary-hit depth)"),
    "glass_back":  ("P_back",     "guide from the glass back face"),
    "straight":    ("P_straight", "guide from the unbent primary ray"),
    "primary_dir": ("P_pdir",     "hit-distance virtual point on the primary ray"),
}


def guide_point(tr, name):
    g = tr[GUIDES[name][0]]
    return np.where(np.isnan(g), tr["P_face"], g)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def render(pose, res, ss, guide, sc, chunk=150_000, shade=True):
    W, H = res
    n = W * H * ss * ss
    gy, gx = np.mgrid[0:H * ss, 0:W * ss]
    uv = np.stack([(gx.ravel() + 0.5) / ss, (gy.ravel() + 0.5) / ss], 1).astype(np.float64)

    col = np.zeros((n, 3))
    shcov = np.zeros(n)
    gmk = np.zeros(n)

    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        D = ray_dirs(pose, uv[sl], res, sc.fov_y)
        tr = trace(D, pose, sc)
        glass = tr["glass"]
        gmk[sl] = glass

        seen = tr["P_straight"] if guide == "air" else tr["P_face"]
        recv = tr["P_straight"] if guide == "air" else guide_point(tr, guide)
        sh = is_shadowed(recv, sc)
        shcov[sl] = sh
        if not shade:
            continue
        k = np.where(sh, sc.unlit, sc.lit)
        facecol = face_albedo(seen, sc) * k[:, None]
        c = facecol
        if guide != "air":
            g = np.nonzero(glass)[0]
            if g.size:
                nn = tr["n_glass"][g]
                inv = -dot(D[g], nn)
                F = sc.fresnel0 + (1.0 - sc.fresnel0) * np.power(
                    np.clip(1.0 - inv, 0.0, 1.0), 5.0)
                refl = sky_color(reflect(D[g], nn), sc)
                c = c.copy()
                c[g] = (F[:, None] * refl
                        + (1.0 - F)[:, None] * facecol[g] * sc.glass_tint[None, :])
        col[sl] = c

    img = col.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
    shm = shcov.reshape(H, ss, W, ss).mean(axis=(1, 3))
    gmk = gmk.reshape(H, ss, W, ss).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0) ** (1.0 / 2.2), shm, gmk
