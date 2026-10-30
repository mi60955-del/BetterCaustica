#!/usr/bin/env python3
"""Which depth does a non-cube object hand to the temporal denoiser?

Reported symptom: through glass, the *texture* of non-full blocks and block
entities is smeared by DLSS-RR, while full cubes are clean.

A full cube is the only Minecraft shape where the voxel cell and the visible
surface coincide.  Everything else -- slab, stairs, fence, pane, chest, sign,
banner, shulker -- shows a surface that is *inside* or *outside* its cell, or
is not in the voxel grid at all.  So any G-buffer depth that comes from a cell
(or that is missing) is exact for a full cube and wrong for the rest, which is
exactly the split the bug report describes.

This probe puts a full cube, a slab and a floating block entity in one scene
and compares four depth conventions for the reprojection that a temporal
denoiser performs:

  exact    - depth of the visible surface          (what the denoiser wants)
  cell     - depth of the bounding voxel cell      (cell == shape only for a cube)
  missing  - object absent, depth of the background behind it
  zero     - no reprojection at all (MV == 0)

Output: per-region smear after an exponential accumulation, for a strafing and
for a rotating camera, plus a sheet of the accumulated images.

A depth error cannot smear a *rotating* camera: the reprojected point stays on
the same primary ray, whose projection does not depend on the depth.  So
"smears while only turning" excludes every depth convention and points at a
zero/absent motion vector instead.  "Smears only while translating" is the
depth signature, and the per-object split tells which depth source is used.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analysis as A
import probe as P

OUT = os.path.join(HERE, "out2")
FOV_Y = math.radians(58.0)
RES = (600, 338)
CAM = np.array([2.55, 2.02, 0.42])
TARGET = np.array([2.60, 0.52, 4.70])
LIGHT = P.unit(np.array([[0.35, 0.82, 0.45]]))[0]
WALL_Z = 7.0
UP = np.array([0.0, 1.0, 0.0])


def look_at(pos, target):
    d = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    d = d / np.linalg.norm(d)
    return math.atan2(d[0], d[2]), math.asin(float(np.clip(d[1], -1.0, 1.0)))


class Obj:
    """Axis aligned box plus the voxel cell the renderer might report instead."""

    def __init__(self, name, lo, hi, cell_hi=None, tex=1.0):
        self.name = name
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(hi, np.float64)
        # cell_hi is the top of the containing voxel cell; None -> not in the grid.
        self.cell_hi = None if cell_hi is None else np.array(
            [self.lo[0], cell_hi, self.lo[2]], np.float64)
        self.cell_lo = np.array([self.lo[0], 0.0, self.lo[2]], np.float64)
        self.tex = float(tex)


OBJS = [
    # full cube: cell == shape, so every convention is exact here
    Obj("full", [0.85, 0.0, 3.85], [1.85, 1.0, 4.85], cell_hi=1.0),
    # slab: shape top at y=0.5, cell top at y=1.0
    Obj("slab", [2.85, 0.0, 3.85], [3.85, 0.5, 4.85], cell_hi=1.0, tex=1.0),
    # block entity: floats, has no cell at all -> a cell lookup finds nothing
    Obj("entity", [4.55, 0.55, 3.95], [5.35, 1.15, 4.75], tex=1.0),
]


def box_hit(lo, hi, O, D):
    """Nearest positive hit of a ray bundle with an AABB."""
    Ds = np.where(np.abs(D) < 1e-12, 1e-12, D)
    inv = 1.0 / Ds
    t0 = (lo[None, :] - O) * inv
    t1 = (hi[None, :] - O) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    tn = tmin.max(axis=1)
    tf = tmax.min(axis=1)
    ok = (tf >= np.maximum(tn, 0.0)) & (tn > 1e-6)
    idx = np.argmax(tmin, axis=1)
    rows = np.arange(D.shape[0])
    n = np.zeros_like(D)
    n[rows, idx] = -np.sign(Ds[rows, idx])
    return tn, ok, n


def plane_hit(nrm, b, O, D):
    den = D @ nrm
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = (b - O @ nrm) / den
    return t, (t > 1e-6)


def pattern(p, n):
    """Framed panel with stripes: a texture whose smear is easy to see."""
    ax = np.argmax(np.abs(n), axis=1)
    u = np.where(ax == 0, p[:, 1], p[:, 0])
    v = np.where(ax == 2, p[:, 1], p[:, 2])
    su = u * 2.0
    sv = v * 8.0
    stripe = 0.55 + 0.45 * np.cos(2.0 * math.pi * sv)
    frame = ((su % 1.0) < 0.16) | ((sv % 1.0) < 0.10)
    base = 0.34 + 0.30 * stripe
    val = np.where(frame, 0.13, base)
    return val[:, None] * np.array([[1.00, 0.86, 0.62]])


def wall_albedo(p):
    q = p * np.array([1.6, 1.6, 1.0])
    band = ((q[:, 0] % 1.0) < 0.9) & ((q[:, 1] % 0.5) < 0.45)
    return np.where(band[:, None], np.array([[0.42, 0.44, 0.50]]),
                    np.array([[0.30, 0.31, 0.35]]))


def trace(pose):
    """Trace one frame.  Returns everything the reprojection needs."""
    uv = P.pixel_centers(RES)
    D = P.ray_dirs(pose, uv, RES, FOV_Y)
    O = pose.pos
    n_px = D.shape[0]

    best_t = np.full(n_px, np.inf)
    best_n = np.zeros_like(D)
    best_id = np.full(n_px, -1, dtype=np.int32)
    t_cell = np.full(n_px, np.inf)

    for i, ob in enumerate(OBJS):
        t, ok, nrm = box_hit(ob.lo, ob.hi, O, D)
        m = ok & (t < best_t)
        best_t = np.where(m, t, best_t)
        best_id = np.where(m, i, best_id)
        best_n = np.where(m[:, None], nrm, best_n)
        if ob.cell_hi is not None:
            cell_hi = np.array([ob.hi[0], ob.cell_hi[1], ob.hi[2]], np.float64)
            tc, okc, _ = box_hit(ob.cell_lo, cell_hi, O, D)
            t_cell = np.where(m, np.where(okc, tc, np.inf), t_cell)
        else:
            t_cell = np.where(m, np.inf, t_cell)

    # floor and back wall
    tf, okf = plane_hit(UP, 0.0, O, D)
    m = okf & (tf > 1e-6) & (tf < best_t)
    best_t = np.where(m, tf, best_t)
    best_id = np.where(m, 100, best_id)
    best_n = np.where(m[:, None], UP[None, :] * np.sign(-D @ UP)[:, None], best_n)

    tw, okw = plane_hit(np.array([0.0, 0.0, 1.0]), WALL_Z, O, D)
    m = okw & (tw < best_t)
    best_t = np.where(m, tw, best_t)
    best_id = np.where(m, 101, best_id)
    best_n = np.where(m[:, None], np.array([0.0, 0.0, -1.0])[None, :], best_n)

    P_world = O[None, :] + D * best_t[:, None]
    alb = np.zeros((n_px, 3))
    is_obj = best_id < len(OBJS)
    alb[is_obj] = pattern(P_world[is_obj], best_n[is_obj])
    is_wall = best_id == 101
    alb[is_wall] = wall_albedo(P_world[is_wall])
    is_floor = best_id == 100
    alb[is_floor] = np.array([[0.26, 0.28, 0.26]])

    lam = np.clip(best_n @ LIGHT, 0.0, 1.0)
    shade = 0.22 + 0.78 * lam
    rgb = np.clip(alb * shade[:, None], 0.0, 1.0)
    rgb = np.where(is_wall[:, None], rgb * 0.85, rgb)

    # depth the background would report for this pixel (used by "missing")
    t_bg = np.where(okw, tw, np.inf)

    H, W = RES[1], RES[0]
    return dict(D=D, uv=uv, t=best_t, id=best_id, n=best_n, alb=alb,
                rgb=rgb.reshape(H, W, 3) ** (1.0 / 2.2), t_cell=t_cell,
                t_bg=t_bg, mask=(best_id.reshape(H, W)))


def obj_mask(fr, name):
    i = [o.name for o in OBJS].index(name)
    return fr["id"] == i


def interior(mask2d, m=3):
    g = mask2d.copy()
    for dy in (-m, 0, m):
        for dx in (-m, 0, m):
            g &= np.roll(np.roll(mask2d, dy, axis=0), dx, axis=1)
    return g


def mv_for(conv, fr, pose, prev_pose, off=None):
    """The motion vector a renderer would write for each pixel.

    `posmv` models history addressed with the *refracted* surface point instead
    of the point on the primary ray: that point carries the static refraction
    offset, so the vector stays nonzero even with a perfectly still camera.
    """
    if conv == "zero":
        return np.zeros_like(fr["uv"])
    if conv in ("exact", "posmv"):
        d = fr["t"]
    elif conv == "cell":
        d = np.where(np.isfinite(fr["t_cell"]), fr["t_cell"], fr["t_bg"])
    elif conv == "missing":
        d = np.where(fr["t"] < fr["t_bg"], fr["t_bg"], fr["t"])
    else:
        raise ValueError(conv)
    Q = pose.pos[None, :] + fr["D"] * np.minimum(d, 1e4)[:, None]
    mv = P.pinhole_project(Q, prev_pose, RES, FOV_Y) - fr["uv"]
    if conv == "posmv":
        # Only geometry with an air gap behind the pane carries the offset; a
        # face flush with the pane has none (measured 0.0 px in the slit probe).
        mv = mv + (off if off is not None else np.array([8.2, 0.0]))[None, :] \
            * (fr["id"] < len(OBJS))[:, None]
    return mv


def accumulate(conv, motion, frames, alpha, step, dps, off=None):
    yaw0, pitch0 = look_at(CAM, TARGET)
    poses = []
    for i in range(frames):
        if motion == "strafe":
            poses.append(P.Pose(CAM + np.array([step * i, 0.0, 0.0]), yaw0, pitch0))
        elif motion == "static":
            poses.append(P.Pose(CAM, yaw0, pitch0))
        else:
            poses.append(P.Pose(CAM, yaw0 + math.radians(dps) * i, pitch0))

    acc = None
    prev = None
    mvn = mve = None
    for i, pose in enumerate(poses):
        fr = trace(pose)
        cur = fr["rgb"]
        if acc is None:
            acc = cur.copy()
        else:
            mv = mv_for(conv, fr, pose, prev["pose"], off)
            mvx = mv_for("exact", fr, pose, prev["pose"])
            w = fr["uv"] + mv
            inb = ((w[:, 0] >= 0) & (w[:, 0] <= RES[0] - 1)
                   & (w[:, 1] >= 0) & (w[:, 1] <= RES[1] - 1))
            hist = A.sample_bilinear(acc, w).reshape(-1, 3)
            aa = np.where(inb, alpha, 1.0)[:, None]
            acc = (aa * cur.reshape(-1, 3) + (1.0 - aa) * hist).reshape(
                RES[1], RES[0], 3)
            # keep the error of the *last* frame so it shares the final masks
            mvn = np.linalg.norm(mv - mvx, axis=1)
            mve = np.linalg.norm(mvx, axis=1)
        prev = dict(pose=pose, fr=fr)
    return acc, prev["fr"]["rgb"], prev["fr"], mvn, mve


# Each object class only ever gets one depth convention in a real renderer:
# a full cube and a slab have a cell, a block entity does not.
REGIONS = [("full cube", "cell", "full", None),
           ("slab top", "cell", "slab", "top"),
           ("slab front", "cell", "slab", "front"),
           ("entity", "cell", "entity", None)]


def region_masks(fr):
    out = {}
    for label, conv, name, face in REGIONS:
        m = obj_mask(fr, name)
        if face == "top":
            m &= fr["n"][:, 1] > 0.7
        elif face == "front":
            m &= np.abs(fr["n"][:, 2]) > 0.7
        out[label] = interior(m.reshape(RES[1], RES[0])).ravel()
    return out


def region_rows(acc, cur, fr, mvn, mve, masks):
    rows = []
    d_all = np.abs(acc.reshape(-1, 3) - cur.reshape(-1, 3)).mean(axis=1) * 255.0
    for label, conv, name, face in REGIONS:
        inner = masks[label]
        if inner.sum() < 40:
            rows.append((label, conv, 0, 0.0, 0.0, 0.0))
            continue
        bias = float(np.median(mvn[inner]))
        flow = float(np.median(mve[inner]))
        eps = bias / flow if flow > 1e-6 else float("nan")
        rows.append((label, conv, int(inner.sum()), bias, eps,
                     float(d_all[inner].mean())))
    return rows


LAB = {"exact": "exact depth", "cell": "cell depth",
       "missing": "background depth", "zero": "MV = 0",
       "posmv": "MV from refracted point"}
# The conventions each camera motion can distinguish.  A still camera is the
# sharpest test of all: every *depth* error becomes an exactly zero vector
# there, so anything that still smears must be a nonzero vector on its own.
MODES = [("strafe", ["exact", "cell", "missing", "zero"]),
         ("turn", ["exact", "cell", "missing", "zero"]),
         ("static", ["exact", "posmv", "zero"])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--alpha", type=float, default=0.22)
    ap.add_argument("--step", type=float, default=0.055)
    ap.add_argument("--dps", type=float, default=0.9)
    ap.add_argument("--posmv-offset", type=float, nargs=2, default=[8.2, 0.0])
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    off = np.array(a.posmv_offset, np.float64)

    results = {}
    base = {}
    for motion, convs in MODES:
        tiles = []
        for conv in convs:
            acc, cur, fr, mvn, mve = accumulate(conv, motion, a.frames,
                                                a.alpha, a.step, a.dps, off)
            masks = region_masks(fr)
            rows = region_rows(acc, cur, fr, mvn, mve, masks)
            flow = float(np.median(mve[fr["id"] < len(OBJS)]))
            results[(motion, conv)] = dict(rows=rows, flow=flow)
            if conv == "exact":
                base[motion] = {r[0]: r[5] for r in rows}
            tiles.append(A.label(A.to_u8(acc), f"{motion}: {LAB[conv]}", 14))
        A.save(A.contact_sheet(tiles, 2, pad=8), f"sheet_{motion}.png", OUT)

    # ------------------------------------------------------------------ print
    for motion, convs in MODES:
        r0 = results[(motion, "exact")]
        print(f"\n[{motion}] content flow {r0['flow']:.2f} px/frame,"
              f" alpha {a.alpha:.2f} ({1/a.alpha:.0f} frame memory)")
        print(f"  {'convention':<10}{'region':<12}{'bias px':>9}{'eps':>8}"
              f"{'smear':>8}{'excess':>8}")
        for conv in convs:
            for label, c2, n, b, eps, smear in results[(motion, conv)]["rows"]:
                ex = smear - base[motion][label]
                e = "   n/a" if eps != eps else f"{eps*100:6.1f}%"
                print(f"  {conv:<10}{label:<12}{b:9.2f}{e:>8}"
                      f"{smear:8.2f}{ex:8.2f}")

    # ----------------------------------------------------------------- report
    L = ["# 非完整方块 / 方块实体：G-buffer 深度约定决定涂抹", "",
         f"场景：完整方块 + 半砖（shape 顶 y=0.5、cell 顶 y=1.0）+ 悬空方块实体（无 cell），"
         f"同一面砖墙背景，{RES[0]}x{RES[1]}，相机 `{CAM.tolist()}`。累积 "
         f"{a.frames} 帧、α={a.alpha}（约 {1/a.alpha:.0f} 帧记忆）、"
         f"平移 {a.step} world/帧 或 旋转 {a.dps}°/帧。", "",
         "每个物体类别在真实渲染器里只会拿到一种深度来源：完整方块与半砖有 voxel cell，"
         "方块实体没有 cell（查不到 → 退化成背景/远平面深度）。下表就是这个映射。", ""]
    for motion in ["strafe", "turn"]:
        r0 = results[(motion, "exact")]
        L.append(f"## {'平移' if motion == 'strafe' else '旋转'}"
                 f"（内容流速 {r0['flow']:.1f} px/帧）")
        L.append("")
        L.append("| 区域 | 深度来源 | MV 偏差 px | 占流速 ε | 涂抹(0-255) | 相对精确深度 |")
        L.append("|---|---|---|---|---|---|")
        for label, c2, n, b, eps, smear in results[(motion, "cell")]["rows"]:
            L.append(f"| {label} | cell | {b:.2f} | {eps*100:.1f}% | {smear:.2f} |"
                     f" {smear - base[motion][label]:+.2f} |")
        for label, c2, n, b, eps, smear in results[(motion, "zero")]["rows"]:
            L.append(f"| {label} | MV=0（最坏参考） | {b:.2f} | {eps*100:.1f}% |"
                     f" {smear:.2f} | {smear - base[motion][label]:+.2f} |")
        L.append("")

    L.append("## 判别法（游戏里两步）")
    L.append("")
    L.append("| 深度约定 | 平移时涂抹增量 | 只旋转时涂抹增量 |")
    L.append("|---|---|---|")
    for conv, name in [("exact", "精确深度"), ("cell", "cell 深度"),
                       ("missing", "背景深度"), ("zero", "MV=0")]:
        st = sum(r[5] - base["strafe"][r[0]] for r in results[("strafe", conv)]["rows"])
        tu = sum(r[5] - base["turn"][r[0]] for r in results[("turn", conv)]["rows"])
        L.append(f"| {name} | {st:+.2f} | {tu:+.2f} |")
    L.append("")
    L.append("深度类错误在**只旋转**时严格为 0：重投影点仍落在同一条主射线上，"
             "投影与深度无关（数学上恒等，不是近似）。所以：")
    L.append("")
    L.append("- 站着只转视角也在涂抹 → MV=0/缺失，不是深度；")
    L.append("- 只有走动/平移时涂抹 → 深度约定问题，且按上表看是哪一类几何。")
    L.append("")
    L.append("## 静止相机：只有“静止时 MV 也不为 0”的约定会涂抹")
    L.append("")
    L.append(f"相机完全不动，累积 {a.frames} 帧。此时精确流为 0，`MV=0` 也就是"
             f"正确答案——所以这张表里除 `posmv` 外都应该为 0。`posmv` 用"
             f"「历史取样点取折射命中点」的约定，静止时 MV 仍等于静态折射偏移 "
             f"({off[0]:.1f}, {off[1]:.1f}) px。")
    L.append("")
    L.append("| 区域 | 约定 | 静止时 MV px | 涂抹(0-255) | 相对精确 |")
    L.append("|---|---|---|---|---|")
    for label, c2, n, b, eps, smear in results[("static", "exact")]["rows"]:
        L.append(f"| {label} | 精确 / MV=0 | {b:.2f} | {smear:.2f} | 0.00 |")
    for label, c2, n, b, eps, smear in results[("static", "zero")]["rows"]:
        L.append(f"| {label} | MV=0 | {b:.2f} | {smear:.2f} | {smear:+.2f} |")
    for label, c2, n, b, eps, smear in results[("static", "posmv")]["rows"]:
        L.append(f"| {label} | 折射命中点 | {b:.2f} | {smear:.2f} | {smear:+.2f} |")
    L.append("")
    L.append("这一格就是判定用户那张截图的关键：如果画面**静止不动**时纹理就是糊的，"
             "那么问题不可能是深度约定（静止时深度误差恒等于 0 位移），"
             "只能是渲染器写进 MV 的向量本身不为 0——即历史取样点用的是折射后的位置，"
             "而不是主射线上的位置。")
    L.append("")
    L.append("## 换算到游戏速度")
    L.append("")
    L.append("拖影宽度 ≈ ε × 内容流速 / α。以行走 4.3 m/s、60 fps、物体距 3 格、"
             "屏幕流速约 40 px/帧 计：")
    L.append("")
    L.append("| 情形 | ε | 每帧 MV 偏差 | 累积拖影 |")
    L.append("|---|---|---|---|")
    for name, eps in [("完整方块 / cell 命中", 0.0),
                      ("半砖顶面（0.5 格 / 5 格）", 0.10),
                      ("方块实体退化成背景深度（约 2 格 / 5 格）", 0.40),
                      ("方块实体无深度（MV≈0）", 1.0)]:
        L.append(f"| {name} | {eps*100:.0f}% | {eps*40:.1f} px | {eps*40/a.alpha:.0f} px |")
    L.append("")
    L.append("示意图：`out2/sheet_strafe.png`、`out2/sheet_turn.png`、"
             "`out2/sheet_static.png`（累积后的最终帧）。")
    with open(os.path.join(OUT, "report_partial.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"\nsheets -> {OUT}/sheet_strafe.png, sheet_turn.png, sheet_static.png")
    print(f"report -> {OUT}/report_partial.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
