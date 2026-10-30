#!/usr/bin/env python3
"""How much texture detail does the refracted mapping leave on screen?

A screen-space reconstructor (DLSS-RR's denoiser included) rebuilds a pixel from
a handful of neighbours.  That only works when neighbouring pixels describe
neighbouring points of the surface the colour came from.  A refracted primary
ray breaks exactly that: pixel `p` reads the destination point `P(p)`, which is
displaced by `d(p) = project(P(p)) - p`, and what matters for detail is not the
displacement but its gradient,

    J(p) = I + grad d(p)          destination screen pixels per screen pixel,

because a reconstruction kernel of radius `r` screen pixels gathers a disc of
radius `r * |J|` *destination* pixels.  |J| > 1 therefore minifies the texture:
one screen pixel has to carry |J| destination pixels, the detail is gone in a
single frame, and the denoiser can only smear what survives.  |J| ~ 1 means the
mapping is a one-to-one lift and nothing is lost -- which is the flush,
pane-parallel case, where the refracted exit point *is* the destination point.

Two things are reported per geometry: |J| (median/p95, and the fraction above
1.5) and the destination disc a 3-pixel kernel covers.  It also renders a fine
checker through the pane at 1 sample per pixel: where |J| > 1 the checker cannot
survive, and that is a still image, no motion required.
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe as P
import analysis as A

OUT = os.path.join(HERE, "out3")
RES = (520, 292)
EYE = np.array([-2.35, 0.48, -1.55])
TARGET = np.array([0.0, -0.05, 2.28])
CHECK_WORLD = 0.05          # checker cell in world units (~4 screen pixels)
KERNEL_R = 1.5              # DLSS-RR spatial kernel radius, screen pixels


def look_at(pos, target):
    d = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    d = d / np.linalg.norm(d)
    return P.Pose(pos, math.atan2(d[0], d[2]), math.asin(float(np.clip(d[1], -1, 1))))


def checker(Pw, sc):
    """A fine checker on the destination plane: max texture detail."""
    u = Pw @ sc.e1
    v = Pw @ sc.e2
    c = (np.floor(u / CHECK_WORLD) + np.floor(v / CHECK_WORLD)).astype(np.int64)
    on = (c % 2) == 0
    return np.where(on[:, None], np.array([[0.86, 0.86, 0.84]]),
                    np.array([[0.10, 0.10, 0.11]]))


def make_scene(kind, gap=0.0):
    """Destination geometries behind the same vertical pane.

    `gap` is the air between the pane's back face and a pane-parallel
    destination face; it is what turns the pane plus its air layer into a
    lens, and it is the one thing a full block touching the glass never has.
    """
    sc = P.Scene(gap=0.0)
    if kind == "flush":                     # full cube touching the pane
        pass                                # pane-parallel face, zero air gap
    elif kind == "gap":                     # same face, `gap` of air behind the pane
        sc.p_face = sc.p_face - sc.n_face * gap
        sc.face_gap = gap
    elif kind == "slab":                    # slab top: horizontal face behind the pane
        sc.n_face = np.array([0.0, 1.0, 0.0])
        sc.p_face = np.array([0.0, -0.55, sc.p_face[2]])
        up = np.array([0.0, 0.0, 1.0])
        sc.e1 = P.unit(np.cross(sc.n_face[None, :], up[None, :]))[0]
        sc.e2 = np.cross(sc.e1, sc.n_face)
        sc.face_gap = 0.0
    else:
        raise ValueError(kind)
    return sc


def pose_at(scale, target=TARGET, eye=EYE):
    """The default view pulled toward the pane by `scale` (1.0 = default)."""
    return look_at(target + (eye - target) * scale, target)


def density(pose, sc, res):
    """|J| per pane pixel plus the destination screen position."""
    uv = P.pixel_centers(res)
    D = P.ray_dirs(pose, uv, res, sc.fov_y)
    tr = P.trace(D, pose, sc)
    Pf = tr["P_face"]
    ok = tr["glass"] & ~tr["tir"] & np.isfinite(Pf).all(axis=1)
    dest = P.pinhole_project(Pf, pose, res, sc.fov_y)

    H, W = res[1], res[0]
    dx = np.full((H, W, 2), np.nan)
    dy = np.full((H, W, 2), np.nan)
    dd = dest.reshape(H, W, 2)
    dd = np.where(ok.reshape(H, W, 1), dd, np.nan)
    dx[:, 1:-1] = (dd[:, 2:] - dd[:, :-2]) * 0.5      # d/dx
    dy[1:-1, :] = (dd[2:, :] - dd[:-2, :]) * 0.5      # d/dy

    # J = I + grad(d) with d = dest - p, and grad(dest) already carries the
    # identity, so J is just the Jacobian of the destination screen position.
    J = np.zeros((H, W, 2, 2))
    J[:, :, 0, 0] = dx[:, :, 0]
    J[:, :, 0, 1] = dy[:, :, 0]
    J[:, :, 1, 0] = dx[:, :, 1]
    J[:, :, 1, 1] = dy[:, :, 1]

    inner = A.interior_mask(ok.reshape(H, W), 2)
    sel = inner & np.isfinite(J).all(axis=(2, 3))
    return dict(uv=uv, tr=tr, dest=dest, ok=ok, J=J, sel=sel, res=res)


def svals(J, sel):
    m = J[sel]
    s = np.linalg.svd(m, compute_uv=False)            # (N,2), descending
    return s[:, 0], s[:, 1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=f"{RES[0]}x{RES[1]}")
    ap.add_argument("--ss", type=int, default=1)
    a = ap.parse_args(argv)
    res = tuple(int(v) for v in a.res.lower().split("x"))
    os.makedirs(OUT, exist_ok=True)
    pose = look_at(EYE, TARGET)
    dist = float(np.linalg.norm(TARGET - EYE))
    px_world = res[1] / (2.0 * math.tan(make_scene("flush").fov_y * 0.5) * dist)

    rows = []
    tiles = []
    for kind, name in [("flush", "full block, flush"), ("gap", "same face, 0.30 air"),
                       ("slab", "slab top, horizontal")]:
        sc = make_scene(kind, gap=0.30) if kind == "gap" else make_scene(kind)
        d = density(pose, sc, res)
        smax, smin = svals(d["J"], d["sel"])
        frac = float(np.mean(smax > 1.5))
        foot = KERNEL_R * float(np.median(smax))
        rows.append(dict(kind=kind, name=name, n=int(d["sel"].sum()),
                         med=float(np.median(smax)), p95=float(np.percentile(smax, 95)),
                         mx=float(smax.max()), mmin=float(np.median(smin)),
                         frac=frac, foot=foot))

        # render the same scene with a maximum-detail checker, 1 spp
        sc.face_albedo = lambda Pw, sc=sc: checker(Pw, sc)
        rgb, cov, gm = P.render(pose, res, a.ss, "truth", sc)
        img = A.to_u8(np.clip(rgb, 0.0, 1.0))
        tiles.append(A.label(img, f"{name}   |J| med {rows[-1]['med']:.2f}", 14))

        # analytic 1:1 reference: the same checker, no glass in the way
        sc2 = make_scene(kind)
        sc2.face_albedo = lambda Pw, sc2=sc2: checker(Pw, sc2)
        sc2.glass_b = np.array([9e9, 9e9, 9e9, 9e9, 9e9, 9e9])   # disable the pane
        rgb2, _, _ = P.render(pose, res, a.ss, "truth", sc2)
        tiles.append(A.label(A.to_u8(np.clip(rgb2, 0, 1)), f"{name}   no pane", 14))

    print(f"resolution {res[0]}x{res[1]}, {a.ss} spp, "
          f"checker cell {CHECK_WORLD} world = {CHECK_WORLD*px_world:.1f} px")
    print(f"\n{'geometry':<22}{'|J| med':>9}{'p95':>7}{'max':>7}{'min':>7}"
          f"{'>1.5':>8}{'kernel footprint':>18}")
    for r in rows:
        print(f"{r['name']:<22}{r['med']:9.2f}{r['p95']:7.2f}{r['mx']:7.2f}"
              f"{r['mmin']:7.2f}{r['frac']*100:7.1f}%"
              f"{r['foot']:12.1f} dest px")

    sheet = A.contact_sheet(tiles, 2, pad=8)
    A.save(sheet, "sheet_warp.png", OUT)

    # ------------------------------------------------------------------ sweep
    # The pane plus the air behind it is a lens: its magnification depends on
    # that air gap and on how close the eye is, which is exactly the pair the
    # user's two observations point at (only geometry with air behind the pane,
    # worst when looking closely).
    gaps = [0.0, 0.05, 0.15, 0.30]
    scales = [1.0, 0.6, 0.35]
    sw = np.zeros((len(gaps), len(scales)))
    for i, g in enumerate(gaps):
        for j, s in enumerate(scales):
            sc = make_scene("gap", gap=g)
            dd = density(pose_at(s), sc, res)
            smax, _ = svals(dd["J"], dd["sel"])
            sw[i, j] = float(np.median(smax)) if smax.size else float("nan")

    print("\n|J| (destination px per screen px); >1 minifies, <1 magnifies")
    print("gap \\ eye" + "".join(f"{s:>9.2f}x" for s in scales))
    for i, g in enumerate(gaps):
        print(f"{g:5.2f}    " + "".join(f"{sw[i, j]:9.3f}" for j in range(len(scales))))
    print("magnification 1/|J| (screen px per destination px)")
    for i, g in enumerate(gaps):
        print(f"{g:5.2f}    " + "".join(f"{1.0 / sw[i, j]:9.2f}" for j in range(len(scales))))

    L2 = ["", "## 空气层 × 相机距离：镜片放大率", "",
          "`|J|` 是「每个屏幕像素覆盖多少目标面像素」。`|J| < 1` 表示画面被放大："
          "每个屏幕像素只前进不足一个目标面像素，纹理被拉伸到更多屏幕上，"
          "细节并没有增加——这是单帧的、与运动无关的软糊，来自玻璃与目标面之间那层空气。", "",
          "| 空气层 | " + " | ".join(f"眼距 {s:.2f}×" for s in scales) + " |",
          "|---|" + "---|" * len(scales)]
    for i, g in enumerate(gaps):
        L2.append(f"| {g:.2f} | " + " | ".join(f"{sw[i, j]:.3f}" for j in range(len(scales))) + " |")
    L2.append("")
    with open(os.path.join(OUT, "report_warp.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L2) + "\n")

    L = ["# 折射映射的屏幕密度：为什么静止时纹理也是糊的", "",
         f"相机 `{EYE.tolist()}` 看向 `{TARGET.tolist()}`，{res[0]}x{res[1]}，"
         f"{a.ss} spp，目标面贴 1/{CHECK_WORLD:g} 世界单位的棋盘格"
         f"（约 {CHECK_WORLD*px_world:.1f} 屏幕像素一格，接近采样极限）。", "",
         "`J(p) = I + ∇d(p)` 是「每个屏幕像素对应多少目标面屏幕像素」。"
         "半径 r 的重建核在目标面上覆盖 `r·|J|` 个像素。", "",
         "| 几何 | median \\|J\\| | p95 | max | min \\|J\\| | \\|J\\|>1.5 | 3 px 核覆盖 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['med']:.2f} | {r['p95']:.2f} | {r['mx']:.2f} |"
                 f" {r['mmin']:.2f} | {r['frac']*100:.1f}% | {r['foot']:.1f} dest px |")
    L += ["",
          "齐平（完整方块贴着玻璃）时 `|J| = 1.00`、核覆盖 1.5 px：折射出射点就是目标点，"
          "映射是 1:1 的，纹理一个像素都不丢。一旦目标面与玻璃之间夹了空气层"
          "（半砖顶面、方块实体的面），`|J|` 远离 1 且逐像素变化，"
          "同一块棋盘格在屏幕上就被压成糊/摩尔纹——这一步不需要任何时间项，"
          "它就是单帧里发生的细节丢失。", "",
          "示意图 `out3/sheet_warp.png`：左列透过玻璃、右列同一画面撤掉玻璃。"]
    with open(os.path.join(OUT, "report_warp.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nsheet -> {OUT}/sheet_warp.png\nreport -> {OUT}/report_warp.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
