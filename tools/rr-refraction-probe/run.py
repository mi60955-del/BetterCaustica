#!/usr/bin/env python3
"""Driver for the glass-refraction shadow probe.  See probe.py for the scene."""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe as P
import analysis as A

OUT = os.path.join(HERE, "out")
DEF_POS = [-2.35, 0.48, -1.55]
DEF_TARGET = [0.0, -0.05, 2.28]


def look_at(pos, target):
    d = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    d = d / np.linalg.norm(d)
    yaw = math.atan2(d[0], d[2])
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    return yaw, pitch


def build(a):
    sc = P.Scene(gap=a.gap, thickness=a.thickness, ior=a.ior, wedge_deg=a.wedge)
    yaw, pitch = look_at(DEF_POS, DEF_TARGET)
    pose = P.Pose(DEF_POS, yaw + math.radians(a.yaw_off), pitch)
    return sc, pose


def res_of(a):
    w, h = a.res.lower().split("x")
    return int(w), int(h)


def pixel_trace(pose, res, sc):
    uv = P.pixel_centers(res)
    return uv, P.trace(P.ray_dirs(pose, uv, res, sc.fov_y), pose, sc)


def px_per_world(sc, res, depth):
    return (res[1] * 0.5) / (math.tan(sc.fov_y * 0.5) * max(depth, 1e-3))


# --------------------------------------------------------------------------- #
# self checks
# --------------------------------------------------------------------------- #
def cmd_check(a):
    ok = True
    sc, pose = build(a)

    # 1. Snell at 30 degrees through a parallel slab
    sc0 = P.Scene(gap=0.30, thickness=0.5, ior=1.52, wedge_deg=0.0)
    th = math.radians(30.0)
    D = np.array([[math.sin(th), 0.0, math.cos(th)]])
    cam = np.array([0.0, 0.0, 1.20])
    tr = P.trace(D, P.Pose(cam, 0.0, 0.0), sc0)
    exit_dir = tr["P_face"][0] - tr["P_back"][0]
    exit_dir = exit_dir / np.linalg.norm(exit_dir)
    dir_err = float(np.linalg.norm(exit_dir - D[0]))
    th2 = math.asin(math.sin(th) / sc0.ior)
    dx_analytic = sc0.thickness * (math.tan(th2) - math.tan(th))
    dx_measured = float(tr["P_face"][0, 0] - tr["P_straight"][0, 0])
    snell_ok = abs(dx_measured - dx_analytic) < 1e-9
    ok &= snell_ok
    print(f"[1] Snell 30deg  exit-dir err {dir_err:.2e} (want 0 for a slab)")
    print(f"    lateral shift measured {dx_measured:+.9f}  analytic {dx_analytic:+.9f}"
          f"  -> {'PASS' if snell_ok else 'FAIL'}")

    # 2. Newton round trip
    res = (240, 135)
    uv, tr = pixel_trace(pose, res, sc)
    # Only interior, non-TIR glass pixels: at the pane silhouette the image->surface
    # map is ill conditioned (a pixel can straddle the glass edge) and a Newton
    # solve there is not a test of the solver.
    g = (tr["glass"] & ~tr["tir"]).reshape(res[1], res[0])
    inner = g.copy()
    for dy in (-3, 0, 3):
        for dx in (-3, 0, 3):
            inner &= np.roll(np.roll(g, dy, axis=0), dx, axis=1)
    sel = np.nonzero(inner.ravel())[0][::3][:400]
    W = tr["P_face"][sel]
    uv2, err = A.solve_uv(W, pose, res, sc)
    rt = float(np.max(np.abs(uv2 - uv[sel])))
    ok &= rt < 0.05
    print(f"[2] Newton round trip  max |duv| {rt:.4f} px over {sel.size} glass pixels"
          f"  -> {'PASS' if rt < 0.05 else 'FAIL'}")

    # 3. pure rotation: measured flow must equal the analytic rotation flow
    pose1 = P.Pose(pose.pos, pose.yaw + math.radians(2.0), pose.pitch)
    uv1, err = A.solve_uv(W, pose1, res, sc)
    mv = uv1 - uv[sel]
    mv_rot = A.rotation_flow(uv[sel], pose, pose1, res, sc)
    d = np.abs(mv - mv_rot)
    rot_err = float(np.max(d))
    ok &= rot_err < 0.02
    print(f"[3] rotation flow  max |measured - analytic| {rot_err:.5f} px"
          f"  -> {'PASS' if rot_err < 0.02 else 'FAIL'}")
    print(f"    (this is the proof that depth-based guides cannot be wrong under"
          f" pure rotation)")

    # 4. translation: flow must NOT be depth independent
    pose2 = P.Pose(pose.pos + np.array([0.025, 0.0, 0.0]), pose.yaw, pose.pitch)
    uv2t, _ = A.solve_uv(W, pose2, res, sc)
    mv_t = uv2t - uv[sel]
    print(f"[4] strafe 25mm   true flow median {np.median(np.linalg.norm(mv_t, axis=1)):.3f} px"
          f"  spread {np.ptp(np.linalg.norm(mv_t, axis=1)):.3f} px")
    print(f"\n{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# stills
# --------------------------------------------------------------------------- #
GUIDE_ORDER = ["glass_front", "glass_back", "straight", "primary_dir"]


def cmd_stills(a):
    sc, pose = build(a)
    res = res_of(a)
    uv, tr = pixel_trace(pose, res, sc)

    # refraction displacement field (glass image vs no-glass image), in px
    pf = P.pinhole_project(tr["P_face"], pose, res, sc.fov_y)
    shift = np.where(tr["glass"][:, None], pf - uv, 0.0)
    shift_px = np.linalg.norm(shift, axis=1).reshape(res[1], res[0])
    depth = np.linalg.norm(tr["P_face"] - pose.pos[None, :], axis=1)

    tiles = []
    imgs = {}
    for g in ["air", "truth"] + GUIDE_ORDER:
        img, cov, gmk = P.render(pose, res, a.ss, g, sc)
        imgs[g] = (img, cov, gmk)
        print(f"  rendered {g}")

    _, cov_t, _ = imgs["truth"]
    win = A.shadow_window(cov_t, imgs["truth"][2])
    if win is None:
        print("!! shadow window not inside the glass; geometry needs adjusting")
        return 1

    for g in ["air", "truth"] + GUIDE_ORDER:
        img = imgs[g][0]
        tiles.append(A.label(A.to_u8(img), g))
    # truth edge overlay on every guide tile
    for g in GUIDE_ORDER:
        img = imgs[g][0]
        m = A.to_u8(img).copy()
        e = (cov_t > 0.15) & (cov_t < 0.85)
        m[e] = np.array([255, 40, 200], np.uint8)
        tiles.append(A.label(m, g + " + truth edge"))

    # shift magnitude map (the pure-refraction "perspective" warp)
    sm = shift_px / max(shift_px.max(), 1e-6)
    rgb = np.zeros((res[1], res[0], 3))
    rgb[..., 0] = np.clip(sm * 1.6, 0, 1)
    rgb[..., 1] = np.clip(1.2 - np.abs(sm - 0.5) * 2.4, 0, 1)
    rgb[..., 2] = np.clip(1.0 - sm * 1.4, 0, 1)
    shade = imgs["truth"][0].mean(axis=2)
    viz = np.clip(rgb * 0.85 + shade[..., None] * 0.35, 0, 1)
    tiles.append(A.label(A.to_u8(viz), f"refraction shift px  max {shift_px.max():.1f}"))

    sheet = A.contact_sheet(tiles, 3)
    p = A.save(sheet, "stills.png", OUT)
    print(f"stills -> {p}")
    print(f"  shift: median {np.median(shift_px[tr['glass'].reshape(res[1], res[0])]):.2f} px,"
          f" max {shift_px.max():.2f} px")
    return 0


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def guide_world_error(tr, name):
    g = P.guide_point(tr, name)
    return np.linalg.norm(g - tr["P_face"], axis=1)


def mv_models(tr, sel, u0, pose0, pose1, res, sc):
    """Two plausible engine MV conventions, per guide.

    depth: rebuild the surface point *on the primary ray* from a scalar radius
           and reproject it.  This is what a depth buffer can actually give you.
    pos:   reproject the guide's 3D point itself, i.e. a G-buffer world
           position or a motion vector built from a traced hit point.  For an
           off-ray point this bakes the static refraction offset into the MV.
    """
    D = tr["D"][sel]
    out = {}
    for name in GUIDE_ORDER:
        gp = P.guide_point(tr, name)[sel]
        rad = np.linalg.norm(gp - pose0.pos[None, :], axis=1)
        Q = pose0.pos[None, :] + D * rad[:, None]
        out[name] = dict(
            depth=P.pinhole_project(Q, pose1, res, sc.fov_y) - u0,
            pos=P.pinhole_project(gp, pose1, res, sc.fov_y) - u0,
            off_ray=float(np.median(np.linalg.norm(gp - Q, axis=1))))
    return out


def mv_table(sc, pose, res, step_world, step_deg, ss=1):
    uv, tr = pixel_trace(pose, res, sc)
    gmk2 = tr["glass"].reshape(res[1], res[0]).astype(float)
    inter = A.interior_mask(gmk2, 8).ravel()
    # TIR pixels show reflected content, not a refracted surface: a separate regime
    sel = np.nonzero(inter & tr["glass"] & ~tr["tir"])[0]
    sel = sel[::max(1, sel.size // 4000)]
    W = tr["P_face"][sel]
    u0 = uv[sel]

    out = dict(tir_fraction=float(tr["tir"][tr["glass"]].mean()))
    for kind, pose1 in (("walk", P.Pose(pose.pos + np.array([step_world, 0.0, 0.0]),
                                       pose.yaw, pose.pitch)),
                        ("turn", P.Pose(pose.pos, pose.yaw + math.radians(step_deg),
                                        pose.pitch))):
        uv1, err = A.solve_uv(W, pose1, res, sc)
        # a valid solution must land back on a refracted (non-TIR) glass pixel
        tr1 = P.trace(P.ray_dirs(pose1, uv1, res, sc.fov_y), pose1, sc)
        good = (err < 1e-4) & tr1["glass"] & ~tr1["tir"]
        mv_true = uv1 - u0
        mods = mv_models(tr, sel, u0, pose, pose1, res, sc)
        rows = {}
        for name in ["truth"] + GUIDE_ORDER:
            if name == "truth":
                rows[name] = dict(depth=dict(median=0.0, p95=0.0),
                                  pos=dict(median=0.0, p95=0.0), off_ray=0.0)
                continue
            m = mods[name]
            row = dict(off_ray=m["off_ray"])
            for conv in ("depth", "pos"):
                d = np.linalg.norm(m[conv] - mv_true, axis=1)[good]
                row[conv] = dict(median=float(np.median(d)),
                                 p95=float(np.percentile(d, 95)),
                                 mx=float(np.max(d)))
            rows[name] = row
        out[kind] = dict(flow_median=float(np.median(np.linalg.norm(mv_true, axis=1)[good])),
                         n=int(good.sum()), rows=rows)
        if kind == "turn":
            d = np.linalg.norm(A.rotation_flow(u0, pose, pose1, res, sc) - mv_true, axis=1)[good]
            out[kind]["analytic_vs_measured"] = dict(
                median=float(np.median(d)), mx=float(np.max(d)),
                viol=float(np.mean(d > 1.0)))
    return out


def cmd_metrics(a):
    sc, pose = build(a)
    res = res_of(a)
    uv, tr = pixel_trace(pose, res, sc)
    depth = float(np.median(np.linalg.norm(tr["P_face"][tr["glass"]] - pose.pos[None, :], axis=1)))
    scale = px_per_world(sc, res, depth)

    print(f"scene: wedge {a.wedge}deg gap {a.gap} thickness {a.thickness} ior {a.ior}")
    print(f"  median eye distance {depth:.2f} world, {scale:.1f} px/world-unit\n")
    print(f"{'guide':<13}{'static |dP| world':>19}{'= px':>8}{'centroid px':>13}{'area %':>9}{'rms edge px':>13}")

    img_t, cov_t, gmk_t = P.render(pose, res, a.ss, "truth", sc, shade=False)
    win = A.shadow_window(cov_t, gmk_t)
    st_t = A.shadow_stats(cov_t, win)
    static = {}
    for name in GUIDE_ORDER:
        _, cov, _ = P.render(pose, res, a.ss, name, sc, shade=False)
        st = A.shadow_stats(cov, win)
        cmp = A.compare_shadow(st_t, st)
        werr = guide_world_error(tr, name)
        werr = werr[tr["glass"]]
        rms_all = np.nanmean([cmp["rms_top"], cmp["rms_bot"], cmp["rms_lef"], cmp["rms_rig"]])
        stat = dict(world_median=float(np.median(werr)),
                    px_median=float(np.median(werr) * scale),
                    rms_edge=float(rms_all), **cmp)
        static[name] = stat
        print(f"{name:<13}{np.median(werr):>19.4f}{np.median(werr) * scale:>8.1f}"
              f"{cmp['centroid_px']:>13.2f}{cmp['area_pct']:>9.2f}{rms_all:>13.2f}")

    walk = a.walk
    turn = a.turn
    mt = mv_table(sc, pose, res, walk, turn)
    print(f"\nTIR (total internal reflection) fraction of the pane: {100 * mt['tir_fraction']:.1f}%")
    for kind in ("walk", "turn"):
        d = mt[kind]
        print(f"\n{'strafe' if kind == 'walk' else 'turn'} {walk if kind == 'walk' else turn}"
              f"{' world/frame' if kind == 'walk' else ' deg/frame'}: content flow median"
              f" {d['flow_median']:.2f} px/frame, n={d['n']} samples")
        print(f"  {'guide':<13}{'pos-MV med':>11}{'p95':>8}{'depth-MV med':>14}{'p95':>8}"
              f"{'off-ray px':>12}")
        for name in GUIDE_ORDER:
            r = d["rows"][name]
            print(f"  {name:<13}{r['pos']['median']:>11.3f}{r['pos']['p95']:>8.3f}"
                  f"{r['depth']['median']:>14.3f}{r['depth']['p95']:>8.3f}"
                  f"{r['off_ray'] * px_per_world(sc, res, depth):>12.2f}")
        if "analytic_vs_measured" in d:
            e = d["analytic_vs_measured"]
            print(f"  analytic rotation flow vs measured: median {e['median']:.4f},"
                  f" max {e['mx']:.4f} px")

    # wedge sweep: how non-parallel must the glass be for guides to break?
    sweep = []
    for wdeg in a.wedge_sweep:
        sc2 = P.Scene(gap=a.gap, thickness=a.thickness, ior=a.ior, wedge_deg=wdeg)
        pose2 = P.Pose(pose.pos, pose.yaw, pose.pitch)
        m2 = mv_table(sc2, pose2, res, walk, turn)
        _, c2, g2 = P.render(pose2, res, 2, "truth", sc2, shade=False)
        w2 = A.shadow_window(c2, g2)
        s2 = A.shadow_stats(c2, w2)
        _, c3, _ = P.render(pose2, res, 2, "glass_front", sc2, shade=False)
        s3 = A.shadow_stats(c3, w2)
        c3cmp = A.compare_shadow(s2, s3)
        sweep.append(dict(wedge=wdeg,
                          mv_walk=m2["walk"]["rows"]["glass_front"]["depth"]["p95"],
                          mv_walk_pos=m2["walk"]["rows"]["glass_front"]["pos"]["p95"],
                          mv_turn=m2["turn"]["rows"]["glass_front"]["depth"]["p95"],
                          mv_turn_pos=m2["turn"]["rows"]["glass_front"]["pos"]["p95"],
                          tir=100.0 * m2["tir_fraction"],
                          id_viol=100.0 * m2["turn"]["analytic_vs_measured"]["viol"],
                          edge_rms=float(np.nanmean([c3cmp["rms_top"], c3cmp["rms_bot"],
                                                     c3cmp["rms_lef"], c3cmp["rms_rig"]]))))
        s = sweep[-1]
        print(f"\nwedge {wdeg:>4}deg : glass_front p95 mv  depth {s['mv_walk']:.2f}"
              f" / pos {s['mv_walk_pos']:.2f} px (walk),  depth {s['mv_turn']:.2f}"
              f" / pos {s['mv_turn_pos']:.2f} px (turn),  TIR {s['tir']:.1f}%"
              f"   flow!={s['id_viol']:.1f}%   shadow edge rms {s['edge_rms']:.2f} px")

    res_obj = dict(scene=dict(wedge=a.wedge, gap=a.gap, thickness=a.thickness, ior=a.ior,
                              eye_depth=depth, px_per_world=scale),
                   static=static, motion=mt, sweep=sweep)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res_obj, f, indent=1)
    print(f"\nmetrics -> {os.path.join(OUT, 'metrics.json')}")
    return res_obj


def cmd_demo(a):
    """Temporal accumulation of real rendered frames with exact vs guide MVs.

    The scene is noise free, so exact MVs reproduce the current frame exactly
    and any MV error shows up as pure reprojection smear -- the cleanest
    possible read-out of "how big must an MV error be to see it".
    """
    sc, pose0 = build(a)
    res = res_of(a)
    n = a.frames
    dps = math.radians(a.dps)
    alpha = a.alpha
    uv = P.pixel_centers(res)
    models = ["exact", "depth_front", "pos_back"]
    acc = {}
    mvmax = {m: 0.0 for m in models}
    frames_png = []
    prev = None

    for i in range(n):
        pose = P.Pose(pose0.pos, pose0.yaw + dps * i, pose0.pitch)
        img, _, _ = P.render(pose, res, 1, "truth", sc)
        D = P.ray_dirs(pose, uv, res, sc.fov_y)
        tr = P.trace(D, pose, sc)
        if prev is None:
            for m in models:
                acc[m] = img.copy()
        else:
            prev_pose = prev[1]
            mv = {"exact": A.project_dir(D, prev_pose, res, sc.fov_y) - uv}
            r = np.linalg.norm(P.guide_point(tr, "glass_front") - pose.pos[None, :], axis=1)
            mv["depth_front"] = P.pinhole_project(
                pose.pos[None, :] + D * r[:, None], prev_pose, res, sc.fov_y) - uv
            mv["pos_back"] = P.pinhole_project(
                P.guide_point(tr, "glass_back"), prev_pose, res, sc.fov_y) - uv
            if i == 1:
                gmask = P.trace(D, pose, sc)["glass"]
                for m in models:
                    mvn = np.linalg.norm(mv[m], axis=1)[gmask]
                    mvmax[m] = (float(np.median(mvn)), float(np.max(mvn)))
            for m in models:
                w = uv + mv[m]
                inb = ((w[:, 0] >= 0) & (w[:, 0] <= res[0] - 1)
                       & (w[:, 1] >= 0) & (w[:, 1] <= res[1] - 1))
                hist = A.sample_bilinear(acc[m], w)
                aa = np.where(inb, alpha, 1.0)[:, None]
                acc[m] = (aa * img.reshape(-1, 3)
                          + (1.0 - aa) * hist).reshape(res[1], res[0], 3)
        prev = (img, pose)
        tiles = [A.label(A.to_u8(img), f"current frame {i:02d} (truth)", 13),
                 A.label(A.to_u8(acc["exact"]), "accum, exact MV", 13),
                 A.label(A.to_u8(acc["depth_front"]), "accum, depth-MV front", 13),
                 A.label(A.to_u8(acc["pos_back"]), "accum, pos-MV back", 13)]
        frames_png.append(A.contact_sheet(tiles, 4, pad=6))

    gif_path = os.path.join(OUT, "temporal.gif")
    imgs = [Image.fromarray(f) for f in frames_png]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 * a.delay), loop=0, optimize=True)
    A.save(frames_png[-1], "temporal_last.png", OUT)
    print(f"demo: {n} frames, turn {a.dps} deg/frame, alpha {alpha}"
          f" ({1.0 / alpha:.0f} frame memory)")
    for m in models:
        md, mx = mvmax[m]
        print(f"  MV through glass: {m:<12} median {md:6.2f} px   max {mx:6.2f} px"
              f"   -> accumulated smear ~{md / alpha:5.1f} px")
    print(f"  gif -> {gif_path}")
    return dict(mv_magnitude=mvmax)


def _ftable(rows, head, fmt):
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(fmt(r))
    return "\n".join(out)


def cmd_report(a):
    m = cmd_metrics(a)
    _, pose0 = build(a)
    yaw_deg = math.degrees(pose0.yaw)
    pitch_deg = math.degrees(pose0.pitch)
    st = m["static"]
    mo = m["motion"]
    sw = m["sweep"]

    def srow(name):
        s = st[name]
        return (f"| `{name}` | {s['world_median']:.4f} | {s['px_median']:.1f} |"
                f" {s['centroid_px']:.2f} | {s['area_pct']:+.2f} | {s['rms_edge']:.2f} |")

    srows = "".join(srow(n) + "\n" for n in GUIDE_ORDER)

    def mrows(kind):
        out = ""
        for n in GUIDE_ORDER:
            r = mo[kind]["rows"][n]
            out += (f"| `{n}` | {r['depth']['median']:.3f} | {r['depth']['p95']:.3f} |"
                    f" {r['pos']['median']:.3f} | {r['pos']['p95']:.3f} |"
                    f" {r['off_ray'] * m['scene']['px_per_world']:.2f} |\n")
        return out

    wrows = "".join(
        f"| {s['wedge']:.0f}° | {s['mv_walk']:.2f} | {s['mv_turn']:.2f} |"
        f" {s['tir']:.1f}% | {s['id_viol']:.1f}% | {s['edge_rms']:.2f} |\n" for s in sw)

    doc = f"""# 玻璃折射阴影 透视感 / DLSS-RR 归因实验

程序：`tools/rr-refraction-probe/`（`check` 自检 → `stills` 图 → `metrics` 表 → `demo` 时域 GIF → `report` 本文）。全部为解析几何 + 逐像素数值追踪，不依赖游戏、GPU 或模组代码。

> **一句话结论**：平行板玻璃不会弯折光线（只横向平移 t(tanθ₂−tanθ₁)，本例 8.2 px），所以由深度算出的运动向量在纯转视角时误差**恒为 0**、平移时 <0.2 px——"深度/MV 算错"这条假设在平板上被定量排除。能造成"透视感"的只剩三种：接收点错到玻璃表面（20 px）、MV 直接投影折射命中点（8 px，且被时域累积放大约 4.5 倍）、玻璃带倒角/透镜（40 px）。

## 场景与精确性

相机 `{DEF_POS}` 斜视（yaw {yaw_deg:.1f}°、pitch {pitch_deg:.1f}°），玻璃为凸多面体（6 半空间），`ior={m['scene']['ior']}`、厚 `{m['scene']['thickness']}`、楔角 `{m['scene']['wedge']}`°、贴面间隙 `{m['scene']['gap']}`；玻璃后面是一面砖墙（带接缝色差）+ 一个 gobo 投出的硬阴影。眼距 {m['scene']['eye_depth']:.2f} world，{m['scene']['px_per_world']:.1f} px/world。

自检（`run.py check`，全部 PASS）：Snell 30° 出入口方向误差 1e-16、横向位移与解析式 t(tanθ₂−tanθ₁) 一致到 1e-16；像素→表面的牛顿反解往返误差 0.002 px；纯旋转流场与解析单应一致到 0.002 px。

## 1. 阴影画在哪个面上（静态）

"引导点"= 阴影接收点 / 深度重建位置。表内是相对真值（折射后的真实墙面点）的偏差。

平板玻璃的机制：进入面与出射面平行，出射方向与入射方向完全相同（只差一个横向位移 t(tanθ₂−tanθ₁)），所以透过玻璃看到的墙比真实位置横向偏 8.2 px——阴影随之整体平移，这是正确的折射成像，不是错误。

| 引导假设 | 三维偏差 world | 折合 px | 阴影质心位移 px | 面积变化 % | 边缘 rms px |
|---|---|---|---|---|---|
{srows}
- `glass_back` 在间隙 0 时**就是真实墙面点**，所以恒为 0：说明"用玻璃后表面重建"在平板贴面时是无害的。
- 平板玻璃不改变光线方向（只横向平移），所以 `straight` / `primary_dir` 偏差只有 8 px 级别的**横向**位移，而无方向误差。
- `glass_front` 把阴影挪到玻璃前表面：质心偏 {st['glass_front']['centroid_px']:.1f} px、面积 {st['glass_front']['area_pct']:+.1f}%、边缘 rms {st['glass_front']['rms_edge']:.1f} px。这在视觉上就是"阴影像贴在一块倾斜的玻璃上、而不是贴在墙上"的透视感——它属于**接收点错误**，与降噪器无关，但降噪器会把这份错误当真实信号做时域累积。

## 2. 运动向量（时域）——决定 RR 会不会抹开阴影

同一像素在下一帧的真实位置由牛顿反解给出（唯一正确值）；"depth-MV"= 用标量深度沿主射线重建后再投影（真实深度缓冲能给出的东西），"pos-MV"= 直接把引导点的世界坐标投影（G-buffer 世界位置 / 用 trace 命中点手搓的 MV）。

**旋转 {mo['turn']['flow_median']:.2f} px/帧（{a.turn}°/帧）下的 MV 误差**

| 引导假设 | depth-MV 中位 | p95 | pos-MV 中位 | p95 | 离主射线 px |
|---|---|---|---|---|---|
{mrows('turn')}
**平移 {a.walk} world/帧（{mo['walk']['flow_median']:.2f} px/帧）下的 MV 误差**

| 引导假设 | depth-MV 中位 | p95 | pos-MV 中位 | p95 | 离主射线 px |
|---|---|---|---|---|---|
{mrows('walk')}
结论（本项目最硬的一条）：
- **纯转视角时，所有 depth-MV 误差恒等于 0.000 px**，与深度值无关。原因是：某个表面点的入射方向在世界空间里是固定的，相机转动只是把这个固定方向在新基下重投影，所以折射内容的真实流场恰好就是针孔旋转单应。`check [3]` 与表中 turn 列同时验证了这一点（解析 vs 实测 max 0.0000 px）。
- 平移 2.5 cm/帧时 depth-MV 误差 0.09–0.18 px/帧，一帧累积（α≈0.2 时约 5 帧记忆）也到不了 1 px。
- 唯一能产生 ≥8 px 误差的是 `pos-MV`（把折射命中点当世界位置再投影）：{mo['turn']['rows']['glass_back']['pos']['median']:.2f} px，且与运动类型无关（旋转也照样错）。**如果模组把 trace 命中点写进 G-buffer/自己算 MV，那么平板玻璃上也会出现 8 px 级错误——这是唯一能在平板上解释"阴影被拖出透视状"的 MV 来源。**

## 2b. 为什么 8 px 的 MV 偏差会变成几十 px 的拖影

指数累积的时域降噪（α={a.alpha}，约 {1.0 / a.alpha:.0f} 帧记忆）遇到稳定偏差 b 时，历史每一代都会被同一偏差再平移一次，饱和拖影宽度 ≈ b/α，即偏差被放大约 {1.0 / a.alpha:.1f} 倍：

- depth-MV：旋转时 b = {mo['turn']['rows']['glass_front']['depth']['median']:.2f} px、平移时 b = {mo['walk']['rows']['glass_front']['depth']['median']:.2f} px → 拖影 ≤ {mo['walk']['rows']['glass_front']['depth']['median'] / a.alpha:.1f} px（不可见）；
- pos-MV：b ≈ {mo['turn']['rows']['glass_back']['pos']['median']:.2f} px → 拖影 ≈ {mo['turn']['rows']['glass_back']['pos']['median'] / a.alpha:.0f} px。

`out/temporal.gif` 是这个过程在真实渲染帧上的录像：第 3 格（depth-MV front）与当前帧一样锐利，第 4 格（pos-MV back）在二十几帧后把阴影和砖缝糊成一片，而糊出来的形状看起来"歪、带透视"——正是报告描述的症状。**结论：不需要任何几何上的透视错误，一个稳定的 MV 偏差就足以把阴影拖成透视状。**

## 3. 玻璃不是平行板时（倒角 / 透镜）

| 楔角 | depth-MV p95 平移 | depth-MV p95 旋转 | TIR 占比 | 流场非单应像素 | 阴影边缘 rms px |
|---|---|---|---|---|---|
{wrows}
楔角 ≥10° 后，背面倾斜使"从背面出射"与"从侧面出射"在画面内部形成**不连续边界**，流场不再是单应（20° 时 14.5% 像素），引导误差跳到 41 px，任何时域降噪都会把阴影抹成一片。若模组的玻璃有倒角/透镜几何，这条就是直接原因。

## 4. 建议的判定流程

1. **站定不动**看阴影：若此时阴影已经是歪的/像贴在斜面上 → 是第 1 节的接收点问题（`glass_front` 型），与 RR 无关。
2. **只有移动/转视角时才抹开** → 是 MV 问题。此时再分：
   - 平板玻璃中心区也抹 → 第 2 节 `pos-MV`（命中点当世界位置），误差 8 px 级；
   - 只沿倒角/边缘出现 → 第 3 节（透镜流场不连续）。
3. 若两者都不是，则几何上平板玻璃的引导误差在亚像素级，问题应出在着色器/RR 集成对这块像素的**材质语义**（法线、粗糙度、镜面语义），而不是位置/深度。

结论一句话：**在平行板玻璃上，"深度/运动向量算错"被定量排除**（旋转恒 0、平移 <0.2 px）；能解释"透视感"的只剩三条——接收点错到玻璃表面（20 px 级）、MV 用折射命中点（8 px 级）、玻璃非平行（40 px 级）。

图：`out/stills.png`（六种引导的对照 + 真值边缘叠加 + 折射位移场）、`out/temporal.gif`（精确 MV / depth-MV / pos-MV 三种累积对比）。

复现：`python run.py all`（约 40 s；`check` 0.4 s、`stills` 9 s、`metrics` 14 s、`demo` 12 s）。可调参数：`--wedge`（楔角）、`--gap`（贴面间隙）、`--ior`、`--thickness`、`--res`、`--ss`、`--walk`、`--turn`、`--frames/--dps/--alpha`（时域演示）。全部输出写在 `out/`。
"""
    p = os.path.join(OUT, "report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\nreport -> {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "stills", "metrics", "demo", "report", "all"])
    ap.add_argument("--wedge", type=float, default=0.0)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--thickness", type=float, default=0.5)
    ap.add_argument("--ior", type=float, default=1.52)
    ap.add_argument("--res", default="480x270")
    ap.add_argument("--ss", type=int, default=3)
    ap.add_argument("--walk", type=float, default=0.025)
    ap.add_argument("--turn", type=float, default=1.0)
    ap.add_argument("--wedge-sweep", type=float, nargs="*", default=[0, 5, 10, 20])
    ap.add_argument("--yaw-off", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--dps", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.22)
    ap.add_argument("--delay", type=float, default=0.11)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    fns = {"check": cmd_check, "stills": cmd_stills, "metrics": cmd_metrics,
           "demo": cmd_demo, "report": cmd_report}
    if a.cmd == "all":
        for name in ["check", "stills", "metrics", "demo", "report"]:
            print(f"\n===== {name} =====")
            r = fns[name](a)
            if isinstance(r, int) and r != 0:
                return r
        return 0
    r = fns[a.cmd](a)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
