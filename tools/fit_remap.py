"""Fit the coverage remap so the slider delivers what it says.

Per sample the field's n and the patch gate are computed ONCE; different remaps are then pure transforms
of those, so a sweep over (threshK, widthBase, widthK) is free. Target: the covered fraction AFTER the
gate equals the slider value, at the shipped default patchiness 1.0 (the gate then decides the clumping,
and lower patchiness over-delivers rather than under-delivers, which reads as a feature).

    python fit_remap.py
"""
import math

import numpy as np

from calibrate_remap import DIR_CELL, DIR_JITTER, PATCH, TYPE, fbm, hash12, smoothstep, value_noise

SPAN, STRIDE = 8192.0, 24.0
xs = np.arange(0.0, SPAN, STRIDE)
gx, gy = np.meshgrid(xs, xs)
PX = (gx + 137.0).ravel()
PY = (gy + 911.0).ravel()
N = PX.size

def sample_field(phase):
    evx, evy = phase, phase * 0.618034
    out_n = np.empty(N)
    out_gate = np.empty(N)
    for i in range(N):
        px, py = float(PX[i]), float(PY[i])
        ang = (value_noise(px / DIR_CELL + 3.0, py / DIR_CELL + 7.0) - 0.5) * (2.0 * DIR_JITTER)
        dx, dy = math.cos(ang), math.sin(ang)
        ppx, ppy = -dy, dx
        broad = 1.0 + (0.5 - 1.0) * TYPE
        sx, sy = px * broad, py * broad
        macro = fbm((sx * dx + sy * dy) / 2048.0 + evx * 0.06,
                    (sx * ppx + sy * ppy) / 1024.0 + evy * 0.06, 4)
        out_gate[i] = smoothstep(0.28, 0.78, macro)
        out_n[i] = macro  # placeholder, replaced below
    return out_n, out_gate

# n needs the warp + fbm; do it in the same loop for cache friendliness
def sample_n(phase):
    evx, evy = phase, phase * 0.618034
    out = np.empty(N)
    for i in range(N):
        px, py = float(PX[i]), float(PY[i])
        ang = (value_noise(px / DIR_CELL + 3.0, py / DIR_CELL + 7.0) - 0.5) * (2.0 * DIR_JITTER)
        dx, dy = math.cos(ang), math.sin(ang)
        ppx, ppy = -dy, dx
        broad = 1.0 + (0.5 - 1.0) * TYPE
        sx, sy = px * broad, py * broad
        macro = fbm((sx * dx + sy * dy) / 2048.0 + evx * 0.06,
                    (sx * ppx + sy * ppy) / 1024.0 + evy * 0.06, 4)
        cross_eff = 16.0 + (40.0 - 16.0) * macro
        cum_cell = max(120.0 * 1.1, 24.0)
        ca = 96.0 + (cum_cell - 96.0) * TYPE
        cc = cross_eff + (cum_cell - cross_eff) * TYPE
        qx = (sx * dx + sy * dy) / ca
        qy = (sx * ppx + sy * ppy) / cc
        qx += (value_noise(qx * 0.35 + 17.0 + evx, qy * 0.35 + 17.0 + evy) - 0.5) * 1.1
        qy += (value_noise(qx * 0.35 + 91.0 + evx, qy * 0.35 + 91.0 + evy) - 0.5) * 1.1
        out[i] = (fbm(qx, qy, 3)
                  - 0.07 * value_noise(qx * 0.25 + 7.0 + evx * 1.7, qy * 0.25 + 7.0 + evy * 1.7)
                  - 0.12 * (1.0 - TYPE) * value_noise(qx * 1.0 + 31.0 + evx * 2.3,
                                                      qy * 6.0 + 31.0 + evy * 2.3))
    return out

if __name__ == "__main__":
    print("sampling the field at %d points, two phases ..." % N)
    n1 = sample_n(0.0)
    n2 = sample_n(1.4)          # a later evolution state, so the fit holds as the sky churns
    g1 = 0.0 * n1               # gate computed inside sample_n would be handy; recompute cheaply:
    # the gate is a smoothstep of the SAME macro the n pass already saw - approximate it by re-deriving
    # from a second pass at the same positions.
    def gate_at(phase):
        out = np.empty(N)
        for i in range(N):
            px, py = float(PX[i]), float(PY[i])
            ang = (value_noise(px / DIR_CELL + 3.0, py / DIR_CELL + 7.0) - 0.5) * (2.0 * DIR_JITTER)
            dx, dy = math.cos(ang), math.sin(ang)
            ppx, ppy = -dy, dx
            broad = 1.0 + (0.5 - 1.0) * TYPE
            sx, sy = px * broad, py * broad
            macro = fbm((sx * dx + sy * dy) / 2048.0 + phase * 0.06,
                        (sx * ppx + sy * ppy) / 1024.0 + phase * 0.618034 * 0.06, 4)
            g = smoothstep(0.28, 0.78, macro)
            out[i] = g + (1.0 - g) * min(max((0.5 - 0.7) / 0.3, 0.0), 1.0)
        return out
    gate1 = gate_at(0.0)
    gate2 = gate_at(1.4)

    print()
    print("current remap (k=1.15, w0=0.10, w1=0.20) delivers:")
    for phase, n, gate in ((0.0, n1, gate1), (1.4, n2, gate2)):
        row = []
        for c in (0.2, 0.35, 0.5, 0.65, 0.8):
            thr = 1.0 - 1.15 * c
            w = 0.10 + 0.20 * c
            cov = np.clip((n - thr) / w, 0.0, 1.0) * gate
            row.append("%.3f" % float((cov > 0.5).mean()))
        print("   phase %.1f: c 0.20/0.35/0.50/0.65/0.80 -> %s" % (phase, " ".join(row)))

    print()
    print("sweep (k, w0, w1) -> mean |delivered - slider| over c in 0.2..0.9, patchiness 1.0:")
    CS = np.arange(0.2, 0.91, 0.1)
    best = None
    for k in (1.15, 1.45, 1.75, 2.05, 2.35):
        for w0 in (0.02, 0.06, 0.10):
            for w1 in (0.20, 0.32, 0.45):
                errs = []
                for phase, n, gate in ((0.0, n1, gate1), (1.4, n2, gate2)):
                    for c in CS:
                        thr = 1.0 - k * c
                        w = w0 + w1 * c
                        cov = np.clip((n - thr) / w, 0.0, 1.0) * gate
                        errs.append(float((cov > 0.5).mean()) - c)
                err = float(np.mean(np.abs(errs)))
                if best is None or err < best[0]:
                    best = (err, k, w0, w1)
                print("   k=%.2f w0=%.2f w1=%.2f  mean|err| %.3f" % (k, w0, w1, err))
    print()
    print("BEST: k=%.2f w0=%.2f w1=%.2f  mean|err| %.3f" % (best[1], best[2], best[3], best[0]))
    k, w0, w1 = best[1], best[2], best[3]
    for phase, n, gate in ((0.0, n1, gate1), (1.4, n2, gate2)):
        row = []
        for c in (0.2, 0.35, 0.5, 0.65, 0.8):
            thr = 1.0 - k * c
            w = w0 + w1 * c
            cov = np.clip((n - thr) / w, 0.0, 1.0) * gate
            row.append("%.3f" % float((cov > 0.5).mean()))
        print("   phase %.1f: -> %s" % (phase, " ".join(row)))
