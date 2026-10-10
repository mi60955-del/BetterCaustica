"""Fit the base coverage remap: threshold = 1 - k*c^gamma, width = w0 + w1*c_local.

The patch gate is a REDISTRIBUTION (zero-mean modulation of the local coverage) rather than an opacity
multiplier, so it no longer caps what the slider can deliver; the base curve is therefore fitted at
patchiness 0 and holds at any patchiness.
"""
import math

import numpy as np

from fit_remap import PX, PY, N, sample_n, DIR_CELL, DIR_JITTER, TYPE, fbm, smoothstep
from calibrate_remap import value_noise

CS = np.arange(0.2, 0.91, 0.05)
GAMMAS = (0.55, 0.7, 0.85, 1.0)


def gate_at(phase):
    out = np.empty(N)
    evx, evy = phase, phase * 0.618034
    for i in range(N):
        px, py = float(PX[i]), float(PY[i])
        ang = (value_noise(px / DIR_CELL + 3.0, py / DIR_CELL + 7.0) - 0.5) * (2.0 * DIR_JITTER)
        dx, dy = math.cos(ang), math.sin(ang)
        ppx, ppy = -dy, dx
        broad = 1.0 + (0.5 - 1.0) * TYPE
        sx, sy = px * broad, py * broad
        macro = fbm((sx * dx + sy * dy) / 2048.0 + evx * 0.06,
                    (sx * ppx + sy * ppy) / 1024.0 + evy * 0.06, 4)
        g = smoothstep(0.28, 0.78, macro)
        out[i] = g + (1.0 - g) * min(max((0.5 - 0.7) / 0.3, 0.0), 1.0)
    return out


def delivered(n, k, w0, w1, gamma, patchiness, gate):
    out = []
    for c in CS:
        if patchiness == 0.0:
            cloc = np.full(N, c)
        else:
            cloc = np.clip(c * (1.0 + patchiness * (gate - 0.5) * 2.0), 0.0, 1.0)
        thr = 1.0 - k * np.power(cloc, gamma)
        w = w0 + w1 * cloc
        cov = np.clip((n - thr) / np.maximum(w, 1e-3), 0.0, 1.0)
        out.append(float((cov > 0.5).mean()))
    return np.array(out)


if __name__ == "__main__":
    print("sampling %d points ..." % N)
    n = sample_n(0.0)
    gate = gate_at(0.0)
    print("E[gate] = %.3f (0.5 keeps the redistribution zero-mean)" % float(gate.mean()))

    best = None
    for gamma in GAMMAS:
        for k in (1.15, 1.45, 1.75, 2.05, 2.35):
            for w0 in (0.04, 0.08):
                for w1 in (0.16, 0.24, 0.32):
                    d = delivered(n, k, w0, w1, gamma, 0.0, None)
                    err = float(np.mean(np.abs(d - CS)))
                    if best is None or err < best[0]:
                        best = (err, k, w0, w1, gamma, d.copy())
    err, k, w0, w1, gamma, d = best
    print("BEST base: threshold = 1 - %.2f * c^%.2f, width = %.2f + %.2f*c   mean|err| %.3f"
          % (k, gamma, w0, w1, err))
    for c, f in zip(CS, d):
        print("   slider %.2f -> covered %.3f" % (c, f))
    d1 = delivered(n, k, w0, w1, gamma, 1.0, gate)
    print("patchiness 1.0 (redistribution):", " ".join("%.2f" % f for f in d1))
    print("slider                         :", " ".join("%.2f" % c for c in CS))
