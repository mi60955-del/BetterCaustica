"""Calibrate the cloud coverage remap against the CURRENT field.

The remap's constants (threshold 1 - 1.15c, width 0.10 + 0.20c) were tuned for the stratus-lattice field
of v22. The cumulus rework (v30) changed the lattice the FBM runs on, which shifts the *distribution* of
n, so the coverage slider no longer means what it says - at 50 % the sky is mostly empty, and the patch
gate multiplies what is left down to a few threads.

This measures the covered fraction of the live field over a large area, per coverage setting, and reports
what the slider actually delivers.

    python calibrate_remap.py
"""
import math

import numpy as np

# ---- the shader's noise, verbatim -------------------------------------------------------------------
def hash12(px, py):
    p3x = (px * 0.1031) % 1.0
    p3y = (py * 0.1031) % 1.0
    p3z = (px * 0.1031) % 1.0
    d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33)
    p3x += d; p3y += d; p3z += d
    return ((p3x + p3y) * p3z) % 1.0

def value_noise(px, py):
    ix, iy = math.floor(px), math.floor(py)
    fx, fy = px - ix, py - iy
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    a = hash12(ix, iy)
    b = hash12(ix + 1, iy)
    c = hash12(ix, iy + 1)
    d = hash12(ix + 1, iy + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def fbm(px, py, octaves):
    s, amp, norm = 0.0, 0.5, 0.0
    for _ in range(octaves):
        s += amp * value_noise(px, py)
        norm += amp
        px, py = px * 2.0 + 31.0, py * 2.0 + 17.0
        amp *= 0.5
    return s / norm

def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)

# ---- the live field (volume.slang), with the constants read off the shader --------------------------
ALONG, CROSS, CROSS_MAX = 96.0, 16.0, 40.0
DIR_JITTER, DIR_CELL = 0.17, 900.0
BROADEN = 0.5
TYPE = 0.7          # the user's cloud-type
PATCH = 1.0         # the user's cloud-patchiness
THICK = 120.0       # the user's cloud-thickness
CUMULUS_CELL = max(THICK * 1.1, 24.0)

def field(px, py, c, phase=0.0, patchiness=PATCH):
    evx, evy = phase, phase * 0.618034
    ang = 0.0 + (value_noise(px / DIR_CELL + 3.0, py / DIR_CELL + 7.0) - 0.5) * (2.0 * DIR_JITTER)
    dx, dy = math.cos(ang), math.sin(ang)
    ppx, ppy = -dy, dx
    s = math.hypot(0, 0)  # placeholder; broaden scales p below
    broad = 1.0 + (BROADEN - 1.0) * TYPE
    sx, sy = px * broad, py * broad
    macro = fbm((sx * dx + sy * dy) / 2048.0 + evx * 0.06,
                (sx * ppx + sy * ppy) / 1024.0 + evy * 0.06, 4)
    cross_eff = CROSS + (CROSS_MAX - CROSS) * macro
    cell_along = ALONG + (CUMULUS_CELL - ALONG) * TYPE
    cell_cross = cross_eff + (CUMULUS_CELL - cross_eff) * TYPE
    qx = (sx * dx + sy * dy) / cell_along
    qy = (sx * ppx + sy * ppy) / cell_cross
    qx += (value_noise(qx * 0.35 + 17.0 + evx, qy * 0.35 + 17.0 + evy) - 0.5) * 1.1
    qy += (value_noise(qx * 0.35 + 91.0 + evx, qy * 0.35 + 91.0 + evy) - 0.5) * 1.1
    n = (fbm(qx, qy, 3)
         - 0.07 * value_noise(qx * 0.25 + 7.0 + evx * 1.7, qy * 0.25 + 7.0 + evy * 1.7)
         - 0.12 * (1.0 - TYPE) * value_noise(qx * 1.0 + 31.0 + evx * 2.3,
                                             qy * 6.0 + 31.0 + evy * 2.3))
    gate = smoothstep(0.28, 0.78, macro)
    gate = gate + (1.0 - gate) * min(max((c - 0.7) / 0.3, 0.0), 1.0)
    cov = min(max((n - (1.0 - 1.15 * c)) / max(0.10 + 0.20 * c, 1e-3), 0.0), 1.0)
    return cov * (1.0 - patchiness + patchiness * gate)

def covered_fraction(c, phase, patchiness, span=8192.0, stride=32.0):
    xs = np.arange(0.0, span, stride)
    gx, gy = np.meshgrid(xs, xs)
    gx, gy = gx.ravel() + 137.0, gy.ravel() + 911.0   # off-lattice offsets
    vec_field = np.vectorize(lambda x, y: field(x, y, c, phase, patchiness))
    v = vec_field(gx, gy)
    return float((v > 0.5).mean()), float((v > 0.1).mean()), float(v.mean())

if __name__ == "__main__":
    print("coverage slider -> actually covered fraction of the sky (cov > 0.5), by patchiness")
    for patch in (1.0, 0.0):
        print("--- patchiness %.1f ---" % patch)
        for c in (0.2, 0.35, 0.5, 0.65, 0.8):
            f50, f10, mean = covered_fraction(c, phase=0.0, patchiness=patch)
            print("  c=%.2f  frac>0.5 %.3f  frac>0.1 %.3f  mean %.3f" % (c, f50, f10, mean))
