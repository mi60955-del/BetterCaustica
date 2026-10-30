"""What a leaked medium does to the face behind the glass.

Reported picture, after the correction that this is not a denoiser artefact: on the face of a
non-full block seen through glass, a sharp-edged circular patch shows the content *behind*
the face, and outside that patch the face's own texture is normal. That is a Snell window,
which needs a critical angle, which needs the ray to arrive at the face while the pipeline
still believes it is inside the denser medium.

A leaked medium alone is not enough to make one: light that entered glass through a plane
face can never exceed the critical angle at a parallel face, so theta_inside is always below
41.1 deg for n = 1.52. The window appears only in the inconsistent state -- the medium says
glass, but the direction carried to the face is the straight *air* direction. Then the face
is tested as glass -> air with an air-space angle, which can reach 90 deg, so everything
beyond theta_c = asin(1/1.52) = 41.1 deg total-internal-reflects and shows the face's own
reflection, while everything inside transmits and shows what is behind the face.

So this probe measures the three numbers that identify that state in a real frame:
the angular radius of the patch (must be 41.1 deg), how sharp its rim is (a TIR boundary is
one pixel wide), and the compression of the transmitted content toward the rim (infinite at
theta_c). It also renders the two states so the look can be compared with the screenshot.
"""

import os

import numpy as np

import analysis as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out6")
N_GLASS = 1.52
THETA_C = np.degrees(np.arcsin(1.0 / N_GLASS))
RES = (520, 340)
SS = 3
CAM = np.array([0.0, 0.0, -0.55])
TARGET = np.array([0.0, 0.0, 1.0])
# A 41 deg cap cannot show its own rim inside a 62 deg frame, so the demo widens the frame:
# the window's rim then sits at 0.73 of the half-height and is plainly visible.
FOV_Y = 100.0
FACE_Z = 0.85
BACK_Z = 2.2
PANE = np.array([[-3.0, -2.0, -0.04], [3.0, 2.0, 0.04]])
FACE_HALF = 3.0


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def basis(eye, target):
    f = unit(target - eye)
    r = unit(np.cross(np.array([0.0, 1.0, 0.0]), f))
    u = np.cross(f, r)
    return f, r, u


def refract(d, n, eta):
    """d, n unit with n facing the incoming ray. Returns (dir, tir)."""
    cosi = -np.dot(d, n)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return d - 2.0 * np.dot(d, n) * n, True
    return unit(eta * d + (eta * cosi - np.sqrt(k)) * n), False


def behind(a, b):
    """What sits behind the face: a strongly marked wall, so 'seen through' is unmistakable."""
    cells = 0.14
    gx = np.floor(a / cells).astype(np.int64)
    gy = np.floor(b / cells).astype(np.int64)
    odd = ((gx + gy) & 1).astype(np.float64)
    shade = 0.35 + 0.55 * odd
    return np.stack([shade * 0.30, shade * 0.55, shade * 0.95], -1)      # blue checker


def face_color(a, b):
    """The face's own texture: warm bricks, so the two states never look alike."""
    row = np.floor(b / 0.11).astype(np.int64)
    xx = a + 0.055 * (row & 1)
    mortar = ((np.mod(b, 0.11) < 0.011) | (np.mod(xx, 0.22) < 0.011)).astype(np.float64)
    tone = 0.70 + 0.14 * np.mod(np.floor(xx / 0.22) * 7 + row * 3, 5) / 4.0
    base = tone * (1.0 - mortar) + 0.28 * mortar
    return np.stack([base, base * 0.86, base * 0.70], -1)


def trace(D, mode):
    eye = CAM[None, :]
    t_face = (FACE_Z - eye[:, 2]) / D[:, 2]
    p = eye + D * t_face[:, None]
    on_face = (np.abs(p[:, 0]) < FACE_HALF) & (np.abs(p[:, 1]) < FACE_HALF) & (t_face > 0.0)

    # pane silhouette, only used to gate the leak (a ray outside the pane never met glass)
    t_pane = (PANE[0, 2] - eye[:, 2]) / D[:, 2]
    q = eye + D * t_pane[:, None]
    in_pane = ((q[:, 0] > PANE[0, 0]) & (q[:, 0] < PANE[1, 0])
               & (q[:, 1] > PANE[0, 1]) & (q[:, 1] < PANE[1, 1]))

    col = face_color(p[:, 0], p[:, 1])
    tir = np.zeros(len(D), bool)
    qcol = np.full((len(D), 2), np.nan)
    if mode == "leak":
        n_face = np.array([0.0, 0.0, -1.0])
        for i in np.nonzero(on_face & in_pane)[0]:
            dd, t = refract(D[i], n_face, N_GLASS)     # glass -> air, with an air-space angle
            tir[i] = t
            if t:
                col[i] = col[i] * 0.45 + 0.05          # own reflection, dimmed
            else:
                t2 = (BACK_Z - p[i, 2]) / dd[2] if abs(dd[2]) > 1e-6 else 1e6
                w = p[i] + dd * t2
                qcol[i] = w[:2]
                col[i] = behind(np.array([w[0]]), np.array([w[1]]))[0]
    return col, on_face, tir, in_pane, qcol


def render(mode, res=RES, ss=SS):
    w, h = res
    f, r, u = basis(CAM, TARGET)
    ty = np.tan(np.radians(FOV_Y) * 0.5)
    xs = (np.arange(w * ss) + 0.5) / ss
    ys = (np.arange(h * ss) + 0.5) / ss
    gx, gy = np.meshgrid(xs, ys)
    sx = (gx / w * 2.0 - 1.0) * ty * (w / h)
    sy = (1.0 - gy / h * 2.0) * ty
    D = unit(f[None, :] + sx.ravel()[:, None] * r[None, :] + sy.ravel()[:, None] * u[None, :])
    col, on_face, tir, in_pane, qcol = trace(D, mode)
    shape = (h * ss, w * ss)
    img = col.reshape(shape + (3,)).reshape(h, ss, w, ss, 3).mean((1, 3))
    return img, dict(on_face=on_face.reshape(shape), tir=tir.reshape(shape),
                     in_pane=in_pane.reshape(shape), q=qcol.reshape(shape + (2,)), D=D)


def main():
    os.makedirs(OUT, exist_ok=True)
    truth, _ = render("truth")
    bug, info = render("leak")
    w, h = RES

    def down(mask):
        """Supersampled mask -> display-resolution coverage in [0,1]."""
        return mask.reshape(h, SS, w, SS).mean((1, 3))

    tir = down(info["tir"])
    on_face = down(info["on_face"])
    in_pane = down(info["in_pane"])
    sel = (on_face > 0.5) & (in_pane > 0.5)
    ang = down(np.degrees(np.arccos(np.clip(info["D"][:, 2], -1.0, 1.0))).reshape(info["tir"].shape))
    tx = sel & (tir < 0.5)
    refl = sel & (tir > 0.5)

    print(f"n = {N_GLASS}   theta_c = {THETA_C:.3f} deg")
    print(f"rays through the pane onto the face: {int(sel.sum())} px")
    print(f"  see-through {int(tx.sum())} px   total internal reflection {int(refl.sum())} px")
    print(f"largest incidence angle that still transmits : {ang[tx].max():7.2f} deg")
    print(f"smallest incidence angle that reflects       : {ang[refl].min():7.2f} deg")
    print(f"  -> boundary error vs theta_c: "
          f"{abs(ang[tx].max() - THETA_C):.3f} / {abs(ang[refl].min() - THETA_C):.3f} deg")

    edge = (tir > 0.02) & (tir < 0.98) & sel
    r_px = np.sqrt(max(int(tx.sum()), 1) / np.pi)
    print(f"window: {int(tx.sum())} px see-through inside a radius of {r_px:.1f} px "
          f"(perimeter ~{2 * np.pi * r_px:.0f} px)")
    print(f"rim: {int(edge.sum())} partial-coverage px => edge thickness "
          f"{edge.sum() / max(2 * np.pi * r_px, 1):.2f} px (1.0 = a one-pixel edge)")

    # compression of the transmitted content: the world step per screen pixel of what is
    # behind the face, at the centre of the window versus against its rim. It has to
    # collapse toward zero at theta_c, which is what makes the rim look like a fan.
    q0, q1 = info["q"][..., 0], info["q"][..., 1]
    step = np.hypot(np.diff(q0, axis=1), np.diff(q1, axis=1))
    tir_ss = info["tir"]
    spans = [np.count_nonzero(tir_ss[y] == 0) for y in range(tir_ss.shape[0])]
    row = int(np.argmax(spans))
    cols = np.nonzero(tir_ss[row] == 0)[0]
    if cols.size > 32:
        mid = float(np.nanmedian(step[row, cols[cols.size // 4: cols.size // 2]]))
        rim = float(np.nanmedian(step[row, cols[-6:-1]]))
        print(f"scanline {row}: world step per px of the content behind the face "
              f"{mid:.5f} at the centre -> {rim:.5f} at the rim "
              f"(x{mid / max(rim, 1e-9):.1f} compression)")

    def mono(m):
        return A.to_u8(np.repeat(m[..., None].astype(np.float64), 3, 2))

    tiles = [
        A.label(A.to_u8(truth), "opaque face (truth)"),
        A.label(A.to_u8(bug), f"leaked medium: Snell window at {THETA_C:.1f} deg"),
        A.label(mono(tx.astype(np.float64)), "see-through (transmitted)"),
        A.label(mono(refl.astype(np.float64)), "total internal reflection (own texture)"),
    ]
    A.save(A.contact_sheet(tiles, 2, pad=8), "sheet_snell.png", OUT)
    print(f"sheet -> {os.path.join(OUT, 'sheet_snell.png')}")


if __name__ == "__main__":
    main()
