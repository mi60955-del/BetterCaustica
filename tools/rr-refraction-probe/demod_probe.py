"""Does a guide/colour mismatch alone destroy texture detail? Demodulation probe.

DLSS-RR does not denoise the radiance image directly. It divides the colour by the guide
albedo, denoises that ratio (which should be pure lighting), and multiplies the albedo back
in. The step is exact only while the guide albedo is the same albedo the colour was lit by.
The guide is written by a separate trace through the glass (guides.slang:177-250), so any
difference between the two -- a different mip, a tint, a face, a normalising constant -- is
left behind in the ratio as texture detail, and the spatial filter that denoises the ratio
treats that detail as noise and removes it. The output then re-modulates a smeared ratio and
comes back with the texture wiped flat, permanently and without any camera motion.

This probe measures that: one texture, one lighting field, an exact colour, and an albedo
guide with a controlled error. The denoiser is a plain spatial filter, deliberately the most
forgiving possible stand-in for RR's kernel, so the numbers here are a lower bound.
"""

import os

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out4")
W, H = 480, 270


def maze(n=16):
    """Blocky spiral/maze albedo, the kind of high-frequency detail that reads as 'smeared'."""
    t = np.zeros((n, n), np.float32)
    lo, hi, v = 0, n - 1, 0.0
    while lo <= hi:
        for i in range(lo, hi + 1):
            t[lo, i] = v
            t[hi, i] = 1.0 - v
        for j in range(lo, hi + 1):
            t[j, hi] = v
            t[j, lo] = 1.0 - v
        lo, hi, v = lo + 1, hi - 1, 1.0 - v if (hi - lo) % 4 else v
    return 0.28 + 0.44 * t


def upsample(t, w, h):
    """Nearest-neighbour to screen, so every texel edge is a hard 1-px step."""
    n = t.shape[0]
    ys = (np.arange(h) * n // h).clip(0, n - 1)
    xs = (np.arange(w) * n // w).clip(0, n - 1)
    return t[ys][:, xs]


def lighting(w, h):
    """Smooth light with one soft shadow quad: low frequency, recoverable by a denoiser."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 0.62 + 0.38 * (xx / w)
    g *= 1.0 - 0.30 * np.exp(-(((xx - 0.30 * w) / (0.22 * w)) ** 2
                               + ((yy - 0.55 * h) / (0.26 * h)) ** 2))
    return g


def box_blur(a, r):
    """Separable box blur, radius r, mirrored edges."""
    r = max(int(round(r)), 0)
    if r == 0:
        return a.copy()
    k = 2 * r + 1
    p = np.pad(a, ((0, 0), (r, r)), mode="reflect")
    c = np.cumsum(np.concatenate([np.zeros((p.shape[0], 1), p.dtype), p], axis=1), axis=1)
    a = (c[:, k:] - c[:, :-k]) / k
    p = np.pad(a, ((r, r), (0, 0)), mode="reflect")
    c = np.cumsum(np.concatenate([np.zeros((1, p.shape[1]), p.dtype), p], axis=0), axis=0)
    return (c[k:, :] - c[:-k, :]) / k


def grad_energy(a):
    gy, gx = np.gradient(a)
    return float(np.mean(np.hypot(gx, gy)))


def run_variant(tex, light, guide, radius=4.0):
    """RR-style demodulate -> spatial denoise -> re-modulate."""
    colour = tex * light
    albedo = np.maximum(guide, 1.0e-3)
    demod = colour / albedo
    demod = box_blur(demod, radius)              # the denoiser's spatial fallback
    return demod * albedo


def main():
    os.makedirs(OUT, exist_ok=True)
    tex = upsample(maze(16), W, H)
    light = lighting(W, H)

    fresnel = 0.86 + 0.14 * np.sin(np.linspace(0.0, 3.0, W))[None, :]   # per-pixel tint error
    variants = [
        ("exact guide", tex),
        ("guide mip +2", box_blur(box_blur(tex, 1), 1)),
        ("guide tint x0.95", tex * 0.95),
        ("guide tint x1.05", tex * 1.05),
        ("guide fresnel tilt", tex * np.broadcast_to(fresnel, (H, W))),
    ]

    truth = tex * light
    e_truth = grad_energy(truth)
    print(f"reference gradient energy (exact colour) = {e_truth:.5f}\n")
    print(f"{'guide':22s} {'grad ratio':>11s} {'tex contrast':>13s} {'mean |err|':>11s}")
    tiles = [("truth texture x light", truth)]
    rows = []
    for name, guide in variants:
        out = run_variant(tex, light, guide)
        ratio = grad_energy(out) / e_truth
        # how much of the *texture* survives: divide the lighting back out and compare to tex
        recovered = out / np.maximum(light, 1.0e-3)
        contrast = float(np.std(recovered) / np.std(tex))
        err = float(np.mean(np.abs(out - truth)))
        rows.append((name, ratio, contrast, err))
        print(f"{name:22s} {ratio:11.3f} {contrast:13.3f} {err:11.4f}")
        tiles.append((f"{name}: grad x{ratio:.2f}", out))
    print("\ngrad ratio 1.00 = detail preserved; tex contrast 1.00 = texture fully recovered")
    print("(denoiser here is a plain 4-px blur, the mildest stand-in for RR's spatial kernel)")

    # ---------------------------------------------------------------- image output
    import analysis as A

    imgs = [np.repeat(A.to_u8(np.clip(t, 0.0, 1.0))[:, :, None], 3, axis=2) for _, t in tiles]
    imgs = [A.label(img, lab) for img, (lab, _) in zip(imgs, tiles)]
    sheet = A.contact_sheet(imgs, 2, pad=8)
    A.save(sheet, "sheet_demod.png", OUT)
    print(f"\nsheet -> {os.path.join(OUT, 'sheet_demod.png')}")


if __name__ == "__main__":
    main()
