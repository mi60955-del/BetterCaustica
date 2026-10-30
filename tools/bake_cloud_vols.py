#!/usr/bin/env python3
"""Bake the nine cloud-density volumes the volumetric cloud layer raymarches.

Binary contract (RtCloudVolumes.java reads it, this tool writes it): an 8-byte header
("BVOL", u8 version 1, u8 reserved, u16 little-endian resolution), then res^3 R8 densities
with x the fastest axis, then y, then z. z is the volume's VERTICAL axis: RtCloudVolumes
uploads each res*res z-slab as one 2D-array layer, and volume.slang's cloudInstanceDensity
maps the layer index to height inside the instance box, layer 0 the cloud's bottom. The
in-layer (x, y) axes are the box's horizontal footprint.

The instance box stretches whatever the volume holds: volume.slang's cloudSpan is a multiple of the
slab thickness while cloudBoxHeight is CLOUD_BOX_ASPECT of that footprint, so every shape is authored
in a domain 1.0 x 1.0 wide and VERTICAL_SCALE tall. Puff geometry is laid out in those world
proportions before sampling into the grid (one domain unit is the same number of blocks on every
axis), which is what stops a baked cumulus from rendering as a squashed pancake.

Shapes are deterministic in their own name, and the script gates its own output before
declaring success: the outer shell must be empty, no axis's per-slice means may be flat or
banded (either one reads in-game as square cards with a grid moire instead of clouds), the
vertical profile must be bottom-heavy, and a contact sheet of three max-projections per
shape lands next to the other cloud previews for an eyeball pass. The shape list is
cross-checked against RtCloudVolumes.RESOURCES at every run — the atlas index order and the
game's Cloud Type slider are one and the same list, and a silent mismatch would ship shapes
under the wrong names. The authored height is also cross-checked against volume.slang's
CLOUD_BOX_ASPECT, because a domain taller or shorter than the box stretches every shape.
"""

import argparse
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RES = 128
# Instance-box height as a fraction of its footprint; must equal volume.slang's CLOUD_BOX_ASPECT
# (checked in main). One domain unit is the same number of blocks on every axis at this value.
VERTICAL_SCALE = 0.5
BORDER_VOXELS = 4     # density ramps to zero over this many voxels at every face
PUFF_CORE = 0.45      # puff falloff: density 1 inside this fraction of the radius
PUFF_SMOOTH_MIN = 0.16  # smooth-min width: softens the crease where two puffs overlap
BILLOW_FALLOFF = 0.72   # worley F1 (cell units) mapped to zero erosion weight past this
# The three-octave billow removes more mass than a single-octave erosion does at the same
# amplitude, and the density cut below defines the silhouette. Applying this gain after that cut
# sets the deck's optical depth without moving the silhouette's shape, so tuning shape cannot
# silently thin the sky.
DENSITY_GAIN = 1.5

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
DEFAULT_OUT = REPO / "src/main/resources/assets/bettercaustica/textures/cloud"
SHEET_OUT = TOOLS / "cloud-preview-out"
JAVA = REPO / "src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtCloudVolumes.java"
VOLUME_SHADER = REPO / "shaders/pipelines/world/volume.slang"


# ---- Noise -----------------------------------------------------------------------------------------

def _hash3(xi, yi, zi, seed):
    """Lattice hash to [0, 2^32): integer avalanche over int64, masked. Deterministic in seed."""
    h = xi * np.int64(374761393) + yi * np.int64(668265263) + zi * np.int64(1274126177)
    h = h + np.int64(seed)
    h = (h ^ (h >> np.int64(13))) * np.int64(1103515245)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFFFF)).astype(np.int64)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1.0e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def perlin3(coords, freq, seed):
    """Gradient noise over `coords` (..., 3) in domain units; freq cells across the domain.
    Returns roughly [0, 1] (a unit-gradient Perlin spans about +-0.9, centred here)."""
    p = coords * np.float32(freq)
    xi = np.floor(p[..., 0]).astype(np.int64)
    yi = np.floor(p[..., 1]).astype(np.int64)
    zi = np.floor(p[..., 2]).astype(np.int64)
    xf = p[..., 0] - xi
    yf = p[..., 1] - yi
    zf = p[..., 2] - zi
    u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
    v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)
    w = zf * zf * zf * (zf * (zf * 6.0 - 15.0) + 10.0)

    rng = np.random.default_rng(seed)
    table = rng.standard_normal((256, 3), dtype=np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True)

    def grad(cx, cy, cz, dx, dy, dz):
        g = table[_hash3(cx, cy, cz, seed) & 255]
        return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

    n = ((1.0 - u) * ((1.0 - v) * grad(xi, yi, zi, xf, yf, zf)
                      + v * grad(xi, yi + 1, zi, xf, yf - 1.0, zf))
         + u * ((1.0 - v) * grad(xi + 1, yi, zi, xf - 1.0, yf, zf)
                + v * grad(xi + 1, yi + 1, zi, xf - 1.0, yf - 1.0, zf)))
    n = ((1.0 - w) * n
         + w * ((1.0 - u) * ((1.0 - v) * grad(xi, yi, zi + 1, xf, yf, zf - 1.0)
                             + v * grad(xi, yi + 1, zi + 1, xf, yf - 1.0, zf - 1.0))
                + u * ((1.0 - v) * grad(xi + 1, yi, zi + 1, xf - 1.0, yf, zf - 1.0)
                       + v * grad(xi + 1, yi + 1, zi + 1, xf - 1.0, yf - 1.0, zf - 1.0))))
    return np.clip(0.5 + 0.6 * n, 0.0, 1.0)


def worley3(coords, cells, seed):
    """F1 distance in CELL units over `coords` (..., 3) in domain units, one jittered point
    per grid cell. 0 at a cell's point, ~1 at the farthest corner of its cell."""
    pts = np.random.default_rng(seed).random((cells, cells, cells, 3), dtype=np.float32)
    p = coords * np.float32(cells)
    base = np.floor(p).astype(np.int64)
    f1 = np.full(p.shape[:-1], 1.0e9, np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx = np.clip(base[..., 0] + dx, 0, cells - 1)
                cy = np.clip(base[..., 1] + dy, 0, cells - 1)
                cz = np.clip(base[..., 2] + dz, 0, cells - 1)
                cpt = pts[cx, cy, cz]
                d = cx[..., None] + cpt - p
                f1 = np.minimum(f1, (d * d).sum(-1))
    return np.sqrt(f1)


def perlin_fbm(coords, freq, seed, octaves=4):
    total = np.zeros(coords.shape[:-1], np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * perlin3(coords, freq * (2 ** o), seed + 101 * o)
        norm += amp
        amp *= 0.5
    return total / norm


# ---- Domain ----------------------------------------------------------------------------------------

def world_grid():
    """Voxel-centre world coordinates as (RES, RES, RES, 3), indexed [z, y, x]: the footprint
    spans [0, 1] on x/y and the height [0, VERTICAL_SCALE] on z."""
    a = (np.arange(RES, dtype=np.float32) + 0.5) / RES
    z = (np.arange(RES, dtype=np.float32) + 0.5) / RES * VERTICAL_SCALE
    zz, yy, xx = np.meshgrid(z, a, a, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1)


# ---- Shapes ----------------------------------------------------------------------------------------

def _deck(rng, footprint, puffs, taper, env_pow, puff_lo, puff_hi,
          flat=0.8, u_lo=0.1, origin=(0.5, 0.5), base=0.0, height=1.0, shrink_top=1.0):
    """One deck of puffs. Heights are fractions of VERTICAL_SCALE (`base`..`base+height`).
    Puff heights are stratified-random rather than laid out in discrete layers — layers of
    puffs sharing one height read in the side projections as vertical striations. Each puff's
    ring follows the envelope footprint * (1 - taper*t)^env_pow; puff radii ride the same
    envelope scaled into [puff_lo, puff_hi] of it. `shrink_top` thins the puffs extra near the
    top (the cauliflower crown). The crown's puffs keep a real radius on purpose: shrinking them
    to a point stacks small blobs on the axis, and the erosion then cuts them loose as floating
    fragments instead of a crown."""
    centers, radii = [], []
    for i in range(puffs):
        t = min((i + rng.uniform(0.25, 0.75)) / max(puffs - 1, 1), 1.0)
        r = footprint * (1.0 - taper * t) ** env_pow
        crown = 1.0 - (1.0 - shrink_top) * t
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rr = r * math.sqrt(rng.uniform(u_lo, 1.0))
        pr = r * crown * rng.uniform(puff_lo, puff_hi)
        centers.append((origin[0] + rr * math.cos(ang) + rng.uniform(-1.0, 1.0) * 0.012,
                        origin[1] + rr * math.sin(ang) + rng.uniform(-1.0, 1.0) * 0.012,
                        (base + t * height) * VERTICAL_SCALE))
        radii.append((pr, pr, pr * flat))
    return centers, radii


def _smin(a, b, k):
    """Polynomial smooth minimum: C1 at the join, so two puffs that overlap weld into one surface
    instead of leaving the crease a plain min() draws along their intersection."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _puff_field(centers, radii, q):
    """Union of soft ellipsoids through a smooth minimum, then a falloff that saturates at
    PUFF_CORE. The smooth minimum keeps the crease between two overlapping puffs from being a
    curvature discontinuity; it cannot bridge puffs that do not overlap, so the deck's connectivity
    still comes from the authoring's footprint and puff count."""
    d = np.full(q.shape[:-1], np.float32(4.0), np.float32)
    for (cx, cy, cz), (rx, ry, rz) in zip(centers, radii):
        dx = (q[..., 0] - cx) / max(rx, 1.0e-4)
        dy = (q[..., 1] - cy) / max(ry, 1.0e-4)
        dz = (q[..., 2] - cz) / max(rz, 1.0e-4)
        d = _smin(d, np.sqrt(dx * dx + dy * dy + dz * dz), PUFF_SMOOTH_MIN)
    return 1.0 - smoothstep(PUFF_CORE, 1.0, d)


def billow3(q, seed, streak=1.0, octaves=3, cells=7.0):
    """Billow field: inverted Worley F1 summed over octaves and normalised to 0..1. One octave
    gives the deck a single lump scale, which reads as dimples on a smooth surface; three give
    the silhouette a large lobe, its shoulders and the small tears along them. Dividing x by
    `streak` lowers the frequency along x, elongating the lobes into the wind-aligned streaks a
    wisp needs."""
    qs = q * np.float32([1.0 / streak, 1.0, 1.0])
    total = np.zeros(q.shape[:-1], np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        f1 = worley3(qs, int(round(cells)), seed + 11 * o)
        total += amp * (1.0 - np.clip(f1 / BILLOW_FALLOFF, 0.0, 1.0))
        norm += amp
        amp *= 0.5
        cells *= 2.0
    return total / norm


def _finish(d, q, seed, *, erosion, detail, gamma, lo, streak=1.0, top_start=0.6, top_pow=1.2,
            base_ramp=0.04):
    """Vertical gradient, billow erosion, interior variation, contrast, border window."""
    t = q[..., 2] / VERTICAL_SCALE
    # Bottom-heavy profile: a thin ramp rounds the base, the top fades over [top_start, 1].
    grad = smoothstep(0.0, base_ramp, t) * (1.0 - smoothstep(top_start, 1.0, t)) ** top_pow
    d = d * grad

    # Erosion chews the boundary into billow lobes. The weight keeps it on the boundary, where the
    # silhouette lives, and off the interior: the three-octave field is high over a much larger area
    # than one octave, and weighting by (1 - d) is what keeps it from cutting the necks between
    # lobes and leaving a body of separate pebbles.
    w = billow3(q, seed + 7, streak)
    d = np.clip(d - (1.0 - d) * erosion * 0.7 * w, 0.0, 1.0)

    # Interior variation keeps the solid core from rendering as polished plastic.
    if detail > 0.0:
        d = d * (1.0 - detail + detail * perlin_fbm(q, 6.0, seed + 31, octaves=3))

    d = np.clip(d, 0.0, 1.0) ** gamma
    d = smoothstep(lo, 1.0, d)
    d = np.clip(d * DENSITY_GAIN, 0.0, 1.0)

    # Border window: the shader's superellipse blends into the volumes' own border fade, so
    # every face must reach zero a few voxels in.
    i = np.arange(RES, dtype=np.float32)
    ramp = smoothstep(0.0, float(BORDER_VOXELS), np.minimum(i, RES - 1.0 - i))
    win = ramp[None, None, :] * ramp[None, :, None] * ramp[:, None, None]
    return d * win


def _shape_puffs(name, rng):
    """Per-shape puff layout. Footprints and heights are world proportions of the instance
    box; the in-game Cloud Type slider order is the dict's insertion order (kept identical to
    RtCloudVolumes.RESOURCES, cross-checked in main)."""
    if name == "cumulus_small.vol":
        return _deck(rng, 0.26, 42, 0.75, 0.8, 0.38, 0.58, flat=0.7)
    if name == "cumulus_towering.vol":
        return _deck(rng, 0.24, 60, 0.5, 0.7, 0.42, 0.62, u_lo=0.0, flat=0.85)
    if name == "cumulus_heavy.vol":
        return _deck(rng, 0.33, 90, 0.55, 0.8, 0.42, 0.62, flat=0.85)
    if name == "cumulus_rising.vol":
        c1, r1 = _deck(rng, 0.30, 30, 0.5, 0.8, 0.42, 0.6, height=0.45)
        c2, r2 = _deck(rng, 0.13, 24, 0.15, 0.8, 0.55, 0.75, u_lo=0.0,
                       origin=(0.5, 0.48), base=0.3, height=0.7)
        return c1 + c2, r1 + r2
    if name == "cumulus_wispy.vol":
        return _deck(rng, 0.30, 34, 0.8, 0.9, 0.34, 0.54, flat=0.65)
    if name == "cumulus_mediocris.vol":
        return _deck(rng, 0.32, 64, 0.6, 0.75, 0.4, 0.6, flat=0.7)
    if name == "cauliflower.vol":
        return _deck(rng, 0.30, 72, 0.65, 1.2, 0.36, 0.54, flat=0.85, shrink_top=0.7)
    if name == "cauliflower_multi.vol":
        cs, rs = [], []
        for (ox, oy), fp, n in (((0.33, 0.34), 0.17, 40), ((0.64, 0.55), 0.21, 48),
                                ((0.44, 0.74), 0.15, 32)):
            c, r = _deck(rng, fp, n, 0.6, 1.1, 0.4, 0.58, flat=0.85,
                         shrink_top=0.7, origin=(ox, oy))
            cs += c
            rs += r
        return cs, rs
    if name == "storm.vol":
        return _deck(rng, 0.35, 110, 0.5, 0.8, 0.45, 0.65, flat=0.9)
    raise KeyError(name)


# Per-shape finish knobs: erosion/detail shape the silhouette, gamma/lo set how much of the
# envelope reads as cloud versus haze. Tune these against the contact sheet (--sheet).
FINISH = {
    "cumulus_small.vol":      dict(erosion=0.55, detail=0.25, gamma=1.25, lo=0.22, top_start=0.55),
    "cumulus_towering.vol":   dict(erosion=0.45, detail=0.22, gamma=1.05, lo=0.16, top_start=0.85),
    "cumulus_heavy.vol":      dict(erosion=0.40, detail=0.20, gamma=1.00, lo=0.14, top_start=0.72),
    "cumulus_rising.vol":     dict(erosion=0.50, detail=0.22, gamma=1.15, lo=0.20, top_start=0.85),
    "cumulus_wispy.vol":      dict(erosion=0.60, detail=0.30, gamma=1.35, lo=0.22, top_start=0.30,
                                   streak=1.9),
    "cumulus_mediocris.vol":  dict(erosion=0.50, detail=0.22, gamma=1.10, lo=0.18, top_start=0.60),
    "cauliflower.vol":        dict(erosion=0.62, detail=0.28, gamma=1.20, lo=0.22, top_start=0.60),
    "cauliflower_multi.vol":  dict(erosion=0.50, detail=0.24, gamma=1.10, lo=0.18, top_start=0.65),
    "storm.vol":              dict(erosion=0.35, detail=0.15, gamma=0.95, lo=0.14, top_start=0.88),
}


def bake(name):
    """One shape to uint8[RES, RES, RES], indexed [z, y, x] (z vertical, layer 0 the bottom)."""
    seed = int.from_bytes(name.encode(), "little") % 0xFFFF
    rng = np.random.default_rng(seed)
    q = world_grid()
    centers, radii = _shape_puffs(name, rng)
    d = _puff_field(centers, radii, q)
    d = _finish(d, q, seed, **dict(FINISH[name]))
    return np.ascontiguousarray(d, dtype=np.float32)


def write_vol(path, d):
    """Serialize to the BVOL contract: the file order is x fastest, then y, then z, so the
    [z, y, x]-indexed array transposes to [x, y, z] before a C-order byte dump."""
    voxels = np.ascontiguousarray(d.transpose(2, 1, 0), dtype=np.float32)
    u8 = np.rint(voxels * 255.0).clip(0, 255).astype(np.uint8)
    path.write_bytes(struct.pack("<4sBBH", b"BVOL", 1, 0, RES) + u8.tobytes())


# ---- Validation ------------------------------------------------------------------------------------

def validate(d, name):
    """The gates the broken 0.2.3 assets failed: a solid extruded block with stripe bands.
    Returns a list of (level, message); level 'fail' makes the run exit non-zero."""
    issues = []
    # Only the outermost voxel must be exactly zero — that is what clamp-to-edge sampling
    # reads when a ray leaves the box. The ramp inside BORDER_VOXELS is the fade itself.
    rim = np.zeros(d.shape, bool)
    rim[0, :, :] = rim[-1, :, :] = rim[:, 0, :] = rim[:, -1, :] = True
    rim[:, :, 0] = rim[:, :, -1] = True
    if d[rim].max() > 0.0:
        issues.append(("fail", f"outermost voxel not zero (max {d[rim].max():.3f})"))

    means = [d.mean(axis=ax) for ax in range(3)]
    for ax, m in zip("xyz", means):
        if m.std() < 0.004:
            issues.append(("fail", f"{ax}: per-slice means constant (std {m.std():.5f}) — "
                                   f"the volume is extruded along {ax}"))
        # A large cliff between ADJACENT slices is the stripe signature; a legitimate profile
        # rises and falls smoothly, and narrow shapes legitimately swing their slice means by
        # tenths as their puffs' cross-sections change.
        jumps = np.abs(np.diff(m))
        if jumps.max() > 0.30:
            issues.append(("fail", f"{ax}: cliffed slice profile (max jump {jumps.max():.3f})"))
        if np.abs(np.diff(d, axis="xyz".index(ax))).mean() > 0.05:
            issues.append(("warn", f"{ax}: high neighbour-difference (striped detail)"))

    zm = d.mean(axis=(1, 2))
    if zm[: RES // 2].mean() <= zm[RES // 2:].mean():
        issues.append(("warn", "vertical profile is not bottom-heavy"))
    frac = (d > 0.01).mean()
    if not 0.04 <= frac <= 0.55:
        issues.append(("warn", f"unusual filled fraction {frac:.3f}"))
    # The mean is over the whole box, and the box is 2:1 wide to tall, so a body that fills its
    # middle third still averages well below the density a squat box would show. The floor is a
    # degenerate-output gate, not a target.
    if not 0.03 < d.mean() < 0.45:
        issues.append(("warn", f"unusual mean density {d.mean():.3f}"))

    filled = f"fill {frac:.2f}  mean {d.mean():.3f}"
    if any(level == "fail" for level, _ in issues):
        print(f"  [FAIL] {name}: {filled}")
    else:
        print(f"  [ ok ] {name}: {filled}")
    for level, msg in issues:
        print(f"         {level}: {msg}")
    return issues


# ---- Contact sheet ---------------------------------------------------------------------------------

def sheet(volumes, out_path):
    """Three max-projections per shape (silhouettes must read as a cloud from all sides) plus
    one horizontal slice at 40% height for the interior texture.

    The two side columns are drawn at the volume's own proportions — VERTICAL_SCALE of the width —
    and with the base at the bottom, so the sheet shows what the instance box will show. Drawn
    square, a 2:1 shape reads as a tower and every crown looks like a spire."""
    cols = (("max x", VERTICAL_SCALE), ("max y", VERTICAL_SCALE),
            ("max z (top-down)", 1.0), ("slice z~40%", 1.0))
    row_h = RES + 18
    img = Image.new("L", (RES * len(cols) + (len(cols) - 1) * 4, row_h * len(volumes) + 4), 40)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    for ri, name in enumerate(volumes):
        d = volumes[name]
        tiles = [np.flipud(d.max(axis=2)), np.flipud(d.max(axis=1)),
                 np.flipud(d.max(axis=0)), d[int(RES * 0.4)]]
        oy = 4 + ri * row_h
        dr.text((4, oy), name, fill=255)
        for ci, (tile, aspect) in enumerate(zip(tiles, cols)):
            ox = ci * (RES + 4)
            if ri == 0:
                dr.text((ox + 2, oy + 12), aspect[0], fill=200)
            h = max(1, int(round(RES * aspect[1])))
            view = Image.fromarray((tile * 255).astype(np.uint8)).resize((RES, h), Image.BILINEAR)
            img.paste(view, (ox, oy + 18 + (RES - h)))
    img.save(out_path)


# ---- Driver ----------------------------------------------------------------------------------------

def java_resources():
    """The atlas order RtCloudVolumes.java ships; the slider's 0..1 maps onto this exact list."""
    m = re.search(r"RESOURCES = \{(.*?)\};", JAVA.read_text(encoding="utf-8"), re.S)
    if not m:
        raise SystemExit(f"cannot find RtCloudVolumes.RESOURCES in {JAVA}")
    return re.findall(r'"([^"]+\.vol)"', m.group(1))


def shader_box_aspect():
    """volume.slang's CLOUD_BOX_ASPECT. The box stretches the volume onto its own aspect, so a
    domain authored at a different height squashes or stretches every shape."""
    m = re.search(r"CLOUD_BOX_ASPECT\s*=\s*([0-9]*\.?[0-9]+)f?\s*;",
                  VOLUME_SHADER.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"cannot find CLOUD_BOX_ASPECT in {VOLUME_SHADER}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for .vol files")
    ap.add_argument("--sheet", type=Path, default=SHEET_OUT / "bake_check_sheet.png",
                    help="contact-sheet output path")
    ap.add_argument("--check-only", action="store_true",
                    help="validate existing .vol files instead of baking")
    args = ap.parse_args()

    names = list(FINISH)
    shipped = java_resources()
    if names != shipped:
        raise SystemExit(f"shape order drifted from RtCloudVolumes.RESOURCES:\n"
                         f"  tool:    {names}\n  shipped: {shipped}")
    aspect = shader_box_aspect()
    if abs(aspect - VERTICAL_SCALE) > 1.0e-6:
        raise SystemExit(f"VERTICAL_SCALE {VERTICAL_SCALE} != volume.slang CLOUD_BOX_ASPECT {aspect}: "
                         f"the instance box would stretch every shape")

    if args.check_only:
        volumes = {}
        for name in names:
            raw = (args.out / name).read_bytes()
            magic, version, _, res = struct.unpack("<4sBBH", raw[:8])
            if magic != b"BVOL" or version != 1 or res != RES or len(raw) != 8 + RES ** 3:
                raise SystemExit(f"{name}: malformed (magic {magic!r}, version {version}, "
                                 f"res {res}, {len(raw)} bytes)")
            d = np.frombuffer(raw, np.uint8, RES ** 3, 8).reshape(RES, RES, RES).transpose(2, 1, 0)
            volumes[name] = d.astype(np.float32) / 255.0
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        volumes = {}
        for name in names:
            print(f"baking {name} ...")
            volumes[name] = bake(name)
            write_vol(args.out / name, volumes[name])

    failed = False
    print(f"validating {len(volumes)} shapes:")
    for name in names:
        if any(level == "fail" for level, _ in validate(volumes[name], name)):
            failed = True
    sheet(volumes, args.sheet)
    print(f"contact sheet: {args.sheet}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
