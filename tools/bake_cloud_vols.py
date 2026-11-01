#!/usr/bin/env python3
"""Bake the nine cloud-density volumes the volumetric cloud layer raymarches.

Binary contract (RtCloudVolumes.java reads it, this tool writes it): an 8-byte header
("BVOL", u8 version 1, u8 reserved, u16 little-endian resolution), then res^3 R8 densities
with x the fastest axis, then y, then z. z is the volume's VERTICAL axis: RtCloudVolumes
uploads each res*res z-slab as one 2D-array layer, and volume.slang's cloudInstanceDensity
maps the layer index to height inside the instance box, layer 0 the cloud's bottom. The
in-layer (x, y) axes are the box's horizontal footprint.

The instance box stretches whatever the volume holds: volume.slang's cloudSpan is the Cloud Size dial
and cloudBoxHeight the Cloud Thickness dial, and the two are independent, so the authored domain's own
proportions are the only aspect a shape is designed in. Every shape is authored 1.0 x 1.0 wide and
VERTICAL_SCALE tall, which makes one domain unit the same number of blocks on x/y and twice as many on
z; keeping the puff geometry in those proportions before sampling is what stops a baked cumulus from
rendering as a squashed pancake.

Silhouettes come from four stacked mechanisms. The puff union is sampled through a coordinate warp, so
the body's outline is swirled rather than spherical. A closing then lays a re-thresholded low-pass of the
field back over itself, which is what carries mass between puffs that do not touch: a smooth minimum
welds only overlapping ones, so without it a deck whose radii fall short reads as a heap of separate
balls. It runs before the density cut because the cut is precisely what turns mid-level density into a
surface, and before the erosion because a continuous body is what lets the erosion carve lobes rather
than cut a body into pebbles. Billow erosion and a finer signed fuzz field then chew and regrow the
boundary band, sampled through a second, higher-frequency warp so their lobes are not grid-aligned. A
final narrow closing relinks the hairline necks that erosion severs, and because a thin tendril averages
near nothing across its window, it closes a severed neck without erasing the wisps.

Shapes are deterministic in their own name, and the script gates its own output before
declaring success: the outer shell must be empty, no axis's per-slice means may be flat or
banded (either one reads in-game as square cards with a grid moire instead of clouds), the
vertical profile must be bottom-heavy, and a contact sheet of three max-projections plus one lit
surface view per shape lands next to the other cloud previews for an eyeball pass. The projection
columns say whether the silhouette reads as a cloud; only the lit one shows what the erosion did,
because a max over a column hides any matter it removed. The shape list is
cross-checked against RtCloudVolumes.RESOURCES at every run — the atlas index order and the
game's Cloud Type slider are one and the same list, and a silent mismatch would ship shapes
under the wrong names. The slice count is cross-checked against volume.slang's
CLOUD_VOLUME_RES, which addresses a height slice inside one shape's band of the atlas.
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
# Authored domain height as a fraction of its width. volume.slang stretches the volume onto the
# instance box, whose height is the Cloud Thickness dial and whose footprint is Cloud Size, so this
# number only fixes the proportions the shapes are DESIGNED in: at 0.5 one domain unit is one stack
# of blocks horizontally and twice that vertically.
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
# Blur radii are in voxels, not world units: the domain is 1.0 wide but VERTICAL_SCALE tall, so a
# voxel spans half as much height as breadth and an equal radius on every axis keeps the authored
# base and crown fade sharper than a world-isotropic radius would.
WELD_RADIUS = 9       # closing width of the pre-erosion weld
WELD_LO = 0.26        # low-pass level below which the weld adds nothing (a lone surface's halo)
WELD_HI = 0.52        # level at which the weld's bridge is full mass
LINK_RADIUS = 3       # closing width of the post-cut relink
LINK_LO = 0.30
LINK_HI = 0.55
BLUR_PASSES = 3       # box-blur repetitions; three approximate one Gaussian of that radius
# An erosion field only carves if it is sparse. The billow field averages 0.29 and reaches 0.87, so
# using it raw lowers the whole deck about equally and leaves the silhouette a slightly smaller copy
# of itself; keeping the band above its own crest turns it into a few local bites. The two thresholds
# are per field because the Perlin wisp field is far tighter than the Worley billow one, and the upper
# one sits near the field's own maximum so a bite ramps in rather than arriving at full depth — a
# narrow band leaves round, bottomed-out craters, which read as a scored ball and not as a cloud.
BITE_LO = 0.26        # billow field value below which nothing is eroded
BITE_HI = 0.82        # value at which the bite reaches full strength
WISP_LO = 0.50        # ditto for the wisp pass, whose field spans about 0.41..0.66
WISP_HI = 0.72
CORE_DENSITY = 0.6    # above this the voxel is solid on every side and a warp moves no surface
BASE_BAND = 0.22      # fraction of the box height the base erosion reaches up into

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


def linear_step(a, b, x):
    """x rescaled from [a, b] to [0, 1] and clamped, with `a` allowed to be a field. This is the
    erosion remap's shape: `a` is how much the erosion field eats at this voxel, and a density of 1
    has to land back on 1 so the interior is untouched however hard the field bites."""
    return np.clip((x - a) / np.maximum(b - a, 1.0e-4), 0.0, 1.0)


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
                origin = np.stack([cx, cy, cz], axis=-1)
                d = origin + cpt - p
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


def box_blur(d, radius, axis):
    """Mean filter along one axis with the edges clamped, through a running sum so the cost is
    linear in the voxel count rather than in the radius."""
    r = int(max(0, min(radius, d.shape[axis] - 1)))
    if r == 0:
        return d
    a = np.moveaxis(d, axis, -1)
    n = a.shape[-1]
    padded = np.empty(a.shape[:-1] + (n + 2 * r,), np.float32)
    padded[..., :r] = a[..., :1]
    padded[..., r:r + n] = a
    padded[..., r + n:] = a[..., n - 1:]
    prefix = np.concatenate([np.zeros(padded.shape[:-1] + (1,), np.float32),
                             np.cumsum(padded, axis=-1, dtype=np.float32)], axis=-1)
    blurred = (prefix[..., 2 * r + 1:2 * r + 1 + n] - prefix[..., 0:n]) / np.float32(2 * r + 1)
    return np.ascontiguousarray(np.moveaxis(blurred, -1, axis), np.float32)


def low_pass(d, radius, passes=BLUR_PASSES):
    """Isotropic-in-voxels smoothing: repeated box blur, which is what the weld and the relink read
    as the deck's low-frequency body."""
    out = d
    for _ in range(passes):
        for axis in range(3):
            out = box_blur(out, radius, axis)
    return out


def close_gaps(d, radius, *, lo, hi, gain):
    """Morphological closing: lay a re-thresholded low-pass back over the field with a max.

    The low-pass alone is not a closing. Two nearby bodies average to a mid-level bridge, but so does
    one body's own surroundings, so a plain blur-max adds a soft glow wherever density exists at all
    and fills nothing. Re-thresholding picks the bridge out of that glow: `lo` sits above the level a
    lone surface reaches outside the deck, so the halo stays zero, and below the level two surfaces
    within `radius` of each other average to, so the gap between them becomes mass."""
    if gain <= 0.0:
        return d
    return np.maximum(d, smoothstep(lo, hi, low_pass(d, radius)) * gain)


def curl_field(q, seed, *, freq, octaves=2, streak=1.0):
    """Curl of a Perlin potential, taken across the voxel grid: a displacement with zero divergence.

    Three independent noise fields — the obvious way to build a vector field — have non-zero
    divergence, so wherever they point inward they compress matter into a knot and where they point
    outward they thin it into a shell. A curl leaves the local mass alone and only swirls it, which is
    what an updraft does to a real cloud. The derivatives are central differences along the grid, each
    axis divided by its own voxel edge because the domain is 1.0 wide but VERTICAL_SCALE tall."""
    qs = q * np.float32([1.0 / streak, 1.0 / streak, 1.0])
    pot = [perlin_fbm(qs, freq, seed + 977 * k, octaves=octaves) for k in range(3)]
    edge = (VERTICAL_SCALE / RES, 1.0 / RES, 1.0 / RES)  # voxel edge on the (z, y, x) axes
    d_pot_dz, d_pot_dy, d_pot_dx = ([np.gradient(p, edge[axis], axis=axis) for p in pot]
                                    for axis in range(3))
    return np.stack([d_pot_dy[2] - d_pot_dz[1],
                     d_pot_dz[0] - d_pot_dx[2],
                     d_pot_dx[1] - d_pot_dy[0]], axis=-1).astype(np.float32)


def warp(q, seed, *, amp, freq, octaves=2, streak=1.0):
    """Displace sample coordinates by a curl field normalised so `amp` is its RMS push in domain
    units. `amp` may itself be a per-voxel field, which is how the warp gets gated to the region
    where it does something. Warping the coordinates rather than adding to the density is the point:
    a warp moves the isosurface itself, so its raggedness survives the density cut, whereas a field
    added after the cut only re-thresholds the same surface."""
    if np.ndim(amp) == 0 and amp <= 0.0:
        return q
    c = curl_field(q, seed, freq=freq, octaves=octaves, streak=streak)
    c /= max(float(np.sqrt((c * c).mean())), 1.0e-9)
    push = np.asarray(amp, np.float32)
    if push.ndim == c.ndim - 1:
        push = push[..., None]  # a per-voxel gate applies to all three components at once
    return (q + c * push).astype(np.float32)


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


def _finish(d, q, qw, seed, *, erosion, detail, gamma, lo, streak=1.0, top_start=0.6, top_pow=1.2,
            base_ramp=0.04, weld=0.0, link=0.0, fuzz=0.0, fuzz_freq=14.0, ewarp=0.0,
            wisp=0.0, wisp_freq=4.0, bottom=0.0, erode=1.0):
    """Vertical gradient, the weld that closes the deck into one body, the gated curl warps and the
    three erosion remaps (billow, wisp, base), interior variation, the density cut, the narrow relink,
    and the border window. `q` is the authored domain and `qw` the body-warped coordinates the puff
    field was sampled through: the profile reads the first so the base stays flat and the crown fade
    stays where it was authored, and every roughening field reads the second so its lobes follow the
    body's swirl instead of the grid."""
    t = q[..., 2] / VERTICAL_SCALE
    # Bottom-heavy profile: a thin ramp rounds the base, the top fades over [top_start, 1].
    grad = smoothstep(0.0, base_ramp, t) * (1.0 - smoothstep(top_start, 1.0, t)) ** top_pow
    d = d * grad

    # The weld: the deck's separate puffs closed into one body, before anything else touches them. A
    # smooth minimum only welds overlapping ones, so an authoring whose radii fall short reads as a
    # heap of balls; and the density cut at the end of this function sharpens each soft puff back into
    # a small hard sphere, which means the connectivity has to be built up here rather than recovered
    # after the cut. Closing first also gives the erosion below a continuous surface to chew, so it
    # carves lobes out of a body instead of cutting a body into pebbles.
    d = close_gaps(d, WELD_RADIUS, lo=WELD_LO, hi=WELD_HI, gain=weld)

    # Where a warp is worth doing: up in the crown, and off the saturated core. A displacement cannot
    # move a voxel that is already solid on every side — it changes nothing about that surface — and
    # the base wants to stay the flat cut the profile authored, so the ramp is what leaves the underside
    # alone. Gating on the low side too, as some formulations do, is a mistake: the erosion lookup has
    # to be carried across the empty air to reach the far side of the deck, and a gate that vanishes
    # outside the body leaves the warp acting on almost nothing at all.
    gate = (0.2 + 0.3 * t * t) * (1.0 - smoothstep(CORE_DENSITY, 1.0, d))

    # The billow field sharpened, which is what makes it an erosion at all. Raw, the three-octave
    # field sits high over most of the domain, so multiplying a density by it lowers the whole deck
    # about equally and carves nothing; keeping only its crests leaves a sparse field whose high values
    # are few and local, and those are the bites the silhouette is made of.
    # Every roughening field below is sampled in deck-scaled coordinates. Their frequencies are
    # absolute in the authored domain, so nine decks of nine sizes otherwise carry nine identical bite
    # marks, and that — not any difference in outline, which survives the whole pipeline intact — is
    # what makes the Cloud Shape slider read as one shape at nine sizes. Scaling the coordinates once
    # moves the warp, the billow, the wisp and the fuzz together, so a small cumulus gets a small swirl
    # and a small bite and a storm gets both large.
    qe = qw * np.float32(erode)
    ew = warp(qe, seed + 401, amp=ewarp * gate, freq=3.0, streak=streak)
    bite = smoothstep(BITE_LO, BITE_HI, billow3(ew, seed + 7, streak))

    # The billow erosion, as a remap rather than a subtraction: the field says how much of the column
    # this voxel still has to clear, and a density of 1 lands back on 1 however hard the field bites.
    # Weighting the old additive form by (1 - d) was an attempt at the same thing — it kept the
    # interior solid — but it still greyed every mid-level voxel it touched, which is what turned a
    # deck's shoulders into a haze. The remap leaves the body alone and eats only the surface.
    d = linear_step(bite * erosion, 1.0, d)

    # The wisp pass: a second remap at a frequency the billow does not reach, carried by a wider curl,
    # and weighted onto the crown. Billow alone gives the whole deck one lump scale; this is what makes
    # the crown fibrous while the body under it stays a single mass, and the reason it is a separate
    # pass and not another billow octave is that it needs its own warp strength to be any good.
    if wisp > 0.0:
        ww = warp(qe, seed + 733, amp=2.0 * ewarp * gate, freq=wisp_freq * 0.5, streak=streak)
        e = perlin_fbm(ww, wisp_freq, seed + 733, octaves=3)
        d = linear_step(smoothstep(WISP_LO, WISP_HI, e) * wisp * smoothstep(0.35, 1.0, t), 1.0, d)

    # The fuzz: a signed modulation of the boundary band at a frequency neither pass above reaches.
    # Every remap can only remove, which thins a rim until it vanishes; this pushes the surface out
    # where the field runs low as well as in where it runs high, so the rim ends up ragged at voxel
    # scale rather than merely eaten away.
    if fuzz > 0.0:
        f = perlin_fbm(ew, fuzz_freq, seed + 601, octaves=2)
        d = np.clip(d - (1.0 - d) * fuzz * (f - 0.5) * 2.0, 0.0, 1.0)

    # The base is a flat cut and no cumulus has one: an updraft chews the underside into ragged
    # shelves while the crown stays rounded. Read the same bite field rather than a new one so the base
    # erodes along the cells the rest of the deck does, and weight it to the bottom of the box so it
    # moves the underside and nothing above it.
    if bottom > 0.0:
        d = d * (1.0 - bite * bottom * (1.0 - smoothstep(0.0, BASE_BAND, t)))

    # Interior variation keeps the solid core from rendering as polished plastic.
    if detail > 0.0:
        d = d * (1.0 - detail + detail * perlin_fbm(qw, 6.0, seed + 31, octaves=3))

    d = np.clip(d, 0.0, 1.0) ** gamma
    d = smoothstep(lo, 1.0, d)
    d = np.clip(d * DENSITY_GAIN, 0.0, 1.0)

    # The relink, at the one scale the cut cannot be blamed for: erosion severs hairline necks the
    # weld just closed, and a floating fragment is the same defect in a new place. A narrow closing
    # rejoins two lobes a couple of voxels across a cut, while a wisp thin enough to be a wisp is
    # still alone inside its window and survives untouched.
    d = close_gaps(d, LINK_RADIUS, lo=LINK_LO, hi=LINK_HI, gain=link)

    # Border window: the shader's superellipse blends into the volumes' own border fade, so
    # every face must reach zero a few voxels in.
    i = np.arange(RES, dtype=np.float32)
    ramp = smoothstep(0.0, float(BORDER_VOXELS), np.minimum(i, RES - 1.0 - i))
    win = ramp[None, None, :] * ramp[None, :, None] * ramp[:, None, None]
    return d * win


def _shape_puffs(name, rng):
    """Per-shape puff layout. Footprints and heights are world proportions of the instance
    box; the in-game Cloud Type slider order is the dict's insertion order (kept identical to
    RtCloudVolumes.RESOURCES, cross-checked in main).

    Puff count and radii are the connectivity of the deck: _finish's closing only bridges gaps of a
    couple of voxels, so a layout whose neighbours sit a whole puff-radius apart stays a heap of
    balls no knob at that stage can fix. Read the contact sheet's slice column for that — separate
    discs there mean the authoring is too sparse, and the count is the dial to move."""
    if name == "cumulus_small.vol":
        return _deck(rng, 0.26, 58, 0.75, 0.8, 0.40, 0.66, flat=0.7)
    if name == "cumulus_towering.vol":
        return _deck(rng, 0.24, 60, 0.5, 0.7, 0.42, 0.62, u_lo=0.0, flat=0.85)
    if name == "cumulus_heavy.vol":
        return _deck(rng, 0.33, 90, 0.55, 0.8, 0.42, 0.62, flat=0.85)
    if name == "cumulus_rising.vol":
        c1, r1 = _deck(rng, 0.30, 40, 0.5, 0.8, 0.44, 0.64, height=0.45)
        c2, r2 = _deck(rng, 0.13, 34, 0.15, 0.8, 0.58, 0.85, u_lo=0.0,
                       origin=(0.5, 0.48), base=0.3, height=0.7)
        return c1 + c2, r1 + r2
    if name == "cumulus_wispy.vol":
        return _deck(rng, 0.30, 56, 0.8, 0.9, 0.42, 0.70, flat=0.65)
    if name == "cumulus_mediocris.vol":
        return _deck(rng, 0.32, 64, 0.6, 0.75, 0.4, 0.6, flat=0.7)
    if name == "cauliflower.vol":
        return _deck(rng, 0.30, 92, 0.65, 1.2, 0.38, 0.62, flat=0.85, shrink_top=0.7)
    if name == "cauliflower_multi.vol":
        cs, rs = [], []
        for (ox, oy), fp, n in (((0.33, 0.34), 0.17, 54), ((0.64, 0.55), 0.21, 62),
                                ((0.44, 0.74), 0.15, 46)):
            c, r = _deck(rng, fp, n, 0.6, 1.1, 0.42, 0.68, flat=0.85,
                         shrink_top=0.7, origin=(ox, oy))
            cs += c
            rs += r
        return cs, rs
    if name == "storm.vol":
        return _deck(rng, 0.35, 110, 0.5, 0.8, 0.45, 0.65, flat=0.9)
    raise KeyError(name)


# Per-shape finish knobs: erosion/wisp/bottom are the three remaps and set how deep each one bites,
# gamma/lo set how much of the envelope reads as cloud versus haze. `swirl` warps the coordinates the
# puff union is sampled through, so the body's outline is bent at its own scale; `ewarp` does the same
# for the erosion fields, and both are ramped up toward the crown and off the saturated core, where
# moving coordinates moves no surface. `weld`/`link` set how much mass the two closings lay into the
# gaps, which is the knob a deck that reads as separate balls needs; `fuzz` is the high-frequency rim
# raggedness. A closing only reaches a couple of voxels, so when the sheet's lit column still shows
# detached balls the fix is the puff count in _shape_puffs, not a bigger weld. `erode` scales every
# roughening field to the deck's own size and is the deck's footprint inverted, because a feature
# scale shared by nine differently sized clouds is what makes nine shapes read as one shape at nine
# sizes. The remap bites harder than the subtraction it replaced, so these erosion numbers are not
# comparable to older ones. Tune against the contact sheet (--sheet).
FINISH = {
    "cumulus_small.vol":      dict(erosion=0.30, detail=0.25, gamma=1.25, lo=0.22, top_start=0.55,
                                   swirl=0.020, ewarp=0.014, fuzz=0.10, weld=0.45, link=0.45,
                                   wisp=0.26, bottom=0.35, erode=1.15),
    "cumulus_towering.vol":   dict(erosion=0.28, detail=0.22, gamma=1.05, lo=0.16, top_start=0.85,
                                   swirl=0.016, ewarp=0.014, fuzz=0.12, weld=0.70, link=0.50,
                                   wisp=0.22, bottom=0.30, erode=1.25),
    "cumulus_heavy.vol":      dict(erosion=0.26, detail=0.20, gamma=1.00, lo=0.14, top_start=0.72,
                                   swirl=0.012, ewarp=0.010, fuzz=0.08, weld=0.40, link=0.40,
                                   wisp=0.20, bottom=0.40, erode=0.90),
    "cumulus_rising.vol":     dict(erosion=0.30, detail=0.22, gamma=1.15, lo=0.20, top_start=0.85,
                                   swirl=0.018, ewarp=0.014, fuzz=0.12, weld=0.60, link=0.50,
                                   wisp=0.24, bottom=0.30, erode=1.00),
    "cumulus_wispy.vol":      dict(erosion=0.32, detail=0.30, gamma=1.35, lo=0.22, top_start=0.30,
                                   streak=1.9, swirl=0.030, ewarp=0.020, fuzz=0.22, weld=0.35,
                                   link=0.45, wisp=0.32, bottom=0.25, erode=1.00),
    "cumulus_mediocris.vol":  dict(erosion=0.30, detail=0.22, gamma=1.10, lo=0.18, top_start=0.60,
                                   swirl=0.020, ewarp=0.014, fuzz=0.14, weld=0.65, link=0.58,
                                   wisp=0.26, bottom=0.35, erode=0.95),
    "cauliflower.vol":        dict(erosion=0.34, detail=0.28, gamma=1.20, lo=0.22, top_start=0.60,
                                   swirl=0.016, ewarp=0.016, fuzz=0.16, weld=0.62, link=0.45,
                                   wisp=0.30, bottom=0.30, erode=1.00),
    "cauliflower_multi.vol":  dict(erosion=0.28, detail=0.24, gamma=1.10, lo=0.18, top_start=0.65,
                                   swirl=0.020, ewarp=0.016, fuzz=0.16, weld=0.45, link=0.55,
                                   wisp=0.26, bottom=0.30, erode=1.55),
    "storm.vol":              dict(erosion=0.24, detail=0.15, gamma=0.95, lo=0.14, top_start=0.88,
                                   swirl=0.014, ewarp=0.012, fuzz=0.14, weld=0.55, link=0.55,
                                   wisp=0.20, bottom=0.45, erode=0.85),
}


def bake(name):
    """One shape to uint8[RES, RES, RES], indexed [z, y, x] (z vertical, layer 0 the bottom)."""
    seed = int.from_bytes(name.encode(), "little") % 0xFFFF
    rng = np.random.default_rng(seed)
    q = world_grid()
    centers, radii = _shape_puffs(name, rng)
    knobs = FINISH[name]
    t = q[..., 2] / VERTICAL_SCALE
    # The body's own swirl, gated to the crown the same way the erosion warps are: a flat base is part
    # of the authored profile, and a warp that reached it would round the deck into a lens.
    qw = warp(q, seed + 11, amp=knobs["swirl"] * (0.2 + 0.3 * t * t), freq=2.5,
              streak=knobs.get("streak", 1.0))
    d = _puff_field(centers, radii, qw)
    rest = {k: v for k, v in knobs.items() if k != "swirl"}
    d = _finish(d, q, qw, seed, **rest)
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

SURF_STEPS = 128          # x samples a preview ray takes across the box's depth
SURF_THRESHOLD = 0.35     # the density a ray calls the visible surface
SURF_SUN = (0.62, -0.30, 0.72)  # from the camera's right and above, in domain units


def surface_view(d, sun=SURF_SUN):
    """First-hit relief: march each ray until it meets the surface, take the normal from the density
    gradient there, and shade by the sun.

    This is the column the max-projections cannot stand in for. A deck this dense is optically thick,
    so a volume render of it is a flat white disc no matter how much carving it holds — the erosion
    only reads as relief. Normals come from the gradient of the sampled slab, whose x spacing is the
    ray step and whose z spacing is not its y spacing, because the box is VERTICAL_SCALE tall over the
    same sample count it is wide."""
    nz, ny, nx = d.shape
    xs = np.clip(np.round((np.arange(SURF_STEPS, dtype=np.float32) + 0.5) / SURF_STEPS * nx)
                 .astype(np.int32), 0, nx - 1)
    slab = d[:, :, xs]
    step_x = (xs[1] - xs[0]) if SURF_STEPS > 1 else 1.0
    gz, gy, gx = np.gradient(slab, VERTICAL_SCALE / nz, 1.0 / ny, step_x / nx,
                             edge_order=2)
    hit = slab > SURF_THRESHOLD
    first = np.argmax(hit, axis=-1)
    found = hit.any(axis=-1)
    pick = lambda axis: np.take_along_axis(axis, first[..., None], axis=-1)[..., 0]
    # Stacked in the array's own (z, y, x) axis order, which is not the (x, y, z) order the sun arrives
    # in; the gradient is already per domain unit on every axis, so the normal needs no rescaling.
    normal = -np.stack([pick(gz), pick(gy), pick(gx)], axis=-1)
    vdir = np.float32([sun[2] * nz / VERTICAL_SCALE, sun[1] * ny, sun[0] * nx])
    vdir /= np.linalg.norm(vdir)
    length = np.maximum(np.linalg.norm(normal, axis=-1), 1.0e-6)
    n = normal / length[..., None]
    # Wrapped diffuse, so the terminator stays soft the way a cloud's does, plus a sky term that keys
    # off the normal's vertical component: the base of a deck is lit by the whole hemisphere.
    lambert = np.clip((n * vdir).sum(-1) * 0.7 + 0.3, 0.0, 1.0)
    up = np.clip(n[..., 0] * 0.5 + 0.5, 0.0, 1.0)
    shade = 0.22 + 0.62 * lambert + 0.16 * up
    return np.where(found, np.clip(shade, 0.0, 1.0), 0.0).astype(np.float32)


def sheet(volumes, out_path):
    """Three max-projections per shape (silhouettes must read as a cloud from all sides) plus one lit
    surface view, which is what the erosion knobs are actually tuned against.

    The two side columns are drawn at the volume's own proportions — VERTICAL_SCALE of the width —
    and with the base at the bottom, so the sheet shows what the instance box will show. Drawn
    square, a 2:1 shape reads as a tower and every crown looks like a spire."""
    cols = (("max x", VERTICAL_SCALE), ("max y", VERTICAL_SCALE),
            ("max z (top-down)", 1.0), ("lit surface", VERTICAL_SCALE))
    row_h = RES + 18
    img = Image.new("L", (RES * len(cols) + (len(cols) - 1) * 4, row_h * len(volumes) + 4), 40)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    for ri, name in enumerate(volumes):
        d = volumes[name]
        tiles = [np.flipud(d.max(axis=2)), np.flipud(d.max(axis=1)),
                 np.flipud(d.max(axis=0)), np.flipud(surface_view(d))]
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


def shader_volume_res():
    """volume.slang's CLOUD_VOLUME_RES. The shader addresses a height slice as `shape * RES + z`
    inside the atlas, so a volume baked at any other slice count would be read at the wrong height
    and into the wrong shape's band."""
    m = re.search(r"CLOUD_VOLUME_RES\s*=\s*([0-9]*\.?[0-9]+)f?\s*;",
                  VOLUME_SHADER.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"cannot find CLOUD_VOLUME_RES in {VOLUME_SHADER}")
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
    res = shader_volume_res()
    if res != float(RES):
        raise SystemExit(f"RES {RES} != volume.slang CLOUD_VOLUME_RES {res}: the shader would read "
                         f"every height slice at the wrong layer")

    # The erosion field gets its own gate because a degenerate one is invisible everywhere else. A
    # billow field that has lost its scale removes almost nothing, the density still validates, and the
    # shipped volumes just look like smooth uncarved loaves — which is how a broken Worley distance
    # passed a whole release. Probed on an empty grid, so this is the field and not a cloud in it.
    probe = billow3(world_grid(), 4242)
    if not 0.15 <= probe.mean() <= 0.55:
        raise SystemExit(f"billow field mean {probe.mean():.3f} outside 0.15..0.55 — the erosion is "
                         f"not biting, and no density gate below will notice")

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
