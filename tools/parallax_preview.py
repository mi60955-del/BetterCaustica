"""Offline replica of the height-field parallax march, with agent-readable verification views.

Ports `sampleParallax` + `parallaxShadow` from `shaders/pipelines/world/closest_hit.rchit.slang`
to numpy, renders one synthetic block face under a fitted perspective camera, and exposes the
march's internals as debug views. Those views are this tool's own — they inspect the march, not
the game's `composite.debug-view` pass, which exists for the RR guides instead.

Materials are generated here rather than read from a resource pack: pack textures hide the
crossing inside authored noise, while a ramp or a staircase has an answer that can be computed
exactly and scored against. `tools/test_parallax_preview.py` pins the SHADER_* constants to the
slang source; `python tools/parallax_preview.py --check` prints the behavioural verdicts.

This is not a pixel-level replica of the path tracer. The march — every UV, depth, layer count,
wall, and shadow number — is the ported code; the lighting on top is a Lambert stand-in whose
only job is to make the displaced sample visible. The face is planar with a fixed tangent frame
(T=+x, B=+y, n=+z), one sprite = one block face = the full texture, so `pageUvScale` is 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

# ---- Constants mirrored from closest_hit.rchit.slang (test-pinned) --------------------------
SHADER_MIN_LAYERS = 8
SHADER_MAX_LAYERS = 64
SHADER_MIN_VIEW_Z = 0.2
SHADER_MAX_LOD = 2.0
SHADER_WALL_GRADIENT = 0.15
SHADER_SMOOTH_MIP_BIAS = 0.75
SHADER_SMOOTH_WALL = 0.35
SHADER_SHADOW_STEPS = 10
SHADER_SHADOW_MIN_ELEVATION = 0.10
SHADER_SHADOW_SOFTNESS = 0.15
SHADER_SHADOW_BIAS = 0.02
# NORMAL_AO_STRENGTH, same file.
NORMAL_AO_STRENGTH = 0.5

# ---- Defaults mirrored from CausticaConfig.java (test-pinned) -------------------------------
DEFAULT_DEPTH_SCALE = 0.25  # Rt.Composite.PARALLAX_DEPTH, sprite-local UV (0.25 = LabPBR quarter block)
DEFAULT_LAYERS_CAP = 32     # Rt.Materials.PARALLAX_LAYERS quality ceiling
DEFAULT_STYLE = 1           # Rt.Materials.PARALLAX_STYLE_SHARP
DEFAULT_SHADOW = True       # Rt.Materials.PARALLAX_SHADOW

STYLE_SMOOTH = 0
STYLE_SHARP = 1

TEXELS = 16  # vanilla block-face resolution; the march's texel grid is built from this
MAX_MIP = 4  # material max LOD for the preview; the 16x16 mip chain ends at a 1x1 level 4


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


# ---- Texture sampling -----------------------------------------------------------------------
def make_mips(channel: np.ndarray) -> list[np.ndarray]:
    """Box-filter mip pyramid, 2x2 averages down to 1x1 — what a mip chain generation produces.

    Trailing channels (an RGB albedo, an RG normal) ride along untouched.
    """
    mips = [channel.astype(np.float32)]
    while mips[-1].shape[0] > 1 or mips[-1].shape[1] > 1:
        prev = mips[-1]
        h, w = prev.shape[:2]
        extra = [(0, 0)] * (prev.ndim - 2)
        padded = np.pad(prev, ((0, h % 2), (0, w % 2), *extra), mode="edge")
        mips.append(0.25 * (padded[0::2, 0::2] + padded[1::2, 0::2]
                            + padded[0::2, 1::2] + padded[1::2, 1::2]))
    return mips


def _bilinear(mip: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear read with clamped addressing; uv in [0,1] maps to texel centres 0.5..n-0.5."""
    h, w = mip.shape[:2]
    x = np.clip(u * w - 0.5, 0.0, w - 1.0)
    y = np.clip(v * h - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if mip.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = mip[y0, x0] * (1.0 - fx) + mip[y0, x1] * fx
    bot = mip[y1, x0] * (1.0 - fx) + mip[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def sample(mips: list[np.ndarray], uv: np.ndarray, lod) -> np.ndarray:
    """Trilinear SampleLevel with a per-pixel lod array: bilinear at floor and floor+1, blended.

    The uv wraps (fract) before sampling — the port of wrapSpriteUv: a block texture repeats per
    face, so a march pulled past the border reads the opposite edge. The tool's mip chain carries no
    page gutter, so clamped addressing stands in for the packer's wrapped gutter at the seam: the
    border texel reads pure instead of blending with the continuation, a half-texel-wide difference
    from the game at wrap lines.
    """
    uv = np.mod(uv, 1.0)
    lod_arr = np.broadcast_to(np.asarray(lod, np.float32), uv.shape[:2])
    lod_arr = np.clip(lod_arr, 0.0, len(mips) - 1.0)
    l0 = np.floor(lod_arr)
    frac = lod_arr - l0
    out = None
    for level in range(len(mips)):
        sel = l0 == level
        if not sel.any():
            continue
        hi_level = min(level + 1, len(mips) - 1)
        val = lerp(_bilinear(mips[level], uv[..., 0], uv[..., 1]),
                   _bilinear(mips[hi_level], uv[..., 0], uv[..., 1]),
                   frac[..., None] if mips[level].ndim == 3 else frac)
        if out is None:
            out = np.zeros(uv.shape[:2] + val.shape[2:], np.float32)
        out[sel] = val[sel]
    return out


# ---- The march (port of sampleParallax + parallaxShadow) ------------------------------------
def resolve_style(style: int) -> dict:
    # The offset itself is continuous in both styles: quantising it to the texel grid draws closed
    # one-texel jump contours over any field whose crossing depth varies smoothly — on a steep
    # multi-level pack texture those bands read as copies of the texture stacked at whole-texel depths.
    sharp = style == STYLE_SHARP
    return {
        "mip_bias": 0.0 if sharp else SHADER_SMOOTH_MIP_BIAS,
        "wall_scale": 1.0 if sharp else SHADER_SMOOTH_WALL,
    }


def sprite_texel_grid(lod: float) -> int:
    """`spriteTexelGrid` with pageUvScale = 1: texels spanned at the LOD the shading reads at."""
    return max(math.floor(TEXELS / 2.0 ** math.floor(lod) + 0.5), 1)


def march(field_mips: list[np.ndarray], uv0: np.ndarray, vts: np.ndarray, lod: float,
          depth_scale: float = DEFAULT_DEPTH_SCALE, layers_cap: int = DEFAULT_LAYERS_CAP,
          style: int = DEFAULT_STYLE, max_lod: float = MAX_MIP,
          light_ts=None, shadow_on: bool = True) -> dict:
    """Vectorised port of the hit shader's relief walk over one sprite.

    `uv0` is (H, W, 2) sprite-local UV and `vts` (H, W, 3) the view direction in tangent space
    (surface toward the viewer). `lod` is the scalar material LOD the shading reads at; the walk
    itself can land on a coarser `march_lod` per pixel when the travel outruns the layer budget.
    """
    st = resolve_style(style)
    slope_x = vts[..., 0] / np.maximum(np.abs(vts[..., 2]), SHADER_MIN_VIEW_Z)
    slope_y = vts[..., 1] / np.maximum(np.abs(vts[..., 2]), SHADER_MIN_VIEW_Z)

    lod_fade = 1.0 - smoothstep(SHADER_MAX_LOD - 1.0, SHADER_MAX_LOD, lod)
    reach = depth_scale * lod_fade
    active = reach > 1.0e-5

    grid = sprite_texel_grid(lod)
    cap = min(max(int(layers_cap), SHADER_MIN_LAYERS), SHADER_MAX_LAYERS)
    march_lod = min(lod + st["mip_bias"], max_lod)

    # The DDA descends the ray through the staircase field one texel boundary at a time — the port of
    # the shader's walk. Depth t is measured inward from the flat surface as a fraction of the height
    # range; per unit t the sample moves -slope·reach in UV, which is -slope·reach·grid texels.
    vel_x = -slope_x * reach * grid
    vel_y = -slope_y * reach * grid
    pos_x = uv0[..., 0] * grid
    pos_y = uv0[..., 1] * grid
    tex_x = np.floor(pos_x)
    tex_y = np.floor(pos_y)

    def column_depth(tx, ty):
        centre = np.stack([(tx + 0.5) / grid, (ty + 0.5) / grid], axis=-1)
        return 1.0 - sample(field_mips, centre, march_lod)

    field = column_depth(tex_x, tex_y)
    walking = active & (field > 0.0)
    sdir_x = np.sign(vel_x)
    sdir_y = np.sign(vel_y)
    ardir_x = 1.0 / np.maximum(np.abs(vel_x), 1.0e-9)
    ardir_y = 1.0 / np.maximum(np.abs(vel_y), 1.0e-9)
    bound_x = (sdir_x * (0.5 - (pos_x - tex_x)) + 0.5) * ardir_x
    bound_y = (sdir_y * (0.5 - (pos_y - tex_y)) + 0.5) * ardir_y

    t = np.zeros_like(field)
    prev_field = field.copy()
    hit_depth = field.copy()
    wall_hit = np.zeros(walking.shape, bool)
    landed = np.zeros(walking.shape, bool)
    last_mx = np.zeros_like(field)
    last_my = np.zeros_like(field)
    used = np.zeros(walking.shape, np.int32)
    for _ in range(cap):
        cont = walking & ~landed
        if not cont.any():
            break
        b = np.minimum(bound_x, bound_y)
        wall_land = cont & (field <= t)
        flat_land = cont & ~wall_land & (field <= b)
        landed |= wall_land | flat_land
        wall_hit |= wall_land
        # the shadow climbs from the landed column's ceiling — the top of the struck face
        hit_depth = np.where(wall_land | flat_land, field, hit_depth)
        step = cont & ~wall_land & ~flat_land
        if not step.any():
            continue
        used[step] += 1
        t = np.where(step, b, t)
        mx = np.where(step & (bound_x <= bound_y), sdir_x, 0.0)
        my = np.where(step & ~(bound_x <= bound_y), sdir_y, 0.0)
        last_mx = np.where(step, mx, last_mx)
        last_my = np.where(step, my, last_my)
        tex_x = np.where(step, tex_x + mx, tex_x)
        tex_y = np.where(step, tex_y + my, tex_y)
        bound_x = np.where(step & (bound_x <= bound_y), bound_x + ardir_x, bound_x)
        bound_y = np.where(step & ~(bound_x <= bound_y), bound_y + ardir_y, bound_y)
        prev_field = np.where(step, field, prev_field)
        field = np.where(step, column_depth(tex_x, tex_y), field)

    # A ray that never left its starting column (a face-on view) or ran out of budget keeps the flat
    # mapping; a landed one shades its reached column's texel centre.
    moved = (last_mx != 0.0) | (last_my != 0.0)
    quantized = landed & moved
    displaced = np.where(quantized[..., None],
                         np.mod(np.stack([(tex_x + 0.5) / grid, (tex_y + 0.5) / grid], axis=-1), 1.0),
                         uv0)
    offset = displaced - uv0
    offset = offset - np.round(offset)   # the shortest signed difference across the wrap seam

    # A wall is the face the ray crossed to enter the reached column, facing back along that
    # boundary's own axis; its strength is the height step across the face.
    wall_amount = np.where(wall_hit,
                           smoothstep(SHADER_WALL_GRADIENT * 0.33, SHADER_WALL_GRADIENT,
                                      np.abs(field - prev_field)) * st["wall_scale"], 0.0)
    wall_dir = np.where(wall_hit[..., None], -np.stack([last_mx, last_my], axis=-1), 0.0)

    # Directional self-shadow through the same field, walked from the landed texel up to the surface
    # plane. Light at or below the elevation floor stays unshadowed: the traced sun shadow owns that
    # transition.
    shadow = np.ones_like(t)
    if shadow_on and light_ts is not None:
        lz = float(light_ts[2])
        eligible = quantized & (hit_depth > 1.0e-4) & (lz > SHADER_SHADOW_MIN_ELEVATION)
        if eligible.any():
            lateral = np.stack([light_ts[0] / lz * reach, light_ts[1] / lz * reach], axis=-1)
            step_depth = hit_depth / float(SHADER_SHADOW_STEPS)
            sh = np.ones_like(t)
            for i in range(1, SHADER_SHADOW_STEPS + 1):
                climbed = step_depth * float(i)
                sample_uv = displaced + lateral * climbed[..., None]
                sf = 1.0 - sample(field_mips, sample_uv, march_lod)
                blocked = (hit_depth - climbed - sf - SHADER_SHADOW_BIAS) / SHADER_SHADOW_SOFTNESS
                sh = np.minimum(sh, 1.0 - np.clip(blocked, 0.0, 1.0))
            sh = lerp(1.0, sh, smoothstep(SHADER_SHADOW_MIN_ELEVATION,
                                          SHADER_SHADOW_MIN_ELEVATION * 2.0, lz))
            shadow = np.where(eligible, sh, 1.0)

    return {
        "uv": displaced,
        "offset": offset,
        "steps_used": used,
        "layers": np.full(walking.shape, cap, np.int32),
        "march_lod": np.float32(march_lod),
        "reach": np.float32(reach),
        "depth_exit": hit_depth,
        "wall_amount": wall_amount,
        "wall_dir": wall_dir,
        "shadow": shadow,
        "surface_depth": 1.0 - sample(field_mips, displaced, march_lod),
        "crossed": quantized,
    }


def untouched(uv0: np.ndarray) -> dict:
    """The no-relief result: what `untouchedParallax` returns and the gates short-circuit to."""
    h, w = uv0.shape[:2]
    return {
        "uv": uv0.copy(), "offset": np.zeros((h, w, 2), np.float32),
        "steps_used": np.zeros((h, w), np.int32),
        "layers": np.full((h, w), SHADER_MIN_LAYERS, np.int32),
        "march_lod": np.zeros((h, w), np.float32), "reach": np.zeros((h, w), np.float32),
        "depth_exit": np.zeros((h, w), np.float32),
        "wall_amount": np.zeros((h, w), np.float32),
        "wall_dir": np.zeros((h, w, 2), np.float32),
        "shadow": np.ones((h, w), np.float32),
        "surface_depth": np.zeros((h, w), np.float32),
        "crossed": np.zeros((h, w), bool),
    }


# ---- Synthetic materials --------------------------------------------------------------------
def _texel_edge_shade(albedo: np.ndarray, strength: float = 0.82) -> np.ndarray:
    """Darker top/left edge per texel, so a displaced sample visibly shifts off its texel."""
    a = albedo.copy()
    a[0, :] *= strength
    a[:, 0] *= strength
    return a


def _labpbr(alpha: np.ndarray, normal_xy=None, ao=None, albedo=None) -> dict:
    """LabPBR _n layout: RG tangent normal (0.5 centred), B ao, A height (1 flush, 0 deepest)."""
    t = alpha.shape[0]
    return {
        "alpha": alpha.astype(np.float32),
        "normal_xy": np.full((t, t, 2), 0.5, np.float32) if normal_xy is None else normal_xy.astype(np.float32),
        "ao": np.ones((t, t), np.float32) if ao is None else ao.astype(np.float32),
        "albedo": np.full((t, t, 3), 0.62, np.float32) if albedo is None else albedo.astype(np.float32),
    }


def material_stair4() -> dict:
    """Four equal hard treads at depths 0, 1/3, 2/3, 1 across the sprite. Binary edges."""
    x = (np.arange(TEXELS) // (TEXELS // 4)) / 3.0
    alpha = 1.0 - np.tile(x[None, :], (TEXELS, 1))
    depth = 1.0 - alpha
    albedo = _texel_edge_shade((0.55 + 0.18 * (1.0 - depth))[..., None] * np.ones(3)[None, None], 0.8)
    return _labpbr(alpha, albedo=albedo)


def material_ramp() -> dict:
    """Linear ramp, flush at x=0 and deepest at x=1: the analytic-answer material.

    A ray with lateral slope s descending through reach R crosses at depth x0/(1+sR), giving an
    exact displaced UV every pixel can be scored against.
    """
    x = np.linspace(0.0, 1.0, TEXELS, endpoint=False)[None, :] + 0.5 / TEXELS
    alpha = np.repeat(1.0 - x, TEXELS, axis=0)
    depth = 1.0 - alpha
    albedo = _texel_edge_shade((0.45 + 0.4 * depth)[..., None] * np.ones(3)[None, None], 0.85)
    return _labpbr(alpha, albedo=albedo)


def material_checker() -> dict:
    """Flush plateaus alternating with 0.75-deep plateaus: walls on every cell boundary."""
    c = (np.add.outer(np.arange(TEXELS), np.arange(TEXELS)) // (TEXELS // 4)) % 2
    alpha = np.where(c == 0, 1.0, 0.25)
    albedo = _texel_edge_shade(np.repeat(np.where(c[..., None] == 0, 0.7, 0.42), 3, axis=-1), 0.9)
    return _labpbr(alpha, albedo=albedo)


def material_pit() -> dict:
    """A flush face with one square pit at full depth: grazing exposes the far inner wall."""
    alpha = np.ones((TEXELS, TEXELS), np.float32)
    lo, hi = TEXELS // 2 - 2, TEXELS // 2 + 2
    alpha[lo:hi, lo:hi] = 0.0
    albedo = np.full((TEXELS, TEXELS, 3), 0.6, np.float32)
    albedo[lo:hi, lo:hi] = 0.35
    return _labpbr(alpha, albedo=albedo)


def material_groove() -> dict:
    """One-texel plank grooves every four texels, half a block deep, with authored wall normals."""
    alpha = np.ones((TEXELS, TEXELS), np.float32)
    normal = np.full((TEXELS, TEXELS, 2), 0.5, np.float32)
    ao = np.ones((TEXELS, TEXELS), np.float32)
    for g in range(1, TEXELS, 4):
        alpha[:, g] = 0.5
        ao[:, g] = 0.55
        normal[:, (g + 1) % TEXELS, 0] = 0.25  # walls lean back out of the groove
        normal[:, (g - 1) % TEXELS, 0] = 0.75
    albedo = np.full((TEXELS, TEXELS, 3), 0.62, np.float32)
    albedo[:, 1::4] = 0.4
    return _labpbr(alpha, normal, ao, albedo)


def material_dune() -> dict:
    """Smooth rounded relief whose per-texel gradient stays far under the wall threshold."""
    t = np.arange(TEXELS) / TEXELS
    depth = 0.5 + 0.15 * np.sin(2 * np.pi * 2 * t)[None, :] * np.sin(2 * np.pi * 2 * t)[:, None]
    alpha = 1.0 - depth
    gy, gx = np.gradient(depth)
    normal = np.stack([0.5 - gx * 2.2, 0.5 + gy * 2.2], axis=-1)
    nlen = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.clip(normal / np.maximum(nlen, 1.0e-6) * 0.35 + 0.5, 0.0, 1.0)
    albedo = _texel_edge_shade((0.5 + 0.25 * depth)[..., None] * np.ones(3)[None, None], 0.88)
    return _labpbr(alpha, normal, None, albedo)


def material_ratchet() -> dict:
    """One-texel sawtooth: forces the layer budget and the coarsen path at grazing views."""
    x = np.arange(TEXELS) / TEXELS
    depth = np.tile(x[None, :], (TEXELS, 1))
    albedo = _texel_edge_shade((0.5 + 0.3 * depth)[..., None] * np.ones(3)[None, None], 0.85)
    return _labpbr(1.0 - depth, albedo=albedo)


MATERIALS = {
    "stair4": material_stair4,
    "ramp": material_ramp,
    "checker": material_checker,
    "pit": material_pit,
    "groove": material_groove,
    "dune": material_dune,
    "ratchet": material_ratchet,
}


def material_mips(mat: dict) -> dict:
    """Per-channel mip pyramids; the normal's two channels filter independently."""
    return {
        "alpha": make_mips(mat["alpha"]),
        "albedo": make_mips(mat["albedo"]),
        "ao": make_mips(mat["ao"]),
        "normal_x": make_mips(mat["normal_xy"][..., 0]),
        "normal_y": make_mips(mat["normal_xy"][..., 1]),
    }


# ---- Perspective camera over one block face -------------------------------------------------
VIEWS = {
    # name: (tilt from the face normal in degrees, azimuth in degrees, eye distance, lod)
    "front": (0.0, 0.0, 2.6, 0.0),
    "tilt15": (15.0, 0.0, 2.6, 0.0),
    "tilt30": (30.0, 0.0, 2.6, 0.0),
    "tilt45": (45.0, 0.0, 2.6, 0.0),
    "tilt65": (65.0, 20.0, 2.6, 0.0),
    "grazing": (78.0, 30.0, 3.4, 0.0),
    "close": (35.0, 15.0, 1.15, 0.0),
    "far": (30.0, 0.0, 16.0, 1.5),
    "walltest": (55.0, 25.0, 2.2, 0.0),
}


def camera_rays(view: str, res: int):
    """Fit a perspective camera to the face [-0.5,0.5]^2 in the plane z=0, normal +z.

    The tangent frame is T=+x, B=+y, n=+z, so the tangent-space view direction is the world one.
    Returns uv0 (H,W,2), vts (H,W,3), a valid (H,W) mask for pixels that hit the face, and the
    view's lod. Perspective is the point: off-axis pixels tilt the view exactly like the game's
    camera does, which is where a head-on face still shows relief near its frame edges.
    """
    tilt, az, dist, lod = VIEWS[view]
    th, ph = math.radians(tilt), math.radians(az)
    eye = dist * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    corners = np.array([[x, y, 0.0] for x in (-0.5, 0.5) for y in (-0.5, 0.5)])
    rel = corners - eye
    depth_c = rel @ fwd
    proj_x = (rel @ right) / depth_c
    proj_y = (rel @ up) / depth_c
    span = 1.06 * max(proj_x.max() - proj_x.min(), proj_y.max() - proj_y.min())
    center = np.array([(proj_x.max() + proj_x.min()) / 2.0, (proj_y.max() + proj_y.min()) / 2.0])

    px = (np.arange(res) + 0.5) / res
    u, v = np.meshgrid(px, px)
    ray = (fwd[None, None]
           + (center[0] + (u - 0.5) * span)[..., None] * right
           + (center[1] + (v - 0.5) * span)[..., None] * up)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -eye[2] / ray[..., 2]
        hit = eye[None, None] + t[..., None] * ray
        valid = np.isfinite(t) & (t > 0) & (np.abs(hit[..., 0]) <= 0.5) & (np.abs(hit[..., 1]) <= 0.5)
    uv0 = np.stack([hit[..., 0] + 0.5, hit[..., 1] + 0.5], axis=-1)
    vts = eye[None, None] - hit
    vts /= np.maximum(np.linalg.norm(vts, axis=-1, keepdims=True), 1.0e-9)
    return uv0, vts, valid, lod


def sun_tangent(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Sun direction in the face's tangent frame from elevation above the plane and azimuth."""
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


# ---- Shading (illustrative Lambert; the march itself is the exact part) ---------------------
def shade(mat: dict, mips: dict, res: dict, vts: np.ndarray, light: np.ndarray,
          lod: float, ambient: float = 0.35) -> np.ndarray:
    """Albedo at the displaced UV under a Lambert sun, with walls and self-shadow folded in."""
    nxy = np.stack([sample(mips["normal_x"], res["uv"], lod),
                    sample(mips["normal_y"], res["uv"], lod)], axis=-1)
    nxy = nxy * 2.0 - 1.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - np.sum(nxy * nxy, axis=-1)))
    nm = np.concatenate([nxy, nz[..., None]], axis=-1)
    nm /= np.maximum(np.linalg.norm(nm, axis=-1, keepdims=True), 1.0e-9)
    wall = res["wall_amount"][..., None]
    wall_n = np.concatenate([res["wall_dir"], np.zeros((*res["wall_dir"].shape[:2], 1))], axis=-1)
    wall_n /= np.maximum(np.linalg.norm(wall_n, axis=-1, keepdims=True), 1.0e-9)
    nm = nm * (1.0 - wall) + wall_n * wall
    nm /= np.maximum(np.linalg.norm(nm, axis=-1, keepdims=True), 1.0e-9)
    nov = np.sum(nm * vts, axis=-1)
    flip = nov < 0.02
    if flip.any():
        nm[flip] = nm[flip] + vts[flip] * (0.02 - nov[flip])[:, None]
        nm[flip] /= np.linalg.norm(nm[flip], axis=-1, keepdims=True)

    albedo = sample(mips["albedo"], res["uv"], lod)
    ao = lerp(1.0, sample(mips["ao"], res["uv"], lod), NORMAL_AO_STRENGTH)
    diffuse = np.maximum(np.sum(nm * light[None, None], axis=-1), 0.0)
    col = albedo * (ambient + diffuse * res["shadow"])[..., None] * ao[..., None]
    return np.clip(col, 0.0, 1.0)


def shade_flat(mat: dict, mips: dict, uv0: np.ndarray, light: np.ndarray, lod: float,
               ambient: float = 0.35) -> np.ndarray:
    """The march-off baseline: flat mapping, flat normal, no walls, no self-shadow."""
    albedo = sample(mips["albedo"], uv0, lod)
    return np.clip(albedo * (ambient + max(float(light[2]), 0.0)), 0.0, 1.0)


# ---- Debug views ----------------------------------------------------------------------------
def _gray(x, lo=0.0, hi=1.0):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


DEBUG_VIEWS = ("shaded", "flat", "diff", "height", "steps", "offset", "wall", "shadow", "analytic")


def debug_image(name: str, mat: dict, mips: dict, res: dict, uv0: np.ndarray, vts: np.ndarray,
                valid: np.ndarray, light: np.ndarray, lod: float, grid: int,
                style: int) -> np.ndarray | None:
    """One debug view as an (H,W,3) display image; neutral grey where no face was hit."""
    if name == "shaded":
        col = shade(mat, mips, res, vts, light, lod)
    elif name == "flat":
        col = shade_flat(mat, mips, uv0, light, lod)
    elif name == "diff":
        col = np.clip(np.abs(shade(mat, mips, res, vts, light, lod)
                             - shade_flat(mat, mips, uv0, light, lod)) * 6.0, 0.0, 1.0)
    elif name == "height":
        col = np.repeat(_gray(res["surface_depth"])[..., None], 3, -1)
    elif name == "steps":
        g = _gray(res["steps_used"].astype(np.float32) / float(SHADER_MAX_LAYERS))
        col = np.where(res["crossed"][..., None],
                       np.stack([g, 0.35 + 0.4 * g, 1.0 - g], axis=-1), 0.08)
    elif name == "offset":
        tex = np.linalg.norm(res["offset"], axis=-1) * grid
        g = _gray(tex, 0.0, 8.0)
        col = np.stack([g, 1.0 - g, np.zeros_like(g)], axis=-1)
    elif name == "wall":
        g = _gray(res["wall_amount"])
        col = np.stack([g, g * 0.6, np.zeros_like(g)], axis=-1)
    elif name == "shadow":
        col = np.repeat(_gray(res["shadow"])[..., None], 3, -1)
    elif name == "analytic":
        if mat["name"] != "ramp":
            return None
        err, mask = ramp_analytic_error(uv0, vts, res, grid, style, lod)
        col = np.stack([_gray(err, 0.0, 1.0), mask.astype(np.float32) * 0.9, np.zeros_like(err)], axis=-1)
    else:
        return None
    out = np.full((*uv0.shape[:2], 3), 0.16, np.float32)
    out[valid] = col[valid]
    return out


def ramp_analytic_error(uv0, vts, res, grid, style, lod):
    """Exact crossing depth for the ramp, scored on the DDA's landed depth in texels of travel.

    The ramp's depth is x itself, so a ray with lateral slope s descending through reach R meets
    it at depth x0/(1+sR); the y drift is unconstrained (the field is y-independent) and is not
    scored. The staircase march lands within one column of that crossing, so the landed depth's
    travel error must stay under a texel. Valid only where the model is exact: sharp style and a
    crossing whose whole walk stays inside the sprite (the columns wrap across the border, and the
    closed form assumes an unwrapped field).
    """
    x0 = uv0[..., 0]
    s = vts[..., 0] / np.maximum(np.abs(vts[..., 2]), SHADER_MIN_VIEW_Z)
    reach = res["reach"]
    exact_d = x0 / (1.0 + s * reach)
    err = np.abs(res["depth_exit"] - exact_d) * (np.abs(s) * reach * grid)
    mask = (style == STYLE_SHARP) & res["crossed"] & (x0 > 0.15) & (x0 < 0.85)
    mask &= (exact_d > 0.02) & (exact_d < 1.0) & ((1.0 + s * reach) > 0.2)
    exact_x = x0 - s * reach * exact_d
    mask &= (exact_x > 0.02) & (exact_x < 0.98)
    return err, mask


# ---- Sheets ---------------------------------------------------------------------------------
def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def label_panel(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(to_u8(img))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(20, 20, 20))
    d.text((4, 2), text, fill=(230, 230, 230))
    return im


def sheet(panels: list[Image.Image], cols: int, pad: int = 4) -> Image.Image:
    rows = math.ceil(len(panels) / cols)
    pw = max(p.width for p in panels) + pad
    ph = max(p.height for p in panels) + pad
    out = Image.new("RGB", (cols * pw + pad, rows * ph + pad), (30, 30, 34))
    for i, p in enumerate(panels):
        out.paste(p, (pad + (i % cols) * pw, pad + (i // cols) * ph))
    return out


# ---- Per-case run + metrics -----------------------------------------------------------------
def run_case(mat_key: str, view: str, style: int = DEFAULT_STYLE,
             depth_scale: float = DEFAULT_DEPTH_SCALE, layers_cap: int = DEFAULT_LAYERS_CAP,
             shadow_on: bool = DEFAULT_SHADOW, lod: float | None = None,
             sun_elevation: float = 35.0, sun_azimuth: float = 65.0,
             res_px: int = 384) -> dict:
    """March one material under one view and carry everything the views and metrics need."""
    mat = MATERIALS[mat_key]()
    mat["name"] = mat_key
    mips = material_mips(mat)
    uv0, vts, valid, view_lod = camera_rays(view, res_px)
    lod = view_lod if lod is None else lod
    light = sun_tangent(sun_elevation, sun_azimuth)
    grid = sprite_texel_grid(lod)

    if depth_scale <= 0.0:
        res = untouched(uv0)
    else:
        res = march(mips["alpha"], uv0, vts, lod, depth_scale, layers_cap, style,
                    light_ts=light, shadow_on=shadow_on)
    return {"mat": mat, "mips": mips, "res": res, "valid": valid, "light": light,
            "uv0": uv0, "vts": vts, "lod": lod, "grid": grid, "style": style}


def case_metrics(case: dict) -> dict:
    res, valid = case["res"], case["valid"]
    sel = valid
    steps = res["steps_used"][sel].astype(np.float32)
    tex = np.linalg.norm(res["offset"], axis=-1)[sel] * case["grid"]
    wall = res["wall_amount"][sel]
    shadow = res["shadow"][sel]
    metrics = {
        "pixels": int(valid.sum()),
        "displaced_fraction": float(np.mean(res["crossed"][sel])) if sel.any() else 0.0,
        "mean_steps": float(steps.mean()) if steps.size else 0.0,
        "p95_steps": float(np.percentile(steps, 95)) if steps.size else 0.0,
        "max_steps": int(steps.max()) if steps.size else 0,
        "mean_offset_texels": float(tex.mean()) if tex.size else 0.0,
        "max_offset_texels": float(tex.max()) if tex.size else 0.0,
        "wall_fraction": float(np.mean(wall > 0.0)) if wall.size else 0.0,
        "max_wall": float(wall.max()) if wall.size else 0.0,
        "min_shadow": float(shadow.min()) if shadow.size else 1.0,
        "mean_march_lod": float(res["march_lod"]),
    }
    if case["mat"]["name"] == "ramp":
        err, mask = ramp_analytic_error(case["uv0"], case["vts"], res, case["grid"],
                                        case["style"], case["lod"])
        if mask.any():
            metrics["analytic_mean_err_texels"] = float(err[mask].mean())
            metrics["analytic_max_err_texels"] = float(err[mask].max())
            metrics["analytic_valid_px"] = int(mask.sum())
    return metrics


# ---- Built-in checks ------------------------------------------------------------------------
def run_checks() -> list[dict]:
    """Decidable verdicts an agent can read without judging a picture. Returns check records."""
    report: list[dict] = []

    def record(name, passed, detail):
        report.append({"name": name, "pass": bool(passed), "detail": detail})
        print(("PASS " if passed else "FAIL ") + name + "  " + detail)

    # 1. A flush field is never displaced, in any view.
    uv0, vts, valid, lod = camera_rays("grazing", 192)
    r = march(make_mips(np.ones((TEXELS, TEXELS), np.float32)), uv0, vts, lod)
    record("flush-untouched", not r["crossed"][valid].any(),
           f"crossed px = {int(r['crossed'][valid].sum())}")

    # 2. Head-on, the view axis hits the face flat: near-axis pixels must not visibly move.
    #    Perspective still tilts pixels away from the axis, so the bound is a visibility one.
    c = run_case("stair4", "front", style=STYLE_SMOOTH, layers_cap=SHADER_MAX_LAYERS)
    axis = c["valid"] & (c["vts"][..., 2] > 0.9999)
    near = float(np.max(np.linalg.norm(c["res"]["offset"][axis], axis=-1)) * c["grid"]) if axis.any() else -1.0
    record("front-axis-still", axis.any() and near < 0.1,
           f"near-axis offset = {near:.2e} texels over {int(axis.sum())} px")

    # 3. The ramp march lands within a column of the analytic crossing (sharp style).
    c = run_case("ramp", "tilt45", style=STYLE_SHARP, layers_cap=SHADER_MAX_LAYERS)
    err, mask = ramp_analytic_error(c["uv0"], c["vts"], c["res"], c["grid"], c["style"], c["lod"])
    good = mask.any() and err[mask].mean() < 0.5 and err[mask].max() < 1.0
    record("ramp-analytic", good,
           f"mean {err[mask].mean() if mask.any() else float('nan'):.4f} tex, "
           f"max {err[mask].max() if mask.any() else float('nan'):.4f} tex over {int(mask.sum())} px")

    # 4. Displacement scales with the depth setting and grows with tilt. Halving the depth also
    #    drops more pixels under the softness knee, so the mean lands slightly above half.
    m = [run_case("ramp", v, style=STYLE_SMOOTH, depth_scale=d)["res"]["offset"][..., 0]
         for v, d in (("tilt30", 0.25), ("tilt45", 0.25), ("tilt45", 0.125))]
    scale = float(np.mean(np.abs(m[2]))) / max(float(np.mean(np.abs(m[1]))), 1e-9)
    tilt = float(np.mean(np.abs(m[1]))) / max(float(np.mean(np.abs(m[0]))), 1e-9)
    record("offset-scaling", 0.45 < scale < 0.62 and tilt > 1.1,
           f"depth halving ratio = {scale:.3f}, tilt45/tilt30 ratio = {tilt:.3f}")

    # 5. The layer budget clamps at the cap on a grazing binary field.
    c = run_case("ratchet", "grazing", layers_cap=SHADER_MIN_LAYERS)
    sel = c["valid"] & c["res"]["crossed"]
    record("layers-cap", bool(sel.any() and (c["res"]["layers"][sel] == SHADER_MIN_LAYERS).all()),
           f"layers uniq = {np.unique(c['res']['layers'][sel])}")

    # 6. Binary steps grow full walls under sharp style; the dune's gentle stairs claim only a
    #    partial wall (its per-column step clears the ramp only partway, and smooth style scales
    #    that down).
    c = run_case("stair4", "walltest", style=STYLE_SHARP)
    stair = float(c["res"]["wall_amount"][c["valid"]].max())
    d = run_case("dune", "walltest", style=STYLE_SMOOTH)
    dune = float(d["res"]["wall_amount"][d["valid"]].max())
    record("wall-response", stair > 0.85 and dune < 0.3, f"stair max wall {stair:.2f}, dune max wall {dune:.3f}")

    # 7. Self-shadow: the deep treads shade when the sun comes across them, not along them, and a
    #    sun at the elevation floor casts nothing (the traced sun shadow owns that transition). The
    #    walk starts at each landed column's ceiling, so contact shading follows the staircase.
    across = run_case("stair4", "tilt30", shadow_on=True, sun_elevation=40.0, sun_azimuth=0.0)
    a_sh = across["res"]["shadow"][across["valid"]]
    along = run_case("stair4", "tilt30", shadow_on=True, sun_elevation=40.0, sun_azimuth=90.0)
    l_sh = along["res"]["shadow"][along["valid"]]
    low = run_case("stair4", "tilt30", shadow_on=True, sun_elevation=3.0, sun_azimuth=0.0)
    lo_sh = low["res"]["shadow"][low["valid"]]
    record("self-shadow", float(a_sh.min()) < 0.3 and np.mean(a_sh < 0.9) > 0.05
           and np.percentile(l_sh, 5) > 0.99 and float(lo_sh.min()) > 0.999,
           f"across min {a_sh.min():.3f} shadowed {np.mean(a_sh < 0.9):.3f}, "
           f"along p05 {np.percentile(l_sh, 5):.3f}, low-sun min {lo_sh.min():.3f}")

    # 8. Tiling: field samples wrap across the sprite border (a block texture repeats per face, so
    #    the wrapped texel is what the neighbouring face shows), and the reach holds full strength to
    #    the border — a reach fade there would compress the border band into a curling flap. The
    #    probe reads stair4's alpha just left of x=0: clamped addressing would return the flush
    #    tread's alpha (1.0), the wrap returns the deep tread across the seam (0.0). The relief is
    #    also required to reach the border band: the DDA quantizes there like anywhere else.
    mips = material_mips(MATERIALS["stair4"]())
    probe = float(sample(mips["alpha"], np.array([[[-0.02, 0.5]]], np.float32), 0.0)[0, 0])
    c = run_case("stair4", "grazing", style=STYLE_SHARP)
    edge = np.minimum(c["uv0"][..., 0], 1.0 - c["uv0"][..., 0]) < 2.0 / 16.0
    border_crossed = float(np.mean(c["res"]["crossed"][c["valid"] & edge]))
    record("tiling", probe < 0.1 and border_crossed > 0.5,
           f"wrap field {probe:.2f} (clamp would read 1.0), border crossed {border_crossed:.2f}")

    # 9. depth 0 leaves every pixel untouched — the A/B baseline the game documents.
    c = run_case("stair4", "tilt65", depth_scale=0.0)
    record("depth-zero-off", not c["res"]["crossed"][c["valid"]].any(),
           f"crossed px = {int(c['res']['crossed'][c['valid']].sum())}")

    # 10. Smooth reads SHADER_SMOOTH_MIP_BIAS coarser than sharp on identical geometry.
    cs = run_case("dune", "tilt45", style=STYLE_SHARP)
    cm = run_case("dune", "tilt45", style=STYLE_SMOOTH)
    delta = cm["res"]["march_lod"] - cs["res"]["march_lod"]
    record("smooth-mip-bias", bool(np.allclose(delta, SHADER_SMOOTH_MIP_BIAS, atol=1e-4)),
           f"lod delta uniq = {np.unique(np.round(delta, 4))}")

    return report


# ---- CLI ------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", default="all",
                        help="comma list or 'all' of: " + ",".join(MATERIALS))
    parser.add_argument("--view", default="front,tilt30,tilt65,grazing,walltest",
                        help="comma list or 'all' of: " + ",".join(VIEWS))
    parser.add_argument("--debug", default="shaded,steps,offset",
                        help="comma list of: " + ",".join(DEBUG_VIEWS))
    parser.add_argument("--style", default="sharp", choices=["sharp", "smooth"])
    parser.add_argument("--depth", type=float, default=DEFAULT_DEPTH_SCALE,
                        help="parallaxDepth in sprite-local UV (game default 0.25, settings UI 25%%)")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS_CAP, help="quality cap, 8..64")
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--lod", type=float, default=None, help="override the view's material LOD")
    parser.add_argument("--sun-elevation", type=float, default=35.0, help="degrees above the face plane")
    parser.add_argument("--sun-azimuth", type=float, default=65.0, help="degrees around the face normal")
    parser.add_argument("--res", type=int, default=384, help="render resolution per view")
    parser.add_argument("--ab", choices=["depth", "style", "shadow", "layers"], default=None,
                        help="side-by-side A/B of one setting, shaded view")
    parser.add_argument("--orbit", action="store_true", help="8-azimuth shaded sweep at tilt 45")
    parser.add_argument("--check", action="store_true", help="run the behavioural checks and exit")
    parser.add_argument("--report", default=None, help="write case metrics as JSON to this path")
    parser.add_argument("--out", default="tools/parallax-preview-out")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    mats = list(MATERIALS) if args.material == "all" else args.material.split(",")
    views = list(VIEWS) if args.view == "all" else args.view.split(",")
    style = STYLE_SHARP if args.style == "sharp" else STYLE_SMOOTH
    shadow_on = not args.no_shadow
    os.makedirs(args.out, exist_ok=True)
    print(f"parallax preview: depth {args.depth} layers {args.layers} style {args.style} "
          f"shadow {shadow_on} res {args.res}")

    if args.check:
        report = run_checks()
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump({"checks": report}, f, indent=1)
            print(f"wrote {args.report}")
        raise SystemExit(0 if all(r["pass"] for r in report) else 1)

    def variants_for(axis: str):
        return {
            "depth": [("off", dict(depth_scale=0.0)), ("on", dict(depth_scale=args.depth))],
            "style": [("sharp", dict(style=STYLE_SHARP)), ("smooth", dict(style=STYLE_SMOOTH))],
            "shadow": [("on", dict(shadow_on=True)), ("off", dict(shadow_on=False))],
            "layers": [("min8", dict(layers_cap=SHADER_MIN_LAYERS)),
                       ("cap64", dict(layers_cap=SHADER_MAX_LAYERS))],
        }[axis]

    if args.orbit:
        for mkey in mats:
            panels = []
            for az in range(0, 360, 45):
                oc = run_case(mkey, "tilt45", style, args.depth, args.layers, shadow_on,
                              args.lod, args.sun_elevation, float(az), args.res)
                img = debug_image("shaded", oc["mat"], oc["mips"], oc["res"], oc["uv0"], oc["vts"],
                                  oc["valid"], oc["light"], oc["lod"], oc["grid"], oc["style"])
                panels.append(label_panel(img, f"az {az}"))
            if not args.no_write:
                path = os.path.join(args.out, f"{mkey}-orbit.png")
                sheet(panels, 4).save(path)
                print(f"wrote {path}")
        return

    metrics_rows = []
    for mkey in mats:
        for view in views:
            case = run_case(mkey, view, style, args.depth, args.layers, shadow_on,
                            args.lod, args.sun_elevation, args.sun_azimuth, args.res)
            row = {"material": mkey, "view": view, "style": args.style, "depth": args.depth,
                   "layers_cap": args.layers, "shadow": shadow_on}
            row.update(case_metrics(case))
            metrics_rows.append(row)
            print(f"{mkey:>8} {view:>9}: steps mean {row['mean_steps']:5.1f} p95 {row['p95_steps']:5.1f} "
                  f"max {row['max_steps']:3d} | offset mean {row['mean_offset_texels']:5.2f} max "
                  f"{row['max_offset_texels']:5.2f} tex | wall {row['wall_fraction']:4.2f} | "
                  f"shadow min {row['min_shadow']:5.3f} | displaced {row['displaced_fraction']:4.2f}"
                  + (f" | analytic max {row['analytic_max_err_texels']:.3f} tex"
                     if "analytic_max_err_texels" in row else ""))

            if args.ab:
                panels = []
                for label, kw in variants_for(args.ab):
                    full = dict(style=style, depth_scale=args.depth, layers_cap=args.layers,
                                shadow_on=shadow_on)
                    full.update(kw)
                    vc = run_case(mkey, view, lod=args.lod, sun_elevation=args.sun_elevation,
                                  sun_azimuth=args.sun_azimuth, res_px=args.res, **full)
                    img = debug_image("shaded", vc["mat"], vc["mips"], vc["res"], vc["uv0"],
                                      vc["vts"], vc["valid"], vc["light"], vc["lod"], vc["grid"],
                                      vc["style"])
                    panels.append(label_panel(img, label))
                if not args.no_write:
                    path = os.path.join(args.out, f"{mkey}-{view}-ab-{args.ab}.png")
                    sheet(panels, 2).save(path)
                    print(f"  wrote {path}")
                continue

            panels = []
            for dbg in args.debug.split(","):
                img = debug_image(dbg, case["mat"], case["mips"], case["res"], case["uv0"],
                                  case["vts"], case["valid"], case["light"], case["lod"],
                                  case["grid"], case["style"])
                if img is None:
                    print(f"  (no {dbg} view for {mkey})")
                    continue
                panels.append(label_panel(img, f"{mkey} {view} {dbg}"))
            if panels and not args.no_write:
                path = os.path.join(args.out, f"{mkey}-{view}-sheet.png")
                sheet(panels, len(panels)).save(path)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"cases": metrics_rows}, f, indent=1)
        print(f"wrote {args.report}")
    if not args.no_write:
        print(f"wrote sheets to {args.out}/")


if __name__ == "__main__":
    main()
