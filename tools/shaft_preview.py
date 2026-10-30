"""Offline preview and cost model of the sun-shaft term (shaders/pipelines/world/volume.slang).

Two questions, both answered without launching the game:

  Is it a volumetric effect?   The shaft term integrates light along the view segment, but its medium
                               is constant and its light is gated by ray-traced sun visibility:
                               `L = structure * strength * phase * E * sum_k(sigma*T_k*V_k)` with
                               `structure = 1 - mean(V)`. This tool renders the segment with the fog
                               density at zero (shafts on) and again with the visibility forced to one.
                               A uniform scattering medium still lights the path under that control;
                               this term must return exactly zero, because the gate is then zero. What
                               survives the fog being off, and vanishes when the shadow probes are
                               silenced, is shadow structure, not air.

  What does it cost?           One shaft call casts SHAFT_STRATA = 8 shadow rays and evaluates the
                               cloud-shadow field 8 times, at the same call sites and under the same
                               gating as the analytic fog's FOG_SUN_BANDS = 4. The tool counts the
                               rays it actually casts, re-counts them at the reach and density
                               extremes to show the count is per-pixel constant, and audits the
                               source: every shaft call sits behind CAUSTICA_VOLUME_PATH, and the
                               volume-free raygen is only selected while shafts are off.

Constants prefixed `SHADER_` are parsed and pinned against the shader source by test_shaft_preview.py,
and the config defaults against CausticaConfig.java the same way. Values are float32 throughout,
matching the shader's arithmetic closely enough to judge shape, energy and probe budget. The preview
is a shape/energy/cost model, not a bit-exact reference: the sky LUT is replaced by an analytic
gradient, the cloud sheet by a two-octave stand-in, and the underwater branches of the integrator are
verified numerically by the test rather than rendered.

Usage:
    python tools/shaft_preview.py                      # report + 4x2 sheet into tools/shaft-preview-out/
    python tools/shaft_preview.py --budget-only        # source audit and probe budget, no rendering
    python tools/shaft_preview.py --width 960 --height 540 --frames 8
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLUME_SLANG = os.path.join(REPO_ROOT, "shaders", "pipelines", "world", "volume.slang")
PRIMARY_SLANG = os.path.join(REPO_ROOT, "shaders", "pipelines", "world", "primary.rgen.slang")
INDIRECT_SLANG = os.path.join(REPO_ROOT, "shaders", "pipelines", "world", "indirect.rgen.slang")
COMPOSITE_JAVA = os.path.join(
    REPO_ROOT, "src", "main", "java", "dev", "comfyfluffy", "caustica", "rt", "RtComposite.java")
CONFIG_JAVA = os.path.join(
    REPO_ROOT, "src", "main", "java", "dev", "comfyfluffy", "caustica", "CausticaConfig.java")
BUILD_GRADLE = os.path.join(REPO_ROOT, "build.gradle")

# --- mirrored from shaders/pipelines/world/volume.slang ------------------------------------------------
SHADER_SHAFT_STRATA = 8                     # :1260; strata along the segment, one shadow ray each
SHAFT_STRATA = SHADER_SHAFT_STRATA          # the marcher's own name for the count
SHADER_SHAFT_PHASE_ENERGY = 0.7             # :1263
SHADER_WATER_SHAFT_STRUCTURE_FLOOR = 0.45   # :1271
SHADER_FOG_SUN_BANDS = 4                    # :1076; the analytic fog tier's probes per segment
SHADER_FOG_MARCH_PROBE_STRATA = 8           # :895; the marched fog tier's probe budget per segment
SHADER_CLOUD_EXTINCTION = 0.22              # :257, 1/block at unit density
SHADER_PHASE_ENERGY = 0.7                   # the `* 0.7` in both integrators' phase
SHADER_JITTER_STEP = 0.6180339887           # the golden-ratio frame walk
SHADER_RAY_TMIN = 1.0e-3                    # trace.slang; a shorter segment integrates nothing

# --- config defaults, mirrored from CausticaConfig.java (Rt.Volume) -------------------------------------
DEFAULT_SHAFT_DENSITY = 0.010               # 1/block
DEFAULT_SHAFT_ANISOTROPY = 0.70
DEFAULT_SHAFT_DISTANCE = 192.0              # blocks
DEFAULT_SHAFT_INTENSITY = 0.6
DEFAULT_WATER_SHAFT_DENSITY = 0.03
DEFAULT_WATER_SHAFT_CEILING = 0.10
DEFAULT_FOG_DENSITY = 0.02                  # before the Java-side sun-elevation factor
DEFAULT_FOG_HEIGHT = 24.0
DEFAULT_FOG_ALBEDO = 0.85
DEFAULT_FOG_ANISOTROPY = 0.50
DEFAULT_FOG_AMBIENT = 1.0
DEFAULT_FOG_SHADOW = True
DEFAULT_FOG_SHADOW_FLOOR = 0.10
DEFAULT_FOG_MARCH_STEPS = 16

# Rendering constants of this port only. The sun's illuminance has no absolute meaning here: the arms
# are compared against each other, so the ratios (surface, fog in-scatter, shaft in-scatter) are what
# the images and the report read.
SUN_ILLUMINANCE = np.array([3.2, 3.05, 2.85], dtype=F32)
SKY_HORIZON = np.array([0.55, 0.62, 0.72], dtype=F32)
SKY_ZENITH = np.array([0.14, 0.24, 0.50], dtype=F32)
SURFACE_AMBIENT = F32(0.30)
SHADOW_RAY_LENGTH = 10000.0


# --- small math ---------------------------------------------------------------------------------------
def frac(x):
    return x - np.floor(x)


def saturate(x):
    return np.clip(x, 0.0, 1.0)


def lerp(a, b, t):
    return a + (b - a) * t


def luminance(rgb):
    return (F32(0.2126) * rgb[..., 0] + F32(0.7152) * rgb[..., 1] + F32(0.0722) * rgb[..., 2])


def hg(cos_theta, g):
    """Henyey-Greenstein phase, shaders/pipelines/world/math.slang:43; integral over the sphere = 1."""
    denom = np.maximum(F32(0.0), 1.0 + g * g - 2.0 * g * cos_theta)
    return ((1.0 - g * g) / (4.0 * math.pi * denom ** 1.5)).astype(F32)


def sky_radiance(rd):
    """Stand-in for the sky view LUT: a zenith-to-horizon gradient."""
    up = saturate(rd[..., 1])
    return lerp(SKY_HORIZON, SKY_ZENITH, up[..., None] ** 0.45)


# --- geometry -----------------------------------------------------------------------------------------
class Scene:
    """Axis-aligned boxes over a ground plane. `water_y` is None for the air scene."""

    def __init__(self, boxes, albedos, ground_y, ground_albedo, water_y=None):
        self.lo = np.array([b[0] for b in boxes], dtype=F32).reshape(-1, 3)
        self.hi = np.array([b[1] for b in boxes], dtype=F32).reshape(-1, 3)
        self.albedo = np.array(albedos, dtype=F32).reshape(-1, 3)
        self.ground_y = F32(ground_y)
        self.ground_albedo = np.array(ground_albedo, dtype=F32)
        self.water_y = None if water_y is None else F32(water_y)


def build_air_scene():
    """A blind-window room, backlit: three horizontal slits in the far wall plus two occluders."""
    boxes = []
    albedos = []
    wall_z0, wall_z1 = 6.0, 6.6
    for y0, y1 in ((0.0, 1.1), (1.5, 2.4), (2.8, 3.7), (4.1, 6.2)):
        boxes.append(((-9.0, y0, wall_z0), (9.0, y1, wall_z1)))
        albedos.append((0.55, 0.54, 0.52))
    for x in (-9.4, 9.0):
        boxes.append(((x, 0.0, -12.0), (x + 0.4, 6.2, wall_z1)))
        albedos.append((0.40, 0.39, 0.38))
    boxes.append(((3.4, 0.0, 2.0), (4.0, 3.2, 2.6)))
    albedos.append((0.42, 0.41, 0.40))
    boxes.append(((-2.4, 0.0, 3.2), (-1.4, 0.8, 4.2)))
    albedos.append((0.50, 0.45, 0.38))
    return Scene(boxes, albedos, ground_y=0.0, ground_albedo=(0.34, 0.33, 0.31))


def _slab(origin, dirs, lo, hi):
    """Entry distance into one box (inf where missed). `dirs` is (..., 3)."""
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    t1 = (lo - origin) * inv
    t2 = (hi - origin) * inv
    entry = np.max(np.minimum(t1, t2), axis=-1)
    exit_ = np.min(np.maximum(t1, t2), axis=-1)
    hit = (exit_ > np.maximum(entry, 0.0)) & (entry > SHADER_RAY_TMIN)
    return np.where(hit, entry, np.inf).astype(F32)


def trace_box_entry(scene, origin, dirs):
    """Nearest box entry distance per ray and the box index (-1 where no box is hit)."""
    best_t = np.full(dirs.shape[:-1], np.inf, dtype=F32)
    best_i = np.full(dirs.shape[:-1], -1, dtype=np.int32)
    for i in range(len(scene.lo)):
        t = _slab(origin, dirs, scene.lo[i], scene.hi[i])
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_i = np.where(closer, i, best_i)
    return best_t, best_i


def trace_ground(scene, origin, dirs):
    """Entry distance of the ground plane (inf when the ray never descends to it)."""
    dy = dirs[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (scene.ground_y - origin[1]) / dy
    return np.where((dy < -1e-9) & (t > SHADER_RAY_TMIN), t, np.inf).astype(F32)


def trace_scene(scene, origin, dirs):
    """Nearest hit: (t, material, normal, albedo). Material -1 means the sky."""
    t_box, i_box = trace_box_entry(scene, origin, dirs)
    t_ground = trace_ground(scene, origin, dirs)
    use_ground = t_ground < t_box
    t = np.where(use_ground, t_ground, t_box).astype(F32)
    mat = np.where(use_ground, -2, i_box)
    normal = np.zeros(dirs.shape, dtype=F32)
    normal[..., 1] = np.where(use_ground, 1.0, 0.0)
    albedo = np.broadcast_to(scene.ground_albedo, dirs.shape).astype(F32, copy=True)
    for i in range(len(scene.lo)):
        sel = mat == i
        if not np.any(sel):
            continue
        albedo[sel] = scene.albedo[i]
        p = origin[None, None, :] + dirs * np.where(sel, t, 0.0)[..., None]
        d_lo = np.abs(p - scene.lo[i])
        d_hi = np.abs(p - scene.hi[i])
        face_axis = np.argmin(np.minimum(d_lo, d_hi), axis=-1)
        n = np.where(d_lo < d_hi, -1.0, 1.0).astype(F32)
        n = n * (np.arange(3)[None, None, :] == face_axis[..., None])
        normal = np.where(sel[..., None], n, normal)
    return t, mat, normal, albedo


class ProbeLog:
    """What the integrators actually spend: one entry per sampler the shader calls.

    A shadow ray and a cloud-field evaluation are the two things a volumetric integrator pays for, and
    the point of the model is that both counts are fixed by the shader's constants rather than by the
    medium or the segment length.
    """

    def __init__(self):
        self.shadow_rays = 0
        self.cloud_field_evals = 0

    def ray(self, count):
        self.shadow_rays += int(count)

    def cloud(self, count):
        self.cloud_field_evals += int(count)


def sun_visibility(scene, points, sun_dir, log, clouds=False, count=None, force_visible=False,
                   frame_index=0):
    """One shadow ray per point: the shader's `visibility(p, sunDir, 10000.0)`.

    Returns grey visibility (this port returns 0/1; the shader may return a stained-glass tint), the
    toward-sun distance to the water surface (0 above it), and the cloud-shadow transmittance.
    `count` overrides the ray counter, for callers whose vectorized port computes pixels the shader
    would have skipped.
    """
    shape = points.shape[:-1]
    n = int(points[..., 0].size) if count is None else int(count)
    if force_visible:
        vis = np.ones(shape, dtype=F32)
    else:
        t_min = np.full(shape, np.inf, dtype=F32)
        rays = np.broadcast_to(sun_dir, points.shape)
        for i in range(len(scene.lo)):
            t_min = np.minimum(t_min, _slab(points, rays, scene.lo[i], scene.hi[i]))
        vis = np.where(t_min < SHADOW_RAY_LENGTH, 0.0, 1.0).astype(F32)
        log.ray(n)
    water_hit_t = np.zeros(shape, dtype=F32)
    if scene.water_y is not None and float(sun_dir[1]) > 1e-4:
        with np.errstate(divide="ignore", invalid="ignore"):
            water_hit_t = np.where(points[..., 1] < scene.water_y,
                                   (scene.water_y - points[..., 1]) / sun_dir[1], 0.0).astype(F32)
    if not clouds:
        return vis, water_hit_t, np.ones(shape, dtype=F32)
    log.cloud(n)
    return vis, water_hit_t, cloud_shadow(points, sun_dir, frame_index)


def cloud_shadow(points, sun_dir, frame_index=0):
    """Stand-in for cloudShadowTransmittance: a planar slab over a two-octave coverage field.

    The shader's coverage is a macro/micro FBM with a domain warp (dozens of hash taps per eval); the
    tap count is not what this preview measures — the number of evaluations is, and ProbeLog counts
    that. Shape and strength follow the same `exp(-cov * tau0 / sunDir.y)`.
    """
    if float(sun_dir[1]) < 0.05:
        return np.ones(points.shape[:-1], dtype=F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = (F32(180.0) - points[..., 1]) / sun_dir[1]
    xz = points[..., (0, 2)] + sun_dir[[0, 2]] * t_plane[..., None]
    cov = F32(0.55) * (0.5 + 0.5 * np.sin(xz[..., 0] * 0.021 + 1.7)
                       * np.cos(xz[..., 1] * 0.017 - 0.4 + frame_index * 0.01))
    cov = saturate(cov) ** 1.5
    # Thin-slab optical scale: the stand-in is meant to read as a light modulation (a few tens of
    # percent), not as an opaque deck, and only the evaluation count is being measured here.
    tau = cov * F32(SHADER_CLOUD_EXTINCTION) * F32(2.5) / max(float(sun_dir[1]), 0.05)
    return np.where(t_plane > 0.0, np.exp(-tau), 1.0).astype(F32)


# --- the height fog (analytic tier) -------------------------------------------------------------------
def fog_exp_stretch_depth(sigma0, height_h, above, dir_y, length):
    return (sigma0 * np.exp(-np.maximum(above, 0.0) / height_h) * (height_h / dir_y)
            * (1.0 - np.exp(-dir_y * length / height_h)))


def fog_optical_depth(sigma0, base_y, height_h, y0, dir_y, t):
    """volume.slang:60, vectorized. All arguments broadcast against each other."""
    sigma0, base_y, height_h, y0, dir_y, t = np.broadcast_arrays(
        *[np.asarray(v, dtype=F32) for v in (sigma0, base_y, height_h, y0, dir_y, t)])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        above = y0 - base_y
        # A level ray takes the flat column; the branch expressions divide by dir_y, so they are
        # evaluated against a non-degenerate slope and discarded by the final selection.
        slope = np.where(np.abs(dir_y) < 1.0e-6, F32(1.0), dir_y)
        flat = sigma0 * np.exp(-np.maximum(above, 0.0) / height_h) * t
        s_base_up = -above / slope
        value_up = np.where(
            above >= 0.0,
            fog_exp_stretch_depth(sigma0, height_h, above, slope, t),
            np.where(t <= s_base_up, sigma0 * t,
                     sigma0 * s_base_up
                     + fog_exp_stretch_depth(sigma0, height_h, 0.0, slope, t - s_base_up)))
        s_base_down = above / (-slope)
        value_down = np.where(
            above <= 0.0, sigma0 * t,
            np.where(t <= s_base_down,
                     fog_exp_stretch_depth(sigma0, height_h, above, slope, t),
                     fog_exp_stretch_depth(sigma0, height_h, above, slope, s_base_down)
                     + sigma0 * (t - s_base_down)))
        out = np.where(np.abs(dir_y) < 1.0e-6, flat, np.where(dir_y > 0.0, value_up, value_down))
        out = np.where(t <= 0.0, 0.0, out)
    return out.astype(F32)


def fog_transmittance(sigma0, base_y, height_h, y0, dir_y, t):
    return np.exp(-fog_optical_depth(sigma0, base_y, height_h, y0, dir_y, t))


def fog_depth_at_weight(sigma0, base_y, height_h, y0, dir_y, t, weight):
    """volume.slang:1097: 18-step bisection on the closed-form optical depth."""
    weight = np.asarray(weight, dtype=F32)
    tau_target = -np.log(np.maximum(1.0 - weight, 1.0e-4))
    full = fog_optical_depth(sigma0, base_y, height_h, y0, dir_y, t)
    lo = np.zeros_like(full, dtype=F32)
    hi = np.ones_like(full, dtype=F32) * np.asarray(t, dtype=F32)
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        reached = fog_optical_depth(sigma0, base_y, height_h, y0, dir_y, mid) < tau_target
        lo = np.where(reached, mid, lo)
        hi = np.where(reached, hi, mid)
    return np.where(weight <= 0.0, 0.0,
                    np.where(full <= tau_target, np.asarray(t, dtype=F32),
                             0.5 * (lo + hi))).astype(F32)


def interleave_gradient_noise(pix):
    """volume.slang:1089, the pixel-fixed jitter the stratum offsets ride."""
    return frac(F32(52.9829189) * frac(pix[..., 0] * F32(0.06711056) + pix[..., 1] * F32(0.00583715)))


def stratum_noise(pix, frame_index):
    return frac(interleave_gradient_noise(pix) + F32(frame_index) * F32(SHADER_JITTER_STEP))


class Medium:
    """The fog medium and the shaft medium as the shader pushes them (one object per arm)."""

    def __init__(self, fog_density=0.0, fog_height=DEFAULT_FOG_HEIGHT, fog_base_y=0.0,
                 fog_albedo=DEFAULT_FOG_ALBEDO, fog_g=DEFAULT_FOG_ANISOTROPY,
                 fog_ambient=DEFAULT_FOG_AMBIENT, fog_shadow=DEFAULT_FOG_SHADOW,
                 fog_shadow_floor=DEFAULT_FOG_SHADOW_FLOOR,
                 shaft_density=DEFAULT_SHAFT_DENSITY, shaft_g=DEFAULT_SHAFT_ANISOTROPY,
                 shaft_distance=DEFAULT_SHAFT_DISTANCE, shaft_intensity=DEFAULT_SHAFT_INTENSITY,
                 water_shaft_density=DEFAULT_WATER_SHAFT_DENSITY,
                 water_shaft_ceiling=DEFAULT_WATER_SHAFT_CEILING):
        self.fog_density = F32(fog_density)
        self.fog_height = F32(fog_height)
        self.fog_base_y = F32(fog_base_y)
        self.fog_albedo = F32(fog_albedo)
        self.fog_g = F32(fog_g)
        self.fog_ambient = F32(fog_ambient)
        self.fog_shadow = bool(fog_shadow)
        self.fog_shadow_floor = F32(fog_shadow_floor)
        self.shaft_density = F32(shaft_density)
        self.shaft_g = F32(shaft_g)
        self.shaft_distance = F32(shaft_distance)
        self.shaft_intensity = F32(shaft_intensity)
        self.water_shaft_density = F32(water_shaft_density)
        self.water_shaft_ceiling = F32(water_shaft_ceiling)


# --- the analytic fog segment (volume.slang:1127) -----------------------------------------------------
def fog_segment_inscatter(scene, ro, rd, t, pix, med, sun_dir, log, allow_shadow_cull=True,
                          count=None, frame_index=0):
    """Single-scatter over one segment: (radiance [H,W,3], fogT [H,W], detail)."""
    sigma0 = med.fog_density
    shape = t.shape
    fog_t = np.ones(shape, dtype=F32)
    albedo = np.full(3, med.fog_albedo, dtype=F32)
    detail = {"probes": 0}
    if float(sigma0) <= 0.0 or float(np.sum(albedo)) == 0.0:
        return np.zeros(shape + (3,), dtype=F32), fog_t, detail

    shadow_cull = bool(allow_shadow_cull and med.fog_shadow and float(sun_dir[1]) > 0.0)
    noise = stratum_noise(pix, frame_index)
    vis = np.ones((1,) + shape, dtype=F32)
    cloud_bands = np.ones((1,) + shape, dtype=F32)
    stratum_depth = np.zeros((1,) + shape, dtype=F32)
    if shadow_cull:
        scatter_full = 1.0 - fog_transmittance(sigma0, med.fog_base_y, med.fog_height,
                                               F32(ro[1]), rd[..., 1], t)
        depths = []
        for k in range(SHADER_FOG_SUN_BANDS):
            start = scatter_full * (F32(k) + noise) / F32(SHADER_FOG_SUN_BANDS)
            depths.append(fog_depth_at_weight(
                sigma0, med.fog_base_y, med.fog_height, F32(ro[1]), rd[..., 1], t,
                start + scatter_full * F32(0.5) / F32(SHADER_FOG_SUN_BANDS)))
        stratum_depth = np.stack(depths, axis=0)
        probe = ro[None, None, None, :] + rd[None] * stratum_depth[..., None]
        # All four strata ride one array here; in the shader each is its own shadow ray and its own
        # cloud-field evaluation, so bill four per call site pixel.
        vis, _, cloud_bands = sun_visibility(scene, probe, sun_dir, log, clouds=True,
                                             count=None if count is None
                                             else int(count) * SHADER_FOG_SUN_BANDS)
        detail["probes"] = SHADER_FOG_SUN_BANDS

    fog_t = fog_transmittance(sigma0, med.fog_base_y, med.fog_height, F32(ro[1]), rd[..., 1], t)
    scatter_weight = 1.0 - fog_t
    phase = hg((rd @ sun_dir).astype(F32), med.fog_g) * F32(SHADER_PHASE_ENERGY)
    ambient = (med.fog_ambient * sky_radiance(rd)).astype(F32)
    sun_part = np.zeros(shape + (3,), dtype=F32)
    if not shadow_cull:
        to_sun = fog_transmittance(sigma0, med.fog_base_y, med.fog_height, F32(ro[1]),
                                   F32(sun_dir[1]), F32(SHADOW_RAY_LENGTH))
        log.cloud(int(count) if count is not None else int(t.size))
        cloud0 = cloud_shadow(np.broadcast_to(ro, shape + (3,)), sun_dir)
        sun_part = (phase * scatter_weight * to_sun * cloud0)[..., None] * SUN_ILLUMINANCE
    else:
        stratum_weight = scatter_weight / F32(SHADER_FOG_SUN_BANDS)
        sun_vis = np.maximum(vis, med.fog_shadow_floor)
        leg = phase[None] * stratum_weight[None] * sun_vis
        for k in range(SHADER_FOG_SUN_BANDS):
            probe = ro[None, None, :] + rd * stratum_depth[k][..., None]
            to_sun = fog_transmittance(sigma0, med.fog_base_y, med.fog_height, probe[..., 1],
                                       F32(sun_dir[1]), F32(SHADOW_RAY_LENGTH))
            sun_part = sun_part + (leg[k] * to_sun * cloud_bands[k])[..., None] * SUN_ILLUMINANCE
    radiance = albedo * (sun_part + scatter_weight[..., None] * ambient)
    return radiance.astype(F32), fog_t, detail


# --- the shaft segment (volume.slang:1293) ------------------------------------------------------------
def shaft_depth_at_weight(sigma, reach, k):
    """volume.slang:1277. `sigma` and `reach` may be per-pixel."""
    sigma = np.asarray(sigma, dtype=F32)
    reach = np.asarray(reach, dtype=F32)
    if k <= 0:
        return np.zeros(np.broadcast(sigma, reach).shape, dtype=F32)
    total = 1.0 - np.exp(-sigma * reach)
    if k >= SHADER_SHAFT_STRATA:
        return np.ones(np.broadcast(sigma, reach).shape, dtype=F32) * reach
    weight = F32(k) * total / F32(SHADER_SHAFT_STRATA)
    return (-np.log(np.maximum(1.0 - weight, 1.0e-6)) / sigma).astype(F32)


def shaft_inscatter(scene, ro, rd, t, pix, med, sun_dir, log, underwater=False, frame_index=0,
                    control_visible=False):
    """Port of shaftInScatter: (radiance, share, detail).

    `control_visible` is the verification control: it replaces the shadow probes with V = 1, i.e. a
    uniformly lit medium. A volumetric effect would still light the path; this term must return exactly
    zero, because `structure = 1 - mean(V)` is then zero.
    """
    shape = t.shape
    zero = np.zeros(shape + (3,), dtype=F32)
    sigma = med.water_shaft_density if underwater else med.shaft_density
    strength = med.shaft_intensity
    empty = {"probes": 0, "structure": np.zeros(shape, dtype=F32), "lit": np.zeros(shape, dtype=F32),
             "phase": np.zeros(shape, dtype=F32)}
    if float(sigma) <= 0.0 or float(strength) <= 0.0 or float(np.max(t)) <= SHADER_RAY_TMIN:
        return zero, np.zeros(shape, dtype=F32), empty

    reach = np.minimum(t, F32(max(float(med.shaft_distance), 1.0)))
    if underwater:
        fog_t = np.ones(shape, dtype=F32)
    else:
        fog_t = fog_transmittance(med.fog_density, med.fog_base_y, med.fog_height, F32(ro[1]),
                                  rd[..., 1], reach)
    total = 1.0 - np.exp(-sigma * reach)
    shaft_weight = total * fog_t
    # volume.slang:1306 — a segment whose scatter weight is below the epsilon casts no probes at all.
    active = shaft_weight > F32(1.0e-4)
    if not np.any(active):
        return zero, np.zeros(shape, dtype=F32), empty
    phase = hg((rd @ sun_dir).astype(F32), med.shaft_g) * F32(SHADER_SHAFT_PHASE_ENERGY)
    if underwater:
        phase = np.minimum(phase, med.water_shaft_ceiling)

    noise = stratum_noise(pix, frame_index)
    weight = total / F32(SHAFT_STRATA)
    active_count = int(np.count_nonzero(active))
    sun = np.zeros(shape + (3,), dtype=F32)
    lit = np.zeros(shape, dtype=F32)
    for k in range(SHAFT_STRATA):
        a = shaft_depth_at_weight(sigma, reach, k)
        b = shaft_depth_at_weight(sigma, reach, k + 1)
        p = ro[None, None, :] + rd * (a + (b - a) * noise)[..., None]
        vis, water_hit_t, cloud = sun_visibility(scene, p, sun_dir, log, clouds=True,
                                                 count=active_count,
                                                 force_visible=control_visible)
        if underwater:
            # The probe crossed the water surface on its way to the sun: the shadow ray reports how
            # much water stands between the sample and the surface, and the caustic rides on it.
            vis = vis * np.exp(-sigma * water_hit_t)
        lit = lit + vis / F32(SHAFT_STRATA)
        sun = sun + (vis * (weight * cloud))[..., None] * SUN_ILLUMINANCE
    structure = saturate(1.0 - lit)
    if underwater:
        # Water keeps a floor on the gate: what reads underwater is the occluder's shadow in the
        # body, not a band a few pixels wide where the path straddles a shadow boundary.
        structure = lerp(F32(SHADER_WATER_SHAFT_STRUCTURE_FLOOR), F32(1.0), structure)
    share = shaft_weight * structure
    radiance = sun * (phase * (strength * fog_t * structure))[..., None]
    detail = {"probes": 0 if control_visible else SHAFT_STRATA,
              "structure": structure, "lit": lit, "phase": phase}
    return radiance.astype(F32), share.astype(F32), detail


# --- camera, shading, frame ---------------------------------------------------------------------------
def camera_rays(pos, target, width, height, fov_deg):
    pos = np.asarray(pos, dtype=F32)
    target = np.asarray(target, dtype=F32)
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=F32))
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    tan_half = math.tan(math.radians(fov_deg) * 0.5)
    aspect = width / height
    xs = ((np.arange(width, dtype=F32) + 0.5) / width * 2.0 - 1.0) * tan_half * aspect
    ys = (1.0 - (np.arange(height, dtype=F32) + 0.5) / height * 2.0) * tan_half
    dirs = (fwd[None, None, :] + right[None, None, :] * xs[None, :, None]
            + up[None, None, :] * ys[:, None, None])
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    return pos, dirs.astype(F32)


def surface_radiance(scene, pos, dirs, t, normal, albedo, sun_dir, hit, log):
    """Lambert plus sky ambient. The scene's own NEE, not part of any arm's budget."""
    points = pos[None, None, :] + dirs * np.where(hit, t, 0.0)[..., None]
    vis, _, cloud = sun_visibility(scene, points, sun_dir, log)
    vis = vis * hit
    cos_sun = np.maximum(0.0, np.sum(normal * sun_dir, axis=-1))
    sun = (cos_sun * vis * cloud)[..., None] * SUN_ILLUMINANCE / math.pi
    lit = albedo * (sun + SURFACE_AMBIENT * sky_radiance(normal))
    return np.where(hit[..., None], lit, sky_radiance(dirs) * F32(0.6)).astype(F32)


def render(scene, width, height, med, sun_dir, camera, fog_on, shafts_on, frames=1,
           control_visible=False):
    """One image: surface * fogT + fog in-scatter + shaft in-scatter.

    The shaft integrator is only called where the view ray meets geometry, which is the callers'
    doctrine (a sky miss already carries the atmosphere's own in-scatter in the sky LUT), so the
    vectorized port masks those pixels out of the probe count.
    """
    pos, dirs = camera_rays(camera[0], camera[1], width, height, camera[2])
    pix = np.stack(np.meshgrid(np.arange(width, dtype=F32), np.arange(height, dtype=F32)), axis=-1)
    t, _, normal, albedo = trace_scene(scene, pos, dirs)
    hit = np.isfinite(t)
    hit_count = int(np.count_nonzero(hit))

    surface_log = ProbeLog()
    base = surface_radiance(scene, pos, dirs, t, normal, albedo, sun_dir, hit, surface_log)

    fog_log = ProbeLog()
    shaft_log = ProbeLog()
    fog_color = np.zeros_like(base)
    fog_t = np.ones(t.shape, dtype=F32)
    if fog_on:
        # Geometry in front: 4 shadow probes. Sky-facing stretch: the closed form without probes. The
        # sky pixels take the second call, so the first one is handed a finite dummy segment length —
        # an infinite t would make the stratum bisection 0/0 and only the discarded pixels would see it.
        t_probe = np.where(hit, t, F32(1.0))
        c_hit, t_hit, _ = fog_segment_inscatter(scene, pos, dirs, t_probe, pix, med, sun_dir, fog_log,
                                                allow_shadow_cull=True, count=hit_count)
        c_sky, t_sky, _ = fog_segment_inscatter(scene, pos, dirs, np.full(t.shape, 256.0, dtype=F32),
                                                pix, med, sun_dir, fog_log,
                                                allow_shadow_cull=False,
                                                count=width * height - hit_count)
        fog_color = np.where(hit[..., None], c_hit, c_sky)
        fog_t = np.where(hit, t_hit, t_sky)

    shaft_color = np.zeros_like(base)
    share = np.zeros(t.shape, dtype=F32)
    structure = np.zeros(t.shape, dtype=F32)
    if shafts_on and hit_count:
        t_segment = np.where(hit, t, F32(0.0))
        accum = np.zeros_like(base)
        share_acc = np.zeros_like(share)
        structure = np.zeros_like(share)
        for f in range(max(1, frames)):
            colour, share_f, detail = shaft_inscatter(
                scene, pos, dirs, t_segment, pix, med, sun_dir, shaft_log,
                underwater=scene.water_y is not None, frame_index=f,
                control_visible=control_visible)
            accum = accum + colour
            share_acc = share_acc + share_f
            structure = detail["structure"]
        shaft_color = accum / F32(max(1, frames))
        share = share_acc / F32(max(1, frames))
        shaft_color = np.where(hit[..., None], shaft_color, 0.0)
        share = np.where(hit, share, 0.0)
        structure = np.where(hit, structure, 0.0)

    color = base * (fog_t if fog_on else F32(1.0))[..., None] + fog_color + shaft_color
    return {"color": color.astype(F32), "base": base, "fog": fog_color, "shaft": shaft_color,
            "fog_t": fog_t, "structure": structure, "share": share, "hit": hit,
            "logs": {"surface": surface_log, "fog": fog_log, "shaft": shaft_log}}


# --- display ------------------------------------------------------------------------------------------
def tonemap(rgb):
    x = np.maximum(rgb, 0.0)
    return (x / (1.0 + x)) ** (1.0 / 2.2)


def gray_image(values, ceiling=None):
    v = np.asarray(values, dtype=F32)
    hi = float(ceiling if ceiling is not None else max(float(np.max(v)), 1e-6))
    return np.repeat((np.clip(v / hi, 0.0, 1.0) ** (1.0 / 2.2))[..., None], 3, axis=-1)


def to_uint8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png(path, image):
    Image.fromarray(to_uint8(image)).save(path)


def gain_for(image, target=0.30, percentile=99.0):
    """Exposure that puts the bright end of `image` at `target`: the shaft term is faint next to a
    fully lit wall, so the panels that show it alone are auto-exposed and say so in their label."""
    peak = float(np.percentile(np.asarray(image, dtype=F32).max(axis=-1), percentile))
    return F32(target / max(peak, 1e-6))


def contact_sheet(panels, path, columns=3, label_height=18):
    h, w = panels[0][1].shape[:2]
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (w * columns, (h + label_height) * rows), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(panels):
        col, row = i % columns, i // columns
        sheet.paste(Image.fromarray(to_uint8(image)),
                    (col * w, row * (h + label_height) + label_height))
        draw.text((col * w + 6, row * (h + label_height) + 4), label, fill=(220, 220, 220))
    sheet.save(path)


# --- cost model ---------------------------------------------------------------------------------------
def probe_budget():
    """What one call spends, per pixel, straight from the shader constants."""
    return [
        ("analytic fog", SHADER_FOG_SUN_BANDS, SHADER_FOG_SUN_BANDS),
        ("sun shafts", SHADER_SHAFT_STRATA, SHADER_SHAFT_STRATA),
        ("marched fog", min(SHADER_FOG_MARCH_PROBE_STRATA, DEFAULT_FOG_MARCH_STEPS),
         min(SHADER_FOG_MARCH_PROBE_STRATA, DEFAULT_FOG_MARCH_STEPS)),
    ]


# --- source audit -------------------------------------------------------------------------------------
def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


CALL_PATTERN = re.compile("|".join(name + "[ 	]*[(]" for name in
                                     ("shaftInScatter", "fogSegmentInScatter")))


def _statement_head(lines, index):
    """Text of the gate that owns the call on `lines[index]`, from the block head to the call.

    Call statements wrap across lines and may open with a local declaration, so the gate cannot be
    read off the call line alone: walk up to the `if (...) {` that opens the enclosing block and stop
    when the indentation leaves it.
    """
    depth = len(lines[index]) - len(lines[index].lstrip())
    first = index
    while first > 0:
        prev = lines[first - 1]
        stripped = prev.strip()
        if stripped == "" or stripped.startswith("//"):
            first -= 1
            continue
        indent = len(prev) - len(prev.lstrip())
        if indent < depth and stripped.startswith("}"):
            break
        first -= 1
        if indent < depth and stripped.endswith("{"):
            break
    return " ".join(line.strip() for line in lines[first:index + 1]
                    if not line.strip().startswith("//"))



def gating_audit():
    """Source checks behind the cost claim: list of (check, ok, evidence).

    A missing source file is a skipped check (`ok is None`) rather than a failure.
    """
    checks = []
    for path, name in ((PRIMARY_SLANG, "primary.rgen.slang"), (INDIRECT_SLANG, "indirect.rgen.slang")):
        text = _read(path)
        if text is None:
            checks.append((f"{name}: readable", None, "file not found"))
            continue
        lines = text.splitlines()
        calls = 0
        water = 0
        for index, line in enumerate(lines):
            match = CALL_PATTERN.search(line)
            if match is None:
                continue
            calls += 1
            gate = _statement_head(lines, index)
            behind_volume_path = ("CAUSTICA_VOLUME_PATH" in gate
                                  or _guarded_by_volume_path(lines, index))
            checks.append((f"{name}:{index + 1} {match.group(0)} stays behind CAUSTICA_VOLUME_PATH",
                           behind_volume_path, gate[-90:]))
            if not match.group(0).startswith("shaftInScatter"):
                continue
            if "waterShaft" in gate or "medium.current.water" in gate:
                water += 1
                checks.append((f"{name}:{index + 1} water shaft is gated by the water shaft switch",
                               "shaftOn" in gate, gate[-90:]))
            else:
                checks.append((f"{name}:{index + 1} air shaft is gated by the shaft switch only",
                               "fogOn" not in gate, gate[-90:]))

        checks.append((f"{name}: shafts and fog share the same call sites", calls >= 2,
                       f"{calls} calls, {water} of them in water"))
    volume = _read(VOLUME_SLANG)
    checks.append(("volume.slang: defines the integrator", volume is not None
                   and "float3 shaftInScatter(" in volume
                   and "fogSegmentInScatter(" in volume, "shaftInScatter + fogSegmentInScatter"))
    java = _read(COMPOSITE_JAVA)
    if java is None:
        checks.append(("RtComposite.java: readable", None, "file not found"))
    else:
        match = re.search(r"boolean leanVolumePath\s*=(.*?);", java, re.S)
        lean = " ".join(match.group(1).split()) if match else "leanVolumePath not found"
        checks.append(("RtComposite.java: the volume-free raygen needs the shafts off",
                       bool(match) and "!shaftOn" in match.group(1), lean))
    gradle = _read(BUILD_GRADLE)
    if gradle is None:
        checks.append(("build.gradle: readable", None, "file not found"))
    else:
        defines = gradle.count("-DCAUSTICA_NO_VOLUME_PATH")
        twins = re.findall(r"world/(?:primary|indirect)_lean(?:_ser)?\.rgen\.spv", gradle)
        checks.append(("build.gradle: three volume-free raygen twins",
                       defines == 3 and len(twins) == 3,
                       f"defines={defines} twins={len(twins)}"))
    return checks


def _guarded_by_volume_path(lines, index):
    """Whether the function containing `lines[index]` opens with a volume-path guard clause."""
    depth = len(lines[index]) - len(lines[index].lstrip())
    first = index
    while first > 0 and (len(lines[first]) - len(lines[first].lstrip()) >= depth
                         or not lines[first].strip()):
        first -= 1
    for line in lines[first:index]:
        if "!CAUSTICA_VOLUME_PATH" in line:
            return True
    return False


# --- report -------------------------------------------------------------------------------------------
def report_budget(width, height):
    pixels = width * height
    rows = probe_budget()
    print()
    print("=== probe budget (per call, from the shader constants) ===")
    print(f"{'term':16s}{'sun rays':>10s}{'cloud evals':>13s}{'rays/frame':>14s}")
    for name, rays, clouds in rows:
        print(f"{name:16s}{rays:>10d}{clouds:>13d}{rays * pixels / 1e6:>11.2f} M")
    print(f"  (frame = {width}x{height} = {pixels / 1e6:.2f} M pixels, one call site per pixel)")
    fog_rays = rows[0][1]
    shaft_rays = rows[1][1]
    print()
    print(f"  sun shafts spend {shaft_rays} probes per pixel against the analytic fog's {fog_rays}:"
          f" {shaft_rays / fog_rays:.2f}x. The count is fixed by the shader's constants — reach and"
          f" density only decide whether the epsilon early-out skips them")
    print(f"  at 1920x1080 the shaft arm alone is {shaft_rays * 1920 * 1080 / 1e6:.2f} M shadow rays per"
          f" frame, i.e. {shaft_rays * 1920 * 1080 / 1e6:.1f} M more than the fog tier spends")


def report_invariance(scene, sun_dir, camera, width=64, height=36):
    """Re-count the spend at the reach and density extremes: the count must not move."""
    print()
    print("=== per-pixel constancy (measured, 64x36 probe frame) ===")
    counts = []
    for reach, density in ((16.0, 0.002), (512.0, 0.05), (DEFAULT_SHAFT_DISTANCE,
                                                          DEFAULT_SHAFT_DENSITY)):
        med = Medium(fog_density=0.0, shaft_distance=reach, shaft_density=density)
        image = render(scene, width, height, med, sun_dir, camera, fog_on=False, shafts_on=True)
        hit = image["hit"]
        log = image["logs"]["shaft"]
        counts.append((reach, density, log.shadow_rays, log.cloud_field_evals,
                       int(np.count_nonzero(hit)),
                       float(np.mean(image["shaft"][hit])) if np.any(hit) else 0.0))
    for reach, density, rays, clouds, hits, level in counts:
        print(f"  reach {reach:5.0f}  density {density:.4f}  ->  {rays / max(hits, 1):5.2f}"
              f" rays per hit pixel, cloud evals {clouds}, mean in-scatter {level:.4f}")
    same = len({round(rays / max(hits, 1), 6) for _, _, rays, _, hits, _ in counts}) == 1
    print(f"  PASS count independent of reach and density: {same}")
    levels = [level for *_, level in counts]
    print(f"  the effect itself moves {max(levels) / max(min(levels), 1e-9):.0f}x across those settings")
    return same


def report_effect(images, med):
    print("\n=== effect: shadow structure, not air ===")
    base = images["off"]["color"]
    delta = images["shafts"]["color"] - base
    control = images["shafts_unshadowed"]["color"] - base
    fog_delta = images["fog"]["color"] - base
    print(f"fog density in the shaft arm : {float(med.fog_density):.4f} / block"
          f"  (fog disabled: the medium is zero)")
    print(f"shaft in-scatter            : mean {float(np.mean(delta)):.4f}"
          f"  max {float(np.max(delta)):.4f}  min {float(np.min(delta)):.6f}")
    print(f"control (V forced to 1)     : max {float(np.max(control)):.6f}  (must be exactly 0)")
    print(f"fog in-scatter              : mean {float(np.mean(fog_delta)):.4f}")
    structure = images["shafts"]["structure"]
    mean_structure = float(np.mean(structure[images["shafts"]["hit"]])) if np.any(images["shafts"]["hit"]) else 0.0
    cv = float(np.std(structure[images["shafts"]["hit"]]) / max(mean_structure, 1e-6)) \
        if np.any(images["shafts"]["hit"]) else 0.0
    print(f"structure gate (hit pixels) : mean {mean_structure:.3f}  coefficient of variation {cv:.2f}")
    ok_control = float(np.max(np.abs(control))) == 0.0
    ok_positive = float(np.min(delta)) >= 0.0
    print(f"\nPASS control is identically zero : {ok_control}")
    print(f"PASS shafts only add light       : {ok_positive}")
    print("verdict: with the fog at zero the segment still carries beams, and silencing the sun probes")
    print("         removes them exactly. The light is the shadow structure of the path, not an integral")
    print("         of a scattering medium; the medium only supplies the weight.")
    return ok_control and ok_positive


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline preview and cost model of the sun-shaft term.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "shaft-preview-out"))
    parser.add_argument("--frames", type=int, default=1, help="temporal accumulations of the jitter")
    parser.add_argument("--reach", type=float, default=DEFAULT_SHAFT_DISTANCE)
    parser.add_argument("--shaft-density", type=float, default=DEFAULT_SHAFT_DENSITY)
    parser.add_argument("--fog-density", type=float, default=DEFAULT_FOG_DENSITY)
    parser.add_argument("--budget-only", action="store_true")
    args = parser.parse_args(argv)

    audit = gating_audit()
    print("=== compile-time gating audit ===")
    for check, ok, evidence in audit:
        mark = "skip" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {check}")
        if ok is False:
            print(f"         {evidence}")
    report_budget(args.width, args.height)
    if args.budget_only:
        return 0 if all(ok is not False for _, ok, _ in audit) else 1

    os.makedirs(args.out, exist_ok=True)
    scene = build_air_scene()
    sun_dir = np.array([0.25, 0.55, 0.80], dtype=F32)
    sun_dir = (sun_dir / np.linalg.norm(sun_dir)).astype(F32)
    camera = ((0.0, 1.7, -10.0), (0.0, 1.5, 6.0), 55.0)

    quiet = Medium(fog_density=0.0, shaft_distance=args.reach, shaft_density=args.shaft_density)
    hazy = Medium(fog_density=args.fog_density, shaft_distance=args.reach,
                  shaft_density=args.shaft_density)
    common = dict(width=args.width, height=args.height, sun_dir=sun_dir, camera=camera,
                  frames=args.frames)
    t0 = time.perf_counter()
    images = {
        "off": render(scene, med=quiet, fog_on=False, shafts_on=False, **common),
        "shafts": render(scene, med=quiet, fog_on=False, shafts_on=True, **common),
        "shafts_unshadowed": render(scene, med=quiet, fog_on=False, shafts_on=True,
                                    control_visible=True, **common),
        "fog": render(scene, med=hazy, fog_on=True, shafts_on=False, **common),
        "both": render(scene, med=hazy, fog_on=True, shafts_on=True, **common),
    }
    elapsed = time.perf_counter() - t0

    labels = ["off (no fog, no shafts)", "shafts, fog density 0 (auto-exposed)",
              "control: V forced to 1 (must be 0)", "fog only", "both",
              "shaft term alone (auto-exposed)", "structure gate (1 - mean V)",
              "shaft share (RR albedo weight)"]
    keys = ["off", "shafts", "shafts_unshadowed", "fog", "both"]

    def auto(image):
        return tonemap(image * gain_for(image))

    panels = [(labels[0], tonemap(images["off"]["color"])),
              (labels[1], auto(images["shafts"]["color"])),
              (labels[2], auto(images["shafts_unshadowed"]["color"])),
              (labels[3], tonemap(images["fog"]["color"])),
              (labels[4], tonemap(images["both"]["color"])),
              (labels[5], auto(images["shafts"]["shaft"])),
              (labels[6], gray_image(images["shafts"]["structure"])),
              (labels[7], gray_image(images["shafts"]["share"]))]
    for key in keys:
        save_png(os.path.join(args.out, f"{key}.png"), tonemap(images[key]["color"]))
    save_png(os.path.join(args.out, "shaft_only.png"), auto(images["shafts"]["shaft"]))
    save_png(os.path.join(args.out, "structure.png"), gray_image(images["shafts"]["structure"]))
    save_png(os.path.join(args.out, "share.png"), gray_image(images["shafts"]["share"]))
    sheet = os.path.join(args.out, "sheet.png")
    contact_sheet(panels, sheet, columns=4)

    print(f"\n=== frame: {args.width}x{args.height}, {elapsed:.2f}s for {len(images)} arms ===")
    ok_effect = report_effect(images, quiet)
    ok_inv = report_invariance(scene, sun_dir, camera)

    print("\n=== measured spend on this frame ===")
    print(f"{'arm':18s}{'hit px':>10s}{'shaft rays':>12s}{'rays/hit':>10s}"
          f"{'cloud evals':>13s}{'fog rays':>10s}{'fog/hit':>9s}")
    for key in keys:
        logs = images[key]["logs"]
        hits = int(np.count_nonzero(images[key]["hit"]))
        shaft_rays = logs["shaft"].shadow_rays
        clouds = logs["shaft"].cloud_field_evals
        fog_rays = logs["fog"].shadow_rays
        print(f"{key:18s}{hits:>10d}{shaft_rays:>12d}{shaft_rays / max(hits, 1):>10.2f}"
              f"{clouds:>13d}{fog_rays:>10d}{fog_rays / max(hits, 1):>9.1f}")

    print(f"\nwrote {sheet}")
    ok = all(o is not False for _, o, _ in audit) and ok_effect and ok_inv
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
