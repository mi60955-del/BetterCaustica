// Volumetric media that are NOT MediumStack members: the air around the terrain. The stack (medium.slang)
// tracks what an interface crossing put the path inside (water, glass); fog and clouds are position-dependent
// global fields the path integrator evaluates per segment, so they must not consume stack depth. Air layers
// add on top of whatever the stack says; the water layer itself is the stack's business (its scattering
// arrives with Medium.scattering, packed without widening the path record).
//
// Everything here reads the WorldPush volume block (fog0/fog1/fogColor/cloud0/cloud1/cloudColor/weather/
// volumeParams) and is gated by worldPush.flags volume bits, all of which are clear by default: a frame with
// every volume switch off evaluates none of it, so the path integral is bit-identical to the pre-volume
// renderer. That identity is the regression baseline the whole module is held to.
import world_common;
import world_core;
import medium;
import math;
import trace;
import sky;
import water;
import bindings;

// One sampled point of the air volume. sigmaS/sigmaA are 1/block, the same unit as medium.slang's
// extinction. `g` is the Henyey–Greenstein anisotropy shared by both coefficients' phase.
public struct VolumeSample {
    public float3 sigmaS;
    public float3 sigmaA;
    public float  g;
    public float3 emission;
};

public VolumeSample emptyVolumeSample() {
    VolumeSample s;
    s.sigmaS = float3(0.0, 0.0, 0.0);
    s.sigmaA = float3(0.0, 0.0, 0.0);
    s.g = 0.0;
    s.emission = float3(0.0, 0.0, 0.0);
    return s;
}

// ---- Analytic height fog ---------------------------------------------------------------------------
//
// Density falls exponentially above the fog base and is constant below it:
//     sigma_t(y) = sigma0 * exp(-max(0, y - baseY) / H)
// baseY arrives already rebased into terrain coordinates (Java subtracts the terrain origin, the same
// anchoring waterAnchor uses), so the profile stays pinned to the world as the terrain rebase shifts.
// The fog's scattering fraction lives Java-side (volume.fog-albedo, default 0.85) and arrives inside
// fogColor.rgb — the shader must not re-scale it.

// Closed-form depth of the pure-exponential stretch: starts `above` blocks above the base and travels
// `len` along the vertical component dirY without crossing the base. Positive for either dirY sign.
// For dirY < 0 the two signed factors cancel, so the large intermediate exp(above/H) only ever loses
// precision, never overflows.
public float fogExpStretchDepth(float sigma0, float heightH, float above, float dirY, float len) {
    return sigma0 * exp(-max(above, 0.0) / heightH) * (heightH / dirY)
            * (1.0 - exp(-dirY * len / heightH));
}

// Optical depth of the profile over [0, t] along a direction with vertical component dirY. Closed form,
// so the analytic fog tier integrates with zero march samples and zero Monte-Carlo noise — the property
// the RR pipeline and the low-end tier both lean on. The clamp at the base makes this piecewise: below
// baseY the density is flat sigma0, above it the exponential integrates in closed form.
public float fogOpticalDepth(float sigma0, float baseY, float heightH, float y0, float dirY, float t) {
    if (t <= 0.0) {
        return 0.0;
    }
    float above = y0 - baseY;
    if (abs(dirY) < 1.0e-6) {
        return sigma0 * exp(-max(above, 0.0) / heightH) * t;
    }
    if (dirY > 0.0) {
        if (above >= 0.0) {
            return fogExpStretchDepth(sigma0, heightH, above, dirY, t);
        }
        float sBase = -above / dirY; // flat-density stretch from below the base up to baseY
        if (t <= sBase) {
            return sigma0 * t;
        }
        return sigma0 * sBase + fogExpStretchDepth(sigma0, heightH, 0.0, dirY, t - sBase);
    }
    // dirY < 0
    if (above <= 0.0) {
        return sigma0 * t;
    }
    float sBase = above / (-dirY); // exponential stretch from above the base down to baseY
    if (t <= sBase) {
        return fogExpStretchDepth(sigma0, heightH, above, dirY, t);
    }
    return fogExpStretchDepth(sigma0, heightH, above, dirY, sBase) + sigma0 * (t - sBase);
}

public float fogTransmittance(float sigma0, float baseY, float heightH, float y0, float dirY, float t) {
    return exp(-fogOpticalDepth(sigma0, baseY, heightH, y0, dirY, t));
}

/**
 * Closed-form single-scatter in-scatter over [0, t], the analytic fog tier's whole lighting term:
 *
 *     L = albedo * (1 - T(t)) * (sunTerm + ambient)
 *
 * The identity integral[T(s)*sigma_t(s) ds] = 1 - T(t) (transmittance's own derivative) makes the marched
 * integral exact for ANY density profile as long as sigma_s/sigma_t stays at `albedo`, which the constant
 * fog albedo gives. `sunTerm` and `ambient` are radiance-shaped terms the caller assembles — the sun's
 * phase-weighted illuminance times its geometric visibility times the analytic transmittance toward it,
 * and the sky-irradiance ambient — so the shadow-ray count stays at one per pixel and no LUT access
 * happens inside this module.
 */
public float3 fogInScatter(float3 albedo, float sigma0, float baseY, float heightH,
                           float y0, float dirY, float t, float3 sunTerm, float3 ambient) {
    float transmittance = fogTransmittance(sigma0, baseY, heightH, y0, dirY, t);
    return albedo * ((1.0 - transmittance) * (sunTerm + ambient));
}

// ---- Heterogeneous density (the march tier's noise field) -------------------------------------------
//
// The analytic profile above is the fog's mean density; the march tier modulates it with a value-noise
// FBM so fog gathers into billows and lets light shafts read as shaped volumes instead of extruded
// cones:
//
//     sigma(p) = sigmaProfile(p.y) * max(0, 1 + amp * (fbm(p) * 2 - 1))
//
// so the multiplier stays in [1-amp, 1+amp] and amp = 0 reproduces the analytic field exactly — the
// property the march-to-analytic convergence test leans on. The field is world-anchored through the
// same trick the wave domain uses (rebased position plus waterAnchor recovers world coordinates), and
// it drifts with the wind at fog1.y blocks per second along fog1.x's azimuth, reading the wave clock
// (waterParams.w, a wall-clock seconds accumulator independent of the waves toggle).

public static const float FOG_NOISE_CELL = 16.0;  // blocks per base FBM cell

// Dave Hoskins' hash13 (https://www.shadertoy.com/view/4djSRW) — same choice sky.rmiss makes for the
// star lattice. Renamed so the two file-local copies can never collide through an import.
float fogHash13(float3 p3) {
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

// Lattice hash wrapped to a 256-cell period per axis. The wrap makes the noise field exactly periodic
// in 256 * FOG_NOISE_CELL = 4096 blocks — the same modulus the waterAnchor anchoring works in — so the
// wind offset can wrap into [0, 4096) and the drift never snaps when the wall clock wraps.
float fogCellHash(float3 c) {
    return fogHash13(c - 256.0 * floor(c / 256.0));
}

float fogValueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float n = lerp(
        lerp(lerp(fogCellHash(i + float3(0.0, 0.0, 0.0)), fogCellHash(i + float3(1.0, 0.0, 0.0)), f.x),
            lerp(fogCellHash(i + float3(0.0, 1.0, 0.0)), fogCellHash(i + float3(1.0, 1.0, 0.0)), f.x),
            f.y),
        lerp(lerp(fogCellHash(i + float3(0.0, 0.0, 1.0)), fogCellHash(i + float3(1.0, 0.0, 1.0)), f.x),
            lerp(fogCellHash(i + float3(0.0, 1.0, 1.0)), fogCellHash(i + float3(1.0, 1.0, 1.0)), f.x),
            f.y),
        f.z);
    return n;
}

// Three octaves, unit gain — octave transforms are integer scalings plus a constant shift, which the
// 256-cell lattice wrap survives exactly, so the whole stack stays 4096-block periodic.
float fogFbm(float3 p) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int o = 0; o < 3; o++) {
        sum += amp * fogValueNoise(p);
        norm += amp;
        p = p * 2.0 + float3(17.0, 9.0, 23.0);
        amp *= 0.5;
    }
    return sum / norm;
}

// ---- Spray band (the water surface's own mist layer) ------------------------------------------------
//
// A thin high-density band living just above the water surface, its density driven by the wave field
// itself: crest steepness (the wave tilt's magnitude) is the density modulation, so spray gathers
// where the surface is actually agitated. Rides the marched tier only — a wavefield-dependent density
// has no closed form for the analytic tier to consume — and is gated by WORLD_FLAG_SPRAY, which Java
// only sets when the marched fog tier is running.
//
// v1 geometry: the band hangs above ONE pushed sea-level plane (water1.w, rebased) rather than the
// true per-position heightfield, because the air integrator has no water-presence knowledge per
// sample. Consequences, both accepted for this tier: water surfaces well above sea level (mountain
// lakes) get no spray, and low-lying terrain near sea level gets a whisper of the band where the
// wave field is agitated. The band is 1.5 blocks and the modulation keeps most of it hugging zero,
// so the approximation reads as coastal ground mist rather than a artefact. Default strength is 0 —
// the whole term is opt-in.
public static const float SPRAY_SIGMA = 0.35; // extinction at full strength and full crest (1/block)

float sprayDensity(WorldPush push, float3 pos) {
    if ((push.flags & WORLD_FLAG_SPRAY) == 0u) {
        return 0.0;
    }
    float above = pos.y - push.water1.w;
    float bandH = max(push.water1.z, 0.05);
    if (above < 0.0 || above >= bandH) {
        return 0.0;
    }
    float band = smoothstep(0.0, 0.35 * bandH, above) * (1.0 - smoothstep(0.6 * bandH, bandH, above));
    // Crest steepness 0..1: the wave tilt (the same gradient the visible ripples use, scaled by the
    // wave-height control like every other consumer) saturates the spray around ~0.12 slope.
    float2 tilt = waterWaveTilt(pos.xz + push.waterAnchor.xy, push.water1.w,
            push.waterParams.w, 0.0) * waterWaveControls().x;
    float crest = clamp(length(tilt) * 8.0, 0.0, 1.0);
    return push.weather.z * SPRAY_SIGMA * band * crest;
}

// The march tier's density at one point. Returns the plain analytic profile while the noise amplitude
// is zero, so the marched integrator degenerates to integrating the analytic field.
// The march tier's density at one point. Returns the plain analytic profile while the noise amplitude
// is zero, so the marched integrator degenerates to integrating the analytic field. It also returns the
// plain profile wherever the profile times the noise's largest multiplier sits below
// FOG_NOISE_MIN_DENSITY: there the 24-lattice-hash FBM can move the sample by less than the density is
// worth, so the modulation is skipped and the segment's optical depth error stays bounded by that
// constant per block of skipped stretch — invisible next to the noise floor the estimator already runs
// at, and the reason the steps above the fog's significant band stop paying for noise.
public static const float FOG_NOISE_MIN_DENSITY = 1.0e-6;

public float fogMarchDensity(WorldPush push, float3 pos) {
    float base = fogDensityAt(push, pos.y);
    float amp = push.fog1.z;
    if (base <= 0.0 || amp <= 0.0 || base * (1.0 + abs(amp)) < FOG_NOISE_MIN_DENSITY) {
        return base + sprayDensity(push, pos);
    }
    // The wind drift arrives pre-accumulated per axis (Java integrates speed x direction over the
    // frame step and wraps each axis in whole 4096-block periods — this lattice's exact period —
    // so neither the hourly clock wrap nor a rebase can snap the field; the old shader-side
    // speed x clock product did exactly that once an hour).
    float2 wind = push.fog1.xy;
    float3 q = (float3(pos.xz + push.waterAnchor.xy, pos.y)
            + float3(wind.x, 0.0, wind.y)) / FOG_NOISE_CELL;
    return base * max(0.0, 1.0 + amp * (fogFbm(q) * 2.0 - 1.0)) + sprayDensity(push, pos);
}

// ---- Cloud layer (baked volumetric clouds, the clouds_unreal scheme) --------------------------------
//
// The layer's datum is cloud0.y, its base, and cloud0.z, how deep the deck stands above it. Instances
// stand on that base as an infinite wind-drifted grid, each rendering one baked 3D density volume
// (the textures/cloud/*.vol assets, uploaded as one 2D-array atlas by RtCloudVolumes) in a box as
// wide as Cloud Size and as deep as Cloud Thickness. The march and the
// lighting are the UE "volumetric raymarcher" scheme the clouds_unreal project ships (Ryan Brucks'
// shaderbits raymarcher): Beer-Lambert extinction per step, a short secondary march toward the
// dominant body for self-shadow, the multi-scatter octave stack on the light leg, sky-irradiance
// ambient.
//
// Instance placement is a deterministic hash grid, so every consumer — the sky miss shader's walk,
// the cheap single-crossing tiers, the NEE shadow probes, the continuation-ray attenuation — derives
// the SAME cloud at the SAME place from the push block alone. The grid reads the EXACT cloud anchor
// (cloud2.xy) and the Java-accumulated drift (cloud2.zw), so the wind slides the instances and no
// terrain rebase can snap them. cloud1.x selects which baked volume every instance renders.
//
// Per-instance variety comes from the same hash: each instance rotates its volume by a quarter-turn
// with an optional mirror and scales it inside a patchiness-driven band, so one texture dresses the
// whole sky without visible repetition. Near full coverage the grid saturates into a solid overcast
// of touching decks; below that, a coarse hash of the parent cell gathers instances into weather
// systems with clear sky between them — the patchiness setting drives how hard.

public static const float CLOUD_GRID_CELL = 176.0;  // blocks per instance-grid cell at the base span
public static const float CLOUD_BOX_SPAN = 0.86;    // instance footprint as a fraction of its cell
// Column optical depth, in density-blocks, every tier prices the deck at: the one number that sets
// how opaque a cloud is. The march converts it into an extinction per block by dividing by the box
// height, and the planar tier and the shadow probes use it directly. Derived from the nine shipped
// volumes: their mean density times this is the column the old slab calibration delivered, measured
// over the set.
public static const float CLOUD_COLUMN_DEPTH = 34.8;
// Height inside the box, as a fraction, where the single-sample consumers read the field: the
// mass-weighted centre of the authored profiles. The planar sheet's crossing, the shadow probes and
// the continuation-ray transmittance all use it, so neither switching tiers nor flying in and out of
// the walk moves the cloud.
public static const float CLOUD_SHEET_HEIGHT = 0.35;
// Vertical thickness of the two taps the planar tier averages, as a fraction of the box height. Wide
// enough to average out one slice's speckle, narrow enough that the two taps describe one height.
public static const float CLOUD_SHEET_BAND = 0.08;
// Vertical resolution of the baked volumes; MUST match RtCloudVolumes.RES, which validates the
// resources against it at load.
public static const float CLOUD_VOLUME_RES = 128.0;
public static const float CLOUD_EXTINCTION = 0.22;      // 1/block at unit density (water-cloud order)
public static const float CLOUD_G = 0.4;                // phase anisotropy, the fog tier's forward doctrine
public static const float3 CLOUD_ALBEDO = float3(0.86, 0.87, 0.90); // BT-709-linear, sky.rmiss space
// The phase a sample is lit by: an isotropic share plus the single-scatter lobe, mostly intact. The lobe
// alone spans nearly two orders of magnitude between looking into the body and looking away from it, which
// drew one and the same deck blown white on the body's side of the sky and black on the other. Taking that
// range out entirely is worse: a deck with no directional falloff reads as one flat bright sheet, and the
// falloff IS what makes the cloud field read as shape. The share below only lifts the extreme - the
// directions more than ~75 degrees off the body, where the lobe has nothing left - and the lobe keeps the
// rest of its range.
public static const float CLOUD_SCATTER_FLOOR = 0.15;   // the isotropic share every sample carries
public static const float CLOUD_SCATTER_LOBE = 0.85;    // how much of the single-scatter lobe survives
// How much of the column above a sample occludes the SKY that lights it. Light reaches a deck's base from
// the whole hemisphere - the sides and the gaps included - so this stays well below the sun leg's own
// occlusion. It is also what makes the base darker than the crown, so it cannot be driven to zero either:
// the deck's own vertical gradient is this weight's whole job.
public static const float CLOUD_AMBIENT_OCCLUSION = 0.20;
// The sky term is the only light a thick base receives: its sun leg saturates the multi-scatter stack, and
// what reaches a viewer under the deck is light that entered through the sides and the gaps. Two constants
// keep that floor from collapsing. The FLOOR is the share of the unoccluded sky a sample keeps at any depth
// (the hemisphere around it is never fully cloud), and the GAIN undoes the irradiance-to-radiance divide
// for a scattering medium, whose in-scatter is not a Lambertian surface's.
public static const float CLOUD_AMBIENT_FLOOR = 0.35;
public static const float CLOUD_AMBIENT_GAIN = 5.0;

// Self-shadow of a sample with the multi-scatter octave stack. The weights (1, 1/2, 1/4) over
// extinctions (tau, tau/2, tau/4) are normalised by their sum, so the result is 1 at tau = 0 and falls
// off far more slowly than exp(-tau): that slow tail IS the missing bounces, and it is what keeps a thick
// deck white by day and moonlit at night.
public float cloudSelfShadow(float tau) {
    float sum = 0.0;
    float weightSum = 0.0;
    float weight = 1.0;
    for (int i = 0; i < 3; i++) {
        sum += weight * exp(-tau * exp2(-float(i)));
        weightSum += weight;
        weight *= 0.5;
    }
    return sum / weightSum;
}

// ---- The instance grid ------------------------------------------------------------------------------
//
// One hash grid answers "which cloud, where, how posed" for every consumer. The cell grows with the
// Cloud Size dial so an instance's footprint always fits inside its cell: a sky of large decks has
// proportionally larger gaps between their centres. Size is the only dial the grid reads, which is
// what keeps the layout still while thickness, density and steps change what the deck is made of.

// Horizontal footprint (full width) of one instance box, in blocks: the Cloud Size dial. It is
// pushed in cloudColor.x and is the ONLY thing that sizes an instance, so changing the slab's
// thickness moves no cloud.
float cloudSpan(WorldPush push) {
    return clamp(push.cloudColor.x, 32.0, 2048.0);
}

// Vertical extent of one instance box, in blocks: the Cloud Thickness dial, directly. The box STANDS
// ON the cloud base, so the volume's flat bottom is the layer's own base and the crown grows upward
// into clear air; a box centered on the slab would hang half a body below the base, where the terrain
// and the planar tier both say the cloud has already ended. Thickness moves nothing else: the
// footprint is Cloud Size, and the column the tiers price is CLOUD_COLUMN_DEPTH.
float cloudBoxHeight(WorldPush push) {
    return clamp(push.cloud0.z, 8.0, 1024.0);
}

float cloudBoxBottom(WorldPush push) {
    return push.cloud0.y;
}

// The height the single-sample consumers read the field at: CLOUD_SHEET_HEIGHT up the box, which is
// where the authored profiles carry their mass. The planar sheet's crossing, the shadow probes and the
// continuation-ray transmittance all use it, so neither switching tiers nor flying in and out of the
// walk moves the cloud.
public float cloudBoxMid(WorldPush push) {
    return cloudBoxBottom(push) + CLOUD_SHEET_HEIGHT * cloudBoxHeight(push);
}

// The walk's extinction per block: the column every tier prices, spread over the deck's actual depth.
// This is what keeps Cloud Thickness a shape dial — a deeper deck carries the same water, so it is
// thinner per unit length and no darker — and it is the only place the box height reaches opacity.
float cloudDensityScale(WorldPush push) {
    return clamp(CLOUD_COLUMN_DEPTH / cloudBoxHeight(push), 0.0, 4.0);
}

float cloudGridCell(WorldPush push) {
    return max(CLOUD_GRID_CELL, cloudSpan(push) / CLOUD_BOX_SPAN);
}

float cloudGridHash(float2 cell, float salt) {
    return fogHash13(float3(cell.x, cell.y, salt));
}

// Density 0..1 of the layer at one drifted-grid position and one normalized box height (0 at the
// cloud base, 1 at the crown). Returns 0 wherever no instance holds the point. `driftedXZ` has already been moved through cloud2 (anchor +
// drift); the instance centers live in the same drifted frame, which is what makes the wind slide
// them. The two adjacent atlas layers are lerped by hand — 2D-array filtering is nearest along the
// layer axis — and the box window keeps a clipped neighbour from drawing a visible box wall on top
// of the volumes' own soft borders.
float cloudInstanceDensity(WorldPush push, float2 driftedXZ, float vertical01) {
    if (push.cloud0.x <= 0.0 || vertical01 < 0.0 || vertical01 > 1.0) {
        return 0.0;
    }
    float cell = cloudGridCell(push);
    float2 cellId = floor(driftedXZ / cell);
    float roll = cloudGridHash(cellId, 17.0);
    // Weather systems: a bilinear-smoothed hash over 4x4-cell parent blocks modulates the occupancy
    // threshold zero-mean, so whole neighbourhoods fill or clear together and the patchiness slider
    // gathers instances into clusters instead of sprinkling them. The interpolation matters: a binary
    // per-block hash drew straight grid-aligned fronts across the sky. Near-full coverage pins the
    // gate open — a solid overcast is a deck of touching clouds, and the slider's 1.0 must deliver it.
    float2 parent = cellId / 4.0;
    float2 pf = floor(parent);
    float2 pu = frac(parent);
    float cluster = lerp(
            lerp(cloudGridHash(pf, 83.0), cloudGridHash(pf + float2(1.0, 0.0), 83.0), pu.x),
            lerp(cloudGridHash(pf + float2(0.0, 1.0), 83.0), cloudGridHash(pf + float2(1.0, 1.0), 83.0), pu.x),
            pu.y);
    float gate = push.cloud0.x >= 0.95
            ? 2.0
            : saturate(push.cloud0.x * (1.0 + push.cloud1.z * (cluster * 2.0 - 1.0)));
    if (roll >= gate) {
        return 0.0;
    }
    float span = cloudSpan(push);
    float2 jitter = float2(cloudGridHash(cellId, 29.0), cloudGridHash(cellId, 43.0)) - 0.5;
    float pose = cloudGridHash(cellId, 61.0);
    float2 center = (cellId + 0.5 + jitter * (1.0 - span / cell)) * cell;
    // Quarter-turns plus a mirror: eight poses out of one texture, so the grid never reads as a stamp.
    float2 q = driftedXZ - center;
    float turn = floor(pose * 3.999);
    float2 r = turn < 0.5 ? q
            : turn < 1.5 ? float2(-q.y, q.x)
            : turn < 2.5 ? -q
            : float2(q.y, -q.x);
    if (frac(pose * 7.31) < 0.5) {
        r.x = -r.x;
    }
    // Patchiness also breathes the footprint, so a sparse sky carries visibly smaller decks.
    float scale = 1.0 + (frac(pose * 3.77) - 0.5) * push.cloud1.z * 0.5;
    float2 local = r / (span * scale) + 0.5;
    // Rounded-square window (a superellipse over the centered coordinates). A plain box window drew
    // straight vertical walls wherever the baked cloud ran to its own border, and every instance read
    // as a square card; the wide ramp blends the crop into the volumes' own border fade.
    float2 cc = abs(local - 0.5) * 2.0;
    float radius = pow(cc.x * cc.x * cc.x + cc.y * cc.y * cc.y, 1.0 / 3.0);
    float window = 1.0 - smoothstep(0.96, 1.12, radius);
    if (window <= 0.0) {
        return 0.0;
    }
    // Shape variety: most instances render the pushed shape; a hash minority renders its neighbour in
    // the slider order, so one sky carries two shapes without the type dial losing its meaning. The
    // clamp rides cloudColor.y, the atlas's last shape index Java pushes alongside.
    float shapeRoll = cloudGridHash(cellId, 97.0);
    float shape = clamp(floor(push.cloud1.x)
            + (shapeRoll < 0.18 ? -1.0 : shapeRoll > 0.82 ? 1.0 : 0.0),
            0.0, max(push.cloudColor.y, 0.0));
    // Atlas layer = shape * RES + height slice; the clamp keeps the hand-made trilinear's second
    // tap inside the shape's own layer band.
    float layerF = min(vertical01 * (CLOUD_VOLUME_RES - 1.0), CLOUD_VOLUME_RES - 1.001);
    float l0 = floor(layerF);
    float layer = shape * CLOUD_VOLUME_RES + l0;
    float a = cloudVolumes.SampleLevel(float3(local.x, local.y, layer), 0.0).r;
    float b = cloudVolumes.SampleLevel(float3(local.x, local.y, layer + 1.0), 0.0).r;
    return lerp(a, b, layerF - l0) * window;
}

// Transmittance of the cloud layer along a sun leg from `pos` (rebased): the density sampled where
// the leg crosses the layer's mid-height plane, with the optical depth thickened by the leg's slant.
// This is the one estimate the ground-level NEE, the fog sun legs and the sheet's own self-shadow all
// share, so a clouded sun dims identically in all of them. Returns 1 while clouds are off — the
// callers' multiply keeps them bit-identical.
public float cloudShadowTransmittance(float3 pos, float3 sunDir) {
    if ((worldPush.flags & WORLD_FLAG_CLOUD) == 0u || worldPush.cloud0.x <= 0.0) {
        return 1.0;
    }
    if (sunDir.y < 0.05) {
        return 1.0; // the body sits at or below the horizon; there is no light left to shadow
    }
    float t = (cloudBoxMid(worldPush) - pos.y) / sunDir.y;
    if (t <= 0.0) {
        return 1.0; // the sample is at or above the layer along the sun leg
    }
    float2 drifted = pos.xz + sunDir.xz * t + worldPush.cloud2.xy + worldPush.cloud2.zw;
    float density = cloudInstanceDensity(worldPush, drifted, 0.5);
    if (density <= 0.0) {
        return 1.0;
    }
    float tau = density * worldPush.cloud0.w * CLOUD_EXTINCTION * CLOUD_COLUMN_DEPTH / sunDir.y;
    return exp(-tau);
}

/** Field sample at one world position's projection, for the debug views. */
public float cloudCoverageWorld(float2 rebasedXZ) {
    if ((worldPush.flags & WORLD_FLAG_CLOUD) == 0u || worldPush.cloud0.x <= 0.0) {
        return 0.0;
    }
    return cloudInstanceDensity(worldPush, rebasedXZ + worldPush.cloud2.xy + worldPush.cloud2.zw,
            CLOUD_SHEET_HEIGHT);
}

// The colour the pushed illuminance is tinted with, and how far it may be pulled off white. The derived
// body's colour is normalised to unit LUMINANCE rather than to its brightest channel, so the deck keeps the
// level the look constant means and only its hue moves: normalising to the brightest channel would dim the
// deck by the colour's own luminance - 0.42 for a one-degree sun - and a warm deck two stops down reads as
// brown. Applied at full strength even the luminance-normalised one-degree sunset colour (red at 2.4, blue
// at 0.12) is an impossible light, and the ACES highlight roll-off turned the deck khaki rather than a
// sunset, which is why the trim is a constant and not the raw colour. A body that is down derives a zero
// colour, and that returns zero: no light, whatever the look constant says.
public static const float CLOUD_LIGHT_TINT = 0.5;

public float3 cloudLightTint(float3 derivedIlluminance) {
    float lum = dot(derivedIlluminance, float3(0.2126, 0.7152, 0.0722));
    if (lum <= 1.0e-9) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 chroma = derivedIlluminance / lum;
    return lerp(float3(1.0, 1.0, 1.0), chroma, CLOUD_LIGHT_TINT);
}

// The planar sheet as sky.rmiss renders it: single scatter of the dominant celestial where the view
// ray crosses the layer's mid-height plane, plus the sky-dome ambient, with the sheet's transmittance
// out for the caller to attenuate the sky behind it (atmosphere, stars, discs) with. The density is
// one instance fetch at the crossing — the same field the shadow probes read, which is exactly what
// makes the sheet and its shadows agree. A camera inside the body sees the mid-height cut; far
// crossings fade into the haze they sit behind.
public float3 planarCloudSkyInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                      out float cloudT) {
    cloudT = 1.0;
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float mid = cloudBoxMid(push);
    bool below = ro.y < mid;
    bool above = ro.y > mid;
    if (below == above) {
        return float3(0.0, 0.0, 0.0);
    }
    if (below ? rd.y < 1.0e-3 : rd.y > -1.0e-3) {
        return float3(0.0, 0.0, 0.0); // the ray runs away from the plane
    }
    float t = (mid - ro.y) / rd.y;
    // Dissipation distance (cloud1.w): the sheet fades over the last 40% of the pushed range and
    // is not rendered at all past it — clouds beyond the fade cost nothing. Deliberately a sheet-
    // only fade: the shadow field stays unfaded, because the layer still exists overhead and a
    // shadow dissolving with distance would smear across the ground as a gradient.
    float fadeDist = max(push.cloud1.w, 64.0);
    float fade = 1.0 - smoothstep(fadeDist * 0.6, fadeDist, t);
    if (fade <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    // Two vertical taps straddling the sheet height average the column the crossing implies, so the
    // flat tier's silhouette carries some of the volume the walk would resolve instead of one
    // slice's speckle.
    float boxH = cloudBoxHeight(push);
    float slabBand = CLOUD_SHEET_BAND;
    float2 drift = ro.xz + rd.xz * t + push.cloud2.xy + push.cloud2.zw;
    float density = 0.5 * (cloudInstanceDensity(push, drift, CLOUD_SHEET_HEIGHT - slabBand)
            + cloudInstanceDensity(push, drift, CLOUD_SHEET_HEIGHT + slabBand));
    if (density <= 0.0) {
        return float3(0.0, 0.0, 0.0); // cloudT stays 1: clear sky, nothing to attenuate with
    }
    // Perspective thickening: a grazing sight line crosses proportionally more slab.
    float tau = density * push.cloud0.w * CLOUD_EXTINCTION * CLOUD_COLUMN_DEPTH / max(abs(rd.y), 0.04);
    // The fade dissolves the slab's OPACITY, not just its glow: radiance and occlusion scale by
    // the same factor (cloudT = 1 - opacity). Fading the radiance alone painted the dissipation
    // band black — a grazing sight line saturates the transmittance to 0 long before the fade
    // ends, so cloudL + 0 x sky approached zero and the far edge read as a black silhouette band.
    float opacity = (1.0 - exp(-tau)) * fade;
    cloudT = 1.0 - opacity;
    // Which body lights the sheet, and what colour its light arrives in, come from the DERIVED celestial —
    // the same derivation world.rgen hands the fog, so the sky, the fog and the sheet cannot disagree about
    // which body is up. Its LEVEL deliberately does not: the atmosphere's transmittance along a
    // one-degree sun leaves a rounding error, and lighting the deck with that literal left the whole deck
    // black at an hour when a real one burns. The level stays the pushed look constant, which is what keeps
    // the deck's brightness on its daytime curve, and what the derived light contributes is a fraction of
    // its chromaticity (see cloudLightTint). A body that is down arrives at zero transmittance, so its tint
    // is nothing and the deck darkens by itself — the atmosphere is the horizon gate, with no separate
    // elevation test needed.
    CelestialLight light = dominantCelestialLight(state, transmittanceLut);
    float3 lightDir = light.dir;
    float lightLevel = dot(lightDir, state.sunDir) >= dot(lightDir, state.moonDir)
            ? state.sunIlluminance : state.moonIlluminance;
    float3 lightIll = lightLevel * cloudLightTint(light.illuminance);
    float tauSelf = density * push.cloud0.w * CLOUD_EXTINCTION * CLOUD_COLUMN_DEPTH
            / max(lightDir.y, 0.05);
    // Multiple-scatter approximation for the slab's self-shadow. A pure single-scatter sheet darkens a
    // thick deck to ambient-only, which reads as blue-grey daytime clouds and as black cut-outs at night;
    // real cloud tops are bright because photons bounce inside them. Three octaves of halved extinction,
    // energy-normalised so an unshadowed sample still reaches exactly 1 — thin edges keep the exact
    // single-scatter value, thick cores keep a lit presence.
    float selfT = cloudSelfShadow(tauSelf);
    float phase = CLOUD_SCATTER_FLOOR + CLOUD_SCATTER_LOBE * hg(dot(rd, lightDir), CLOUD_G) * 0.7;
    // The sheet's ambient is the sky's HEMISPHERE irradiance — the same bake the fog and the water read,
    // evaluated through sky.slang so the three cannot drift apart — and not the radiance of a single
    // direction. A zenith sample is the coldest, dimmest part of a sunset sky, so a deck hanging under a
    // bright warm horizon band came out grey. INV_PI turns the cos-weighted integral back into the average
    // radiance the albedo multiplies. The weight is clamped: it is a config lane, and the night this
    // shipped a lane-order slip fed a growing drift accumulator into it and the sky glowed. A bounded read
    // makes that class of accident merely wrong instead of blinding.
    float3 ambient = skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0)) * INV_PI
            * clamp(push.cloudColor.w, 0.0, 2.0) * CLOUD_AMBIENT_GAIN;
    return CLOUD_ALBEDO * opacity * (phase * lightIll * selfT + ambient);
}

// ---- Volumetric cloud march (the raymarcher tier) ----------------------------------------------------
//
// The walk spans the instance box — standing on cloudBase, Cloud Thickness deep — while the planar
// tier prices the same body in one crossing: both price CLOUD_COLUMN_DEPTH, and cloudDensityScale
// spreads that column over the box's height. So this tier is not a second
// parameterisation of the clouds: it is the same layer, walked instead of estimated, and both tiers
// answer the same question at the same scale (which is why the step count can be turned down and the
// two must not disagree about how much light gets through).
//
// What the walk adds over a single crossing is the baked volume's real 3D shape and a light march
// through it — the two things a plane crossing cannot express, and the reason the extra cost buys
// anything at all. The march is the clouds_unreal raymarcher's: jittered primary steps accumulating
// Beer-Lambert extinction, and per in-cloud sample a short secondary march toward the dominant body
// whose accumulated tau feeds the same multi-scatter octave stack the sheet uses, so both tiers grade
// identically under the same light.

public static const float CLOUD_STEP_MIN = 8.0;        // steps below this alias visibly at the rims
// Samples the walk spends per slab thickness when the crossing is close to vertical. The baked shape
// is resolved at that rate; the pushed step lane is a CEILING the crossing grows into as the
// perspective thickens it (1/|rd.y|), not a rate every ray must pay.
public static const float CLOUD_STEPS_PER_THICKNESS = 12.0;
// Secondary (light) march: steps and their minimum world-space length. Three taps is what the baked
// volumes need — their density is smooth on the scale of a slice, so a longer march buys aliasing
// samples, not accuracy — and the multi-scatter stack on top of it carries the depth the raw
// single-scatter tau would miss.
public static const int CLOUD_LIGHT_STEPS = 3;
public static const float CLOUD_LIGHT_STEP_MIN = 24.0; // blocks per light leg

// Light-march optical depth from one point toward the body: the instance field re-evaluated along
// the leg, bounded, early-outing once the tau is deep enough that the octave stack is saturated
// anyway. The leg leaves the sample's own instance quickly at off-vertical sun angles, and the
// neighbour it enters carries the rest — the grid lookup is the same one the primary step used.
float cloudLightTau(WorldPush push, float3 p, float3 lightDir, float boxH) {
    float stepLen = max(boxH, CLOUD_LIGHT_STEP_MIN) / float(CLOUD_LIGHT_STEPS);
    float tau = 0.0;
    float3 lp = p;
    for (int s = 0; s < CLOUD_LIGHT_STEPS; s++) {
        lp += lightDir * stepLen;
        float v = (lp.y - cloudBoxBottom(push)) / boxH;
        if (v < 0.0 || v > 1.0) {
            break; // the leg left the slab; whatever is above it no longer occludes this sample
        }
        tau += cloudInstanceDensity(push,
                lp.xz + push.cloud2.xy + push.cloud2.zw, v) * CLOUD_EXTINCTION * stepLen
                * cloudDensityScale(push);
        if (tau > 4.0) {
            break; // the stack's output is flat past this; the taps would be pure waste
        }
    }
    return tau;
}

// Volumetric cloud slab as sky.rmiss renders it: a jittered walk across the slab accumulating
// in-scatter and transmittance. `pix` seeds the pixel-fixed blue-noise offset that advances one
// golden-ratio stride per frame — a per-frame re-roll is a signal the denoiser cannot accumulate,
// while a rotating pixel-fixed pattern is one it averages. The fade and the multi-scatter doctrine
// are the planar tier's, so the tiers degrade into each other instead of jumping.
public float3 volumetricCloudInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                       float2 pix, out float cloudT) {
    cloudT = 1.0;
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0 || abs(rd.y) < 1.0e-6) {
        // A ray parallel to the slab never crosses its thickness; a zero-coverage layer has nothing to
        // cross. Either way cloudT stays 1 and the caller's compose is a no-op.
        return float3(0.0, 0.0, 0.0);
    }
    float boxBottom = cloudBoxBottom(push);
    float boxH = cloudBoxHeight(push);
    float boxTop = boxBottom + boxH;
    float ta = (boxBottom - ro.y) / rd.y;
    float tb = (boxTop - ro.y) / rd.y;
    float t0 = max(min(ta, tb), 0.0);
    float t1 = max(ta, tb);
    if (t1 <= t0) {
        return float3(0.0, 0.0, 0.0); // the camera is above the layer looking away, or below looking down
    }
    // Dissipation distance (cloud1.w): same sheet-only fade the planar tier applies, and applied to
    // the slab's OPACITY rather than to its glow alone — fading the radiance alone painted the
    // dissipation band black.
    float fadeDist = max(push.cloud1.w, 64.0);
    float fade = 1.0 - smoothstep(fadeDist * 0.6, fadeDist, t0);
    if (fade <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    // The same body selection and the same tinted-level rule as the sheet, so the tier toggle cannot
    // change the light the deck is drawn under.
    CelestialLight light = dominantCelestialLight(state, transmittanceLut);
    float3 lightDir = light.dir;
    float lightLevel = dot(lightDir, state.sunDir) >= dot(lightDir, state.moonDir)
            ? state.sunIlluminance : state.moonIlluminance;
    float3 lightIll = lightLevel * cloudLightTint(light.illuminance);
    float cosTheta = dot(rd, lightDir);
    // Beer-Powder along the raymarcher's formula: 2/(1 + exp(-2 cos)) is 1 side-on and swells toward 2
    // when the view runs with the light, which is what makes a back-lit rim read as a lit rim.
    float phase = CLOUD_SCATTER_FLOOR + CLOUD_SCATTER_LOBE
            * hg(cosTheta, CLOUD_G) * 0.7 * (2.0 / (1.0 + exp(-2.0 * cosTheta)));
    float3 ambient = skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0)) * INV_PI
            * clamp(push.cloudColor.w, 0.0, 2.0) * CLOUD_AMBIENT_GAIN;

    float steps = clamp(push.volumeParams.w, CLOUD_STEP_MIN, 96.0);
    // The budget scales with the crossing: an overhead sight line crosses the box's own height,
    // and CLOUD_STEPS_PER_THICKNESS samples resolve what the walk exists for (the baked shape).
    // Spending the full pushed lane there would only sample cells the step was already wider than,
    // so the lane is a ceiling the crossing grows into as the perspective thickens it — grazing rays
    // pay it, vertical ones pay the floor.
    steps = clamp(ceil((t1 - t0) / max(boxH / CLOUD_STEPS_PER_THICKNESS, 1.0)),
            CLOUD_STEP_MIN, steps);
    float dt = (t1 - t0) / steps;
    float jitter = frac(interleaveGradientNoise(pix)
            + float(push.frameIndex) * 0.6180339887);
    float3 L = float3(0.0, 0.0, 0.0);
    float T = 1.0;
    for (int i = 0; i < int(steps); i++) {
        float3 p = ro + rd * (t0 + (float(i) + jitter) * dt);
        float v = (p.y - boxBottom) / boxH;
        float density = cloudInstanceDensity(push,
                p.xz + push.cloud2.xy + push.cloud2.zw, v);
        // cloudDensityScale spreads the calibrated column optical depth over the box's actual height,
        // so the walk resolves the same deck the planar tier prices rather than a denser one.
        float tauCell = density * push.cloud0.w * CLOUD_EXTINCTION * dt * cloudDensityScale(push);
        if (tauCell > 1.0e-4) {
            float cellT = exp(-tauCell);
            // The light march: the baked shape's own occlusion toward the body, graded through the
            // same multi-scatter octaves the sheet's column estimate uses. The ambient leg is
            // occluded by the same tau at a softened weight — what a viewer mostly sees of a deck
            // is its BASE, where the direct sun is gone and the brightness comes from light that
            // has bounced many times inside the cloud. Occluding the sky by the full sun tau
            // crushed the base to a dark sheet; that softening is CLOUD_AMBIENT_OCCLUSION, and it
            // is also the only term that can light a base at all.
            float tauSun = cloudLightTau(push, p, lightDir, boxH) * push.cloud0.w;
            float3 sunT = cloudSelfShadow(tauSun);
            float ambT = max(cloudSelfShadow(tauSun * CLOUD_AMBIENT_OCCLUSION),
                    CLOUD_AMBIENT_FLOOR);
            float3 scatter = CLOUD_ALBEDO * (phase * lightIll * sunT + ambient * ambT);
            L += T * (1.0 - cellT) * scatter;
            T *= cellT;
            if (T < 3.0e-3) {
                break;
            }
        }
    }
    // Fade dissolves the slab's opacity, with the radiance following it at the same scale — half a fade
    // is half a cloud, never a glowing sheet that has stopped blocking.
    float opacity = (1.0 - T) * fade;
    cloudT = 1.0 - opacity;
    return L * fade;
}

// Extinction only, at the planar tier's price: a continuation ray (a reflection, a refraction, an
// indirect bounce) must be dimmed by the deck it passed through without paying a walk per bounce —
// the "Pass B does extinction only" contract. One instance fetch at the slab's mid-crossing and the
// same slant-thickened tau the planar tier uses keeps a bounce's attenuation the MEAN of what the
// primary ray walked.
public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0 || abs(rd.y) < 1.0e-6) {
        return 1.0;
    }
    float boxBottom = cloudBoxBottom(push);
    float boxH = cloudBoxHeight(push);
    float boxTop = boxBottom + boxH;
    if (rd.y > 0.0 ? ro.y >= boxTop : ro.y <= boxBottom) {
        return 1.0; // the ray runs away from the layer
    }
    float ta = (boxBottom - ro.y) / rd.y;
    float tb = (boxTop - ro.y) / rd.y;
    float tMid = 0.5 * (max(min(ta, tb), 0.0) + max(ta, tb));
    float vMid = (ro.y + rd.y * tMid - boxBottom) / boxH;
    float density = cloudInstanceDensity(push,
            ro.xz + rd.xz * tMid + push.cloud2.xy + push.cloud2.zw, vMid);
    if (density <= 0.0) {
        return 1.0;
    }
    float tau = density * push.cloud0.w * CLOUD_EXTINCTION * CLOUD_COLUMN_DEPTH / max(abs(rd.y), 0.04);
    return exp(-tau);
}
// ---- Unified air-volume sampler --------------------------------------------------------------------
//
// The marcher's per-step density source (the marched tier walks this; the analytic tier above bypasses
// it). The density sources meet here: fog carries the whole sample while it is the only enabled layer,
// cloud and spray join it as their layers come online, and the water layer is explicitly NOT sampled — while the
// path is inside water the stack's medium owns extinction and scattering, so mixing the air field in
// would double-count the volume the surface crossing already describes.
public VolumeSample sampleVolume(float3 pos, bool inWater, WorldPush push) {
    VolumeSample s = emptyVolumeSample();
    if (inWater || (push.flags & WORLD_FLAG_FOG) == 0u) {
        return s;
    }
    float sigmaT;
    if ((push.flags & WORLD_FLAG_VOLUME_MARCH) != 0u) {
        sigmaT = fogMarchDensity(push, pos);
    } else {
        sigmaT = fogDensityAt(push, pos.y);
    }
    float3 albedo = max(push.fogColor.rgb, float3(0.0, 0.0, 0.0));
    s.sigmaS = albedo * sigmaT;
    s.sigmaA = (float3(1.0, 1.0, 1.0) - albedo) * sigmaT;
    s.g = push.fog0.w;
    return s;
}

public float fogDensityAt(WorldPush push, float y) {
    float sigma0 = push.fog0.x;
    if (sigma0 <= 0.0) {
        return 0.0;
    }
    return sigma0 * exp(-max(y - push.fog0.z, 0.0) / max(push.fog0.y, 1.0e-4));
}

// ---- Marched tier: heterogeneous fog with per-sample occlusion --------------------------------------
//
// The quality tier Phase 2 adds: uniform stepping over the segment with a pixel-fixed blue-noise
// jitter and FBM-modulated density. Every sample's measured visibility gates that sample's sun term
// and nothing else: the density stays the pushed profile, so the transmittance the caller attenuates
// with is a property of the air rather than of where the sun happens to reach. Only light is culled.
// The visibility itself is measured at the stratum budget below rather than traced once per sample.
//
// Convergence: with noise amplitude 0 and the shadow switch off every sample sees the plain analytic
// profile, and the midpoint-quadrature integral converges to the closed form as the step count rises —
// VolumeFogTest pins that numerically, 8/16/32/64 steps against fogOpticalDepth.
//
// Cost: the march covers only the segment's stretch inside the medium's significant band (see the span
// clamp below), spending steps density evals there, and the shadow probes ride a stratum budget — at
// most FOG_MARCH_PROBE_STRATA geometry shadow rays and cloud-shadow field evaluations per segment,
// wherever in the band the non-empty samples land. The early-outs matter: a segment entirely above the
// band returns without marching, samples past negligible transmittance stop the march, and the probes
// fire only where the medium is non-zero. volumeParams.z (max integration distance) stays reserved —
// truncating the integral would desynchronise the out fogT from the throughput attenuation, which must
// agree.
public static const float FOG_SPAN_TAU = 2.0e-3;      // bound on the optical depth the span clamp excludes
public static const int FOG_MARCH_PROBE_STRATA = 8;   // shadow-probe budget per marched segment

// The radiance a fog sample scatters toward the eye from the sky: the SKY ALONG THE VIEW RAY itself,
// not the hemisphere's average irradiance. This is the term that decides whether a thick fog reads as
// air or as a dark wall, because the in-scatter converges to `albedo * fogSkyRadiance(rd)` as the
// transmittance falls. Sampling the same radiance the miss shader would have returned is what makes a
// fully opaque fog dissolve into the sky it hides; the hemisphere average cannot do that — a sunset's
// energy sits in a few degrees above the horizon while the average is dominated by the cold upper dome,
// so a fog bank built on it reads several stops darker than the sky behind it at exactly the elevation
// where the eye compares the two. Below-horizon rays mirror to their horizon elevation for the reason
// world.rmiss mirrors an escaping ray: the view LUT's lower half is the space floor, and air seen
// downward is haze, not space.
public float3 fogSkyRadiance(float3 rd) {
    SkyState state = skyState(worldPush);
    return skyDiffuseRadiance(skyViewLut, float3(rd.x, abs(rd.y), rd.z), state);
}

public float3 fogSegmentInScatterMarched(float3 ro, float3 rd, float t, float2 pix,
                                         bool shadowAllowed, out float fogT) {
    fogT = 1.0;
    float sigma0 = worldPush.fog0.x;
    float3 albedo = max(worldPush.fogColor.rgb, float3(0.0, 0.0, 0.0));
    if (sigma0 <= 0.0 || dot(albedo, float3(1.0, 1.0, 1.0)) == 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    // The significant band. The density is a function of height alone and is bounded by the analytic
    // profile times the noise's largest multiplier, so above
    //     yTop = baseY + H * ln(sigma0 * (1 + |amp|) / eps),  eps = FOG_SPAN_TAU / t,
    // every block of path carries less than eps of density: the excluded stretch's optical depth — and
    // with it the error in both the returned transmittance and the skipped in-scatter — is bounded by
    // FOG_SPAN_TAU whatever the geometry, orders below anything the estimator can resolve. The march
    // covers only the segment's stretch at or below yTop: a segment entirely above returns without
    // marching, and one crossing the height concentrates its whole step budget on the density that can
    // actually read instead of striding the empty tail above the fog.
    float peak = sigma0 * (1.0 + abs(worldPush.fog1.z));
    float eps = FOG_SPAN_TAU / t;
    if (peak <= eps) {
        return float3(0.0, 0.0, 0.0);
    }
    float yTop = worldPush.fog0.z + max(worldPush.fog0.y, 1.0e-4) * log(peak / eps);
    // The spray band is a second density source at its own height; while it is live its top joins the
    // significant range. Conservatively the span then also covers the fog-free stretch between band
    // and fog top — a few wasted steps, never a wrong one.
    if ((worldPush.flags & WORLD_FLAG_SPRAY) != 0u) {
        yTop = max(yTop, worldPush.water1.w + max(worldPush.water1.z, 0.05));
    }
    float yEnd = ro.y + rd.y * t;
    if (min(ro.y, yEnd) > yTop) {
        return float3(0.0, 0.0, 0.0); // the whole segment sits above the medium
    }
    float t0 = 0.0;
    float t1 = t;
    if (max(ro.y, yEnd) > yTop && abs(rd.y) > 1.0e-6) {
        float tTop = clamp((yTop - ro.y) / rd.y, 0.0, t);
        if (rd.y > 0.0) {
            t1 = tTop;
        } else {
            t0 = tTop;
        }
    }
    if (t1 <= t0) {
        return float3(0.0, 0.0, 0.0);
    }

    int steps = int(clamp(worldPush.volumeParams.x, 1.0, 64.0));
    float jitterStrength = clamp(worldPush.volumeParams.y, 0.0, 1.0);
    bool shadow = shadowAllowed
            && (worldPush.flags & WORLD_FLAG_VOLUME_SHADOW) != 0u
            && celestialLight.halfAngle > 0.0;

    float dt = (t1 - t0) / float(steps);
    // Pixel-fixed interleave gradient with the golden-ratio frame walk — the analytic tier's exact
    // doctrine: a per-frame re-roll is a signal the denoiser cannot accumulate, while this pattern
    // dithers step edges into a stable grain the reconstruct averages out.
    float u = 0.5 + (frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887) - 0.5) * jitterStrength;

    float phase = hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7;
    float3 ambient = worldPush.fogColor.w * fogSkyRadiance(rd);
    float3 sunDir = celestialLight.dir;

    float3 L = float3(0.0, 0.0, 0.0);
    float tau = 0.0;
    // The probes ride a stratum budget: the marched samples split into FOG_MARCH_PROBE_STRATA equal
    // ranges, and each stratum's first non-empty sample fires the geometry shadow ray and the
    // cloud-shadow field evaluation once, with the whole stratum reusing both. A stratum spans
    // steps/FOG_MARCH_PROBE_STRATA samples, so shadow-volume edges quantise at that scale per frame —
    // the pixel-fixed jitter walks the sample positions inside the strata, and the reconstruct resolves
    // the quantisation the same way it resolves the step dither. At or below the stratum count every
    // sample probes, which is the short-segment case where per-sample resolution actually reads.
    int lastStratum = -1;
    bool probeValid = false;
    float stratumVis = 1.0;
    float stratumCloud = 1.0;
    for (int i = 0; i < steps; i++) {
        int stratum = (i * FOG_MARCH_PROBE_STRATA) / steps;
        if (stratum != lastStratum) {
            lastStratum = stratum;
            probeValid = false;
        }
        float tMid = t0 + (float(i) + u) * dt;
        float3 p = ro + rd * tMid;
        float sigma = fogMarchDensity(worldPush, p);
        float vis = 1.0;
        if (sigma > 0.0) {
            if (!probeValid) {
                stratumVis = shadow ? luminance(visibility(p, sunDir, 10000.0).transmittance) : 1.0;
                stratumCloud = cloudShadowTransmittance(p, sunDir);
                probeValid = true;
            }
            vis = stratumVis;
        }
        // The medium is the density profile; the visibility gates the sun term below. Thinning the
        // density instead would make the fog's opacity a function of a probe estimate — a
        // per-frame-walking, stochastic signal a temporal reconstruct cannot accumulate, and one that
        // deletes the medium wherever the sun is occluded (the air over the sea reading as a dark band
        // while the sky-facing segment, which traces no probes, kept its full density).
        float sigmaEff = sigma;
        if (sigmaEff <= 0.0) {
            continue;
        }
        float tauMid = tau + sigmaEff * 0.5 * dt;
        float tMid_transmittance = exp(-tauMid);
        if (tMid_transmittance < 0.004) {
            break;
        }
        float3 sunTerm = float3(0.0, 0.0, 0.0);
        // The floor rides the sun term, not the medium: a fully shaded stratum keeps this fraction of
        // its direct light, which softens the fog's shadow edges instead of making the air vanish in them.
        float sunVis = max(vis, fogShadowFloor());
        if (sunVis > 0.0) {
            // Toward-sun transmittance: the analytic profile — the noise modulation of the sun leg is
            // deliberately not marched (a second full march per sample for a second-order softening of
            // the shafts is not a trade this tier makes). The cloud shadow rides on top: the same field
            // the sheet renders and the ground NEE reads, so a clouded sun thins the shafts instead of
            // lighting them.
            float sunFogT = fogTransmittance(sigma0, worldPush.fog0.z, worldPush.fog0.y,
                    p.y, sunDir.y, 10000.0);
            sunTerm = phase * celestialLight.illuminance
                    * (sunVis * sunFogT * stratumCloud);
        }
        L += tMid_transmittance * (albedo * sigmaEff) * (sunTerm + ambient) * dt;
        tau += sigmaEff * dt;
    }
    fogT = exp(-tau);
    return L;
}

// ---- The water column ------------------------------------------------------------------------------
//
// The column carries the same structure-gated shaft layer the air uses, on the medium pushed in
// `waterShaft`: its own extinction decides how far the layer is worth integrating, its ceiling caps the
// forward lobe so the surface you look up at cannot go white, and the gate draws the beams out of the
// shadow structure.
//
// The body's absorption (`composite.water-opacity`) is untouched: it lives in the surface-radiance path
// (`medium.extinction`), not in this term, so water still darkens with depth exactly as it did.

// ---- Segment-level lighting assembly ---------------------------------------------------------------
//
// The analytic fog tier's single-scatter over one path segment. The MEDIUM is the height profile alone:
// density and transmittance must not depend on where the sun happens to reach. Tying them to visibility
// makes the fog's opacity a function of a four-probe estimate — a stochastic, per-frame-walking signal
// no temporal reconstruct can accumulate, and one that deletes the medium wherever the probes land in
// shadow. That is what turned the air over the sea into a dark band while the sky-facing segment, which
// traces no probes at all, kept its full density: the two halves of the horizon were integrating
// different media.
//
// Visibility therefore gates LIGHT, not matter. FOG_SUN_BANDS probes placed at equal in-scatter-weight
// depths (bisection on the closed form; pixel-fixed IGN jitter walked by a golden-ratio sequence, so the
// reconstruct resolves their quantisation) each contribute their own measured visibility to their own
// stratum's direct term, and a stratum the sun does not reach contributes none of it. The sky ambient is
// not gated at all — shadowed air still scatters sky light, which is what makes a backlit valley read as
// blue haze instead of as clear air.
//
// Writes the segment's fog transmittance so the caller attenuates throughput with the same quantity
// the in-scatter used. `celestialLight` (world_core) must have been derived for this frame and
// converted to ACEScg, exactly as world_indirect's raygen does; callers in a pass that skips NEE set
// it only when they actually call this.
public static const int FOG_SUN_BANDS = 4;

// Direct-sun fraction a fully shaded stratum keeps — the live slider volume.fog-shadow-floor riding
// fog1.w. A lighting control, never a density one: the medium is the same everywhere, so the slider
// decides how hard the fog's shadow edges read (0 = direct sun fully culled inside shadow, 0.95 = the
// sun reaches almost everywhere) without changing how much air the ray crosses.
public float fogShadowFloor() {
    return clamp(worldPush.fog1.w, 0.0, 0.95);
}

// Pixel-fixed hash for the per-band probe jitter. Deliberately NOT frame-indexed: a per-frame
// re-roll is a signal the denoiser cannot accumulate, while a static per-pixel pattern dithers the
// shadow-volume edges into a stable grain the reconstruct resolves into soft edges.
public float interleaveGradientNoise(float2 pix) {
    return frac(52.9829189 * frac(dot(pix, float2(0.06711056, 0.00583715))));
}

// Depth at which the segment's cumulative in-scatter weight reaches `weight` (equivalently, where
// transmittance drops to 1 - weight), by bisection on the closed-form optical depth. Monotone, so the
// bisection always brackets. This is what places the sun-visibility probes: equal weight intervals
// contribute equal Monte-Carlo weight regardless of where the density puts them.
public float fogDepthAtWeight(float sigma0, float baseY, float heightH,
                              float y0, float dirY, float t, float weight) {
    float tauTarget = -log(max(1.0 - weight, 1.0e-4));
    if (weight <= 0.0) {
        return 0.0;
    }
    if (fogOpticalDepth(sigma0, baseY, heightH, y0, dirY, t) <= tauTarget) {
        return t;
    }
    float lo = 0.0;
    float hi = t;
    for (int i = 0; i < 18; i++) {
        float mid = 0.5 * (lo + hi);
        if (fogOpticalDepth(sigma0, baseY, heightH, y0, dirY, mid) < tauTarget) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

// Analytic fog transmittance from the segment start toward the sun, used when the shadow switch is
// off and no probe exists to evaluate it from. Takes the segment's (possibly culled) sigma so the
// toward-sun transmittance agrees with the medium the segment actually integrates.
public float fogT_toSunRo(float3 ro, float3 sunDir, float sigma) {
    return fogTransmittance(sigma, worldPush.fog0.z, worldPush.fog0.y,
            ro.y, sunDir.y, 10000.0);
}

public float3 fogSegmentInScatter(float3 ro, float3 rd, float t, float2 pix,
                                  bool allowShadowCull, out float fogT) {
    // Tier dispatch: the march flag routes the whole segment to the heterogeneous integrator, which
    // reuses the same shadow switch, phase, ambient and lighting doctrine. `allowShadowCull` keeps its
    // meaning — the sky-facing miss segment passes false and its march runs without shadow rays, the
    // same cost gate the analytic tier's probes honour.
    //
    // The march tier exists to resolve a heterogeneous FIELD: the density noise and the spray band.
    // With both off it integrates exactly the profile the closed form already owns, so every step it
    // spends buys nothing but the jitter of its own quadrature — a per-frame signal the reconstruct
    // cannot accumulate, on a frame that has no field to resolve. Take the closed form then; the flag
    // stays the tier selector for frames that actually carry one.
    bool heterogeneous = worldPush.fog1.z > 0.0 || (worldPush.flags & WORLD_FLAG_SPRAY) != 0u;
    if ((worldPush.flags & WORLD_FLAG_VOLUME_MARCH) != 0u && worldPush.volumeParams.x >= 1.0
            && heterogeneous) {
        return fogSegmentInScatterMarched(ro, rd, t, pix, allowShadowCull, fogT);
    }
    float sigma0 = worldPush.fog0.x;
    fogT = 1.0;
    float3 albedo = max(worldPush.fogColor.rgb, float3(0.0, 0.0, 0.0));
    if (sigma0 <= 0.0 || dot(albedo, float3(1.0, 1.0, 1.0)) == 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }

    bool shadowCull = allowShadowCull
            && (worldPush.flags & WORLD_FLAG_VOLUME_SHADOW) != 0u
            && celestialLight.halfAngle > 0.0;

    // One shadow probe per equal-weight stratum, placed on the field with a pixel-fixed,
    // frame-walking jitter. Each sample's visibility is kept for its own stratum's sun term; nothing
    // here touches the medium, so the transmittance below is a property of the air alone.
    float vis[FOG_SUN_BANDS];
    float stratumDepth[FOG_SUN_BANDS];
    float noise = frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887);
    for (int k = 0; k < FOG_SUN_BANDS; k++) {
        vis[k] = 1.0;
        stratumDepth[k] = 0.0;
    }
    if (shadowCull) {
        float scatterWeightFull = 1.0 - fogTransmittance(sigma0, worldPush.fog0.z,
                worldPush.fog0.y, ro.y, rd.y, t);
        if (scatterWeightFull > 1.0e-4) {
            for (int k = 0; k < FOG_SUN_BANDS; k++) {
                float stratumStart = scatterWeightFull * (float(k) + noise) / float(FOG_SUN_BANDS);
                stratumDepth[k] = fogDepthAtWeight(sigma0, worldPush.fog0.z, worldPush.fog0.y,
                        ro.y, rd.y, t, stratumStart + scatterWeightFull * 0.5 / float(FOG_SUN_BANDS));
                vis[k] = luminance(visibility(ro + rd * stratumDepth[k], celestialLight.dir,
                        10000.0).transmittance);
            }
        }
    }

    // The medium: the height profile, unmodulated by light. `fogT` is what the caller attenuates its
    // throughput with, so it has to be the same quantity every frame for a given ray — a visibility-
    // scaled sigma made it a per-pixel lottery and let a shadowed stretch read as clear air.
    float sigma = sigma0;
    fogT = fogTransmittance(sigma, worldPush.fog0.z, worldPush.fog0.y, ro.y, rd.y, t);
    float scatterWeight = 1.0 - fogT;
    // HG lobe scaled to 70% energy: a pure forward lobe blows out when the view aligns with the
    // sun down a bright shaft. No isotropic floor — a floor at side/back angles reads as a uniform
    // milk wash, and the strata probes already light the lit medium geometrically.
    float phase = hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7;
    float3 ambient = worldPush.fogColor.w * fogSkyRadiance(rd);
    float3 sunPart = float3(0.0, 0.0, 0.0);
    if (!shadowCull) {
        // Unshadowed sun term: the closed form with V = 1 everywhere, sun transmittance taken at
        // the segment start. The cloud shadow still applies — geometry shadows are off, the layer
        // above is not.
        sunPart = phase * celestialLight.illuminance * (scatterWeight
                * fogT_toSunRo(ro, celestialLight.dir, sigma)
                * cloudShadowTransmittance(ro, celestialLight.dir));
    } else if (scatterWeight > 1.0e-4) {
        // Per-stratum sun in-scatter: the stratum's share of the medium times its own measured
        // visibility times the closed-form fog transmittance from its probe toward the sun. Strata
        // whose probe landed in shadow contribute none of it — that IS the culling, and it is light
        // only. The cloud shadow multiplies the same field every other sun leg reads.
        float stratumWeight = scatterWeight / float(FOG_SUN_BANDS);
        for (int k = 0; k < FOG_SUN_BANDS; k++) {
            float3 probe = ro + rd * stratumDepth[k];
            float sunFogT = fogTransmittance(sigma, worldPush.fog0.z, worldPush.fog0.y,
                    probe.y, celestialLight.dir.y, 10000.0);
            float sunVis = max(vis[k], fogShadowFloor());
            sunPart += phase * celestialLight.illuminance * (stratumWeight * sunVis * sunFogT
                    * cloudShadowTransmittance(probe, celestialLight.dir));
        }
    }
    return albedo * (sunPart + scatterWeight * ambient);
}

// ---- Sun shafts: one integrator for the air and for the water --------------------------------------
//
// Airlight: the sun's beams made visible by the medium itself, with no height fog involved. The two are
// separate media on purpose. The fog's transmittance is what makes a scene hazy, so god rays that
// "need fog" always arrive with a veil; a shaft medium adds light and attenuates nothing, which is the
// only way to have beams in clear air. The same integrator runs underwater, where the medium is the
// water body's own scattering instead of the pushed air density.
//
// The integral is the segment's single scatter over a uniform medium,
//
//     L = structure * strength * phase * E * ∫_0^min(t, reach) sigma*exp(-sigma*s) * V(s) * cloudShadow(s) ds
//
// split into SHAFT_STRATA strata of equal in-scatter weight, so the strata follow the medium's own
// e-folding instead of even spacing: air (sigma ~ 1e-3) puts them almost uniformly along the reach,
// while water (sigma ~ 0.1-0.5) packs them into the few blocks the light actually crosses. A point-
// sampled sun visibility V sits inside each stratum. There is no ambient term: sky light scattered
// toward the eye is not a beam, and adding it would only lift the shade the beams have to contrast
// with. The water column has no ambient of its own any more; see the section at the end.
//
// `structure` is what makes this read as a Tyndall beam instead of as fog. A path that never leaves the
// light scatters uniformly along its whole length, and that smooth lift — added to every sunlit wall,
// every open field, the sky-facing half of every room — is indistinguishable from a mist veil. It is
// also the one part of this integral the sky LUT already renders as the atmosphere's own in-scatter.
// So the term is faded out by the lit fraction of the path: fully-lit air contributes nothing, air that
// straddles a shadow boundary contributes in full, and half-lit air keeps half. What survives is exactly the shadow structure the
// eye reads as a shaft, which is also why the beams are only drawn where the sun is occluded somewhere
// along the line of sight — the physical reason they are visible at all.
//
// The trade is deliberate: a camera that sits inside a beam and looks along it sees no lift from that
// beam, because along such a path every probe is lit and there is no edge to see. That is the same
// bargain every screen-space god-ray effect makes, and the alternative — a veil over the whole frame —
// is what this gate exists to remove.
//
// V is the same ray-traced sun visibility the terrain NEE uses, sampled once per stratum and jittered
// by the pixel-fixed interleave hash walked by the golden-ratio frame index — the doctrine the fog's
// strata follow, so the temporal reconstruct resolves the probe positions into continuous shadow
// volumes instead of a per-frame re-roll it cannot accumulate. The stratum count is higher than the
// fog's four because the lit fraction needs enough samples to place the shadow boundary: at four, the
// gate quantises in quarters and the beam edges step.
//
// Callers integrate only the stretch in front of geometry — never a sky miss. The sky LUT already
// carries the atmosphere's own in-scatter, so airlight along a ray that escapes to the sky would count
// that scattering twice and read as an aureole around the sun rather than as a beam.
public static const int SHAFT_STRATA = 8;

// The fog's own HG energy scaling, so a footer and a beam lit by the same sun agree in level.
public static const float SHAFT_PHASE_ENERGY = 0.7;

// Underwater the term is shaped differently from the air one, for two reasons that only exist in water.
// Its forward lobe has to be CAPPED (`waterShaft.y`, volume.water-shaft-fill): a long grazing path to
// the surface plus the peak of the lobe is an order of magnitude over the water it sits in front of, and
// the surface you look up at goes white. And its gate keeps a floor (`WATER_SHAFT_STRUCTURE_FLOOR`),
// because the gate alone peaks in a band a few pixels wide when the occluder is a bank or a tree, which
// reads as no shafts at all rather than as shafts.
public static const float WATER_SHAFT_STRUCTURE_FLOOR = 0.45;

// Depth at which the medium's cumulative in-scatter weight reaches k/SHAFT_STRATA of the stretch's
// total: the inverse of exp(-sigma*s). Equal weight per stratum is what keeps a dense medium sampling
// the blocks the light actually crosses, and it degenerates to even spacing in the thin limit, which is
// exactly where the air medium sits.
public float shaftDepthAtWeight(float sigma, float reach, int k) {
    if (k <= 0) {
        return 0.0;
    }
    if (k >= SHAFT_STRATA) {
        return reach;
    }
    float total = 1.0 - exp(-sigma * reach);
    float weight = float(k) * total / float(SHAFT_STRATA);
    return -log(max(1.0 - weight, 1.0e-6)) / sigma;
}

// `sigma` is the medium's extinction (pushed air density, or the water body's grey extinction), `g` its
// phase anisotropy, and `underwater` selects the water's surface leg: the sun ray has to cross the
// water surface, so it picks up the rest of the body's Beer-Lambert and the wave focus at the exit
// point, which is what lights an underwater shaft in the first place.
public float3 shaftInScatter(float3 ro, float3 rd, float t, float2 pix,
                             float sigma, float g, bool underwater, out float shaftShare) {
    shaftShare = 0.0;
    float strength = worldPush.shaft.w;
    if (sigma <= 0.0 || strength <= 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    if (dot(celestialLight.illuminance, float3(1.0, 1.0, 1.0)) <= 0.0) {
        return float3(0.0, 0.0, 0.0); // no body above the horizon: nothing to scatter
    }
    float reach = min(t, max(worldPush.shaft.z, 1.0));
    // Air beams are still swallowed by fog. Water's own extinction is already inside sigma, and the air
    // fog does not exist below the surface.
    float fogT = underwater ? 1.0 : fogTransmittance(worldPush.fog0.x, worldPush.fog0.z,
            worldPush.fog0.y, ro.y, rd.y, reach);
    float total = 1.0 - exp(-sigma * reach);
    float shaftWeight = total * fogT;
    if (shaftWeight <= 1.0e-4) {
        return float3(0.0, 0.0, 0.0); // a hit right in front of the eye: too little medium to read
    }
    float phase = hg(dot(rd, celestialLight.dir), g) * SHAFT_PHASE_ENERGY;
    if (underwater) {
        // The cap is the whole reason the underwater layer survives a look straight at the sun: with the
        // lobe at its peak and a long grazing path to the surface, the physical value runs an order of
        // magnitude over the water it sits in front of, and the surface turns white. A ceiling keeps the
        // water body's own level while the shadow structure still draws the beams.
        phase = min(phase, worldPush.waterShaft.y);
    }
    float noise = frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887);
    float weight = total / float(SHAFT_STRATA);
    bool waterWaves = (worldPush.flags & 16u) != 0u;
    float caustic = 1.0;
    bool causticKnown = false;
    float3 sun = float3(0.0, 0.0, 0.0);
    float lit = 0.0;
    for (int k = 0; k < SHAFT_STRATA; k++) {
        float a = shaftDepthAtWeight(sigma, reach, k);
        float b = shaftDepthAtWeight(sigma, reach, k + 1);
        float3 p = ro + rd * (a + (b - a) * noise);
        VisibilityResult probe = visibility(p, celestialLight.dir, 10000.0);
        float3 vis = probe.transmittance;
        if (underwater && probe.waterHitT > 0.0) {
            // The underwater stretch of the sun leg up to the surface, then the focus the wave field
            // gives the light as it crosses. One caustic sample serves the whole shaft: every probe sits
            // within a few blocks of the surface entry, where the pattern does not move enough to pay
            // the wave solve eight times over.
            vis *= exp(-sigma * probe.waterHitT);
            if (!causticKnown) {
                caustic = waterWaves
                        ? waterCaustic(p + celestialLight.dir * probe.waterHitT, celestialLight.dir,
                                probe.waterHitT)
                        : 1.0;
                causticKnown = true;
            }
            vis *= caustic;
        }
        // Per-channel, so a beam that has crossed stained glass carries its tint like every other sun
        // path does; solids and cutout leaves terminate it. The gate reads the grey mean: what matters
        // there is whether the path is lit, not in what colour.
        lit += dot(vis, float3(0.3333333, 0.3333333, 0.3333333));
        sun += vis * (weight * cloudShadowTransmittance(p, celestialLight.dir));
    }
    lit *= 1.0 / float(SHAFT_STRATA);
    // Linear in the lit fraction, deliberately: a smoothstep here multiplies the estimator's own
    // quantisation (one probe is 1/SHAFT_STRATA of the ramp) by its slope, and at a forward-scattering
    // phase the term is bright enough that those steps read as contour arcs across a lit wall. Linear
    // keeps every step at 1/SHAFT_STRATA of the term, which the reconstruct averages away.
    float structure = clamp(1.0 - lit, 0.0, 1.0);
    if (underwater) {
        // Water does not use the gate the way air does. The gate's product is largest exactly where a
        // path is half lit, which for a big occluder — a bank, a tree, a hull — is a band a few pixels
        // wide; open water and deep shade both fall to zero, and the view reads as "no shafts at all".
        // What reads underwater is the SHADOW the occluder casts into the water, so the term keeps its
        // visibility modulation and only leans on the gate for contrast: fully-lit water keeps 45%,
        // shadow-crossing water the lot. The forward peak is capped separately (see `phase`), which is
        // what keeps this from becoming a white sheet.
        structure = lerp(WATER_SHAFT_STRUCTURE_FLOOR, 1.0, structure);
    }
    shaftShare = shaftWeight * structure;
    return sun * (phase * celestialLight.illuminance * (strength * fogT * structure));
}
