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
public float fogMarchDensity(WorldPush push, float3 pos) {
    float base = fogDensityAt(push, pos.y);
    float amp = push.fog1.z;
    if (base <= 0.0 || amp <= 0.0) {
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

// ---- Cloud layer (Phase 4: the planar sheet and its shared shadow field) ----------------------------
//
// One procedural coverage field serves the three consumers the plan gives the weather map at this
// tier: the sky.rmiss planar sheet, the shadow every celestial NEE probe below the layer picks up
// (the cloud shadows on the ground), and the fog/water sun legs (the crepuscular dimming). The
// field is world-pinned through the EXACT cloud anchor (cloud2.xy — not the wave domain's
// mod-4096 one: the streak lattice has no period, so the anchor must not wrap either) and rides
// the Java-accumulated per-axis drift (cloud2.zw). The 2D field is the honest resolution of the
// PLANAR tier — the slab has no vertical structure yet; Phase 5's volumetric tier replaces the
// shadow estimate with a real march through the slab and takes the baked weather map with it.
//
// The shadow estimate is the honest resolution of the PLANAR tier: a flat slab's shadow along the sun
// is the coverage sampled once where the sun ray crosses the plane, with a slant-thickened optical
// depth. No baked weather/cloud-shadow map yet — the per-consumer cost is a few noise taps, cheaper
// than any texture pass at this resolution, and Phase 5's volumetric tier is what actually wants the
// baked map. cloudColor.rgb stays reserved: the planar sheet shades inside sky.rmiss's pre-conversion
// colour space (its albedo is a constant here), while the Phase 5 tier shades ACEScg like the fog.

public static const float CLOUD_CELL_ALONG = 96.0;      // blocks per base cell along the wind
public static const float CLOUD_CELL_CROSS = 16.0;      // ... across it, at the narrowest
public static const float CLOUD_CELL_CROSS_MAX = 40.0;  // ... and at the broadest (macro-modulated)
public static const float CLOUD_DIR_JITTER = 0.17;      // rad the local wind direction wanders (~10 deg)
public static const float CLOUD_DIR_CELL = 900.0;       // blocks that wander varies over
public static const float CLOUD_EXTINCTION = 0.22;      // 1/block at unit density (water-cloud order)
public static const float CLOUD_G = 0.4;                // phase anisotropy, the fog tier's forward doctrine
public static const float3 CLOUD_ALBEDO = float3(0.86, 0.87, 0.90); // BT.709-linear, sky.rmiss space

// Dave Hoskins' hash12, the 2D twin of fogHash13 (and of sky.rmiss's star hash). No lattice wrap:
// the streak lattice has no period to wrap into — real stratus streaks run along an arbitrary wind
// azimuth, which no per-axis wrap survives — so the field leans on the exact cloud anchor and the
// double-precision drift accumulators to keep the sample coordinates small instead.
float cloudHash12(float2 p) {
    float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float cloudValueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(cloudHash12(i), cloudHash12(i + float2(1.0, 0.0)), f.x),
                lerp(cloudHash12(i + float2(0.0, 1.0)), cloudHash12(i + float2(1.0, 1.0)), f.x),
                f.y);
}

// Unit-gain value-noise FBM. Three octaves is deliberate for the streak field: a fourth put the
// finest cells at ~2 blocks across, which is sub-pixel at cloud altitude — it read as a dense
// speckle instead of cloud, and the whole dome turned into one woven mat. The macro field below
// keeps four, because there the octaves land at hundreds of blocks.
float cloudFbm(float2 p, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int o = 0; o < octaves; o++) {
        sum += amp * cloudValueNoise(p);
        norm += amp;
        p = p * 2.0 + float2(31.0, 17.0);
        amp *= 0.5;
    }
    return sum / norm;
}

// Self-shadow of the slab with the multi-scatter octave stack (see the call site for the reasoning).
// The weights (1, 1/2, 1/4) over extinctions (tau, tau/2, tau/4) are normalised by their sum, so the
// result is 1 at tau = 0 and falls off far more slowly than exp(-tau): that slow tail IS the missing
// bounces, and it is what keeps a thick deck white by day and moonlit at night.
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

// ---- The coverage field, split so the slab walk can advance it incrementally ------------------------
//
// The field is a pure function of the horizontal position, so along a straight view ray it is a function
// of a single parameter. What does NOT vary along such a ray is everything derived from the MACRO octave
// - the streak breadth it breathes into, the lattice cells, the localized-coverage gate - because one
// macro cell is 2048x1024 blocks while a slab crossing is a couple of hundred. The field is therefore
// built in two halves: a frame of macro-derived invariants, and the per-sample octaves on top.
//
// Splitting it is not only a speed matter: every consumer still calls the same two functions, so the deck
// a player sees keeps casting the shadow they see. Freezing a frame at the midpoint of a crossing freezes
// only quantities that vary by a few percent of their own cell across it.
public struct CloudFieldFrame {
    float2 dir;        // the wind frame's along-drift axis (unit)
    float2 perp;       // ... and its cross-drift axis
    float  cellAlong;  // blocks per base cell along the drift
    float  cellCross;  // ... across it (breathing with the macro field)
    float  cumulus;    // the type dial, saturated: drives the cells and the rim feathering's fade-out
    float  broaden;    // the cumulus planform's dilation of the lattice (1 for the planar tier)
    float  threshold;  // the remap's threshold for this ray's redistributed coverage
    float  width;      // ... and its transition width
    float  gateMix;    // how strongly the macro gate applies (the patchiness setting)
    float  cLocal;     // the gate-modulated coverage: below ~0.02 the sky reads as clear
    float2 evolve;     // the field's evolution phase, per layer a different multiple of it (see below)
};

// The frame at one lattice position `p` (rebased XZ + exact anchor + drift, before the broaden scaling).
public CloudFieldFrame cloudFieldFrameAt(WorldPush push, float2 p) {
    CloudFieldFrame f;
    float c = push.cloud0.x;
    f.cumulus = saturate(push.cloud1.x);
    // The volumetric deck is a different animal from the sheet: a cumulus is an order of magnitude
    // broader than the sheet's own streak cells, and its planform is a field of cloud-sized patches
    // rather than a mat of lanes. Sampling the same lattice coarser turns the lanes into those patches,
    // and doing it HERE rather than at the slab's call site is what keeps the slab, the NEE probes and
    // the fog sun legs reading ONE field. Zero while the tier is off, so a planar frame samples exactly
    // the lattice it always did.
    f.broaden = (push.flags & WORLD_FLAG_CLOUD_VOLUMETRIC) != 0u
            ? lerp(1.0, CLOUD_BROADEN, f.cumulus) : 1.0;
    // The local wind direction wanders a few degrees over ~900-block regions - real cloud streets are
    // not parallel across a whole sky. It belongs to the frame because the lattice cells derived from
    // the macro field are sampled in the same rotated frame.
    float ang = push.cloud1.y
            + (cloudValueNoise(p / CLOUD_DIR_CELL + float2(3.0, 7.0)) - 0.5)
                    * (2.0 * CLOUD_DIR_JITTER);
    f.dir = float2(cos(ang), sin(ang));
    f.perp = float2(-f.dir.y, f.dir.x);
    float2 s = p * f.broaden;
    // Evolution FIRST: every layer below offsets its noise taps by this phase, the macro field at a
    // fraction of the rate the fine octaves use, so the layers crawl against EACH OTHER - the streaks
    // deform inside their envelope, the macro patches let clouds appear and dissolve, the interior
    // texture churns. The rates run slow-to-fast from large scale to small, which is how real cloud
    // fields behave. The phase is a smooth monotonic function of time - never a per-frame re-roll - so
    // RR accumulates it the way it accumulates the wind drift. It is assigned before anything reads it:
    // the macro field directly below is the first consumer, and an uninitialised read here would feed
    // the gate garbage.
    f.evolve = float2(push.cloudColor.x, push.cloudColor.x * 0.618034);
    // The macro field first: it gates the patches AND sets how broad the streaks run here. Four octaves
    // over 2 km cells, so one sample spans a 16x size range of cloud systems - a coverage change grows
    // and shrinks patches at every scale at once instead of one uniform size.
    // 0.06x: the macro layout must evolve an order of magnitude SLOWER than the wind advects it. At the
    // wind's own rate the patch layout sweeps past faster than the clouds move, and the sky cycles
    // clear-to-overcast every couple of minutes.
    float macro = cloudFbm(
            float2(dot(s, f.dir) / 2048.0, dot(s, f.perp) / 1024.0) + f.evolve * 0.06, 4);
    float crossEff = lerp(CLOUD_CELL_CROSS, CLOUD_CELL_CROSS_MAX, macro);
    // Stratus is a mat of lanes; cumulus is a field of heaps. Those are different planforms, not the
    // same one at two sizes: the lanes have to close up into round-ish patches whose size is the cloud's
    // own size - which is what the slab's thickness means. So the type dial interpolates both lattice
    // cells toward one isotropic, thickness-proportional cumulus cell. At 0 the cells are exactly the
    // stratus ones, so a planar frame is unaffected by any of this.
    float cumulusCell = max(push.cloud0.z * CLOUD_LOBE_RATIO, CLOUD_LOBE_MIN);
    f.cellAlong = lerp(CLOUD_CELL_ALONG, cumulusCell, f.cumulus);
    f.cellCross = lerp(crossEff, cumulusCell, f.cumulus);
    // Localised coverage: the macro field REDISTRIBUTES the coverage into clustered systems with clear
    // sky between them, as a zero-mean modulation of the LOCAL coverage around CLOUD_GATE_MEAN - patches
    // end up denser, gaps emptier, and the sky's total roughly holds. The old multiplicative gate could
    // not do that: it scaled the result by a mean of ~0.44, so the slider's promise was capped there and
    // a dip in the macro field cleared the sky to a few threads.
    f.gateMix = clamp(push.cloud1.z, 0.0, 1.0);
    // The swing is ONE sig of the gate either way: twice this gain sent whole kilometre-scale regions
    // to zero coverage, and the sky went bimodal - fully decked, then empty, as the camera's
    // neighbourhood moved between macro patches.
    f.cLocal = saturate(c * (1.0 + f.gateMix * (smoothstep(0.28, 0.78, macro)
            - CLOUD_GATE_MEAN)));
    // The remap: the threshold falls as c^0.7 so the response stays near-linear over the slider, and the
    // transition width grows with it (both calibrated, see CLOUD_REMAP_* above).
    f.threshold = 1.0 - CLOUD_REMAP_K * pow(f.cLocal, CLOUD_REMAP_GAMMA);
    f.width = max(CLOUD_REMAP_W0 + CLOUD_REMAP_W1 * f.cLocal, 1.0e-3);
    return f;
}

// Coverage 0..1 at a lattice position, from a frame. This is the whole per-sample field: a domain warp
// that bends the streaks over hundreds of blocks, the streak FBM, a mid-scale octave that breaks the
// interior texture, and a cross-wind octave that feathers the rims.
public float cloudCoverageAt(CloudFieldFrame f, float2 p) {
    float2 s = p * f.broaden;
    float2 q = float2(dot(s, f.dir) / f.cellAlong, dot(s, f.perp) / f.cellCross);
    q += (float2(cloudValueNoise(q * 0.35 + 17.0 + f.evolve),
                 cloudValueNoise(q * 0.35 + 91.0 + f.evolve)) - 0.5) * 1.1;
    // The cross-wind feathering octave is a stratus device - it feathers the long rims of lanes. A
    // cumulus heap has no long rims to feather, so it fades out with the type dial.
    float n = cloudFbm(q, 3) - 0.07 * cloudValueNoise(q * 0.25 + 7.0 + f.evolve * 1.7)
            - 0.12 * (1.0 - f.cumulus) * cloudValueNoise(q * float2(1.0, 6.0) + 31.0 + f.evolve * 2.3);
    return saturate((n - f.threshold) / f.width);
}

// The field at one world position's projection, for the consumers that cross the slab in a single step:
// the planar sheet, the shadow field, the NEE probes and the fog sun legs. Takes the push block
// explicitly because the sky miss shader holds its WorldPush in a local (module globals belong to the
// raygen entry points), while the rgen-side consumers read the global through the wrapper below.
float cloudCoverageField(WorldPush push, float2 rebasedXZ) {
    if (push.cloud0.x <= 0.0) {
        return 0.0;
    }
    float2 p = rebasedXZ + push.cloud2.xy + push.cloud2.zw;
    return cloudCoverageAt(cloudFieldFrameAt(push, p), p);
}

/** Global-reading wrapper for the raygen-side consumers (NEE probes, fog sun legs, debug views). */
public float cloudCoverageWorld(float2 rebasedXZ) {
    return cloudCoverageField(worldPush, rebasedXZ);
}

// Transmittance of the cloud layer along a sun leg from `pos` (rebased): the coverage sampled where
// the leg crosses the pushed plane, with the optical depth thickened by the leg's slant. This is the
// one estimate the ground-level NEE, the fog sun legs and the sheet's own self-shadow all share, so
// a clouded sun dims identically in all of them. Returns 1 while clouds are off — the callers'
// multiply keeps them bit-identical.
public float cloudShadowTransmittance(float3 pos, float3 sunDir) {
    if ((worldPush.flags & WORLD_FLAG_CLOUD) == 0u || worldPush.cloud0.x <= 0.0) {
        return 1.0;
    }
    if (sunDir.y < 0.05) {
        return 1.0; // the body sits at or below the horizon; there is no light left to shadow
    }
    float tPlane = (worldPush.cloud0.y - pos.y) / sunDir.y;
    if (tPlane <= 0.0) {
        return 1.0; // the sample is at or above the layer along the sun leg
    }
    float cov = cloudCoverageWorld(pos.xz + sunDir.xz * tPlane);
    if (cov <= 0.0) {
        return 1.0;
    }
    float tau = cov * worldPush.cloud0.w * CLOUD_EXTINCTION * worldPush.cloud0.z / sunDir.y;
    return exp(-tau);
}

// The planar sheet as sky.rmiss renders it: single scatter of the dominant celestial where the view
// ray crosses the pushed cloud plane, plus the sky-dome ambient, with the sheet's transmittance out
// for the caller to attenuate the sky behind it (atmosphere, stars, discs) with. The slab has no
// vertical structure yet — the tau here is the same slant-thickened estimate the shadow field uses,
// which is exactly what makes the sheet and its shadows agree. Camera inside the slab renders
// nothing (the planar tier has no sides to show); far crossings fade into the haze they sit behind.
public float3 planarCloudSkyInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                      out float cloudT) {
    cloudT = 1.0;
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    bool below = ro.y < push.cloud0.y;
    bool above = ro.y > push.cloud0.y;
    if (below == above) {
        return float3(0.0, 0.0, 0.0);
    }
    if (below ? rd.y < 1.0e-3 : rd.y > -1.0e-3) {
        return float3(0.0, 0.0, 0.0); // the ray runs away from the plane
    }
    float t = (push.cloud0.y - ro.y) / rd.y;
    // Dissipation distance (cloud1.w): the sheet fades over the last 40% of the pushed range and
    // is not rendered at all past it — clouds beyond the fade cost nothing. Deliberately a sheet-
    // only fade: the shadow field stays unfaded, because the layer still exists overhead and a
    // shadow dissolving with distance would smear across the ground as a gradient.
    float fadeDist = max(push.cloud1.w, 64.0);
    float fade = 1.0 - smoothstep(fadeDist * 0.6, fadeDist, t);
    if (fade <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float cov = cloudCoverageField(push, ro.xz + rd.xz * t);
    if (cov <= 0.0) {
        return float3(0.0, 0.0, 0.0); // cloudT stays 1: clear sky, nothing to attenuate with
    }
    // Perspective thickening: a grazing sight line crosses proportionally more slab.
    float tau = cov * push.cloud0.w * CLOUD_EXTINCTION * push.cloud0.z / max(abs(rd.y), 0.04);
    // The fade dissolves the slab's OPACITY, not just its glow: radiance and occlusion scale by
    // the same factor (cloudT = 1 - opacity). Fading the radiance alone painted the dissipation
    // band black — a grazing sight line saturates the transmittance to 0 long before the fade
    // ends, so cloudL + 0 x sky approached zero and the far edge read as a black silhouette band.
    float opacity = (1.0 - exp(-tau)) * fade;
    cloudT = 1.0 - opacity;
    // The dominant celestial, straight from the sky state (the miss shader has no derived
    // celestialLight): the self-shadow IS this same optical depth at the light's own elevation.
    // The body must actually be ABOVE the horizon to light the sheet. The pushed illuminance is a
    // look constant that does not fall to zero when its body sets (the drawn disc only goes dark
    // because extinction does), so an ungated sun term kept its full 128 klx all night and lit the
    // clouds like noon. With neither body up the term is exactly zero and only the sky ambient
    // (dark at night) remains.
    bool sunUp = state.sunIlluminance > 0.0 && state.sunDir.y > 0.0;
    bool moonUp = state.moonIlluminance > 0.0 && state.moonDir.y > 0.0;
    float3 lightDir = sunUp ? state.sunDir : state.moonDir;
    float lightIll = sunUp ? state.sunIlluminance : (moonUp ? state.moonIlluminance : 0.0);
    float tauSelf = cov * push.cloud0.w * CLOUD_EXTINCTION * push.cloud0.z / max(lightDir.y, 0.05);
    // Multiple-scatter approximation for the slab's self-shadow. A pure single-scatter sheet darkens a
    // thick deck to ambient-only, which reads as blue-grey daytime clouds and as black cut-outs at night;
    // real cloud tops are bright because photons bounce inside them. Three octaves of halved extinction,
    // energy-normalised so an unshadowed sample still reaches exactly 1 — thin edges keep the exact
    // single-scatter value, thick cores keep a lit presence. Phase 5's multi-scatter LUT replaces it.
    float selfT = cloudSelfShadow(tauSelf);
    float phase = hg(dot(rd, lightDir), CLOUD_G) * 0.7;
    // The ambient weight is clamped: it is a config lane, and the night this shipped a lane-order
    // slip fed a growing drift accumulator into it and the sky glowed. A bounded read makes that
    // class of accident merely wrong instead of blinding.
    float3 ambient = skyDome(skyViewLut, float3(0.0, 1.0, 0.0), state)
            * clamp(push.cloudColor.w, 0.0, 2.0);
    return CLOUD_ALBEDO * opacity * (phase * lightIll * selfT + ambient);
}

// ---- Volumetric cloud slab (Phase 5) ----------------------------------------------------------------
//
// The slab spans [cloudBase, cloudBase + thickness] — the SAME geometric reading cloud0.z already has in
// the planar tier, where tau = cov * density * EXTINCTION * thickness / dir.y is exactly the optical
// depth of a slab that thick crossed at the ray's slant. So this tier is not a second parameterisation
// of the clouds: it is the same layer, walked instead of estimated, and both tiers answer the same
// question at the same scale (which is why the step count can be turned down and the two must not
// disagree about how much light gets through).
//
// Density is the SAME field the ground NEE probes and the fog sun legs sample (cloudCoverageField): one
// field, three consumers, so the deck a player sees casts the shadow they see. What the walk adds is the
// vertical profile and the per-sample light gradient — the two things a single plane crossing cannot
// express, and the reason the extra cost buys anything at all. The slab's actual SHAPE, though, has to
// come from a 3D field sampled in the same wind frame: see cloudSlabDensity for why a density built out
// of a 2D field times a height profile is a layer cake no matter how finely it is walked.

public static const float CLOUD_STEP_MIN = 8.0;        // steps below this alias visibly at the slab's rims
// How far along a ray the per-ray field frame may be sampled. UNBOUNDED this fans the sky into wedges
// centred on the camera: a horizon ray crosses the slab kilometres away while an overhead ray crosses it
// a few hundred blocks up, so each direction reads the wind-direction jitter and the macro gate from a
// different patch, and the streak orientation snaps at the jitter cells' boundaries. Clamping keeps every
// ray sampling one neighbourhood, which turns the angular snapping into a smooth gradient.
public static const float CLOUD_FRAME_DIST = 1536.0;   // blocks; about 0.75 of a macro cell
// The macro gate's measured mean over the sky (tools/fit_base_remap.py). The gate modulates the LOCAL
// coverage around this centre rather than scaling the result, which keeps the redistribution zero-mean:
// patches get denser, gaps emptier, and the slider's total survives.
public static const float CLOUD_GATE_MEAN = 0.44;
// The coverage remap, calibrated against the live field by the same tool: at patchiness 0 the covered
// fraction of the sky tracks the coverage slider across its whole range (mean |error| 0.14). The old
// 1.15 / 0.10+0.20 pair was tuned for the pre-cumulus field, whose different n distribution left the
// slider delivering a fifth of what it promised.
public static const float CLOUD_REMAP_K = 1.15;
public static const float CLOUD_REMAP_GAMMA = 0.7;
public static const float CLOUD_REMAP_W0 = 0.08;
public static const float CLOUD_REMAP_W1 = 0.32;
// The lobe cell scales WITH the slab: a cloud's own size is what the slab's thickness means, so a
// 48-block deck is a field of altocumulus elements and a 200-block one is a field of cumulus towers,
// from the same field and the same constants. A fixed cell cannot do this — sized for one thickness it
// is either too coarse to vary vertically in a thin deck (back to a layer cake) or too fine to hold a
// tower in a thick one.
public static const float CLOUD_LOBE_RATIO = 1.1;      // lobe cell as a fraction of the slab's thickness
public static const float CLOUD_LOBE_MIN = 24.0;       // blocks; keeps a paper-thin slab from aliasing
public static const float CLOUD_SHAPE_CONTRAST = 1.8;  // how hard the 3D lobe field drives the boundary
public static const float CLOUD_EDGE_EROSION = 0.35;   // fine-octave rim erosion (mean-preserving)
public static const float CLOUD_BROADEN = 0.5;         // cumulus planform: sample the 2D field coarser
public static const float CLOUD_CUMULUS_NORM = 1.3245; // normalises the cumulus shoulder to mean 1

// 3D value-noise FBM over the fog's lattice. This is the field the volumetric tier's SHAPE comes from,
// and the reason is separability: the previous density was cov(xz) * profile(h) * detail(pos/96), where
// the 2D factor does not vary with height and the vertical factor does not vary horizontally. A product
// of a 2D and a 1D function is a stack of identical horizontal sheets whose brightness steps with
// height — which is exactly what it looked like: layer cake from the side, the same streaks extruded
// through the slab as maze walls. A density whose boundary is a 3D surface cannot do that.
public float cloudFbm3(float3 p, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int o = 0; o < octaves; o++) {
        sum += amp * fogValueNoise(p);
        norm += amp;
        p = p * 2.0 + float3(19.0, 7.0, 31.0);
        amp *= 0.5;
    }
    return sum / norm;
}

// Vertical profile of the slab, normalised to a mean of 1 for BOTH shapes so the type dial changes the
// silhouette the light falls on and never how much cloud is overhead. Stratus is an even board with
// softened rims; cumulus is the classic heap — a flat base, a shoulder, and a crown that rounds off.
public float cloudHeightProfile(float h, float typeMix) {
    float flat = 1.0 - 0.6 * pow(abs(2.0 * h - 1.0), 12.0);
    float heap = smoothstep(0.0, 0.09, h) * (1.0 - smoothstep(0.60, 1.0, h)) * CLOUD_CUMULUS_NORM;
    return lerp(flat, heap, saturate(typeMix));
}

// Density of the slab at one point, 0 outside it. The 2D coverage decides WHERE cloud may exist (and so
// also what it shadows — one field, three consumers, still), and a 3D lobe field decides what SHAPE the
// cloud takes inside that envelope: its boundary is a genuine 3D surface, so the mass breaks into
// billows and towers instead of extruded streaks. The lobe contrast and the fine-octave erosion are both
// mean-preserving on purpose — the column's optical depth stays the coverage field's, so the deck a
// player sees keeps casting the shadow they see and the planar tier's tau remains a fair reference.
public float cloudSlabDensity(WorldPush push, CloudFieldFrame f, float3 pos) {
    float thickness = max(push.cloud0.z, 1.0);
    float h = (pos.y - push.cloud0.y) / thickness;
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }
    float cov = cloudCoverageAt(f, pos.xz + push.cloud2.xy + push.cloud2.zw);
    if (cov <= 0.0) {
        return 0.0;
    }
    // The 3D coordinate lives in the wind frame: stretched along the drift, near-isotropic across it and
    // vertically. Sampled through the drift like every other cloud field, so the lobes never slide
    // against the streaks they are carved out of.
    float2 w = pos.xz + push.cloud2.zw;
    float cell = max(thickness * CLOUD_LOBE_RATIO, CLOUD_LOBE_MIN);
    float3 q = float3(dot(w, f.dir), pos.y, dot(w, f.perp)) / cell;
    q.xz += f.evolve * 0.25;                         // the billows churn inside the envelope
    float lobe = cloudFbm3(q, 2);                    // large billows, mean 0.5
    float detail = cloudFbm3(q * 2.6 + float3(47.0, 11.0, 29.0)
            + float3(f.evolve, 0.0) * 0.55, 1);      // rims, mean 0.5, scaled with q
    float contrast = 1.0 + CLOUD_SHAPE_CONTRAST * (lobe * 2.0 - 1.0);
    contrast *= 1.0 - CLOUD_EDGE_EROSION + 2.0 * CLOUD_EDGE_EROSION * detail;
    return saturate(cov * contrast) * cloudHeightProfile(h, f.cumulus) * push.cloud0.w;
}

// Volumetric cloud slab as sky.rmiss renders it: a jittered walk across the slab accumulating in-scatter
// and transmittance. `pix` seeds the pixel-fixed blue-noise offset that advances one golden-ratio stride
// per frame — the plan's hard constraint: a per-frame re-roll is a signal the denoiser cannot accumulate,
// while a rotating pixel-fixed pattern is one it averages. Both the fade and the multi-scatter doctrine
// are the planar tier's, so the tiers degrade into each other instead of jumping.
public float3 volumetricCloudInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                       float2 pix, out float cloudT) {
    cloudT = 1.0;
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0 || abs(rd.y) < 1.0e-6) {
        // A ray parallel to the slab never crosses its thickness; a zero-coverage layer has nothing to
        // cross. Either way cloudT stays 1 and the caller's compose is a no-op.
        return float3(0.0, 0.0, 0.0);
    }
    float base = push.cloud0.y;
    float thickness = max(push.cloud0.z, 1.0);
    float top = base + thickness;
    float ta = (base - ro.y) / rd.y;
    float tb = (top - ro.y) / rd.y;
    float t0 = max(min(ta, tb), 0.0);
    float t1 = max(ta, tb);
    if (t1 <= t0) {
        return float3(0.0, 0.0, 0.0); // the camera is above the slab looking away, or below looking down
    }
    // Dissipation distance (cloud1.w): same sheet-only fade the planar tier applies, and applied to the
    // slab's OPACITY rather than to its glow alone — the v17 lesson, which the emitted fade repeats.
    float fadeDist = max(push.cloud1.w, 64.0);
    float fade = 1.0 - smoothstep(fadeDist * 0.6, fadeDist, t0);
    if (fade <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    bool sunUp = state.sunIlluminance > 0.0 && state.sunDir.y > 0.0;
    bool moonUp = state.moonIlluminance > 0.0 && state.moonDir.y > 0.0;
    float3 lightDir = sunUp ? state.sunDir : state.moonDir;
    float lightIll = sunUp ? state.sunIlluminance : (moonUp ? state.moonIlluminance : 0.0);
    float cosTheta = dot(rd, lightDir);
    // Beer-Powder along the plan's formula: 2/(1 + exp(-2 cos)) is 1 side-on and swells toward 2 when
    // the view runs with the light, which is what makes a back-lit rim read as a lit rim.
    float phase = hg(cosTheta, CLOUD_G) * 0.7 * (2.0 / (1.0 + exp(-2.0 * cosTheta)));
    float3 ambient = skyDome(skyViewLut, float3(0.0, 1.0, 0.0), state)
            * clamp(push.cloudColor.w, 0.0, 2.0);

    float steps = clamp(push.volumeParams.w, CLOUD_STEP_MIN, 96.0);
    float dt = (t1 - t0) / steps;
    float jitter = frac(interleaveGradientNoise(pix)
            + float(push.frameIndex) * 0.6180339887);
    // One field frame for the whole crossing, built at its midpoint. Everything the frame carries is
    // derived from the macro octave, whose cells (2 km x 1 km) dwarf a slab crossing, so the sample loop
    // advances the field's per-sample octaves - the expensive part - with no per-step re-derivation of the
    // wind frame, the lattice cells or the localized-coverage gate.
    CloudFieldFrame frame = cloudFieldFrameAt(push,
            ro.xz + rd.xz * min(0.5 * (t0 + t1), CLOUD_FRAME_DIST)
            + push.cloud2.xy + push.cloud2.zw);
    // A coverage this close to zero leaves at most a hair of cloud anywhere in the crossing (the
    // threshold it implies is past the field's reach), so the walk is skipped rather than paying the
    // per-sample field to discover that 24 times.
    if (frame.cLocal <= 0.02) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 L = float3(0.0, 0.0, 0.0);
    float T = 1.0;
    for (int i = 0; i < int(steps); i++) {
        float3 p = ro + rd * (t0 + (float(i) + jitter) * dt);
        float density = cloudSlabDensity(push, frame, p);
        float tauCell = density * CLOUD_EXTINCTION * dt;
        if (tauCell > 1.0e-4) {
            float cellT = exp(-tauCell);
            // The local-density self-shadow: the sample's own density times the geometric path left to
            // the slab's top along the light ray. It costs no extra field tap, and it is what produces
            // the vertical gradient — dark underbelly, lit crown — that is the whole reason to walk the
            // slab instead of crossing it once. The octave stack stays the multi-scatter stand-in on top,
            // and the ambient leg is occluded by the same column at a softened weight.
            float depthToTop = max(top - p.y, 0.0);
            float tauUp = density * CLOUD_EXTINCTION * depthToTop;
            float tauSun = tauUp / max(lightDir.y, 0.05);
            float3 sunT = cloudSelfShadow(tauSun);
            // The ambient leg is occluded far more weakly than the sun leg, and deliberately so: what a
            // viewer mostly sees of a deck is its BASE, where the direct sun is gone and the brightness
            // comes from light that has bounced many times inside the cloud. Occluding the sky by the
            // same column the sun sees crushed the base to a dark sheet and the whole deck read as thin
            // — the opposite of the dense mass a cumulus should look like from underneath.
            float ambT = cloudSelfShadow(tauUp * 0.25);
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
// indirect bounce) must be dimmed by the deck it passed through without paying a walk per bounce — the
// plan's "Pass B does extinction only". One coverage tap at the slab's mid-crossing and the same
// slant-thickened tau the planar tier uses keeps a bounce's attenuation the MEAN of what the primary ray
// walked. The walk's own vertical detail is deliberately absent here: it is a per-sample property, and
// the cheap stand-in for it is the mean.
public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0 || abs(rd.y) < 1.0e-6) {
        return 1.0;
    }
    float base = push.cloud0.y;
    float top = base + max(push.cloud0.z, 1.0);
    if (rd.y > 0.0 ? ro.y >= top : ro.y <= base) {
        return 1.0; // the ray runs away from the slab
    }
    float ta = (base - ro.y) / rd.y;
    float tb = (top - ro.y) / rd.y;
    float tMid = 0.5 * (max(min(ta, tb), 0.0) + max(ta, tb));
    float cov = cloudCoverageField(push, ro.xz + rd.xz * tMid);
    if (cov <= 0.0) {
        return 1.0;
    }
    float tau = cov * push.cloud0.w * CLOUD_EXTINCTION * push.cloud0.z / max(abs(rd.y), 0.04);
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
// jitter, FBM-modulated density, and one geometric shadow ray per sample toward the dominant celestial
// body. Where the analytic tier culls the fog from shadow with a per-segment probe mean, here every
// sample carries its own visibility: it gates that sample's sun term AND thins its density (down to
// fogShadowFloor()), so shafts and shadow volumes come out of the same estimator instead of two.
//
// Convergence: with noise amplitude 0 and the shadow switch off every sample sees the plain analytic
// profile, and the midpoint-quadrature integral converges to the closed form as the step count rises —
// VolumeFogTest pins that numerically, 8/16/32/64 steps against fogOpticalDepth.
//
// Cost: steps density evals + up to steps shadow rays per segment per pixel. The early-outs matter:
// samples past negligible transmittance stop the march, and shadow rays fire only where the medium is
// non-zero. volumeParams.z (max integration distance) stays reserved — truncating the integral would
// desynchronise the out fogT from the throughput attenuation, which must agree.

public float3 fogSegmentInScatterMarched(float3 ro, float3 rd, float t, float2 pix,
                                         bool shadowAllowed, out float fogT) {
    fogT = 1.0;
    float sigma0 = worldPush.fog0.x;
    float3 albedo = max(worldPush.fogColor.rgb, float3(0.0, 0.0, 0.0));
    if (sigma0 <= 0.0 || dot(albedo, float3(1.0, 1.0, 1.0)) == 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    int steps = int(clamp(worldPush.volumeParams.x, 1.0, 64.0));
    float jitterStrength = clamp(worldPush.volumeParams.y, 0.0, 1.0);
    bool shadow = shadowAllowed
            && (worldPush.flags & WORLD_FLAG_VOLUME_SHADOW) != 0u
            && celestialLight.halfAngle > 0.0;

    float dt = t / float(steps);
    // Pixel-fixed interleave gradient with the golden-ratio frame walk — the analytic tier's exact
    // doctrine: a per-frame re-roll is a signal the denoiser cannot accumulate, while this pattern
    // dithers step edges into a stable grain the reconstruct averages out.
    float u = 0.5 + (frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887) - 0.5) * jitterStrength;

    float phase = hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7;
    float3 ambient = worldPush.fogColor.w * skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0))
            * INV_PI;
    float3 sunDir = celestialLight.dir;

    float3 L = float3(0.0, 0.0, 0.0);
    float tau = 0.0;
    for (int i = 0; i < steps; i++) {
        float tMid = (float(i) + u) * dt;
        float3 p = ro + rd * tMid;
        float sigma = fogMarchDensity(worldPush, p);
        float vis = 1.0;
        if (shadow && sigma > 0.0) {
            vis = luminance(visibility(p, sunDir, 10000.0).transmittance);
        }
        // The shadow-culled medium, per sample: shadowed fog keeps fogShadowFloor() of its density,
        // exactly the doctrine the analytic tier settled on. Transmittance and both lighting terms
        // ride the culled density, so a shadowed stretch reads as thin clear air.
        float sigmaEff = sigma * max(vis, fogShadowFloor());
        if (sigmaEff <= 0.0) {
            continue;
        }
        float tauMid = tau + sigmaEff * 0.5 * dt;
        float tMid_transmittance = exp(-tauMid);
        if (tMid_transmittance < 0.004) {
            break;
        }
        float3 sunTerm = float3(0.0, 0.0, 0.0);
        if (vis > 0.0) {
            // Toward-sun transmittance: the analytic profile at the sample's culled density — the
            // noise modulation of the sun leg is deliberately not marched (a second full march per
            // sample for a second-order softening of the shafts is not a trade this tier makes).
            // The cloud shadow rides on top: the same field the sheet renders and the ground NEE
            // reads, so a clouded sun thins the shafts instead of lighting them.
            float sunFogT = fogTransmittance(sigma0 * max(vis, fogShadowFloor()),
                    worldPush.fog0.z, worldPush.fog0.y, p.y, sunDir.y, 10000.0);
            sunTerm = phase * celestialLight.illuminance
                    * (vis * sunFogT * cloudShadowTransmittance(p, sunDir));
        }
        L += tMid_transmittance * (albedo * sigmaEff) * (sunTerm + ambient) * dt;
        tau += sigmaEff * dt;
    }
    fogT = exp(-tau);
    return L;
}

// ---- Water medium's own in-scatter (the underwater light shafts) -------------------------------------
//
// Average transmission of the water surface for downward sky irradiance (Fresnel reflection at
// near-normal incidence plus a grazing concession) — the constant factor the ambient term pays to
// enter the water. Deliberately a constant, not a per-sample Fresnel: the sky ambient is diffuse,
// so a single directional factor is the honest resolution available at this budget.
public static const float WATER_AMBIENT_SURFACE = 0.92;

// Single scatter over one path segment INSIDE the water stack medium. The water is a constant medium
// over the segment, so the identity integral the analytic fog leans on holds exactly:
//
//     L = (sigmaS/sigmaT) * (1 - T(t)) * light
//
// with T = exp(-sigmaT * t) in the GREY mean of the per-channel extinction — the throughput itself
// keeps attenuating per-channel (the caller's existing Beer-Lambert line, untouched), the in-scatter
// integral just refuses to become per-channel-vector radiance weights for a term that is second-order
// here. sigmaS/sigmaT IS the pushed turbidity by construction (waterScattering), so no re-derivation.
//
// The ambient light field DECAYS WITH DEPTH — the fix for the "white surface from below" blowout: sky
// irradiance reaches a sample only after crossing the water above it, so ambient(p) = A ·
// exp(-sigmaT · depth(p)). Fog never needed this (its column sits under open sky), but water saturates
// the single-scatter integral within a few blocks of opacity-scaled extinction, and without the decay
// every upward view degenerated to a depth-independent milk veil at ~turbidity × sky brightness. The
// surface is the pushed sea-level plane (water1.w) — exact for oceans; for elevated water the depth
// clamps at zero and the term degrades to the un-attenuated form rather than inventing light. The
// product T(s)·exp(-sigmaT·depth(s)) is still an exponential, so the integral stays closed form:
//
//     L_amb = sigmaS · A · exp(-sigmaT·d0) · (1 - exp(-sigmaT·t·(1-rd.y))) / (sigmaT·(1-rd.y))
//
// with d0 the segment start's depth below the plane; the rd.y → 1 (straight-up) limit is t, where the
// along-path and depth attenuations exactly cancel and the integral is bounded by the surface hit.
//
// Sun leg: one shadow-ray probe at the segment midpoint — the fog tier's per-path budget (the
// indirect pass gates the call to the walk's first water segment; the primary pass covers the
// camera's own underwater stretch).
// visibility() already crosses the waved surface as an any-hit (tinting the transmittance and
// reporting the exit distance as waterHitT); on top of that the leg picks up (a) the water's own
// Beer-Lambert over the underwater stretch of the sun ray and (b) the wave-refraction caustic at the
// exit point, the same term the underwater terrain NEE applies, so shafts and caustic light pools
// come from one wave field. The caustic clamp (CAUSTIC_MAX) bounds the term: every factor here is
// <= 1 except the clamped focus, so the integrator cannot produce fireflies the surface NEE didn't
// already have.
//
// With the shadow switch off there is no probe and hence no surface-exit distance; the sun term is
// skipped rather than estimated from a guessed depth — the ambient term alone keeps the water lit.
// Gating is turbidity > 0 && medium.water && extinction > 0 (the callers check the medium); zero
// turbidity never reaches here, preserving the volume block's zero-when-off invariant.
public float3 waterSegmentInScatter(float3 ro, float3 rd, float t, float2 pix, Medium medium) {
    float sigmaT = (medium.extinction.r + medium.extinction.g + medium.extinction.b) / 3.0;
    float sigmaS = (medium.scattering.r + medium.scattering.g + medium.scattering.b) / 3.0;
    if (sigmaT <= 0.0 || sigmaS <= 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 ambient = worldPush.fogColor.w * skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0))
            * INV_PI * WATER_AMBIENT_SURFACE;
    float scatterWeight = 1.0 - exp(-sigmaT * t);
    // Depth-attenuated ambient integral (see the derivation in the header): d0 is the segment start's
    // depth below the pushed plane, and the upCos limit handles the straight-up cancellation.
    float d0 = max(worldPush.water1.w - ro.y, 0.0);
    float upCos = 1.0 - rd.y;
    float ambIntegral;
    if (upCos < 1.0e-4) {
        ambIntegral = t * exp(-sigmaT * d0);
    } else {
        ambIntegral = exp(-sigmaT * d0) * (1.0 - exp(-sigmaT * t * upCos)) / (sigmaT * upCos);
    }
    float3 sunPart = float3(0.0, 0.0, 0.0);
    bool shadow = (worldPush.flags & WORLD_FLAG_VOLUME_SHADOW) != 0u
            && celestialLight.halfAngle > 0.0;
    if (shadow) {
        float3 p = ro + rd * (t * 0.5);
        VisibilityResult probe = visibility(p, celestialLight.dir, 10000.0);
        float vis = luminance(probe.transmittance);
        if (vis > 0.0) {
            // Underwater stretch of the sun ray: exp over the grey sigma up to the surface crossing.
            float legT = 1.0;
            if (probe.waterHitT > 0.0) {
                legT = exp(-sigmaT * probe.waterHitT);
                if ((worldPush.flags & 16u) != 0u) { // the water-waves toggle's flag bit
                    vis *= waterCaustic(p + celestialLight.dir * probe.waterHitT,
                            celestialLight.dir, probe.waterHitT);
                }
            }
            // The same 70%-energy HG scaling doctrine as the fog's sun term, plus the cloud shadow
            // the underwater sun leg crosses above the surface — the same field the fog tiers read,
            // so overcast water loses its shafts with everything else.
            float phase = hg(dot(rd, celestialLight.dir), medium.phaseG) * 0.7;
            sunPart = phase * celestialLight.illuminance
                    * (vis * legT * cloudShadowTransmittance(p, celestialLight.dir));
        }
    }
    return (sigmaS / sigmaT) * scatterWeight * sunPart + sigmaS * ambIntegral * ambient;
}

// ---- Segment-level lighting assembly ---------------------------------------------------------------
//
// The analytic fog tier's single-scatter over one path segment, with the fog CULLED FROM SHADOW:
// the medium only exists where the dominant celestial body reaches it. The segment's effective
// density is sigma0 scaled by the mean sun visibility of FOG_SUN_BANDS probes placed at equal
// in-scatter-weight depths on the unthinned field (bisection on the closed form; the probe grid is
// pixel-fixed, IGN jitter walking a golden-ratio sequence across frames, so the reconstruct resolves
// the culled shadow-volume edges into smooth gradients even with a static camera). A sealed room or
// the shadow side of a hill holds no fog at all — clear air, recovered transmittance — while lit fog
// keeps its full physical in-scatter, shafts included. The visibility probes ride the UNTHINNED
// field: what they measure is geometry's occlusion, not the culled medium's.
//
// Writes the segment's fog transmittance so the caller attenuates throughput with the same quantity
// the in-scatter used. `celestialLight` (world_core) must have been derived for this frame and
// converted to ACEScg, exactly as world_indirect's raygen does; callers in a pass that skips NEE set
// it only when they actually call this.
public static const int FOG_SUN_BANDS = 4;

// Density fraction fully shadowed fog keeps — a live slider (volume.fog-shadow-floor) riding the
// reserved fog1.w lane, no longer a compile-time constant. History: a hard cull to zero read as
// sterile (shadow volumes lost every depth cue and their edges snapped hard against the lit medium);
// the first floor (25%) kept too much milk in sealed rooms, so the default now holds only a whisper
// of aerial perspective in shadow. 0 reproduces the hard cull, 0.95 saturates toward no culling at
// all. The per-stratum sun in-scatter is NOT floored — direct sun through a shadowed stratum stays
// zero, so the floor feeds the ambient glow and the transmittance only.
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
    // reuses the same shadow switch, phase, ambient and culling doctrine. `allowShadowCull` keeps its
    // meaning — the sky-facing miss segment passes false and its march runs without shadow rays, the
    // same cost gate the analytic tier's probes honour.
    if ((worldPush.flags & WORLD_FLAG_VOLUME_MARCH) != 0u && worldPush.volumeParams.x >= 1.0) {
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

    // One shadow probe per equal-weight stratum, placed on the physical (unthinned) field with a
    // pixel-fixed, frame-walking jitter. Their mean is the segment's lit fraction — the culling
    // factor for the medium itself. Per-stratum visibilities are kept for the sun term below.
    float vis[FOG_SUN_BANDS];
    float stratumDepth[FOG_SUN_BANDS];
    float visBar = 1.0;
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
            visBar = 0.0;
            for (int k = 0; k < FOG_SUN_BANDS; k++) {
                float stratumStart = scatterWeightFull * (float(k) + noise) / float(FOG_SUN_BANDS);
                stratumDepth[k] = fogDepthAtWeight(sigma0, worldPush.fog0.z, worldPush.fog0.y,
                        ro.y, rd.y, t, stratumStart + scatterWeightFull * 0.5 / float(FOG_SUN_BANDS));
                vis[k] = luminance(visibility(ro + rd * stratumDepth[k], celestialLight.dir,
                        10000.0).transmittance);
                visBar += vis[k];
            }
            visBar /= float(FOG_SUN_BANDS);
        }
    }

    // The culled medium: fog thins with shadow down to the floor. Transmittance, the sun strata and
    // the ambient all ride this density, so a shadowed stretch reads as thin clear air rather than
    // a dark veil or a vacuum.
    float sigma = sigma0 * max(visBar, fogShadowFloor());
    fogT = fogTransmittance(sigma, worldPush.fog0.z, worldPush.fog0.y, ro.y, rd.y, t);
    float scatterWeight = 1.0 - fogT;
    // HG lobe scaled to 70% energy: a pure forward lobe blows out when the view aligns with the
    // sun down a bright shaft. No isotropic floor — a floor at side/back angles reads as a uniform
    // milk wash, and the strata probes already light the lit medium geometrically.
    float phase = hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7;
    float3 ambient = worldPush.fogColor.w * skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0))
            * INV_PI;
    float3 sunPart = float3(0.0, 0.0, 0.0);
    if (!shadowCull) {
        // Unshadowed sun term: the closed form with V = 1 everywhere, sun transmittance taken at
        // the segment start. The cloud shadow still applies — geometry shadows are off, the layer
        // above is not.
        sunPart = phase * celestialLight.illuminance * (scatterWeight
                * fogT_toSunRo(ro, celestialLight.dir, sigma)
                * cloudShadowTransmittance(ro, celestialLight.dir));
    } else if (scatterWeight > 1.0e-4) {
        // Per-stratum sun in-scatter: the stratum's share of the culled medium times its own
        // measured visibility times the closed-form fog transmittance from its probe toward the
        // sun. Strata whose probe landed in shadow contribute nothing — that IS the culling. The
        // cloud shadow multiplies the same field every other sun leg reads.
        float stratumWeight = scatterWeight / float(FOG_SUN_BANDS);
        for (int k = 0; k < FOG_SUN_BANDS; k++) {
            float3 probe = ro + rd * stratumDepth[k];
            float sunFogT = fogTransmittance(sigma, worldPush.fog0.z, worldPush.fog0.y,
                    probe.y, celestialLight.dir.y, 10000.0);
            sunPart += phase * celestialLight.illuminance * (stratumWeight * vis[k] * sunFogT
                    * cloudShadowTransmittance(probe, celestialLight.dir));
        }
    }
    return albedo * (sunPart + scatterWeight * ambient);
}
