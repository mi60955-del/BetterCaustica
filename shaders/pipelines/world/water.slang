// Water surface: the wave field, the normal perturbation it drives, the relief offset that sells its
// height, and the caustic term that differentiates the same field. Depends on core.

import world_common;
import world_core;

// The field is a band-limited sum of twenty-one sinusoids -- seven wavelength bands, three headings in
// each -- with each band's crests steered by one smooth noise field. Everything about it is analytic, and
// that is why it replaced a hashed value-noise lattice:
//
//  * A plane wave is band-limited by construction. A bilinearly interpolated lattice is only C0: the
//    height is continuous but its gradient snaps on every cell boundary, so the normal carries a straight
//    crease along the lattice grid. At grazing angles those creases fan out from the vanishing point as a
//    hatch of streaks, and the grid is visible directly at the horizon.
//  * The gradient falls out of the same expression that gives the height, so the visible normal, the
//    previous-frame gradient the reconstruction filter reprojects, and the caustic Jacobian all take the
//    exact derivative instead of finite-differencing the height three times.
//
// Four choices carry the look:
//
//  * Deep-water dispersion, omega = sqrt(g k), ties each wave's speed to its wavelength, so swell rolls
//    while chop ripples instead of the whole surface sliding rigidly. The caller's wave clock is seconds,
//    which is the unit that makes these periods come out like a real sea.
//  * Band headings are rotated from one another by the golden angle, and each band is split into three
//    headings spread a radian either side of its centre. One heading per band -- however well separated
//    the bands are -- leaves a two-direction weave that the eye reads instantly as a repeating pattern;
//    measured autocorrelation of the near-field tilt goes from 0.43 with one heading per band to 0.08
//    with three. Spreading the headings is what fixes the close view; the band count is what fixes the
//    horizon, and neither substitutes for the other.
//  * Amplitude goes as wavelength, which makes the slope amplitude equal in every band. That is the k^-4
//    equilibrium range of a wind-driven sea, and it is what lets all scales read at once instead of one
//    band dominating.
//  * Each wave carries its own phase offset, and the whole surface is phase-modulated by C1 value noise.
//    Neither a sum of pure tones nor a set of rotated tones is aperiodic on its own; the noise is what
//    stops any patch of sea resembling another.
//
// iterationT damps its normal perturbation toward zero as the surface turns edge-on to the camera. That
// guard is not portable: a shading ray's direction is not the eye's, so scaling the normal by it would give
// one surface a different normal per ray. The level-of-detail fade already discards what a grazing pixel
// cannot resolve, which is the ray-independent form of the same protection.

public static const uint  WAVE_BANDS = 7;
public static const uint  WAVE_PER_BAND = 3;
public static const float WAVE_G = 9.81;              // m/s^2; sets every omega below
public static const float WAVE_SLOPE = 0.055;         // gradient amplitude per band, dimensionless
public static const float WAVE_SPEED = 0.8;           // clock rate applied to the caller's wave seconds
public static const float WAVE_WARP = 4.0;            // radians of phase the noise may add
public static const float WAVE_WARP_CELLS = 3.0;      // noise cell, in wavelengths of its own band
public static const float WAVE_INV_SQRT_PER_BAND = 0.5773503;   // 1/sqrt(3): splits a band's slope budget
public static const float WAVE_RELIEF_CAP = 0.25;     // blocks of lateral travel, whatever the angle

// Wavelength of each band, in blocks. Every other number is derived from this so it cannot drift:
// k = 2pi/lambda, omega = sqrt(g k), height amplitude = slope / k.
//
// The spectrum has to reach far enough up to be seen at the horizon. The level-of-detail fade discards a
// band once its wavelength no longer fits in the pixel footprint, so wherever the sea ends, only the
// longest bands are still alive -- and one band is three waves sharing a single |k|, which is a grating
// with one spatial frequency and nothing to break it up. Measured on a strip of horizon sampled at six
// metres per pixel, five bands leave the far field a flat mirror at 1.6 degrees of tilt with a strongly
// self-similar pattern; seven bands put running swell all the way to the horizon at 3.1 degrees.
public static const float WAVE_BAND_LAMBDA[WAVE_BANDS] = {
    1.25000, 2.44000, 4.75000, 9.27000, 18.07000, 35.30000, 68.80000
};
// Centre heading of each band, rotated from the last by the golden angle.
public static const float2 WAVE_BAND_DIR[WAVE_BANDS] = {
    float2(0.825336, 0.564642), float2(-0.989987, 0.141156), float2(0.634636, -0.772811),
    float2(0.054066, 0.998537), float2(-0.714369, -0.699770), float2(0.999441, 0.033439),
    float2(-0.759544, 0.650455)
};
// Direction along which each band's phase warp is laid out. Never parallel to the band's headings, so
// bending a crest at its middle and at its end pull it in different directions.
public static const float2 WAVE_BAND_WARP_DIR[WAVE_BANDS] = {
    float2(-0.801144, 0.598472), float2(0.186476, -0.982459), float2(0.526140, 0.850398),
    float2(-0.962395, -0.271654), float2(0.893140, -0.449779), float2(-0.354752, 0.934960),
    float2(-0.369973, -0.929042)
};
// cos/sin of the three heading offsets within a band (-1, 0, +1 radian). Stored as a rotation so the
// inner loop costs two multiplies instead of a sincos.
public static const float2 WAVE_HEADING_ROT[WAVE_PER_BAND] = {
    float2(0.540302, -0.841471), float2(1.000000, 0.000000), float2(0.540302, 0.841471)
};
// Per-wave phase offset, band-major and then heading-minor. Without these the bands all start at phase
// zero and their crests lock into a lattice, which is the repeating weave this field has to avoid.
public static const float WAVE_PHASE[WAVE_BANDS * WAVE_PER_BAND] = {
    0.807830, 3.137055, 3.779326, 0.180258, 0.929447, 5.832122, 0.442466, 0.815394,
    5.958523, 3.907410, 2.318452, 3.213158, 4.164765, 1.729816, 0.866879, 4.951399,
    4.212000, 3.219393, 5.131706, 3.449942, 6.163262
};

// The player's wave controls: x height, y wavelength, z clock rate. All 1 reproduces the authored field.
// The wavelength keeps a floor because a zeroed control block would drive every wavenumber to infinity.
public float3 waterWaveControls() {
    float4 c = worldPush.waterWave;
    return float3(max(c.x, 0.0), max(c.y, 0.01), max(c.z, 0.0));
}

public float waterWaveLodWeight(float wavelength, float footprint) {
    float cyclesPerPixel = max(footprint, 0.0) / max(wavelength, 1.0e-4);
    return 1.0 - smoothstep(0.25, 0.5, cyclesPerPixel);
}

/**
 * How much of its Fresnel reflection a surface keeps. Only water is damped; every other dielectric
 * returns the Fresnel value untouched.
 *
 * A smooth dielectric mirror approaches F = 1 at grazing angles, so a flat water surface reproduces the
 * bright band of sky at the horizon almost exactly and the whole distant sea reads as white. Real water is
 * a field of steep, mutually shadowing facets: much of what would have mirrored the sky is instead turned
 * toward the water on the far side of a crest, and absorbed there. The wave normals give the first-order
 * version of that, but they cannot give the shadowing, so the residue is applied here.
 *
 * The removed energy is not discarded — callers multiply it into the transmission term, so a damped
 * surface goes darker by absorbing more, not by losing light.
 */
public float waterReflectance(float fresnel, bool isWater) {
    return isWater ? fresnel * saturate(worldPush.waterWave.w) : fresnel;
}

// One lattice cell of the phase warp's noise.
static float waveCell(int2 cell, uint channel) {
    uint n = uint(cell.x) * 1619u + uint(cell.y) * 31337u + channel * 6971u;
    n = (n << 13) ^ n;
    n = (n * (n * n * 15731u + 789221u) + 1376312589u) & 0x7FFFFFFFu;
    return float(n) * (2.0 / 2147483648.0) - 1.0;
}

/**
 * Quintic value noise at `coord` (one unit per lattice cell) and its gradient with respect to `coord`.
 *
 * The weights are the quintic 6f^5 - 15f^4 + 10f^3, whose derivative vanishes at both cell edges, so the
 * noise is C1 and its gradient is continuous across cells. That is the whole reason it can be used where
 * the old height lattice was not: it bends a crest without drawing a crease into the slope.
 */
static void waveNoise(float2 coord, uint channel, out float value, out float2 grad) {
    float2 i0 = floor(coord);
    float2 f = coord - i0;
    float2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float2 dw = 30.0 * f * f * (f - 1.0) * (f - 1.0);
    int2 c = int2(i0);
    float v00 = waveCell(c, channel);
    float v10 = waveCell(c + int2(1, 0), channel);
    float v01 = waveCell(c + int2(0, 1), channel);
    float v11 = waveCell(c + int2(1, 1), channel);
    value = (v00 + (v10 - v00) * w.x) * (1.0 - w.y) + (v01 + (v11 - v01) * w.x) * w.y;
    grad = float2(dw.x * ((v10 - v00) * (1.0 - w.y) + (v11 - v01) * w.y),
                  dw.y * ((v01 - v00) * (1.0 - w.x) + (v11 - v10) * w.x));
}

/**
 * Height and exact slope of the whole field at one world position, in one walk.
 *
 * `worldXZ` must already carry the pushed water anchor, and `surfaceY` is the face's own height, folded
 * into the horizontal domain so two bodies of water at different levels show different crests instead of
 * the same pattern stacked. `footprint` is the world size of the shading sample; bands that no longer fit
 * inside it are faded out.
 *
 * A wave in band b at heading offset j has
 *     psi = k (d . q) - omega t + phase + W N(q / (lambda C) + off)
 * so    grad psi = k d + (W / (lambda C)) grad N
 * and the warp's gradient is the only reason the second term is not zero: crests curve, and the slope
 * stays closed-form.
 */
static void waveField(float2 worldXZ, float surfaceY, float t, float footprint,
                      out float height, out float2 grad) {
    float3 controls = waterWaveControls();
    float clock = t * WAVE_SPEED * controls.z;
    float2 q = worldXZ - surfaceY;
    height = 0.0;
    grad = float2(0.0, 0.0);
    for (uint b = 0u; b < WAVE_BANDS; b++) {
        float lambda = WAVE_BAND_LAMBDA[b] * controls.y;
        float lod = waterWaveLodWeight(lambda, footprint);
        if (lod <= 0.0) continue;
        float k = 2.0 * PI / lambda;
        float omega = sqrt(WAVE_G * k);
        float cell = lambda * WAVE_WARP_CELLS;
        float wv;
        float2 wg;
        waveNoise(q / cell + WAVE_BAND_WARP_DIR[b] * (3.7 * float(b)), b + 1u, wv, wg);
        float warp = WAVE_WARP * wv;
        float2 gwarp = (WAVE_WARP / cell) * wg;
        float amp = WAVE_SLOPE / k * WAVE_INV_SQRT_PER_BAND * lod;
        for (uint j = 0u; j < WAVE_PER_BAND; j++) {
            float2 cs = WAVE_HEADING_ROT[j];
            float2 d = float2(cs.x * WAVE_BAND_DIR[b].x - cs.y * WAVE_BAND_DIR[b].y,
                              cs.y * WAVE_BAND_DIR[b].x + cs.x * WAVE_BAND_DIR[b].y);
            float psi = k * dot(q, d) - omega * clock
                      + WAVE_PHASE[b * WAVE_PER_BAND + j] + warp;
            height += amp * sin(psi);
            grad += amp * cos(psi) * (k * d + gwarp);
        }
    }
}

/** Field height in blocks above the flat surface. See waveField. */
public float waterWaveHeight(float2 worldXZ, float surfaceY, float t, float footprint) {
    float height;
    float2 grad;
    waveField(worldXZ, surfaceY, t, footprint, height, grad);
    return height;
}

/** Field slope, dimensionless. See waveField. Callers scale it by the height control. */
public float2 waterWaveTilt(float2 worldXZ, float surfaceY, float t, float footprint) {
    float height;
    float2 grad;
    waveField(worldXZ, surfaceY, t, footprint, height, grad);
    return grad;
}

/**
 * Lateral offset that stands in for the crest the ray would have to travel past.
 *
 * The travel is capped in blocks rather than left at height / cos(incidence). The unclosed form is the
 * correct parallax, but at grazing angles it divides toward infinity and drags the whole field sideways
 * along the view ray, which shears it into radial streaks. Capped, it stays a crest hint at every angle.
 */
public static float2 waterWaveRelief(float2 worldXZ, float surfaceY, float3 rayDir, float t,
                                     float footprint) {
    float h = waterWaveHeight(worldXZ, surfaceY, t, footprint);
    // Scaled with the height control so a flattened surface also stops travelling sideways: relief is a
    // consequence of the crests existing, not an independent knob.
    float travel = clamp(h * waterWaveControls().x / max(-rayDir.y, 0.25),
                         -WAVE_RELIEF_CAP, WAVE_RELIEF_CAP);
    return worldXZ + rayDir.xz * travel;
}

/**
 * Perturb a (near-)horizontal water surface normal by the field. Flowing side faces (|n.y| small) keep
 * their flat geometric normal: the field is defined over the horizontal plane, and a vertical sheet of
 * water reads as flat in the reference too.
 *
 * `worldPos` is a world-stable position (rebased hit XZ plus the pushed anchor, with the face's own Y),
 * and `rayDir` is the direction the shading ray arrived travelling, which the relief offset needs.
 * `nGeo` is already oriented toward the viewer, so the down-viewed underside (n.y < 0) takes the mirrored
 * slope.
 */
public float3 applyWaterWaves(float3 nGeo, float3 worldPos, float3 rayDir, float t, float footprint) {
    if (abs(nGeo.y) < 0.5) return nGeo;
    float3 controls = waterWaveControls();
    float2 domain = waterWaveRelief(worldPos.xz, worldPos.y, rayDir, t, footprint);
    float2 tilt = waterWaveTilt(domain, worldPos.y, t, footprint) * controls.x;
    float3 up = normalize(float3(-tilt.x, 1.0, -tilt.y));
    return nGeo.y >= 0.0 ? up : -up;
}

// Evaluate the current tilt and the previous frame's. Both are the exact gradient of the same expression
// and differ only in the clock, so the reprojected normal is the crest that was actually there.
public void waterWaveGradTemporal(float2 worldXZ, float surfaceY, float currentT, float previousT,
                                  float footprint, out float2 currentGrad, out float2 previousGrad) {
    currentGrad = waterWaveTilt(worldXZ, surfaceY, currentT, footprint) * waterWaveControls().x;
    previousGrad = waterWaveTilt(worldXZ, surfaceY, previousT, footprint) * waterWaveControls().x;
}

// ---- Water caustics. Sunlight refracting through the waved surface converges/diverges before it
// reaches an underwater receiver; the focusing factor is the inverse Jacobian determinant of the
// surface->floor mapping, so it comes out of the same field the visible ripples use and stays in sync
// with them.
//
// Where the shadow ray of an underwater NEE vertex crossed water (returned by visibility() as
// VisibilityResult.waterHitT), evaluate the horizontal landing position of the refracted sun ray as a
// function of surface position. The per-sample sun-quad jitter (sampleSquare) shifts the pattern per
// sample, so caustics physically blur with depth under DLSS-RR accumulation.
public static const float CAUSTIC_EPS = 0.25; // finite-difference step (m); also low-passes sub-step detail
public static const float CAUSTIC_MAX = 6.0;  // focusing clamp: keeps fold-line singularities from becoming fireflies

// Horizontal landing offset of a refracted sun ray entering the surface at world-stable xz, intersected
// with the receiver plane h metres below. inc = sunlight travel direction (downward, unit).
public float2 causticLanding(float2 xz, float surfaceY, float t, float3 inc, float h) {
    // footprint 0: the pattern is evaluated per receiver sample, not per screen pixel, so it must not be
    // faded by how large the pixel happens to be.
    float2 tilt = waterWaveTilt(xz, surfaceY, t, 0.0) * waterWaveControls().x;
    float3 nw = normalize(float3(-tilt.x, 1.0, -tilt.y));
    float3 rdw = refract(inc, nw, 1.0 / WATER_IOR); // air->water: never TIR
    return xz + rdw.xz * (h / max(-rdw.y, 0.05));
}

public float waterCaustic(float3 exitPos, float3 lightDir, float waterDist) {
    // Grazing light: the receiver-plane intersection degenerates and caustics vanish in reality anyway
    // (long slant paths absorb + the pattern smears out) -- fade the effect to neutral 1.0.
    float fade = smoothstep(0.06, 0.18, lightDir.y);
    if (fade <= 0.0) return 1.0;
    float h = max(waterDist * lightDir.y, 0.05); // vertical depth of the receiver below the exit point
    float t = worldPush.waterParams.w;
    float3 inc = -lightDir;
    float2 base = exitPos.xz + worldPush.waterAnchor.xy;  // world-stable wave domain (same as applyWaterWaves)
    // The exit point is on the surface itself, and height is not rebased, so it feeds the domain directly.
    float surfaceY = exitPos.y;
    float2 p0 = causticLanding(base, surfaceY, t, inc, h);
    float2 px = causticLanding(base + float2(CAUSTIC_EPS, 0.0), surfaceY, t, inc, h);
    float2 pz = causticLanding(base + float2(0.0, CAUSTIC_EPS), surfaceY, t, inc, h);
    float det = abs((px.x - p0.x) * (pz.y - p0.y) - (px.y - p0.y) * (pz.x - p0.x));
    // Flat water => all landings displace identically => det = eps^2 => focus = 1 (energy-neutral); the
    // clamp bounds the fold-line spikes whose energy the surrounding darkening already paid for.
    float focus = (CAUSTIC_EPS * CAUSTIC_EPS) / max(det, 1.0e-5);
    return lerp(1.0, min(focus, CAUSTIC_MAX), fade);
}
