// Physically based sky: Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique" (EGSR). Three LUTs, all produced by compute passes in this directory:
//
//   sky_transmittance.comp  256x64   static  transmittance from a point to space, by (altitude, cos zenith)
//   sky_multiscatter.comp    32x32   static  second-and-higher-order scattering, by (altitude, cos sun zenith)
//   sky_view.comp           192x216  per-frame  the sky dome itself, one 192x108 slice per celestial body
//
// This module holds everything the three passes and world.rmiss/world.rgen share, so the LUT and the
// shaded result can never be computed from different physics.
//
// The multiple-scattering LUT supplies twilight and anti-solar illumination directly, so the sky needs no
// authored fill or crossfade between lighting states.
//
// Everything is photometric: illuminance in lux goes in and luminance in cd/m² comes out. Colour is
// linear BT.709 inside this module and crosses to ACEScg exactly
// once, at the consumer's output.
//
// Lengths are KILOMETRES and coefficients 1/km throughout. The LUT parameterisations take square roots of
// differences of radii around 6360 vs 6460; in metres those differences lose most of their fp32
// significand and the horizon row of the sky-view LUT visibly quantises.
module sky;

import world_common;

public static const float SKY_PI = 3.14159265359;

// ---- Medium. Rayleigh/Mie/ozone coefficients are Hillaire's Earth reference values.
public static const float ATMOS_BOTTOM_KM = 6360.0;
public static const float ATMOS_TOP_KM    = 6460.0;
public static const float3 RAYLEIGH_SCATTERING = float3(0.005802, 0.013558, 0.033100);
public static const float  RAYLEIGH_SCALE_H_KM = 8.0;
public static const float  MIE_SCATTERING      = 0.003996;
public static const float  MIE_EXTINCTION      = 0.004440;   // scattering + absorption
public static const float  MIE_SCALE_H_KM      = 1.2;
public static const float  MIE_G               = 0.8;
// Ozone absorbs without scattering, in a tent peaking at 25 km. Its green-heavy absorption is what keeps
// a noon zenith deep blue and turns twilight through purple rather than straight from orange to grey.
public static const float3 OZONE_ABSORPTION = float3(0.000650, 0.001881, 0.000085);
public static const float  OZONE_CENTER_KM  = 25.0;
public static const float  OZONE_WIDTH_KM   = 15.0;

// ---- LUT dimensions. Mirrored in RtSkyLut.java; the two must agree exactly (each pass hard-codes its
// own extent for the bounds check, and the parameterisations below apply the half-texel correction).
public static const int TRANSMITTANCE_LUT_W = 256;
public static const int TRANSMITTANCE_LUT_H = 64;
public static const int MULTISCATTER_LUT_W  = 32;
public static const int MULTISCATTER_LUT_H  = 32;
public static const int SKY_VIEW_LUT_W      = 192;
public static const int SKY_VIEW_LUT_H      = 108;
// The sky-view image stacks one slice per celestial body: rows [0,108) are the sun's contribution and
// rows [108,216) the moon's. Two independent slices let moonlight and sunlight scatter through the same
// atmosphere without crossfading between the lighting solutions.
public static const int SKY_VIEW_BODY_COUNT = 2;
public static const int SKY_VIEW_LUT_TOTAL_H = SKY_VIEW_LUT_H * SKY_VIEW_BODY_COUNT;
public static const int SKY_VIEW_BODY_SUN  = 0;
public static const int SKY_VIEW_BODY_MOON = 1;

// Raymarch step counts. The sky-view LUT is amortised over the whole frame, so it can afford a count that
// leaves no visible banding at the horizon, which is where the parameterisation concentrates its texels.
public static const int TRANSMITTANCE_STEPS = 40;
public static const int SKY_VIEW_STEPS      = 32;
public static const int MULTISCATTER_STEPS  = 20;
public static const int MULTISCATTER_DIRS   = 64; // 8x8 sphere directions per texel

public struct MediumSample {
    public float3 scattering;  // Rayleigh + Mie (the part that redirects light)
    public float3 extinction;  // scattering + absorption (Mie absorption + ozone)
    public float3 rayleigh;
    public float  mie;
};

public MediumSample sampleMedium(float altitudeKm) {
    float rayleighDensity = exp(-max(altitudeKm, 0.0) / RAYLEIGH_SCALE_H_KM);
    float mieDensity = exp(-max(altitudeKm, 0.0) / MIE_SCALE_H_KM);
    float ozoneDensity = max(0.0, 1.0 - abs(altitudeKm - OZONE_CENTER_KM) / OZONE_WIDTH_KM);

    MediumSample m;
    m.rayleigh = RAYLEIGH_SCATTERING * rayleighDensity;
    m.mie = MIE_SCATTERING * mieDensity;
    m.scattering = m.rayleigh + float3(m.mie);
    m.extinction = m.rayleigh + float3(MIE_EXTINCTION * mieDensity)
                 + OZONE_ABSORPTION * ozoneDensity;
    return m;
}

public float rayleighPhase(float cosTheta) {
    return 3.0 / (16.0 * SKY_PI) * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks, the phase function Hillaire's coefficients are fitted against.
public float miePhase(float cosTheta) {
    float g = MIE_G;
    float k = 3.0 / (8.0 * SKY_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(max(1.0 + g * g - 2.0 * g * cosTheta, 1.0e-4), 1.5);
}

public static const float ISOTROPIC_PHASE = 1.0 / (4.0 * SKY_PI);

/** Nearest non-negative ray/sphere (centred at the origin) intersection distance, or -1 for a miss. */
public float raySphereNearest(float3 origin, float3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    disc = sqrt(disc);
    float t0 = -b - disc;
    float t1 = -b + disc;
    if (t1 < 0.0) {
        return -1.0;
    }
    return t0 < 0.0 ? t1 : t0;
}

/** Farthest ray/sphere intersection distance (the exit point), or -1 when the ray misses entirely. */
public float raySphereFarthest(float3 origin, float3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b + sqrt(disc);
}

// ---- Transmittance LUT parameterisation (Bruneton). The mapping is by DISTANCE to the top of the
// atmosphere rather than by the angle itself, which is what keeps texels dense exactly where the optical
// depth changes fastest: the grazing directions around the horizon.
public float2 transmittanceLutUv(float radiusKm, float cosZenith) {
    float h = sqrt(max(ATMOS_TOP_KM * ATMOS_TOP_KM - ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM, 0.0));
    float rho = sqrt(max(radiusKm * radiusKm - ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM, 0.0));
    float discriminant = radiusKm * radiusKm * (cosZenith * cosZenith - 1.0) + ATMOS_TOP_KM * ATMOS_TOP_KM;
    float d = max(0.0, -radiusKm * cosZenith + sqrt(max(discriminant, 0.0)));
    float dMin = ATMOS_TOP_KM - radiusKm;
    float dMax = rho + h;
    return float2((d - dMin) / max(dMax - dMin, 1.0e-6), rho / max(h, 1.0e-6));
}

public void transmittanceLutParams(float2 uv, out float radiusKm, out float cosZenith) {
    float h = sqrt(max(ATMOS_TOP_KM * ATMOS_TOP_KM - ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM, 0.0));
    float rho = h * uv.y;
    radiusKm = sqrt(rho * rho + ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM);
    float dMin = ATMOS_TOP_KM - radiusKm;
    float dMax = rho + h;
    float d = dMin + uv.x * (dMax - dMin);
    cosZenith = d == 0.0 ? 1.0
            : (h * h - rho * rho - d * d) / (2.0 * radiusKm * d);
    cosZenith = clamp(cosZenith, -1.0, 1.0);
}

// Sampling any of these LUTs must address texel CENTRES: the parameterisations above are defined on the
// closed [0,1] range, while a sampler's [0,1] spans centre-to-centre of the outermost texels.
public float2 unitToSubUv(float2 uv, float2 size) {
    return (uv * (size - 1.0) + 0.5) / size;
}

public float2 subUvToUnit(float2 uv, float2 size) {
    return (uv * size - 0.5) / (size - 1.0);
}

public float3 sampleTransmittance(Sampler2D lut, float radiusKm, float cosZenith) {
    float2 uv = transmittanceLutUv(radiusKm, cosZenith);
    uv = unitToSubUv(saturate(uv), float2(float(TRANSMITTANCE_LUT_W), float(TRANSMITTANCE_LUT_H)));
    return lut.SampleLevel(uv, 0.0).rgb;
}

/**
 * Transmittance from the viewer to space along a world direction (world up = +Y), zero below the local
 * horizon.
 *
 * <p>The horizon gate is NOT redundant with the LUT. The transmittance parameterisation runs from
 * straight up to the ground-tangent ray and stops there, so every direction below the horizon clamps to
 * that last column — about 5% in red at sea level, not zero. Sampling the raw LUT for a set sun therefore
 * hands back ~6,000 lux of deep red light that never goes away, which would leave the world lit from
 * below all night and keep the sun's disc drawn under the ground. (The raymarch has no such problem: its
 * per-sample `lightVisibility` already gates on the same geometry.)
 *
 * <p>{@code angularRadius} is the light's own angular size, so a body sets over its own width instead of
 * switching off at a mathematical line.
 */
public float3 transmittanceToSpace(Sampler2D lut, float viewerRadiusKm, float3 dir, float angularRadius) {
    // cos of the zenith angle of the ground-tangent ray: the viewer's horizon, which dips further below
    // horizontal the higher the viewer is.
    float horizonCos = horizonZenithCos(viewerRadiusKm);
    float softness = max(sin(angularRadius), 1.0e-4);
    float aboveHorizon = smoothstep(horizonCos - softness, horizonCos + softness, dir.y);
    return sampleTransmittance(lut, viewerRadiusKm, dir.y) * aboveHorizon;
}

// ---- Multiple-scattering LUT parameterisation: linear in cos(sun zenith) and in altitude. The stored
// quantity is per unit illuminance, so one LUT serves the sun and the moon alike.
public float2 multiScatterLutUv(float radiusKm, float cosSunZenith) {
    float2 uv = float2(cosSunZenith * 0.5 + 0.5,
                       (radiusKm - ATMOS_BOTTOM_KM) / max(ATMOS_TOP_KM - ATMOS_BOTTOM_KM, 1.0e-6));
    return saturate(uv);
}

public void multiScatterLutParams(float2 uv, out float radiusKm, out float cosSunZenith) {
    cosSunZenith = clamp(uv.x * 2.0 - 1.0, -1.0, 1.0);
    radiusKm = lerp(ATMOS_BOTTOM_KM, ATMOS_TOP_KM, uv.y);
    // Keep off the exact boundaries: a sample sitting precisely on the ground or on the outer shell makes
    // the ray/sphere tests degenerate. The bottom uses the same fp32-derived floor as the viewer — a
    // smaller nudge would be lost to the cancellation in `r*r - bottom*bottom` (see
    // MIN_VIEWER_ALTITUDE_KM), leaving the LUT's ground row as garbage rather than as ground.
    radiusKm = clamp(radiusKm, ATMOS_BOTTOM_KM + MIN_VIEWER_ALTITUDE_KM,
                     ATMOS_TOP_KM - MIN_VIEWER_ALTITUDE_KM);
}

public float3 sampleMultiScatter(Sampler2D lut, float radiusKm, float cosSunZenith) {
    float2 uv = multiScatterLutUv(radiusKm, cosSunZenith);
    uv = unitToSubUv(uv, float2(float(MULTISCATTER_LUT_W), float(MULTISCATTER_LUT_H)));
    return lut.SampleLevel(uv, 0.0).rgb;
}

// Smallest viewer altitude the fp32 geometry can actually resolve, and the floor every consumer of a
// radius here clamps to.
//
// At radius == ATMOS_BOTTOM_KM the model degenerates: the horizon collapses onto the horizontal, the
// ray/ground test returns its tangent root at t == 0 so `tGround > 0` reads false, and the march runs
// straight through the planet instead of stopping at it.
//
// The floor has to be far larger than it looks, for two reasons that both come from working in km at
// planet scale:
//
//   * one fp32 ULP at 6360 km is 0.488 m, so any altitude below half a metre is not representable at all
//     — `6360.0001f` IS `6360.0f`, and a floor of a tenth of a metre silently does nothing;
//   * every horizon test computes `r*r - bottom*bottom`, a subtraction of two numbers near 4.0e7 where
//     the ULP is 4 km². The exact value is `2*bottom*h`, so at 0.5 m altitude that difference carries 26%
//     error, at 5 m 0.6%, and only past a few tens of metres does it stop mattering.
//
// 50 m is ~100 ULP of margin, leaves the horizon dip accurate to 0.001 degrees, and costs nothing: it is
// 0.05% of the 100 km shell, and with RtComposite's 100-blocks-per-km scale it only raises cameras below
// about y = 68 — which it moves by a fifth of a degree of horizon dip, well under the width of the
// drawn sun. The alternative is reformulating every ray/sphere test to take altitude instead of radius
// (the exact `2*bottom*h + h*h` form); that is worth doing only if the viewer ever needs to sit within
// metres of the shell, which nothing here does.
public static const float MIN_VIEWER_ALTITUDE_KM = 0.05;

/** Cosine of the zenith angle of the ground-tangent direction: the viewer's geometric horizon. */
public float horizonZenithCos(float viewerRadiusKm) {
    return -sqrt(max(1.0 - (ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM)
            / (viewerRadiusKm * viewerRadiusKm), 0.0));
}

// ---- Sky-view LUT parameterisation (Hillaire). Azimuth is measured FROM the light, which the dome is
// symmetric about, so two angles describe the whole sphere. The elevation mapping is split at the
// geometric horizon and squared on each side, putting the most texels exactly where the sky's gradient is
// steepest, keeping the horizon continuous and well resolved.
public float2 skyViewLutUv(float viewerRadiusKm, float viewZenithCos, float lightViewCos, int body) {
    float horizonSin = sqrt(max(viewerRadiusKm * viewerRadiusKm
            - ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM, 0.0)) / max(viewerRadiusKm, 1.0e-6);
    float horizonAngle = acos(clamp(horizonSin, -1.0, 1.0)); // angle from the DOWN axis to the horizon
    float zenithHorizonAngle = SKY_PI - horizonAngle;        // angle from the UP axis to the horizon
    float viewZenithAngle = acos(clamp(viewZenithCos, -1.0, 1.0));

    float v;
    if (viewZenithAngle < zenithHorizonAngle) {
        float coord = viewZenithAngle / max(zenithHorizonAngle, 1.0e-6);
        coord = 1.0 - sqrt(max(1.0 - coord, 0.0));
        v = coord * 0.5;
    } else {
        float coord = (viewZenithAngle - zenithHorizonAngle) / max(horizonAngle, 1.0e-6);
        v = sqrt(max(coord, 0.0)) * 0.5 + 0.5;
    }
    float u = sqrt(max(0.5 - 0.5 * lightViewCos, 0.0));

    float2 uv = unitToSubUv(float2(u, saturate(v)),
            float2(float(SKY_VIEW_LUT_W), float(SKY_VIEW_LUT_H)));
    // Fold the slice offset in after the half-texel correction so a body's rows never bleed into its
    // neighbour's: v is already inside [0.5/108, 1-0.5/108] of its own slice.
    return float2(uv.x, (uv.y + float(body)) / float(SKY_VIEW_BODY_COUNT));
}

public void skyViewLutParams(float2 uv, float viewerRadiusKm,
                             out float viewZenithCos, out float lightViewCos) {
    float2 unit = subUvToUnit(saturate(uv), float2(float(SKY_VIEW_LUT_W), float(SKY_VIEW_LUT_H)));
    float horizonSin = sqrt(max(viewerRadiusKm * viewerRadiusKm
            - ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM, 0.0)) / max(viewerRadiusKm, 1.0e-6);
    float horizonAngle = acos(clamp(horizonSin, -1.0, 1.0));
    float zenithHorizonAngle = SKY_PI - horizonAngle;

    float viewZenithAngle;
    if (unit.y < 0.5) {
        float coord = 1.0 - unit.y * 2.0;
        viewZenithAngle = zenithHorizonAngle * (1.0 - coord * coord);
    } else {
        float coord = unit.y * 2.0 - 1.0;
        viewZenithAngle = zenithHorizonAngle + horizonAngle * coord * coord;
    }
    viewZenithCos = cos(viewZenithAngle);
    lightViewCos = clamp(1.0 - 2.0 * unit.x * unit.x, -1.0, 1.0);
}

/**
 * The canonical frame the sky-view LUT is baked in: world up is +Y and the light's azimuth is placed on
 * +X. Any world configuration maps onto it, because the dome is rotationally symmetric about up and
 * mirror-symmetric about the light's meridian.
 */
public void skyViewLocalDirections(float lightZenithCos, float viewZenithCos, float lightViewCos,
                                   out float3 viewDir, out float3 lightDir) {
    float lightZenithSin = sqrt(max(1.0 - lightZenithCos * lightZenithCos, 0.0));
    lightDir = float3(lightZenithSin, lightZenithCos, 0.0);
    float viewZenithSin = sqrt(max(1.0 - viewZenithCos * viewZenithCos, 0.0));
    float lightViewSin = sqrt(max(1.0 - lightViewCos * lightViewCos, 0.0));
    viewDir = float3(viewZenithSin * lightViewCos, viewZenithCos, viewZenithSin * lightViewSin);
}

/** Cosine of the azimuth between a view direction and a light, both in world space (up = +Y). */
public float lightViewCosine(float3 viewDir, float3 lightDir) {
    float2 viewAzimuth = viewDir.xz;
    float2 lightAzimuth = lightDir.xz;
    float viewLen = length(viewAzimuth);
    float lightLen = length(lightAzimuth);
    if (viewLen < 1.0e-5 || lightLen < 1.0e-5) {
        return 1.0; // straight up/down, or a light at the zenith: azimuth is undefined and irrelevant
    }
    return clamp(dot(viewAzimuth / viewLen, lightAzimuth / lightLen), -1.0, 1.0);
}

public struct ScatteringResult {
    public float3 luminance;      // cd/m²
    public float3 multiScatAs1;   // per-unit-illuminance response used to build the multiscatter LUT
};

/**
 * Single-scattering raymarch with an optional multiple-scattering lookup, following Hillaire's
 * IntegrateScatteredLuminance.
 *
 * <p>{@code lightAngularRadius} softens the planet's shadow on the atmosphere over the light's finite
 * angular size instead of switching it per sample, preventing the transition from landing on the
 * primary march's sample lattice.
 *
 * <p>{@code clipGround} false marches straight past the planet to the far side of the shell and adds no
 * ground bounce. Evaluated exactly along the ground-tangent direction that is the limit of the
 * above-horizon sky, which is what {@code sky_view.comp} blends the below-horizon rows toward.
 */
public ScatteringResult integrateScatteredLuminance(
        float3 originKm, float3 dir, float3 lightDir, float3 illuminance,
        float lightAngularRadius, int steps, bool applyPhase,
        bool useMultiScatter, Sampler2D transmittanceLut, Sampler2D multiScatterLut,
        float groundAlbedo, bool clipGround) {
    ScatteringResult result;
    result.luminance = float3(0.0);
    result.multiScatAs1 = float3(0.0);

    float tTop = raySphereFarthest(originKm, dir, ATMOS_TOP_KM);
    if (tTop <= 0.0) {
        return result; // outside the atmosphere looking away from it
    }
    float tGround = raySphereNearest(originKm, dir, ATMOS_BOTTOM_KM);
    bool hitGround = clipGround && tGround > 0.0;
    float tMax = hitGround ? tGround : tTop;

    float cosTheta = dot(dir, lightDir);
    float phaseR = applyPhase ? rayleighPhase(cosTheta) : ISOTROPIC_PHASE;
    float phaseM = applyPhase ? miePhase(cosTheta) : ISOTROPIC_PHASE;
    float shadowSoftness = max(sin(lightAngularRadius), 1.0e-4);

    float3 throughput = float3(1.0);
    float dt = tMax / float(steps);
    for (int i = 0; i < steps; i++) {
        float t = (float(i) + 0.5) * dt;
        float3 p = originKm + dir * t;
        float radius = length(p);
        float3 up = p / max(radius, 1.0e-6);
        float altitude = radius - ATMOS_BOTTOM_KM;
        MediumSample medium = sampleMedium(altitude);
        float3 stepTransmittance = exp(-medium.extinction * dt);

        // Planet shadow, softened over the light's angular radius: `clearance` is how far the light sits
        // above this sample's own geometric horizon, which rises as the sample does — so high-altitude
        // samples stay lit well after the ground below them is in shadow. That altitude dependence is
        // what makes a real twilight glow, and it is the reason no artificial fill term is needed.
        float horizonDipSin = sqrt(max(1.0 - (ATMOS_BOTTOM_KM * ATMOS_BOTTOM_KM) / (radius * radius), 0.0));
        float clearance = dot(up, lightDir) + horizonDipSin;
        float lightVisibility = smoothstep(-shadowSoftness, shadowSoftness, clearance);

        float cosLightZenith = dot(up, lightDir);
        float3 transmittanceToLight = sampleTransmittance(transmittanceLut, radius, cosLightZenith);
        float3 inScatter = lightVisibility * transmittanceToLight
                * (medium.rayleigh * phaseR + medium.mie * phaseM);
        if (useMultiScatter) {
            // Multiple scattering arrives from every direction, so it couples to the total scattering
            // coefficient rather than to a phase function.
            inScatter += sampleMultiScatter(multiScatterLut, radius, cosLightZenith) * medium.scattering;
        }

        // Analytic integration of the constant-medium segment (Hillaire): exact for the step rather than
        // a midpoint sample, which is what lets this run at 32 steps without banding.
        float3 safeExtinction = max(medium.extinction, float3(1.0e-9));
        float3 scattered = illuminance * inScatter;
        result.luminance += throughput * (scattered - scattered * stepTransmittance) / safeExtinction;

        float3 msIntegral = (medium.scattering - medium.scattering * stepTransmittance) / safeExtinction;
        result.multiScatAs1 += throughput * msIntegral;

        throughput *= stepTransmittance;
    }

    if (hitGround && groundAlbedo > 0.0) {
        // Lambertian ground bounce. Almost always hidden behind terrain in a Minecraft world, but it is
        // what keeps a below-horizon direction (over a void, or through the world edge) a plausible dim
        // colour instead of black, with no separate below-horizon code path to seam against.
        float3 p = originKm + dir * tMax;
        float3 up = p / max(length(p), 1.0e-6);
        float cosLightZenith = dot(up, lightDir);
        float3 transmittanceToLight = sampleTransmittance(transmittanceLut, ATMOS_BOTTOM_KM, cosLightZenith);
        float ndl = max(cosLightZenith, 0.0);
        result.luminance += throughput * illuminance * transmittanceToLight * ndl
                * (groundAlbedo / SKY_PI);
    }
    return result;
}

// ---- Celestial geometry ------------------------------------------------------------------------

/**
 * World direction of a celestial body from Minecraft's own eased angle (EnvironmentAttributes.SUN_ANGLE /
 * MOON_ANGLE, radians) and the look package's noon tilt toward south. Angle 0 puts the body at its
 * highest point. Matches vanilla's arc: east-west travel, tilted toward +Z at its peak.
 */
public float3 celestialDirection(float angleRad, float noonTiltRad) {
    float peak = cos(angleRad);
    return float3(-sin(angleRad), cos(noonTiltRad) * peak, sin(noonTiltRad) * peak);
}

/** The pole the sun/moon arc about, and the axis the starfield wheels on. */
public float3 celestialAxis(float noonTiltRad) {
    return float3(0.0, -sin(noonTiltRad), cos(noonTiltRad));
}

/**
 * Tangent frame for a celestial body's square, kept level with the horizon: {@code right} is horizontal
 * and {@code up} lies in the vertical plane through the body. This keeps the sun and moon from rolling
 * about their own centres as they cross the sky. The same frame
 * builds the NEE shadow-ray square in math.slang, so the sampled light and the drawn body stay identical.
 */
public void celestialSquareFrame(float3 bodyDir, out float3 right, out float3 up) {
    // Near the zenith the horizontal reference degenerates; the arc's tilt keeps the sun/moon well away
    // from it in practice, so any stable fallback will do.
    float3 reference = abs(bodyDir.y) > 0.9995 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    right = normalize(cross(reference, bodyDir));
    up = cross(bodyDir, right);
}

/**
 * Gnomonic projection of {@code dir} into a body's tangent square, in units of its half-angle:
 * {@code local} within [-1,1]² is on the body. A square (not a disc) because vanilla draws both bodies as
 * textured quads, and the sprite's own alpha then shapes what is actually visible.
 */
public bool celestialSquareLocal(float3 dir, float3 bodyDir, float halfAngle, out float2 local) {
    local = float2(2.0, 2.0);
    float c = dot(dir, bodyDir);
    if (c <= 1.0e-3) {
        return false;
    }
    float3 right;
    float3 up;
    celestialSquareFrame(bodyDir, right, up);
    float3 planar = dir / c; // the point where the ray pierces the body's tangent plane
    local = float2(dot(planar, right), dot(planar, up)) / tan(halfAngle);
    return all(abs(local) < float2(1.0, 1.0));
}

/**
 * Circular counterpart of {@link celestialSquareLocal}: the gnomonic offset from a body's centre in units
 * of its angular radius, with the tangent frame needed to place a light direction on its face. The test is
 * against a radius rather than a square, and it reaches slightly past the limb so the caller can fade the
 * edge instead of staircasing a body that subtends under a degree.
 */
public bool celestialDiscLocal(float3 dir, float3 bodyDir, float halfAngle, out float2 local,
                               out float3 right, out float3 up) {
    local = float2(2.0, 2.0);
    right = float3(0.0);
    up = float3(0.0);
    float c = dot(dir, bodyDir);
    if (c <= 1.0e-3) {
        return false;
    }
    celestialSquareFrame(bodyDir, right, up);
    float3 planar = dir / c;
    local = float2(dot(planar, right), dot(planar, up)) / tan(halfAngle);
    return dot(local, local) <= 1.25;
}

/**
 * Disc radiance for a circular body, keeping the drawn body's total power equal to the illuminance it
 * delivers: L = E / projected area = E / (pi tan^2 halfAngle). This is the circular counterpart of the
 * square convention in {@link discRadiance}, and it is what lets the sun's apparent size and the softness
 * of its shadow be the same authored number rather than two that can drift apart.
 */
public float3 discRadianceCircular(float illuminance, float halfAngle) {
    float t = tan(halfAngle);
    return float3(illuminance / max(SKY_PI * t * t, 1.0e-12));
}

// ---- Per-frame sky state ------------------------------------------------------------------------
//
// WorldPush carries only what the CPU alone can know: Minecraft's four eased celestial angles, its star
// brightness, the moon phase, and the immutable look-package constants. Every direction, colour, level,
// and transmittance is derived here so atmospheric physics has one implementation.

// DimensionType.Skybox's ordinal order (the shader keeps the numeric push, not an enum).
public static const int SKYBOX_NONE      = 0;
public static const int SKYBOX_OVERWORLD = 1;
public static const int SKYBOX_END       = 2;

// Vanilla's buildEndSky() paints every end-sky texel with this flat vertex colour (ARGB 0xFF282828).
public static const float END_SKY_VERTEX_GRAY = 40.0 / 255.0;

// Vanilla's end sky cube is camera-centred with faces at +-100 and the flash quad rides 100 up, both in
// the sky renderer's own units. Only their ratio reaches the shaders: the cube edge and the flash height.
public static const float END_SKY_CUBE_HALF = 100.0;
public static const float END_FLASH_HEIGHT  = 100.0;
// Vanilla's renderEndFlash() scales the flash quad to +-60 about its centre.
public static const float END_FLASH_HALF    = 60.0;

public struct SkyState {
    public float3 sunDir;
    public float3 moonDir;
    public float  sunIlluminance;      // lux, top of atmosphere
    public float  moonIlluminance;     // lux, already scaled by the visible phase
    public float  moonFullIlluminance; // lux, un-scaled: the surface brightness a phase-lit disc shows
    public float  viewerRadiusKm;      // planet centre to viewer
    public float  sunAngularRadius;    // NEE sampling half-angle, also the planet-shadow softness
    public float  moonAngularRadius;
    public float  sunDiscHalfAngle;    // drawn size, decoupled from the NEE radius (vanilla quads are huge)
    public float  moonDiscHalfAngle;
    public float  starAngle;
    public float  starLuminance;       // cd/m², MC's star brightness times the look package's anchor
    public float  airglowLuminance;    // cd/m², the real moonless-night floor (airglow + unresolved stars)
    public float  noonTilt;
    public float  groundAlbedo;
    public float  horizonSoften;       // radians below the horizon over which the ground fades in
    public float  moonPhaseIndex;      // 0 full .. 4 new
    public float  skyLightFactor;      // 0 in a dimension with no sky light (Nether, End)
    public float3 dimensionSkyColor;   // linear BT.709 sky colour the dimension supplies instead
    public int    skybox;              // DimensionType.Skybox ordinal (SKYBOX_*)
    public float  flashIntensity;      // end flash, eased 0..1 (0 when absent or hidden)
    public float  flashXAngle;         // end flash angles, rad, verbatim from EndFlashState
    public float  flashYAngle;
    public float3 skyLightColor;       // linear BT.709, SKY_LIGHT_COLOR: flash + ambient tint
    public float  ambientLight;        // DimensionType.ambientLight scalar (End 0.25)
    public float3 ambientLightColor;   // linear BT.709, AMBIENT_LIGHT_COLOR
    public float3 fogColor;            // linear BT.709, FOG_COLOR: the end dome's horizon colour
    public float  flashIlluminanceLux; // end flash NEE illuminance at full intensity
    public float  flashAngularRadius;  // end flash NEE half-angle: penumbra softness of the glow
    public float  flashQuadLuminance;  // end flash drawn quad scale (cd/m² of the sprite's white point)
    public float  ambientIlluminanceLux; // end ambient irradiance at ambientLight 1.0
    public float  skyBoxGain;          // multiplier on the sampled end_sky box
    public float  skyGradientGain;     // multiplier on the attribute-colour end dome gradient
    public float4 endFlashUv;
    public float4 sunUv;
    public float4 moonUv;
};

/** Minecraft's moon phases run 0 = full to 4 = new, then mirror back toward full through phase 7. */
public float moonLitFraction(float moonPhaseIndex) {
    return abs(moonPhaseIndex - 4.0) / 4.0;
}

public SkyState skyState(WorldPush push) {
    SkyState s;
    s.noonTilt = push.skyLook1.x;
    s.sunDir = celestialDirection(push.celestial.x, s.noonTilt);
    s.moonDir = celestialDirection(push.celestial.y, s.noonTilt);
    s.sunIlluminance = push.skyLook0.x;
    // A fixed floor plus the visible phase: a new moon still casts a little directional light, which is
    // a gameplay decision the look package owns rather than an astronomical one.
    float phaseFixed = push.skyLook1.w;
    s.moonIlluminance = push.skyLook0.y
            * (phaseFixed + (1.0 - phaseFixed) * moonLitFraction(push.skyLook2.w));
    // A lit patch of regolith is as bright at first quarter as at full; only how much of the disc is lit
    // changes. A disc drawn from the phase-scaled value above would therefore fade rather than wane.
    s.moonFullIlluminance = push.skyLook0.y;
    s.viewerRadiusKm = ATMOS_BOTTOM_KM + max(push.skyLook2.z, MIN_VIEWER_ALTITUDE_KM);
    s.sunAngularRadius = push.skyLook1.y;
    s.moonAngularRadius = push.skyLook1.z;
    s.sunDiscHalfAngle = push.skyLook2.x;
    s.moonDiscHalfAngle = push.skyLook2.y;
    s.starAngle = push.celestial.z;
    s.starLuminance = push.celestial.w * push.skyLook0.w;
    s.airglowLuminance = push.skyLook0.z;
    s.groundAlbedo = push.skyLook3.x;
    s.horizonSoften = push.skyLook3.y;
    s.moonPhaseIndex = push.skyLook2.w;
    s.skyLightFactor = push.skyEnvironment.x;
    s.dimensionSkyColor = push.skyEnvironment.yzw;
    s.skybox = int(push.skyboxState.x);
    s.flashIntensity = max(push.skyboxState.y, 0.0);
    s.flashXAngle = push.skyboxState.z;
    s.flashYAngle = push.skyboxState.w;
    s.skyLightColor = push.dimensionLight.xyz;
    s.ambientLight = max(push.dimensionLight.w, 0.0);
    s.ambientLightColor = push.dimensionAmbient.xyz;
    s.fogColor = push.dimensionFog.xyz;
    s.flashIlluminanceLux = max(push.skyLook4.x, 0.0);
    s.flashQuadLuminance = max(push.skyLook4.y, 0.0);
    s.ambientIlluminanceLux = max(push.skyLook4.z, 0.0);
    s.skyBoxGain = push.skyLook4.w;
    s.flashAngularRadius = push.skyLook3.z;
    s.skyGradientGain = push.skyLook3.w;
    s.endFlashUv = push.endFlashUv;
    s.sunUv = push.sunUv;
    s.moonUv = push.moonUv;
    return s;
}

/**
 * Radiance of a dimension that has no sky light. Such a dimension still carries a sky colour, which
 * vanilla shows as the haze filling distant view; it is not an emissive dome, so only its chromaticity is
 * taken and it is given the night airglow's luminance -- the dimmest level the exposure controller already
 * handles -- rather than any absolute brightness the colour itself might imply.
 *
 * <p>The End replaces that flat stand-in with its own dome: the dimension's fog colour at and below the
 * horizon easing to its sky colour overhead (vanilla's attributes are FOG_COLOR #181318 against
 * SKY_COLOR #000000), plus ambient_light as an isotropic floor so unlit surfaces read without a flash.
 * The floor's tint is the sum of the two attribute colours the vanilla lightmap consumes for the same
 * purpose (SKY_LIGHT_COLOR + AMBIENT_LIGHT_COLOR), taken raw so the authored illuminance stays
 * photometric. The box texture from {@code end_sky.png} is layered on top by the miss shader only: it is
 * presentation, and the SH bake below intentionally integrates the smooth dome.
 */
public float3 dimensionSkyRadiance(SkyState s, float3 dir) {
    float3 sky = max(s.dimensionSkyColor, float3(0.0));
    if (s.skybox == SKYBOX_END) {
        float3 fog = max(s.fogColor, float3(0.0));
        float3 gradient = lerp(fog, sky, smoothstep(0.0, 0.5, dir.y)) * s.skyGradientGain;
        float3 ambientChroma = max(s.skyLightColor, float3(0.0))
                + max(s.ambientLightColor, float3(0.0));
        float3 ambient = ambientChroma * (s.ambientLight * s.ambientIlluminanceLux / SKY_PI);
        return gradient + ambient;
    }
    float luma = dot(sky, float3(0.2126, 0.7152, 0.0722));
    if (luma <= 0.0) {
        return float3(0.0);
    }
    return sky * (s.airglowLuminance / luma);
}

public struct CelestialLight {
    public float3 dir;
    public float3 illuminance;  // linear BT.709, lux at the viewer
    public float  halfAngle;
};

/** Atmosphere in-scatter for a world direction: both bodies' slices of this frame's sky-view LUT. */
public float3 skyDome(Sampler2D viewLut, float3 dir, SkyState state) {
    float2 sunUv = skyViewLutUv(state.viewerRadiusKm, dir.y,
            lightViewCosine(dir, state.sunDir), SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(state.viewerRadiusKm, dir.y,
            lightViewCosine(dir, state.moonDir), SKY_VIEW_BODY_MOON);
    return viewLut.SampleLevel(sunUv, 0.0).rgb + viewLut.SampleLevel(moonUv, 0.0).rgb;
}

/**
 * The radiance the diffuse sky basis integrates: the dome plus the isotropic airglow floor, blended toward
 * the dimension's own flat sky colour by the same factor world.rmiss uses. Resolved stars are deliberately
 * absent — their integral over the sphere is negligible next to either term, and a point-like source is
 * exactly what a one-sample estimator turns into a firefly.
 */
public float3 skyDiffuseRadiance(Sampler2D viewLut, float3 dir, SkyState state) {
    float3 sky = skyDome(viewLut, dir, state) + state.airglowLuminance;
    return lerp(dimensionSkyRadiance(state, dir), sky, state.skyLightFactor);
}

// ---- End flash ----------------------------------------------------------------------------------
//
// Vanilla's end flash is one quad in the celestials atlas (end_flash.png: a radial purple glow with a
// black surround, drawn additively), hinged at (0, END_FLASH_HEIGHT, 0) and re-oriented every 600 ticks
// by EndFlashState's angles. renderEndFlash() rotates the pose by Y(180° - yAngle) then X(-90° - xAngle)
// about the camera BEFORE the translate, so with xAngle 0 the quad hangs on the horizon at azimuth
// yAngle and xAngle swings it up toward the zenith (the datapack's range is [-60°, +10°]).
//
// Both consumers below derive from one direction: the quad's centre, which is also where its glow peaks.

/**
 * World direction of the end flash's centre for EndFlashState's angles (rad). This is also the quad
 * plane's unit normal: the quad is rotated rigidly about the viewer before its height translate, so the
 * plane always passes through height*direction with the normal facing away from the viewer -- vanilla
 * renders it unculled and so do we.
 */
public float3 endFlashCentreDirection(float xAngleRad, float yAngleRad) {
    return float3(-cos(xAngleRad) * sin(yAngleRad), -sin(xAngleRad),
                  cos(yAngleRad) * cos(xAngleRad));
}

/**
 * The end flash as NEE's one directional light. Vanilla delivers it through the lightmap with the
 * dimension's SKY_LIGHT_COLOR tint, so the traced equivalent takes that tint at the look package's
 * authored illuminance and a wide half-angle (the glow is tens of degrees across, nothing like the
 * sun's disc) -- reusing the existing single-celestial NEE keeps the shadow-ray budget unchanged.
 * Zero intensity returns the absent light (half-angle 0), which callers skip without tracing.
 */
public CelestialLight endFlashLight(SkyState s) {
    CelestialLight light;
    light.halfAngle = 0.0;
    light.dir = float3(0.0, 1.0, 0.0);
    light.illuminance = float3(0.0);
    if (s.flashIntensity <= 0.0) {
        return light;
    }
    float3 centreDir = endFlashCentreDirection(s.flashXAngle, s.flashYAngle);
    light.dir = centreDir;
    light.illuminance = s.skyLightColor * (s.flashIlluminanceLux * s.flashIntensity);
    light.halfAngle = s.flashAngularRadius;
    return light;
}

// ---- Diffuse sky irradiance basis.
//
// The dome is convolved once per frame into order-2 real spherical harmonics with the Lambert transfer
// already folded in, so a shading point gets its entire unoccluded sky irradiance from a 9-term sum instead
// of the single cosine sample it used to pay for. That sample was the dominant variance at night: with the
// sun 24 EV above the moon, the diffuse continuation ray's sky hit IS the image, and its value swings by
// whatever direction the sample happened to land on.
//
// sky_lut/irradiance.comp bakes it, world_indirect.rgen evaluates it. Both go through the functions below
// so the two can never disagree about the basis.
public static const int SKY_IRRADIANCE_COEFFS = 9;
public static const int SKY_IRRADIANCE_SAMPLES = 512;

/**
 * Orthonormal real SH, up = +Y, indexed k = 0..8 as (l,m): 00, 1-,10,1+, 2-2,2-1,20,2+1,2+2.
 *
 * <p>Only self-consistency matters here — bake and evaluate use this same table — so the m<0/m>0 axis
 * assignment is ours to choose rather than a convention to obey.
 */
public float shBasis(int k, float3 d) {
    if (k == 0) return 0.28209479;
    if (k == 1) return 0.48860251 * d.z;
    if (k == 2) return 0.48860251 * d.y;
    if (k == 3) return 0.48860251 * d.x;
    if (k == 4) return 1.09254843 * d.x * d.z;
    if (k == 5) return 1.09254843 * d.z * d.y;
    if (k == 6) return 0.31539157 * (3.0 * d.y * d.y - 1.0);
    if (k == 7) return 1.09254843 * d.x * d.y;
    return 0.54627422 * (d.x * d.x - d.z * d.z);
}

/**
 * The cos-lobe (Lambert) convolution scale for coefficient k: pi, 2pi/3, pi/4 for l = 0, 1, 2. Applying it
 * at bake time is what lets the evaluate side be a plain sum of `coefficient * basis`.
 */
public float shLambertTransfer(int k) {
    if (k == 0) return SKY_PI;
    if (k < 4) return 2.0 * SKY_PI / 3.0;
    return SKY_PI / 4.0;
}

/**
 * Equal-weight spherical quadrature point i of n: a golden-angle spiral, so consecutive rows never line up
 * and the coverage of a fixed n is stable enough for the basis to be identical frame to frame.
 */
public float3 skyQuadratureDir(int i, int n) {
    float k = float(i) + 0.5;
    float y = 1.0 - 2.0 * k / float(n);
    float r = sqrt(max(0.0, 1.0 - y * y));
    float phi = k * 2.399963229728653; // 2*pi*(1 - 1/golden ratio)
    return float3(r * cos(phi), y, r * sin(phi));
}

/** Irradiance (the cos-weighted integral of the dome over the hemisphere about n) from baked coefficients. */
public float3 skyIrradiance(RWTexture2D<float4> lut, float3 n) {
    float3 e = float3(0.0, 0.0, 0.0);
    for (int k = 0; k < SKY_IRRADIANCE_COEFFS; ++k) {
        e += lut.Load(int2(k, 0)).rgb * shBasis(k, n);
    }
    // A clamped-normal SH reconstruction can dip below zero between lobes; irradiance cannot.
    return max(e, float3(0.0, 0.0, 0.0));
}

/**
 * The single directional light NEE samples: whichever of the two bodies actually delivers more light
 * after atmospheric extinction.
 *
 * <p>There is deliberately no crossfade between them. The extinction along a horizon-grazing path is
 * enormous (roughly 70 air masses), so both bodies are already within a rounding error of zero at the
 * moment the choice flips, making the handoff invisible without an authored smoothstep.
 *
 * <p>A dimension whose sky-light factor is zero has no sun or moon at all, and the returned half-angle is
 * zero as well so callers can skip the shadow ray rather than sample a light of zero strength. The End
 * takes that same slot with its own body: the end flash, whose vanilla delivery is a lightmap term
 * independent of sky light -- see {@link endFlashLight}.
 */
public CelestialLight dominantCelestialLight(SkyState s, Sampler2D transmittanceLut) {
    CelestialLight absent;
    absent.dir = s.sunDir;
    absent.illuminance = float3(0.0);
    absent.halfAngle = 0.0;
    if (s.skybox == SKYBOX_END) {
        return endFlashLight(s);
    }
    if (s.skyLightFactor <= 0.0) {
        return absent;
    }

    float3 sunLux = s.sunIlluminance
            * transmittanceToSpace(transmittanceLut, s.viewerRadiusKm, s.sunDir, s.sunAngularRadius);
    // Moonlight is sunlight reflected off a grey body: its colour is the sun's, not a cool tint. The blue
    // cast of a moonlit night is the viewer's own scotopic response, which belongs in the look transform,
    // not in the light's spectrum.
    float3 moonLux = s.moonIlluminance
            * transmittanceToSpace(transmittanceLut, s.viewerRadiusKm, s.moonDir, s.moonAngularRadius);

    CelestialLight light;
    if (dot(sunLux, float3(0.2126, 0.7152, 0.0722)) >= dot(moonLux, float3(0.2126, 0.7152, 0.0722))) {
        light.dir = s.sunDir;
        light.illuminance = sunLux;
        light.halfAngle = s.sunAngularRadius;
    } else {
        light.dir = s.moonDir;
        light.illuminance = moonLux;
        light.halfAngle = s.moonAngularRadius;
    }
    // A partial factor (a datapack may author one) dims the body rather than switching it off, so the
    // atmosphere's own contribution and this directional light stay in the same proportion.
    light.illuminance *= s.skyLightFactor;
    return light;
}
