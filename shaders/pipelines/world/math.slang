// Stateless math: luminance, GGX (D/G/VNDF), Fresnel, Henyey-Greenstein, the RR specular-albedo
// fit, the PCG stream, the camera's own frame, and direction sampling. No bindings, no rays, and no module
// state beyond the lens lattice's step.

import world_common;
import world_core;
import sky;

public float luminance(float3 c) { return dot(c, ACESCG_LUMA); }

// GGX normal distribution.
// NOTE: the 1e-7 denominator guard is not merely a divide-by-zero epsilon, and it is load-bearing. At
// the lobe peak d == a2, so the guard dominates PI*d*d until a2 >> 1.8e-4, i.e. rough >> 0.116; every
// smoother material has its NEE highlight peak truncated. That truncation is what keeps the sun from
// being counted twice: a specular continuation sets showCelestial, so the BSDF-sampled ray may hit the
// sun disc in world.rmiss while NEE is also adding a sun term for the same lobe, with no MIS weighting
// between them. Removing the guard restores the physical peak AND the double count, and makes the
// single-sample sun estimator explode wherever the lobe is tighter than the disc (~0.0047 rad). The
// real fix is MIS (or a disc-aware specular light sample), not an epsilon edit — left as-is.
public float ggxD(float ndh, float alpha) {
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1.0e-7);
}

// Smith GGX masking term for one direction (separable form; G2 = G1(v) * G1(l)).
public float ggxG1(float ndx, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * ndx / (ndx + sqrt(a2 + (1.0 - a2) * ndx * ndx) + 1.0e-7);
}

public float3 fresnelSchlick(float cosT, float3 f0) {
    float m = clamp(1.0 - cosT, 0.0, 1.0);
    float m2 = m * m;
    return f0 + (1.0 - f0) * (m2 * m2 * m);
}

// Henyey–Greenstein phase function (with 4π normalisation so it integrates to 1 over the sphere).
// g=0 → isotropic; g>0 → forward-scatter peak at cosT=1 (light and view aligned through a thin slab).
public static const float SSS_G        = 0.6; // forward-scatter anisotropy; leaves/grass are quite directional
public static const float SSS_STRENGTH = 1.0; // 1.0 ≈ 2.5× the Lambertian at peak (sss=1, cosT=1, backNdl=1)
// Keep SSS on the first two Pass B hits. Pass A's primary/interface prefix is outside this local depth.
public static const int   MAX_SSS_INDIRECT_DEPTH = 1;
public float hg(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(0.0, 1.0 + g2 - 2.0 * g * cosT), 1.5));
}

// NVIDIA RR guide-buffer helper: integrated/view-dependent specular reflectivity, not raw F0.
// alpha is GGX alpha (== the linear roughness materials store), NoV is the first-hit view cosine.
public float3 rrSpecularAlbedo(float3 specularColor, float alpha, float NoV) {
    NoV = abs(NoV);
    float NoV2 = NoV * NoV;
    float alpha2 = alpha * alpha;
    float4 X = float4(1.0, NoV, NoV2, NoV * NoV2);
    float4 Y = float4(1.0, alpha, alpha2, alpha * alpha2);

    float2 m1 = float2(
            dot(float2(0.99044, -1.28514), X.xy),
            dot(float2(1.29678, -0.755907), X.xy));
    float3 Xxyw = float3(X.x, X.y, X.w);
    float3 m2 = float3(
            dot(float3(1.0, 2.92338, 59.4188), Xxyw),
            dot(float3(20.3225, -27.0302, 222.592), Xxyw),
            dot(float3(121.563, 626.13, 316.627), Xxyw));
    float bias = dot(m1, Y.xy) / max(dot(m2, float3(Y.x, Y.y, Y.w)), 1.0e-7);

    float2 m3 = float2(
            dot(float2(0.0365463, 3.32707), X.xy),
            dot(float2(9.0632, -9.04756), X.xy));
    float3 Xxzw = float3(X.x, X.z, X.w);
    float3 m4 = float3(
            dot(float3(1.0, 3.59685, -1.36772), Xxzw),
            dot(float3(9.04401, -16.3174, 9.22949), Xxzw),
            dot(float3(5.56589, 19.7886, -20.2123), Xxzw));
    float scale = dot(m3, Y.xy) / max(dot(m4, float3(Y.x, Y.y, Y.w)), 1.0e-7);

    bias *= clamp(specularColor.g * 50.0, 0.0, 1.0);
    return specularColor * max(0.0, scale) + float3(max(0.0, bias), max(0.0, bias), max(0.0, bias));
}

// Exact (unpolarized) Fresnel reflectance for a dielectric interface. cosI is the incidence cosine on
// the incoming side; etaI/etaT are the IORs the ray travels from / into. Returns 1.0 on total internal
// reflection (sin of the transmitted angle >= 1), which matches refract() returning the zero vector —
// so reflect/refract selection and TIR stay consistent.
public float fresnelDielectric(float cosI, float etaI, float etaT) {
    float sinT2 = (etaI * etaI) / (etaT * etaT) * max(0.0, 1.0 - cosI * cosI);
    if (sinT2 >= 1.0) {
        return 1.0;
    }
    float cosT = sqrt(1.0 - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaI * cosT - etaT * cosI) / (etaI * cosT + etaT * cosI);
    return 0.5 * (rs * rs + rp * rp);
}

// PCG hash RNG.
public uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
public float rndf(inout uint s) {
    // Convert the high 24 bits, which are exactly representable as float. Converting all 32 bits first
    // lets the top 128 uint values round to 2^32, incorrectly returning 1.0 and breaking `< probability`
    // tests (most seriously the F == 1 total-internal-reflection branch).
    return float(pcg(s) >> 8u) * (1.0 / 16777216.0);
}

public float3 primaryRayDir(float2 ndc) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

public float primaryRayConeSpread(float2 ndc, float2 size, float3 dir) {
    float2 onePixelNdc = 2.0 / size;
    float3 dx = primaryRayDir(ndc + float2(onePixelNdc.x, 0.0));
    float3 dy = primaryRayDir(ndc + float2(0.0, onePixelNdc.y));
    return max(max(length(cross(dir, dx)), length(cross(dir, dy))), RAY_CONE_MIN_SPREAD);
}

// The plastic number's two reciprocal powers: the per-frame step of the 2D low-discrepancy lattice the
// lens samples walk.
public static const float2 LENS_LATTICE_STEPS = float2(0.7548776662, 0.5698402910);

// The frame the sensor sits in: the optical axis and two unit vectors spanning the plane it is
// perpendicular to, read off the same unprojection the pinhole rays come from. View bobbing is baked into
// the projection, so it is inside this frame instead of something the host has to describe. Which way
// `right` and `up` face is irrelevant to a circular aperture, so only the unprojection's own x axis is
// orthogonalised against the axis and `up` is taken from the cross product.
public void sensorFrame(out float3 axis, out float3 right, out float3 up) {
    float4 centreNearH = mul(worldPush.invViewProj, float4(0.0, 0.0, 1.0, 1.0));
    float4 centreFarH = mul(worldPush.invViewProj, float4(0.0, 0.0, 0.0, 1.0));
    float4 edgeNearH = mul(worldPush.invViewProj, float4(1.0, 0.0, 1.0, 1.0));
    float3 centreNear = centreNearH.xyz / centreNearH.w;
    axis = normalize((centreFarH.xyz / centreFarH.w) - centreNear);
    float3 toEdge = (edgeNearH.xyz / edgeNearH.w) - centreNear;
    right = normalize(toEdge - axis * dot(toEdge, axis));
    up = cross(axis, right);
}

// One point of a circular aperture, in units of its radius. Successive frames of one pixel walk the
// plastic-number lattice, which covers the aperture evenly instead of clumping — that is the difference
// between a temporal filter averaging a bokeh disc and one averaging a few scattered dots inside it. The
// lattice is rotated by a phase drawn from the pixel's own stream, because one shared point across the
// whole screen translates the image rather than defocusing it; the frame index is masked because fp32
// stops resolving a fraction of it long before a session ends, and a wrapped walk is still an even one.
public float2 sampleLensDisk(uint frameIndex, uint pixelSeed) {
    uint phaseSeed = pcg(pixelSeed);
    float phaseX = rndf(phaseSeed);
    float phaseY = rndf(phaseSeed);
    float2 u = frac(float(frameIndex & 1023u) * LENS_LATTICE_STEPS + float2(phaseX, phaseY));
    // Area-preserving map of the unit square onto the unit disk: sqrt on the radius, so the samples hold a
    // constant density per unit of area instead of crowding the centre.
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return r * float2(cos(phi), sin(phi));
}

// Cosine-weighted hemisphere sample about n (Malley's method). For a Lambertian BRDF this is the
// importance-sampling match: BRDF*cos/pdf = (albedo/PI)*cos / (cos/PI) = albedo, so the continuation
// throughput is just *= albedo with no PI/cos bookkeeping left over.
public float3 cosineDir(float3 n, inout uint s) {
    float u1 = rndf(s);
    float u2 = rndf(s);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float3 t = normalize(abs(n.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    return normalize(local.x * t + local.y * b + local.z * n);
}

// Soft shadows: sample a direction within the light's SQUARE angular extent about `axis`. MC's sun/moon
// are square quads, so the NEE shadow ray samples a square (not a cone) — the same square, in the same
// horizon-levelled tangent frame (sky.celestialSquareFrame), that world.rmiss draws the visible body in.
// Averaged over frames by DLSS-RR this yields soft penumbrae that widen with occluder distance
// (contact-hardening) for free. halfAngle <= 0 ⇒ exact direction (hard).
public float3 sampleSquare(float3 axis, float halfAngle, inout uint s) {
    float3 right;
    float3 up;
    celestialSquareFrame(axis, right, up);
    float t = tan(halfAngle);
    float u = (rndf(s) * 2.0 - 1.0) * t;
    float v = (rndf(s) * 2.0 - 1.0) * t;
    return normalize(axis + u * right + v * up);
}

// Sample a GGX visible-normal (VNDF, Heitz 2018) about geometric normal n for view ve, returning the
// sampled microfacet normal (half-vector) in world space. Paired with the separable Smith term, the
// importance-sampling weight reduces to F * G1(NdotL) (applied by the caller).
public float3 sampleGGXVNDF(float3 n, float3 ve, float alpha, inout uint s) {
    float3 t = normalize(abs(n.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    float3 Ve = float3(dot(ve, t), dot(ve, b), dot(ve, n)); // view in tangent space (z = n)
    float a = alpha;
    float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);
    float u1 = rndf(s);
    float u2 = rndf(s);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float ss = 0.5 * (1.0 + Vh.z);
    p2 = (1.0 - ss) * sqrt(1.0 - p1 * p1) + ss * p2;
    float3 Nh = p1 * T1 + p2 * T2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * Vh;
    float3 Ne = normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z))); // microfacet normal, tangent space
    return normalize(Ne.x * t + Ne.y * b + Ne.z * n);                 // -> world space
}
