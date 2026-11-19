// Participating media: the tint-to-extinction mappings and the depth-2 medium stack the dielectric
// interface pushes and pops. Depends on core only.

import world_common;
import world_core;

// Per-channel Beer–Lambert absorption of a water body. The coefficient is Photon's underwater absorption
// (0.20 red, 0.08 green, 0.04 blue per block in Rec.709) filtered into working space by the same matrix
// the Rayleigh coefficient uses, so red is swallowed roughly four times as fast as blue and depth reads as
// teal rather than as grey. That chromatic split is the whole point: a single grey sigma at this magnitude
// turns every submerged view into flat murk.
//
// `opacity` scales the absorption, so 0 reads as fully clear water. The painted value is Photon's look at
// the default opacity of 3, which keeps existing configs on the same murk level they had before.
public static const float3 WATER_ABSORPTION = float3(0.151676, 0.087885, 0.047682);
public static const float WATER_OPACITY_DEFAULT = 3.0;
// The biome tint only bends the channel ratio: it multiplies a unit-mean hue weight, so a blue ocean body
// loses blue last and a swamp body stays browner, while the overall murk stays the opacity's job.
public static const float WATER_BIOME_CONTRAST = 0.6;
public float3 waterExtinction(float3 tint, float opacity) {
    float3 lift = float3(1.0, 1.0, 1.0) - clamp(tint, 0.0, 1.0);
    float meanLift = max((lift.r + lift.g + lift.b) / 3.0, 1.0e-3);
    float3 hue = lerp(float3(1.0, 1.0, 1.0), lift / meanLift, WATER_BIOME_CONTRAST);
    return WATER_ABSORPTION * hue * (max(opacity, 0.0) / WATER_OPACITY_DEFAULT);
}

// A non-water dielectric's tint is a filter over one block of travel rather than a biome water colour, so
// it maps to extinction by inverting Beer-Lambert over that reference distance: transmittance t after one
// block means sigma = -ln(t). `transmission` scales that transmittance, so it keeps the sense the
// material's authored factor has everywhere else — 1 passes the tint through unchanged, 0 is opaque.
// Clamped away from zero so a fully saturated texel stays finite instead of producing an infinite sigma.
public static const float VOLUME_TINT_REFERENCE_DISTANCE = 1.0; // blocks
public float3 volumeExtinction(float3 tint, float transmission) {
    float3 blockTransmittance = clamp(clamp(tint, 0.0, 1.0) * clamp(transmission, 0.0, 1.0),
            1.0e-3, 1.0);
    return -log(blockTransmittance) / VOLUME_TINT_REFERENCE_DISTANCE;
}

// ---- Participating medium the path is currently inside.
//
// Refraction needs the RELATIVE index across an interface, so the renderer has to know what it is
// travelling through, not just what it is hitting. `ior` drives Snell/Fresnel; `extinction` drives the
// per-segment Beer-Lambert attenuation.
//
// The stack is depth 2 (current + the one it will return to) held in named fields, NOT an array. A
// dynamically indexed local array lands in scratch memory, and this raygen is already register-bound —
// paying an occupancy hit for nesting that Minecraft does not produce would be a bad trade. Depth 2
// covers air->water->glass and air->glass->water, which is the realistic worst case; anything deeper
// degrades to air on the way out, and because `entering` is re-derived per face from geometry rather
// than toggled, the path re-synchronises at the next crossing instead of staying corrupted.
public struct Medium {
    public float ior;
    public float3 extinction;
    // Scattering of the medium itself (1/block, same unit as extinction). Air and every non-water
    // dielectric carry zero; water's sigmaS is DERIVED from its extinction and worldPush.water1.x —
    // see waterScattering below. Because the derivation is a pure function of the packed extinction
    // and a per-frame global, no scattering lane had to be added to the 48-byte PackedPathSegment:
    // the resumed record re-derives instead of unpacking.
    public float3 scattering;
    // Henyey–Greenstein anisotropy of `scattering`.
    public float phaseG;
    public bool water; // drives the wave-refraction caustic on submerged receivers; nothing else is water-specific
};

public struct MediumStack {
    public Medium current;
    public Medium outer;
};

public Medium airMedium() {
    Medium m;
    m.ior = 1.0;
    m.extinction = float3(0.0, 0.0, 0.0);
    m.scattering = float3(0.0, 0.0, 0.0);
    m.phaseG = 0.0;
    m.water = false;
    return m;
}

public MediumStack makeMediumStack(Medium start) {
    MediumStack s;
    s.current = start;
    s.outer = airMedium();
    return s;
}

public void mediumPush(inout MediumStack stack, Medium entered) {
    stack.outer = stack.current;
    stack.current = entered;
}

// True when the path is travelling through something denser than air, i.e. a dielectric face reached now can
// be a legitimate exit. A face that reads as an exit while the stack still carries air was never entered —
// vanilla culls the entry quad, a bias step started past it, or the path began inside the cell — so there is
// no crossing to refract through at all (see PAYLOAD_DIELECTRIC_THIN).
public bool mediumInsideDenser(MediumStack stack) {
    return stack.current.ior > 1.0 + 1e-3;
}

public void mediumPop(inout MediumStack stack) {
    stack.current = stack.outer;
    stack.outer = airMedium();
}

// The water body's own scattering: `water1.x` is the in-scatter's single-scatter fraction sigmaS/sigmaT,
// so scattering = fraction * extinction keeps the two coefficients energy-consistent by construction
// (sigmaS <= sigmaT for every channel, with equality at a fraction of 1). Because the fraction is a
// per-frame global rather than a per-segment quantity, it is DERIVED from the packed extinction wherever
// the medium is materialised — at the interface (makeDielectricMedium) and again when a packed record
// resumes (segment.unpackPathSegment) — instead of being carried in the 48-byte path record. A zero
// fraction yields zero scattering, so a frame with the water fog off leaves the record byte-identical to
// one from before the fog existed.
public float3 waterScattering(float3 extinction) {
    float fraction = worldPush.water1.x;
    return fraction > 0.0 ? fraction * extinction : float3(0.0, 0.0, 0.0);
}

// Water's absorption is the calibrated per-block coefficient bent by the biome tint's channel ratio; any
// other volume dielectric's tint is a filter over a reference block of travel. Both end up as per-channel
// extinction; `waterOpacity` (worldPush.waterOpacity) scales only the water branch.
public Medium makeDielectricMedium(bool isWater, float ior, float3 tint, float transmission,
        float waterOpacity) {
    Medium m;
    m.ior = ior;
    m.extinction = isWater ? waterExtinction(tint, waterOpacity)
            : volumeExtinction(tint, transmission);
    m.scattering = isWater ? waterScattering(m.extinction) : float3(0.0, 0.0, 0.0);
    m.phaseG = isWater ? worldPush.water1.y : 0.0;
    m.water = isWater;
    return m;
}
