// Participating media: the tint-to-extinction mappings and the depth-2 medium stack the dielectric
// interface pushes and pops. Depends on core only.

// Per-channel Beer–Lambert extinction from a water body's biome tint (carried in the payload's albedo view
// for water hits, or worldPush.waterParams.xyz for the camera's own biome when starting submerged). A blue
// ocean tint (low red) absorbs red fastest → bluer with depth; swamp green-brown shifts the hue. The floor
// keeps even a white-tinted body very slightly absorbing so deep water never reads as clear vacuum.

import world_common;
import world_core;

public static const float WATER_DENSITY = 0.1;
public static const float3 WATER_ABSORB_FLOOR = float3(0.015, 0.010, 0.008);
// `opacity` scales the whole extinction (floor included), so 0 reads as fully clear water rather
// than as a floor-only residue. Above 1 the tint-driven channel spread (red dies fastest → blue
// water) is frozen at its 1.0 strength and only the achromatic mean keeps growing — scaling the
// spread too would leave blue as the last surviving channel and read as saturated dye instead of
// the grey-black of genuinely opaque water. Below 1 the whole curve scales, matching 1.0 exactly.
public float3 waterExtinction(float3 tint, float opacity) {
    float3 base = WATER_ABSORB_FLOOR
            + WATER_DENSITY * (float3(1.0, 1.0, 1.0) - clamp(tint, 0.0, 1.0));
    float mean = (base.r + base.g + base.b) / 3.0;
    float chromaScale = min(opacity, 1.0);
    return mean * opacity + (base - mean) * chromaScale;
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

public void mediumPop(inout MediumStack stack) {
    stack.current = stack.outer;
    stack.outer = airMedium();
}

// The water medium's own scattering: `turbidity` (worldPush.water1.x) is the single-scatter fraction
// sigmaS/sigmaT, so scattering = turbidity * extinction keeps the two coefficients energy-consistent
// by construction (sigmaS <= sigmaT for every channel, with equality at turbidity 1). Because the
// fraction is a per-frame global rather than a per-segment quantity, it is DERIVED from the packed
// extinction wherever the medium is materialised — at the interface (makeDielectricMedium) and again
// when a packed record resumes (segment.unpackPathSegment) — instead of being carried in the 48-byte
// path record. Zero turbidity yields zero scattering, so the packed record needs no change at all and
// the pre-water-fog packing stays byte-identical.
public float3 waterScattering(float3 extinction) {
    float turbidity = worldPush.water1.x;
    return turbidity > 0.0 ? turbidity * extinction : float3(0.0, 0.0, 0.0);
}

// Water's tint is a biome colour whose absorption is calibrated per block of depth; any other volume
// dielectric's tint is a filter over a reference block of travel. Both end up as per-channel extinction;
// `waterOpacity` (worldPush.waterOpacity) scales only the water branch.
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
