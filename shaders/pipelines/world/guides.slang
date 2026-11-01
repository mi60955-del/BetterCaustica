// DLSS-RR guide state and everything that resolves it: the gv_* captures, the foreground specular
// surface and its reflection motion vector, and the image writes.
// PRIMARY PASS ONLY. Depends on trace, water and segment.

// Primary-visibility guide attributes. Pass A traces exactly one sample per pixel; a split reflection
// leaf supplies the spec endpoint and its transmission leaf may replace the ordinary tuple.

import world_common;
import world_core;
import bindings;
import math;
import medium;
import water;
import trace;

public static float3 gv_normal = float3(0.0, 0.0, 0.0);
public static float3 gv_albedo = float3(0.0, 0.0, 0.0);
// Fraction of the camera ray the air volume covers before the guide endpoint. The guides describe ONE
// layer per pixel, so a fog-dominated pixel has to describe the fog: its radiance is not the surface's
// shaded colour, and handing RR the surface's normal, roughness, albedo and mirror motion tells the
// reconstruct that a smooth medium is material — it keeps the medium's own estimator grain as if it were
// texture detail, and it reprojects that grain with the wrong motion. `writeGuides` blends the whole tuple
// toward the medium by this factor instead.
public static float gv_fogOpacity = 0.0;
// The diffuse-albedo guide's alpha lane is not consumed by DLSS-RR. Preserve whether the visible
// guide endpoint emits so downstream exposure metering can cap large lava/glowstone populations
// without another material-ID image.
public static bool gv_emissive = false;
public static float gv_rough = 0.0;
public static float3 gv_hitCamRel = float3(0.0, 0.0, 0.0);       // primary-surface hit position relative to the current camera
#ifdef CAUSTICA_INTERFACE_DEBUG
// Diagnostic lanes for writeGuides: the primary dielectric interface's decision (0 filter, 1/3 volume
// entry, 2/3 volume exit, 1 total internal reflection), the medium it was in, and the hit's material id.
public static float gv_dbgDecision = 0.0;
public static float gv_dbgIor = 0.0;
public static float gv_dbgMaterial = 0.0;
#endif
// World-space displacement of the motion-guide surface since the previous frame (0 for static
// terrain/sky, an entity's per-vertex/object motion for a dynamic hit).
public static float3 gv_motionObjDisp = float3(0.0, 0.0, 0.0);
// Transmission replaces the ordinary guide tuple with the visible destination, so RR's specular inputs
// (specular albedo + reflection MV) need the foreground glass/water interface kept separately. This is
// the surface the reflection physically lives on; for an opaque primary hit it is that hit itself.
public struct SpecSurface {
    public float3 camRel;         // interface position relative to the current camera
    public float3 normal;         // FP32: reflection reprojection amplifies normal quantization at distance
    public float3 previousNormal; // the same normal one frame ago; equal to `normal` unless the surface animates
    public float3 biasNormal;     // geometric normal, used only to offset a ray origin off the surface
    public float3 motionPrev;     // current-minus-previous displacement of the reflecting surface
    public float roughness;
    public float3 albedo;         // specular albedo fed to DLSS-RR for demodulation (0 = pure diffuse)
};
public static SpecSurface gv_spec = {};
// Set when the foreground specular surface is a wave-perturbed water face. Its normal can aim the
// mirror below the plane, and the radiance path lifts such a direction to the grazing floor
// (water.slang's waterReflectedDir); the probe below has to lift it identically, or the reflected
// image and the motion vector it is reprojected with describe two different scenes on exactly the
// pixels where the lift fires.
public static bool gv_specWaveMirror = false;
// The share of the light leaving this pixel that left along the mirror lobe: 1 for a metal, which has no
// diffuse lobe to compete with it, the Fresnel term of the face for a dielectric interface (water, ice,
// a pane), and Schlick's reflectance for an opaque non-metal, which is what makes a floor a mirror at
// grazing incidence. Everything else is 0 and reads as "this pixel shows a surface". The lens's focus
// meter reads it through resolveSpecularGuides: a pixel is showing a reflected image only where that
// image is a real share of what leaves it, and one comparison covers all three cases because the physics
// already puts the answer in the number.
public static float gv_reflectShare = 0.0;

// Above this share a reflected image is what the pixel shows rather than a guest on the surface. A quarter
// is where a reflection stops being a sheen and starts being the picture: a lake's far half clears it (a
// grazing water face keeps most of its Fresnel term), the near water under the player's feet does not — it
// is showing the bottom — and an ordinary floor only clears it at the incidence where it genuinely becomes
// a mirror.
public static const float REFLECT_IMAGE_SHARE = 0.25;
// The far end of the focus meter's depth image. A reflection that escapes to the sky images the sky, so
// it belongs at the far stop rather than at the mirror, and this is what it is written as: the meter
// clamps everything past its own far distance to that stop, and the blur ceiling clamps to the focal
// plane, which is the same "behind everything" answer without a value that could overflow the multiply.
public static const float REFLECT_IMAGE_MAX = 1.0e6;

// Static surfaces reuse one normal for all three roles. Water overrides them (wave-displaced shading
// normal now and last frame, flat geometric normal for the bias) via the six-argument form.
public SpecSurface makeSpecSurface(float3 camRel, float3 normal, float3 previousNormal,
                            float3 biasNormal, float3 motionPrev, float roughness, float3 albedo) {
    SpecSurface s;
    s.camRel = camRel;
    s.normal = normal;
    s.previousNormal = previousNormal;
    s.biasNormal = biasNormal;
    s.motionPrev = motionPrev;
    s.roughness = roughness;
    s.albedo = albedo;
    return s;
}

public SpecSurface makeSpecSurface(float3 camRel, float3 normal, float roughness, float3 albedo) {
    return makeSpecSurface(camRel, normal, normal, normal,
            float3(0.0, 0.0, 0.0), roughness, albedo);
}

public SpecSurface makeSpecSurface(float3 camRel, float3 normal, float3 motionPrev,
                            float roughness, float3 albedo) {
    return makeSpecSurface(camRel, normal, normal, normal, motionPrev, roughness, albedo);
}
// angle can land behind the eye even though the reflector itself is comfortably in view.
public float2 projectPrevNdc(float3 worldPos, out bool valid) {
    float4 clip = mul(worldPush.prevViewProj, float4(worldPos - worldPush.camOffset + worldPush.camDelta, 1.0));
    valid = clip.w > 0.0;
    return valid ? clip.xy / clip.w : float2(0.0, 0.0);
}

// Previous-frame screen position of a planar reflection. The reflected image of a world point P seen
// in a planar reflector is the MIRROR image V = mirror(P) across the surface plane: the eye sees V
// along a straight line, so the reflection appears at proj(V). Project the mirror image directly.
// Gotcha: intersecting a reflection ray with the surface plane divides by dot(mirrorDir, n), which
// explodes at grazing angles and produces non-zero MV in a static scene. Mirroring is division-free, so
// static MV is bit-exact zero at any incidence. The reflector is assumed static (terrain / water);
// reflectedMotionPrev carries the reflected content's own displacement.
public float2 previousReflectionNdc(float3 surfacePos, float3 surfaceNormal, float3 surfaceMotionPrev,
                             float3 reflectedWorldPos, float3 reflectedMotionPrev, out bool valid) {
    float3 n = normalize(surfaceNormal); // mirror(P) is orientation-independent, so no viewer flip needed
    float3 previousSurfacePos = surfacePos - surfaceMotionPrev;
    float3 prevHit = reflectedWorldPos - reflectedMotionPrev;
    float3 mirroredHit = prevHit - 2.0 * n * dot(prevHit - previousSurfacePos, n);
    return projectPrevNdc(mirroredHit, valid);
}

// Reflection guides remain owned by the physical foreground interface even when transmission later
// replaces the ordinary guide tuple. The same non-SER probe resolves both motion and, for exact
// mirrors, the reflected terminal's diffuse content. Destinations without a stable diffuse albedo keep
// the foreground material reflectance: sky, emissive surfaces, metals, and another dielectric/mirror
// would otherwise give RR a zero or recursively defined demodulation signal.
//
// `imageDepth` is where the picture this pixel shows sits when that picture is a reflection: the axial
// distance from the eye to the reflected image, or 0 where this pixel is not showing one. It rides this
// probe because the probe already answers where the mirror ray lands, so the lens's focus meter pays no
// ray of its own for it.
public float2 resolveSpecularGuides(SpecSurface surface, float3 primaryDir,
                             float2 size, float primaryConeSpread,
                             out float3 specAlbedo, out float imageDepth) {
    specAlbedo = surface.albedo;
    imageDepth = 0.0;
    if (length(surface.normal) < 0.5
            || max(surface.albedo.r, max(surface.albedo.g, surface.albedo.b)) <= 0.001
            || surface.roughness > SPEC_MOTION_ALPHA_MAX) {
        return float2(0.0, 0.0);
    }

    float3 surfacePos = surface.camRel + worldPush.camOffset;
    float3 n = normalize(surface.normal);
    float3 specDir = gv_specWaveMirror ? waterReflectedDir(primaryDir, n, surface.biasNormal)
            : reflect(primaryDir, n);
    // Hoisted out of the mirror-motion return below: the interface's own axial distance is the scale the
    // reflected image's distance is expressed in.
    float4 curClip = mul(worldPush.curViewProj, float4(surface.camRel, 1.0));

    // Pass A stops radiance traversal at the first split, so reflection motion keeps one dedicated,
    // deterministic guide probe. CULL_SPECULAR must match the mask the reflected radiance ray will take
    // in the indirect pass: if the probe excluded particles while the reflected image includes them, the
    // motion vectors here would reproject a different scene than the one being denoised.
    // tmin 0.0, exactly like the radiance reflection it mirrors: the offset origin already clears the
    // reflector, and any extra tmin would let the guide miss a surface the radiance ray still hits.
    traceGuide(CULL_SECONDARY | CULL_SPECULAR,
            offsetSurfaceOrigin(surfacePos, surface.biasNormal, specDir, SURF_BIAS),
            0.0, specDir, 10000.0,
            max(length(surface.camRel) * primaryConeSpread, RAY_CONE_MIN_WIDTH),
            max(primaryConeSpread, RAY_CONE_MIN_SPREAD));
    float3 reflectedHit;
    float3 reflectedMotionPrev;
    // Distance along the mirror ray to whatever the reflection shows; 0 is the miss sentinel the probe
    // leaves, and a miss means the reflection is the sky.
    float reflectedRange = payload.hitT;
    if (payload.hitT > 0.0) {
        reflectedHit = surfacePos + specDir * payload.hitT;
        uint reflectedMaterial = payloadMaterial();
        // Closest-hit writes motionPrev for every material. Terrain/water are zero; dynamic opaque and
        // dielectric entities carry their per-vertex/object displacement.
        reflectedMotionPrev = payload.motionPrev;
        if (surface.roughness <= MIRROR_ALPHA_MAX
                && (reflectedMaterial == MATERIAL_OPAQUE
                    || reflectedMaterial == MATERIAL_PARTICLE)
                && payloadEmission() <= 0.0) {
            float3 reflectedAlbedo = payloadAlbedo();
            float3 reflectedDiffuse = reflectedMaterial == MATERIAL_PARTICLE
                    ? reflectedAlbedo
                    : reflectedAlbedo * (1.0 - clamp(payloadMetalness(), 0.0, 1.0));
            if (max(reflectedDiffuse.r, max(reflectedDiffuse.g, reflectedDiffuse.b)) > 0.001) {
                specAlbedo = surface.albedo * reflectedDiffuse;
            }
        }
    } else {
        reflectedHit = surfacePos + specDir * 1.0e6;
        reflectedMotionPrev = float3(0.0, 0.0, 0.0);
        reflectedRange = 1.0e6;
    }
    // A plane reflector images a point at |image - eye| = (eye to mirror) + (mirror to point), and the eye
    // sees that image down the very ray it is looking along, so the image's axial distance is the
    // interface's own scaled by how much further the path went: |CM| + r against |CM|. No second
    // projection, no angle, and it stays exact for a wave-perturbed face, because the mirror symmetry that
    // produces the image does not care which way the normal leaned to send the ray there.
    float reach = length(surface.camRel);
    if (gv_reflectShare >= REFLECT_IMAGE_SHARE && curClip.w > 0.0 && reach > 1.0e-4) {
        imageDepth = min(curClip.w * (1.0 + reflectedRange / reach), REFLECT_IMAGE_MAX);
    }
    float3 previousN = length(surface.previousNormal) >= 0.5
            ? normalize(surface.previousNormal) : n;
    bool prevValid;
    float2 prevNdc = previousReflectionNdc(
            surfacePos, previousN, surface.motionPrev,
            reflectedHit, reflectedMotionPrev, prevValid);
    // The mirror's current screen position is where the reflector itself projects. For a primary hit that
    // is the pixel it was seen through -- the interface lies on the sampled ray -- but the reflector is
    // kept as its own point so the vector describes the surface the reflection physically lives on rather
    // than whatever endpoint the ordinary tuple later moved to.
    return prevValid && curClip.w > 0.0
            ? (prevNdc - curClip.xy / curClip.w) * 0.5 * size : float2(0.0, 0.0);
}

public void setTransmissionGuide(float3 hitCamRel, float3 motionPrev, float3 normal,
                          float roughness, float3 diffuseAlbedo, bool emissive) {
    gv_hitCamRel = hitCamRel;
    gv_motionObjDisp = motionPrev;
    gv_normal = normal;
    gv_rough = roughness;
    gv_albedo = diffuseAlbedo;
    gv_emissive = emissive;
}

// Deterministic ordinary guide behind the first transmitted interface. This is guide-only work:
// radiance reflection/transmission continuations are queued at the first split and traced by Pass B.
public void resolveTransmissionGuide(float3 surfacePos, float3 transmittedDir,
                              float3 surfaceBiasNormal, MediumStack medium, float rayBias,
                              float rayConeWidth, float rayConeSpread, float3 guideFilter,
                              float interfaceRoughness) {
    if (dot(transmittedDir, transmittedDir) <= 0.0) return;

    float3 direction = normalize(transmittedDir);
    float3 ro = offsetSurfaceOrigin(surfacePos, surfaceBiasNormal, direction, rayBias);

    // The camera interface consumed bounce 0; the remaining configured bounce budget is the natural
    // cap for deterministic guide crossings too.
    for (uint crossing = 0u; crossing < worldPush.maxBounces; ++crossing) {
        // tmin 0.0, matching the radiance trace this probe stands in for. The mesher recesses a
        // translucent face by TRANSLUCENT_INSET (2e-4) and the transmitted ray restarts
        // INSET_TRANSMIT_BIAS (1e-4) into that gap, so a partial block or block-entity face sitting
        // flush against the glass is 1e-4 from this origin. RAY_TMIN (1e-3) skipped it: the guide then
        // described whatever lay behind while the color was the block in front, and DLSS-RR turned that
        // block into a translucent ghost.
        traceGuide(CULL_PRIMARY, ro, 0.0, direction, 10000.0,
                rayConeWidth, rayConeSpread);
        if (payload.hitT <= 0.0) {
            setTransmissionGuide((ro + direction * 1.0e6) - worldPush.camOffset,
                    float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), 1.0,
                    guideFilter * SKY_DIFF_ALBEDO, false);
            return;
        }

        uint material = payloadMaterial();
        float3 interfacePos = ro + direction * payload.hitT;
        if (material == MATERIAL_OPAQUE || material == MATERIAL_PARTICLE) {
            // What sharpens or blurs this radiance is the interface it crossed, not the surface behind
            // it: a flat glass face transmits a delta, so the destination's own roughness describes a
            // reflection this pixel never shows. Reporting the destination's roughness instead tells
            // the reconstructor to expect texture-scale variance exactly where the path is sharp, and
            // it averages the texture behind the glass into a smear.
            float endpointRoughness = material == MATERIAL_PARTICLE
                    ? 1.0 : interfaceRoughness;
            float3 hitAlbedo = payloadAlbedo();
            float3 endpointAlbedo = material == MATERIAL_PARTICLE
                    ? hitAlbedo
                    : hitAlbedo * (1.0 - clamp(payloadMetalness(), 0.0, 1.0));
            setTransmissionGuide(interfacePos - worldPush.camOffset, payload.motionPrev,
                    payloadNormal(), endpointRoughness, guideFilter * endpointAlbedo,
                    payloadEmission() > 0.0);
            return;
        }
        if (material != MATERIAL_WATER && material != MATERIAL_DIELECTRIC) return;

        bool isWater = material == MATERIAL_WATER;
        bool entering = payloadDielectricEntering();
        float3 geometricNormal = normalize(payloadNormal());
        float3 interfaceNormal = geometricNormal;
        if (isWater && (worldPush.flags & 16u) != 0u) {
            float waterFootprint = rayConeWidth
                    / max(abs(dot(-direction, geometricNormal)), 0.2);
            interfaceNormal = applyWaterWaves(geometricNormal,
                    float3(interfacePos.xz + worldPush.waterAnchor.xy, interfacePos.y),
                    direction, worldPush.waterParams.w, waterFootprint);
        }

        float transmission = clamp(payloadTransmission(), 0.0, 1.0);
        Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0),
                                              payloadAlbedo(), transmission,
                                              worldPush.waterOpacity);
        // A face that bounds no volume has the same medium on both sides (see PAYLOAD_DIELECTRIC_THIN): it
        // filters the ray and leaves the stack alone, so no critical angle is invented at it. This has to
        // agree with raygen's interface or the guide would describe a crossing the radiance path never takes.
        bool thin = !isWater && (payload.flags & PAYLOAD_DIELECTRIC_THIN) != 0u;
        // Same rule as raygen: an exit needs an entry, and this traversal must agree with the radiance path.
        if (!thin && !entering && !mediumInsideDenser(medium)) {
            thin = !isWater;
        }
        float etaT = thin || entering ? entered.ior : medium.outer.ior;
        // A thin face still bends the ray exactly as an entry would: that bend is the refraction a pane
        // gives, and with the near-side index it can never reach the critical angle.
        float3 nextDirection = refract(direction, interfaceNormal, medium.current.ior / etaT);
        if (dot(nextDirection, nextDirection) <= 0.0) {
            // Never let a TIR reflection become ordinary diffuse/depth.
            setTransmissionGuide(interfacePos - worldPush.camOffset,
                    isWater ? float3(0.0, 0.0, 0.0) : float3(payload.motionPrev),
                    interfaceNormal, 0.0, float3(0.0, 0.0, 0.0), false);
            return;
        }
        if (!isWater && (thin || entering)) {
            guideFilter *= payloadAlbedo();
        }
        if (!thin) {
            if (entering) {
                mediumPush(medium, entered);
            } else {
                mediumPop(medium);
            }
        }
        direction = normalize(nextDirection);
        ro = offsetSurfaceOrigin(interfacePos, geometricNormal, direction,
                isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS);
    }
}

// Resolve the captured gv_* guide state into the DLSS-RR guide images after Pass A queues continuations.
public void writeGuides(int2 pix, float3 primaryDir, float2 size, float primaryConeSpread) {
    // Reflection remains owned by the foreground interface even when a dielectric replaced the ordinary
    // tuple with the transmitted destination.
    float3 specAlbedo;
    float imageDepth;
    float2 specMotion = resolveSpecularGuides(
            gv_spec, primaryDir, size, primaryConeSpread, specAlbedo, imageDepth);

    // Hardware (non-linear, reversed-Z) depth for DLSS-RR + Frame Generation: project the camera-relative
    // primary hit through the forward view-projection and take ndc z/w — exactly the value a rasterizer
    // would write. Reversed-Z (near=1, far=0), so DLSS gets the DepthInverted flag; sky's far hit -> ~0.
    // Clear transmission uses destination depth so the entire ordinary tuple describes one layer.
    float4 curClip = mul(worldPush.curViewProj, float4(gv_hitCamRel, 1.0));
    float depth = curClip.w > 0.0 ? curClip.z / curClip.w : 0.0;

    // The lens ceiling the next frame needs, in the one unit its geometry works in: the axial distance
    // this pixel's pinhole ray travelled. A surface's blur disc is radius * |1/depth - 1/focus| in
    // tangent space, so bounding the disc means bounding the radius against the depth the pixel actually
    // showed -- and this frame's depth is precisely what the bound is for, so it cannot be consulted
    // before the ray is built. Two things read it: the per-pixel ceiling, and the autofocus meter that
    // decides where the focal plane sits next. Both need it only while the lens is on.
    //
    // With the reflection-focus switch up, a pixel whose mirror lobe is what it shows reports the
    // reflected image's distance instead, and that is the same number for both readers: the meter then
    // puts the focal plane on the thing behind the mirror, and the ceiling stops narrowing a mirror
    // pixel's lens as if it were a surface standing far in front of that plane.
    if (worldPush.dof.x > 0.0) {
        float focusDepth = (worldPush.flags & WORLD_FLAG_DOF_REFLECTION) != 0u && imageDepth > 0.0
                ? imageDepth : curClip.w;
        focusDepthCur[pix] = focusDepth > 0.0 ? focusDepth : 0.0;
    }

    // Motion vector: the guide point's own displacement on screen, its previous-frame projection minus
    // where it projects now. Both terms come from the same world point, so a static scene answers exactly
    // zero. Projecting the point rather than subtracting the jittered ray's ndc is the same answer while
    // the camera is a pinhole -- the hit lies on that ray, so its projection recovers the sub-pixel offset
    // -- and it is the only answer that stays right when the guide point is not on the sampled ray at all,
    // which is what a transmitted destination is. Scaled from NDC into render-pixel space.
    // For a dynamic surface, previous-frame position is the current point minus its world-space
    // displacement; subtracting gv_motionObjDisp de-cameras the MV.
    float4 prevClip = mul(worldPush.prevViewProj,
            float4(gv_hitCamRel + worldPush.camDelta - gv_motionObjDisp, 1.0));
    // Guard both perspective divides the way the depth above does. A guide point can land behind either
    // camera — most easily a transmitted destination seen at a grazing angle, or one that leaves the
    // frustum as the camera turns — and w at or below zero turns the divide into an arbitrarily large
    // vector that DLSS-RR reprojects with. No valid position exists in that case, so report no motion and
    // let RR fall back instead of handing it garbage.
    float2 motion = prevClip.w > 0.0 && curClip.w > 0.0
            ? (prevClip.xy / prevClip.w - curClip.xy / curClip.w) * 0.5 * size
            : float2(0.0, 0.0);

    // The medium owns this pixel to the extent the camera ray crosses it. Every guide RR uses to
    // classify the signal moves with that fraction: a camera-facing normal and full roughness (a
    // volume has no lobe), the fog's own albedo (the term RR demodulates against), and a specular
    // channel that fades out with the fog — including its motion, which is the MIRROR image's motion
    // and describes nothing about a volume sitting in front of the mirror. Leaving those guides on the
    // surface is what made the reconstruct treat the fog's dither as material detail and reproject it
    // along the reflection.
    float op = clamp(gv_fogOpacity, 0.0, 1.0);
    float3 endNormal = length(gv_normal) > 0.5 ? normalize(gv_normal) : -primaryDir;
    float3 blendedNormal = lerp(endNormal, -primaryDir, op);
    float3 guideNormal = length(blendedNormal) > 1.0e-4 ? normalize(blendedNormal) : -primaryDir;
    float guideRough = lerp(gv_rough, 1.0, op);
    float3 guideAlbedo = lerp(gv_albedo, max(worldPush.fogColor.rgb, float3(0.0)), op);
    float specWeight = 1.0 - op;

    // Guide buffers: first-hit or coherently replaced attributes consumed by the denoiser/DLSS-RR.
    // Depth deliberately stays the endpoint's: DLSS-RR is not its only reader — frame generation
    // reprojects with it and the exposure histogram weights its populations by it, so moving a
    // fog-dominated pixel's depth off the surface would trade this artifact for two others.
#ifdef CAUSTICA_INTERFACE_DEBUG
    // Diagnostic build: the albedo guide carries the interface decision instead of albedo, so one
    // screenshot of debug view 2 reads out what the dielectric interface saw at each pixel.
    // The albedo guide is ACEScg and debug view 2 maps it through acesCgToSrgb, whose Rec.709
    // matrix has coefficients above 1 and negative off-diagonals: written at full scale the lanes
    // both clip and mix, which makes the decision and exit cases indistinguishable. Quarter scale
    // keeps every channel inside [0, 1] so the three lanes stay invertible.
    guideAlbedo = float3(gv_dbgDecision, gv_dbgIor, gv_dbgMaterial) * 0.25;
#endif
    gNormal[pix] = float4(guideNormal, guideRough);
    gAlbedo[pix] = float4(guideAlbedo, gv_emissive ? 1.0 : 0.0);
    gDepth[pix] = depth;
    gMotion[pix] = motion;
    gSpecAlbedo[pix] = float4(specAlbedo * specWeight, 1.0);
    gSpecMotion[pix] = specMotion * specWeight;
}
