// Primary/guide pass of the wavefront renderer.
//
// Traces one camera radiance ray, captures DLSS-RR guides, and writes one or two resumable
// continuations per pixel. At the first eligible dielectric it queues both post-interface branches and
// stops; every later radiance trace belongs to Pass B. Deterministic guide probes may continue through
// refraction, but Pass A does no NEE, RIS, SSS, or BSDF sampling.
//
// Perspective setup: invViewProj = inverse(proj * viewRot) maps clip to camera-relative world. Primary
// rays are reconstructed from two unprojected points (near and far plane) rather than assuming the eye
// is at the origin, because Minecraft bakes view bobbing into the projection, so the effective eye comes
// from the matrix. camOffset shifts camera-relative space into the terrain's rebased coordinates. Depth
// is reversed-Z (near=1, far=0).
//
// Depth of field puts its lens on that near plane. The bobbing eye is not a point these unprojected
// coordinates can name, while the near plane is, and sliding a lens sample along its own pixel ray tilts
// that ray only by aperture * nearDistance / focus^2.
//
// Pass A deliberately uses ordinary TraceRay. Its short, visually coherent direct/interface walk is
// latency-sensitive and does not amortize an SER reorder barrier.
import world_common;
import world_core;
import bindings;
import math;
import medium;
import segment;
import water;
import trace;
import volume;
import sky;
import guides;

// Perf switch: true by default, false when built with -DCAUSTICA_NO_VOLUME_PATH. That build compiles the
// volumetric scatter out of this raygen entirely; it is traced instead of this one on frames where every
// volumetric switch is off, because code that never executes still costs registers — 5 ms of the frame.
#ifndef CAUSTICA_NO_VOLUME_PATH
#define CAUSTICA_VOLUME_PATH true
#else
#define CAUSTICA_VOLUME_PATH false
#endif


// Volume debug image (debug views 10..13): r fog transmittance along the camera ray, g the fog
// in-scatter's luminance, b the cloud coverage the camera ray reads off the cloud plane, a the
// cloud-shadow transmittance the sun leg from the camera would ride. The cloud lanes live since
// Phase 4; a cloud-off frame leaves them at their inert defaults (0 and 1) through the same push
// gates every other consumer uses. Gated by WORLD_FLAG_VOLUME_DEBUG, so the ordinary frame never
// touches this image.
void writeVolumeDebug(int2 pix, float3 ro, float3 rd, float t, bool allowShadowCull) {
    if (!CAUSTICA_VOLUME_PATH) {
        volumeDebug[pix] = float4(1.0, 0.0, 0.0, 1.0);
        return;
    }
    float fogT;
    float3 insc = fogSegmentInScatter(ro, rd, t, pix, allowShadowCull, fogT);
    float cloudDensity = 0.0;
    float cloudShadow = 1.0;
    if ((worldPush.flags & WORLD_FLAG_CLOUD) != 0u && worldPush.cloud0.x > 0.0) {
        float tPlane = (cloudBoxMid(worldPush) - ro.y) / rd.y;
        if (tPlane > 0.0) {
            cloudDensity = cloudCoverageWorld(ro.xz + rd.xz * tPlane);
        }
        if (celestialLight.halfAngle > 0.0) {
            cloudShadow = cloudShadowTransmittance(ro, celestialLight.dir);
        }
    }
    volumeDebug[pix] = float4(fogT, luminance(insc), cloudDensity, cloudShadow);
}

// Pass A's camera segment walk. `guidesOnly` runs the same walk for a second ray purely to land the
// gv_* guide state: it writes no radiance, so it skips the prefix (and with it the fog and shaft
// integrators, which is most of what the walk costs) and skips queueing the transmitted branch, which
// would otherwise overwrite the record the radiance pass already wrote.
public PathSegment tracePrimary(PathSegment seg,
                                DevicePtr<PackedPathSegment> queue,
                                uint splitRecord,
                                int2 pix,
                                bool volumeDebugOn,
                                bool guidesOnly,
                                out bool prefixInImage,
                                out uint nextRecord) {
    float3 ro = seg.ro;
    float3 rd = seg.rd;
    float3 throughput = seg.throughput;
    MediumStack medium = seg.medium;
    float rayConeWidth = seg.rayConeWidth;
    float rayConeSpread = max(seg.rayConeSpread, RAY_CONE_MIN_SPREAD);
    uint seed = seg.seed;
    bool showCelestial = seg.showCelestial;
    bool waterWaves = (worldPush.flags & 16u) != 0u;
    bool fogOn = (worldPush.flags & WORLD_FLAG_FOG) != 0u;
    bool shaftOn = (worldPush.flags & WORLD_FLAG_SHAFTS) != 0u;
    bool edgeFadeOn = (worldPush.flags & WORLD_FLAG_EDGE_FADE) != 0u;
    nextRecord = PATH_NO_NEXT;
    prefixInImage = false;
    // Per-pixel specular guide state: only the water interface below sets it.
    gv_specWaveMirror = false;
    // Reset with it: a pixel whose hit sets neither of the two surfaces below is showing a surface, not a
    // reflection, and the lens's focus meter reads the zero that leaves.
    gv_reflectShare = 0.0;

    {
        int bounce = seg.bounce;
        PathSegment terminal = makePathSegment(ro, rd, throughput, medium,
                                               rayConeWidth, rayConeSpread, seed, bounce,
                                               showCelestial);
        traceRadiance(CULL_PRIMARY, ro, 0.0, rd, 10000.0,
                      showCelestial, rayConeWidth, rayConeSpread, true);

        // The volume debug image rides on the camera ray's own trace, before the guide probes below
        // overwrite the payload's hit distance.
        if (volumeDebugOn) {
            writeVolumeDebug(pix, ro, rd, payload.hitT >= 0.0 ? payload.hitT : 10000.0,
                    payload.hitT >= 0.0);
        }

        if (payload.hitT < 0.0) {
            if (bounce == 0) {
                gv_normal = float3(0.0, 0.0, 0.0);
                gv_albedo = SKY_DIFF_ALBEDO;
                gv_emissive = false;
                gv_rough = 1.0;
                gv_hitCamRel = rd * 1.0e6;
                gv_motionObjDisp = float3(0.0, 0.0, 0.0);
                gv_spec = makeSpecSurface(gv_hitCamRel, gv_normal, 1.0, SKY_SPEC_ALBEDO);
            }
            return terminal;
        }

        float3 n = payloadNormal();
        float3 hitPos = ro + rd * payload.hitT;
        uint material = payloadMaterial();
        if (material != MATERIAL_WATER && material != MATERIAL_DIELECTRIC) {
            float rough = material == MATERIAL_PARTICLE
                    ? 1.0 : clamp(payloadRoughness(), 0.0, 1.0);
            bool exactSpecular = material == MATERIAL_OPAQUE && isDeltaAlpha(rough);
            rough = exactSpecular ? 0.0 : rough;
            float metal = material == MATERIAL_PARTICLE
                    ? 0.0 : clamp(payloadMetalness(), 0.0, 1.0);
            float3 hitAlbedo = payloadAlbedo();
            float3 diffAlb = material == MATERIAL_PARTICLE
                    ? hitAlbedo : hitAlbedo * (1.0 - metal);
            if (bounce == 0) {
#ifdef CAUSTICA_INTERFACE_DEBUG
                // Opaque first hit: decision lane 0 marks "no interface here", and the other two lanes
                // report the medium the segment was written with and the material kind. Reading these is
                // what separates "the glass surface was misclassified" from "an opaque face is being
                // shaded while the stack still carries a dense medium".
                gv_dbgDecision = 0.0;
                gv_dbgIor = clamp((medium.current.ior - 1.0) / 0.6, 0.0, 1.0);
                gv_dbgMaterial = float(material) / 3.0;
#endif
                gv_normal = n;
                gv_hitCamRel = hitPos - worldPush.camOffset;
                gv_motionObjDisp = payload.motionPrev;
                gv_emissive = payloadEmission() > 0.0;
                if (material == MATERIAL_PARTICLE) {
                    gv_albedo = hitAlbedo;
                    gv_rough = 1.0;
                    gv_spec = makeSpecSurface(gv_hitCamRel, n, 1.0,
                                              float3(0.0, 0.0, 0.0));
                } else {
                    float3 v = -rd;
                    gv_albedo = diffAlb;
                    gv_rough = rough;
                    gv_spec = makeSpecSurface(gv_hitCamRel, n, float3(payload.motionPrev), rough,
                                              rrSpecularAlbedo(payload.f0, rough, dot(n, v)));
                    // How much of this pixel's light leaves along the mirror lobe. A metal has only that
                    // lobe, so all of it; a non-metal keeps Schlick's share, which is a few per cent face-on
                    // and most of the surface at grazing incidence, where any floor is a mirror. The focus
                    // meter reads it to decide whether this pixel shows a reflected image.
                    float3 mirrorShare = fresnelSchlick(clamp(dot(n, v), 0.0, 1.0), payload.f0);
                    gv_reflectShare = clamp(metal + (1.0 - metal)
                            * max(mirrorShare.r, max(mirrorShare.g, mirrorShare.b)), 0.0, 1.0);
                }
                // A pixel whose radiance is mostly not the surface converges its diffuse guide toward
                // mid-grey by how much of it isn't: RR demodulates against gv_albedo, and neither the fog in
                // front nor the render edge's fade dissolving the surface is that surface's albedo, so
                // leaving the surface's value there would brighten dark surfaces and darken bright ones
                // wherever either sits in front. Guarded so a frame with both off keeps the plain write.
                if (CAUSTICA_VOLUME_PATH && (fogOn || edgeFadeOn)) {
                    float airT = fogOn ? airTransmittance(ro, rd, payload.hitT) : 1.0;
                    float coverage = edgeFadeOn ? edgeFadeCoverage(ro, rd, payload.hitT) : 0.0;
                    float awayFromSurface = 1.0 - airT * (1.0 - coverage);
                    if (awayFromSurface > 0.0) {
                        gv_albedo = lerp(gv_albedo, float3(0.5, 0.5, 0.5), awayFromSurface);
                    }
                }
                // The shafts deliberately do NOT join this blend. Their term is high-frequency by
                // construction — it is gated on the shadow structure along the ray — so writing it
                // into the albedo guide tells the reconstruct it is material and it keeps the
                // estimator's own grain instead of averaging it out, which put a dashed lattice along
                // every block edge. Left as radiance it is denoised like any other light.
            }

            return terminal;
        }

        // This hit is consumed in pass A, so carry its travelled-medium absorption and cone growth into
        // the record for the next trace. The terminal hit above is deliberately not consumed here.
        if (any(medium.current.extinction > 0.0)) {
            throughput *= exp(-medium.current.extinction * payload.hitT);
        }

        // Fog over the camera-to-interface stretch. The terminal hit is consumed by Pass B, which
        // re-traces it and integrates the fog itself; the dielectric branches resume from the interface,
        // so this is the only place that stretch can be consumed for them. The transmittance rides the
        // branch throughputs; the in-scatter is radiance COMMON to both Fresnel branches and cannot ride
        // either record, so it is written into outImage here once and flagged: the indirect pass
        // accumulates on top of the stored value instead of overwriting it. The write is guarded so a
        // fog-off frame never touches outImage from this pass.
        prefixInImage = false;
        // A guide probe carries no radiance: the pass that owns this pixel already wrote its prefix, and
        // re-running the fog and shaft integrators for a result that is thrown away is most of what the
        // probe would otherwise cost.
        if (!guidesOnly) {
            // Both air terms are radiance common to the two Fresnel branches, so they are summed into
            // one prefix write. The transmittance only comes from the fog: the shafts are airlight and
            // deliberately attenuate nothing.
            float3 prefixRadiance = float3(0.0, 0.0, 0.0);
            bool hasPrefix = false;
            if (!any(medium.current.extinction > 0.0)) {
                if (CAUSTICA_VOLUME_PATH && fogOn) {
                    // The in-scatter call owns the segment's sigma (the indoor sky-exposure ladder may
                    // thin it), and its out fogT is that same thinned transmittance — attenuate
                    // throughput with it so the two agree instead of re-integrating at full density.
                    float prefixFogT;
                    prefixRadiance += fogSegmentInScatter(ro, rd, payload.hitT, pix, true, prefixFogT);
                    throughput *= prefixFogT;
                    hasPrefix = true;
                }
                if (CAUSTICA_VOLUME_PATH && shaftOn) {
                    float shaftShare;
                    prefixRadiance += shaftInScatter(ro, rd, payload.hitT, pix,
                            worldPush.shaft.x, worldPush.shaft.y, false, shaftShare);
                    hasPrefix = true;
                }
            }
            if (hasPrefix) {
                outImage[pix] = float4(clamp(prefixRadiance * worldPush.preExposure,
                        float3(0.0, 0.0, 0.0), float3(HALF_MAX, HALF_MAX, HALF_MAX)), 1.0);
                prefixInImage = true;
            }
            // The water medium's own scattering over the camera-to-interface stretch (the camera started
            // submerged): the same prefix doctrine as the fog branch above — the radiance is common to
            // both Fresnel continuations and can ride neither record, so it is written into outImage once
            // and flagged, and the indirect pass accumulates on top. The two branches never co-fire: the
            // fog stands down while any extinction is present, the water term needs the water medium.
            if (CAUSTICA_VOLUME_PATH && medium.current.water && shaftOn) {
                float shaftShare;
                float3 prefixWater = shaftInScatter(ro, rd, payload.hitT, pix, worldPush.waterShaft.x,
                        medium.current.phaseG, true, shaftShare);
                outImage[pix] = float4(clamp(prefixWater * worldPush.preExposure,
                        float3(0.0, 0.0, 0.0), float3(HALF_MAX, HALF_MAX, HALF_MAX)), 1.0);
                prefixInImage = true;
            }
        }
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0),
                           RAY_CONE_MIN_WIDTH);

        bool isWater = material == MATERIAL_WATER;
        // Only the wave perturbation can aim this interface's mirror below its own plane; a flat water
        // plane and every glass/ice face reflect into the half-space they were hit from.
        bool waveNormal = isWater && waterWaves;
        bool entering = payloadDielectricEntering();
        float3 geometricNormal = n;
        float3 previousNormal = n;
        if (waveNormal) {
            float3 waterPos = float3(hitPos.xz + worldPush.waterAnchor.xy, hitPos.y);
            float waterFootprint = rayConeWidth / max(abs(dot(-rd, geometricNormal)), 0.2);
            if (bounce == 0 && abs(geometricNormal.y) >= 0.5) {
                float2 currentGrad;
                float2 previousGrad;
                waterWaveGradTemporal(waterPos.xz, waterPos.y, worldPush.waterParams.w,
                                      worldPush.waterAnchor.z, waterFootprint, currentGrad, previousGrad);
                float orientation = geometricNormal.y >= 0.0 ? 1.0 : -1.0;
                n = orientation * normalize(float3(-currentGrad.x, 1.0, -currentGrad.y));
                previousNormal = orientation
                        * normalize(float3(-previousGrad.x, 1.0, -previousGrad.y));
            } else {
                n = applyWaterWaves(geometricNormal, waterPos, rd,
                                    worldPush.waterParams.w, waterFootprint);
            }
        }

        float transmission = clamp(payloadTransmission(), 0.0, 1.0);
        Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0),
                                              payloadAlbedo(), transmission,
                                              worldPush.waterOpacity);
        // A face that bounds no volume (see PAYLOAD_DIELECTRIC_THIN: a pane, a decorative translucent layer, a
        // translucent model sheet) has the medium the ray arrived with on BOTH sides. Reading etaI from the stack and
        // etaT from the stack's outer medium there describes a crossing the geometry does not have, and
        // wherever the incoming direction is more than asin(1/ior) from the face normal that invented exit
        // total-internal-reflects — the background lands on the sheet as a hard-edged disc. Such a face is a
        // filter: Fresnel against its own index on the near side, bent once as an entry (which is the
        // refraction a pane is seen to give), stack untouched, and its tint absorbed here because no medium
        // will do it downstream.
        bool thinDielectric = !isWater && (payload.flags & PAYLOAD_DIELECTRIC_THIN) != 0u;
        // An exit needs an entry. When the face reads as an exit but the path still carries air, some culled
        // or skipped quad means this crossing was never made, and deriving etaI/etaT anyway invents an
        // interface that total-internal-reflects past the critical angle. Falls back to the same filter.
        bool unmatchedExit = !entering && !mediumInsideDenser(medium);
        thinDielectric = thinDielectric || (!isWater && unmatchedExit);
        float etaI = medium.current.ior;
        float etaT = thinDielectric || entering ? entered.ior : medium.outer.ior;
        float F = waterReflectance(fresnelDielectric(clamp(dot(-rd, n), 0.0, 1.0), etaI, etaT), isWater);
        float3 transmittedDir = refract(rd, n, etaI / etaT);
        float3 thinTint = thinDielectric ? payloadAlbedo() : float3(1.0, 1.0, 1.0);
        float transmitBias = isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS;

        if (bounce == 0) {
            gv_normal = n;
            gv_rough = 0.0;
            gv_albedo = float3(0.0, 0.0, 0.0);
            // An emissive pane (the nether portal) is a light source to the RIS emitter sampling, not a
            // surface to blur against whatever shows through it. Its radiance is added by Pass B at the
            // far face.
            gv_emissive = payloadEmission() > 0.0;
            gv_hitCamRel = hitPos - worldPush.camOffset;
            gv_motionObjDisp = isWater ? float3(0.0, 0.0, 0.0) : payload.motionPrev;
            gv_spec = makeSpecSurface(gv_hitCamRel, n, previousNormal, geometricNormal,
                                      isWater ? float3(0.0, 0.0, 0.0) : float3(payload.motionPrev),
                                      gv_rough, float3(F, F, F));
            gv_specWaveMirror = waveNormal;
            // The face splits what leaves it by Fresnel and nothing else, so this is the reflection's share
            // of the pixel outright — the same number the split below hands the reflected branch.
            gv_reflectShare = F;
#ifdef CAUSTICA_INTERFACE_DEBUG
            // Quarter-scale decision codes so all four survive the ACEScg encode: 0.25 sheet filter,
            // 0.5 entry, 0.75 exit, 1.0 total internal reflection.
            gv_dbgDecision = dot(transmittedDir, transmittedDir) > 0.0
                    ? (thinDielectric ? 0.25 : (entering ? 0.5 : 0.75)) : 1.0;
            gv_dbgIor = clamp((medium.current.ior - 1.0) / 0.6, 0.0, 1.0);
            gv_dbgMaterial = float(payload.flags & PAYLOAD_MATERIAL_MASK) / 3.0;
#endif
            if (dot(transmittedDir, transmittedDir) > 0.0) {
                MediumStack guideMedium = medium;
                // A thin face leaves the stack exactly as it found it: it opens no medium to cross into.
                if (!thinDielectric) {
                    if (entering) {
                        mediumPush(guideMedium, entered);
                    } else {
                        mediumPop(guideMedium);
                    }
                }
                // A volume tints only on the way in, because the medium carries the absorption from there.
                // A sheet has no medium behind it, so it absorbs on every crossing.
                bool tintPath = !isWater && (thinDielectric || entering);
                resolveTransmissionGuide(hitPos, transmittedDir, geometricNormal,
                        guideMedium, transmitBias, rayConeWidth, rayConeSpread,
                        tintPath ? payloadAlbedo() : float3(1.0, 1.0, 1.0),
                        clamp(payloadRoughness(), 0.0, 1.0));
            }
        }

        // Split once, write both post-interface continuations, and stop Pass A radiance traversal.
        // Pass B owns every radiance trace after this point. A guide probe owns no continuation at all:
        // its split record would overwrite the one the radiance pass queued for this pixel.
        bool splitEligible = dot(transmittedDir, transmittedDir) > 0.0
                && F > 0.0 && F < 1.0;
        if (splitEligible && !guidesOnly) {
            MediumStack transmittedMedium = medium;
            if (!thinDielectric) {
                if (entering) {
                    mediumPush(transmittedMedium, entered);
                } else {
                    mediumPop(transmittedMedium);
                }
            }
            float3 deferredDir = normalize(transmittedDir);
            PathSegment deferred = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, deferredDir, transmitBias),
                    deferredDir, throughput * (1.0 - F) * thinTint, transmittedMedium,
                    rayConeWidth, rayConeSpread, seed, bounce + 1,
                    true);
            queue[splitRecord] = packPathSegment(deferred, PATH_NO_NEXT);
            nextRecord = splitRecord;
            float3 reflectedDir = waveNormal ? waterReflectedDir(rd, n, geometricNormal) : reflect(rd, n);
            PathSegment reflected = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, reflectedDir, SURF_BIAS),
                    reflectedDir, throughput * F, medium,
                    rayConeWidth, rayConeSpread, seed ^ 0xa511e9b3u, bounce + 1,
                    true);
            return reflected;
        }

        bool hasTransmission = dot(transmittedDir, transmittedDir) > 0.0;
        PathSegment continuation;
        if (!hasTransmission || F >= 1.0) {
            float3 reflectedDir = waveNormal ? waterReflectedDir(rd, n, geometricNormal) : reflect(rd, n);
            continuation = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, reflectedDir, SURF_BIAS),
                    reflectedDir, throughput * F, medium,
                    rayConeWidth, rayConeSpread, seed, bounce + 1,
                    true);
        } else {
            float3 deferredDir = normalize(transmittedDir);
            if (!thinDielectric) {
                if (entering) {
                    mediumPush(medium, entered);
                } else {
                    mediumPop(medium);
                }
            }
            continuation = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, deferredDir, transmitBias),
                    deferredDir, throughput * (1.0 - F) * thinTint, medium,
                    rayConeWidth, rayConeSpread, seed, bounce + 1,
                    true);
        }
        return continuation;
    }
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 pix = int2(dispatchIndex);
    float2 size = float2(dimensions);
    float2 uv = (float2(pix) + 0.5) / size;
    float2 jndc = (uv + worldPush.jitter / size) * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 pinholeDir = normalize(farP - nearP);
    float rayConeSpread = primaryRayConeSpread(jndc, size, pinholeDir);
    uint pixelSeed = dispatchIndex.x * 1973u + dispatchIndex.y * 9277u + 26699u;
    uint seed = pixelSeed ^ (worldPush.frameIndex * 2654435761u);
    float3 origin = nearP + worldPush.camOffset;
    float3 dir = pinholeDir;
    bool lensSampled = false;
    // The lens: this pixel's pinhole ray is moved sideways inside the sensor plane and re-aimed at the
    // point where it crosses the focal plane. That plane keeps the pinhole's radiance, and a surface at
    // range s lands off the pinhole ray by radius * (1 - s / focus) — so the disc closes at the focal
    // plane, settles at radius / focus of the frame's tangent span out at the horizon, and opens to
    // radius / s on this side of it. One aperture point per frame, and the reconstruct averages the run of
    // them into a disc, exactly as it averages the sun square into a penumbra.
    if (worldPush.dof.x > 0.0 && worldPush.dof.y > 0.0) {
        float3 axis;
        float3 lensRight;
        float3 lensUp;
        sensorFrame(axis, lensRight, lensUp);
        // The host already holds the horizon inside the ceiling (it softens at radius / focus). What it
        // cannot see is this pixel's own surface, and a surface in front of the focal plane softens at
        // radius / depth instead, which runs away as depth falls -- a wall half a block from the eye asks
        // for a disc a hundred pixels wide, and no denoiser fills that from one aperture point per
        // frame. Last frame's pinhole distance closes it: solve radius * (1/depth - 1/focus) <= ceiling.
        float radius = worldPush.dof.x;
        // Clamped into [0, focus] because a texel no frame has written yet hands back whatever
        // vkAllocateMemory held, and a NaN or a negative distance must not reach the ray: inside the
        // clamp, garbage can only ask for a narrower lens, never a wider one. A zero ceiling lane is the
        // host's way of saying the cap is switched off, and then the dial is the whole lens.
        float behind = worldPush.dof.y - clamp(focusDepthPrev[pix], 0.0, worldPush.dof.y);
        if (worldPush.dof.z > 0.0 && behind > 1.0e-3) {
            float seen = worldPush.dof.y - behind;
            radius = min(radius, worldPush.dof.z * seen * worldPush.dof.y / behind);
        }
        float2 aperture = sampleLensDisk(worldPush.frameIndex, pixelSeed) * radius;
        float3 lensPoint = nearP + aperture.x * lensRight + aperture.y * lensUp;
        float3 focalPoint = nearP
                + pinholeDir * (worldPush.dof.y / max(dot(pinholeDir, axis), 1.0e-4));
        origin = lensPoint + worldPush.camOffset;
        dir = normalize(focalPoint - lensPoint);
        lensSampled = true;
    }
    MediumStack cameraMedium = makeMediumStack((worldPush.flags & 1u) != 0u
            ? makeDielectricMedium(true, WATER_IOR, worldPush.waterParams.xyz, 1.0,
                    worldPush.waterOpacity)
            : airMedium());
    // The fog in-scatter's light comes from the same derived celestial body the indirect pass uses;
    // deriving it costs two LUT fetches, so it only happens on a debug frame that will actually read it.
    bool volumeDebugOn = (worldPush.flags & WORLD_FLAG_VOLUME_DEBUG) != 0u;
    // The pre-interface fog and the underwater in-scatter both need the derived celestial body the
    // indirect pass uses; deriving it costs two LUT fetches, so it happens only on frames that will
    // actually read it (a volume-debug frame, fog frames, or a submerged camera with water scattering on).
    bool fogOn = (worldPush.flags & WORLD_FLAG_FOG) != 0u;
    bool shaftOn = (worldPush.flags & WORLD_FLAG_SHAFTS) != 0u;
    bool cameraWaterScatter = (worldPush.flags & 1u) != 0u && worldPush.water1.x > 0.0;
    if (fogOn || shaftOn || volumeDebugOn || cameraWaterScatter) {
        celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
        celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    }
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    uint baseRecordCount = dimensions.x * dimensions.y;
    uint splitRecord = baseRecordCount + pixelIndex;
    DevicePtr<PackedPathSegment> queue = DevicePtr<PackedPathSegment>(pc.pathQueueAddr);

    seed = pcg(seed);
    PathSegment current = makePathSegment(origin, dir, float3(1.0, 1.0, 1.0),
            cameraMedium, 0.0, rayConeSpread, seed, 0, true);
    uint nextRecord;
    bool prefixInImage;
    PathSegment terminal = tracePrimary(
            current, queue, splitRecord, pix, volumeDebugOn, false, prefixInImage, nextRecord);
    PackedPathSegment packedTerminal = packPathSegment(terminal, nextRecord);
    if (prefixInImage) {
        packedTerminal.pathFlags |= PATH_PREFIX_IN_IMAGE;
    }
    queue[pixelIndex] = packedTerminal;

    // The guides describe what the pinhole sees, not what the lens sample hit. DLSS-RR classifies and
    // reprojects with them and its contract is a pinhole camera, while a lens lands its hit off this
    // pixel by its circle of confusion -- guides taken from that ray hand the reconstruct a depth,
    // normal, albedo and mirror motion that walk across scene layers every frame, which it paints over
    // as a smear instead of resolving as blur. Re-walking the same bounce-0 segment on the pinhole ray
    // lands them back on this pixel; the radiance above stays on the lens ray, so the defocus is still
    // traced rather than post-smudged. The probe writes no radiance and queues no continuation, and it is
    // the feature's real cost: with a lens on, Pass A walks the bounce-0 segment twice per pixel.
    // `dof.w` is the player's say over that cost, and with it off the guides carry the aperture point.
    if (lensSampled && worldPush.dof.w > 0.0) {
        bool probePrefix;
        uint probeNext;
        tracePrimary(makePathSegment(nearP + worldPush.camOffset, pinholeDir, float3(1.0, 1.0, 1.0),
                        cameraMedium, 0.0, rayConeSpread, seed, 0, true),
                queue, splitRecord, pix, false, true, probePrefix, probeNext);
    }

    writeGuides(pix, pinholeDir, size, rayConeSpread);
}
