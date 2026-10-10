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

// Volume debug image (debug views 10..13): r fog transmittance along the camera ray, g the fog
// in-scatter's luminance, b the cloud coverage the camera ray reads off the cloud plane, a the
// cloud-shadow transmittance the sun leg from the camera would ride. The cloud lanes live since
// Phase 4; a cloud-off frame leaves them at their inert defaults (0 and 1) through the same push
// gates every other consumer uses. Gated by WORLD_FLAG_VOLUME_DEBUG, so the ordinary frame never
// touches this image.
void writeVolumeDebug(int2 pix, float3 ro, float3 rd, float t, bool allowShadowCull) {
    float fogT;
    float3 insc = fogSegmentInScatter(ro, rd, t, pix, allowShadowCull, fogT);
    float cloudDensity = 0.0;
    float cloudShadow = 1.0;
    if ((worldPush.flags & WORLD_FLAG_CLOUD) != 0u && worldPush.cloud0.x > 0.0) {
        float tPlane = (worldPush.cloud0.y - ro.y) / rd.y;
        if (tPlane > 0.0) {
            cloudDensity = cloudCoverageWorld(ro.xz + rd.xz * tPlane);
        }
        if (celestialLight.halfAngle > 0.0) {
            cloudShadow = cloudShadowTransmittance(ro, celestialLight.dir);
        }
    }
    volumeDebug[pix] = float4(fogT, luminance(insc), cloudDensity, cloudShadow);
}

public PathSegment tracePrimary(PathSegment seg,
                                DevicePtr<PackedPathSegment> queue,
                                uint splitRecord,
                                int2 pix,
                                bool volumeDebugOn,
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
    nextRecord = PATH_NO_NEXT;
    prefixInImage = false;

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
                gv_motionUseRefracted = false;
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
                gv_normal = n;
                gv_hitCamRel = hitPos - worldPush.camOffset;
                gv_motionUseRefracted = false;
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
                }
                // A fog-dominated pixel's diffuse guide converges toward mid-grey by the fog's opacity:
                // RR demodulates against gv_albedo, and the fog term in front of the surface is not that
                // surface's albedo, so leaving it would brighten dark surfaces and darken bright ones
                // wherever fog sits in front. Guarded so the fog-off guide write stays untouched.
                if (fogOn) {
                    float fogOpacity = 1.0 - fogTransmittance(worldPush.fog0.x, worldPush.fog0.z,
                            worldPush.fog0.y, ro.y, rd.y, payload.hitT);
                    if (fogOpacity > 0.0) {
                        gv_albedo = lerp(gv_albedo, float3(0.5, 0.5, 0.5), fogOpacity);
                    }
                }
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
        if (fogOn && !any(medium.current.extinction > 0.0)) {
            // The in-scatter call owns the segment's sigma (the indoor sky-exposure ladder may thin
            // it), and its out fogT is that same thinned transmittance — attenuate throughput with
            // it so the two agree instead of re-integrating at full density here.
            float prefixFogT;
            float3 prefixRadiance = fogSegmentInScatter(ro, rd, payload.hitT, pix, true, prefixFogT);
            throughput *= prefixFogT;
            outImage[pix] = float4(clamp(prefixRadiance * worldPush.preExposure,
                    float3(0.0, 0.0, 0.0), float3(HALF_MAX, HALF_MAX, HALF_MAX)), 1.0);
            prefixInImage = true;
        }
        // The water medium's own scattering over the camera-to-interface stretch (the camera started
        // submerged): the same prefix doctrine as the fog branch above — the radiance is common to
        // both Fresnel continuations and can ride neither record, so it is written into outImage once
        // and flagged, and the indirect pass accumulates on top. The two branches never co-fire: the
        // fog stands down while any extinction is present, the water term needs the water medium.
        if (medium.current.water && worldPush.water1.x > 0.0) {
            float3 prefixWater = waterSegmentInScatter(ro, rd, payload.hitT, pix, medium.current);
            outImage[pix] = float4(clamp(prefixWater * worldPush.preExposure,
                    float3(0.0, 0.0, 0.0), float3(HALF_MAX, HALF_MAX, HALF_MAX)), 1.0);
            prefixInImage = true;
        }
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0),
                           RAY_CONE_MIN_WIDTH);

        bool isWater = material == MATERIAL_WATER;
        bool entering = payloadDielectricEntering();
        float3 geometricNormal = n;
        float3 previousNormal = n;
        if (isWater && waterWaves) {
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
        float etaI = medium.current.ior;
        float etaT = entering ? entered.ior : medium.outer.ior;
        float F = waterReflectance(fresnelDielectric(clamp(dot(-rd, n), 0.0, 1.0), etaI, etaT), isWater);
        float3 transmittedDir = refract(rd, n, etaI / etaT);
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
            gv_motionUseRefracted = false;
            gv_motionObjDisp = isWater ? float3(0.0, 0.0, 0.0) : payload.motionPrev;
            gv_spec = makeSpecSurface(gv_hitCamRel, n, previousNormal, geometricNormal,
                                      isWater ? float3(0.0, 0.0, 0.0) : float3(payload.motionPrev),
                                      gv_rough, float3(F, F, F));
            if (dot(transmittedDir, transmittedDir) > 0.0) {
                MediumStack guideMedium = medium;
                if (entering) {
                    mediumPush(guideMedium, entered);
                } else {
                    mediumPop(guideMedium);
                }
                resolveTransmissionGuide(hitPos, transmittedDir, geometricNormal,
                        guideMedium, transmitBias, rayConeWidth, rayConeSpread,
                        !isWater && entering
                                ? payloadAlbedo() : float3(1.0, 1.0, 1.0));
            }
        }

        // Split once, write both post-interface continuations, and stop Pass A radiance traversal.
        // Pass B owns every radiance trace after this point.
        bool splitEligible = dot(transmittedDir, transmittedDir) > 0.0
                && F > 0.0 && F < 1.0;
        if (splitEligible) {
            MediumStack transmittedMedium = medium;
            if (entering) {
                mediumPush(transmittedMedium, entered);
            } else {
                mediumPop(transmittedMedium);
            }
            float3 deferredDir = normalize(transmittedDir);
            PathSegment deferred = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, deferredDir, transmitBias),
                    deferredDir, throughput * (1.0 - F), transmittedMedium,
                    rayConeWidth, rayConeSpread, seed, bounce + 1,
                    true);
            queue[splitRecord] = packPathSegment(deferred, PATH_NO_NEXT);
            nextRecord = splitRecord;
            float3 reflectedDir = reflect(rd, n);
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
            float3 reflectedDir = reflect(rd, n);
            continuation = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, reflectedDir, SURF_BIAS),
                    reflectedDir, throughput * F, medium,
                    rayConeWidth, rayConeSpread, seed, bounce + 1,
                    true);
        } else {
            float3 deferredDir = normalize(transmittedDir);
            if (entering) {
                mediumPush(medium, entered);
            } else {
                mediumPop(medium);
            }
            continuation = makePathSegment(
                    offsetSurfaceOrigin(hitPos, geometricNormal, deferredDir, transmitBias),
                    deferredDir, throughput * (1.0 - F), medium,
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
    float3 origin = nearP + worldPush.camOffset;
    float3 dir = normalize(farP - nearP);
    float rayConeSpread = primaryRayConeSpread(jndc, size, dir);
    uint seed = (dispatchIndex.x * 1973u + dispatchIndex.y * 9277u + 26699u)
            ^ (worldPush.frameIndex * 2654435761u);
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
    bool cameraWaterScatter = (worldPush.flags & 1u) != 0u && worldPush.water1.x > 0.0;
    if (fogOn || volumeDebugOn || cameraWaterScatter) {
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
            current, queue, splitRecord, pix, volumeDebugOn, prefixInImage, nextRecord);
    PackedPathSegment packedTerminal = packPathSegment(terminal, nextRecord);
    if (prefixInImage) {
        packedTerminal.pathFlags |= PATH_PREFIX_IN_IMAGE;
    }
    queue[pixelIndex] = packedTerminal;

    writeGuides(pix, dir, jndc, size, rayConeSpread);
}
