// Ray miss = sky. Two sky-view LUT fetches (sun slice + moon slice) give the whole atmosphere; the disc
// sprites, the starfield and the night airglow floor are added on top. Everything physical lives in
// sky.slang, shared verbatim with the LUT bakes, so the sky a ray sees and the sky the LUT holds cannot
// diverge.
//
// Atmospheric scattering is amortized into the per-frame LUT. Analytic stars, airglow, and textured
// celestial discs are combined with that physical result here.
//
// A sky-light-free dimension with skybox END (the End) renders no atmosphere at all: its own dome
// gradient plus ambient floor, the end_sky.png cube, and the end flash quad. See "The End" below.
//
// Celestial discs are gated by a packed payload flag, which raygen sets per-ray from the path state: true
// on the primary ray and after every specular/dielectric bounce (so the sun/moon appear in water, glass
// and metal reflections/refractions), false after a diffuse vertex did sun/moon NEE — otherwise the disc
// would be counted twice against the direct light and show up as fireflies. The atmosphere and the stars
// are added on every miss at any bounce: stars are not a NEE light, so they never double-count.
import world_common;
import sky;
import volume;
import bindings;

// Vanilla celestials atlas (sun + moon-phase sprites), bound by RtComposite. Sampled with an explicit LOD
// (a miss shader has no derivatives). Keeping the real sprites means a resource pack's sun and moon still
// show up, which a procedural disc would silently drop.
// Drawn disc radiance is derived from illuminance and the solid angle the body is actually DRAWN at
// (L = E / Ω), deliberately not from its true surface luminance. Vanilla's sun quad subtends ~60x the real
// sun, i.e. ~3900x its solid angle; painting it at the physical 1.6e9 cd/m² would put ~3900x the sun's
// power into every path that sees it. Specular and dielectric bounces do see it and have already taken the
// sun through NEE, so that double-count would go from harmless to dominant, with fireflies off every
// glossy lobe. Deriving from illuminance keeps the drawn body's total power equal to the real one's at
// whatever size it is drawn, and costs nothing visually: the result still sits ~15 EV over an 18%-grey
// noon surface, which the ACES 2.0 output transform renders as pure white either way.
float3 discRadiance(float illuminance, float halfAngle) {
    float side = 2.0 * tan(halfAngle);      // the square spans this much on a side, gnomonically
    return float3(illuminance / max(side * side, 1.0e-8));
}

// Dave Hoskins' hash13 (https://www.shadertoy.com/view/4djSRW): well-distributed and pattern-free.
float hash13(float3 p3) {
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

// Rotate v about a unit axis by angle (Rodrigues). Used to wheel the starfield about the celestial pole.
float3 rotateAxis(float3 v, float3 axis, float ang) {
    float c = cos(ang), s = sin(ang);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// Procedural starfield, a faithful port of vanilla's buildStars(): ~1500 discrete stars at random sphere
// directions, each a small SQUARE billboard of random size and rotation. A cube-map grid is the spatial
// accelerator — at most one star per cell, at a hashed position inset from the cell edges (so its square
// never crosses into a neighbour), drawn as a hash-rotated square of hashed size. The lookup direction is
// rotated about the celestial pole by -starAngle so the field wheels with world time like the real sky.
static const float STAR_CELLS     = 21.0;  // cube-face grid resolution; with the threshold gives ~1500 stars
static const float STAR_THRESHOLD = 0.86;  // cell occupancy cutoff (higher = fewer stars)
static const float STAR_HALF      = 0.045; // star half-size as a fraction of a cell (≈ vanilla 0.11° at centre)
float3 stars(float3 dir, SkyState state, float3 extinction) {
    if (state.starLuminance <= 0.0 || dir.y <= 0.0) {
        return float3(0.0);
    }
    // Negative angle: rotating the LOOKUP direction is the inverse of rotating star geometry, so
    // -starAngle makes the field wheel the SAME way the sun and moon do.
    float3 sdir = rotateAxis(dir, normalize(celestialAxis(state.noonTilt)), -state.starAngle);
    // Cube-map face + 2D gnomonic uv → uniform square cells, no pole/equator distortion.
    float3 a = abs(sdir);
    float2 uv;
    float face;
    if (a.x >= a.y && a.x >= a.z)      { uv = sdir.yz / a.x; face = sdir.x > 0.0 ? 0.0 : 1.0; }
    else if (a.y >= a.z)               { uv = sdir.xz / a.y; face = sdir.y > 0.0 ? 2.0 : 3.0; }
    else                               { uv = sdir.xy / a.z; face = sdir.z > 0.0 ? 4.0 : 5.0; }
    float2 g = uv * STAR_CELLS;
    float3 id = float3(floor(g), face);
    if (hash13(id) < STAR_THRESHOLD) {
        return float3(0.0); // no star in this cell
    }
    float sz = STAR_HALF * (0.6 + 0.8 * hash13(id + 53.0));
    float inset = sz * 1.5; // keeps the rotated square clear of the cell edge
    float2 center = inset + (1.0 - 2.0 * inset) * float2(hash13(id + 11.0), hash13(id + 23.0));
    float2 q = frac(g) - center;
    float ang = hash13(id + 91.0) * 6.2831853;            // random per-star rotation (vanilla zRot)
    float cs = cos(ang), sn = sin(ang);
    q = float2(cs * q.x - sn * q.y, sn * q.x + cs * q.y); // GLSL mat2(cs,-sn,sn,cs) * q, written out
    float box = max(abs(q.x), abs(q.y));                  // Chebyshev distance → square
    float star = 1.0 - smoothstep(sz * 0.6, sz, box);     // soft-edged square (helps DLSS stability)
    if (star <= 0.0) {
        return float3(0.0);
    }
    float bright = 0.55 + 0.45 * hash13(id + 71.0);       // mild per-star brightness variation
    // Per-star colour temperature: a hash picks a spot on a warm-orange to blue-white ramp, matching real
    // star populations. Extinction reddens and dims them toward the horizon on the atmosphere's curve.
    float3 starColor = lerp(float3(0.78, 0.60, 0.42), float3(0.50, 0.62, 0.95), hash13(id + 137.0));
    return star * bright * starColor * state.starLuminance * extinction;
}

// ---- Realistic bodies --------------------------------------------------------------
//
// With the vanilla sprites switched off, each body becomes a circular disc at the angular radius the
// lighting is already sampled at. That coupling is the reason for using the NEE radius rather than the
// true ~0.27 degrees: radiance is illuminance divided by projected area, so the authored pair
// (moonIlluminance 5 lux, moonAngularRadius 1.5 degrees) works out at ~2300 cd/m2 against the real Moon's
// ~2500, and the sun lands within a factor of five of its true photosphere. Drawing either at its real
// angular size would instead concentrate the same authored power into a far smaller disc and blow both
// bodies out. It also means a body's apparent size and the softness of the shadow it casts can no longer
// disagree, which they deliberately could under the sprite path.

// Fraction of the disc radius spent on the limb transition, so a body under a degree across does not
// staircase at the sample rate a path tracer delivers.
static const float DISC_EDGE = 0.04;

// Radians the Moon's terminator axis leans off horizontal. The ecliptic meets the horizon at an angle,
// which is why a crescent sometimes smiles and sometimes stands on its side; without a lean every phase
// would sit level and the Moon would look like a clock hand.
static const float MOON_AXIS_TILT = 0.41;

// One lattice cell of the Moon's mare field.
static float moonCell(int2 c) {
    uint n = uint(c.x) * 1619u + uint(c.y) * 31337u + 4021u;
    n = (n << 13) ^ n;
    n = (n * (n * n * 15731u + 789221u) + 1376312589u) & 0x7FFFFFFFu;
    return float(n) * (1.0 / 2147483648.0);
}

// Quintic value noise. The weights' derivative vanishes at both cell edges, so the field is C1 and a mare
// boundary cannot turn into a crease on the limb the way a bilinear lattice would.
static float moonNoise(float2 coord) {
    float2 i0 = floor(coord);
    float2 f = coord - i0;
    float2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    int2 c = int2(i0);
    float a = moonCell(c);
    float b = moonCell(c + int2(1, 0));
    float d = moonCell(c + int2(0, 1));
    float e = moonCell(c + int2(1, 1));
    return (a + (b - a) * w.x) * (1.0 - w.y) + (d + (e - d) * w.x) * w.y;
}

// Relative surface brightness. The absolute level is already fixed by the illuminance and the disc area,
// so this only carries the pattern: dark maria, brighter mottled highlands.
static float moonPattern(float2 p) {
    float mare = smoothstep(0.54, 0.72, moonNoise(p * 2.3 + float2(11.0, 7.0)));
    float mottle = moonNoise(p * 7.5 + float2(3.0, 19.0)) * 2.0 - 1.0;
    return 1.0 - 0.55 * mare + 0.16 * mottle;
}

/** The sun as a limb-darkened photosphere. `local` is the disc offset in units of its radius. */
static float3 realisticSun(SkyState state, float2 local, float3 extinction) {
    float r = length(local);
    float mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    float edge = 1.0 - smoothstep(1.0 - DISC_EDGE, 1.0 + DISC_EDGE, r);
    // Centre-to-limb darkening of a ~5800 K photosphere, to second order in (1 - mu).
    float limb = 1.0 - 0.76 * (1.0 - mu) - 0.12 * (1.0 - mu) * (1.0 - mu);
    return discRadianceCircular(state.sunIlluminance, state.sunAngularRadius) * limb * edge * extinction;
}

/**
 * The Moon as a sphere, not a sprite: the terminator falls out of the sun direction expressed in the
 * body's own frame, so the crescent is oriented correctly for wherever the sun happens to be rather than
 * being looked up from a phase texture.
 */
static float3 realisticMoon(SkyState state, float2 local, float3 sunLocal, float3 extinction) {
    float r = length(local);
    float mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    float edge = 1.0 - smoothstep(1.0 - DISC_EDGE, 1.0 + DISC_EDGE, r);
    float3 n = normalize(float3(local.x, local.y, mu));   // surface normal, +z toward the viewer
    float cosI = max(0.0, dot(n, sunLocal));
    if (cosI <= 0.0) {
        return float3(0.0);                                // the unlit side of the terminator
    }
    // Mostly Lommel-Seeliger: unlike Lambert it keeps craters and mare edges visible near the terminator,
    // which is where the surface reads as a surface. Scaled so a full moon's centre stays near unity.
    float lambert = cosI;
    float lommel = cosI / max(cosI + mu, 1.0e-3);
    float shade = lerp(lambert, lommel * 2.0, 0.65);
    return discRadianceCircular(state.moonFullIlluminance, state.moonAngularRadius)
            * moonPattern(local) * shade * edge * extinction;
}

// ---- The End -------------------------------------------------------------------------------------
//
// A sky-light-free dimension with skybox END renders vanilla's end sky instead of an atmosphere: the
// dome gradient plus ambient floor from sky.dimensionSkyRadiance, the end_sky.png cube on top, and the
// end flash quad wherever the payload may see celestials. The flash is also the dimension's NEE
// directional light (sky.endFlashLight), so like the sun/moon discs the quad is gated behind
// showCelestial to keep the direct hit and the NEE sample from double-counting.

/**
 * Texture coordinate (in repeats, 0..16) of vanilla's buildEndSky() cube for a view direction: six quads
 * rotated to the faces of a +-100 cube, the base quad carrying u along its local x and v along its local
 * z with the texture repeated 16x16 per face. The per-face u/v orientation below is that rotation table
 * spelled out, which is what makes the noise pattern continuous across cube edges exactly as vanilla's
 * is.
 */
float2 endSkyCubeUv(float3 dir) {
    float3 a = abs(dir);
    float3 p = dir * (END_SKY_CUBE_HALF / max(max(a.x, a.y), a.z));
    if (a.x >= a.y && a.x >= a.z) {
        // x = -100 (rotationZ(-90)) / x = +100 (rotationZ(+90))
        float u = dir.x > 0.0 ? (p.y + END_SKY_CUBE_HALF) : (END_SKY_CUBE_HALF - p.y);
        return float2(u, p.z + END_SKY_CUBE_HALF) * (16.0 / (2.0 * END_SKY_CUBE_HALF));
    }
    if (a.y >= a.z) {
        // y = -100 (identity) / y = +100 (rotationX(180))
        float2 uv = dir.y > 0.0
                ? float2(END_SKY_CUBE_HALF - p.z, END_SKY_CUBE_HALF - p.x)
                : float2(p.x, p.z) + END_SKY_CUBE_HALF;
        return uv * (16.0 / (2.0 * END_SKY_CUBE_HALF));
    }
    // z = -100 (rotationX(+90)) / z = +100 (rotationX(-90))
    float v = dir.z > 0.0 ? (p.y + END_SKY_CUBE_HALF) : (END_SKY_CUBE_HALF - p.y);
    return float2(p.x + END_SKY_CUBE_HALF, v) * (16.0 / (2.0 * END_SKY_CUBE_HALF));
}

/** The end_sky.png cube: sampled with wrapping, linearised, and painted by vanilla's vertex grey. */
float3 endSkyBoxRadiance(float3 dir, SkyState state) {
    float3 texel = endSkyTex.SampleLevel(endSkyCubeUv(dir), 0.0).rgb;
    return srgbToLinear(texel) * (END_SKY_VERTEX_GRAY * state.skyBoxGain);
}

/**
 * The end flash: vanilla's single quad at height END_FLASH_HEIGHT along the flash centre direction,
 * spanning +-END_FLASH_HALF in its own plane, textured from the celestials atlas' end_flash sprite
 * (a radial purple glow on black, drawn additively -- the black surround contributes nothing).
 */
float3 endFlashQuadRadiance(float3 dir, SkyState state) {
    if (state.flashIntensity <= 0.0) {
        return float3(0.0);
    }
    float x = state.flashXAngle;
    float y = state.flashYAngle;
    float cx = cos(x);
    float sx = sin(x);
    float cy = cos(y);
    float sy = sin(y);
    float3 n = endFlashCentreDirection(x, y);
    float denom = dot(dir, n);
    if (denom <= 1.0e-5) {
        return float3(0.0); // the plane only spans directions with a positive normal component
    }
    float3 offset = dir * (END_FLASH_HEIGHT / denom) - n * END_FLASH_HEIGHT;
    // The quad's local x/z axes after the same Y(180 - y) then X(-90 - x) rotation vanilla applies.
    float3 axisX = float3(-cy, 0.0, -sy);
    float3 axisZ = float3(-sx * sy, cx, sx * cy);
    float2 local = float2(dot(offset, axisX), dot(offset, axisZ)) / END_FLASH_HALF;
    if (max(abs(local.x), abs(local.y)) > 1.0) {
        return float3(0.0);
    }
    float2 uv = lerp(state.endFlashUv.xy, state.endFlashUv.zw, local * 0.5 + 0.5);
    float3 texel = srgbToLinear(celestialsAtlas.SampleLevel(uv, 0.0).rgb);
    return texel * (state.flashQuadLuminance * state.flashIntensity);
}

/** Atmosphere in-scatter for a world direction: both bodies' slices of this frame's sky-view LUT. */
[shader("miss")]
void main(inout Payload payload) {
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    SkyState state = skyState(worldPush);
    float3 dir = normalize(WorldRayDirection());
    bool showCelestial = (payload.flags & PAYLOAD_SHOW_CELESTIAL) != 0u;

    float skyFactor = saturate(state.skyLightFactor);
    bool endSky = state.skybox == SKYBOX_END && skyFactor <= 0.0;
    float3 color;
    if (endSky) {
        // No atmosphere is sampled on this path: the sky-view/transmittance LUTs are never baked in a
        // sky-light-free dimension, and the end sky is dome + box + flash on its own.
        color = dimensionSkyRadiance(state, dir) + endSkyBoxRadiance(dir, state);
        if (showCelestial) {
            color += endFlashQuadRadiance(dir, state);
        }
    } else {
        // The dome. Below-horizon rays that escaped the loaded world sample the LUT's space-black
        // floor; while the fog is actually rendering (flag up AND non-zero density — the md5
        // baseline keeps a fog-off frame's raw dome), mirror those rays to their horizon elevation
        // instead: real air seen downward is haze, not space, and thin fog no longer lets the void
        // band through the horizon.
        bool fogVoidFill = (worldPush.flags & WORLD_FLAG_FOG) != 0u && worldPush.fog0.x > 0.0;
        float3 domeDir = fogVoidFill && dir.y < 0.0 ? float3(dir.x, -dir.y, dir.z) : dir;
        color = skyDome(skyViewLut, domeDir, state);
        // Airglow plus integrated unresolved starlight: the real floor of a moonless night sky (~1e-3 cd/m²),
        // isotropic because it is emitted by the upper atmosphere itself rather than scattered from a body.
        // Multiple scattering is in the LUT, so this remains a flat, physically small constant.
        color += state.airglowLuminance;

        // Extinction along the view ray: stars redden and dim into the horizon on exactly the curve the sky
        // itself follows, because it is the same LUT that produced the sky. Zero angular radius — a star is a
        // point, so it sets at the horizon line.
        float3 extinction = transmittanceToSpace(transmittanceLut, state.viewerRadiusKm, dir, 0.0);
        color += stars(dir, state, extinction);

        if (showCelestial) {
            // Each body is drawn either from the celestials sprite, which keeps a resource pack's art winning,
            // or as a physical disc. The disc test runs first so the extinction lookup is only paid for by the
            // fraction of a percent of rays that actually land on a body.
            bool vanillaSun = (worldPush.flags & 32u) != 0u;
            bool vanillaMoon = (worldPush.flags & 64u) != 0u;
            float2 local;
            if (vanillaSun) {
                if (celestialSquareLocal(dir, state.sunDir, state.sunDiscHalfAngle, local)) {
                    // Tint from the body's CENTRE, not from this ray. The drawn quad spans ~33 degrees (vanilla's
                    // size, not the real sun's half degree), so per-ray extinction would redden its lower limb by
                    // several stops relative to its upper one — an accurate answer to a question about a body that
                    // is 60x too large. The centre keeps the sprite one colour, as vanilla draws it.
                    float3 sunExtinction = transmittanceToSpace(transmittanceLut, state.viewerRadiusKm,
                            state.sunDir, state.sunAngularRadius);
                    float2 uv = lerp(state.sunUv.xy, state.sunUv.zw, local * 0.5 + 0.5);
                    float3 texel = celestialsAtlas.SampleLevel(uv, 0.0).rgb;
                    // Vanilla's sun sprite bakes a soft glow gradient into its texels; scaling that raw to HDR
                    // makes a huge ring around the disc. Raising the texture luminance to a high power keeps only
                    // the bright core and collapses the painted halo to ~0 — the real halo comes from the Mie
                    // in-scatter in the LUT, which is where it belongs.
                    float core = pow(clamp(dot(texel, texel) * 0.45, 0.0, 1.0), 6.0);
                    color += discRadiance(state.sunIlluminance, state.sunDiscHalfAngle) * sunExtinction * core;
                }
            } else {
                float2 disc;
                float3 right;
                float3 up;
                if (celestialDiscLocal(dir, state.sunDir, state.sunAngularRadius, disc, right, up)) {
                    float3 sunExtinction = transmittanceToSpace(transmittanceLut, state.viewerRadiusKm,
                            state.sunDir, state.sunAngularRadius);
                    color += realisticSun(state, disc, sunExtinction);
                }
            }
            if (vanillaMoon) {
                if (celestialSquareLocal(dir, state.moonDir, state.moonDiscHalfAngle, local)) {
                    float2 uv = lerp(state.moonUv.xy, state.moonUv.zw, local * 0.5 + 0.5);
                    float3 texel = srgbToLinear(celestialsAtlas.SampleLevel(uv, 0.0).rgb);
                    float3 moonExtinction = transmittanceToSpace(transmittanceLut, state.viewerRadiusKm,
                            state.moonDir, state.moonAngularRadius);
                    // The phase sprite already carries the lit shape; a contrast ramp suppresses its faint painted
                    // halo while keeping that shape.
                    float shape = smoothstep(0.0, 1.0, min(length(texel), 1.0));
                    color += discRadiance(state.moonIlluminance, state.moonDiscHalfAngle)
                           * moonExtinction * texel * shape;
                }
            } else {
                float2 disc;
                float3 right;
                float3 up;
                if (celestialDiscLocal(dir, state.moonDir, state.moonAngularRadius, disc, right, up)) {
                    float3 moonExtinction = transmittanceToSpace(transmittanceLut, state.viewerRadiusKm,
                            state.moonDir, state.moonAngularRadius);
                    // The phase, not the sun direction. Minecraft drives SUN_ANGLE and MOON_ANGLE from two
                    // separate tracks that stay close to opposite all night, so the real angle between them
                    // reads as "full" whatever the phase is; the only place the phase actually exists is the
                    // MOON_PHASE index. One turn of index 0..7 is one turn of the sun about the Moon, and the
                    // sine alone puts the lit limb on alternating sides for waning and waxing, so no separate
                    // sign is needed.
                    float phaseAngle = state.moonPhaseIndex * (SKY_PI / 4.0);
                    float2 limb = float2(cos(MOON_AXIS_TILT), sin(MOON_AXIS_TILT)) * sin(phaseAngle);
                    float3 sunLocal = normalize(float3(limb.x, limb.y, cos(phaseAngle)));
                    color += realisticMoon(state, disc, sunLocal, moonExtinction);
                }
            }
            // No separate horizon gate on either body. A set sun or moon goes dark because its extinction is
            // zero below the horizon, matching the NEE visibility test.
        }

        // A dimension with no sky light has no atmosphere, no airglow, no stars and no sun or moon: its own
        // sky colour stands in for the dome. A partial factor blends rather than switches, so a datapack that
        // authors an in-between value keeps the atmosphere and the dimension colour in proportion.
        if (skyFactor < 1.0) {
            color = lerp(dimensionSkyRadiance(state, dir), color, skyFactor);
        }

        // The cloud layer, in whichever tier is switched on. The planar sheet (Phase 4) crosses the layer
        // once and shades that single crossing; the volumetric slab (Phase 5) walks through it and gets a
        // real vertical profile and light gradient. The tier is the flag, and only the primary ray pays
        // for a walk: a continuation ray — a reflection, a refraction, an indirect bounce — rides the
        // deck's MEAN extinction instead, which is the plan's "Pass B does extinction only". The bounce-0
        // test is showCelestial, the same bit this shader already reads for the sun and moon.
        // End-sky dimensions keep their dome untouched either way.
        if (!endSky) {
            float cloudT;
            float3 cloudL;
            if ((worldPush.flags & WORLD_FLAG_CLOUD_VOLUMETRIC) != 0u) {
                if (showCelestial) {
                    cloudL = volumetricCloudInScatter(worldPush, WorldRayOrigin(), dir, state,
                            float2(DispatchRaysIndex().xy), cloudT);
                } else {
                    cloudL = float3(0.0, 0.0, 0.0);
                    cloudT = volumetricCloudTransmittanceFast(worldPush, WorldRayOrigin(), dir);
                }
            } else {
                cloudL = planarCloudSkyInScatter(worldPush, WorldRayOrigin(), dir, state, cloudT);
            }
            if (cloudT < 1.0) {
                color = cloudL + cloudT * color;
            }
        }
    }

    // The sky goes out in absolute scene units at full fp32 — packSky writes the payload's three surface
    // words as raw floats rather than halves, which is why the sun disc's ~1e5 cd/m² needs neither a
    // pre-exposure scale nor a clamp to stay representable. The fp16 ceiling reappears only at world.rgen's
    // outImage store, where the finite-half clamp is applied. packSky also
    // overwrites the normal view, so there is no separate normal clear below.
    packSky(payload, bt709ToAcesCg(max(color, float3(0.0))));
    payload.hitT = -1.0;
    payload.motionPrev = half3(0.0h, 0.0h, 0.0h);
    payload.f0 = half3(0.0h, 0.0h, 0.0h);
    payload.flags = 0u;
    payload.roughMetal = packHalf2(float2(1.0, 0.0));
    payload.emissionSss = packHalf2(float2(0.0, 0.0));
    payload.iorTransmission = packHalf2(float2(1.0, 0.0));
    payload.rayCone = 0u;
}
