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
// The medium is Rayleigh air: a height profile times a per-channel coefficient.
//
//     sigma_t(p) = rho(y) * FOG_RAYLEIGH_COEFF
//     rho(y)     = density0 * exp(-max(0, y - baseY) / H)
//
// The profile falls exponentially above the fog base and is constant below it; baseY arrives already
// rebased into terrain coordinates (Java subtracts the terrain origin, the same anchoring waterAnchor
// uses), so the profile stays pinned to the world as the terrain rebase shifts.
//
// The coefficient carries the colour, and that is the point of the term: blue is extinguished about
// four times faster than red per block, so distance blues the terrain out while the low sun reddens
// through the same air — the aerial perspective every landscape has. At the default 0.0078 the
// e-folding distance is 280 blocks for blue and 1.1 km for red. FOG_RAYLEIGH_COEFF is that
// coefficient in ACEScg, the push's working space, which is what makes a white surface lose each
// channel at the rate the source filter says; AP1's primaries are far wider than Rec.709's, so the
// same filter reads different distances on this file's channels than Photon's Rec.709 ones.
//
// The fog's scattering fraction lives Java-side (volume.fog-albedo) and arrives inside fogColor.rgb —
// the shader must not re-scale it.

// Photon's Rayleigh air: from_srgb(0.31, 0.67, 1.00) in linear Rec.709 times the 0.5 its airmass
// integral folds in, carried through the push's Rec.709 -> ACEScg matrix. 1/block per unit of pushed
// density.
public static const float3 FOG_RAYLEIGH_COEFF = float3(0.116688, 0.195699, 0.457982);
// Largest channel. The marched tier's significant-band bound has to hold for the channel that dies
// first, so it prices density against this coefficient.
public static const float FOG_RAYLEIGH_PEAK = 0.457982;
// ACEScg luminance of the coefficient (AP1 weights 0.2722/0.6741/0.0537): the scalar proxy wherever
// one number has to stand for the medium — the sun-probe strata, the shaft attenuation, the volume
// debug image.
public static const float FOG_RAYLEIGH_MEAN = 0.188272;

// Closed-form depth of the pure-exponential stretch: starts `above` blocks above the base and travels
// `len` along the vertical component dirY without crossing the base. Positive for either dirY sign.
// For dirY < 0 the two signed factors cancel, so the large intermediate exp(above/H) only ever loses
// precision, never overflows. `density0` is the profile's value at the base, not an extinction: the
// coefficient multiplies it at the call sites that need one.
public float fogExpStretchDepth(float density0, float heightH, float above, float dirY, float len) {
    return density0 * exp(-max(above, 0.0) / heightH) * (heightH / dirY)
            * (1.0 - exp(-dirY * len / heightH));
}

// Optical depth of the profile over [0, t] along a direction with vertical component dirY, in units of
// the coefficient (multiply by FOG_RAYLEIGH_COEFF for the medium's own depth). Closed form, so the
// analytic fog tier integrates with zero march samples and zero Monte-Carlo noise — the property the RR
// pipeline and the low-end tier both lean on. The clamp at the base makes this piecewise: below baseY
// the density is flat density0, above it the exponential integrates in closed form.
public float fogOpticalDepth(float density0, float baseY, float heightH, float y0, float dirY, float t) {
    if (t <= 0.0) {
        return 0.0;
    }
    float above = y0 - baseY;
    if (abs(dirY) < 1.0e-6) {
        return density0 * exp(-max(above, 0.0) / heightH) * t;
    }
    if (dirY > 0.0) {
        if (above >= 0.0) {
            return fogExpStretchDepth(density0, heightH, above, dirY, t);
        }
        float sBase = -above / dirY; // flat-density stretch from below the base up to baseY
        if (t <= sBase) {
            return density0 * t;
        }
        return density0 * sBase + fogExpStretchDepth(density0, heightH, 0.0, dirY, t - sBase);
    }
    // dirY < 0
    if (above <= 0.0) {
        return density0 * t;
    }
    float sBase = above / (-dirY); // exponential stretch from above the base down to baseY
    if (t <= sBase) {
        return fogExpStretchDepth(density0, heightH, above, dirY, t);
    }
    return fogExpStretchDepth(density0, heightH, above, dirY, sBase) + density0 * (t - sBase);
}

// Per-channel transmittance of the height profile alone, with no border haze. This is the sun legs'
// quantity: they deliberately exclude the flat term.
public float3 fogTransmittanceRGB(float density0, float baseY, float heightH,
                                  float y0, float dirY, float t) {
    return exp(-fogOpticalDepth(density0, baseY, heightH, y0, dirY, t) * FOG_RAYLEIGH_COEFF);
}

// ---- The world border's haze -----------------------------------------------------------------------
//
// Walking up to a world border closes the air in. Vanilla did it by driving the linear fog's start and end
// toward zero; there is no linear fog here, so the same thing happens to the medium the path tracer already
// integrates: a FLAT extinction added to the height profile, sharing the fog's albedo, phase, sun and sky
// terms and flag.
//
// Being an addition to sigma_t rather than a second layer is what makes it cheap and what makes it exact at
// once: the identity integral[T(s)·sigma_t(s) ds] = 1 - T(t) the tiers' in-scatter rests on holds for any
// profile, so it holds for the sum, and the closed form needs one multiply-add more rather than a second
// exponential to compose two transmittances. Being flat is its only real difference from the height fog —
// no stretch of air stays clear of it — which is also why the marched tier's span clamp has to stand down
// while it is live.
//
// The ramp itself is a per-frame scalar the CPU reads off WorldBorder: the border is not geometry, so a
// shader could not find it, and the distance that matters is the camera's. The haze therefore hangs over
// the whole view instead of falling in a band at the border's own plane, which is the shape vanilla's
// border fog had too.
//
// The one place the flat term is deliberately left out is a leg toward the SUN (fogT_toSunRo and the
// marched tier's sunFogT). Those integrate to the sky, and a medium with no falloff has infinite optical
// depth that way: including it would delete the sun term, and with it every light shaft, at haze levels far
// too thin to see on the view ray. The border's job is to veil what the eye reaches, not to dim the sun.

// The flat extinction the border adds to the air, in 1/block. Zero unless the border is within its ramp.
// Deliberately NOT scaled by the Rayleigh coefficient: the veil is grey mist, not sky, and a haze that
// reddened with distance the way the air does would read as weather rather than as the world ending.
public float boundarySigma(WorldPush push) {
    return push.boundaryFog.x;
}

// Optical depth of the air the fog integrators actually walk: the height profile's per-channel depth
// plus the border's flat share. `ro` is camera-relative throughout the renderer.
//
// The tier's in-scatter rests on integral[T(s)*sigma_t(s) ds] = 1 - T(t) — transmittance's own
// derivative — which holds for ANY profile as long as sigma_s/sigma_t stays at the scattering albedo.
// That is what lets the closed form price the height profile and the flat term together with one
// multiply-add and no march samples.
public float3 airOpticalDepth(float3 ro, float3 rd, float t) {
    return fogOpticalDepth(worldPush.fog0.x, worldPush.fog0.z, worldPush.fog0.y, ro.y, rd.y, t)
            * FOG_RAYLEIGH_COEFF + boundarySigma(worldPush) * t;
}

public float3 airTransmittance(float3 ro, float3 rd, float t) {
    return exp(-airOpticalDepth(ro, rd, t));
}

// ---- The render edge's fade ------------------------------------------------------------------------
//
// The loaded world stops at the render distance, so the far hills end in a hard stepped cliff against the
// sky. This dissolves that edge: a coverage factor that rises smoothly from zero at edgeFade.y to
// edgeFade.x at edgeFade.z, applied as an alpha-over toward the sky behind the ray.
//
// It is deliberately NOT a medium. A medium scatters — albedo, phase, the sun's illuminance, the sky's
// irradiance — so a haze built out of one both lights the view up and leaves the surface untouched, which
// reads as fog laid over everything rather than as terrain going away. A fade only takes the surface away
// and leaves the sky that was already behind it, so the coverage IS the blend weight: no density, no
// optical depth, no in-scatter, and no cost beyond the sky fetch on segments that actually reach the band.
//
// Distances are measured from the EYE rather than along the segment, so a mirror, a pane of glass or a
// refraction showing the far side of the world carries the same fade the direct view does — priced along
// the segment instead, a second bounce would restart the ramp and the boundary would reappear in every
// reflection.
//
// A ray that escapes to the sky needs no special case: the radiance it returns IS the sky, so fading it
// toward the sky is a no-op by construction. That is what lets every segment price the same expression
// without asking how the trace ended.
public float edgeFadeCoverage(float3 ro, float3 rd, float t) {
    float peak = worldPush.edgeFade.x;
    if (peak <= 0.0) {
        return 0.0;
    }
    float begin = worldPush.edgeFade.y;
    // One block of band, so a start fraction at its clamp cannot make the smoothstep's width zero.
    float edge = max(worldPush.edgeFade.z, begin + 1.0);
    // The trace runs in the terrain's REBASED space, where the eye sits at camOffset rather than at the
    // origin: measuring the hit point's own length would describe its distance from the terrain origin, and
    // that moves with the player's coordinates instead of with the view. Subtract the anchor and the
    // quantity is the eye distance the coverage is defined against, on the camera segment and on every
    // bounce alike.
    float3 eyeRelative = ro + rd * t - worldPush.camOffset;
    return peak * smoothstep(begin, edge, length(eyeRelative));
}

// What the fade dissolves into: the SKY ALONG THE RAY ITSELF, via the same lookup the fog's in-scatter
// uses (fogSkyRadiance), so a segment the fade has fully dissolved and a segment fog has made fully
// opaque land on the same colour — the one the miss shader would have returned for that direction. The
// zenith constant this lookup replaced could not do that: at a sunset the dissolved silhouette stayed
// zenith-blue while the sky behind it went orange, and the wash read as a colour of its own. The mirror
// on rd.y is what licenses direction per ray: the view LUT's lower half is the space floor, and terrain
// dissolving past the edge is haze, not space. (Clouds stay uncomposited here, matching the fog: the
// fill is the clear dome, so a cloud deck overhead does not repaint the horizon's dissolve.)
public float3 edgeFadeSky(float3 rd) {
    return fogSkyRadiance(rd);
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

// The march tier's density at one point. Returns the plain analytic profile while the noise amplitude
// is zero, so the marched integrator degenerates to integrating the analytic field.
// The march tier's density at one point. Returns the plain analytic profile while the noise amplitude
// is zero, so the marched integrator degenerates to integrating the analytic field. It also returns the
// plain profile wherever the profile times the noise's largest multiplier sits below
// FOG_NOISE_MIN_DENSITY: there the 24-lattice-hash FBM can move the sample by less than the density is
// worth, so the modulation is skipped and the segment's optical depth error stays bounded by that
// constant per block of skipped stretch — invisible next to the noise floor the estimator already runs
// at, and the reason the steps above the fog's significant band stop paying for noise.
public static const float FOG_NOISE_MIN_DENSITY = 1.0e-6;

public float fogMarchDensity(WorldPush push, float3 pos) {
    float base = fogDensityAt(push, pos.y);
    float amp = push.fog1.z;
    if (base <= 0.0 || amp <= 0.0 || base * (1.0 + abs(amp)) < FOG_NOISE_MIN_DENSITY) {
        return base;
    }
    // The wind drift arrives pre-accumulated per axis (Java integrates speed x direction over the
    // frame step and wraps each axis in whole 4096-block periods — this lattice's exact period —
    // so neither the hourly clock wrap nor a rebase can snap the field; the old shader-side
    // speed x clock product did exactly that once an hour).
    float2 wind = push.fog1.xy;
    float3 q = (float3(pos.xz + push.waterAnchor.xy, pos.y)
            + float3(wind.x, 0.0, wind.y)) / FOG_NOISE_CELL;
    return base * max(0.0, 1.0 + amp * (fogFbm(q) * 2.0 - 1.0));
}

// Cloud code has two compile-time cut lines. CAUSTICA_NO_VISIBLE_CLOUD_PATH removes the sky layer,
// segment out-scatter and continuation attenuation while retaining the shadow field. Adding
// CAUSTICA_NO_CLOUD_PATH removes the shadow field too, leaving only inert stubs for the fog/shaft
// sun legs and the volume debug image. Both variants keep those callers compiling without carrying
// the shape sampler or its register footprint.
#ifndef CAUSTICA_NO_CLOUD_PATH

// ---- Cloud stack (procedural layered cumulus) ------------------------------------------------------
//
// The layer is a stack of four procedurally generated decks, not a placed asset:
//
//   low    cumulus humilis/mediocris, standing on the Cloud Altitude dial
//   high   cumulus congestus, one gap above the low deck: the towering bodies
//   alto   an altostratus sheet far above both, thin and wide
//   cirrus a curl-warped cirrus sheet higher still
//
// Every deck answers the same three questions the same way, which is what keeps them one system
// instead of four effects:
//
//   coverage   a low-frequency noise field thresholded against the layer's coverage lane
//   profile    a flat base and a domed crown: the horizontal shape is multiplied by a vertical
//              profile that reaches zero at the base and at the crown, and the two together decide
//              where the body is solid
//   erosion    a fourth-power noise cut that eats the isosurface's rim, with a horizontal shear
//              that leans the crown downwind
//
// Densities are authored per block (see the layer table), so a march step's optical depth is the
// sample's shape times its layer's density times the step length — there is no separate calibrated
// column to keep in step. The Cloud Thickness dial scales the two cumulus decks' heights only; the
// horizontal scales are per-layer constants, because a taller deck carrying larger cells is what
// makes the towering layer read as the same sky's weather system rather than as the low deck moved
// up.
//
// Everything below samples one 512-square noise atlas (RtCloudNoise) in tile-relative units, so the
// whole sky is deterministic from the push block: the same coverage, wind and height always produce
// the same cloud, and no two consumers can disagree about where a body stands.

public static const int CLOUD_LAYER_COUNT = 4;
public static const int CLOUD_LOW = 0;
public static const int CLOUD_HIGH = 1;
public static const int CLOUD_ALTO = 2;
public static const int CLOUD_CIRRUS = 3;

// Stack geometry, as multiples of the Cloud Thickness dial. The towering deck's base sits 0.65 dials
// above the low deck's and stands 1.21 dials deep against the low deck's 0.56, so at the default 232
// the pair spans 312..442 and 463..744: two decks that overlap in silhouette without ever sharing a
// body, which is what lets the low layer shade the high one's base from below.
public static const float CLOUD_LOW_TALLNESS = 0.56;
public static const float CLOUD_HIGH_BASE = 0.65;
public static const float CLOUD_HIGH_TALLNESS = 1.21;
// The high sheets are pinned relative to the cumulus base rather than to the world, so raising the
// altitude dial lifts the whole sky instead of pushing the cirrus under the towering deck.
public static const float CLOUD_ALTO_OFFSET = 4688.0;
public static const float CLOUD_CIRRUS_OFFSET = 6688.0;
// Both sheets are five blocks thick and are crossed, never really walked: the shape field is a
// column, so the thickness only decides how much optical depth a vertical sight line picks up.
public static const float CLOUD_SHEET_TALLNESS = 5.0;

// Authored extinction at unit shape, in 1/block, multiplied by the Cloud Density dial. A cumulus body
// is 0.5/block — a two-block mean free path — which is what makes its isosurface solid instead of
// milky. The sheets are two orders of magnitude thinner, and that contrast is the whole difference
// between a deck and a veil.
public static const float CLOUD_LOW_DENSITY = 0.50;
public static const float CLOUD_HIGH_DENSITY = 0.60;
public static const float CLOUD_ALTO_DENSITY = 0.10;
public static const float CLOUD_CIRRUS_DENSITY = 0.02;

// The sheets' coverage, as a fraction of the cumulus lane's saturated value: a high overcast should
// thin out when the player asks for a clear sky, and thicken with the dial, so one number keeps
// meaning "how cloudy".
// The high sheets clear LESS of the mask than the cumulus decks, so they stay sparse filaments. A
// full-coverage sheet was the first thing to turn the sky into one flat grey wash: two thin decks at
// 5000 and 7000 blocks cover a far larger cone of sky than a cumulus body does.
public static const float CLOUD_ALTO_COVERAGE = 1.05;
public static const float CLOUD_CIRRUS_COVERAGE = 0.95;

// Shape: two noise fields added into one mask, thresholded against coverage and normalised by the
// square root of it. The normalisation keeps a half-covered sky's bodies as solid as an overcast's
// instead of fading every cloud uniformly: the threshold moves, the density behind it does not.
//
// The FINE field must carry the local shape. The large field is sampled at a 20 km tile, so within
// one view it is essentially a constant: giving it most of the weight made the whole sky clear or the
// whole sky overcast depending on where the player stood (a 130% coverage sky produced one square
// cloud and nothing else). The fine field has a 2 km tile and std 0.19, so it puts several distinct
// bodies in the same view; the large field only modulates which neighbourhood gets more of them.
// Linear fine, not squared: squaring flattened its low end and handed the decision back to `big`.
// Mean mask ~0.98 at the authored 100% — 0.7 nearly clear, 1.0 broken cumulus, 1.2 two thirds
// covered, 1.8 solid overcast.
public static const float CLOUD_MASK_LARGE = 0.60;
public static const float CLOUD_MASK_FINE = 1.30;
public static const float CLOUD_MASK_BIAS = 0.10;
// Patchiness trades the large field's weight against the fine one's: at 0 the silhouette is shredded
// evenly by small cells, at 1 whole weather systems decide their neighbourhood. Kept small — a wide
// swing turns the dial into an empty/full switch instead of a clustering control.
public static const float CLOUD_MASK_PATCH = 0.35;
public static const float CLOUD_SHAPE_SOFTNESS = 0.20;
// The vertical profile. BASE_ROUND is how many blocks the underside curves in over; TOP_POWER is the
// dome's taper — a smooth 1-(1-t)^p falls away through the whole upper half of the body instead of
// cutting off in the last few blocks, which is what keeps a crown from reading as a flat lid.
public static const float CLOUD_BASE_ROUND = 26.0;
public static const float CLOUD_TOP_POWER = 2.2;
// Fourth-power erosion depth, and the share of it that comes from the base carve. The exponent is
// what turns smooth noise into a cauliflower rim: a linear cut leaves the isosurface's own blur on
// the silhouette, and that blur is what reads as volumetric fog instead of cloud.
public static const float CLOUD_EROSION = 0.55;
public static const float CLOUD_EROSION_BASE = 0.75;
// Horizontal shear: the crown leans up to this many blocks downwind at the deck's top, on a 1.5-power
// ramp. Without it a deck is a prism; with it, a congestus leans the way the real one does.
public static const float CLOUD_SHEAR = 26.0;

// Tiles, in blocks per noise period. The cumulus pair deliberately differs: 5 km cells with 700 m
// detail for the low deck puts a body's width within a few multiples of its height — the aspect
// real cumulus have, which the shipped 20 km tables did not (a 1000-5000 block body on a
// 130-block deck read as one flat plate from any raised camera). The towering deck keeps the same
// proportion at its own scale.
public static const float CLOUD_LOW_FIELD_TILE = 5000.0;
public static const float CLOUD_LOW_DETAIL_TILE = 700.0;
public static const float CLOUD_HIGH_FIELD_TILE = 9000.0;
public static const float CLOUD_HIGH_DETAIL_TILE = 1800.0;
public static const float CLOUD_ALTO_FIELD_TILE = 100000.0;
public static const float CLOUD_ALTO_DETAIL_TILE = 2600.0;
public static const float CLOUD_CIRRUS_FIELD_TILE = 60000.0;
public static const float CLOUD_CIRRUS_DETAIL_TILE = 4200.0;
// One cell of the erosion field, in blocks. Small enough that a rim carries several cells across a
// march step, so the cut reads as grain, and large enough that a distant deck does not turn into
// noise the reconstruct cannot resolve.
public static const float CLOUD_EROSION_CELL = 12.0;
// The erosion field's vertical scroll amplifies its v coordinate 193-fold, so one of its texels
// covers only a few dozen blocks along a grazing sight line — far below a pixel at cloud distance.
// Sampling it two mip levels coarser than the shape field removes the comb of fine stripes that
// otherwise appears along every cloud base near the horizon; 48 march steps made it worse, not
// better, because the grain is a texture-frequency problem rather than a step-count one.
public static const float CLOUD_EROSION_MIP = 2.0;
// Third, finer erosion octave, blended into the cut with this weight. The two base octaves land at
// 12 and 6.6 blocks; this one re-uses the same cell field at 3.6, so a rim carries grain at three
// scales instead of reading as one smooth bite.
public static const float CLOUD_EROSION_FINE = 0.35;
// The coverage fields' height shear, in world blocks of address shift per block of height above the
// deck's own base. One diagonal slice through a body's fine field decorrelates within a few blocks
// of climb while the large field needs the whole deck, which is what makes a body billow instead of
// extruding one horizontal blob from base to lid. Anchored to the base, not to the camera, so the
// pattern is world-fixed whatever altitude the viewer takes. The rate stays low on purpose: the
// fine field's features are only a few blocks across at the shipped tiles, so even this rate
// decorrelates them fully, and a higher one stretched those features into the near-horizontal
// streaks a side view reads as layers.
public static const float2 CLOUD_FIELD_SCROLL = float2(1.5, -1.1);
// Tiling suppression for every coverage field: a warp fetched at an incommensurate fraction of the
// field's own period (0.27 — the ratio 1/0.27 must stay clear of any near-integer, or the warp lands
// on a near-same phase one field period away and the wrap shows through), displacing the sample by
// this fraction of the period at most. The composite of field and warp then repeats only at a period
// no sky ever reaches, so the same blob cluster never recurs at the tile interval the wrap alone
// would give it.
public static const float CLOUD_WARP_TILE = 0.27;
public static const float CLOUD_WARP_AMOUNT = 0.18;
// The interior texture depth, from the fine fetch's spare alpha channel: how far below its local
// value a solid body's density may swing. The floor keeps the body solid — this shades a crown, it
// does not hollow it.
public static const float CLOUD_INNER_TEXTURE = 0.28;
// Per-column crown height. A column's own top comes from the coverage mask read at one fixed
// reference height (a fraction of the deck's tallness), so the fetch never moves with the sample's
// y and a body domes with its own shape: the crown rises where the column is most saturated and
// slopes down at its rim, instead of every column ending on the same horizontal lid — the flat
// plate a flying camera reads as a cut-off slab.
public static const float CLOUD_CROWN_REF = 0.35;
public static const float CLOUD_HEIGHT_RANGE = 0.55;
public static const float CLOUD_CROWN_MIN = 0.25;
public static const float CLOUD_CROWN_POWER = 0.8;
// Curl strength on the fine field, as a fraction of the detail tile. The value-noise lattice puts
// every blob's centre on a regular grid, and a sky of grid-aligned bodies reads as a tiled texture
// however well the wrap itself is broken; the swirl scatters the centres and chews up any streak
// the height shear leaves behind.
public static const float CLOUD_FINE_CURL = 0.10;
// Target march step for the volumetric tier, in blocks: two erosion cells. A coarser step puts whole
// cells behind one sample and quantizes the body's surface to the step length; a finer one buys
// aliasing samples, not shape.
public static const float CLOUD_MARCH_STEP = 24.0;
// The light leg's edge fade and cap, in optical depths. The leg's sample ramps to zero over this many
// blocks inside the slab, so its contribution vanishes continuously as the sun crosses a deck's lid or
// floor instead of drawing a brightness step there; the leg is capped once it is deep enough that
// nothing downstream of the normalised shadow mask can move.
public static const float CLOUD_LIGHT_EDGE = 16.0;
public static const float CLOUD_LIGHT_BREAK = 16.0;
// The dissipation distance the two high sheets ride, as a multiple of the cumulus fade: a sheet at
// 5000 blocks is crossed tens of kilometres out, and the cumulus fade would erase it before it
// reached the horizon band it exists to paint.
public static const float CLOUD_SHEET_FADE_MULT = 8.0;
// Assumed angular size of one render pixel, in radians: the mip the shape field is sampled at is
// derived from it. Advisory only — a coarser device gets a slightly sharper cloud, never a crawling
// one, because the erosion is already faded past its own cell size.
public static const float CLOUD_TEXEL_ANGLE = 0.0011;
// Height inside the low deck the single-sample consumers read the field at, as a fraction of its
// tallness: the mass-weighted centre of the profile. The planar sheet's crossing, the shadow probes
// and the continuation-ray transmittance all use it, so neither switching tiers nor flying in and
// out of the walk moves the cloud.
public static const float CLOUD_SHEET_HEIGHT = 0.35;

public static const float3 CLOUD_ALBEDO = float3(0.86, 0.87, 0.90); // BT-709-linear, sky.rmiss space
// The direct look level is pushed at 128000, well above what the exposure can resolve on a cloud top.
// Scaling it before the powder and shadow stack is what lets the crown keep a gradient instead of
// clipping to a flat white.
public static const float CLOUD_DIRECT_GAIN = 0.65;
// How much of the column above a sample occludes the SKY that lights it, as a share of the direct
// leg's own occlusion. Light reaches a deck's base from the whole hemisphere, so this stays below
// the sun leg — but it is the only term that can darken a base, and a value small enough to leave
// the base within a few percent of the crown flattened the whole deck into one white sheet.
public static const float CLOUD_AMBIENT_OCCLUSION = 0.45;
// The sky term is the only light a thick base receives: its sun leg saturates the multi-scatter
// stack, and what reaches a viewer under the deck is light that entered through the sides and the
// gaps. FLOOR is the share of the unoccluded sky a sample keeps at any depth; GAIN undoes the
// irradiance-to-radiance divide for a scattering medium.
public static const float CLOUD_AMBIENT_FLOOR = 0.15;
public static const float CLOUD_AMBIENT_GAIN = 2.0;
// The light leg returns an OPTICAL DEPTH: a 130-block cumulus at 0.5/block puts tens of them on even
// the shortest leg. The multi-scatter gain needs a normalised 0..1 shadow mask, so the leg is divided
// by this depth before 1-exp(-tau) turns it into one. Eight depths is where a crown and a base part
// company: the top of a body still sees light, its underside does not.
public static const float CLOUD_LIGHT_DEPTH_SCALE = 8.0;

// ---- Shape field -----------------------------------------------------------------------------------

/** One deck's geometry and material, resolved from the push block. */
public struct CloudLayer {
    public int   kind;        // 0 cumulus, 1 altostratus sheet, 2 cirrus sheet
    public float base;        // rebased blocks
    public float top;
    public float tallness;
    public float fieldTile;
    public float detailTile;
    public float coverage;
    public float density;
    public float windScale;
};

float cloudBoxHeight(WorldPush push) {
    return clamp(push.cloud0.z, 8.0, 1024.0);
}

float cloudBoxBottom(WorldPush push) {
    return push.cloud0.y;
}

CloudLayer cloudLayerOf(WorldPush push, int index) {
    float thickness = cloudBoxHeight(push);
    float base = cloudBoxBottom(push);
    // The dial is the threshold; weather adds to it, which is how rain decks the sky — the bodies
    // grow into each other instead of a different field being drawn.
    float coverage = clamp(push.cloud0.x + push.cloud1.y, 0.0, 2.5);
    float density = clamp(push.cloud0.w, 0.0, 4.0);
    // Cloud Size scales every tile, so a dial that makes the bodies larger makes them larger in every
    // deck at once. The layer tables' tiles are the 1.0 reference.
    float sizeScale = clamp(push.cloudColor.x, 0.25, 4.0);
    CloudLayer l;
    l.kind = 0;
    l.base = base;
    l.tallness = thickness * CLOUD_LOW_TALLNESS;
    l.fieldTile = CLOUD_LOW_FIELD_TILE * sizeScale;
    l.detailTile = CLOUD_LOW_DETAIL_TILE * sizeScale;
    l.coverage = coverage;
    l.density = CLOUD_LOW_DENSITY * density;
    l.windScale = 1.0;
    if (index == CLOUD_HIGH) {
        l.base = base + thickness * CLOUD_HIGH_BASE;
        l.tallness = thickness * CLOUD_HIGH_TALLNESS;
        l.fieldTile = CLOUD_HIGH_FIELD_TILE * sizeScale;
        l.detailTile = CLOUD_HIGH_DETAIL_TILE * sizeScale;
        l.density = CLOUD_HIGH_DENSITY * density;
        l.windScale = 1.35;
    } else if (index == CLOUD_ALTO) {
        l.kind = 1;
        l.base = base + CLOUD_ALTO_OFFSET;
        l.tallness = CLOUD_SHEET_TALLNESS;
        l.fieldTile = CLOUD_ALTO_FIELD_TILE * sizeScale;
        l.detailTile = CLOUD_ALTO_DETAIL_TILE * sizeScale;
        l.coverage = CLOUD_ALTO_COVERAGE * saturate(coverage);
        l.density = CLOUD_ALTO_DENSITY * density;
        l.windScale = 2.2;
    } else if (index == CLOUD_CIRRUS) {
        l.kind = 2;
        l.base = base + CLOUD_CIRRUS_OFFSET;
        l.tallness = CLOUD_SHEET_TALLNESS;
        l.fieldTile = CLOUD_CIRRUS_FIELD_TILE * sizeScale;
        l.detailTile = CLOUD_CIRRUS_DETAIL_TILE * sizeScale;
        l.coverage = CLOUD_CIRRUS_COVERAGE * saturate(coverage);
        l.density = CLOUD_CIRRUS_DENSITY * density;
        l.windScale = 1.8;
    }
    l.top = l.base + l.tallness;
    return l;
}

// The height the single-sample consumers read the field at: CLOUD_SHEET_HEIGHT up the low deck.
public float cloudBoxMid(WorldPush push) {
    return cloudBoxBottom(push) + CLOUD_SHEET_HEIGHT * cloudLayerOf(push, CLOUD_LOW).tallness;
}

// Mip level for a shape sample `dist` blocks away: one texel of the layer's field should never be
// smaller than a pixel. Past the erosion's own cell the cut is faded out instead, so this only has
// to keep the distant decks from crawling.
float cloudLod(float dist, float tile) {
    return clamp(log2(max(dist, 1.0) * CLOUD_TEXEL_ANGLE * 512.0 / tile), 0.0, 5.0);
}

/** One sample of the noise atlas. Channels: r fine detail, g/b the cell field, a the interior texture. */
float4 cloudNoiseSample(float2 uv, float lod) {
    return cloudNoise.SampleLevel(uv, lod);
}

// The coverage fields' shared warp: one fetch at an incommensurate fraction of the period of the
// tile the caller is about to sample, returned as a fraction of that tile. Each field warps at its
// OWN scale — a warp shared from the field tile would leave the detail field's own wrap showing
// through, because a smooth displacement barely moves across one detail period.
float2 cloudFieldWarp(float2 xz, float tile, float2 wind) {
    return (cloudNoiseSample((xz + wind) / (tile * CLOUD_WARP_TILE), 1.0).rg - 0.5)
            * CLOUD_WARP_AMOUNT;
}

// A coverage field's sample address: the deck's wind offset, the height shear above its own base,
// and the warp scaled to this field's tile. Every big/fine fetch goes through here — including the
// march's cheap rejection — or the rejection and the density it prunes would disagree about what
// the field says.
float2 cloudFieldAddress(float2 xz, float aboveBase, float tile, float2 wind, float2 warp) {
    return xz + wind + aboveBase * (CLOUD_FIELD_SCROLL / tile) + warp * tile;
}

// The erosion field: one atlas texel per lattice cell, two independently scrolled channels summing
// to a field that varies in all three axes without a 3D texture.
//
// The vertical offset MUST use the CONTINUOUS p.y. Flooring it to an integer cell and stepping the
// atlas row by 193 per cell put a hard discontinuity every cell — the cross-section came out as a
// layer cake of horizontal stripes — because no amount of bilinear filtering smooths a jump. A
// continuous diagonal scroll through the same tile keeps the field 3D-looking and seamless; keying a
// channel blend off frac(p.y) instead left a derivative kink at every cell boundary, which read as
// faint horizontal stripes across a deck's cross-section.
float cloudCellNoise(float3 p, float lod) {
    float2 uv = (p.xz + p.y * float2(0.0, 193.0)) * (1.0 / 512.0);
    float a = cloudNoiseSample(uv, lod + CLOUD_EROSION_MIP).g;
    // A second, independent scroll along the same tile keeps the two channels from being one field
    // sampled twice. Both offsets are continuous, so neither introduces a cell boundary's stripe.
    float2 uv2 = (p.xz * 1.7 + p.y * float2(11.0, 193.0)) * (1.0 / 512.0);
    float b = cloudNoiseSample(uv2, lod + CLOUD_EROSION_MIP).b;
    return a * 0.6 + b * 0.4;
}

// 2D gradient hash for the cirrus warp: two independent members per lattice point, normalised so the
// curl below is scale-free.
float2 cloudGradient(float2 p) {
    float3 p3 = frac(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    float2 h = frac((p3.xx + p3.yz) * p3.zy);
    return normalize(h - 0.5 + float2(1.0e-4, 1.0e-4));
}

// Curl of the gradient field: divergence-free, so the warp swirls the cirrus into filaments instead
// of smearing it into a blob.
float2 cloudCurl(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float2 g0 = cloudGradient(i);
    float2 g1 = cloudGradient(i + float2(1.0, 0.0));
    float2 g2 = cloudGradient(i + float2(0.0, 1.0));
    float2 g3 = cloudGradient(i + float2(1.0, 1.0));
    float v0 = dot(g0, f);
    float v1 = dot(g1, f - float2(1.0, 0.0));
    float v2 = dot(g2, f - float2(0.0, 1.0));
    float v3 = dot(g3, f - float2(1.0, 1.0));
    return float2(
            ((v1 - v0) * (1.0 - u.y) + (v3 - v2) * u.y) * du.x,
            ((v2 - v0) * (1.0 - u.x) + (v3 - v1) * u.x) * du.y);
}

// The horizontal mask every deck shares: a large field plus a squared fine one, thresholded against
// the layer's coverage. The coverage is normalised out of the result, so the body behind the
// threshold is as solid at 0.4 coverage as at 1.6.
float cloudMaskLarge(float patchiness) {
    return CLOUD_MASK_LARGE * (1.0 + CLOUD_MASK_PATCH * (saturate(patchiness) * 2.0 - 1.0));
}

float cloudMaskFine(float patchiness) {
    return CLOUD_MASK_FINE * (1.0 - CLOUD_MASK_PATCH * (saturate(patchiness) * 2.0 - 1.0));
}

float cloudCoverageMask(float big, float fine, float cov, float patchiness) {
    float mask = cloudMaskLarge(patchiness) * big + cloudMaskFine(patchiness) * fine + CLOUD_MASK_BIAS;
    // A narrow smoothstep, not the linear ramp: inside the threshold the body is SOLID (shape 1) and
    // outside it there is clear sky. The linear ramp left every deck a smooth grey wash with holes —
    // fog, not cumulus — no matter how high the extinction was behind it.
    float q = saturate(((cov - mask) * rsqrt(max(cov, 1.0e-3))) / CLOUD_SHAPE_SOFTNESS);
    return q * q * (3.0 - 2.0 * q);
}

// Cumulus: coverage threshold, then the vertical profile, then the erosion cut. `wind` is the layer's
// own drifted XZ offset in blocks, `lod` the distance-derived sample level. `warp` and `big` are the
// coverage field's own fetch, made by the caller: the march rejects with `big` before it pays for the
// profile, so handing that value back in is what keeps a hit from fetching the same field twice.
float cloudCumulusDensity(CloudLayer l, float3 p, float2 wind, float lod, bool eroded,
                          float2 warp, float big) {
    float cov = l.coverage;
    if (cov <= 1.0e-3) {
        return 0.0;
    }
    float aboveBase = p.y - l.base;
    // The flow phase scrolls the FINE field against the body, at its own rate: the body is carried by
    // the wind, the detail inside it billows, and the two never move as one sheet. The same fetch's
    // alpha is an independent octave stack reserved for the interior texture.
    float2 flow = float2(worldPush.cloudColor.z * 1.7, -worldPush.cloudColor.z * 1.1);
    float2 fineWind = flow - wind * 0.6;
    float2 fineWarp = cloudFieldWarp(p.xz, l.detailTile, fineWind);
    float2 fineAddr = cloudFieldAddress(p.xz, aboveBase, l.detailTile, fineWind, fineWarp);
    // The curl displaces the fine address with a divergence-free swirl so the lattice's blob
    // centres stop landing on a grid. It is part of the shape field every consumer reads, shadow
    // field and light legs included, so the detail dial leaves it on: a body's coverage, crown and
    // base — and the shadow it casts — must not move under a close-view switch.
    float2 curl = cloudCurl(fineAddr / l.detailTile) * CLOUD_FINE_CURL;
    bool detailOn = worldPush.cloudColor.y > 0.5;
    float4 detail = cloudNoiseSample((fineAddr + curl * l.detailTile) / l.detailTile, lod);
    float fine = 1.0 - detail.r;
    float shape = cloudCoverageMask(big, fine, cov, worldPush.cloud1.z);
    if (shape <= 0.0) {
        return 0.0;
    }
    // Per-column crown: the column's own top comes from the coverage mask read at one fixed
    // reference height, so the height field never moves with the sample's y and a body domes
    // with its own shape — crown rising where the column is most saturated, sloping down at its
    // rim — instead of every column ending on the deck's one horizontal lid.
    float2 refAddr = cloudFieldAddress(p.xz, l.tallness * CLOUD_CROWN_REF, l.detailTile,
            fineWind, fineWarp);
    float refFine = 1.0 - cloudNoiseSample((refAddr + curl * l.detailTile) / l.detailTile, lod).r;
    float hmask = cloudMaskLarge(worldPush.cloud1.z) * big
            + cloudMaskFine(worldPush.cloud1.z) * refFine + CLOUD_MASK_BIAS;
    float hexc = saturate(((cov - hmask) * rsqrt(max(cov, 1.0e-3))) / CLOUD_HEIGHT_RANGE);
    float colTall = max(l.tallness * (CLOUD_CROWN_MIN
            + (1.0 - CLOUD_CROWN_MIN) * pow(hexc, CLOUD_CROWN_POWER)), 1.0);
    float below = p.y - l.base;
    float above = l.base + colTall - p.y;
    // Close the body on its own silhouette before the profile runs: a sample outside the box has no
    // density whatever the column says, and these two clamps are what make the lid and the floor read
    // as the cloud's own boundaries rather than as cuts through it.
    shape = min(min(shape, saturate(above)), saturate(below));
    // The underside curves in over the first few tens of blocks, so a base looks like a base.
    float bottom = 1.0 - pow(1.0 - saturate(below / CLOUD_BASE_ROUND), 5.0);
    // The crown tapers through the upper half of the column instead of cutting off at its top: at
    // the top the term is zero, so the subtraction below reads "no density" there, and a cloud seen
    // from below closes on a rounded crown rather than on a plane.
    float top = saturate(above / colTall);
    top = 1.0 - pow(1.0 - top, CLOUD_TOP_POWER);
    shape = max(shape - 1.0 + top * bottom, 0.0);
    if (shape <= 1.0e-3) {
        return 0.0;
    }
    // Interior texture: the thresholded mask saturates to a solid body, so without this modulation a
    // crown is one smooth dome. The alpha octave swings the density while keeping the body solid —
    // the erosion below bites where this made the body thinner, and the two read as one billow.
    shape *= 1.0 - CLOUD_INNER_TEXTURE * (1.0 - detail.a);
    if (!eroded || !detailOn) {
        // The shadow field and the far decks read the un-eroded shape, shrunk by the mean depth of
        // the cut so the shadow a deck casts matches the body a close look sees; the detail-off close
        // view lands here too, skipping the three erosion fetches below.
        return max(shape - CLOUD_EROSION * 0.5, 0.0);
    }
    // Wind shear: the crown leans downwind on a 1.5-power ramp measured from 20 blocks above the
    // base, so the bottom stays planted while the top drifts.
    float3 q = p;
    q.xz += pow(max(below - 20.0, 0.0) / max(l.tallness * 0.2, 1.0), 1.5) * CLOUD_SHEAR;
    q.xz -= wind * 0.25;
    float falloff = 1.0 - saturate(above / max(colTall * 0.75, 1.0));
    float e1 = 1.0 - cloudCellNoise(q / CLOUD_EROSION_CELL, lod);
    float e2 = cloudCellNoise(q / (CLOUD_EROSION_CELL * 0.55), lod);
    float e3 = 1.0 - cloudCellNoise(q / (CLOUD_EROSION_CELL * 0.30), lod);
    float om = 1.0 - shape;
    float erosion = e1 * sqrt(om);
    erosion += abs(e2 - falloff) * CLOUD_EROSION_BASE * om * om * (1.0 - falloff * 0.5);
    erosion += e3 * CLOUD_EROSION_FINE * om * (1.0 - falloff * 0.5);
    erosion = erosion * erosion;
    erosion *= erosion;
    return max(shape - erosion * CLOUD_EROSION, 0.0);
}

// A high sheet. Altostratus is a wide, soft, low-contrast field whose threshold is squared, so the
// sheet reads denser in its middle than at its edges; cirrus is the same threshold over a
// curl-warped detail, which is what turns a smooth sheet into filaments.
float cloudSheetDensity(CloudLayer l, float3 p, float2 wind, float lod) {
    float cov = l.coverage;
    if (cov <= 1.0e-3) {
        return 0.0;
    }
    float aboveBase = p.y - l.base;
    float2 warp = cloudFieldWarp(p.xz, l.fieldTile, wind);
    float2 bigAddr = cloudFieldAddress(p.xz, aboveBase, l.fieldTile, wind, warp);
    float big = cloudNoiseSample(bigAddr / l.fieldTile, lod).b;
    float fine;
    if (l.kind == 2) {
        // The curl warps the DETAIL sampling coordinates, so a cirrus sheet breaks into filaments that
        // flow rather than the smooth blobs a plain threshold would give it. The detail dial gates it
        // like the cumulus erosion: off keeps the filament threshold but skips the three curl fetches.
        float2 c = (p.xz - wind) / l.detailTile;
        float2 curl = float2(0.0, 0.0);
        if (worldPush.cloudColor.y > 0.5) {
            curl = cloudCurl(c * 3.0) * 0.5 + cloudCurl(c * 7.0) * 0.25 + cloudCurl(c * 19.0) * 0.125;
        }
        float a = 1.0 - cloudNoiseSample(c * 0.6 + curl * 0.35, lod + 1.0).r;
        float b = 1.0 - cloudNoiseSample(c * 2.3 + curl * 0.9, lod + 1.0).b;
        fine = a * 0.6 + b * 0.4;
    } else {
        // The red channel, not the blue the big field reads: one channel at two scales would threshold
        // the same blobs twice and the sheet would carry no detail of its own.
        float2 fineWind = -wind * 1.4;
        float2 fineWarp = cloudFieldWarp(p.xz, l.detailTile, fineWind);
        float2 fineAddr = cloudFieldAddress(p.xz, aboveBase, l.detailTile, fineWind, fineWarp);
        fine = 1.0 - cloudNoiseSample(fineAddr / l.detailTile, lod).r;
    }
    // Every deck, sheets included, clears the same threshold. The numbers differ (their own coverage
    // and density lanes) but the shape language is one, which is why a high sheet reads as weather
    // rather than as a different effect pasted over the cumulus.
    float shape = cloudCoverageMask(big, fine, cov, worldPush.cloud1.z);
    float below = p.y - l.base;
    float above = l.top - p.y;
    if (below <= 0.0 || above <= 0.0) {
        return 0.0;
    }
    // A sheet's own slab: two blocks of ramp so the plane has an edge instead of a step.
    shape = min(shape, saturate(above * 0.4));
    return min(shape, saturate(below * 0.4));
}

// The coverage fields' own fetch for a cumulus sample, for the callers that need it without paying for
// the profile and the erosion: the march's column rejection, and the light leg's slab midpoint.
float cloudCumulusMaskBig(CloudLayer l, float3 p, float2 wind, float lod, out float2 warp) {
    warp = cloudFieldWarp(p.xz, l.fieldTile, wind);
    float2 addr = cloudFieldAddress(p.xz, p.y - l.base, l.fieldTile, wind, warp);
    return cloudNoiseSample(addr / l.fieldTile, lod).b;
}

// Density 0..1 at one point of one deck.
public float cloudLayerDensity(WorldPush push, int index, float3 p, float2 wind, float lod, bool eroded) {
    CloudLayer l = cloudLayerOf(push, index);
    if (l.kind == 0) {
        float2 warp;
        float big = cloudCumulusMaskBig(l, p, wind, lod, warp);
        return cloudCumulusDensity(l, p, wind, lod, eroded, warp, big);
    }
    return cloudSheetDensity(l, p, wind, lod);
}

// XZ wind offset for one deck, in blocks. The Java drift is one world-space offset; each deck scales
// it so the stack shears instead of sliding as a rigid sheet.
float2 cloudLayerWind(WorldPush push, int index) {
    CloudLayer l = cloudLayerOf(push, index);
    return (push.cloud2.xy + push.cloud2.zw) * l.windScale;
}

// ---- Light ---------------------------------------------------------------------------------------

// The phase a sample is lit by: four Henyey-Greenstein lobes interpolated by the powder term. The
// strong forward lobes are the sun-facing rim of a body; the weak ones carry the flat middle; the
// back lobe is the silver lining when the body sits between the eye and the sun. A constant blend of
// two lobes cannot do this — the powder weight is what makes a thin rim and a thick core light
// differently at the same view angle.
float cloudPhase4(float cosTheta, float powder) {
    float a = lerp(hg(cosTheta, 0.80), hg(cosTheta, 0.55), powder);
    float b = lerp(hg(cosTheta, 0.35), hg(cosTheta, 0.10), powder);
    return lerp(a, b, powder) + powder * 2.0 * hg(-cosTheta, 0.25);
}

// Density-based Beer-Powder: the sample's own density, not the view angle or the light leg, decides
// how much of the strong forward lobe it keeps. A thin rim at the isosurface is lit almost entirely by
// the forward scatter (the silver lining); a solid interior sits on the flat lobes. Bounded above so a
// thin sample cannot be multiplied past the direct level the look constant sets.
float cloudPowder(float shape) {
    return min(exp(-4.0 * exp(-4.0 * shape)) * 3.5, 1.0);
}

// The one body lighting the stack, resolved once per ray so no deck can disagree about the hour.
public struct CloudLight {
    public float3 dir;
    public float3 direct;     // illuminance * tint * gain
    public float3 ambient;    // sky irradiance term
    public float  cosTheta;   // dot(view, light)
};

// The colour the pushed illuminance is tinted with, and how far it may be pulled off white. The
// derived body's colour is normalised to unit LUMINANCE rather than to its brightest channel, so the
// deck keeps the level the look constant means and only its hue moves: normalising to the brightest
// channel would dim the deck by the colour's own luminance — 0.42 for a one-degree sun — and a warm
// deck two stops down reads as brown.
public static const float CLOUD_LIGHT_TINT = 0.5;

public float3 cloudLightTint(float3 derivedIlluminance) {
    float lum = dot(derivedIlluminance, float3(0.2126, 0.7152, 0.0722));
    if (lum <= 1.0e-9) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 chroma = derivedIlluminance / lum;
    return lerp(float3(1.0, 1.0, 1.0), chroma, CLOUD_LIGHT_TINT);
}

// The elevation the deck's direct level rides. cloudLightTint returns unit luminance by construction,
// so without this ramp the deck is lit at the pushed look constant from noon all the way to the
// horizon gate and then loses the entire direct term in one step — and that step lands on the frame
// where dominantCelestialLight hands the light to the moon, 128000 lux to 5.
public static const float CLOUD_LEVEL_ZERO_ALT = -0.012;  // cos zenith, -0.69 degrees
public static const float CLOUD_LEVEL_FULL_ALT = 0.080;   // cos zenith, +4.6 degrees

float cloudDaylightLevel(float sunDirY) {
    return smoothstep(CLOUD_LEVEL_ZERO_ALT, CLOUD_LEVEL_FULL_ALT, sunDirY);
}

CloudLight cloudLightOf(SkyState state, float3 rd) {
    CloudLight light;
    CelestialLight body = dominantCelestialLight(state, transmittanceLut);
    light.dir = body.dir;
    bool sunDominant = dot(body.dir, state.sunDir) >= dot(body.dir, state.moonDir);
    float level = sunDominant ? state.sunIlluminance : state.moonIlluminance;
    if (sunDominant) {
        level *= cloudDaylightLevel(state.sunDir.y);
    }
    light.direct = level * cloudLightTint(body.illuminance) * CLOUD_DIRECT_GAIN;
    // The sky term is the HEMISPHERE irradiance the fog and the water read, not the radiance of one
    // direction; INV_PI turns the cos-weighted integral back into the average radiance the albedo
    // multiplies. The weight is clamped because it is a config lane.
    light.ambient = skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0)) * INV_PI
            * clamp(worldPush.cloudColor.w, 0.0, 2.0) * CLOUD_AMBIENT_GAIN;
    light.cosTheta = dot(rd, body.dir);
    return light;
}

// One sample's radiance. `shape` is the density 0..1, `tauSun` the optical depth toward the body and
// `height01` the sample's position inside the deck (0 at the base, 1 at the crown).
//
// The direct term is the multi-scatter approximation, but its exponent takes a NORMALISED shadow
// mask, not the raw optical depth: a 130-block cumulus at 0.5/block puts tens of optical depths on
// even the shortest light leg, and exp(-5·tau) would be zero everywhere — the whole sky renders as
// one flat, ambient-only grey sheet. Dividing by CLOUD_LIGHT_DEPTH_SCALE before 1-exp(-tau) maps that
// leg back to [0, 1], so an unshadowed rim keeps 6.28 of the bounced-light gain, a deep core keeps
// 6.28·exp(-5) and the deck keeps a lit crown over a darker base instead of a uniform slab.
float3 cloudSampleRadiance(float shape, float tauSun, float height01, CloudLight light) {
    float powder = cloudPowder(shape);
    float phase = cloudPhase4(light.cosTheta, powder);
    float sunMask = 1.0 - exp(-tauSun / CLOUD_LIGHT_DEPTH_SCALE);
    float beer = 6.28 * exp(-5.0 * sunMask);
    float3 direct = beer * light.direct * phase;
    float occl = height01 * height01;
    float ambOcclusion = lerp(CLOUD_AMBIENT_FLOOR, 1.0,
            lerp(1.0, exp2(-5.0 * saturate(shape)), occl * CLOUD_AMBIENT_OCCLUSION));
    float3 ambient = light.ambient * ambOcclusion;
    return CLOUD_ALBEDO * (direct + ambient);
}

// ---- Sampling the stack along a ray ----------------------------------------------------------------

// The stretch of [0, tMax] a ray spends inside one deck, or an empty range.
bool cloudLayerSpan(CloudLayer l, float3 ro, float3 rd, float tMax, out float t0, out float t1) {
    t0 = 0.0;
    t1 = 0.0;
    if (abs(rd.y) < 1.0e-6) {
        // A ray parallel to a sheet never enters it unless it starts inside.
        if (ro.y <= l.base || ro.y >= l.top) {
            return false;
        }
        t1 = tMax;
        return t1 > 0.0;
    }
    float ta = (l.base - ro.y) / rd.y;
    float tb = (l.top - ro.y) / rd.y;
    t0 = max(min(ta, tb), 0.0);
    t1 = min(max(ta, tb), tMax);
    return t1 > t0;
}

// The optical depth a sample sees toward the body through its own deck: the stretch of the sun ray
// that still lies between the deck's base and its crown, priced by one density sample at the midpoint
// of each of the leg's sub-segments. The span is clipped analytically rather than sampled, so the leg
// stays continuous where the sun crosses a deck's lid or floor — a sampled span would step the shadow
// mask as the crossing moves. The sample count is the Cloud Light Steps dial (cloud1.x, clamped to
// 1..8): 1 is the single midpoint tap the tier shipped with, more resolve the light gradient through
// the body. `CLOUD_LIGHT_BREAK` caps the leg, whose only consumer is the normalised multi-scatter mask.
float cloudSunTau(WorldPush push, int index, CloudLayer l, float3 p, float2 wind, float3 ro,
                  float3 sunDir) {
    float tallness = max(l.tallness, 1.0);
    float below = p.y - l.base;
    float legLen = max(tallness, 24.0);
    if (abs(sunDir.y) > 1.0e-6) {
        // Distance to the plane the leg leaves through, measured from the sample: the crown for a
        // rising sun, the base for a setting one.
        float exit = sunDir.y > 0.0 ? (tallness - below) / sunDir.y : below / (-sunDir.y);
        legLen = min(legLen, max(exit, 0.0));
    }
    if (legLen <= 0.0) {
        return 0.0;
    }
    int steps = int(clamp(push.cloud1.x, 1.0, 8.0));
    float dt = legLen / float(steps);
    float tau = 0.0;
    for (int i = 0; i < steps; i++) {
        float3 lp = p + sunDir * ((float(i) + 0.5) * dt);
        float v = lp.y - l.base;
        // Each tap's weight ramps to zero over its last blocks inside the slab, so the leg's result
        // is continuous in p wherever it meets the deck's lid or floor.
        float edge = min(v, tallness - v);
        float tap = cloudLayerDensity(push, index, lp, wind, cloudLod(length(lp - ro), l.detailTile),
                false) * l.density * dt;
        tau += tap * saturate(edge / CLOUD_LIGHT_EDGE);
    }
    return min(tau, CLOUD_LIGHT_BREAK);
}

// March one deck between t0 and t1 and composite it into the running (L, T). `steps` is this deck's
// share of the budget, `jitter` the per-pixel phase, `fade` the dissipation applied to its opacity.
void cloudMarchLayer(WorldPush push, int index, float3 ro, float3 rd, float t0, float t1,
                     float steps, float jitter, float fade, CloudLight light,
                     inout float3 L, inout float T) {
    CloudLayer l = cloudLayerOf(push, index);
    if (l.coverage <= 1.0e-3) {
        return;
    }
    float2 wind = cloudLayerWind(push, index);
    float dt = (t1 - t0) / max(steps, 1.0);
    float tallness = max(l.tallness, 1.0);
    for (int i = 0; i < int(steps); i++) {
        float t = t0 + (float(i) + jitter) * dt;
        float3 p = ro + rd * t;
        float dist = length(p - ro);
        float lod = cloudLod(dist, l.detailTile);
        // One warp fetch and one field fetch reject the empty majority of the sky before the profile,
        // the erosion and the light leg are paid for. The address is the density's own, through the
        // shared helpers, so the rejection and the fetches it prunes cannot disagree — and the fetch
        // it kept is handed to the density below rather than repeated.
        float2 warp;
        float big = cloudCumulusMaskBig(l, p, wind, lod, warp);
        // A lower bound on the mask: the fine term is non-negative, so a big field that already clears
        // the threshold on its own cannot leave any density behind.
        if (cloudMaskLarge(worldPush.cloud1.z) * big + CLOUD_MASK_BIAS >= l.coverage) {
            continue;
        }
        float density = cloudCumulusDensity(l, p, wind, lod, true, warp, big);
        if (density <= 1.0e-4) {
            continue;
        }
        float height01 = saturate((p.y - l.base) / tallness);
        // The vertical weight: the base quarter absorbs far less than the crown, which is what gives
        // a deck a translucent, cooler underside instead of a solid black floor.
        float faded = density * pow(saturate(height01 * 4.0), 2.0);
        float tauCell = faded * l.density * dt;
        if (tauCell <= 1.0e-5) {
            continue;
        }
        float tauSun = cloudSunTau(push, index, l, p, wind, ro, light.dir);
        float3 radiance = cloudSampleRadiance(density, tauSun, height01, light);
        float absorbed = (1.0 - exp(-tauCell)) * fade;
        L += T * absorbed * radiance;
        T *= 1.0 - absorbed;
        if (T < 3.0e-3) {
            return;
        }
    }
}

// The stack's planar compositing, shared by the sky miss shader and the segment out-scatter. `tMax`
// is the reach of the ray: a surface nearer than a deck cannot be veiled by it, so the same function
// serves a ray that escapes and one that stops on geometry.
float3 cloudPlanarStack(WorldPush push, float3 ro, float3 rd, SkyState state, float tMax,
                        out float cloudT) {
    cloudT = 1.0;
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    CloudLight light = cloudLightOf(state, rd);
    float3 L = float3(0.0, 0.0, 0.0);
    float T = 1.0;
    bool upward = rd.y >= 0.0;
    for (int n = 0; n < CLOUD_LAYER_COUNT; n++) {
        int index = upward ? n : CLOUD_LAYER_COUNT - 1 - n;
        CloudLayer l = cloudLayerOf(push, index);
        if (l.coverage <= 1.0e-3) {
            continue;
        }
        float mid = l.base + 0.5 * l.tallness;
        bool below = ro.y < mid;
        if (below == (ro.y > mid)) {
            continue;
        }
        if (below ? rd.y < 1.0e-3 : rd.y > -1.0e-3) {
            continue; // the ray runs away from the plane
        }
        float t = (mid - ro.y) / rd.y;
        if (t <= 0.0 || t >= tMax) {
            continue;
        }
        // Dissipation distance: the visible layer fades over the last 40% of the pushed range and is
        // not rendered past it. The shadow field has no per-ray fade; its relevance is decided by the
        // host's WORLD_FLAG_CLOUD_SHADOW gate.
        float fadeDist = max(push.cloud1.w, 64.0) * (l.kind == 0 ? 1.0 : CLOUD_SHEET_FADE_MULT);
        float fade = 1.0 - smoothstep(fadeDist * 0.6, fadeDist, t);
        if (fade <= 0.0) {
            continue;
        }
        float3 p = ro + rd * t;
        float2 wind = cloudLayerWind(push, index);
        float lod = cloudLod(t, l.detailTile);
        float density = cloudLayerDensity(push, index, p, wind, lod, true);
        if (density <= 0.0) {
            continue;
        }
        // Perspective thickening: a grazing sight line crosses proportionally more body. The
        // self-shadow is the stack's own shadow field at the crossing, so the sheet and the ground it
        // shades agree about how much light is left.
        float tau = density * l.density * l.tallness / max(abs(rd.y), 0.04);
        float shadowT = cloudShadowTransmittance(p, light.dir);
        float tauSun = -log(max(shadowT, 1.0e-3));
        float3 radiance = cloudSampleRadiance(density, tauSun, 0.5, light);
        float opacity = (1.0 - exp(-tau)) * fade;
        L += T * opacity * radiance;
        T *= 1.0 - opacity;
        if (T < 3.0e-3) {
            break;
        }
    }
    cloudT = T;
    return L;
}

// ---- The stack's public surface ---------------------------------------------------------------------

// Transmittance of the stack along a sun leg from `pos` (rebased): each cumulus and alto deck is
// sampled along the leg's stretch through its slab — one tap at the mid-height plane by default,
// the Cloud Shadow Steps dial's count averaged symmetrically around it — and the optical depth is
// thickened by the leg's slant. This is the one estimate the ground-level NEE, the fog sun legs and
// the sheets' own self-shadow share, so a clouded sun dims identically in all of them. Returns 1
// while the shadow field is off — the callers' multiply keeps them bit-identical.
public float cloudShadowTransmittance(float3 pos, float3 sunDir) {
    if ((worldPush.flags & WORLD_FLAG_CLOUD_SHADOW) == 0u || worldPush.cloud0.x <= 0.0) {
        return 1.0;
    }
    // The body's own elevation gate. A sun at the horizon has no light left to shadow, but switching
    // the shadow off at a fixed elevation put a step on every surface the deck shades: at 2.87 degrees
    // the shadow vanished between two frames. Fade it out over the last few degrees instead, and clamp
    // the leg's denominator with the same floor so the tau being blended stays finite.
    float shadowGate = smoothstep(0.0, 0.06, sunDir.y);
    if (shadowGate <= 0.0) {
        return 1.0;
    }
    float denom = max(sunDir.y, 0.02);
    float tau = 0.0;
    // Cirrus is deliberately absent: at 0.005/block over five blocks its column is a few percent, and
    // paying a fetch per NEE site for that is not a trade this field makes.
    for (int index = CLOUD_LOW; index <= CLOUD_ALTO; index++) {
        CloudLayer l = cloudLayerOf(worldPush, index);
        if (l.coverage <= 1.0e-3) {
            continue;
        }
        float mid = l.base + 0.5 * l.tallness;
        float t = (mid - pos.y) / denom;
        if (t <= 0.0) {
            continue;
        }
        float2 wind = cloudLayerWind(worldPush, index);
        // The quality dial's sample count, spread symmetrically around the mid-plane crossing: 1 is
        // the single tap the field shipped with, more average the leg's own density profile through
        // the slab. A sample outside the slab reads zero shape, so the spread needs no clipping.
        int steps = int(clamp(worldPush.volumeParams.z, 1.0, 4.0));
        float legLen = l.tallness / denom;
        float densitySum = 0.0;
        for (int i = 0; i < steps; i++) {
            float ti = t + (float(i) - 0.5 * float(steps - 1)) * (legLen / float(steps));
            float3 p = pos + sunDir * ti;
            densitySum += cloudLayerDensity(worldPush, index, p, wind,
                    cloudLod(length(p - pos), l.detailTile), false);
        }
        tau += (densitySum / float(steps)) * l.density * l.tallness / denom;
    }
    return lerp(1.0, exp(-tau), shadowGate);
}

/** Field sample at one world position's projection, for the debug views. */
public float cloudCoverageWorld(float2 rebasedXZ) {
    if ((worldPush.flags & WORLD_FLAG_CLOUD) == 0u || worldPush.cloud0.x <= 0.0) {
        return 0.0;
    }
    CloudLayer low = cloudLayerOf(worldPush, CLOUD_LOW);
    float2 wind = cloudLayerWind(worldPush, CLOUD_LOW);
    float3 p = float3(rebasedXZ.x, low.base + 0.5 * low.tallness, rebasedXZ.y);
    return cloudLayerDensity(worldPush, CLOUD_LOW, p, wind, 0.0, true);
}

#ifndef CAUSTICA_NO_VISIBLE_CLOUD_PATH

// The planar tier as sky.rmiss renders it: every deck crossed once, front to back, composited into
// one transmittance. A camera inside a deck sees its mid-height cut; far crossings fade into the haze
// they sit behind.
public float3 planarCloudSkyInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                      out float cloudT) {
    return cloudPlanarStack(push, ro, rd, state, 1.0e12, cloudT);
}

// The stack's out-scattering on a segment that ENDS ON GEOMETRY. sky.rmiss composites the layer only
// for rays that escape, so a ray that crosses the deck and then lands on terrain — the view from above
// the layer, a reflection or a refraction that finds a hillside — otherwise draws that surface as if
// no cloud lay in front of it. The crossings are clipped to [0, tMax]: a surface nearer than a deck
// cannot be veiled by it. Gated on its own flag so a frame with all volumetrics off never reaches the
// density fetch.
public float3 cloudSegmentOutScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                     float tMax, out float cloudT) {
    if ((push.flags & WORLD_FLAG_CLOUD_OUTSCATTER) == 0u) {
        cloudT = 1.0;
        return float3(0.0, 0.0, 0.0);
    }
    return cloudPlanarStack(push, ro, rd, state, tMax, cloudT);
}

// ---- Volumetric march (the raymarcher tier) ---------------------------------------------------------
//
// The walk spans every deck the ray meets, front to back. The budget is a SAMPLE count shared across
// the decks in proportion to the slab span each one catches, capped by a target step of two erosion
// cells: a grazing sight line through a slab needs an order more samples than a vertical one, and a
// fixed per-deck split starves exactly the ray that needs the help. The planar tier prices the same
// stack in one crossing per deck, with the same shape field, the same layer densities and the same
// light, so the two tiers degrade into each other instead of into a different sky.
public float3 volumetricCloudInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                       float2 pix, out float cloudT) {
    cloudT = 1.0;
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0) {
        // A zero-coverage layer has nothing to cross; cloudT stays 1 and the caller's compose is a
        // no-op. A ray parallel to every deck is handled per deck by cloudLayerSpan.
        return float3(0.0, 0.0, 0.0);
    }
    if (push.volumeParams.w < 1.0) {
        return cloudPlanarStack(push, ro, rd, state, 1.0e12, cloudT);
    }
    CloudLight light = cloudLightOf(state, rd);
    float budget = clamp(push.volumeParams.w, 1.0, 96.0);
    // R2 rank-1 lattice: a fixed per-pixel phase whose neighbouring values are maximally separated,
    // rotated one golden-ratio stride per frame so RR averages it instead of seeing a re-roll.
    float jitter = frac(dot(pix, float2(0.7548776662, 0.5698402910))
            + float(push.frameIndex) * 0.6180339887);
    bool upward = rd.y >= 0.0;
    // First pass: the slab span each deck catches, so the budget lands where the cloud actually is.
    float spanSum = 0.0;
    float spans[CLOUD_LAYER_COUNT];
    for (int n = 0; n < CLOUD_LAYER_COUNT; n++) {
        int index = upward ? n : CLOUD_LAYER_COUNT - 1 - n;
        CloudLayer l = cloudLayerOf(push, index);
        spans[n] = 0.0;
        if (l.coverage <= 1.0e-3) {
            continue;
        }
        // The fade distance is also the walk's reach. A ray parallel to a deck and already inside it
        // would otherwise span an unbounded [t0, t1] and take steps kilometres long.
        float fadeDist = max(push.cloud1.w, 64.0) * (l.kind == 0 ? 1.0 : CLOUD_SHEET_FADE_MULT);
        float t0;
        float t1;
        if (!cloudLayerSpan(l, ro, rd, fadeDist, t0, t1)) {
            continue;
        }
        if (1.0 - smoothstep(fadeDist * 0.6, fadeDist, t0) <= 0.0) {
            continue;
        }
        spans[n] = t1 - t0;
        spanSum += spans[n];
    }
    float3 L = float3(0.0, 0.0, 0.0);
    float T = 1.0;
    for (int n = 0; n < CLOUD_LAYER_COUNT; n++) {
        if (spans[n] <= 0.0) {
            continue;
        }
        int index = upward ? n : CLOUD_LAYER_COUNT - 1 - n;
        CloudLayer l = cloudLayerOf(push, index);
        float fadeDist = max(push.cloud1.w, 64.0) * (l.kind == 0 ? 1.0 : CLOUD_SHEET_FADE_MULT);
        float t0;
        float t1;
        cloudLayerSpan(l, ro, rd, fadeDist, t0, t1);
        float fade = 1.0 - smoothstep(fadeDist * 0.6, fadeDist, t0);
        // Step split: the deck's share of the budget, capped by its own shape scale. A vertical sight
        // line through a 130-block deck needs six steps, not ninety-six; a grazing one takes whatever
        // share the span proportion grants it, minimum four so a sliver crossing still resolves.
        float steps;
        if (l.kind == 0) {
            float ideal = ceil(spans[n] / CLOUD_MARCH_STEP);
            float share = budget * spans[n] / max(spanSum, 1.0e-3);
            steps = clamp(min(ideal, share), 4.0, budget);
        } else {
            // A sheet is five blocks thick and a single crossing is already its whole column.
            steps = 2.0;
        }
        cloudMarchLayer(push, index, ro, rd, t0, t1, steps, jitter, fade, light, L, T);
        if (T < 3.0e-3) {
            break;
        }
    }
    cloudT = T;
    return L;
}

// Extinction only, at the planar tier's price: a continuation ray (a reflection, a refraction, an
// indirect bounce) must be dimmed by the deck it passed through without paying a walk per bounce —
// the "Pass B does extinction only" contract. One fetch per deck at its mid-crossing and the same
// slant-thickened tau the planar tier uses keeps a bounce's attenuation the MEAN of what the primary
// ray walked.
public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {
    if ((push.flags & WORLD_FLAG_CLOUD) == 0u || push.cloud0.x <= 0.0 || abs(rd.y) < 1.0e-6) {
        return 1.0;
    }
    float tau = 0.0;
    for (int index = 0; index < CLOUD_LAYER_COUNT; index++) {
        CloudLayer l = cloudLayerOf(push, index);
        if (l.coverage <= 1.0e-3) {
            continue;
        }
        if (rd.y > 0.0 ? ro.y >= l.top : ro.y <= l.base) {
            continue; // the ray runs away from the deck
        }
        float mid = l.base + 0.5 * l.tallness;
        float t = (mid - ro.y) / rd.y;
        if (t <= 0.0) {
            continue;
        }
        float3 p = ro + rd * t;
        float2 wind = cloudLayerWind(push, index);
        float density = cloudLayerDensity(push, index, p, wind, cloudLod(t, l.detailTile), false);
        tau += density * l.density * l.tallness / max(abs(rd.y), 0.04);
    }
    return exp(-tau);
}

#else // CAUSTICA_NO_VISIBLE_CLOUD_PATH: the shadow field remains, the visible layer is inert.

public float3 planarCloudSkyInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                      out float cloudT) {
    float unused = push.cloud0.x + ro.x + rd.y + state.skyLightFactor;
    cloudT = 1.0;
    return float3(0.0, 0.0, 0.0) * unused;
}

public float3 cloudSegmentOutScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                     float tMax, out float cloudT) {
    float unused = push.cloud0.x + ro.x + rd.y + state.skyLightFactor + tMax;
    cloudT = 1.0;
    return float3(0.0, 0.0, 0.0) * unused;
}

public float3 volumetricCloudInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                       float2 pix, out float cloudT) {
    float unused = push.cloud0.x + ro.x + rd.y + state.skyLightFactor + pix.x;
    cloudT = 1.0;
    return float3(0.0, 0.0, 0.0) * unused;
}

public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {
    return 1.0 + 0.0 * (push.cloud0.x + ro.x + rd.y);
}

#endif // CAUSTICA_NO_VISIBLE_CLOUD_PATH

#else // CAUSTICA_NO_CLOUD_PATH: every cloud consumer is inert.

public float cloudBoxMid(WorldPush push) {
    return 0.0 * push.cloud0.y;
}

public float cloudShadowTransmittance(float3 pos, float3 sunDir) {
    return 1.0 + 0.0 * (pos.x + sunDir.y);
}

public float cloudCoverageWorld(float2 rebasedXZ) {
    return 0.0 * rebasedXZ.x;
}

public float3 planarCloudSkyInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                      out float cloudT) {
    float unused = push.cloud0.x + ro.x + rd.y + state.skyLightFactor;
    cloudT = 1.0;
    return float3(0.0, 0.0, 0.0) * unused;
}

public float3 cloudSegmentOutScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                     float tMax, out float cloudT) {
    float unused = push.cloud0.x + ro.x + rd.y + state.skyLightFactor + tMax;
    cloudT = 1.0;
    return float3(0.0, 0.0, 0.0) * unused;
}

public float3 volumetricCloudInScatter(WorldPush push, float3 ro, float3 rd, SkyState state,
                                       float2 pix, out float cloudT) {
    float unused = push.cloud0.x + ro.x + rd.y + state.skyLightFactor + pix.x;
    cloudT = 1.0;
    return float3(0.0, 0.0, 0.0) * unused;
}

public float volumetricCloudTransmittanceFast(WorldPush push, float3 ro, float3 rd) {
    return 1.0 + 0.0 * (push.cloud0.x + ro.x + rd.y);
}

#endif // CAUSTICA_NO_CLOUD_PATH
// ---- Unified air-volume sampler --------------------------------------------------------------------
//
// One point sample of the air: the medium's contract, which the tiers integrate in closed form rather
// than by calling this. `density` comes from the marched tier's FBM field or from the plain profile, and
// the Rayleigh coefficient turns it into a per-channel extinction; the sampling rules are the tiers'
// own — the border's haze joins after the modulation, the render-edge fade is not a medium, and water
// excludes the air entirely because the stack's medium already owns that volume.
public VolumeSample sampleVolume(float3 pos, bool inWater, WorldPush push) {
    VolumeSample s = emptyVolumeSample();
    if (inWater || (push.flags & WORLD_FLAG_FOG) == 0u) {
        return s;
    }
    float density;
    if ((push.flags & WORLD_FLAG_VOLUME_MARCH) != 0u) {
        density = fogMarchDensity(push, pos);
    } else {
        density = fogDensityAt(push, pos.y);
    }
    // The border's haze joins the medium AFTER the tier's own modulation, so the billow field never
    // scales it and both tiers describe the same air. It is added to every channel alike: the Rayleigh
    // coefficient multiplies only the profile, so the veil stays grey. The render edge's fade is
    // deliberately absent: it is not a medium at all, and a point query has no view ray to measure a
    // coverage against.
    float3 sigmaT = density * FOG_RAYLEIGH_COEFF + boundarySigma(push);
    float3 albedo = max(push.fogColor.rgb, float3(0.0, 0.0, 0.0));
    s.sigmaS = albedo * sigmaT;
    s.sigmaA = (float3(1.0, 1.0, 1.0) - albedo) * sigmaT;
    s.g = push.fog0.w;
    return s;
}

// The height profile's density at one altitude — the number FOG_RAYLEIGH_COEFF turns into an
// extinction. Zero while the pushed density is off, which is what a fog-off frame reads.
public float fogDensityAt(WorldPush push, float y) {
    float density0 = push.fog0.x;
    if (density0 <= 0.0) {
        return 0.0;
    }
    return density0 * exp(-max(y - push.fog0.z, 0.0) / max(push.fog0.y, 1.0e-4));
}

// ---- Marched tier: heterogeneous fog with per-sample occlusion --------------------------------------
//
// The quality tier Phase 2 adds: uniform stepping over the segment with a pixel-fixed blue-noise
// jitter and FBM-modulated density. Every sample's measured visibility gates that sample's sun term
// and nothing else: the density stays the pushed profile, so the transmittance the caller attenuates
// with is a property of the air rather than of where the sun happens to reach. Only light is culled.
// The visibility itself is measured at the stratum budget below rather than traced once per sample.
//
// Convergence: with noise amplitude 0 and the shadow switch off every sample sees the plain analytic
// profile, and the midpoint-quadrature integral converges to the closed form as the step count rises —
// VolumeFogTest pins that numerically, 8/16/32/64 steps against fogOpticalDepth.
//
// Cost: the march covers only the segment's stretch inside the medium's significant band (see the span
// clamp below), spending steps density evals there, and the shadow probes ride a stratum budget — at
// most FOG_MARCH_PROBE_STRATA geometry shadow rays and cloud-shadow field evaluations per segment,
// wherever in the band the non-empty samples land. The early-outs matter: a segment entirely above the
// band returns without marching, samples past negligible transmittance stop the march, and the probes
// fire only where the medium is non-zero. volumeParams.z carries the cloud shadow field's sample
// count, not a fog integration distance — truncating the fog integral would desynchronise the out
// fogT from the throughput attenuation, which must agree.
public static const float FOG_SPAN_TAU = 2.0e-3;      // bound on the optical depth the span clamp excludes
public static const int FOG_MARCH_PROBE_STRATA = 8;   // shadow-probe budget per marched segment

// The radiance a fog sample scatters toward the eye from the sky: the SKY ALONG THE VIEW RAY itself,
// not the hemisphere's average irradiance. This is the term that decides whether a thick fog reads as
// air or as a dark wall, because the in-scatter converges to `albedo * fogSkyRadiance(rd)` as the
// transmittance falls. Sampling the same radiance the miss shader would have returned is what makes a
// fully opaque fog dissolve into the sky it hides; the hemisphere average cannot do that — a sunset's
// energy sits in a few degrees above the horizon while the average is dominated by the cold upper dome,
// so a fog bank built on it reads several stops darker than the sky behind it at exactly the elevation
// where the eye compares the two. Below-horizon rays mirror to their horizon elevation for the reason
// world.rmiss mirrors an escaping ray: the view LUT's lower half is the space floor, and air seen
// downward is haze, not space.
public float3 fogSkyRadiance(float3 rd) {
    SkyState state = skyState(worldPush);
    return skyDiffuseRadiance(skyViewLut, float3(rd.x, abs(rd.y), rd.z), state);
}

public float3 fogSegmentInScatterMarched(float3 ro, float3 rd, float t, float2 pix,
                                         bool shadowAllowed, out float3 fogT) {
    fogT = float3(1.0, 1.0, 1.0);
    float density0 = worldPush.fog0.x;
    float sigmaB = boundarySigma(worldPush);
    float3 albedo = max(worldPush.fogColor.rgb, float3(0.0, 0.0, 0.0));
    if (density0 + sigmaB <= 0.0 || dot(albedo, float3(1.0, 1.0, 1.0)) == 0.0
            || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    // The significant band. The density is a function of height alone and is bounded by the analytic
    // profile times the noise's largest multiplier, so above
    //     yTop = baseY + H * ln(peak / eps),
    //     peak = density0 * (1 + |amp|) * FOG_RAYLEIGH_PEAK + sigmaB,  eps = FOG_SPAN_TAU / t,
    // every block of path carries less than eps of extinction in the channel that dies first (blue),
    // and every other channel strictly less. The excluded stretch's optical depth — and with it the
    // error in both the returned transmittance and the skipped in-scatter — is therefore bounded by
    // FOG_SPAN_TAU whatever the geometry and whichever channel the eye weighs, orders below anything
    // the estimator can resolve. The march covers only the segment's stretch at or below yTop: a
    // segment entirely above returns without marching, and one crossing the height concentrates its
    // whole step budget on the density that can actually read instead of striding the empty tail above
    // the fog.
    float peak = density0 * (1.0 + abs(worldPush.fog1.z)) * FOG_RAYLEIGH_PEAK + sigmaB;
    float eps = FOG_SPAN_TAU / t;
    if (peak <= eps) {
        return float3(0.0, 0.0, 0.0);
    }
    // A flat medium has no significant band: the border's sigma is the same density at every altitude, so
    // the clamp that trims the empty stretch above the fog would trim real density with it. Put the top out
    // of reach while it is live — the march then covers the whole segment — rather than re-deriving the span
    // around a profile that does not fall off.
    float yTop = sigmaB > 0.0 ? 1.0e30
            : worldPush.fog0.z + max(worldPush.fog0.y, 1.0e-4) * log(peak / eps);
    float yEnd = ro.y + rd.y * t;
    if (min(ro.y, yEnd) > yTop) {
        return float3(0.0, 0.0, 0.0); // the whole segment sits above the medium
    }
    float t0 = 0.0;
    float t1 = t;
    if (max(ro.y, yEnd) > yTop && abs(rd.y) > 1.0e-6) {
        float tTop = clamp((yTop - ro.y) / rd.y, 0.0, t);
        if (rd.y > 0.0) {
            t1 = tTop;
        } else {
            t0 = tTop;
        }
    }
    if (t1 <= t0) {
        return float3(0.0, 0.0, 0.0);
    }

    int steps = int(clamp(worldPush.volumeParams.x, 1.0, 64.0));
    float jitterStrength = clamp(worldPush.volumeParams.y, 0.0, 1.0);
    bool shadow = shadowAllowed
            && (worldPush.flags & WORLD_FLAG_VOLUME_SHADOW) != 0u
            && celestialLight.halfAngle > 0.0;

    float dt = (t1 - t0) / float(steps);
    // Pixel-fixed interleave gradient with the golden-ratio frame walk — the analytic tier's exact
    // doctrine: a per-frame re-roll is a signal the denoiser cannot accumulate, while this pattern
    // dithers step edges into a stable grain the reconstruct averages out.
    float u = 0.5 + (frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887) - 0.5) * jitterStrength;

    float phase = hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7;
    float3 ambient = worldPush.fogColor.w * fogSkyRadiance(rd);
    float3 sunDir = celestialLight.dir;

    float3 L = float3(0.0, 0.0, 0.0);
    float3 tau = float3(0.0, 0.0, 0.0);
    // The probes ride a stratum budget: the marched samples split into FOG_MARCH_PROBE_STRATA equal
    // ranges, and each stratum's first non-empty sample fires the geometry shadow ray once, with the
    // whole stratum reusing it. A stratum spans steps/FOG_MARCH_PROBE_STRATA samples, so shadow-volume
    // edges quantise at that scale per frame — the pixel-fixed jitter walks the sample positions inside
    // the strata, and the reconstruct resolves the quantisation the same way it resolves the step
    // dither. At or below the stratum count every sample probes, which is the short-segment case where
    // per-sample resolution actually reads.
    int lastStratum = -1;
    bool probeValid = false;
    float stratumVis = 1.0;
    // One cloud-shadow evaluation for the whole marched span, at its middle: the deck above a segment
    // does not change faster than the field itself, and the per-stratum signal that must vary is the
    // geometry visibility below. A field walk per stratum bought cloud-shadow grain the reconstruct
    // cannot see under the shaft term it multiplies.
    float cloudShadow = shadow
            ? cloudShadowTransmittance(ro + rd * (0.5 * (t0 + t1)), sunDir)
            : 1.0;
    for (int i = 0; i < steps; i++) {
        int stratum = (i * FOG_MARCH_PROBE_STRATA) / steps;
        if (stratum != lastStratum) {
            lastStratum = stratum;
            probeValid = false;
        }
        float tMid = t0 + (float(i) + u) * dt;
        float3 p = ro + rd * tMid;
        float density = fogMarchDensity(worldPush, p);
        float3 sigmaT = density * FOG_RAYLEIGH_COEFF + sigmaB;
        float vis = 1.0;
        if (density + sigmaB > 0.0) {
            if (!probeValid) {
                stratumVis = shadow ? luminance(visibility(p, sunDir, 10000.0).transmittance) : 1.0;
                probeValid = true;
            }
            vis = stratumVis;
        }
        // The medium is the density profile; the visibility gates the sun term below. Thinning the
        // density instead would make the fog's opacity a function of a probe estimate — a
        // per-frame-walking, stochastic signal a temporal reconstruct cannot accumulate, and one that
        // deletes the medium wherever the sun is occluded (the air over the sea reading as a dark band
        // while the sky-facing segment, which traces no probes, kept its full density).
        if (density + sigmaB <= 0.0) {
            continue;
        }
        float3 tauMid = tau + sigmaT * 0.5 * dt;
        float3 tMid_transmittance = exp(-tauMid);
        // Blue dies first. Once the surviving luminance is below the eye's floor the channels have
        // already diverged to a fixed ratio, so their recorded value is negligible too and the rest of
        // the segment can only add that ratio times nothing.
        if (luminance(tMid_transmittance) < 0.004) {
            break;
        }
        float3 sunTerm = float3(0.0, 0.0, 0.0);
        // The floor rides the sun term, not the medium: a fully shaded stratum keeps this fraction of
        // its direct light, which softens the fog's shadow edges instead of making the air vanish in them.
        float sunVis = max(vis, fogShadowFloor());
        if (sunVis > 0.0) {
            // Toward-sun transmittance: the analytic profile — the noise modulation of the sun leg is
            // deliberately not marched (a second full march per sample for a second-order softening of
            // the shafts is not a trade this tier makes). The cloud shadow rides on top: the same field
            // the sheet renders and the ground NEE reads, so a clouded sun thins the shafts instead of
            // lighting them. Per channel, so a low sun reddens through the air it crosses.
            float3 sunFogT = fogTransmittanceRGB(density0, worldPush.fog0.z, worldPush.fog0.y,
                    p.y, sunDir.y, 10000.0);
            sunTerm = phase * celestialLight.illuminance
                    * (sunVis * sunFogT * cloudShadow);
        }
        L += tMid_transmittance * (albedo * sigmaT) * (sunTerm + ambient) * dt;
        tau += sigmaT * dt;
    }
    fogT = exp(-tau);
    return L;
}

// ---- The water column: its own in-scatter ----------------------------------------------------------
//
// A path inside water crosses two media that are not the air fog: the body's own in-scatter (here) and
// the structure-gated shaft layer further down. They are separate terms on purpose. This one is the
// body's glow — the reason a submerged view has depth instead of reading as a dark window — and it is
// lit by the sky and the sun exactly once per segment. The shaft layer is the sun's beams, gated on the
// shadow structure, and it attenuates nothing.
//
// The body's constant medium makes the single-scatter integral exact rather than marched:
//
//     L = sigmaS * (ambient integral) + (sigmaS/sigmaT) * (1 - T) * sun
//
// with T = exp(-sigmaT * t) in the GREY mean of the per-channel extinction. The throughput itself keeps
// attenuating per channel through the caller's Beer-Lambert line; this term refuses to become a
// per-channel vector weight for a contribution that is second-order here. sigmaS/sigmaT is the pushed
// fog density by construction (medium.waterScattering), so no coefficient is derived twice.
//
// The ambient field DECAYS WITH DEPTH. Sky irradiance reaches a sample only after crossing the water
// above it, so ambient(p) = A * exp(-sigmaT * depth(p)). Without that decay every upward view
// degenerates to a depth-independent milk veil at the fog's own brightness. The surface is the pushed
// sea-level plane (water1.w) — exact for oceans; for elevated water the depth clamps at zero and the
// term degrades to the un-attenuated form rather than inventing light. The product T(s) *
// exp(-sigmaT * depth(s)) is still an exponential, so the integral stays closed form:
//
//     L_amb = sigmaS * A * exp(-sigmaT * d0) * (1 - exp(-sigmaT * t * (1 - rd.y))) / (sigmaT * (1 - rd.y))
//
// with d0 the segment start's depth below the plane; the rd.y -> 1 (straight-up) limit is t, where the
// along-path and depth attenuations exactly cancel and the integral is bounded by the surface hit.
//
// Sun leg: one shadow-ray probe at the segment midpoint, the same one-probe-per-segment budget the
// analytic fog tier runs. visibility() already crosses the waved surface as an any-hit (tinting the
// transmittance and reporting the exit distance as waterHitT); on top of that the leg picks up the
// body's Beer-Lambert over the underwater stretch of the sun ray and the wave-refraction caustic at the
// exit point — the same term the underwater terrain NEE applies, so the glow and the caustic light
// pools come from one wave field. With the shadow switch off there is no probe and hence no exit
// distance, so the sun term is skipped rather than estimated from a guessed depth; the ambient term
// alone keeps the water lit.
public static const float WATER_AMBIENT_SURFACE = 0.92; // surface transmission for downward sky light

public float3 waterSegmentInScatter(float3 ro, float3 rd, float t, float2 pix, Medium medium) {
    float sigmaT = (medium.extinction.r + medium.extinction.g + medium.extinction.b) / 3.0;
    float sigmaS = (medium.scattering.r + medium.scattering.g + medium.scattering.b) / 3.0;
    if (sigmaT <= 0.0 || sigmaS <= 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 ambient = worldPush.water1.z * skyIrradiance(skyIrradianceLut, float3(0.0, 1.0, 0.0))
            * INV_PI * WATER_AMBIENT_SURFACE;
    float scatterWeight = 1.0 - exp(-sigmaT * t);
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
            float legT = 1.0;
            if (probe.waterHitT > 0.0) {
                legT = exp(-sigmaT * probe.waterHitT);
                if ((worldPush.flags & 16u) != 0u) { // the water-waves toggle's flag bit
                    vis *= waterCaustic(p + celestialLight.dir * probe.waterHitT,
                            celestialLight.dir, probe.waterHitT);
                }
            }
            // The fog's own 70%-energy HG scaling, plus the cloud shadow the sun leg crosses above the
            // surface — the same field every other sun term reads, so overcast water loses its beams
            // with everything else.
            float phase = hg(dot(rd, celestialLight.dir), medium.phaseG) * SHAFT_PHASE_ENERGY;
            sunPart = phase * celestialLight.illuminance
                    * (vis * legT * cloudShadowTransmittance(p, celestialLight.dir));
        }
    }
    return sigmaS * ambIntegral * ambient
            + (sigmaS / sigmaT) * scatterWeight * sunPart;
}

// ---- Segment-level lighting assembly ---------------------------------------------------------------
//
// The analytic fog tier's single-scatter over one path segment. The MEDIUM is the height profile alone:
// density and transmittance must not depend on where the sun happens to reach. Tying them to visibility
// makes the fog's opacity a function of a four-probe estimate — a stochastic, per-frame-walking signal
// no temporal reconstruct can accumulate, and one that deletes the medium wherever the probes land in
// shadow. That is what turned the air over the sea into a dark band while the sky-facing segment, which
// traces no probes at all, kept its full density: the two halves of the horizon were integrating
// different media.
//
// Visibility therefore gates LIGHT, not matter. FOG_SUN_BANDS probes placed at equal in-scatter-weight
// depths (the closed form's inverse; pixel-fixed IGN jitter walked by a golden-ratio sequence, so the
// reconstruct resolves their quantisation) each contribute their own measured visibility to their own
// stratum's direct term, and a stratum the sun does not reach contributes none of it. The sky ambient is
// not gated at all — shadowed air still scatters sky light, which is what makes a backlit valley read as
// blue haze instead of as clear air.
//
// Writes the segment's fog transmittance so the caller attenuates throughput with the same quantity
// the in-scatter used. `celestialLight` (world_core) must have been derived for this frame and
// converted to ACEScg, exactly as world_indirect's raygen does; callers in a pass that skips NEE set
// it only when they actually call this.
public static const int FOG_SUN_BANDS = 4;

// Direct-sun fraction a fully shaded stratum keeps — the live slider volume.fog-shadow-floor riding
// fog1.w. A lighting control, never a density one: the medium is the same everywhere, so the slider
// decides how hard the fog's shadow edges read (0 = direct sun fully culled inside shadow, 0.95 = the
// sun reaches almost everywhere) without changing how much air the ray crosses.
public float fogShadowFloor() {
    return clamp(worldPush.fog1.w, 0.0, 0.95);
}

// Pixel-fixed hash for the per-band probe jitter. Deliberately NOT frame-indexed: a per-frame
// re-roll is a signal the denoiser cannot accumulate, while a static per-pixel pattern dithers the
// shadow-volume edges into a stable grain the reconstruct resolves into soft edges.
public float interleaveGradientNoise(float2 pix) {
    return frac(52.9829189 * frac(dot(pix, float2(0.06711056, 0.00583715))));
}

// Depth at which the profile's optical depth alone reaches `tau`, along a ray that starts `above`
// blocks over the base and travels along dirY. The analytic inverse of fogOpticalDepth: each stretch
// is a constant or a single exponential, so each solves for its length directly. A target past the
// finite column above a rising sample has no root, and the caller clamps the (large) result to the
// segment.
public float fogExpDepthAtTau(float density0, float baseY, float heightH, float y0, float dirY,
                              float tau) {
    if (density0 <= 0.0 || tau <= 0.0) {
        return 0.0;
    }
    float h = max(heightH, 1.0e-4);
    float above = y0 - baseY;
    if (abs(dirY) < 1.0e-6) {
        // A level ray sees one constant density: the column is linear in the depth.
        float rho = density0 * exp(-max(above, 0.0) / h);
        return rho > 0.0 ? tau / rho : 0.0;
    }
    if (dirY > 0.0) {
        if (above >= 0.0) {
            float column = density0 * exp(-above / h) * h / dirY;
            return -(h / dirY) * log(max(1.0 - tau / max(column, 1.0e-30), 1.0e-20));
        }
        // Rising through the flat stretch below the base first, then into the exponential.
        float sBase = -above / dirY;
        float flat = density0 * sBase;
        if (tau <= flat) {
            return tau / density0;
        }
        float column = density0 * h / dirY;
        return sBase - (h / dirY) * log(max(1.0 - (tau - flat) / max(column, 1.0e-30), 1.0e-20));
    }
    if (above <= 0.0) {
        return tau / density0;
    }
    // Falling from above the base: the exponential stretch down to baseY, then the flat column.
    float sBase = above / (-dirY);
    float toBase = density0 * h * (1.0 - exp(-above / h)) / (-dirY);
    if (tau <= toBase) {
        float column = density0 * exp(-above / h) * h / dirY;
        return -(h / dirY) * log(max(1.0 - tau / min(column, -1.0e-30), 1.0e-20));
    }
    return sBase + (tau - toBase) / density0;
}

// Depth at which the segment's cumulative in-scatter weight reaches `weight` (equivalently, where
// transmittance drops to 1 - weight). This is what places the sun-visibility probes: equal weight
// intervals contribute equal Monte-Carlo weight regardless of where the density puts them. `density0`
// is the pushed profile, not an extinction — callers pass it scaled by FOG_RAYLEIGH_MEAN so a single
// scalar tracks the perceptual middle of the per-channel medium. `sigmaFlat` is the border haze's share
// of the same medium, so the depth is solved on the medium the segment actually integrates rather than
// on the height profile alone.
//
// The exponential inverts in closed form (fogExpDepthAtTau), so the common no-border case is a direct
// solve. The flat haze is the one term with no elementary inverse; with it live, its own bound
// (tau/sigmaFlat) caps the solve and three Newton steps on the full optical depth land the root — the
// optical depth is monotone with a closed-form derivative, so the iteration is a rounding step rather
// than a search.
public float fogDepthAtWeight(float density0, float sigmaFlat, float baseY, float heightH,
                              float3 ro, float3 rd, float t, float weight) {
    float tauTarget = -log(max(1.0 - weight, 1.0e-4));
    if (weight <= 0.0) {
        return 0.0;
    }
    if (fogOpticalDepth(density0, baseY, heightH, ro.y, rd.y, t) + sigmaFlat * t <= tauTarget) {
        return t;
    }
    float depth = fogExpDepthAtTau(density0, baseY, heightH, ro.y, rd.y, tauTarget);
    if (sigmaFlat <= 0.0) {
        return clamp(depth, 0.0, t);
    }
    depth = clamp(min(depth, tauTarget / sigmaFlat), 0.0, t);
    for (int i = 0; i < 3; i++) {
        float tau = fogOpticalDepth(density0, baseY, heightH, ro.y, rd.y, depth) + sigmaFlat * depth;
        float slope = density0 * exp(-max(ro.y + rd.y * depth - baseY, 0.0) / heightH) + sigmaFlat;
        depth = clamp(depth - (tau - tauTarget) / max(slope, 1.0e-20), 0.0, t);
    }
    return depth;
}

// Analytic fog transmittance from the segment start toward the sun, used when the shadow switch is
// off and no probe exists to evaluate it from. Takes the segment's density so the toward-sun
// transmittance agrees with the medium the segment actually integrates. Per channel, so a low sun
// reddens through the air it has to cross — the counterpart of the blue the view ray picks up.
public float3 fogT_toSunRo(float3 ro, float3 sunDir, float density0) {
    return fogTransmittanceRGB(density0, worldPush.fog0.z, worldPush.fog0.y,
            ro.y, sunDir.y, 10000.0);
}

public float3 fogSegmentInScatter(float3 ro, float3 rd, float t, float2 pix,
                                  bool allowShadowCull, out float3 fogT) {
    // Tier dispatch: the march flag routes the whole segment to the heterogeneous integrator, which
    // reuses the same shadow switch, phase, ambient and lighting doctrine. `allowShadowCull` keeps its
    // meaning — the sky-facing miss segment passes false and its march runs without shadow rays, the
    // same cost gate the analytic tier's probes honour.
    //
    // The march tier exists to resolve a heterogeneous FIELD: the density noise. With it off the
    // integrator would reproduce the profile the closed form already owns, so every step it spends buys
    // nothing but the jitter of its own quadrature — a per-frame signal the reconstruct cannot
    // accumulate, on a frame that has no field to resolve. Take the closed form then; the flag stays the
    // tier selector for frames that actually carry one.
    //
    // A record compiled with CAUSTICA_NO_MARCH_PATH pins the field off: nothing in it can reach the
    // integrator, so the compiler drops it from the SPIR-V and the raygen stops paying its registers.
    // The renderer dispatches those records only while the march tier is off (see RtComposite).
    bool heterogeneous = worldPush.fog1.z > 0.0;
#if defined(CAUSTICA_NO_MARCH_PATH)
    heterogeneous = false;
#endif
    if ((worldPush.flags & WORLD_FLAG_VOLUME_MARCH) != 0u && worldPush.volumeParams.x >= 1.0
            && heterogeneous) {
        return fogSegmentInScatterMarched(ro, rd, t, pix, allowShadowCull, fogT);
    }
    float density0 = worldPush.fog0.x;
    float sigmaB = boundarySigma(worldPush);
    fogT = float3(1.0, 1.0, 1.0);
    float3 albedo = max(worldPush.fogColor.rgb, float3(0.0, 0.0, 0.0));
    if (density0 + sigmaB <= 0.0 || dot(albedo, float3(1.0, 1.0, 1.0)) == 0.0
            || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }

    bool shadowCull = allowShadowCull
            && (worldPush.flags & WORLD_FLAG_VOLUME_SHADOW) != 0u
            && celestialLight.halfAngle > 0.0;

    // The medium: the height profile and the border's floor, neither of them modulated by light. `fogT`
    // is what the caller attenuates its throughput with, so it has to be the same quantity every frame
    // for a given ray — a visibility-scaled sigma made it a per-pixel lottery and let a shadowed stretch
    // read as clear air. It is per channel: that is the aerial perspective. `scatterWeight` is the share
    // of the medium each channel's light is scattered by, and `scatterTotal` its luminance — the one
    // scalar that can stand for "how much air is in front of this pixel" when deciding whether a probe
    // is worth tracing.
    float3 transmittance = airTransmittance(ro, rd, t);
    fogT = transmittance;
    float3 scatterWeight = 1.0 - transmittance;
    float scatterTotal = luminance(scatterWeight);

    // One shadow probe per equal-weight stratum, placed on the field with a pixel-fixed,
    // frame-walking jitter. Each sample's visibility is kept for its own stratum's sun term; nothing
    // here touches the medium, so the transmittance below is a property of the air alone.
    float vis[FOG_SUN_BANDS];
    float stratumDepth[FOG_SUN_BANDS];
    float noise = frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887);
    for (int k = 0; k < FOG_SUN_BANDS; k++) {
        vis[k] = 1.0;
        stratumDepth[k] = 0.0;
    }
    // The cloud field is evaluated ONCE for the segment, at the half-weight depth: a deck's shadow is
    // a smooth multiplier at the scale of hundreds of blocks, so the four bands would read the same
    // number to within noise while each paid a full three-deck walk for it. The geometry shadow the
    // probes measure is what varies band to band, and that is untouched.
    float cloudShadow = 1.0;
    if (shadowCull) {
        if (scatterTotal > 1.0e-4) {
            float midDepth = fogDepthAtWeight(density0 * FOG_RAYLEIGH_MEAN, sigmaB,
                    worldPush.fog0.z, worldPush.fog0.y, ro, rd, t, 0.5);
            cloudShadow = cloudShadowTransmittance(ro + rd * midDepth, celestialLight.dir);
            for (int k = 0; k < FOG_SUN_BANDS; k++) {
                float stratumStart = scatterTotal * (float(k) + noise) / float(FOG_SUN_BANDS);
                stratumDepth[k] = fogDepthAtWeight(density0 * FOG_RAYLEIGH_MEAN, sigmaB,
                        worldPush.fog0.z, worldPush.fog0.y,
                        ro, rd, t, stratumStart + scatterTotal * 0.5 / float(FOG_SUN_BANDS));
                vis[k] = luminance(visibility(ro + rd * stratumDepth[k], celestialLight.dir,
                        10000.0).transmittance);
            }
        }
    }

    // HG lobe scaled to 70% energy: a pure forward lobe blows out when the view aligns with the
    // sun down a bright shaft. No isotropic floor — a floor at side/back angles reads as a uniform
    // milk wash, and the strata probes already light the lit medium geometrically.
    float phase = hg(dot(rd, celestialLight.dir), worldPush.fog0.w) * 0.7;
    float3 ambient = worldPush.fogColor.w * fogSkyRadiance(rd);
    float3 sunPart = float3(0.0, 0.0, 0.0);
    if (!shadowCull) {
        // Unshadowed sun term: the closed form with V = 1 everywhere, sun transmittance taken at
        // the segment start. The cloud shadow still applies — geometry shadows are off, the layer
        // above is not.
        sunPart = phase * celestialLight.illuminance * (scatterWeight
                * fogT_toSunRo(ro, celestialLight.dir, density0)
                * cloudShadowTransmittance(ro, celestialLight.dir));
    } else if (scatterTotal > 1.0e-4) {
        // Per-stratum sun in-scatter: the stratum's share of the medium times its own measured
        // visibility times the closed-form fog transmittance from its probe toward the sun. Strata
        // whose probe landed in shadow contribute none of it — that IS the culling, and it is light
        // only. The segment's single cloud-shadow sample multiplies the sum, so a clouded sun thins
        // the shafts exactly as it thins every other sun leg.
        float3 stratumWeight = scatterWeight / float(FOG_SUN_BANDS);
        for (int k = 0; k < FOG_SUN_BANDS; k++) {
            float3 probe = ro + rd * stratumDepth[k];
            float3 sunFogT = fogTransmittanceRGB(density0, worldPush.fog0.z, worldPush.fog0.y,
                    probe.y, celestialLight.dir.y, 10000.0);
            float sunVis = max(vis[k], fogShadowFloor());
            sunPart += phase * celestialLight.illuminance * (stratumWeight * sunVis * sunFogT);
        }
        sunPart *= cloudShadow;
    }
    return albedo * (sunPart + scatterWeight * ambient);
}

// ---- Sun shafts: one integrator for the air and for the water --------------------------------------
//
// Airlight: the sun's beams made visible by the medium itself, with no height fog involved. The two are
// separate media on purpose. The fog's transmittance is what makes a scene hazy, so god rays that
// "need fog" always arrive with a veil; a shaft medium adds light and attenuates nothing, which is the
// only way to have beams in clear air. The same integrator runs underwater, where the medium is the
// water body's own scattering instead of the pushed air density.
//
// The integral is the segment's single scatter over a uniform medium,
//
//     L = structure * strength * phase * E * ∫_0^min(t, reach) sigma*exp(-sigma*s) * V(s) * cloudShadow(s) ds
//
// split into SHAFT_STRATA strata of equal in-scatter weight, so the strata follow the medium's own
// e-folding instead of even spacing: air (sigma ~ 1e-3) puts them almost uniformly along the reach,
// while water (sigma ~ 0.1-0.5) packs them into the few blocks the light actually crosses. A point-
// sampled sun visibility V sits inside each stratum. There is no ambient term: sky light scattered
// toward the eye is not a beam, and adding it would only lift the shade the beams have to contrast
// with. The water body's own glow is a separate integral (waterSegmentInScatter), which is what keeps
// the two separable: a player can want beams without the milky water they sit in.
//
// `structure` is what makes this read as a Tyndall beam instead of as fog. A path that never leaves the
// light scatters uniformly along its whole length, and that smooth lift — added to every sunlit wall,
// every open field, the sky-facing half of every room — is indistinguishable from a mist veil. It is
// also the one part of this integral the sky LUT already renders as the atmosphere's own in-scatter.
// So the term is faded out by the lit fraction of the path: fully-lit air contributes nothing, air that
// straddles a shadow boundary contributes in full, and half-lit air keeps half. What survives is exactly the shadow structure the
// eye reads as a shaft, which is also why the beams are only drawn where the sun is occluded somewhere
// along the line of sight — the physical reason they are visible at all.
//
// The trade is deliberate: a camera that sits inside a beam and looks along it sees no lift from that
// beam, because along such a path every probe is lit and there is no edge to see. That is the same
// bargain every screen-space god-ray effect makes, and the alternative — a veil over the whole frame —
// is what this gate exists to remove.
//
// V is the same ray-traced sun visibility the terrain NEE uses, sampled once per stratum and jittered
// by the pixel-fixed interleave hash walked by the golden-ratio frame index — the doctrine the fog's
// strata follow, so the temporal reconstruct resolves the probe positions into continuous shadow
// volumes instead of a per-frame re-roll it cannot accumulate. The stratum count is higher than the
// fog's four because the lit fraction needs enough samples to place the shadow boundary: at four, the
// gate quantises in quarters and the beam edges step.
//
// Callers integrate only the stretch in front of geometry — never a sky miss. The sky LUT already
// carries the atmosphere's own in-scatter, so airlight along a ray that escapes to the sky would count
// that scattering twice and read as an aureole around the sun rather than as a beam.
public static const int SHAFT_STRATA = 8;

// The fog's own HG energy scaling, so a footer and a beam lit by the same sun agree in level.
public static const float SHAFT_PHASE_ENERGY = 0.7;

// Underwater the term is shaped differently from the air one, for two reasons that only exist in water.
// Its forward lobe has to be CAPPED (`waterShaft.y`, volume.water-shaft-fill): a long grazing path to
// the surface plus the peak of the lobe is an order of magnitude over the water it sits in front of, and
// the surface you look up at goes white. And its gate keeps a floor (`WATER_SHAFT_STRUCTURE_FLOOR`),
// because the gate alone peaks in a band a few pixels wide when the occluder is a bank or a tree, which
// reads as no shafts at all rather than as shafts.
public static const float WATER_SHAFT_STRUCTURE_FLOOR = 0.45;

// Depth at which the medium's cumulative in-scatter weight reaches k/SHAFT_STRATA of the stretch's
// total: the inverse of exp(-sigma*s). Equal weight per stratum is what keeps a dense medium sampling
// the blocks the light actually crosses, and it degenerates to even spacing in the thin limit, which is
// exactly where the air medium sits.
public float shaftDepthAtWeight(float sigma, float reach, int k) {
    if (k <= 0) {
        return 0.0;
    }
    if (k >= SHAFT_STRATA) {
        return reach;
    }
    float total = 1.0 - exp(-sigma * reach);
    float weight = float(k) * total / float(SHAFT_STRATA);
    return -log(max(1.0 - weight, 1.0e-6)) / sigma;
}

// `sigma` is the medium's extinction (pushed air density, or the water body's grey extinction), `g` its
// phase anisotropy, and `underwater` selects the water's surface leg: the sun ray has to cross the
// water surface, so it picks up the rest of the body's Beer-Lambert and the wave focus at the exit
// point, which is what lights an underwater shaft in the first place.
public float3 shaftInScatter(float3 ro, float3 rd, float t, float2 pix,
                             float sigma, float g, bool underwater, out float shaftShare) {
    shaftShare = 0.0;
    float strength = worldPush.shaft.w;
    if (sigma <= 0.0 || strength <= 0.0 || t <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    if (dot(celestialLight.illuminance, float3(1.0, 1.0, 1.0)) <= 0.0) {
        return float3(0.0, 0.0, 0.0); // no body above the horizon: nothing to scatter
    }
    float reach = min(t, max(worldPush.shaft.z, 1.0));
    // Air beams are still swallowed by fog — the height profile and the border's haze alike, since a beam
    // that crosses the veiled air has to lose its own share of it. The air's transmittance is per channel
    // while the beam's own extinction is a scalar, so the shaft takes its luminance: a beam is read by how
    // much of it survives, and the haze's hue belongs to the fog, not to the beam. Water's own extinction
    // is already inside sigma, and the air fog does not exist below the surface.
    float fogT = underwater ? 1.0 : luminance(airTransmittance(ro, rd, reach));
    float total = 1.0 - exp(-sigma * reach);
    float shaftWeight = total * fogT;
    if (shaftWeight <= 1.0e-4) {
        return float3(0.0, 0.0, 0.0); // a hit right in front of the eye: too little medium to read
    }
    float phase = hg(dot(rd, celestialLight.dir), g) * SHAFT_PHASE_ENERGY;
    if (underwater) {
        // The cap is the whole reason the underwater layer survives a look straight at the sun: with the
        // lobe at its peak and a long grazing path to the surface, the physical value runs an order of
        // magnitude over the water it sits in front of, and the surface turns white. A ceiling keeps the
        // water body's own level while the shadow structure still draws the beams.
        phase = min(phase, worldPush.waterShaft.y);
    }
    float noise = frac(interleaveGradientNoise(pix)
            + float(worldPush.frameIndex) * 0.6180339887);
    float weight = total / float(SHAFT_STRATA);
    bool waterWaves = (worldPush.flags & 16u) != 0u;
    // The cloud shadow is one multiplier for the whole stretch, sampled at its middle stratum: the
    // deck above a beam is the same deck at every probe, so a field walk per stratum only re-read the
    // same number eight times. The probes keep their own geometry visibility — that is the term the
    // beams are made of.
    float cloudShadow = cloudShadowTransmittance(
            ro + rd * shaftDepthAtWeight(sigma, reach, SHAFT_STRATA / 2), celestialLight.dir);
    float caustic = 1.0;
    bool causticKnown = false;
    float3 sun = float3(0.0, 0.0, 0.0);
    float lit = 0.0;
    for (int k = 0; k < SHAFT_STRATA; k++) {
        float a = shaftDepthAtWeight(sigma, reach, k);
        float b = shaftDepthAtWeight(sigma, reach, k + 1);
        float3 p = ro + rd * (a + (b - a) * noise);
        VisibilityResult probe = visibility(p, celestialLight.dir, 10000.0);
        float3 vis = probe.transmittance;
        if (underwater && probe.waterHitT > 0.0) {
            // The underwater stretch of the sun leg up to the surface, then the focus the wave field
            // gives the light as it crosses. One caustic sample serves the whole shaft: every probe sits
            // within a few blocks of the surface entry, where the pattern does not move enough to pay
            // the wave solve eight times over.
            vis *= exp(-sigma * probe.waterHitT);
            if (!causticKnown) {
                caustic = waterWaves
                        ? waterCaustic(p + celestialLight.dir * probe.waterHitT, celestialLight.dir,
                                probe.waterHitT)
                        : 1.0;
                causticKnown = true;
            }
            vis *= caustic;
        }
        // Per-channel, so a beam that has crossed stained glass carries its tint like every other sun
        // path does; solids and cutout leaves terminate it. The gate reads the grey mean: what matters
        // there is whether the path is lit, not in what colour.
        lit += dot(vis, float3(0.3333333, 0.3333333, 0.3333333));
        sun += vis * weight;
    }
    lit *= 1.0 / float(SHAFT_STRATA);
    // Linear in the lit fraction, deliberately: a smoothstep here multiplies the estimator's own
    // quantisation (one probe is 1/SHAFT_STRATA of the ramp) by its slope, and at a forward-scattering
    // phase the term is bright enough that those steps read as contour arcs across a lit wall. Linear
    // keeps every step at 1/SHAFT_STRATA of the term, which the reconstruct averages away.
    float structure = clamp(1.0 - lit, 0.0, 1.0);
    if (underwater) {
        // Water does not use the gate the way air does. The gate's product is largest exactly where a
        // path is half lit, which for a big occluder — a bank, a tree, a hull — is a band a few pixels
        // wide; open water and deep shade both fall to zero, and the view reads as "no shafts at all".
        // What reads underwater is the SHADOW the occluder casts into the water, so the term keeps its
        // visibility modulation and only leans on the gate for contrast: fully-lit water keeps 45%,
        // shadow-crossing water the lot. The forward peak is capped separately (see `phase`), which is
        // what keeps this from becoming a white sheet.
        structure = lerp(WATER_SHAFT_STRUCTURE_FLOOR, 1.0, structure);
    }
    shaftShare = shaftWeight * structure;
    return sun * (phase * celestialLight.illuminance * (strength * fogT * structure * cloudShadow));
}
