// Lens flare: the three artifacts a fast lens throws around bright sources, summed into one image that the
// display mapper adds to the exposed frame before the tone curve.
//
//   Starburst  Diffraction at the aperture blades is a Fourier transform of the blade polygon, which shows
//              up as a star of spikes whose count and angles follow the blade count. Rendered here as a ray
//              integral through the hot field: `samples` taps along each of `spikes` directions, an
//              exponential envelope so the spike tapers, and a per-texel offset so the taps of neighbouring
//              texels interleave instead of forming a dotted chain.
//   Halos      A defocused aperture images the iris as a ring around every source. Each channel gathers at a
//              slightly different radius, which is exactly how a real lens splits the ring into a chromatic
//              rim: the red ring is the widest, the blue the tightest.
//   Lens dirt  Dust on the front element is far outside the plane of focus, so it never resolves: it shows
//              up as the scene's light scattered through a blurry speck field. Procedural rather than a
//              texture, so there is no asset to ship and the pattern is identical on every machine.
//
// Every stage is a fixed number of texture fetches per texel. The flare image this pass writes is the display
// size divided by the resolution divisor (an eighth by default, i.e. 1/64 of the pixels) and the source field
// it reads is a fixed eighth, which is what keeps three lens artifacts affordable next to the bloom pyramid.
// Both sizes are decided on the host, so nothing here may assume a texel size.

import display_common;
import flare_bindings;

[[vk::push_constant]] LensFlarePush pc;

static const float3 ACESCG_LUMA = float3(0.2722287168, 0.6740817658, 0.0536895174);
static const float TAU = 6.28318530718;

static const float SPIKE_GAIN = 0.012;
// Envelope along a spike: weight exp(-falloff * f) with f the fraction of the way out. Without it the
// normalised sum barely decays and every ray reads as a flat wedge instead of tapering to a point.
static const float SPIKE_FALLOFF = 3.0;
static const float HALO_GAIN = 0.04;
// Vertical ring radius as a fraction of the horizontal one: an aperture seen off-axis projects as an
// ellipse, so the halo is not a circle.
static const float HALO_ELLIPSE = 0.90;

// Fetches per ring tap. The ring is a circle in texel space, so its circumference grows with the radius;
// fixing the fetch count per tap (rather than per ring) keeps the ring continuous at every radius.
static const int HALO_SUBTAPS = 2;
static const float HALO_CHROMA_R = 1.0;
static const float HALO_CHROMA_G = 0.0;
static const float HALO_CHROMA_B = -1.0;

static const float DIRT_GAIN = 0.42;
static const float DIRT_SMUDGE_FREQ = 2.6;
static const float DIRT_EMPTY = 0.42;
// Soft ceiling on the glow that lights the dust: the glow is the frame's brightest signal, and an exposed sun
// would otherwise turn every grain into an opaque blot instead of a lit speck.
static const float DIRT_LIGHT_CEILING = 8.0;
// Three octaves of specks, in cells per lens-image height: fine grains over large flecks, each octave dimmer
// than the one below it. Stronger than these and the field stops reading as a dirty element.
static const float DIRT_FREQ_COARSE = 6.0;
static const float DIRT_FREQ_MEDIUM = 13.0;
static const float DIRT_FREQ_FINE = 27.0;
static const float DIRT_WEIGHT_COARSE = 1.0;
static const float DIRT_WEIGHT_MEDIUM = 0.7;
static const float DIRT_WEIGHT_FINE = 0.45;
static const uint DIRT_SEED_COARSE = 0x51edu;
static const uint DIRT_SEED_MEDIUM = 0x6a4du;
static const uint DIRT_SEED_FINE = 0x8321u;
// Each cell owns at most one speck: its centre stays inside the middle of the cell, its radius varies within
// the cell, and its brightness is the fraction of the remaining hash range. Most cells are empty, which is
// what keeps a dirty element from reading as fog.
static const float DIRT_CELL_MARGIN = 0.20;
static const float DIRT_CELL_SPREAD = 0.60;
static const float DIRT_SPECK_MIN_RADIUS = 0.11;
static const float DIRT_SPECK_RADIUS_SPREAD = 0.17;
static const float DIRT_SPECK_BODY = 0.70;
static const float DIRT_RIM_CENTER = 0.82;
static const float DIRT_RIM_WIDTH = 0.06;
static const float DIRT_SPECK_RIM = 0.30;
// Low-frequency unevenness in how grimy the element is, so its specks are not spread with uniform density.
static const float DIRT_GRIME_FLOOR = 0.55;
static const float DIRT_GRIME_RANGE = 0.45;
static const uint DIRT_SMUDGE_SEED = 0x2a17u;

float3 hotAt(float2 uv) {
    return hotImage.SampleLevel(uv, 0.0).rgb;
}

// Diffraction spikes. Taps march out at a uniform spacing, so the source's own width bridges the gaps between
// them, and carry the envelope exp(-falloff * f) so a spike fades towards its tip. The sum is normalised by
// the weights of one spike, so the tap count only trades smoothness for cost. `dither` slides the tap chain
// within its own spacing, which makes neighbouring texels sample different fractions of the spike and turns
// what would be a row of beads into a continuous ray once the display pass interpolates between them.
float3 starburst(float2 uv, float aspect, float dither) {
    int spikes = max(pc.spikes, 3);
    int samples = max(pc.samples, 1);
    float weightSum = 0.0;
    for (int i = 0; i < samples; i++) {
        float f = (float(i) + dither) / float(samples);
        weightSum += exp(-SPIKE_FALLOFF * f);
    }

    float3 acc = float3(0.0);
    for (int k = 0; k < spikes; k++) {
        float angle = 3.14159265359 * float(k) / float(spikes);
        // Divided by the aspect ratio so a spike is straight in pixels, not in UV: the image is wider than
        // it is tall.
        float2 direction = float2(cos(angle) / aspect, sin(angle));
        for (int i = 0; i < samples; i++) {
            float f = (float(i) + dither) / float(samples);
            float2 offset = direction * (f * pc.spikeLength);
            // Both signs of the offset: a gather only carries light from the side it reads, so a single
            // direction would light one half of a spike line and leave a one-armed star. The spike lines are
            // symmetric about the source, and so is this sum.
            acc += (hotAt(uv + offset) + hotAt(uv - offset)) * exp(-SPIKE_FALLOFF * f);
        }
    }
    return acc * (float(spikes) * SPIKE_GAIN * pc.starburst / max(weightSum, 1.0e-6));
}

// One aperture ring, gathered at a single radius. The ring is as thick as the source is wide, so a distant
// lamp makes a thin ring and a close sun a fat one -- which is what the lens does. The tap phases are rotated
// by a per-texel offset for the same reason as the spike dither: a shared phase chain would leave a budget of
// a few dozen taps as a string of beads.
float3 ring(float2 uv, float aspect, float radius, int taps, float dither) {
    // Two fetches per tap: at the default divisor a ring of this radius is about a hundred and twenty texels
    // around, so two dozen taps five texels apart would leave a string of beads instead of a ring. Halving the
    // spacing to about two texels lets bilinear filtering close the gaps; a coarser divisor halves the
    // circumference and the sample rate has to make the spacing back up.
    int count = taps * HALO_SUBTAPS;
    float3 sum = float3(0.0);
    for (int i = 0; i < count; i++) {
        float angle = TAU * (float(i) + dither) / float(count);
        sum += hotAt(uv + float2(cos(angle) / aspect * radius, sin(angle) * radius * HALO_ELLIPSE));
    }
    return sum / float(count);
}

// The chromatic fringe comes from gathering each channel at its own radius rather than from tinting a single
// ring: the wider red ring and the tighter blue one separate where the ring crosses a gradient, and overlap
// into a neutral where it does not.
float3 halo(float2 uv, float aspect, float dither) {
    int taps = max(pc.haloTaps, 8);
    float3 red = ring(uv, aspect, pc.haloRadius * (1.0 + HALO_CHROMA_R * pc.dispersion), taps, dither);
    float3 green = ring(uv, aspect, pc.haloRadius * (1.0 + HALO_CHROMA_G * pc.dispersion), taps, dither + 0.37);
    float3 blue = ring(uv, aspect, pc.haloRadius * (1.0 + HALO_CHROMA_B * pc.dispersion), taps, dither + 0.71);
    return float3(red.r, green.g, blue.b) * (HALO_GAIN * pc.halo);
}

// --- lens dirt ------------------------------------------------------------------------------------------
//
// A jittered lattice of soft-rimmed specks, three octaves, plus a low-frequency smudge: what a dust grain
// outside the focal plane looks like is a dim body with a brighter rim, and what a dirty element looks like
// is a handful of grains of very different sizes on top of uneven grime. No texture is sampled, so the
// pattern costs integer maths instead of bandwidth.

uint hashUint(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float hashFloat(int2 cell, uint seed) {
    uint x = uint(cell.x) * 0x9e3779b1u ^ uint(cell.y) * 0x85ebca77u;
    return float(hashUint(x ^ seed) >> 8) * (1.0 / 16777216.0);
}

float valueNoise(float2 p, float frequency, uint seed) {
    float2 q = p * frequency;
    int2 base = int2(floor(q));
    float2 f = q - float2(base);
    f = f * f * (3.0 - 2.0 * f);
    float top = lerp(hashFloat(base, seed), hashFloat(base + int2(1, 0), seed), f.x);
    float bottom = lerp(hashFloat(base + int2(0, 1), seed), hashFloat(base + int2(1, 1), seed), f.x);
    return lerp(top, bottom, f.y);
}

float dustLayer(float2 p, float frequency, uint seed) {
    float2 q = p * frequency;
    int2 base = int2(floor(q));
    float field = 0.0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int2 cell = base + int2(dx, dy);
            float offsetU = hashFloat(cell, seed);
            float offsetV = hashFloat(cell, seed + 0x68bc21ebu);
            float radiusHash = hashFloat(cell + int2(0x01234567, 0x07654321), seed + 0x02e5be93u);
            float brightHash = hashFloat(cell + int2(0x0badf00d, 0x00c0ffee), seed + 0x5f356495u);
            float2 center = float2(cell) + DIRT_CELL_MARGIN
                    + DIRT_CELL_SPREAD * float2(offsetU, offsetV);
            float radius = DIRT_SPECK_MIN_RADIUS + DIRT_SPECK_RADIUS_SPREAD * radiusHash;
            // Most cells stay empty: a uniform speck field reads as fog, not as a dirty element.
            float bright = saturate((brightHash - DIRT_EMPTY) / (1.0 - DIRT_EMPTY));
            float dist = length(q - center) / radius;
            float body = saturate(1.0 - dist);
            body = body * body * (3.0 - 2.0 * body);
            float rimDistance = dist - DIRT_RIM_CENTER;
            float rim = exp(-rimDistance * rimDistance / DIRT_RIM_WIDTH);
            field += bright * (body * DIRT_SPECK_BODY + rim * DIRT_SPECK_RIM);
        }
    }
    return saturate(field);
}

float3 lensDirt(float2 uv, float aspect) {
    // Grain coordinates are square in pixels, so a speck is round on screen and keeps its size in the frame
    // at any resolution or aspect ratio.
    float2 p = float2(uv.x * aspect, uv.y);
    float scale = max(pc.dirtScale, 0.01);
    float field = dustLayer(p, DIRT_FREQ_COARSE * scale, DIRT_SEED_COARSE)
            + dustLayer(p, DIRT_FREQ_MEDIUM * scale, DIRT_SEED_MEDIUM) * DIRT_WEIGHT_MEDIUM
            + dustLayer(p, DIRT_FREQ_FINE * scale, DIRT_SEED_FINE) * DIRT_WEIGHT_FINE;
    float grime = valueNoise(p, DIRT_SMUDGE_FREQ * scale, DIRT_SMUDGE_SEED);
    field = saturate(field) * (DIRT_GRIME_FLOOR + DIRT_GRIME_RANGE * grime);

    // The light that reaches the dust is the exposed bloom glow: it is already the frame's bright signal
    // spread the way the lens spreads it. Ceilinged so a sun does not turn every grain into an opaque blot.
    float3 light = max(glowImage.SampleLevel(uv, 0.0).rgb, float3(0.0));
    light = light / (1.0 + light / DIRT_LIGHT_CEILING);
    return light * (field * DIRT_GAIN * pc.dirt);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint width, height;
    flareImage.GetDimensions(width, height);
    int2 pixel = int2(dispatchId.xy);
    if (uint(pixel.x) >= width || uint(pixel.y) >= height) {
        return;
    }

    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    float aspect = float(width) / float(height);
    // One per-texel random phase shared by all three artifacts, so their taps interleave across neighbours.
    float dither = hashFloat(pixel, 0x1f35u);
    float3 flare = float3(0.0);
    if (pc.starburst > 0.0) {
        flare += starburst(uv, aspect, dither);
    }
    if (pc.halo > 0.0) {
        flare += halo(uv, aspect, dither);
    }
    if (pc.dirt > 0.0) {
        flare += lensDirt(uv, aspect);
    }
    flareImage[pixel] = float4(flare, 1.0);
}
