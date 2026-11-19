// Rain streaks, vertex stage: one camera-facing quad per drop, generated from SV_VertexID alone.
//
// Each drop is a vertical strip in the world — its length runs along the fall direction, its width
// along the camera's horizontal right axis — so the quad always reads as a streak rather than as a
// billboard that turns to face the eye. The colour is resolved here, once per vertex, rather than per
// fragment: it is the same sky radiance for every drop in the frame, and the fragment stage is left
// with the depth test and the streak's own shape.
import rain;
import world_common;
import sky;
import bindings;

[[vk::push_constant]] RainPush pc;

// Two triangles over (left,top), (right,top), (left,bottom), (right,bottom). Corner x is -1..1 across
// the streak, y is 0 at its top and 1 at its bottom.
static const float2 STREAK_CORNERS[6] = {
    float2(-1.0, 0.0), float2(1.0, 0.0), float2(-1.0, 1.0),
    float2(-1.0, 1.0), float2(1.0, 0.0), float2(1.0, 1.0),
};

// The sky radiance a streak is lit by is a fraction of the dome's, not the whole of it: a drop is a
// thin scattering shell that only intercepts a sliver of the light crossing its column, and the sky
// the pixels behind it already carry is not removed by the blend.
static const float RAIN_RADIANCE_GAIN = 0.55;

// Near-field fade, in blocks from the eye: a drop that lands closer than this would fill a screen
// corner with a band wider than a block.
static const float RAIN_NEAR_FADE_START = 0.5;
static const float RAIN_NEAR_FADE_END = 1.6;

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;         // x 0..1 across the streak, y 0..1 from its top to its bottom
    [[vk::location(1)]] float opacity;     // this drop's own fade: disc edge, near field, slab ends
    [[vk::location(2)]] float3 radiance;   // ACEScg, already pre-exposed: colour per unit coverage
};

[shader("vertex")]
VertexOutput main(uint vertexIndex : SV_VertexID) {
    VertexOutput output;
    ConstPtr<WorldPush> push = ConstPtr<WorldPush>(pc.worldPushAddr);

    uint drop = vertexIndex / 6u;
    float2 corner = STREAK_CORNERS[vertexIndex - drop * 6u];

    // Camera-relative placement. The disc follows the eye horizontally; its base is snapped to whole
    // blocks so stepping off a ledge does not drag the whole field down with the camera, which would
    // read as the rain being glued to the player's helmet.
    float phase = rainDropPhase(drop, pc.time, pc.speed);
    float2 wind = pc.wind.xy;
    // How long this drop has been falling since it entered the slab's top edge. The wind carries it
    // sideways over exactly that time, so its drift and the streak's lean are one motion rather than
    // two that could disagree.
    float fallTime = phase * RAIN_SLAB_HEIGHT / max(pc.speed, 1.0e-3);
    float3 centre;
    centre.xz = rainDropOffset(drop) + wind * fallTime;
    centre.y = floor(push[0].camOffset.y) - push[0].camOffset.y
            + RAIN_SLAB_BOTTOM + RAIN_SLAB_HEIGHT * (1.0 - phase);

    // The streak runs along the drop's velocity — down the fall axis, leaning with the wind — and is
    // centred on the drop, so the fades below are symmetric about it.
    float3 fall = normalize(float3(wind.x, -pc.speed, wind.y));
    float3 position = centre
            - fall * (pc.length * (corner.y - 0.5))
            + pc.camRight.xyz * (pc.width * 0.5 * corner.x);

    output.position = mul(push[0].curViewProj, float4(position, 1.0));
    output.uv = float2(corner.x * 0.5 + 0.5, corner.y);

    // The drop's brightness falls off in the last couple of blocks at either end of the slab, where a
    // drop is either about to leave the field or has just been placed in it. Without it the two ends of
    // the fall would be visible as drops appearing and vanishing on two flat planes.
    float slabFade = saturate(phase * 8.0) * saturate((1.0 - phase) * 8.0);

    // The eye's own neighbourhood is empty: a streak is a distant object in every practical view, and
    // the one that happens to land on the camera would be a screen-wide bar. The disc's rim fades for
    // the mirror reason — it is where the field stops, and a hard edge there would turn every wide
    // look into a circle of rain drawn on the fog.
    float nearFade = smoothstep(RAIN_NEAR_FADE_START, RAIN_NEAR_FADE_END, length(centre));
    float discFade = 1.0 - smoothstep(RAIN_DISC_RADIUS * 0.6, RAIN_DISC_RADIUS, length(centre.xz));
    output.opacity = slabFade * nearFade * discFade;

    // Lit by the sky directly overhead: the zenith is the part of the dome a falling column actually
    // sees from underneath, and going through skyState/skyDiffuseRadiance here means the streaks take
    // the same weather dimming and the same time of day as every other surface the sky lights.
    SkyState state = skyState(push[0]);
    float3 skyRadiance = skyDiffuseRadiance(skyViewLut, float3(0.0, 1.0, 0.0), state);
    output.radiance = bt709ToAcesCg(max(skyRadiance, float3(0.0)))
            * (push[0].preExposure * RAIN_RADIANCE_GAIN);
    return output;
}
