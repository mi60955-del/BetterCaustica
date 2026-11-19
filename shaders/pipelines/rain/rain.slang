// Shared definitions for the rain-streak overlay: the push block, the drop field's geometry and the
// hash that places each drop.
//
// A streak is a camera-facing quad built entirely from SV_VertexID — six vertices per drop, no vertex
// buffer and no per-drop upload. The field is a disc of drops around the eye, which is the same
// camera-attached placement vanilla's own rain uses: what a frame draws is always the rain in front of
// the player rather than a fixed patch of world they walk out of.
module rain;

// Horizontal radius of the drop disc, in blocks. Wide enough that the streaks still cross the view when
// looking along the horizon, and narrow enough that the disc's edge is past the depth of field.
public static const float RAIN_DISC_RADIUS = 22.0;
// Vertical span of the falling slab, in blocks, and where its bottom edge sits relative to the eye.
// The bottom is below eye level on purpose: a slab that started at the eye would leave a horizon-wide
// band of clear air in a look-down view, which reads as the rain being switched off below the player.
public static const float RAIN_SLAB_HEIGHT = 16.0;
public static const float RAIN_SLAB_BOTTOM = -6.0;

/** One frame of rain-streak geometry, pushed into both stages. Exactly 64 bytes. */
public struct RainPush {
    // The frame's WorldPush: the camera-relative view-projection, the rebased camera offset and the
    // pre-exposure every scene write is scaled by, so the streaks land in the same units as the trace.
    public uint64_t worldPushAddr;
    // Display-resolution target size, in texels. The fragment turns SV_Position.xy into the UV the
    // render-resolution depth guide is sampled with.
    public float2 displaySize;
    // xyz the camera's horizontal right axis (world space): the streak quad's width axis. Horizontal
    // rather than perpendicular to the view so a streak stays vertical on screen at any pitch.
    public float4 camRight;
    // How settled the weather is, 0..1: the drops scale with the rain level, and this ramps the
    // streaks' opacity in so a drizzle arrives as sparse faint streaks instead of popping in.
    public float intensity;
    // Streak length along the fall direction, in blocks.
    public float length;
    // Fall speed in blocks per second.
    public float speed;
    // Streak width along camRight, in blocks.
    public float width;
    // The player's brightness dial 0..1, an opacity gain on top of intensity.
    public float brightness;
    // The wave clock in seconds. Shared with the water surface, so everything that animates on the
    // weather advances on one clock.
    public float time;
    // Drops this frame; the draw covers six vertices each.
    public uint dropCount;
    public float reserved;
    // xy the wind's horizontal velocity in blocks per second — the same wind the streaks lean along and
    // the drops drift with. zw reserved.
    public float4 wind;
};

/**
 * One round of an integer avalanche hash (PCG's output permutation followed by a multiply-xorshift).
 * Every per-drop constant below is a different salt of the drop index, so the drop's position, its
 * phase in the slab and its length are independent draws rather than one value reused three ways.
 */
public uint rainHash(uint value) {
    value = value * 747796405u + 2891336453u;
    uint word = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    return (word >> 22u) ^ word;
}

/** Hash to 0..1 with 24 bits of mantissa — fine enough that no two adjacent drops land on one point. */
public float rainHashUnit(uint value) {
    return float(rainHash(value) >> 8u) * (1.0 / 16777216.0);
}

/**
 * A drop's horizontal offset from the eye. sqrt of the radius sample is what makes the distribution
 * uniform over the disc's area instead of crowding the centre.
 */
public float2 rainDropOffset(uint drop) {
    float radius = sqrt(rainHashUnit(drop * 4u + 1u)) * RAIN_DISC_RADIUS;
    float angle = rainHashUnit(drop * 4u + 2u) * 6.283185307179586;
    return float2(cos(angle), sin(angle)) * radius;
}

/** The drop's position in the falling slab: 0 at the top edge, 1 at the bottom. */
public float rainDropPhase(uint drop, float time, float speed) {
    float start = rainHashUnit(drop * 4u + 3u);
    return frac(start + time * speed / RAIN_SLAB_HEIGHT);
}
