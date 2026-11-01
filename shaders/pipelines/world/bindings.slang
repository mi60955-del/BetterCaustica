// Complete descriptor and push-constant interface for the world ray-tracing pipeline.
// Every world stage imports this module instead of redeclaring its own subset.

import world_common;

[[vk::push_constant]] public WorldPushConstants pc;

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> outImage;

[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> gNormal;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> gAlbedo;
[[vk::binding(5, 0)]] [format("r32f")] public RWTexture2D<float> gDepth;
[[vk::binding(6, 0)]] [format("rg16f")] public RWTexture2D<float2> gMotion;
[[vk::binding(7, 0)]] [format("rgba16f")] public RWTexture2D<float4> gSpecAlbedo;
[[vk::binding(8, 0)]] [format("rg16f")] public RWTexture2D<float2> gSpecMotion;

[[vk::binding(2, 0)]] public Sampler2D blockAlbedoAtlas;
[[vk::binding(9, 0)]] public Sampler2D celestialsAtlas;
[[vk::binding(10, 0)]] public Sampler2D skyViewLut;
[[vk::binding(11, 0)]] public Sampler2D transmittanceLut;
// Order-2 SH of this frame's sky dome with the Lambert transfer applied, 9 texels of RGB. Written by
// sky_lut/irradiance.comp and read by the indirect pass as the closed-form diffuse sky irradiance.
// Declared as a storage image because that is how it is allocated, even though the trace only loads it.
[[vk::binding(12, 0)]] [format("rgba16f")] public RWTexture2D<float4> skyIrradianceLut;

// Bounce-0 diffuse RIS reservoir, one image per side of a host-side ping-pong so a thread can read the
// previous frame's pixel — anywhere on screen after reprojection — while writing its own. Lanes are
// (light index as float, W, M, linear depth of the point that was shaded). W rather than the raw weight
// sum is what persists so the merge can re-weight the survivor by this frame's p-hat over the p-hat it was
// selected under.
// The companion RG16F pair carries the survivor's coordinates (s,t) on the emitter rectangle. They are
// stored because the merge reweights the survivor by its own p-hat, so that p-hat must belong to the same
// rectangle point whose weight went into the sum: re-drawing it each frame is unbiased in expectation but
// leaves the two quantities describing different samples, which measured as a large magnitude bias. Half
// precision is enough — a quadrature audit put the quantisation's contribution to that bias below its own
// noise floor.
[[vk::binding(13, 0)]] [format("rgba32f")] public RWTexture2D<float4> reservoirPrev;
[[vk::binding(14, 0)]] [format("rgba32f")] public RWTexture2D<float4> reservoirCur;
[[vk::binding(15, 0)]] [format("rg16f")] public RWTexture2D<float2> reservoirSamplePrev;
[[vk::binding(16, 0)]] [format("rg16f")] public RWTexture2D<float2> reservoirSampleCur;

[[vk::binding(0, 1)]] public Sampler2D entityAlbedoTex[];
[[vk::binding(1, 1)]] public Sampler2D materialSurface0Tex[];
[[vk::binding(2, 1)]] public Sampler2D materialNormalAoTex[];
[[vk::binding(3, 1)]] public Sampler2D materialSurface1Tex[];

// Vanilla end_sky.png (textures/environment/end_sky.png), bound by RtComposite with a linear, wrapping
// sampler: the end-sky cube repeats it 16x16 per face, which only REPEAT addressing can express. Sampled
// by the world miss shader's end-sky branch; the block-atlas fallback covers the frames before the
// texture is resolvable.
[[vk::binding(17, 0)]] public Sampler2D endSkyTex;

// Per-pixel volumetric diagnostics (fog transmittance / in-scatter luminance / cloud density / cloud
// shadow), written by the primary raygen only when WORLD_FLAG_VOLUME_DEBUG is set and shown by the debug
// present pass. Render resolution, matching the guides.
[[vk::binding(18, 0)]][format("rgba16f")] public RWTexture2D<float4> volumeDebug;

// The baked cloud-density volumes, one 2D-array atlas for every shape the cloud layer can place
// (RtCloudVolumes): RES layers per shape, layer = shape * RES + volume z. The cloud integrators in
// volume.slang fetch slices through it and lerp between adjacent layers themselves (2D-array filtering
// is nearest along the layer axis).
[[vk::binding(19, 0)]] public Sampler2DArray cloudVolumes;

// The lens ceiling's per-pixel history: the last frame's pinhole view depth, which the next frame's
// primary raygen reads before it builds its ray. Render resolution like the guides it sits beside, but it
// is not a guide -- DLSS-RR never samples it, and it holds the axial distance the lens math needs rather
// than the reversed-Z value RR reprojects with. A ping-pong pair for the same reason the reservoirs are
// one: a thread reads a texel another frame's trace wrote, so the two sides must never be the same image.
[[vk::binding(20, 0)]][format("r32f")] public RWTexture2D<float> focusDepthPrev;
[[vk::binding(21, 0)]][format("r32f")] public RWTexture2D<float> focusDepthCur;
