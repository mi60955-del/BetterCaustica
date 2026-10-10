import display_common;

[[vk::binding(0, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> gNormal;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> gAlbedo;
[[vk::binding(3, 0)]] [format("r32f")] public RWTexture2D<float> gDepth;
[[vk::binding(4, 0)]] [format("rg16f")] public RWTexture2D<float2> gMotion;
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture2D<float4> gSpecAlbedo;
[[vk::binding(6, 0)]] [format("rg16f")] public RWTexture2D<float2> gSpecMotion;
[[vk::binding(7, 0)]] [format("rgba16f")] public RWTexture2D<float4> sceneImage;
[[vk::binding(8, 0)]] [format("r32f")] public RWTexture2D<float> exposureImage;
[[vk::binding(9, 0)]] public StructuredBuffer<ExposureState, Std430DataLayout> exposureState;

// Volumetric diagnostics the primary raygen wrote this frame; the volume debug views read one channel
// each. Same render-resolution image the world pipeline has bound at its own binding 18.
[[vk::binding(10, 0)]] [format("rgba16f")] public RWTexture2D<float4> volumeDebug;
