// The gather's accumulation targets on the storage side: the far and near premultiplied fields for the
// SDR and (when HDR output is on) the PQ HDR channel, at half the display resolution. The compose pair
// the display mapper wrote and the lens depth the primary pass published are on the sampled side.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> farImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> nearImage;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> hdrFarImage;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> hdrNearImage;
[[vk::binding(4, 0)]] public Sampler2D sourceImage;
[[vk::binding(5, 0)]] public Sampler2D hdrSourceImage;
// The axial distance, in blocks, each pixel's pinhole ray travelled. Written by the primary pass this
// frame (the double-buffered cur side); a depth is the one number the circle of confusion needs.
[[vk::binding(6, 0)]] [format("r32f")] public RWTexture2D<float> focusDepth;
