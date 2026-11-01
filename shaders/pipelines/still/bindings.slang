// The accumulated still render: an fp32 accumulator the finished display frame is added into, and the
// 8-bit image the average is divided out to for readback. The accumulator is float because the whole
// point is that a sum of many 8-bit frames is worth more than any single one of them — an 8-bit
// accumulator would only ever keep the last frame it saw.
[[vk::binding(0, 0)]] [format("rgba32f")] public RWTexture2D<float4> accumulatorImage;
[[vk::binding(1, 0)]] public Sampler2D sourceImage;
[[vk::binding(2, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
