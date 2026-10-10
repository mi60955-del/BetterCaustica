// Midpoint frame generation inputs: the two displayed frames at swapchain resolution, and the ray-traced
// motion guide at trace resolution that says where each surface was when the previous frame was captured.
[[vk::binding(0, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] public Sampler2D currentImage;
[[vk::binding(2, 0)]] public Sampler2D previousImage;
[[vk::binding(3, 0)]] [format("rg16f")] public RWTexture2D<float2> motionImage;
