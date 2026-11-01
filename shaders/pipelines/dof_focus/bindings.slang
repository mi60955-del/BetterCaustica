[[vk::binding(0, 0)]] [format("r32f")] public RWTexture2D<float> focusDepth;
[[vk::binding(1, 0)]] public RWStructuredBuffer<float2, Std430DataLayout> focusMeter;
