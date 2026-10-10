import display_common;

[[vk::binding(0, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> rtImage;
[[vk::binding(2, 0)]] [format("r32f")] public RWTexture2D<float> exposureImage;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> hdrImage;
[[vk::binding(4, 0)]] public Sampler3D toneLut;
[[vk::binding(5, 0)]] public Sampler3D hdrToneLut;
[[vk::binding(6, 0)]] public Sampler3D lookLut;
[[vk::binding(7, 0)]] public Sampler2D bloomImage;
// Same-frame auto-exposure state, written by the exposure resolve immediately before this pass. The
// Purkinje shift needs the metered ABSOLUTE scene luminance, which only evScene carries: the applied
// exposure scalar has already normalised it out of the image.
[[vk::binding(8, 0)]] public StructuredBuffer<ExposureState, Std430DataLayout> exposureState;
