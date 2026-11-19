// Descriptor interface of the rain-streak pass. One set, two bindings; both images are sampled with a
// combined image sampler, so the pass needs no storage access and no separate LUT copies.
[[vk::binding(0, 0)]] public Sampler2D depthImage;
[[vk::binding(1, 0)]] public Sampler2D skyViewLut;
