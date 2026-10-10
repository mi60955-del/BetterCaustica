// Diffuse sky irradiance bake, once per frame right after the sky-view LUT it reads.
//
// Convolves the dome into order-2 real spherical harmonics with the Lambert transfer applied, giving
// world_indirect.rgen a closed-form unoccluded sky irradiance for any normal. Without it, a shading point's
// entire sky contribution was the single cosine-sampled continuation ray that happened to escape, whose
// value swings by whatever direction it landed on — dominant at night, where the moon sits 24 EV below the
// sun and that sample is most of the image.
//
// One thread per coefficient, each walking the same fixed quadrature: no atomics, no clear pass, and the
// result is bit-stable for a given sky. 9 threads is deliberately tiny — the whole pass is ~512 dome
// fetches per coefficient, and it runs once per frame ahead of a trace that costs orders of magnitude more.
//
// Shares sky.slang with the other bakes and with world.rmiss, so the basis can never disagree with the
// dome it is integrating.
import world_common;
import sky;
import bindings;

[[vk::push_constant]] PushAddr pcAddr;

// SKY_IRRADIANCE_COEFFS — one thread per order-2 SH coefficient.
[shader("compute")]
[numthreads(9, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= uint(SKY_IRRADIANCE_COEFFS) || tid.y != 0u || tid.z != 0u) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pcAddr.worldPushAddr)[0];
    SkyState sky = skyState(push);
    int k = int(tid.x);

    // Equal-weight quadrature over the sphere: every sample carries 4pi/n. The Lambert scale turns the
    // radiance coefficient into an irradiance coefficient here so the evaluate side is a plain sum.
    float weight = 4.0 * SKY_PI / float(SKY_IRRADIANCE_SAMPLES) * shLambertTransfer(k);
    float3 acc = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < SKY_IRRADIANCE_SAMPLES; i++) {
        float3 dir = skyQuadratureDir(i, SKY_IRRADIANCE_SAMPLES);
        acc += skyDiffuseRadiance(skyViewLut, dir, sky) * shBasis(k, dir);
    }
    skyIrradianceImage[int2(k, 0)] = float4(max(acc * weight, float3(0.0, 0.0, 0.0)), 1.0);
}
