// Probe: can the stock FidelityFX SDK Vulkan backend generate a frame on this driver?
//
// The question is not "does the API return OK". A dispatch can return OK while the
// reprojection reads a mis-scaled motion vector, and the picture is simply wrong. The scene is
// built so that the right answer is computable without FSR: one red bar and one striped world,
// each with an exactly known per-frame displacement, so the generated frame is correct only if
// the bar lands at the mean of its previous and current positions.
//
// Four arms feed FSR the same scene under four different motion-vector conventions. Exactly one
// can be right, and the measurement decides which - AMD's documentation contradicts the shader
// source about these units, so a citation would not settle it (see README.md).
//
// Two readings come out of the same run for free: whether HUDLessColor recomposes a screen-fixed
// square cleanly, and which Vulkan names the SDK asks the device for a function pointer to and
// cannot get.

#include "env.h"
#include "ffx_host.h"
#include "image_io.h"

#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/ffx_upscale.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// windows.h defines max/min as macros, which breaks every std::max in this file.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {

// --- options --------------------------------------------------------------------------

struct Options {
    std::string dllPath = "amd_fidelityfx_vk.dll";
    std::string shaderPath = "scene.comp.spv";
    // Path to ffxshim.dll. When set, the arm runs through the shim instead of calling the SDK
    // directly, which is how the shim's hand-built resource descriptions get checked.
    std::string viaShim;
    std::string outDir = ".";
    // Which function pointer the SDK is handed for its own Vulkan lookups: `fallback` answers
    // instance-level names through vkGetInstanceProcAddr, `strict` behaves like the plain
    // device query, and `none` gives the SDK a null pointer so it uses its own.
    std::string shim = "fallback";
    uint32_t width = 512;
    uint32_t height = 288;
    // 0 means "same as the display size". Setting it lower reproduces the arrangement the game
    // actually runs: colour and output at display resolution, depth and motion at a smaller
    // render resolution. The probe passed at equal sizes, so whether the split matters is a
    // question to answer here rather than one to answer with a game launch.
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t frames = 24;
    // Colour noise amplitude, mirroring a path-traced frame's denoise residual.
    float noise = 0.0f;
    bool validation = false;
    bool noHudless = false;
    bool debugView = false;
    bool controlOnly = false;
    // Handing ffx-api our own allocator is diagnostic-only: it releases some blocks without
    // calling back, so the registry is a hint rather than a leak report.
    bool guard = false;
    std::vector<std::string> onlyArms;
};

// FSR divides `motionVectorScale` by the motion vector target size inside the shader and adds
// the product to a UV coordinate, so the pairing that produces a UV displacement is a raw
// vector in render pixels with the scale left at one.
struct Arm {
    const char* name;
    bool mvInUv;       // value stored into the motion image: UV, or render pixels
    bool scaleIsSize;  // motionVectorScale = (width, height) instead of (1, 1)
    float sign;
    const char* expectation;
};

const Arm kArms[] = {
    {"pixels_s1",  false, false,  1.0f, "vectors in pixels, scale 1: divides down to UV"},
    {"pixels_sW",  false, true,   1.0f, "vectors in pixels, scale = size: UV ends up pixel-sized"},
    {"pixels_neg", false, false, -1.0f, "sign flipped: history read ahead of the camera"},
    {"uv_s1",      true,  false,  1.0f, "vectors already in UV, divided once more"},
};

// --- scene ----------------------------------------------------------------------------

struct SceneState {
    double objX = 0.0;
    double objPrevX = 0.0;
    double objHalfW = 0.0;
    double hudLeft = 0.0;
    double hudTop = 0.0;
    double hudW = 0.0;
    double hudH = 0.0;
};

// The bar oscillates at a different rate from the world pan, so no single vector describes the
// frame and a wrong reprojection cannot accidentally agree with a global offset. All positions
// are normalised against the target width, which is what lets the colour and the guides be
// rendered at different resolutions and still agree about where the bar is.
SceneState sceneAt(double t, uint32_t /*width*/, uint32_t /*height*/) {
    const double twoPi = 6.283185307179586;
    SceneState s;
    s.objX = 0.5 + 0.22 * std::sin(twoPi * t / 34.0 + 1.1);
    s.objPrevX = 0.5 + 0.22 * std::sin(twoPi * (t - 1.0) / 34.0 + 1.1);
    s.objHalfW = 0.045;
    s.hudW = 0.18;
    s.hudH = 0.22;
    s.hudLeft = 0.03;
    s.hudTop = 0.72;
    return s;
}

double panAt(double t, uint32_t /*width*/) {
    return 0.06 * std::sin(6.283185307179586 * t / 60.0);
}

// Mirrors the push-constant block in shaders/scene.comp: no vector wider than two floats, so
// nothing needs the 16-byte alignment the GLSL rules would otherwise insert.
struct PushConstants {
    float displaySize[2];
    float guideSize[2];
    float writeColor;
    float writeGuides;
    float mvOutScale[2];
    float mvOutSign;
    float panX;
    float panPrevX;
    float objX;
    float objPrevX;
    float objHalfW;
    float stripePeriod;
    float hudLeft;
    float hudTop;
    float hudW;
    float hudH;
    float noiseAmp;
    float frameSeed;
};

static_assert(sizeof(PushConstants) == 84, "push constants must match scene.comp");

// --- measurement ----------------------------------------------------------------------

struct Measurement {
    double centroid = -1.0;   // weighted mean x of reddish pixels; -1 when nothing was found
    double coverage = 0.0;    // pixels that read as saturated red
    double hudFill = 0.0;     // fraction of the HUD rectangle that reads as saturated green
};

// The bar is pure red on a grey world and a green HUD, so `red - max(green, blue)` isolates it
// and neither distractor contributes.
Measurement measure(const std::vector<uint8_t>& rgba, uint32_t w, uint32_t h, const SceneState& s) {
    Measurement m;
    double sumWeight = 0.0;
    double sumX = 0.0;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const size_t o = (size_t(y) * w + x) * 4;
            const double r = rgba[o];
            const double g = rgba[o + 1];
            const double b = rgba[o + 2];
            const double red = std::max(0.0, r - std::max(g, b) - 90.0) / 165.0;
            if (red > 0.0) {
                sumWeight += red;
                sumX += red * (x + 0.5);
            }
            if (red > 0.5) {
                m.coverage += 1.0;
            }
        }
    }
    if (sumWeight > 0.0) {
        m.centroid = sumX / sumWeight;
    }

    const uint32_t x0 = uint32_t(s.hudLeft * w), x1 = uint32_t((s.hudLeft + s.hudW) * w);
    const uint32_t y0 = uint32_t(s.hudTop * h), y1 = uint32_t((s.hudTop + s.hudH) * h);
    const double area = double(x1 - x0) * double(y1 - y0);
    double green = 0.0;
    for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
            const size_t o = (size_t(y) * w + x) * 4;
            if (rgba[o + 1] > 180.0 && rgba[o] < 90.0 && rgba[o + 2] < 90.0) {
                green += 1.0;
            }
        }
    }
    m.hudFill = area > 0.0 ? green / area : 0.0;
    return m;
}

FloatImage rgbaToFloat(const std::vector<uint8_t>& rgba, uint32_t w, uint32_t h) {
    FloatImage out(w, h);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        out.data[i * 3 + 0] = rgba[i * 4 + 0] / 255.0f;
        out.data[i * 3 + 1] = rgba[i * 4 + 1] / 255.0f;
        out.data[i * 3 + 2] = rgba[i * 4 + 2] / 255.0f;
    }
    return out;
}

void writePngFromRgba(const std::string& path, const std::vector<uint8_t>& rgba, uint32_t w, uint32_t h) {
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    writePng(path, w, h, rgb);
}

// --- ffx resource construction ---------------------------------------------------------

// Every image this probe hands the SDK sits in VK_IMAGE_LAYOUT_GENERAL, and UNORDERED_ACCESS is
// the only FFX resource state that maps to GENERAL. Passing a read state instead makes the
// backend treat the image as being in SHADER_READ_ONLY_OPTIMAL and it stops issuing the
// transitions that the writes here require.
constexpr uint32_t kState = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;

FfxApiResource apiResource(const Img& img, uint32_t additionalUsage) {
    FfxApiResourceDescription description =
            ffxApiGetImageResourceDescriptionVK(img.image, img.createInfo, additionalUsage);
    // The helper copies mipLevels through and the backend feeds it straight into
    // VkImageViewCreateInfo::levelCount, where zero is invalid.
    description.mipCount = 1;
    return ffxApiGetResourceVK(reinterpret_cast<void*>(img.image), description, kState);
}

// --- scene pipeline --------------------------------------------------------------------

struct ScenePipeline {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

ScenePipeline createScenePipeline(Env& env, VkShaderModule module, const Img& colorHud,
                                  const Img& colorHudless, const Img& depth, const Img& motion) {
    ScenePipeline p;

    VkDescriptorSetLayoutBinding bindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;
    checkOk(vkCreateDescriptorSetLayout(env.device, &layoutInfo, nullptr, &p.layout),
            "vkCreateDescriptorSetLayout(scene)");

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    checkOk(vkCreateDescriptorPool(env.device, &poolInfo, nullptr, &p.pool),
            "vkCreateDescriptorPool(scene)");

    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = p.pool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &p.layout;
    checkOk(vkAllocateDescriptorSets(env.device, &setInfo, &p.set), "vkAllocateDescriptorSets(scene)");

    const Img* images[4] = {&colorHud, &colorHudless, &depth, &motion};
    VkDescriptorImageInfo infos[4];
    VkWriteDescriptorSet writes[4];
    for (uint32_t i = 0; i < 4; ++i) {
        infos[i] = {};
        infos[i].imageView = images[i]->view;
        infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(env.device, 4, writes, 0, nullptr);

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &p.layout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &range;
    checkOk(vkCreatePipelineLayout(env.device, &plInfo, nullptr, &p.pipelineLayout),
            "vkCreatePipelineLayout(scene)");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage = stage;
    cp.layout = p.pipelineLayout;
    checkOk(vkCreateComputePipelines(env.device, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipeline),
            "vkCreateComputePipelines(scene)");
    return p;
}

void destroyScenePipeline(Env& env, ScenePipeline& p) {
    if (p.pipeline) vkDestroyPipeline(env.device, p.pipeline, nullptr);
    if (p.pipelineLayout) vkDestroyPipelineLayout(env.device, p.pipelineLayout, nullptr);
    if (p.pool) vkDestroyDescriptorPool(env.device, p.pool, nullptr);
    if (p.layout) vkDestroyDescriptorSetLayout(env.device, p.layout, nullptr);
    p = ScenePipeline{};
}

// Owns everything the arm hands to FFX, so that an early return after a failed context
// creation cannot leak an image the device still references.
struct ArmResources {
    Env& env;
    Img colorHud;
    Img colorHudless;
    Img depth;
    Img motion;
    Img output;
    ScenePipeline scene;

    ArmResources(Env& e, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh, VkShaderModule module)
        : env(e)
        , colorHud(createImage(e, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        , colorHudless(createImage(e, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        , depth(createImage(e, rw, rh, VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        , motion(createImage(e, rw, rh, VK_FORMAT_R16G16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        , output(createImage(e, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        , scene(createScenePipeline(e, module, colorHud, colorHudless, depth, motion)) {}

    ~ArmResources() {
        destroyScenePipeline(env, scene);
        destroyImage(env, output);
        destroyImage(env, motion);
        destroyImage(env, depth);
        destroyImage(env, colorHudless);
        destroyImage(env, colorHud);
    }
};

// One frame's scene: colour at display size, guides at render size. Two dispatches, because the
// two targets have different extents and the shader bounds-checks against whichever it is
// writing. The motion vector is a normalised displacement multiplied by the guide width, so it
// comes out in *guide* pixels - which is the unit FSR expects for a vector at render resolution.
void dispatchScene(const ScenePipeline& scene, VkCommandBuffer cmd, uint32_t W, uint32_t H,
                   uint32_t RW, uint32_t RH, const SceneState& s, double t, const Arm& arm,
                   float noiseAmp) {
    PushConstants push{};
    push.displaySize[0] = float(W);
    push.displaySize[1] = float(H);
    push.guideSize[0] = float(RW);
    push.guideSize[1] = float(RH);
    push.mvOutScale[0] = arm.mvInUv ? 1.0f / float(RW) : 1.0f;
    push.mvOutScale[1] = arm.mvInUv ? 1.0f / float(RH) : 1.0f;
    push.mvOutSign = arm.sign;
    push.panX = float(panAt(t, W));
    push.panPrevX = float(panAt(t - 1.0, W));
    push.objX = float(s.objX);
    push.objPrevX = float(s.objPrevX);
    push.objHalfW = float(s.objHalfW);
    push.stripePeriod = 0.09375f;  // 48 px of a 512 px target, expressed as a fraction
    push.hudLeft = float(s.hudLeft);
    push.hudTop = float(s.hudTop);
    push.hudW = float(s.hudW);
    push.hudH = float(s.hudH);
    push.noiseAmp = noiseAmp;
    push.frameSeed = float(t);

    auto run = [&](float colorFlag, float guideFlag, uint32_t gridW, uint32_t gridH) {
        push.writeColor = colorFlag;
        push.writeGuides = guideFlag;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayout,
                                0, 1, &scene.set, 0, nullptr);
        vkCmdPushConstants(cmd, scene.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(cmd, (gridW + 15) / 16, (gridH + 15) / 16, 1);
    };
    run(1.0f, 0.0f, W, H);
    run(0.0f, 1.0f, RW, RH);
}

// --- one arm ---------------------------------------------------------------------------

struct ArmResult {
    std::string name;
    // The frame call goes through native/ffx_shim instead of the SDK's own entry points, which
    // collapses configure/prepare/generate into one return code.
    bool viaShim = false;
    bool contextUp = false;
    ffxReturnCode_t createRc = FFX_API_RETURN_ERROR;
    ffxReturnCode_t configureRc = FFX_API_RETURN_ERROR;
    ffxReturnCode_t prepareRc = FFX_API_RETURN_ERROR;
    ffxReturnCode_t generateRc = FFX_API_RETURN_ERROR;
    std::string failure;
    std::string versionName;
    uint64_t versionId = 0;
    uint64_t vramBytes = 0;
    double measuredCentroid = -1.0;
    double errMid = 0.0;
    double errPrev = 0.0;
    double errCur = 0.0;
    double coverage = 0.0;
    double hudFill = 0.0;
    double blendMad = 0.0;
};

ArmResult runArm(FfxHost& host, Env& env, const Options& opt, const Arm& arm,
                 VkShaderModule sceneModule) {
    ArmResult result;
    result.name = arm.name;

    const uint32_t W = opt.width;
    const uint32_t H = opt.height;
    const uint32_t RW = opt.renderWidth != 0 ? opt.renderWidth : W;
    const uint32_t RH = opt.renderHeight != 0 ? opt.renderHeight : H;

    // --- context ---------------------------------------------------------------------
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = env.device;
    backend.vkPhysicalDevice = env.physicalDevice;
    backend.vkDeviceProcAddr = opt.shim == "none" ? nullptr : ffxDeviceProcAddrShim;

    ffxCreateContextDescFrameGeneration createFg{};
    createFg.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    // The probe's depth is reversed (1 at the near plane) and has a finite far plane, which is
    // the same shape as Caustica's gDepth.
    createFg.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED;
    createFg.displaySize = {W, H};
    createFg.maxRenderSize = {RW, RH};
    createFg.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    if (opt.debugView) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    }

    // Effect description first, backend chained after it - the order the SDK's own sample
    // builds with ffx::CreateContext(createFg, backendDesc).
    createFg.header.pNext = &backend.header;
    backend.header.pNext = nullptr;

    ffxContext context = nullptr;
    std::fprintf(stderr, "[arm %s] ffxCreateContext ...\n", arm.name);
    result.createRc = host.createContext(&context, &createFg.header,
                                         opt.guard ? ffxGuardedCallbacks() : nullptr);
    std::fprintf(stderr, "[arm %s] ffxCreateContext -> %s\n", arm.name,
                 ffxReturnCodeName(result.createRc));
    if (result.createRc != FFX_API_RETURN_OK) {
        result.failure = std::string("ffxCreateContext: ") + ffxReturnCodeName(result.createRc);
        return result;
    }
    result.contextUp = true;

    ffxQueryGetProviderVersion version{};
    version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    host.query(&context, &version.header);
    result.versionId = version.versionId;
    if (version.versionName) {
        result.versionName = version.versionName;
    }

    try {
        ArmResources res(env, W, H, RW, RH, sceneModule);

        uint64_t frameId = 1;
        const float fovY = 1.2217305f;  // 70 degrees, matching nothing in particular
        std::vector<uint8_t> prevColor;
        std::vector<uint8_t> curColor;
        std::vector<uint8_t> generated;
        double prevObjX = 0.0;
        double curObjX = 0.0;
        SceneState lastScene{};

        for (uint32_t t = 0; t < opt.frames; ++t) {
            const SceneState s = sceneAt(double(t), W, H);
            lastScene = s;

            const uint32_t dispatchFlags =
                    FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY
                    | (opt.debugView ? FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW : 0u);

            ffxConfigureDescFrameGeneration configure{};
            configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            // No swapchain and no callbacks: the app keeps its own VkSwapchainKHR and drives
            // both dispatches itself, which is what Caustica's presenter already does.
            configure.frameGenerationEnabled = true;
            configure.allowAsyncWorkloads = false;
            configure.flags = dispatchFlags;
            configure.generationRect = {0, 0, int32_t(W), int32_t(H)};
            configure.frameID = frameId;
            configure.HUDLessColor = opt.noHudless ? FfxApiResource{} : apiResource(res.colorHudless, 0);
            result.configureRc = host.configure(&context, &configure.header);

            ffxDispatchDescFrameGenerationPrepare prepare{};
            prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prepare.frameID = frameId;
            prepare.flags = dispatchFlags;
            prepare.commandList = nullptr;  // filled once the command buffer exists
            prepare.renderSize = {RW, RH};
            prepare.jitterOffset = {0.0f, 0.0f};
            const float scale = arm.scaleIsSize ? float(RW) : 1.0f;
            prepare.motionVectorScale = {scale, scale};
            prepare.frameTimeDelta = 16.7f;
            prepare.cameraNear = 0.05f;
            prepare.cameraFar = 1000.0f;
            prepare.cameraFovAngleVertical = fovY;
            prepare.viewSpaceToMetersFactor = 1.0f;

            ffxDispatchDescFrameGenerationPrepareCameraInfo camera{};
            camera.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
            camera.cameraPosition[0] = float(panAt(double(t), W));
            camera.cameraUp[1] = 1.0f;
            camera.cameraRight[0] = 1.0f;
            camera.cameraForward[2] = -1.0f;
            prepare.header.pNext = &camera.header;
            camera.header.pNext = nullptr;

            VkCommandBuffer cmd = beginCommands(env);
            dispatchScene(res.scene, cmd, W, H, RW, RH, s, double(t), arm, opt.noise);
            memoryBarrier(cmd);

            prepare.commandList = cmd;
            prepare.depth = apiResource(res.depth, 0);
            prepare.motionVectors = apiResource(res.motion, 0);
            result.prepareRc = host.dispatch(&context, &prepare.header);

            ffxDispatchDescFrameGeneration generate{};
            generate.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
            generate.commandList = cmd;
            generate.presentColor = apiResource(res.colorHud, 0);
            generate.outputs[0] = apiResource(res.output, FFX_API_RESOURCE_USAGE_UAV);
            generate.numGeneratedFrames = 1;
            generate.reset = t == 0;
            generate.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
            generate.minMaxLuminance[0] = 1.0f;
            generate.minMaxLuminance[1] = 100.0f;
            generate.generationRect = {0, 0, int32_t(W), int32_t(H)};
            generate.frameID = frameId;
            result.generateRc = host.dispatch(&context, &generate.header);

            submitCommands(env, cmd);

            if (result.configureRc != FFX_API_RETURN_OK
                    || result.prepareRc != FFX_API_RETURN_OK
                    || result.generateRc != FFX_API_RETURN_OK) {
                char buffer[192];
                std::snprintf(buffer, sizeof(buffer),
                              "frame %u: configure=%s prepare=%s generate=%s", t,
                              ffxReturnCodeName(result.configureRc),
                              ffxReturnCodeName(result.prepareRc),
                              ffxReturnCodeName(result.generateRc));
                result.failure = buffer;
                break;
            }

            // Only the two frames the measurement needs are copied back; reading every frame
            // would change what the timing-sensitive parts of the algorithm see.
            if (t + 2 == opt.frames) {
                prevColor = readImage(env, res.colorHud);
                prevObjX = s.objX * W;
            }
            if (t + 1 == opt.frames) {
                curColor = readImage(env, res.colorHud);
                generated = readImage(env, res.output);
                curObjX = s.objX * W;
            }
            ++frameId;
        }

        FfxApiEffectMemoryUsage usage{};
        ffxQueryDescFrameGenerationGetGPUMemoryUsage memQuery{};
        memQuery.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
        memQuery.gpuMemoryUsageFrameGeneration = &usage;
        host.query(&context, &memQuery.header);
        result.vramBytes = usage.totalUsageInBytes;

        if (result.failure.empty() && !generated.empty()) {
            const Measurement gm = measure(generated, W, H, lastScene);
            const Measurement cm = measure(curColor, W, H, lastScene);
            const Measurement pm = measure(prevColor, W, H, lastScene);

            result.measuredCentroid = gm.centroid;
            result.coverage = gm.coverage;
            result.hudFill = gm.hudFill;
            result.errMid = gm.centroid - 0.5 * (prevObjX + curObjX);
            result.errPrev = gm.centroid - pm.centroid;
            result.errCur = gm.centroid - cm.centroid;

            const FloatImage g = rgbaToFloat(generated, W, H);
            const FloatImage a = rgbaToFloat(prevColor, W, H);
            const FloatImage b = rgbaToFloat(curColor, W, H);
            FloatImage blended(W, H);
            for (size_t i = 0; i < blended.data.size(); ++i) {
                blended.data[i] = 0.5f * (a.data[i] + b.data[i]);
            }
            // A generated frame that is nothing but the two sources averaged scores the same
            // as the blend itself; anything better than that came from the vector field.
            result.blendMad = meanAbsDifference(g, blended);

            const std::string base = opt.outDir + "/arm_" + arm.name;
            writePngFromRgba(base + "_gen.png", generated, W, H);
            writePngFromRgba(base + "_cur.png", curColor, W, H);
            writePngFromRgba(base + "_prev.png", prevColor, W, H);
        }
    } catch (const std::exception& e) {
        if (result.failure.empty()) {
            result.failure = e.what();
        }
    }

    host.destroyContext(&context, nullptr);
    return result;
}

// --- reporting and cli ------------------------------------------------------------------

// Control: bring up a context for a different effect on the same device with the same callback
// and the same backend description. If this one survives where frame generation faults, the
// shared Vulkan backend start-up is not what breaks and the frame-generation path is.
bool controlCreateUpscale(FfxHost& host, Env& env, const Options& opt) {
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = env.device;
    backend.vkPhysicalDevice = env.physicalDevice;
    backend.vkDeviceProcAddr = opt.shim == "none" ? nullptr : ffxDeviceProcAddrShim;

    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    create.maxRenderSize = {opt.width, opt.height};
    create.maxUpscaleSize = {opt.width, opt.height};
    create.fpMessage = nullptr;
    create.header.pNext = &backend.header;
    backend.header.pNext = nullptr;

    ffxContext context = nullptr;
    const ffxReturnCode_t rc =
            host.createContext(&context, &create.header, opt.guard ? ffxGuardedCallbacks() : nullptr);
    std::printf("[control] ffxCreateContext(upscale) -> %s\n", ffxReturnCodeName(rc));
    if (rc == FFX_API_RETURN_OK && context != nullptr) {
        host.destroyContext(&context, nullptr);
    }
    return rc == FFX_API_RETURN_OK;
}

// --- the shim, as a second way to reach the same SDK -----------------------------------

// native/ffx_shim.cpp rebuilds FfxApiResourceDescription by hand rather than calling
// ffxApiGetImageResourceDescriptionVK, because a shim has no VkImageCreateInfo to give the
// helper. A field that disagrees with AMD's produces calls that all return OK and a frame that
// is quietly wrong, so the shim runs over this same scene and its bar centroid is compared
// against the direct path's. That is the only check that can say anything about it.
struct ShimApi {
    using FnLoad = int(__cdecl*)(const char*);
    using FnCreate = int(__cdecl*)(void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                   int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                   int32_t, uint64_t*, char*, int32_t);
    using FnFrame = int(__cdecl*)(void*, uint64_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                  float, float, float, float, float, float, float, float, float,
                                  const float*, const float*, const float*, const float*,
                                  uint64_t, int32_t, uint64_t, int32_t, uint64_t, int32_t,
                                  uint64_t, int32_t, uint32_t, uint32_t, uint64_t, int32_t, int32_t);
    using FnVoid = void(__cdecl*)();
    using FnInt = int(__cdecl*)();
    using FnCString = const char*(__cdecl*)();

    HMODULE module = nullptr;
    FnLoad load = nullptr;
    FnCreate create = nullptr;
    FnFrame frame = nullptr;
    FnVoid destroy = nullptr;
    FnVoid unload = nullptr;
    FnInt available = nullptr;
    FnCString lastMessage = nullptr;
    std::string version;

    bool open(const std::string& path, std::string& error) {
        module = LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (module == nullptr) {
            error = "LoadLibraryExA(" + path + ") failed";
            return false;
        }
        auto symbol = [this](const char* name) { return GetProcAddress(module, name); };
        load = reinterpret_cast<FnLoad>(symbol("ffxshim_load"));
        create = reinterpret_cast<FnCreate>(symbol("ffxshim_create"));
        frame = reinterpret_cast<FnFrame>(symbol("ffxshim_frame"));
        destroy = reinterpret_cast<FnVoid>(symbol("ffxshim_destroy"));
        unload = reinterpret_cast<FnVoid>(symbol("ffxshim_unload"));
        available = reinterpret_cast<FnInt>(symbol("ffxshim_available"));
        auto message = reinterpret_cast<FnCString>(symbol("ffxshim_last_message"));
        auto versionOf = reinterpret_cast<FnCString>(symbol("ffxshim_version"));
        if (!load || !create || !frame || !destroy || !unload || !available) {
            error = "ffxshim.dll is missing exports (rebuild native/ffx_shim)";
            FreeLibrary(module);
            module = nullptr;
            return false;
        }
        if (message) {
            lastMessage = message;
        }
        return true;
    }
};

ArmResult runArmViaShim(ShimApi& shim, Env& env, const Options& opt, const Arm& arm,
                        VkShaderModule sceneModule) {
    ArmResult result;
    result.name = arm.name;
    result.viaShim = true;

    const uint32_t W = opt.width;
    const uint32_t H = opt.height;
    const uint32_t RW = opt.renderWidth != 0 ? opt.renderWidth : W;
    const uint32_t RH = opt.renderHeight != 0 ? opt.renderHeight : H;
    const int32_t rgba8 = VK_FORMAT_R8G8B8A8_UNORM;
    const int32_t r32f = VK_FORMAT_R32_SFLOAT;
    const int32_t rg16f = VK_FORMAT_R16G16_SFLOAT;

    char version[64] = {0};
    uint64_t vram = 0;
    result.createRc = static_cast<ffxReturnCode_t>(shim.create(
            env.device, env.physicalDevice, reinterpret_cast<void*>(&vkGetDeviceProcAddr),
            W, H, RW, RH, rgba8, opt.noHudless ? 0 : rgba8,
            1 /* depthInverted */, 0 /* depthInfinite */, 0 /* hdr */,
            0 /* display-resolution vectors */, 0 /* jittered vectors */, opt.debugView ? 1 : 0,
            &vram, version, static_cast<int32_t>(sizeof(version))));
    if (result.createRc != FFX_API_RETURN_OK) {
        result.failure = "ffxshim_create returned " + std::to_string(result.createRc);
        if (shim.lastMessage) {
            result.failure += std::string("; ") + shim.lastMessage();
        }
        return result;
    }
    result.contextUp = true;
    result.vramBytes = vram;
    result.versionName = version[0] ? version : "via ffxshim";

    try {
        ArmResources res(env, W, H, RW, RH, sceneModule);
        uint64_t frameId = 1;
        std::vector<uint8_t> prevColor;
        std::vector<uint8_t> curColor;
        std::vector<uint8_t> generated;
        double prevObjX = 0.0;
        double curObjX = 0.0;
        SceneState lastScene{};
        float cameraPosition[3] = {static_cast<float>(panAt(0.0, W)), 0.0f, 0.0f};
        const float cameraUp[3] = {0.0f, 1.0f, 0.0f};
        const float cameraRight[3] = {1.0f, 0.0f, 0.0f};
        const float cameraForward[3] = {0.0f, 0.0f, -1.0f};

        for (uint32_t t = 0; t < opt.frames; ++t) {
            const SceneState s = sceneAt(double(t), W, H);
            lastScene = s;

            cameraPosition[0] = float(panAt(double(t), W));

            VkCommandBuffer cmd = beginCommands(env);
            dispatchScene(res.scene, cmd, W, H, RW, RH, s, double(t), arm, opt.noise);
            memoryBarrier(cmd);

            const float scale = arm.scaleIsSize ? float(RW) : 1.0f;
            // VkImage is a handle typedef over a pointer, so it needs an explicit cast to the
            // uint64_t the shim's flat ABI takes.
            auto handle = [](VkImage image) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image)); };
            const int rc = shim.frame(
                    cmd, frameId, t == 0 ? 1 : 0, RW, RH, W, H,
                    0.0f, 0.0f, scale, scale, 16.7f,
                    0.05f, 1000.0f, 1.2217305f, 1.0f,
                    cameraPosition, cameraUp, cameraRight, cameraForward,
                    handle(res.depth.image), r32f,
                    handle(res.motion.image), rg16f,
                    handle(res.colorHud.image), rgba8,
                    opt.noHudless ? 0 : handle(res.colorHudless.image), opt.noHudless ? 0 : rgba8,
                    opt.noHudless ? 0 : W, opt.noHudless ? 0 : H,
                    handle(res.output.image), rgba8,
                    opt.debugView ? 1 : 0);
            submitCommands(env, cmd);

            result.generateRc = static_cast<ffxReturnCode_t>(rc);
            result.configureRc = result.generateRc;
            result.prepareRc = result.generateRc;
            if (rc != FFX_API_RETURN_OK) {
                result.failure = "ffxshim_frame returned " + std::to_string(rc);
                if (shim.lastMessage) {
                    result.failure += std::string("; ") + shim.lastMessage();
                }
                break;
            }

            if (t + 2 == opt.frames) {
                prevColor = readImage(env, res.colorHud);
                prevObjX = s.objX * W;
            }
            if (t + 1 == opt.frames) {
                curColor = readImage(env, res.colorHud);
                generated = readImage(env, res.output);
                curObjX = s.objX * W;
            }
            ++frameId;
        }

        if (result.failure.empty() && !generated.empty()) {
            const Measurement gm = measure(generated, W, H, lastScene);
            const Measurement cm = measure(curColor, W, H, lastScene);
            const Measurement pm = measure(prevColor, W, H, lastScene);
            result.measuredCentroid = gm.centroid;
            result.coverage = gm.coverage;
            result.hudFill = gm.hudFill;
            result.errMid = gm.centroid - 0.5 * (prevObjX + curObjX);
            result.errPrev = gm.centroid - pm.centroid;
            result.errCur = gm.centroid - cm.centroid;

            const std::string base = opt.outDir + "/shim_" + arm.name;
            writePngFromRgba(base + "_gen.png", generated, W, H);
        }
    } catch (const std::exception& e) {
        if (result.failure.empty()) {
            result.failure = e.what();
        }
    }

    shim.destroy();
    return result;
}

// Exactly one of the two ways in is live per run, and every exit path has to close it.
void releaseNatives(ShimApi& shim, FfxHost& host, bool useShim) {
    if (useShim) {
        if (shim.unload) {
            shim.unload();
        }
    } else {
        releaseNatives(shim, host, useShim);
    }
}

int report(const ArmResult& r) {
    if (r.viaShim) {
        std::printf("  rc: create=%s frame=%s\n", ffxReturnCodeName(r.createRc),
                    ffxReturnCodeName(r.generateRc));
    } else {
        std::printf("  rc: create=%s configure=%s prepare=%s generate=%s\n",
                    ffxReturnCodeName(r.createRc), ffxReturnCodeName(r.configureRc),
                    ffxReturnCodeName(r.prepareRc), ffxReturnCodeName(r.generateRc));
    }
    if (!r.contextUp) {
        std::printf("  VERDICT: no context - %s\n", r.failure.c_str());
        return 3;
    }
    std::printf("  provider: 0x%llx \"%s\"  vram=%llu B\n",
                static_cast<unsigned long long>(r.versionId), r.versionName.c_str(),
                static_cast<unsigned long long>(r.vramBytes));
    if (!r.failure.empty()) {
        std::printf("  VERDICT: ran but failed - %s\n", r.failure.c_str());
        return 3;
    }
    std::printf("  bar centroid in generated frame = %.2f  (errMid=%+.2f errPrev=%+.2f errCur=%+.2f)\n",
                r.measuredCentroid, r.errMid, r.errPrev, r.errCur);
    std::printf("  bar coverage=%.0f px  hudFill=%.3f  mad(against a plain blend)=%.5f\n",
                r.coverage, r.hudFill, r.blendMad);

    const bool tracks = std::fabs(r.errMid) < 2.0
                     && std::fabs(r.errMid) < std::fabs(r.errPrev)
                     && std::fabs(r.errMid) < std::fabs(r.errCur);
    std::printf("  VERDICT: %s\n",
                tracks ? "generated frame puts the bar at the midpoint"
                       : "generated frame does NOT track the midpoint");
    return tracks ? 0 : 2;
}

void usage() {
    std::printf(
        "ffxprobe - does stock FSR 3.1 frame generation work on this Vulkan driver?\n"
        "\n"
        "  --dll PATH       amd_fidelityfx_vk.dll (default: next to the executable)\n"
        "  --shader PATH    scene.comp.spv (default: next to the executable)\n"
        "  --width N        render and display width  (default 512)\n"
        "  --height N       render and display height (default 288)\n"
        "  --render-width N guide (depth/motion) width, 0 = display width (default 0)\n"
        "  --render-height N guide height, 0 = display height (default 0)\n"
        "  --frames N       frames per arm            (default 24)\n"
        "  --arm NAME       run only this arm; repeatable\n"
        "  --shim MODE      fallback | strict | none (how vkGetDeviceProcAddr is answered)\n"
        "  --control        create only an upscaler context, then stop\n"
        "  --guard          allocate the SDK's memory through a guarded allocator\n"
        "  --via-shim PATH  run the arms through ffxshim.dll instead of the SDK directly\n"
        "  --noise F        per-frame colour noise amplitude, 0..1 (default 0)\n"
        "  --no-hudless     leave HUDLessColor empty\n"
        "  --debug-view     ask FSR for its debug view and enable API checking\n"
        "  --validation     turn on the Vulkan validation layer\n"
        "  --out DIR        where to write the PNGs (default: current directory)\n"
        "\n"
        "exit: 0 some arm tracked the midpoint, 2 no arm did, 3 a call failed\n");
}

bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        std::string value;
        auto next = [&](bool& ok) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", a.c_str());
                ok = false;
                return std::string();
            }
            return std::string(argv[++i]);
        };
        bool ok = true;
        if (a == "--help" || a == "-h") {
            usage();
            std::exit(0);
        } else if (a == "--dll") {
            opt.dllPath = next(ok);
        } else if (a == "--shader") {
            opt.shaderPath = next(ok);
        } else if (a == "--via-shim") {
            opt.viaShim = next(ok);
        } else if (a == "--out") {
            opt.outDir = next(ok);
        } else if (a == "--arm") {
            value = next(ok);
            opt.onlyArms.push_back(value);
        } else if (a == "--shim") {
            opt.shim = next(ok);
        } else if (a == "--control") {
            opt.controlOnly = true;
        } else if (a == "--guard") {
            opt.guard = true;
        } else if (a == "--width") {
            opt.width = uint32_t(std::stoul(next(ok)));
        } else if (a == "--height") {
            opt.height = uint32_t(std::stoul(next(ok)));
        } else if (a == "--render-width") {
            opt.renderWidth = uint32_t(std::stoul(next(ok)));
        } else if (a == "--render-height") {
            opt.renderHeight = uint32_t(std::stoul(next(ok)));
        } else if (a == "--frames") {
            opt.frames = uint32_t(std::stoul(next(ok)));
        } else if (a == "--noise") {
            opt.noise = std::stof(next(ok));
        } else if (a == "--no-hudless") {
            opt.noHudless = true;
        } else if (a == "--debug-view") {
            opt.debugView = true;
        } else if (a == "--validation") {
            opt.validation = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return false;
        }
        if (!ok) {
            return false;
        }
    }
    if (opt.shim != "fallback" && opt.shim != "strict" && opt.shim != "none") {
        std::fprintf(stderr, "--shim must be fallback, strict or none\n");
        return false;
    }
    const uint32_t renderW = opt.renderWidth != 0 ? opt.renderWidth : opt.width;
    const uint32_t renderH = opt.renderHeight != 0 ? opt.renderHeight : opt.height;
    if (renderW > opt.width || renderH > opt.height) {
        std::fprintf(stderr, "the guide size cannot exceed the display size\n");
        return false;
    }
    if (opt.frames < 3) {
        std::fprintf(stderr, "--frames must be at least 3 so that a previous frame exists\n");
        return false;
    }
    return true;
}

std::string absolutePath(const std::string& path) {
    char buffer[MAX_PATH];
    const DWORD n = GetFullPathNameA(path.c_str(), MAX_PATH, buffer, nullptr);
    if (n == 0 || n >= MAX_PATH) {
        return path;
    }
    return std::string(buffer, n);
}

// --- crash attribution -----------------------------------------------------------------

void printModuleOffset(const char* label, const void* address) {
    if (!address) {
        std::fprintf(stderr, "%s null\n", label);
        return;
    }
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                         | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module)
        && module) {
        char path[MAX_PATH];
        if (GetModuleFileNameA(module, path, MAX_PATH)) {
            const char* base = std::strrchr(path, '\\');
            const auto offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(module);
            std::fprintf(stderr, "%s %s + 0x%llx\n", label, base ? base + 1 : path,
                         static_cast<unsigned long long>(offset));
            return;
        }
    }
    std::fprintf(stderr, "%s %p (no module)\n", label, address);
}

// A call through a null function pointer inside amd_fidelityfx_vk.dll is otherwise just an
// address with no story. The backtrace names the caller, and the shim log says whether the
// pointer that came back null was one this program handed the SDK or one the SDK lost
// internally - which is the difference between a bug in the integration and a bug in the DLL.
LONG WINAPI unhandledException(PEXCEPTION_POINTERS info) {
    const EXCEPTION_RECORD* record = info->ExceptionRecord;
    const uint32_t code = static_cast<uint32_t>(record->ExceptionCode);
    // 0x4001...' range is debug/output events that merely pass through the handler.
    if ((code & 0xC0000000u) != 0xC0000000u) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Reading a faulting stack can itself fault, and a handler that re-enters turns one crash
    // into an unreadable wall of them.
    static bool reentered = false;
    if (reentered) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    reentered = true;
    std::fprintf(stderr, "\n[crash] exception 0x%08lx\n", static_cast<unsigned long>(code));
    printModuleOffset("[crash] fault at ", record->ExceptionAddress);
    ffxReportGuardViolations();

#ifdef _M_X64
    // A call through a null pointer leaves the caller's return address at the top of the
    // stack, and the bytes just before it are the call instruction itself. The displacement
    // in that instruction is an offset into FFX's own function table, which names the entry
    // point that came back null - something no amount of reading the headers would give.
    const uint64_t rsp = info->ContextRecord->Rsp;
    std::fprintf(stderr, "[crash] Rip=%p Rsp=%p\n",
                 reinterpret_cast<void*>(static_cast<uintptr_t>(info->ContextRecord->Rip)),
                 reinterpret_cast<void*>(static_cast<uintptr_t>(rsp)));
    const uint8_t* atRsp = reinterpret_cast<const uint8_t*>(rsp);
    uint64_t returnAddress = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t candidate = 0;
        memcpy(&candidate, atRsp + i * 8, 8);
        HMODULE owner = nullptr;
        if (candidate && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                                            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                            reinterpret_cast<LPCSTR>(static_cast<uintptr_t>(candidate)), &owner)
            && owner) {
            returnAddress = candidate;
            std::fprintf(stderr, "[crash] return address at rsp+%d\n", i * 8);
            break;
        }
    }
    if (returnAddress) {
        printModuleOffset("[crash] called from ", reinterpret_cast<const void*>(returnAddress));
        const uint8_t* before = reinterpret_cast<const uint8_t*>(returnAddress) - 16;
        std::fprintf(stderr, "[crash] bytes before it:");
        for (int i = 0; i < 16; ++i) {
            std::fprintf(stderr, " %02x", before[i]);
        }
        std::fprintf(stderr, "\n");
    }
#endif

    void* frames[24] = {};
    const uint16_t captured = static_cast<uint16_t>(
            RtlCaptureStackBackTrace(1, 24, frames, nullptr));
    for (uint16_t i = 0; i < captured; ++i) {
        char label[32];
        std::snprintf(label, sizeof(label), "[crash] frame %2u  ", i);
        printModuleOffset(label, frames[i]);
    }

    if (FfxHost* host = ffxActiveHost()) {
        std::fprintf(stderr, "[crash] shim answered %u lookups, %u needed the instance fallback\n",
                     host->deviceLookups, static_cast<unsigned>(host->deviceLookupMisses.size()));
        for (const std::string& miss : host->deviceLookupMisses) {
            std::fprintf(stderr, "[crash]   not a device command: %s\n", miss.c_str());
        }
        const size_t from = host->shimLog.size() > 24 ? host->shimLog.size() - 24 : 0;
        for (size_t i = from; i < host->shimLog.size(); ++i) {
            std::fprintf(stderr, "[crash]   proc %s%s\n", host->shimLog[i].c_str(),
                         host->shimLog[i].back() == '!' ? "  <- instance fallback" : "");
        }
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

}  // namespace

int main(int argc, char** argv) {
    // A probe that dies part way through must still show what it got up to.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, unhandledException);

    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        return 2;
    }

    FfxHost host;
    ShimApi shim;
    const bool useShim = !opt.viaShim.empty();
    std::string error;
    if (useShim) {
        const std::string shimPath = absolutePath(opt.viaShim);
        if (!shim.open(shimPath, error)) {
            std::fprintf(stderr, "ffxprobe: %s\n", error.c_str());
            return 3;
        }
        // The shim resolves the SDK itself; --dll names it so both modes test the same binary.
        const int loadRc = shim.load(absolutePath(opt.dllPath).c_str());
        std::printf("[ffxprobe] via %s: load=%d available=%d\n", shimPath.c_str(), loadRc,
                    shim.available());
        if (loadRc != FFX_API_RETURN_OK || !shim.available()) {
            std::fprintf(stderr, "ffxprobe: ffxshim_load failed (%d)\n", loadRc);
            return 3;
        }
    } else {
        const std::string dll = absolutePath(opt.dllPath);
        if (!ffxLoad(host, dll, error)) {
            std::fprintf(stderr, "ffxprobe: %s\n", error.c_str());
            return 3;
        }
        std::printf("[ffxprobe] loaded %s\n", dll.c_str());
    }

    std::vector<std::string> notes;
    Env env;
    try {
        const std::string deviceName = envCreate(env, opt.validation, notes);
        std::printf("[ffxprobe] device: %s\n", deviceName.c_str());
        std::printf("[ffxprobe] this driver advertises %u device extensions\n",
                    deviceExtensionCount(env.physicalDevice));
        if (!useShim) {
            ffxBindShim(host, env.instance);
            ffxSetShimFallback(opt.shim != "strict");
            host.logShim = true;
            std::printf("[ffxprobe] message callback: %s\n",
                        ffxReturnCodeName(ffxEnableMessages(FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE)));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ffxprobe: %s\n", e.what());
        releaseNatives(shim, host, useShim);
        return 3;
    }

    VkShaderModule sceneModule = VK_NULL_HANDLE;
    try {
        sceneModule = loadShaderModule(env, opt.shaderPath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ffxprobe: %s\n", e.what());
        envDestroy(env);
        releaseNatives(shim, host, useShim);
        return 3;
    }

    for (const std::string& note : notes) {
        std::printf("[note] %s\n", note.c_str());
    }
    std::printf("[ffxprobe] %ux%u, %u frames per arm, hudless=%s, validation=%s, shim=%s\n",
                opt.width, opt.height, opt.frames, opt.noHudless ? "off" : "on",
                env.validation ? "on" : "off", opt.shim.c_str());

    if (opt.controlOnly) {
        const bool up = controlCreateUpscale(host, env, opt);
        vkDestroyShaderModule(env.device, sceneModule, nullptr);
        envDestroy(env);
        releaseNatives(shim, host, useShim);
        std::printf("[ffxprobe] control %s\n", up ? "succeeded" : "failed");
        return up ? 0 : 3;
    }

    int worst = 0;
    bool anyTracked = false;
    for (const Arm& arm : kArms) {
        if (!opt.onlyArms.empty()) {
            const auto found = std::find(opt.onlyArms.begin(), opt.onlyArms.end(), std::string(arm.name));
            if (found == opt.onlyArms.end()) {
                continue;
            }
        }
        std::printf("\n[arm %s] %s\n", arm.name, arm.expectation);
        const size_t missesBefore = host.deviceLookupMisses.size();
        ArmResult r;
        try {
            r = useShim ? runArmViaShim(shim, env, opt, arm, sceneModule)
                        : runArm(host, env, opt, arm, sceneModule);
        } catch (const std::exception& e) {
            std::printf("  threw: %s\n", e.what());
            worst = std::max(worst, 3);
            continue;
        }
        const int code = report(r);
        worst = std::max(worst, code);
        anyTracked = anyTracked || code == 0;

        if (host.deviceLookupMisses.size() > missesBefore) {
            std::printf("  vkGetDeviceProcAddr could not answer:");
            for (size_t i = missesBefore; i < host.deviceLookupMisses.size(); ++i) {
                std::printf(" %s", host.deviceLookupMisses[i].c_str());
            }
            std::printf("\n");
        }
        if (env.validation) {
            std::printf("  validation errors so far: %u\n", validationErrorCount());
        }
        if (opt.guard) {
            std::printf("  allocations written past their end: %u\n", ffxReportGuardViolations());
        }
    }

    vkDestroyShaderModule(env.device, sceneModule, nullptr);
    envDestroy(env);
    ffxUnload(host);

    const int exitCode = anyTracked ? 0 : (worst == 0 ? 2 : worst);
    std::printf("\n[ffxprobe] exit %d\n", exitCode);
    return exitCode;
}
