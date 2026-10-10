// Flat-C shim over the FidelityFX SDK's unified API, so that Java never has to reproduce an
// FFX struct layout.
//
// ffx-api is already a C ABI, but its entry points take six nested descriptor structs
// (ffxCreateBackendVKDesc, ffxCreateContextDescFrameGeneration, ffxConfigureDescFrameGeneration,
// ffxDispatchDescFrameGenerationPrepare, its chained camera-info extension, and
// ffxDispatchDescFrameGeneration), each carrying FfxApiResource values whose description is
// itself a struct with a union in it. Hand-encoding those as Panama MemoryLayouts would be a
// silent-mismatch bug waiting to happen, so they live here, compiled against AMD's real headers,
// and the boundary is scalars and raw Vulkan handles.
//
// The SDK is reached through LoadLibrary + GetProcAddress rather than an import library:
// ffx_api.h declares its entry points with __declspec(dllexport), and the SDK ships no .lib for
// the path this project vendors.
//
// Every resource handed to FFX here is a VkImage cast to void*, in VK_IMAGE_LAYOUT_GENERAL,
// described with mipCount 1. That is what ffx-api's Vulkan backend expects: RegisterResourceVK
// does `reinterpret_cast<VkImage>(resource)`, makes its own image views, and
// getVKImageLayoutFromResourceState maps FFX_API_RESOURCE_STATE_UNORDERED_ACCESS to GENERAL.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <cstdio>
#include <cstring>
#include <string>

// ffx-api's own headers use FFX_API for its component exports, but that macro comes from
// ffx_core.h, which the ffx-api include set does not pull in.
#define FFX_API __declspec(dllexport)

namespace {

PfnFfxCreateContext g_createContext = nullptr;
PfnFfxDestroyContext g_destroyContext = nullptr;
PfnFfxConfigure g_configure = nullptr;
PfnFfxQuery g_query = nullptr;
PfnFfxDispatch g_dispatch = nullptr;

HMODULE g_sdkModule = nullptr;
ffxContext g_context = nullptr;
ffxReturnCode_t g_lastReturn = FFX_API_RETURN_OK;
int g_lastVkResult = VK_SUCCESS;
char g_lastMessage[512] = {0};
char g_versionName[64] = {0};

// ffxApiMessage hands a wchar_t*, which is UTF-16 on Windows. Transcode to UTF-8 with non-ASCII
// folded to '?' so the string survives a GBK console on the Java side.
void WINAPI sdkMessage(uint32_t type, const wchar_t* message) {
    char text[512];
    text[0] = '\0';
    if (message) {
        WideCharToMultiByte(CP_UTF8, 0, message, -1, text, static_cast<int>(sizeof(text)), "?", nullptr);
    }
    std::snprintf(g_lastMessage, sizeof(g_lastMessage), "%s: %s",
                  type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warn", text);
}

// Java hands paths over as UTF-8, and the A-series Windows APIs would read the same bytes through
// the system's ANSI code page, which mojibakes every non-ASCII install path (CJK usernames,
// localized directory names). Transcode once and stay in the W API family end to end.
std::wstring wideFromUtf8(const char* text) {
    int length = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (length <= 0) {
        return std::wstring();
    }
    std::wstring wide(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, wide.data(), length);
    wide.resize(static_cast<size_t>(length) - 1);  // drop the terminator the conversion wrote
    return wide;
}

// The inverse, for log lines: non-ASCII folds to '?' so the text survives a GBK console.
std::string utf8FromWide(const std::wstring& wide) {
    int length = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()),
                                     nullptr, 0, "?", nullptr);
    if (length <= 0) {
        return std::string();
    }
    std::string text(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()),
                        text.data(), length, "?", nullptr);
    return text;
}

FfxApiResource importedImage(VkImage image, uint32_t width, uint32_t height, VkFormat format,
                             uint32_t additionalUsage) {
    FfxApiResourceDescription description{};
    description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    description.format = ffxApiGetSurfaceFormatVK(format);
    description.width = width;
    description.height = height;
    // depth is the texture-array layer count, and these are single-layer 2D images.
    description.depth = 1;
    // ffx-api feeds mipCount straight into VkImageViewCreateInfo::levelCount, where 0 is invalid.
    description.mipCount = 1;
    description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    // READ_ONLY plus the caller's UAV bit: the backend picks VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE for
    // inputs and STORAGE_IMAGE for outputs from exactly these flags.
    description.usage = FFX_API_RESOURCE_USAGE_READ_ONLY | additionalUsage;
    return ffxApiGetResourceVK(reinterpret_cast<void*>(image), description,
                               FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
}

}  // namespace

extern "C" {

/** Whether amd_fidelityfx_vk.dll resolved with all five unified-API entry points. */
FFX_API int ffxshim_available(void) {
    return g_createContext && g_destroyContext && g_configure && g_query && g_dispatch ? 1 : 0;
}

/**
 * Load the SDK. dllPath is a UTF-8 byte string naming either a full path to
 * amd_fidelityfx_vk.dll or a directory to look in; empty means "next to this shim".
 * Returns 0 on success, an FfxApiReturnCodes value otherwise.
 */
FFX_API int ffxshim_load(const char* dllPath) {
    std::wstring path;
    if (dllPath != nullptr && dllPath[0] != '\0') {
        path = wideFromUtf8(dllPath);
        const DWORD attributes = GetFileAttributesW(path.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
            if (!path.empty() && path.back() != L'\\') {
                path += L'\\';
            }
            path += L"amd_fidelityfx_vk.dll";
        }
    } else {
        wchar_t modulePath[MAX_PATH];
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&ffxshim_available), &self)
                || GetModuleFileNameW(self, modulePath, MAX_PATH) == 0) {
            return FFX_API_RETURN_ERROR;
        }
        std::wstring directory(modulePath);
        const size_t separator = directory.find_last_of(L"\\/");
        if (separator == std::wstring::npos) {
            return FFX_API_RETURN_ERROR;
        }
        directory.erase(separator + 1);
        path = directory + L"amd_fidelityfx_vk.dll";
    }

    // LOAD_WITH_ALTERED_SEARCH_PATH so the SDK's own imports (vulkan-1.dll, VCRUNTIME140.dll)
    // resolve starting from its directory rather than the JVM's.
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (module == nullptr) {
        std::snprintf(g_lastMessage, sizeof(g_lastMessage), "error: LoadLibraryExW(%s) failed (%lu)",
                      utf8FromWide(path).c_str(), static_cast<unsigned long>(GetLastError()));
        return FFX_API_RETURN_ERROR;
    }
    g_createContext = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
    g_destroyContext = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
    g_configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(module, "ffxConfigure"));
    g_query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
    g_dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
    if (!g_createContext || !g_destroyContext || !g_configure || !g_query || !g_dispatch) {
        FreeLibrary(module);
        std::snprintf(g_lastMessage, sizeof(g_lastMessage), "error: %s lacks the unified API exports",
                      utf8FromWide(path).c_str());
        return FFX_API_RETURN_NO_PROVIDER;
    }
    g_sdkModule = module;

    ffxConfigureDescGlobalDebug1 debug{};
    debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
    debug.fpMessage = sdkMessage;
    debug.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
    // A null context addresses the SDK's global state, which is where the message sink lives.
    g_configure(nullptr, &debug.header);
    return FFX_API_RETURN_OK;
}

/**
 * Create the frame-generation context.
 *
 * The two sizes are separate because FSR interpolates at display resolution from guides at render
 * resolution: displayWidth/Height is the swapchain extent, maxRenderWidth/Height is what the ray
 * tracer renders at, and depth/motion vectors are expected at the render size unless
 * displayResMotionVectors is set.
 *
 * backbufferVkFormat and hudlessVkFormat are real VkFormat values; the conversion to
 * FfxApiSurfaceFormat happens here. Pass 0 for hudlessVkFormat to run without a HUD-less input.
 *
 * Writes the SDK's own device-memory estimate to *outVramBytes when that pointer is non-null, and
 * the implementation's version name into versionOut. Returns an FfxApiReturnCodes value.
 */
FFX_API int ffxshim_create(void* vkDevice, void* vkPhysicalDevice, void* vkGetDeviceProcAddr,
                           uint32_t displayWidth, uint32_t displayHeight,
                           uint32_t maxRenderWidth, uint32_t maxRenderHeight,
                           int32_t backbufferVkFormat, int32_t hudlessVkFormat,
                           int32_t depthInverted, int32_t depthInfinite, int32_t hdrColor,
                           int32_t displayResMotionVectors, int32_t jitteredMotionVectors,
                           int32_t debugChecking,
                           uint64_t* outVramBytes, char* versionOut, int32_t versionOutLength) {
    // Written before anything can fail, so the caller never reads an undefined estimate.
    if (outVramBytes != nullptr) {
        *outVramBytes = 0;
    }
    if (!ffxshim_available()) {
        return FFX_API_RETURN_NO_PROVIDER;
    }
    if (g_context != nullptr) {
        // One context at a time: the caller destroys before recreating, so a second create here
        // would leak the first and its device memory.
        return FFX_API_RETURN_ERROR;
    }

    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = static_cast<VkDevice>(vkDevice);
    backend.vkPhysicalDevice = static_cast<VkPhysicalDevice>(vkPhysicalDevice);
    backend.vkDeviceProcAddr = static_cast<PFN_vkGetDeviceProcAddr>(vkGetDeviceProcAddr);

    ffxCreateContextDescFrameGeneration createFg{};
    createFg.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    if (depthInverted) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED;
    }
    if (depthInfinite) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE;
    }
    if (hdrColor) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    if (displayResMotionVectors) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    }
    if (jitteredMotionVectors) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
    }
    if (debugChecking) {
        createFg.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    }
    createFg.displaySize = {displayWidth, displayHeight};
    createFg.maxRenderSize = {maxRenderWidth, maxRenderHeight};
    createFg.backBufferFormat = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(backbufferVkFormat));

    // A HUD-less input may be a different format from the backbuffer (for example RGBA8 UNORM
    // while the swapchain is 10:10:10:2), which the SDK cannot infer.
    ffxCreateContextDescFrameGenerationHudless hudless{};
    if (hudlessVkFormat != 0) {
        hudless.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
        hudless.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(hudlessVkFormat));
        createFg.header.pNext = &backend.header;
        backend.header.pNext = &hudless.header;
        hudless.header.pNext = nullptr;
    } else {
        createFg.header.pNext = &backend.header;
        backend.header.pNext = nullptr;
    }

    g_lastReturn = g_createContext(&g_context, &createFg.header, nullptr);
    if (g_lastReturn != FFX_API_RETURN_OK) {
        g_context = nullptr;
        return g_lastReturn;
    }

    ffxQueryGetProviderVersion version{};
    version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    g_query(&g_context, &version.header);
    if (versionOut != nullptr && versionOutLength > 0) {
        versionOut[0] = '\0';
        if (version.versionName != nullptr) {
            std::strncpy(versionOut, version.versionName, static_cast<size_t>(versionOutLength) - 1);
            versionOut[versionOutLength - 1] = '\0';
            std::snprintf(g_versionName, sizeof(g_versionName), "%s", versionOut);
        }
    }

    if (outVramBytes != nullptr) {
        FfxApiEffectMemoryUsage usage{};
        ffxQueryDescFrameGenerationGetGPUMemoryUsage query{};
        query.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
        query.gpuMemoryUsageFrameGeneration = &usage;
        if (g_query(&g_context, &query.header) == FFX_API_RETURN_OK) {
            *outVramBytes = usage.totalUsageInBytes;
        }
    }
    return FFX_API_RETURN_OK;
}

/**
 * Record one generated frame: configure, prepare, and generate, all into the same command buffer.
 *
 * frameId must increase by exactly one between calls. FSR treats any other step as a discontinuity
 * and resets its history, so a skipped or repeated id silently costs quality.
 *
 * Motion vectors are expected as previous-position-minus-current-position in render pixels, with
 * motionVectorScaleX/Y left at 1: the runtime divides the scale by the motion vector target size
 * and then adds the result to a UV, so pixels with scale 1 is what produces a UV displacement.
 *
 * cameraPosition/Up/Right/Forward may be null; from FSR 3.1.4 onwards they are wanted, and passing
 * all-zero draws a warning out of the SDK's own debug checking.
 *
 * Returns an FfxApiReturnCodes value; 0 means the generated frame is in outputImage.
 */
FFX_API int ffxshim_frame(void* commandBuffer, uint64_t frameId, int32_t reset,
                          uint32_t renderWidth, uint32_t renderHeight,
                          uint32_t displayWidth, uint32_t displayHeight,
                          float jitterX, float jitterY,
                          float motionVectorScaleX, float motionVectorScaleY,
                          float frameTimeDeltaMs,
                          float cameraNear, float cameraFar, float cameraFovAngleVertical,
                          float viewSpaceToMetersFactor,
                          const float* cameraPosition, const float* cameraUp,
                          const float* cameraRight, const float* cameraForward,
                          uint64_t depthImage, int32_t depthVkFormat,
                          uint64_t motionImage, int32_t motionVkFormat,
                          uint64_t presentColorImage, int32_t presentColorVkFormat,
                          uint64_t hudlessImage, int32_t hudlessVkFormat,
                          uint32_t hudlessWidth, uint32_t hudlessHeight,
                          uint64_t outputImage, int32_t outputVkFormat,
                          int32_t debugView) {
    if (g_context == nullptr) {
        return FFX_API_RETURN_ERROR;
    }
    // DRAW_DEBUG_VIEW is implemented by the frame-interpolation *swapchain* module, which
    // NO_SWAPCHAIN_CONTEXT_NOTIFY bypasses, so asking for it here produces nothing. Tear lines
    // are drawn by the inpainting pass inside the interpolation component itself, which does run,
    // and AMD documents them precisely as the "is a generated frame reaching the display" test.
    const uint32_t flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY
                         | (debugView ? FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES : 0u);

    ffxConfigureDescFrameGeneration configure{};
    configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    // No swapchain and no callbacks: the application keeps its own VkSwapchainKHR and drives both
    // dispatches itself, which is how the present path is already built for DLSS-FG.
    configure.frameGenerationEnabled = true;
    configure.allowAsyncWorkloads = false;
    configure.flags = flags;
    configure.generationRect = {0, 0, static_cast<int32_t>(displayWidth), static_cast<int32_t>(displayHeight)};
    configure.frameID = frameId;
    // The HUD-less target is a copy of the main render target, and the swapchain extent can differ
    // from it by a pixel in either axis. Describing it with the display size would make the SDK read
    // outside the image, so it carries its own dimensions.
    if (hudlessImage != 0 && hudlessVkFormat != 0 && hudlessWidth != 0 && hudlessHeight != 0) {
        configure.HUDLessColor = importedImage(reinterpret_cast<VkImage>(hudlessImage),
                                               hudlessWidth, hudlessHeight,
                                               static_cast<VkFormat>(hudlessVkFormat), 0);
    }
    g_lastReturn = g_configure(&g_context, &configure.header);
    if (g_lastReturn != FFX_API_RETURN_OK) {
        return g_lastReturn;
    }

    ffxDispatchDescFrameGenerationPrepare prepare{};
    prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    prepare.frameID = frameId;
    prepare.flags = flags;
    prepare.commandList = commandBuffer;
    prepare.renderSize = {renderWidth, renderHeight};
    prepare.jitterOffset = {jitterX, jitterY};
    prepare.motionVectorScale = {motionVectorScaleX, motionVectorScaleY};
    prepare.frameTimeDelta = frameTimeDeltaMs;
    prepare.cameraNear = cameraNear;
    prepare.cameraFar = cameraFar;
    prepare.cameraFovAngleVertical = cameraFovAngleVertical;
    prepare.viewSpaceToMetersFactor = viewSpaceToMetersFactor;
    prepare.depth = importedImage(reinterpret_cast<VkImage>(depthImage), renderWidth, renderHeight,
                                 static_cast<VkFormat>(depthVkFormat), 0);
    prepare.motionVectors = importedImage(reinterpret_cast<VkImage>(motionImage), renderWidth, renderHeight,
                                         static_cast<VkFormat>(motionVkFormat), 0);

    ffxDispatchDescFrameGenerationPrepareCameraInfo camera{};
    if (cameraPosition != nullptr || cameraUp != nullptr || cameraRight != nullptr || cameraForward != nullptr) {
        camera.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
        if (cameraPosition != nullptr) {
            std::memcpy(camera.cameraPosition, cameraPosition, sizeof(camera.cameraPosition));
        }
        if (cameraUp != nullptr) {
            std::memcpy(camera.cameraUp, cameraUp, sizeof(camera.cameraUp));
        }
        if (cameraRight != nullptr) {
            std::memcpy(camera.cameraRight, cameraRight, sizeof(camera.cameraRight));
        }
        if (cameraForward != nullptr) {
            std::memcpy(camera.cameraForward, cameraForward, sizeof(camera.cameraForward));
        }
        prepare.header.pNext = &camera.header;
        camera.header.pNext = nullptr;
    }
    g_lastReturn = g_dispatch(&g_context, &prepare.header);
    if (g_lastReturn != FFX_API_RETURN_OK) {
        return g_lastReturn;
    }

    ffxDispatchDescFrameGeneration generate{};
    generate.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    generate.commandList = commandBuffer;
    generate.presentColor = importedImage(reinterpret_cast<VkImage>(presentColorImage),
                                         displayWidth, displayHeight,
                                         static_cast<VkFormat>(presentColorVkFormat), 0);
    generate.outputs[0] = importedImage(reinterpret_cast<VkImage>(outputImage),
                                       displayWidth, displayHeight,
                                       static_cast<VkFormat>(outputVkFormat),
                                       FFX_API_RESOURCE_USAGE_UAV);
    // FSR 3.1.x on this SDK generates one frame per call; multi-frame generation arrived with the
    // DX12-only FSR4 backend.
    generate.numGeneratedFrames = 1;
    generate.reset = reset != 0;
    generate.backbufferTransferFunction =
            static_cast<VkFormat>(presentColorVkFormat) == VK_FORMAT_R16G16B16A16_SFLOAT
                    ? FFX_API_BACKBUFFER_TRANSFER_FUNCTION_PQ
                    : FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    generate.minMaxLuminance[0] = 1.0f;
    generate.minMaxLuminance[1] = 100.0f;
    generate.generationRect = {0, 0, static_cast<int32_t>(displayWidth), static_cast<int32_t>(displayHeight)};
    generate.frameID = frameId;
    g_lastReturn = g_dispatch(&g_context, &generate.header);
    return g_lastReturn;
}

/** Destroy the context. Safe to call when none exists. */
FFX_API void ffxshim_destroy(void) {
    if (g_context != nullptr) {
        g_destroyContext(&g_context, nullptr);
        g_context = nullptr;
    }
}

/** Unload the SDK. Call only after ffxshim_destroy, at device teardown. */
FFX_API void ffxshim_unload(void) {
    ffxshim_destroy();
    if (g_sdkModule != nullptr) {
        FreeLibrary(g_sdkModule);
        g_sdkModule = nullptr;
    }
    g_createContext = nullptr;
    g_destroyContext = nullptr;
    g_configure = nullptr;
    g_query = nullptr;
    g_dispatch = nullptr;
}

/** The most recent non-OK return code, so a failure can name the call that produced it. */
FFX_API int ffxshim_last_result(void) {
    return static_cast<int>(g_lastReturn);
}

/** The SDK's own last diagnostic line, transcoded to UTF-8. Empty when it has not spoken. */
FFX_API const char* ffxshim_last_message(void) {
    return g_lastMessage;
}

/** Version name reported by the provider that was actually selected, e.g. "1.1.3". */
FFX_API const char* ffxshim_version(void) {
    return g_versionName;
}

}  // extern "C"
