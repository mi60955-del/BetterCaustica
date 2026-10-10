// Access to the Caustica NGX shim from the probe.
//
// The shim (native/ngx_shim/ngx_shim.cpp) is compiled straight into the probe rather than
// duplicated, so the probe exercises the identical NGX call sequence the mod uses -- the
// declarations here exist to be conflict-checked against those definitions at compile time.
// Passing --shim-dll switches to loading a deployed ngxshim.dll instead, which is how the
// shipped binary itself gets A/B tested.

#pragma once

#include <vulkan/vulkan.h>

#include <string>

#if defined(_WIN32)
#define NGX_SHIM_EXPORT __declspec(dllexport)
#else
#define NGX_SHIM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

NGX_SHIM_EXPORT int ngxshim_last_result();
NGX_SHIM_EXPORT int ngxshim_required_extensions(int wantDevice, char* outBuf, int bufLen);
NGX_SHIM_EXPORT int ngxshim_init(unsigned long long appId, const wchar_t* dataPath,
                                 VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                                 void* getInstanceProcAddr, void* getDeviceProcAddr,
                                 const wchar_t* featureDllPath);
NGX_SHIM_EXPORT int ngxshim_dlss_available();
NGX_SHIM_EXPORT int ngxshim_dlssd_available();
NGX_SHIM_EXPORT int ngxshim_query_optimal_dlssd(unsigned int displayWidth, unsigned int displayHeight,
                                                int quality, unsigned int* outRenderWidth,
                                                unsigned int* outRenderHeight, float* outSharpness);
NGX_SHIM_EXPORT void* ngxshim_create_dlssd(VkCommandBuffer cmd,
                                           unsigned int renderWidth, unsigned int renderHeight,
                                           unsigned int displayWidth, unsigned int displayHeight,
                                           int quality, int featureFlags, int renderPreset);
NGX_SHIM_EXPORT int ngxshim_evaluate_dlssd(VkCommandBuffer cmd, void* feature,
                                           VkImageView colorView, VkImage colorImage, int colorFormat,
                                           VkImageView depthView, VkImage depthImage, int depthFormat,
                                           VkImageView mvView, VkImage mvImage, int mvFormat,
                                           VkImageView diffuseAlbedoView, VkImage diffuseAlbedoImage, int diffuseAlbedoFormat,
                                           VkImageView specularAlbedoView, VkImage specularAlbedoImage, int specularAlbedoFormat,
                                           VkImageView normalsView, VkImage normalsImage, int normalsFormat,
                                           VkImageView specularMotionView, VkImage specularMotionImage, int specularMotionFormat,
                                           VkImageView specularHitDistanceView, VkImage specularHitDistanceImage, int specularHitDistanceFormat,
                                           VkImageView outputView, VkImage outputImage, int outputFormat,
                                           unsigned int renderWidth, unsigned int renderHeight,
                                           unsigned int displayWidth, unsigned int displayHeight,
                                           float jitterX, float jitterY, float mvScaleX, float mvScaleY,
                                           int reset, float frameTimeMs,
                                           float* worldToViewMatrix, float* viewToClipMatrix);
NGX_SHIM_EXPORT void ngxshim_release(void* feature);
NGX_SHIM_EXPORT void ngxshim_shutdown(VkDevice device);

}  // extern "C"

struct NgxApi {
    int (*requiredExtensions)(int wantDevice, char* outBuf, int bufLen) = nullptr;
    int (*init)(unsigned long long appId, const wchar_t* dataPath,
                VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                void* getInstanceProcAddr, void* getDeviceProcAddr,
                const wchar_t* featureDllPath) = nullptr;
    int (*lastResult)() = nullptr;
    int (*dlssAvailable)() = nullptr;
    int (*dlssdAvailable)() = nullptr;
    int (*queryOptimalDlssd)(unsigned int, unsigned int, int,
                             unsigned int*, unsigned int*, float*) = nullptr;
    void* (*createDlssd)(VkCommandBuffer, unsigned int, unsigned int, unsigned int, unsigned int,
                         int, int, int) = nullptr;
    int (*evaluateDlssd)(VkCommandBuffer, void*,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         VkImageView, VkImage, int,
                         unsigned int, unsigned int, unsigned int, unsigned int,
                         float, float, float, float, int, float,
                         float*, float*) = nullptr;
    void (*release)(void*) = nullptr;
    void (*shutdown)(VkDevice) = nullptr;

    std::string source;
    void* module = nullptr;  // HMODULE when loaded from a DLL, else null
};

// `shimDllPath` empty selects the statically linked shim.
void loadNgxApi(NgxApi& api, const std::string& shimDllPath);
void unloadNgxApi(NgxApi& api);

// UTF-8 to UTF-16, which is what the shim's wchar_t paths want.
std::wstring utf8ToWide(const std::string& text);

// NGX signals success with 0x1 and failure with 0xBAD00000 | code, so a plain "!= 0" test
// would call every success a failure.
inline bool ngxFailed(int result) {
    return (static_cast<unsigned>(result) & 0xFFF00000u) == 0xBAD00000u;
}

std::string ngxResultToString(int result);
