// Runtime load of amd_fidelityfx_vk.dll and the two pieces of glue the unified API needs
// from the application: a Vulkan entry-point lookup and a message callback.
//
// Loaded by path rather than linked, because the SDK ships no import library on the path this
// project keeps its spike on, and because the mod will resolve the same five symbols through
// Panama FFM. The five functions are the whole public surface; everything else is reached
// through descriptor structs passed by pointer.

#pragma once

#include <ffx_api/ffx_api.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <string>
#include <vector>

struct FfxHost {
    void* module = nullptr;
    PfnFfxCreateContext createContext = nullptr;
    PfnFfxDestroyContext destroyContext = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;

    // Names the SDK asked the device for and the device-level lookup could not answer. The
    // shim substitutes the instance-level pointer for these, so a non-empty list is a record
    // of what would have been a null call had the fallback not existed.
    std::vector<std::string> deviceLookupMisses;
    // Every name in the order it was requested, suffixed with '!' when only the instance
    // fallback could answer. Printed when the SDK faults, because a call through a null
    // function pointer is otherwise indistinguishable from any other crash in the DLL.
    std::vector<std::string> shimLog;
    bool logShim = false;
    uint32_t deviceLookups = 0;
};

bool ffxLoad(FfxHost& host, const std::string& dllPath, std::string& error);
void ffxUnload(FfxHost& host);

// The host the shim reports to, so a crash handler can print what the shim saw last.
FfxHost* ffxActiveHost();

// Binds the host the shim reports to, and the instance the fallback resolves against. Both
// have to outlive every frame-generation context, which they do: the host owns the DLL and
// the probe destroys the instance after the last context.
void ffxBindShim(FfxHost& host, VkInstance instance);

// Turning the fallback off makes the shim behave exactly like the plain device query, which is
// what distinguishes "the callback handed the SDK a bad pointer" from "the SDK called a
// pointer it never got".
void ffxSetShimFallback(bool enabled);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ffxDeviceProcAddrShim(VkDevice device, const char* name);

// Installs the SDK's own diagnostic callback at the given level. Pass
// FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE during bring-up.
ffxReturnCode_t ffxEnableMessages(uint32_t debugLevel);

// --- guarded allocator -----------------------------------------------------------------
//
// ffxCreateContext takes allocation callbacks, so the probe can hand the SDK memory whose tail
// is poisoned. If the SDK writes past a block, the poison says how far - which is a readable
// result, where a corrupted function pointer called three statements later is not.
constexpr uint64_t kGuardBytes = 4096;

struct GuardedAllocation {
    void*    ptr = nullptr;
    uint64_t size = 0;
    // Bytes of the tail that no longer hold the poison pattern.
    uint64_t overwritten = 0;
};

// Callbacks wired to the registry below. Pass this as ffxCreateContext's third argument.
const ffxAllocationCallbacks* ffxGuardedCallbacks();

// Prints every block whose tail was written through. Returns the number of offenders.
uint32_t ffxReportGuardViolations();

const char* ffxReturnCodeName(ffxReturnCode_t code);
