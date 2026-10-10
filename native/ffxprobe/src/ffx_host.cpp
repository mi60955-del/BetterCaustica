#include "ffx_host.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace {

FfxHost* g_host = nullptr;
VkInstance g_shimInstance = VK_NULL_HANDLE;
bool g_shimFallback = true;

void wideToAscii(const wchar_t* input, char* out, size_t outSize) {
    out[0] = '\0';
    if (!input) {
        return;
    }
    // '?' as the default char keeps the line ASCII on a CP936 console, where raw UTF-8 bytes
    // from a non-ASCII message would render as mojibake.
    WideCharToMultiByte(CP_UTF8, 0, input, -1, out, static_cast<int>(outSize), "?", nullptr);
}

void VKAPI_CALL ffxMessageCallback(uint32_t type, const wchar_t* message) {
    char buffer[512];
    wideToAscii(message, buffer, sizeof(buffer));
    std::fprintf(stderr, "[ffx %s] %s\n",
                 type == FFX_API_MESSAGE_TYPE_ERROR ? "ERROR" : "warn", buffer);
    std::fflush(stderr);
}

void* loadSymbol(void* module, const char* name) {
    return reinterpret_cast<void*>(
            GetProcAddress(static_cast<HMODULE>(module), name));
}

std::vector<GuardedAllocation> g_allocations;

void* guardedAlloc(void* /*userData*/, uint64_t size) {
    if (size == 0) {
        return nullptr;
    }
    unsigned char* block = static_cast<unsigned char*>(malloc(static_cast<size_t>(size) + kGuardBytes));
    if (!block) {
        return nullptr;
    }
    memset(block + size, 0xA5, kGuardBytes);
    GuardedAllocation entry;
    entry.ptr = block;
    entry.size = size;
    g_allocations.push_back(entry);
    return block;
}

void guardedFree(void* /*userData*/, void* mem) {
    if (!mem) {
        return;
    }
    for (size_t i = 0; i < g_allocations.size(); ++i) {
        if (g_allocations[i].ptr == mem) {
            g_allocations.erase(g_allocations.begin() + static_cast<long>(i));
            break;
        }
    }
    free(mem);
}

ffxAllocationCallbacks g_guardedCallbacks = {nullptr, guardedAlloc, guardedFree};

}  // namespace

bool ffxLoad(FfxHost& host, const std::string& dllPath, std::string& error) {
    // LOAD_WITH_ALTERED_SEARCH_PATH makes the DLL's own directory the starting point for its
    // imports (vulkan-1.dll, VCRUNTIME140.dll), which is what a side-by-side deployment needs.
    HMODULE module = LoadLibraryExA(dllPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        std::ostringstream out;
        out << "LoadLibraryExA(" << dllPath << ") failed with error " << GetLastError();
        error = out.str();
        return false;
    }
    host.module = module;
    host.createContext = reinterpret_cast<PfnFfxCreateContext>(loadSymbol(module, "ffxCreateContext"));
    host.destroyContext = reinterpret_cast<PfnFfxDestroyContext>(loadSymbol(module, "ffxDestroyContext"));
    host.configure = reinterpret_cast<PfnFfxConfigure>(loadSymbol(module, "ffxConfigure"));
    host.query = reinterpret_cast<PfnFfxQuery>(loadSymbol(module, "ffxQuery"));
    host.dispatch = reinterpret_cast<PfnFfxDispatch>(loadSymbol(module, "ffxDispatch"));

    if (!host.createContext || !host.destroyContext || !host.configure || !host.query || !host.dispatch) {
        error = "amd_fidelityfx_vk.dll is missing one of the five unified-API exports";
        FreeLibrary(module);
        host = FfxHost{};
        return false;
    }
    return true;
}

void ffxUnload(FfxHost& host) {
    if (host.module) {
        FreeLibrary(static_cast<HMODULE>(host.module));
        host.module = nullptr;
    }
    host.createContext = nullptr;
    host.destroyContext = nullptr;
    host.configure = nullptr;
    host.query = nullptr;
    host.dispatch = nullptr;
    if (g_host == &host) {
        g_host = nullptr;
        g_shimInstance = VK_NULL_HANDLE;
    }
}

void ffxBindShim(FfxHost& host, VkInstance instance) {
    g_host = &host;
    g_shimInstance = instance;
}

FfxHost* ffxActiveHost() {
    return g_host;
}

void ffxSetShimFallback(bool enabled) {
    g_shimFallback = enabled;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ffxDeviceProcAddrShim(VkDevice device, const char* name) {
    if (g_host) {
        ++g_host->deviceLookups;
    }
    void* address = reinterpret_cast<void*>(vkGetDeviceProcAddr(device, name));
    const bool fromDevice = address != nullptr;
    if (!fromDevice && g_host) {
        // Recorded whether or not the fallback is on: this list is the diagnosis, not the
        // remedy. vkGetDeviceProcAddr must return NULL for instance-level commands.
        g_host->deviceLookupMisses.emplace_back(name);
    }
    if (!fromDevice && g_shimFallback && g_shimInstance != VK_NULL_HANDLE) {
        address = reinterpret_cast<void*>(vkGetInstanceProcAddr(g_shimInstance, name));
    }
    if (g_host && g_host->logShim) {
        g_host->shimLog.emplace_back(fromDevice ? name : std::string(name) + "!");
    }
    return reinterpret_cast<PFN_vkVoidFunction>(address);
}

ffxReturnCode_t ffxEnableMessages(uint32_t debugLevel) {
    if (!g_host || !g_host->configure) {
        return FFX_API_RETURN_ERROR;
    }
    ffxConfigureDescGlobalDebug1 desc{};
    desc.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
    desc.header.pNext = nullptr;
    desc.fpMessage = ffxMessageCallback;
    desc.debugLevel = debugLevel;
    // A null context means "global state", which is where the message sink lives.
    return g_host->configure(nullptr, &desc.header);
}

const char* ffxReturnCodeName(ffxReturnCode_t code) {
    switch (code) {
        case FFX_API_RETURN_OK: return "OK";
        case FFX_API_RETURN_ERROR: return "ERROR";
        case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "ERROR_UNKNOWN_DESCTYPE";
        case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "ERROR_RUNTIME_ERROR";
        case FFX_API_RETURN_NO_PROVIDER: return "NO_PROVIDER";
        case FFX_API_RETURN_ERROR_MEMORY: return "ERROR_MEMORY";
        case FFX_API_RETURN_ERROR_PARAMETER: return "ERROR_PARAMETER";
        default: return "UNKNOWN";
    }
}

const ffxAllocationCallbacks* ffxGuardedCallbacks() {
    return &g_guardedCallbacks;
}

uint32_t ffxReportGuardViolations() {
    uint32_t offenders = 0;
    // The registry itself is the first thing to know about: if it is empty the SDK never
    // called back, and then no amount of guard reading says anything about its writes.
    // Blocks the SDK released without going through dealloc stay listed here with the memory
    // already returned to the OS, so each one is checked for being committed before its tail
    // is read.
    std::fprintf(stderr, "[guard] SDK allocations not returned through dealloc: %u\n",
                 static_cast<unsigned>(g_allocations.size()));
    for (const GuardedAllocation& entry : g_allocations) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(entry.ptr, &info, sizeof(info)) || info.State != MEM_COMMIT) {
            std::fprintf(stderr, "[guard]   block %p size %llu is no longer committed\n",
                         entry.ptr, static_cast<unsigned long long>(entry.size));
            continue;
        }
        const unsigned char* tail = static_cast<const unsigned char*>(entry.ptr) + entry.size;
        uint64_t intact = 0;
        while (intact < kGuardBytes && tail[intact] == 0xA5) {
            ++intact;
        }
        std::fprintf(stderr, "[guard]   block %p size %llu%s\n", entry.ptr,
                     static_cast<unsigned long long>(entry.size),
                     intact == kGuardBytes ? "" : "  <- WRITTEN PAST");
        if (intact == kGuardBytes) {
            continue;
        }
        ++offenders;
        std::fprintf(stderr,
                     "[guard]     first damaged byte at +%llu, i.e. %llu of the %llu-byte tail"
                     " was overwritten\n",
                     static_cast<unsigned long long>(intact),
                     static_cast<unsigned long long>(kGuardBytes - intact),
                     static_cast<unsigned long long>(kGuardBytes));
    }
    return offenders;
}
