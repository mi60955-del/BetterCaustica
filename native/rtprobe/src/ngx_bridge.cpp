#include "ngx_bridge.h"

#include "vk.h"

#include <cstring>
#include <sstream>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// Single source of truth for the probe's NGX access: the shim is *included*, not linked as
// a separate object, so the declarations at the top of ngx_bridge.h are checked against the
// real definitions by the compiler and the build cannot drift.
#include "../../ngx_shim/ngx_shim.cpp"

namespace {

}  // namespace

std::wstring utf8ToWide(const std::string& utf8) {
    if (utf8.empty()) {
        return std::wstring();
    }
    int length = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), int(utf8.size()), nullptr, 0);
    if (length <= 0) {
        length = MultiByteToWideChar(CP_ACP, 0, utf8.c_str(), int(utf8.size()), nullptr, 0);
        if (length <= 0) {
            return std::wstring();
        }
        std::wstring out(size_t(length), L'\0');
        MultiByteToWideChar(CP_ACP, 0, utf8.c_str(), int(utf8.size()), out.data(), length);
        return out;
    }
    std::wstring out(size_t(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), int(utf8.size()), out.data(), length);
    return out;
}

std::string ngxResultToString(int result) {
    if (!ngxFailed(result)) {
        std::ostringstream out;
        out << "success (0x" << std::hex << static_cast<unsigned>(result) << ")";
        return out.str();
    }

    const char* name = "unknown";
    switch (static_cast<unsigned>(result) & 0xFFFFu) {
        case 1:  name = "FAIL_FeatureNotSupported"; break;
        case 2:  name = "FAIL_PlatformError"; break;
        case 3:  name = "FAIL_FeatureAlreadyExists"; break;
        case 4:  name = "FAIL_FeatureNotFound"; break;
        case 5:  name = "FAIL_InvalidParameter"; break;
        case 6:  name = "FAIL_ScratchBufferTooSmall"; break;
        case 7:  name = "FAIL_NotInitialized"; break;
        case 8:  name = "FAIL_UnsupportedInputFormat"; break;
        case 9:  name = "FAIL_RWFlagMissing"; break;
        case 10: name = "FAIL_MissingInput"; break;
        case 11: name = "FAIL_UnableToInitializeFeature"; break;
        case 12: name = "FAIL_OutOfDate"; break;
        case 13: name = "FAIL_OutOfGPUMemory"; break;
        case 14: name = "FAIL_UnsupportedFormat"; break;
        case 15: name = "FAIL_UnableToWriteToAppDataPath"; break;
        case 16: name = "FAIL_UnsupportedParameter"; break;
        case 17: name = "FAIL_Denied"; break;
        case 18: name = "FAIL_NotImplemented"; break;
        default: break;
    }
    std::ostringstream out;
    out << name << " (0x" << std::hex << static_cast<unsigned>(result) << ")";
    return out.str();
}

void loadNgxApi(NgxApi& api, const std::string& shimDllPath) {
    if (shimDllPath.empty()) {
        api.requiredExtensions = &ngxshim_required_extensions;
        api.init = &ngxshim_init;
        api.lastResult = &ngxshim_last_result;
        api.dlssAvailable = &ngxshim_dlss_available;
        api.dlssdAvailable = &ngxshim_dlssd_available;
        api.queryOptimalDlssd = &ngxshim_query_optimal_dlssd;
        api.createDlssd = &ngxshim_create_dlssd;
        api.evaluateDlssd = &ngxshim_evaluate_dlssd;
        api.release = &ngxshim_release;
        api.shutdown = &ngxshim_shutdown;
        api.source = "statically linked (native/ngx_shim/ngx_shim.cpp)";
        return;
    }

    std::wstring wide = utf8ToWide(shimDllPath);
    HMODULE module = LoadLibraryW(wide.c_str());
    if (!module) {
        std::ostringstream out;
        out << "LoadLibraryW failed (GetLastError=" << GetLastError() << ") for " << shimDllPath;
        fail(out.str());
    }
    api.module = module;

    auto resolve = [&](const char* name) -> FARPROC {
        FARPROC address = GetProcAddress(module, name);
        if (!address) {
            std::ostringstream out;
            out << "shim DLL " << shimDllPath << " does not export " << name;
            fail(out.str());
        }
        return address;
    };

    api.requiredExtensions = reinterpret_cast<decltype(api.requiredExtensions)>(resolve("ngxshim_required_extensions"));
    api.init = reinterpret_cast<decltype(api.init)>(resolve("ngxshim_init"));
    api.lastResult = reinterpret_cast<decltype(api.lastResult)>(resolve("ngxshim_last_result"));
    api.dlssAvailable = reinterpret_cast<decltype(api.dlssAvailable)>(resolve("ngxshim_dlss_available"));
    api.dlssdAvailable = reinterpret_cast<decltype(api.dlssdAvailable)>(resolve("ngxshim_dlssd_available"));
    api.queryOptimalDlssd = reinterpret_cast<decltype(api.queryOptimalDlssd)>(resolve("ngxshim_query_optimal_dlssd"));
    api.createDlssd = reinterpret_cast<decltype(api.createDlssd)>(resolve("ngxshim_create_dlssd"));
    api.evaluateDlssd = reinterpret_cast<decltype(api.evaluateDlssd)>(resolve("ngxshim_evaluate_dlssd"));
    api.release = reinterpret_cast<decltype(api.release)>(resolve("ngxshim_release"));
    api.shutdown = reinterpret_cast<decltype(api.shutdown)>(resolve("ngxshim_shutdown"));
    api.source = shimDllPath;
}

void unloadNgxApi(NgxApi& api) {
    if (api.module) {
        FreeLibrary(static_cast<HMODULE>(api.module));
        api.module = nullptr;
    }
    api = NgxApi{};
}
