// rtprobe -- a fast, offscreen validator for the RT + DLSS Ray Reconstruction pipeline.
//
// It exists because validating the game's ray tracer means booting Minecraft, walking into a
// world, and reading a log. This runs the same NGX call sequence against a synthetic scene in
// seconds, and reports numbers instead of impressions:
//
//   * which DLSS-RR DLL was actually loaded (path, size, FileVersion, SHA-256)
//   * whether NGX reports Ray Reconstruction as available, and every VkResult in the chain
//   * the render -> display resolution NGX picks for the requested quality
//   * how much high-frequency energy the denoiser removed
//   * how much the output still changes between two otherwise-identical frames, which is the
//     signal that RR's temporal accumulation is actually doing something
//
// Guide buffers follow Caustica's contract exactly (see shaders/probe.comp), so a result here
// transfers to the mod.

#include "vk.h"
#include "ngx_bridge.h"
#include "image_io.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// --------------------------------------------------------------------------------------
// small math
// --------------------------------------------------------------------------------------

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float lengthOf(Vec3 v) { return std::sqrt(dot(v, v)); }
Vec3 normalize(Vec3 v) {
    float len = lengthOf(v);
    return len > 1.0e-12f ? v * (1.0f / len) : Vec3{};
}

// Row-major 4x4. This is also exactly the layout NGX wants, so the same array is handed to
// both the compute shader (via std140 row reads) and to ngxshim_evaluate_dlssd.
struct Mat4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 out;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[i * 4 + k] * b.m[k * 4 + j];
            }
            out.m[i * 4 + j] = sum;
        }
    }
    return out;
}

Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 upRef) {
    Vec3 z = normalize(eye - target);
    Vec3 x = normalize(cross(upRef, z));
    Vec3 y = cross(z, x);
    Mat4 out;
    out.m[0] = x.x; out.m[1] = x.y; out.m[2] = x.z; out.m[3] = -dot(x, eye);
    out.m[4] = y.x; out.m[5] = y.y; out.m[6] = y.z; out.m[7] = -dot(y, eye);
    out.m[8] = z.x; out.m[9] = z.y; out.m[10] = z.z; out.m[11] = -dot(z, eye);
    out.m[12] = 0.0f; out.m[13] = 0.0f; out.m[14] = 0.0f; out.m[15] = 1.0f;
    return out;
}

// Reversed-Z: clip.z = near (constant), clip.w = -z_view, so ndc.z = near / -z_view gives 1 at
// the near plane and decays toward 0 with distance. That is why DLSS-RR gets DepthInverted.
Mat4 reversedZProjection(float fovY, float aspect, float nearPlane) {
    float f = 1.0f / std::tan(fovY * 0.5f);
    Mat4 out;
    out.m[0] = f / aspect; out.m[1] = 0.0f; out.m[2] = 0.0f;  out.m[3] = 0.0f;
    out.m[4] = 0.0f;       out.m[5] = f;    out.m[6] = 0.0f;  out.m[7] = 0.0f;
    out.m[8] = 0.0f;       out.m[9] = 0.0f; out.m[10] = 0.0f; out.m[11] = nearPlane;
    out.m[12] = 0.0f;      out.m[13] = 0.0f; out.m[14] = -1.0f; out.m[15] = 0.0f;
    return out;
}

// Halton(2,3) minus 0.5, matching CausticaJitter so the probe's jitter statistics are the
// same as the game's.
float halton(int index, int base) {
    float f = 1.0f;
    float result = 0.0f;
    int i = index;
    while (i > 0) {
        f /= float(base);
        result += f * float(i % base);
        i /= base;
    }
    return result;
}

int jitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth) {
    float ratio = float(displayWidth) / float(std::max(1u, renderWidth));
    return std::max(32, int(std::ceil(8.0f * ratio * ratio)));
}

// --------------------------------------------------------------------------------------
// options
// --------------------------------------------------------------------------------------

struct Options {
    uint32_t displayWidth = 1920;
    uint32_t displayHeight = 1080;
    uint32_t renderWidth = 0;   // 0 -> ask NGX for the quality's optimal size
    uint32_t renderHeight = 0;
    int quality = 2;            // 2 = Quality (3 UP, 0 Perf, 1 Balanced, 2 Quality, 5 DLAA)
    int preset = 6;             // 6 = Preset F, the DLSS 4.5 ray reconstruction network
    int frames = 48;
    int hold = 8;
    int gpuIndex = -1;
    bool listGpus = false;
    bool validation = false;
    bool dumpGuides = false;
    bool noRr = false;
    bool quiet = false;
    std::string motion = "orbit";
    std::string outDir = "rtprobe-out";
    std::string dllDir = "E:/Minecraft/26.2/caustica-ngx/natives/windows-x64";
    std::string shimDll;
    std::string shader;
    float exposure = 1.0f;
    float key = 0.18f;
    float fovY = 45.0f;
    float nearPlane = 0.1f;
    float sunIntensity = 1.0f;
};

void printUsage() {
    std::printf(
        "rtprobe -- offscreen RT + DLSS Ray Reconstruction validator\n"
        "\n"
        "  --display WxH     output (display) resolution          [1920x1080]\n"
        "  --render WxH      force the internal render resolution [NGX optimal for --quality]\n"
        "  --quality N       3=UltraPerf 0=Perf 1=Balanced 2=Quality 5=DLAA   [2]\n"
        "  --preset N        RR render preset, 6 = Preset F (DLSS 4.5)         [6]\n"
        "  --frames N        frames to trace                                    [48]\n"
        "  --hold N          final frames with a frozen camera pose (temporal test) [8]\n"
        "  --motion MODE     orbit | static                                     [orbit]\n"
        "  --out DIR         output directory                                   [rtprobe-out]\n"
        "  --dll-dir PATH    directory holding nvngx_dlssd.dll (what NGX loads)\n"
        "  --shim-dll PATH   load this ngxshim.dll instead of the linked shim\n"
        "  --shader PATH     explicit probe.comp.spv\n"
        "  --dump-guides     also write the guide buffers as PNG\n"
        "  --no-rr           skip NGX entirely (baseline trace only)\n"
        "  --gpu N           pick a physical device by index\n"
        "  --list-gpus       list devices and exit\n"
        "  --validation      enable the Khronos validation layer\n"
        "  --exposure F      preview tonemap bias     [1.0]\n"
        "  --quiet           suppress the human-readable summary\n"
        "\n"
        "Set NGXSHIM_VERBOSE=1 in the environment to get the shim's own per-call logging and\n"
        "NGX core logging; it is read at load time, so it cannot be enabled from inside.\n");
}

bool parseSize(const std::string& text, uint32_t& width, uint32_t& height) {
    size_t at = text.find('x');
    if (at == std::string::npos) {
        at = text.find('X');
    }
    if (at == std::string::npos) {
        return false;
    }
    width = uint32_t(std::stoul(text.substr(0, at)));
    height = uint32_t(std::stoul(text.substr(at + 1)));
    return width > 0 && height > 0;
}

Options parseArgs(int argc, char** argv) {
    Options options;
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            fail(std::string("missing value for ") + argv[i]);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else if (arg == "--display") {
            if (!parseSize(next(i), options.displayWidth, options.displayHeight)) {
                fail("--display wants WxH");
            }
        } else if (arg == "--render") {
            if (!parseSize(next(i), options.renderWidth, options.renderHeight)) {
                fail("--render wants WxH");
            }
        } else if (arg == "--quality") {
            options.quality = std::stoi(next(i));
        } else if (arg == "--preset") {
            options.preset = std::stoi(next(i));
        } else if (arg == "--frames") {
            options.frames = std::stoi(next(i));
        } else if (arg == "--hold") {
            options.hold = std::stoi(next(i));
        } else if (arg == "--motion") {
            options.motion = next(i);
        } else if (arg == "--out") {
            options.outDir = next(i);
        } else if (arg == "--dll-dir") {
            options.dllDir = next(i);
        } else if (arg == "--shim-dll") {
            options.shimDll = next(i);
        } else if (arg == "--shader") {
            options.shader = next(i);
        } else if (arg == "--gpu") {
            options.gpuIndex = std::stoi(next(i));
        } else if (arg == "--exposure") {
            options.exposure = std::stof(next(i));
        } else if (arg == "--dump-guides") {
            options.dumpGuides = true;
        } else if (arg == "--no-rr") {
            options.noRr = true;
        } else if (arg == "--list-gpus") {
            options.listGpus = true;
        } else if (arg == "--validation") {
            options.validation = true;
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else {
            fail("unknown argument " + arg + " (try --help)");
        }
    }

    options.frames = std::max(options.frames, 3);
    options.hold = std::max(0, std::min(options.hold, options.frames - 1));
    if (options.motion != "orbit" && options.motion != "static") {
        fail("--motion must be orbit or static");
    }
    return options;
}

// --------------------------------------------------------------------------------------
// scene
// --------------------------------------------------------------------------------------

struct Scene {
    std::vector<float> positions;   // 3 per vertex
    std::vector<float> normals;     // 3 per vertex
    std::vector<float> albedos;     // 3 per vertex
    std::vector<float> params;      // 6 per vertex: roughness, metalness, emissive, emission rgb
    std::vector<uint32_t> indices;
    std::vector<float> emitterPos;  // 4 per emitter: xyz, radius
    std::vector<float> emitterRad;  // 4 per emitter: linear colour, intensity
};

uint32_t addVertex(Scene& scene, Vec3 p, Vec3 n, Vec3 albedo, float roughness, float metalness,
                   Vec3 emission) {
    uint32_t index = uint32_t(scene.positions.size() / 3);
    scene.positions.insert(scene.positions.end(), {p.x, p.y, p.z});
    scene.normals.insert(scene.normals.end(), {n.x, n.y, n.z});
    scene.albedos.insert(scene.albedos.end(), {albedo.x, albedo.y, albedo.z});
    scene.params.insert(scene.params.end(),
                        {roughness, metalness, lengthOf(emission) > 0.0f ? 1.0f : 0.0f,
                         emission.x, emission.y, emission.z});
    return index;
}

void addTriangle(Scene& scene, uint32_t a, uint32_t b, uint32_t c) {
    scene.indices.insert(scene.indices.end(), {a, b, c});
}

void addQuad(Scene& scene, Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 normal, Vec3 albedo,
             float roughness, float metalness) {
    uint32_t ia = addVertex(scene, a, normal, albedo, roughness, metalness, Vec3{});
    uint32_t ib = addVertex(scene, b, normal, albedo, roughness, metalness, Vec3{});
    uint32_t ic = addVertex(scene, c, normal, albedo, roughness, metalness, Vec3{});
    uint32_t id = addVertex(scene, d, normal, albedo, roughness, metalness, Vec3{});
    addTriangle(scene, ia, ib, ic);
    addTriangle(scene, ia, ic, id);
}

void addSphere(Scene& scene, Vec3 centre, float radius, int segments, int rings, Vec3 albedo,
               float roughness, float metalness, Vec3 emission) {
    uint32_t base = uint32_t(scene.positions.size() / 3);
    for (int i = 0; i <= rings; ++i) {
        float theta = kPi * float(i) / float(rings);
        for (int j = 0; j <= segments; ++j) {
            float phi = 2.0f * kPi * float(j) / float(segments);
            Vec3 dir{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
            addVertex(scene, centre + dir * radius, dir, albedo, roughness, metalness, emission);
        }
    }
    int stride = segments + 1;
    for (int i = 0; i < rings; ++i) {
        for (int j = 0; j < segments; ++j) {
            uint32_t v00 = base + uint32_t(i * stride + j);
            uint32_t v01 = v00 + 1;
            uint32_t v10 = base + uint32_t((i + 1) * stride + j);
            uint32_t v11 = v10 + 1;
            if (i != 0) {
                addTriangle(scene, v00, v10, v11);
            }
            if (i != rings - 1) {
                addTriangle(scene, v00, v11, v01);
            }
        }
    }
}

void addEmitter(Scene& scene, Vec3 centre, float radius, Vec3 colour, float intensity) {
    addSphere(scene, centre, radius, 24, 12, Vec3{}, 1.0f, 0.0f, colour * intensity);
    scene.emitterPos.insert(scene.emitterPos.end(), {centre.x, centre.y, centre.z, radius});
    scene.emitterRad.insert(scene.emitterRad.end(), {colour.x, colour.y, colour.z, intensity});
}

// A checkerboard floor (high-frequency detail the denoiser must not smear), a roughness x
// metalness grid of spheres under two coloured emitters, and a sky dome. The roughness ramp
// runs from a near-perfect mirror down to fully rough so every RR path gets exercised.
void buildScene(Scene& scene) {
    const float half = 24.0f;
    const float tile = 1.5f;
    int tiles = int(2.0f * half / tile);
    for (int i = 0; i < tiles; ++i) {
        for (int j = 0; j < tiles; ++j) {
            float x0 = -half + float(i) * tile;
            float z0 = -half + float(j) * tile;
            bool light = ((i + j) & 1) == 0;
            Vec3 albedo = light ? Vec3{0.55f, 0.55f, 0.56f} : Vec3{0.19f, 0.19f, 0.20f};
            float roughness = light ? 0.35f : 0.62f;
            addQuad(scene,
                    {x0, 0.0f, z0},
                    {x0 + tile, 0.0f, z0},
                    {x0 + tile, 0.0f, z0 + tile},
                    {x0, 0.0f, z0 + tile},
                    {0.0f, 1.0f, 0.0f}, albedo, roughness, 0.0f);
        }
    }

    const float roughness[6] = {0.0003f, 0.08f, 0.18f, 0.35f, 0.60f, 0.95f};
    const Vec3 albedo[6] = {{0.92f, 0.92f, 0.94f}, {0.88f, 0.68f, 0.30f}, {0.26f, 0.52f, 0.86f},
                            {0.82f, 0.26f, 0.24f}, {0.30f, 0.70f, 0.42f}, {0.72f, 0.66f, 0.84f}};
    const float metalness[3] = {0.0f, 0.5f, 1.0f};
    const float sphereRadius = 0.62f;
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 3; ++row) {
            Vec3 centre{(float(col) - 2.5f) * 1.55f, sphereRadius, (float(row) - 1.0f) * 1.55f};
            addSphere(scene, centre, sphereRadius, 40, 20, albedo[col], roughness[col],
                      metalness[row], Vec3{});
        }
    }

    addEmitter(scene, {-4.2f, 5.2f, -3.4f}, 1.15f, {1.00f, 0.85f, 0.62f}, 55.0f);
    addEmitter(scene, {4.6f, 4.2f, 1.2f}, 0.85f, {0.62f, 0.78f, 1.00f}, 40.0f);
}

struct CameraPose {
    Vec3 position;
    Vec3 target;
};

CameraPose cameraAt(int frame, const Options& options) {
    float angleDegrees = 18.0f;
    if (options.motion == "orbit") {
        angleDegrees += 0.35f * float(frame);
    }
    float angle = angleDegrees * kPi / 180.0f;
    const float distance = 8.6f;
    const float height = 3.0f;
    return {{std::sin(angle) * distance, height, std::cos(angle) * distance}, {0.0f, 1.05f, 0.0f}};
}

// --------------------------------------------------------------------------------------
// frame uniforms (std140, mirrors the block in probe.comp)
// --------------------------------------------------------------------------------------

struct FrameUniforms {
    float viewProj[16];
    float prevViewProj[16];
    float camPos[4];
    float prevCamPos[4];
    float camRight[4];
    float camUp[4];
    float camFwd[4];
    float jitterRender[4];
    float displayCounts[4];
    float sunDir[4];
    float sunColor[4];
};

// --------------------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------------------

std::string exeDirectory() {
    wchar_t buffer[MAX_PATH * 4] = {};
    DWORD length = GetModuleFileNameW(nullptr, buffer, DWORD(std::size(buffer)));
    if (length == 0) {
        return ".";
    }
    std::wstring path(buffer, length);
    size_t slash = path.find_last_of(L"\\/");
    std::wstring directory = slash == std::wstring::npos ? L"." : path.substr(0, slash);
    std::string utf8;
    int needed = WideCharToMultiByte(CP_UTF8, 0, directory.c_str(), int(directory.size()),
                                     nullptr, 0, nullptr, nullptr);
    utf8.resize(size_t(std::max(0, needed)));
    if (needed > 0) {
        WideCharToMultiByte(CP_UTF8, 0, directory.c_str(), int(directory.size()), utf8.data(),
                            needed, nullptr, nullptr);
    }
    return utf8;
}

std::string resolveShaderPath(const Options& options) {
    if (!options.shader.empty()) {
        if (!std::filesystem::exists(options.shader)) {
            fail("--shader path does not exist: " + options.shader);
        }
        return options.shader;
    }
    std::vector<std::string> candidates{
        (std::filesystem::path(exeDirectory()) / "probe.comp.spv").string(),
        "probe.comp.spv",
        (std::filesystem::path(exeDirectory()) / ".." / "probe.comp.spv").string(),
        (std::filesystem::path(exeDirectory()) / ".." / ".." / "probe.comp.spv").string(),
    };
    for (const std::string& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    fail("cannot find probe.comp.spv; pass --shader");
}

std::string driverVersionString(uint32_t version) {
    std::ostringstream out;
    out << ((version >> 22) & 0x3FFu) << "." << ((version >> 14) & 0xFFu) << "."
        << ((version >> 6) & 0xFFu) << "." << (version & 0x3Fu);
    return out.str();
}

std::string apiVersionString(uint32_t version) {
    std::ostringstream out;
    out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
        << VK_VERSION_PATCH(version);
    return out.str();
}

std::string hexOf(const unsigned char* bytes, size_t count) {
    static const char* digits = "0123456789ABCDEF";
    std::string out;
    out.reserve(count * 2);
    for (size_t i = 0; i < count; ++i) {
        out.push_back(digits[bytes[i] >> 4]);
        out.push_back(digits[bytes[i] & 0xF]);
    }
    return out;
}

std::string hashOfFile(const std::string& path, LPCWSTR algorithmId) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    if (BCryptOpenAlgorithmProvider(&algorithm, algorithmId, nullptr, 0) < 0) {
        return "";
    }
    DWORD hashLength = 0, produced = 0;
    BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hashLength),
                      sizeof(hashLength), &produced, 0);

    BCRYPT_HASH_HANDLE hash = nullptr;
    std::vector<unsigned char> digest(hashLength);
    std::string result;
    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0) {
        std::ifstream file(path, std::ios::binary);
        if (file) {
            std::vector<char> buffer(1 << 20);
            while (file) {
                file.read(buffer.data(), std::streamsize(buffer.size()));
                std::streamsize got = file.gcount();
                if (got > 0) {
                    BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()), ULONG(got), 0);
                }
            }
            if (BCryptFinishHash(hash, digest.data(), hashLength, 0) >= 0) {
                result = hexOf(digest.data(), digest.size());
            }
        }
        BCryptDestroyHash(hash);
    }
    BCryptCloseAlgorithmProvider(algorithm, 0);
    return result;
}

std::string sha256OfFile(const std::string& path) {
    return hashOfFile(path, BCRYPT_SHA256_ALGORITHM);
}

// Recorded for every run so an A/B of two runs reduces to comparing one line, instead of
// eyeballing two images that may differ so subtly that neither is obviously "better".
std::string md5OfFile(const std::string& path) {
    return hashOfFile(path, BCRYPT_MD5_ALGORITHM);
}

struct DllFacts {
    std::string path;
    std::string fileVersion;
    std::string sha256;
    uint64_t size = 0;
    bool exists = false;
};

DllFacts inspectDll(const std::string& directory) {
    DllFacts facts;
    std::filesystem::path path = std::filesystem::path(directory) / "nvngx_dlssd.dll";
    facts.path = path.string();
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        return facts;
    }
    facts.exists = true;
    facts.size = std::filesystem::file_size(path, error);
    facts.sha256 = sha256OfFile(facts.path);

    DWORD handle = 0;
    DWORD bytes = GetFileVersionInfoSizeW(path.wstring().c_str(), &handle);
    if (bytes) {
        std::vector<char> data(bytes);
        if (GetFileVersionInfoW(path.wstring().c_str(), 0, bytes, data.data())) {
            VS_FIXEDFILEINFO* info = nullptr;
            UINT length = 0;
            if (VerQueryValueW(data.data(), L"\\", reinterpret_cast<LPVOID*>(&info), &length)
                    && info) {
                std::ostringstream out;
                out << HIWORD(info->dwFileVersionMS) << "." << LOWORD(info->dwFileVersionMS) << "."
                    << HIWORD(info->dwFileVersionLS) << "." << LOWORD(info->dwFileVersionLS);
                facts.fileVersion = out.str();
            }
        }
    }
    return facts;
}

// --------------------------------------------------------------------------------------
// readback
// --------------------------------------------------------------------------------------

VkDeviceSize bytesPerPixel(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R32_SFLOAT: return 4;
        case VK_FORMAT_R16G16_SFLOAT: return 4;
        default: return 0;
    }
}

FloatImage readFloatImage(const Probe& probe, const Image& image) {
    FloatImage out(image.width, image.height);
    VkDeviceSize bytes = VkDeviceSize(image.width) * image.height * bytesPerPixel(image.format);
    if (bytes == 0) {
        fail("readFloatImage: unsupported format");
    }

    Buffer staging = createBuffer(probe, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                         | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 false);
    VkCommandBuffer commandBuffer = beginCommands(probe);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {image.width, image.height, 1};
    vkCmdCopyImageToBuffer(commandBuffer, image.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1,
                           &region);
    submitCommands(probe, commandBuffer);

    const unsigned char* raw = static_cast<const unsigned char*>(staging.mapped);
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            size_t pixel = size_t(y) * image.width + x;
            float* dst = out.pixel(x, y);
            if (image.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
                for (int c = 0; c < 3; ++c) {
                    uint16_t half = 0;
                    std::memcpy(&half, raw + pixel * 8 + size_t(c) * 2, sizeof(half));
                    dst[c] = halfToFloat(half);
                }
            } else if (image.format == VK_FORMAT_R32_SFLOAT) {
                float value = 0.0f;
                std::memcpy(&value, raw + pixel * 4, sizeof(value));
                dst[0] = dst[1] = dst[2] = value;
            } else {
                for (int c = 0; c < 2; ++c) {
                    uint16_t half = 0;
                    std::memcpy(&half, raw + pixel * 4 + size_t(c) * 2, sizeof(half));
                    dst[c] = halfToFloat(half);
                }
                dst[2] = 0.0f;
            }
        }
    }
    destroyBuffer(probe, staging);
    return out;
}

// --------------------------------------------------------------------------------------
// report
// --------------------------------------------------------------------------------------

std::string jsonEscape(const std::string& text) {
    std::string out;
    for (char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

double roundTo(double value, int digits) {
    double scale = std::pow(10.0, digits);
    return std::round(value * scale) / scale;
}

// --------------------------------------------------------------------------------------
// acceleration structures
// --------------------------------------------------------------------------------------

struct Accel {
    Buffer blasBuffer, tlasBuffer, scratch, vertexBuffer, indexBuffer, instanceBuffer;
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    uint32_t triangleCount = 0;
    uint64_t blasBytes = 0;
    uint64_t tlasBytes = 0;
};

void buildAcceleration(Probe& probe, const Scene& scene, Accel& accel) {
    accel.triangleCount = uint32_t(scene.indices.size() / 3);

    accel.vertexBuffer = createBuffer(probe, scene.positions.size() * sizeof(float),
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                              | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                              | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                              | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      true);
    uploadBuffer(probe, accel.vertexBuffer, scene.positions.data(),
                 scene.positions.size() * sizeof(float));

    accel.indexBuffer = createBuffer(probe, scene.indices.size() * sizeof(uint32_t),
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                             | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     true);
    uploadBuffer(probe, accel.indexBuffer, scene.indices.data(),
                 scene.indices.size() * sizeof(uint32_t));

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = accel.vertexBuffer.address;
    geometry.geometry.triangles.vertexStride = 12;
    geometry.geometry.triangles.maxVertex = uint32_t(scene.positions.size() / 3 - 1);
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress = accel.indexBuffer.address;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t primitiveCount = accel.triangleCount;
    VkAccelerationStructureBuildSizesInfoKHR sizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    probe.getAccelerationStructureBuildSizes(probe.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                             &buildInfo, &primitiveCount, &sizes);
    accel.blasBytes = sizes.accelerationStructureSize;

    accel.blasBuffer = createBuffer(probe, sizes.accelerationStructureSize,
                                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    VkAccelerationStructureCreateInfoKHR createInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = accel.blasBuffer.buffer;
    createInfo.size = sizes.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (probe.createAccelerationStructure(probe.device, &createInfo, nullptr, &accel.blas) != VK_SUCCESS) {
        fail("failed to create the bottom-level acceleration structure");
    }

    // The scratch offset has a device-specific alignment requirement, so over-allocate and
    // round the device address up rather than assuming the buffer's own alignment is enough.
    VkDeviceSize scratchSize = sizes.buildScratchSize + probe.scratchAlignment;
    accel.scratch = createBuffer(probe, scratchSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                         | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = accel.blas;
    VkDeviceAddress blasAddress = probe.getAccelerationStructureDeviceAddress(probe.device, &addressInfo);

    // One instance, identity transform, facing culling disabled so winding never matters.
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blasAddress;

    accel.instanceBuffer = createBuffer(probe, sizeof(instance),
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       true);
    uploadBuffer(probe, accel.instanceBuffer, &instance, sizeof(instance));

    VkAccelerationStructureGeometryKHR tlasGeometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeometry.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeometry.geometry.instances.data.deviceAddress = accel.instanceBuffer.address;

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuild.geometryCount = 1;
    tlasBuild.pGeometries = &tlasGeometry;

    uint32_t instanceCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    probe.getAccelerationStructureBuildSizes(probe.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                             &tlasBuild, &instanceCount, &tlasSizes);
    accel.tlasBytes = tlasSizes.accelerationStructureSize;

    accel.tlasBuffer = createBuffer(probe, tlasSizes.accelerationStructureSize,
                                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    VkAccelerationStructureCreateInfoKHR tlasCreate{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    tlasCreate.buffer = accel.tlasBuffer.buffer;
    tlasCreate.size = tlasSizes.accelerationStructureSize;
    tlasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (probe.createAccelerationStructure(probe.device, &tlasCreate, nullptr, &accel.tlas) != VK_SUCCESS) {
        fail("failed to create the top-level acceleration structure");
    }

    // Both builds need scratch, and the second must not start before the first finishes.
    VkDeviceSize scratchForBlas = sizes.buildScratchSize;
    VkDeviceSize scratchForTlas = tlasSizes.buildScratchSize;
    if (scratchForBlas + scratchForTlas + probe.scratchAlignment > accel.scratch.size) {
        destroyBuffer(probe, accel.scratch);
        accel.scratch = createBuffer(probe, scratchForBlas + scratchForTlas + probe.scratchAlignment,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
    }

    auto alignUp = [](VkDeviceAddress address, VkDeviceSize alignment) {
        VkDeviceSize a = alignment ? alignment : 1;
        return (address + a - 1) & ~(a - 1);
    };

    buildInfo.dstAccelerationStructure = accel.blas;
    buildInfo.scratchData.deviceAddress = alignUp(accel.scratch.address, probe.scratchAlignment);

    tlasBuild.dstAccelerationStructure = accel.tlas;
    tlasBuild.scratchData.deviceAddress = alignUp(accel.scratch.address + scratchForBlas,
                                                  probe.scratchAlignment);

    VkAccelerationStructureBuildRangeInfoKHR blasRange{};
    blasRange.primitiveCount = accel.triangleCount;
    VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
    tlasRange.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* blasRanges = &blasRange;
    const VkAccelerationStructureBuildRangeInfoKHR* tlasRanges = &tlasRange;

    VkCommandBuffer commandBuffer = beginCommands(probe);
    VkAccelerationStructureBuildGeometryInfoKHR buildA = buildInfo;
    probe.cmdBuildAccelerationStructures(commandBuffer, 1, &buildA, &blasRanges);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                          | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
    VkAccelerationStructureBuildGeometryInfoKHR buildB = tlasBuild;
    probe.cmdBuildAccelerationStructures(commandBuffer, 1, &buildB, &tlasRanges);
    submitCommands(probe, commandBuffer);
}

}  // namespace

// --------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        Options options = parseArgs(argc, argv);

        if (options.listGpus) {
            std::vector<std::string> devices = enumeratePhysicalDevices();
            for (size_t i = 0; i < devices.size(); ++i) {
                std::printf("[%zu] %s\n", i, devices[i].c_str());
            }
            return 0;
        }

        std::filesystem::create_directories(options.outDir);
        std::string shaderPath = resolveShaderPath(options);

        // --- NGX extension requirements, queried before the device exists -----------------
        NgxApi ngx;
        loadNgxApi(ngx, options.shimDll);

        std::vector<std::string> notes;
        std::vector<std::string> ngxInstanceExtensions, ngxDeviceExtensions;
        {
            char buffer[8192];
            int count = ngx.requiredExtensions(0, buffer, int(sizeof(buffer)));
            if (count > 0) {
                std::istringstream stream(buffer);
                std::string line;
                while (std::getline(stream, line)) {
                    if (!line.empty()) {
                        ngxInstanceExtensions.push_back(line);
                    }
                }
            }
            count = ngx.requiredExtensions(1, buffer, int(sizeof(buffer)));
            if (count > 0) {
                std::istringstream stream(buffer);
                std::string line;
                while (std::getline(stream, line)) {
                    if (!line.empty()) {
                        ngxDeviceExtensions.push_back(line);
                    }
                }
            }
        }

        // --- device ---------------------------------------------------------------------
        Probe probe;
        ProbeConfig config;
        config.instanceExtensions = ngxInstanceExtensions;
        config.deviceExtensions = ngxDeviceExtensions;
        config.gpuIndex = options.gpuIndex;
        config.validation = options.validation;
        std::string deviceName = createProbe(probe, config, notes);

        // --- scene + acceleration structures --------------------------------------------
        Scene scene;
        buildScene(scene);
        Accel accel;
        buildAcceleration(probe, scene, accel);

        // --- guide images ---------------------------------------------------------------
        const VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                                           | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                           | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        struct GuideSet {
            Image color, normals, albedo, depth, motion, specAlbedo, specMotion, rr;
            uint32_t renderWidth = 0, renderHeight = 0;
        };
        GuideSet guides;

        auto createGuides = [&](uint32_t renderWidth, uint32_t renderHeight) {
            guides.renderWidth = renderWidth;
            guides.renderHeight = renderHeight;
            guides.color = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imageUsage);
            guides.normals = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imageUsage);
            guides.albedo = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imageUsage);
            guides.depth = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R32_SFLOAT, imageUsage);
            guides.motion = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R16G16_SFLOAT, imageUsage);
            guides.specAlbedo = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imageUsage);
            guides.specMotion = createImage(probe, renderWidth, renderHeight, VK_FORMAT_R16G16_SFLOAT, imageUsage);
            guides.rr = createImage(probe, options.displayWidth, options.displayHeight,
                                    VK_FORMAT_R16G16B16A16_SFLOAT, imageUsage);
            for (const Image* image : {&guides.color, &guides.normals, &guides.albedo, &guides.depth,
                                       &guides.motion, &guides.specAlbedo, &guides.specMotion,
                                       &guides.rr}) {
                transitionToGeneral(probe, *image);
            }
        };

        // --- NGX init -------------------------------------------------------------------
        bool ngxInitialized = false;
        bool rrAvailable = false;
        int initResult = 0;
        int capabilityResult = 0;
        std::string capabilityName = "(not queried)";

        if (!options.noRr) {
            std::wstring dataPath = utf8ToWide(std::filesystem::absolute(options.outDir).string());
            std::wstring featurePath = utf8ToWide(options.dllDir);
            initResult = ngx.init(0, dataPath.c_str(), probe.instance, probe.physicalDevice,
                                  probe.device, reinterpret_cast<void*>(vkGetInstanceProcAddr),
                                  reinterpret_cast<void*>(vkGetDeviceProcAddr), featurePath.c_str());
            if (ngxFailed(initResult)) {
                notes.push_back("NGX init failed: " + ngxResultToString(initResult));
            } else {
                ngxInitialized = true;
                rrAvailable = ngx.dlssdAvailable() != 0;
            }
            if (ngxInitialized) {
                capabilityResult = ngx.dlssdAvailable();
                capabilityName = ngxResultToString(capabilityResult);
            }
        }

        // --- initial render size --------------------------------------------------------
        uint32_t renderWidth = options.renderWidth;
        uint32_t renderHeight = options.renderHeight;
        if (renderWidth == 0 || renderHeight == 0) {
            renderWidth = options.displayWidth;
            renderHeight = options.displayHeight;
        }
        createGuides(renderWidth, renderHeight);

        // --- descriptors -----------------------------------------------------------------
        VkDescriptorSetLayoutBinding bindings[16]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t i = 0; i < 7; ++i) {
            bindings[2 + i].binding = 2 + i;
            bindings[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[2 + i].descriptorCount = 1;
            bindings[2 + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[9 + i].binding = 9 + i;
            bindings[9 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[9 + i].descriptorCount = 1;
            bindings[9 + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 16;
        layoutInfo.pBindings = bindings;
        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(probe.device, &layoutInfo, nullptr, &descriptorLayout) != VK_SUCCESS) {
            fail("vkCreateDescriptorSetLayout failed");
        }

        VkDescriptorPoolSize poolSizes[4] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},
        };
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(probe.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            fail("vkCreateDescriptorPool failed");
        }

        VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        setInfo.descriptorPool = descriptorPool;
        setInfo.descriptorSetCount = 1;
        setInfo.pSetLayouts = &descriptorLayout;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(probe.device, &setInfo, &descriptorSet) != VK_SUCCESS) {
            fail("vkAllocateDescriptorSets failed");
        }

        // --- buffers --------------------------------------------------------------------
        Buffer uniformBuffer = createBuffer(probe, sizeof(FrameUniforms),
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            false);
        auto storageBufferFor = [&](const std::vector<float>& data, size_t minimumFloats) {
            size_t bytes = std::max(data.size(), minimumFloats) * sizeof(float);
            Buffer buffer = createBuffer(probe, bytes,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         false);
            std::vector<float> padded(std::max(data.size(), minimumFloats), 0.0f);
            std::copy(data.begin(), data.end(), padded.begin());
            uploadBuffer(probe, buffer, padded.data(), padded.size() * sizeof(float));
            return buffer;
        };
        Buffer normalBuffer = storageBufferFor(scene.normals, 3);
        Buffer albedoBuffer = storageBufferFor(scene.albedos, 3);
        Buffer paramBuffer = storageBufferFor(scene.params, 6);
        Buffer emitterPosBuffer = storageBufferFor(scene.emitterPos, 4);
        Buffer emitterRadBuffer = storageBufferFor(scene.emitterRad, 4);
        Buffer indexBuffer = createBuffer(probe, scene.indices.size() * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          false);
        uploadBuffer(probe, indexBuffer, scene.indices.data(), scene.indices.size() * sizeof(uint32_t));

        auto writeDescriptor = [&](uint32_t binding, VkDescriptorType type, VkBuffer buffer,
                                   VkImageView view, VkFormat format, VkImageLayout layout) {
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = descriptorSet;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = type;
            VkDescriptorBufferInfo bufferInfo{buffer, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo imageInfo{nullptr, view, layout};
            if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                write.pImageInfo = &imageInfo;
            } else {
                write.pBufferInfo = &bufferInfo;
            }
            (void) format;
            vkUpdateDescriptorSets(probe.device, 1, &write, 0, nullptr);
        };

        VkWriteDescriptorSetAccelerationStructureKHR accelWrite{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        accelWrite.accelerationStructureCount = 1;
        accelWrite.pAccelerationStructures = &accel.tlas;
        VkWriteDescriptorSet tlasDescriptor{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        tlasDescriptor.dstSet = descriptorSet;
        tlasDescriptor.dstBinding = 0;
        tlasDescriptor.descriptorCount = 1;
        tlasDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        tlasDescriptor.pNext = &accelWrite;
        vkUpdateDescriptorSets(probe.device, 1, &tlasDescriptor, 0, nullptr);

        writeDescriptor(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, accel.vertexBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, normalBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, albedoBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, paramBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, indexBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, emitterPosBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, emitterRadBuffer.buffer, VK_NULL_HANDLE,
                        VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);

        // --- pipeline -------------------------------------------------------------------
        VkShaderModule shader = loadShaderModule(probe, shaderPath);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(probe.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            fail("vkCreatePipelineLayout failed");
        }
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shader;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(probe.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            fail("vkCreateComputePipelines failed");
        }

        auto bindGuideImages = [&]() {
            writeDescriptor(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.color.view,
                            guides.color.format, VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptor(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.normals.view,
                            guides.normals.format, VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptor(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.albedo.view,
                            guides.albedo.format, VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptor(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.depth.view,
                            guides.depth.format, VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptor(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.motion.view,
                            guides.motion.format, VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptor(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.specAlbedo.view,
                            guides.specAlbedo.format, VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptor(8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, guides.specMotion.view,
                            guides.specMotion.format, VK_IMAGE_LAYOUT_GENERAL);
        };
        bindGuideImages();

        // --- NGX feature -----------------------------------------------------------------
        // DLSS-RR signals InvalidParameter when the render:display ratio does not match the
        // requested quality, and the optimal-size query only works once the DLSSD feature
        // library has been loaded by a first create. So: create 1:1 (the shim maps that to
        // DLAA), let the library register, ask for the real size, rebuild.
        void* feature = nullptr;
        uint32_t featureRenderWidth = guides.renderWidth;
        uint32_t featureRenderHeight = guides.renderHeight;
        int createResult = 0;
        int queryResult = 0;
        bool queryAttempted = false;
        uint32_t queriedRenderWidth = 0, queriedRenderHeight = 0;
        float queriedSharpness = 0.0f;

        if (ngxInitialized && rrAvailable) {
            {
                VkCommandBuffer commandBuffer = beginCommands(probe);
                feature = ngx.createDlssd(commandBuffer, featureRenderWidth, featureRenderHeight,
                                          options.displayWidth, options.displayHeight,
                                          options.quality, 75, options.preset);
                createResult = ngx.lastResult();
                submitCommands(probe, commandBuffer);
            }
            if (!feature) {
                notes.push_back("ngxshim_create_dlssd failed: " + ngxResultToString(createResult));
            } else if (options.renderWidth == 0 && options.renderHeight == 0) {
                queryAttempted = true;
                queryResult = ngx.queryOptimalDlssd(options.displayWidth, options.displayHeight,
                                                   options.quality, &queriedRenderWidth,
                                                   &queriedRenderHeight, &queriedSharpness);
                if (!ngxFailed(queryResult) && queriedRenderWidth > 0 && queriedRenderHeight > 0
                        && (queriedRenderWidth != featureRenderWidth
                            || queriedRenderHeight != featureRenderHeight)) {
                    // Rebuild the guide images at the size NGX actually wants.
                    ngx.release(feature);
                    feature = nullptr;
                    for (Image* image : {&guides.color, &guides.normals, &guides.albedo, &guides.depth,
                                         &guides.motion, &guides.specAlbedo, &guides.specMotion,
                                         &guides.rr}) {
                        destroyImage(probe, *image);
                    }
                    createGuides(queriedRenderWidth, queriedRenderHeight);
                    bindGuideImages();

                    VkCommandBuffer commandBuffer = beginCommands(probe);
                    feature = ngx.createDlssd(commandBuffer, queriedRenderWidth, queriedRenderHeight,
                                              options.displayWidth, options.displayHeight,
                                              options.quality, 75, options.preset);
                    createResult = ngx.lastResult();
                    submitCommands(probe, commandBuffer);
                    if (feature) {
                        featureRenderWidth = queriedRenderWidth;
                        featureRenderHeight = queriedRenderHeight;
                    } else {
                        notes.push_back("rebuild at the NGX optimal size failed: "
                                        + ngxResultToString(createResult));
                    }
                }
            }
        }

        const uint32_t renderWidthFinal = guides.renderWidth;
        const uint32_t renderHeightFinal = guides.renderHeight;
        const int phaseCount = jitterPhaseCount(renderWidthFinal, options.displayWidth);

        // --- frame loop -----------------------------------------------------------------
        Vec3 sunDirection = normalize(Vec3{0.42f, 0.72f, 0.55f});
        Vec3 sunColour{1.0f, 0.95f, 0.88f};

        Mat4 prevViewProj;
        Vec3 prevCameraPosition{};
        bool havePrevious = false;

        const int readbackA = std::max(0, options.frames - 1 - options.hold);
        const int readbackB = options.frames - 1;

        FloatImage noisyA, noisyB, rrA, rrB;
        // Captured on the last moving frame, never on the final one: the tail of the run holds
        // the camera still so the temporal metric can settle, and a frozen camera drives every
        // motion vector to zero. Dumping that frame would show "all MVs are zero" and look like
        // a bug in a probe that is working correctly.
        FloatImage dumpNormals, dumpAlbedo, dumpDepth, dumpMotion, dumpSpecAlbedo, dumpSpecMotion;
        int evalResult = 0;
        bool evalFailed = false;
        std::string evalMessage;

        FrameUniforms uniforms{};
        std::vector<double> frameMilliseconds;
        frameMilliseconds.reserve(size_t(options.frames));

        for (int frame = 0; frame < options.frames; ++frame) {
            auto frameStart = std::chrono::steady_clock::now();

            // Freeze the camera for the tail frames so the temporal metric measures the
            // denoiser settling rather than the scene changing underneath it.
            int poseFrame = frame;
            if (options.hold > 0 && frame >= options.frames - options.hold) {
                poseFrame = options.frames - options.hold;
            }
            CameraPose pose = cameraAt(poseFrame, options);

            float aspect = float(options.displayWidth) / float(options.displayHeight);
            Mat4 view = lookAt(pose.position, pose.target, {0.0f, 1.0f, 0.0f});
            Mat4 projection = reversedZProjection(options.fovY * kPi / 180.0f, aspect, options.nearPlane);
            Mat4 viewProjection = multiply(projection, view);

            if (!havePrevious) {
                prevViewProj = viewProjection;
                prevCameraPosition = pose.position;
                havePrevious = true;
            }

            // Halton(2,3) - 0.5, advancing every frame even while the camera is frozen.
            int jitterIndex = (frame % phaseCount) + 1;
            float jitterX = halton(jitterIndex, 2) - 0.5f;
            float jitterY = halton(jitterIndex, 3) - 0.5f;

            Vec3 forward = normalize(pose.target - pose.position);
            Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
            Vec3 up = cross(right, forward);
            float tanHalfFovY = std::tan(options.fovY * kPi / 360.0f);

            std::memcpy(uniforms.viewProj, viewProjection.m, sizeof(uniforms.viewProj));
            std::memcpy(uniforms.prevViewProj, prevViewProj.m, sizeof(uniforms.prevViewProj));
            uniforms.camPos[0] = pose.position.x;
            uniforms.camPos[1] = pose.position.y;
            uniforms.camPos[2] = pose.position.z;
            uniforms.camPos[3] = float(frame);
            uniforms.prevCamPos[0] = prevCameraPosition.x;
            uniforms.prevCamPos[1] = prevCameraPosition.y;
            uniforms.prevCamPos[2] = prevCameraPosition.z;
            uniforms.prevCamPos[3] = 0.0f;
            uniforms.camRight[0] = right.x;
            uniforms.camRight[1] = right.y;
            uniforms.camRight[2] = right.z;
            uniforms.camRight[3] = tanHalfFovY * aspect;
            uniforms.camUp[0] = up.x;
            uniforms.camUp[1] = up.y;
            uniforms.camUp[2] = up.z;
            uniforms.camUp[3] = tanHalfFovY;
            uniforms.camFwd[0] = forward.x;
            uniforms.camFwd[1] = forward.y;
            uniforms.camFwd[2] = forward.z;
            uniforms.camFwd[3] = 0.0f;
            uniforms.jitterRender[0] = jitterX;
            uniforms.jitterRender[1] = jitterY;
            uniforms.jitterRender[2] = float(renderWidthFinal);
            uniforms.jitterRender[3] = float(renderHeightFinal);
            uniforms.displayCounts[0] = float(options.displayWidth);
            uniforms.displayCounts[1] = float(options.displayHeight);
            uniforms.displayCounts[2] = float(scene.emitterPos.size() / 4);
            uniforms.displayCounts[3] = 0.0f;
            uniforms.sunDir[0] = sunDirection.x;
            uniforms.sunDir[1] = sunDirection.y;
            uniforms.sunDir[2] = sunDirection.z;
            uniforms.sunDir[3] = options.sunIntensity;
            uniforms.sunColor[0] = sunColour.x;
            uniforms.sunColor[1] = sunColour.y;
            uniforms.sunColor[2] = sunColour.z;
            uniforms.sunColor[3] = 0.0f;

            std::memcpy(uniformBuffer.mapped, &uniforms, sizeof(uniforms));
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = uniformBuffer.memory;
            range.size = VK_WHOLE_SIZE;
            vkFlushMappedMemoryRanges(probe.device, 1, &range);

            VkCommandBuffer commandBuffer = beginCommands(probe);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                                    1, &descriptorSet, 0, nullptr);
            vkCmdDispatch(commandBuffer, (renderWidthFinal + 7) / 8, (renderHeightFinal + 7) / 8, 1);
            memoryBarrier(commandBuffer);

            if (feature) {
                int result = ngx.evaluateDlssd(commandBuffer, feature,
                                               guides.color.view, guides.color.image, int(guides.color.format),
                                               guides.depth.view, guides.depth.image, int(guides.depth.format),
                                               guides.motion.view, guides.motion.image, int(guides.motion.format),
                                               guides.albedo.view, guides.albedo.image, int(guides.albedo.format),
                                               guides.specAlbedo.view, guides.specAlbedo.image, int(guides.specAlbedo.format),
                                               guides.normals.view, guides.normals.image, int(guides.normals.format),
                                               guides.specMotion.view, guides.specMotion.image, int(guides.specMotion.format),
                                               0, 0, 0,
                                               guides.rr.view, guides.rr.image, int(guides.rr.format),
                                               renderWidthFinal, renderHeightFinal,
                                               options.displayWidth, options.displayHeight,
                                               // Caustica negates the jitter it reported to NGX; mirror that.
                                               -jitterX, -jitterY, 1.0f, 1.0f,
                                               frame == 0 ? 1 : 0,
                                               1000.0f / 60.0f,
                                               uniforms.viewProj, uniforms.viewProj);
                evalResult = result;
                if (ngxFailed(result) && !evalFailed) {
                    evalFailed = true;
                    evalMessage = ngxResultToString(result);
                }
            }

            memoryBarrier(commandBuffer);
            submitCommands(probe, commandBuffer);

            if (frame == readbackA || frame == readbackB) {
                FloatImage noisyFrame = readFloatImage(probe, guides.color);
                if (frame == readbackA) {
                    noisyA = noisyFrame;
                    if (feature) {
                        rrA = readFloatImage(probe, guides.rr);
                    }
                    if (options.dumpGuides) {
                        dumpNormals = readFloatImage(probe, guides.normals);
                        dumpAlbedo = readFloatImage(probe, guides.albedo);
                        dumpDepth = readFloatImage(probe, guides.depth);
                        dumpMotion = readFloatImage(probe, guides.motion);
                        dumpSpecAlbedo = readFloatImage(probe, guides.specAlbedo);
                        dumpSpecMotion = readFloatImage(probe, guides.specMotion);
                    }
                } else {
                    noisyB = noisyFrame;
                    if (feature) {
                        rrB = readFloatImage(probe, guides.rr);
                    }
                }
            }

            prevViewProj = viewProjection;
            prevCameraPosition = pose.position;

            auto frameEnd = std::chrono::steady_clock::now();
            frameMilliseconds.push_back(
                    std::chrono::duration<double, std::milli>(frameEnd - frameStart).count());
        }

        // --- guide dumps ----------------------------------------------------------------
        std::vector<std::string> writtenFiles;
        auto tonemap = [&](const FloatImage& image, const std::string& name) {
            std::string path = (std::filesystem::path(options.outDir) / name).string();
            if (writePng(path, image.width, image.height, tonemapToRgb8(image, options.exposure, options.key))) {
                writtenFiles.push_back(path);
            }
        };
        auto guidePng = [&](const FloatImage& image, const std::string& name, float lo, float hi) {
            std::string path = (std::filesystem::path(options.outDir) / name).string();
            if (writePng(path, image.width, image.height, mapToRgb8(image, lo, hi))) {
                writtenFiles.push_back(path);
            }
        };

        std::string noisyExr, rrExr;
        if (!noisyB.data.empty()) {
            tonemap(noisyB, "rt-noisy.png");
            noisyExr = (std::filesystem::path(options.outDir) / "rt-noisy.exr").string();
            writeExrHalf(noisyExr, noisyB);
        }
        if (!rrB.data.empty()) {
            tonemap(rrB, "rr-output.png");
            rrExr = (std::filesystem::path(options.outDir) / "rr-output.exr").string();
            writeExrHalf(rrExr, rrB);
        }

        if (options.dumpGuides && !dumpNormals.data.empty()) {
            guidePng(dumpNormals, "guide-normals.png", -1.0f, 1.0f);
            guidePng(dumpAlbedo, "guide-albedo.png", 0.0f, 1.0f);
            guidePng(dumpDepth, "guide-depth.png", 0.0f, 1.0f);
            guidePng(dumpMotion, "guide-motion.png", -6.0f, 6.0f);
            guidePng(dumpSpecAlbedo, "guide-specalbedo.png", 0.0f, 1.0f);
            guidePng(dumpSpecMotion, "guide-specmotion.png", -6.0f, 6.0f);
        }

        // --- metrics --------------------------------------------------------------------
        double highFrequencyNoisy = 0.0, highFrequencyRr = 0.0;
        double temporalNoisy = 0.0, temporalRr = 0.0;
        double rrVersusNaiveUpscale = 0.0;
        double denoiseRatio = 0.0, temporalRatio = 0.0;
        bool haveMetrics = false;

        if (!noisyB.data.empty()) {
            FloatImage noisyDisplay = resampleNearest(noisyB, options.displayWidth, options.displayHeight);
            highFrequencyNoisy = highFrequencyEnergy(noisyDisplay);
            if (!rrB.data.empty()) {
                highFrequencyRr = highFrequencyEnergy(rrB);
                rrVersusNaiveUpscale = meanAbsDifference(rrB, noisyDisplay);
                if (highFrequencyNoisy > 1.0e-9) {
                    denoiseRatio = highFrequencyRr / highFrequencyNoisy;
                }
            }
            if (!noisyA.data.empty()) {
                temporalNoisy = meanAbsDifference(noisyA, noisyB);
            }
            if (!rrA.data.empty() && !rrB.data.empty()) {
                temporalRr = meanAbsDifference(rrA, rrB);
                if (temporalNoisy > 1.0e-9) {
                    temporalRatio = temporalRr / temporalNoisy;
                }
            }
            haveMetrics = true;
        }

        double noisyMeanLuma = 0.0, rrMeanLuma = 0.0;
        ImageStats noisyStats, rrStats;
        if (!noisyB.data.empty()) {
            noisyStats = computeStats(noisyB);
            noisyMeanLuma = noisyStats.meanLuma;
        }
        if (!rrB.data.empty()) {
            rrStats = computeStats(rrB);
            rrMeanLuma = rrStats.meanLuma;
        }

        DllFacts dll = inspectDll(options.dllDir);
        std::string alternateShim = options.shimDll.empty() ? "" : options.shimDll;

        // Two distinct states that must never be reported the same way: NGX was deliberately
        // skipped, versus NGX ran and produced nothing. Formatting both as a zeroed index would
        // read as "RR achieved perfect denoising" when in fact RR never executed.
        const bool rrRan = feature != nullptr;
        const std::string skippedText = "skipped (--no-rr)";
        const std::string initText = options.noRr ? skippedText : ngxResultToString(initResult);
        const std::string createText = options.noRr ? skippedText : ngxResultToString(createResult);
        const std::string queryText = options.noRr ? skippedText
                : (queryAttempted ? ngxResultToString(queryResult) : "skipped (--render given)");
        const std::string evalText = options.noRr ? skippedText : ngxResultToString(evalResult);
        const bool haveTemporalA = !noisyA.data.empty();
        const bool haveTemporalRrA = !rrA.data.empty();

        auto numberOrNull = [](bool have, double value, int digits) -> std::string {
            if (!have) {
                return "null";
            }
            std::ostringstream out;
            out << roundTo(value, digits);
            return out.str();
        };

        // --- report ---------------------------------------------------------------------
        double totalMilliseconds = 0.0;
        for (double ms : frameMilliseconds) {
            totalMilliseconds += ms;
        }
        double averageMilliseconds = frameMilliseconds.empty()
                ? 0.0 : totalMilliseconds / double(frameMilliseconds.size());

        std::ostringstream report;
        report << "{\n";
        report << "  \"schema\": \"rtprobe/1\",\n";
        report << "  \"device\": {\n";
        report << "    \"name\": \"" << jsonEscape(deviceName) << "\",\n";
        report << "    \"driver\": \"" << driverVersionString(probe.deviceProps.driverVersion) << "\",\n";
        report << "    \"api\": \"" << apiVersionString(probe.deviceProps.apiVersion) << "\",\n";
        report << "    \"deviceId\": " << probe.deviceProps.deviceID << ",\n";
        report << "    \"vendorId\": " << probe.deviceProps.vendorID << "\n";
        report << "  },\n";
        report << "  \"ngx\": {\n";
        report << "    \"shim\": \"" << jsonEscape(ngx.source) << "\",\n";
        report << "    \"dllDir\": \"" << jsonEscape(options.dllDir) << "\",\n";
        report << "    \"dllPath\": \"" << jsonEscape(dll.path) << "\",\n";
        report << "    \"dllExists\": " << (dll.exists ? "true" : "false") << ",\n";
        report << "    \"dllBytes\": " << dll.size << ",\n";
        report << "    \"dllFileVersion\": \"" << jsonEscape(dll.fileVersion) << "\",\n";
        report << "    \"dllSha256\": \"" << jsonEscape(dll.sha256) << "\",\n";
        report << "    \"initialized\": " << (ngxInitialized ? "true" : "false") << ",\n";
        report << "    \"skipped\": " << (options.noRr ? "true" : "false") << ",\n";
        report << "    \"initResult\": \"" << jsonEscape(initText) << "\",\n";
        report << "    \"rayReconstructionAvailable\": " << (rrAvailable ? "true" : "false") << ",\n";
        report << "    \"capabilityQuery\": \"" << jsonEscape(capabilityName) << "\",\n";
        report << "    \"requiredInstanceExtensions\": " << ngxInstanceExtensions.size() << ",\n";
        report << "    \"requiredDeviceExtensions\": " << ngxDeviceExtensions.size() << ",\n";
        report << "    \"alternateShimDll\": \"" << jsonEscape(alternateShim) << "\"\n";
        report << "  },\n";
        report << "  \"frame\": {\n";
        report << "    \"displayWidth\": " << options.displayWidth << ",\n";
        report << "    \"displayHeight\": " << options.displayHeight << ",\n";
        report << "    \"renderWidth\": " << renderWidthFinal << ",\n";
        report << "    \"renderHeight\": " << renderHeightFinal << ",\n";
        report << "    \"quality\": " << options.quality << ",\n";
        report << "    \"preset\": " << options.preset << ",\n";
        report << "    \"featureFlags\": 75,\n";
        report << "    \"frames\": " << options.frames << ",\n";
        report << "    \"holdFrames\": " << options.hold << ",\n";
        report << "    \"jitterPhases\": " << phaseCount << ",\n";
        report << "    \"motion\": \"" << jsonEscape(options.motion) << "\",\n";
        report << "    \"triangles\": " << accel.triangleCount << ",\n";
        report << "    \"blasBytes\": " << accel.blasBytes << ",\n";
        report << "    \"tlasBytes\": " << accel.tlasBytes << "\n";
        report << "  },\n";
        report << "  \"dlssd\": {\n";
        report << "    \"ran\": " << (rrRan ? "true" : "false") << ",\n";
        report << "    \"created\": " << (feature ? "true" : "false") << ",\n";
        report << "    \"createResult\": \"" << jsonEscape(createText) << "\",\n";
        report << "    \"queryOptimalResult\": \"" << jsonEscape(queryText) << "\",\n";
        report << "    \"queriedRenderWidth\": " << queriedRenderWidth << ",\n";
        report << "    \"queriedRenderHeight\": " << queriedRenderHeight << ",\n";
        report << "    \"queriedSharpness\": " << roundTo(queriedSharpness, 4) << ",\n";
        report << "    \"lastEvaluateResult\": \"" << jsonEscape(evalText) << "\",\n";
        report << "    \"evaluateFailed\": " << (evalFailed ? "true" : "false") << ",\n";
        report << "    \"evaluateMessage\": \"" << jsonEscape(evalMessage) << "\"\n";
        report << "  },\n";
        report << "  \"metrics\": {\n";
        report << "    \"available\": " << (haveMetrics ? "true" : "false") << ",\n";
        report << "    \"highFrequencyNoisyUpscaled\": " << numberOrNull(haveMetrics, highFrequencyNoisy, 6) << ",\n";
        report << "    \"highFrequencyRayReconstruction\": " << numberOrNull(rrRan, highFrequencyRr, 6) << ",\n";
        report << "    \"highFrequencyRatio\": " << numberOrNull(rrRan && denoiseRatio > 0.0, denoiseRatio, 4) << ",\n";
        report << "    \"temporalDeltaNoisy\": " << numberOrNull(haveTemporalA, temporalNoisy, 6) << ",\n";
        report << "    \"temporalDeltaRayReconstruction\": " << numberOrNull(haveTemporalRrA, temporalRr, 6) << ",\n";
        report << "    \"temporalRatio\": " << numberOrNull(haveTemporalRrA && temporalRatio > 0.0, temporalRatio, 4) << ",\n";
        report << "    \"rrVsNaiveUpscale\": " << numberOrNull(rrRan, rrVersusNaiveUpscale, 6) << ",\n";
        report << "    \"noisyMeanLuma\": " << numberOrNull(haveMetrics, noisyMeanLuma, 6) << ",\n";
        report << "    \"rrMeanLuma\": " << numberOrNull(rrRan, rrMeanLuma, 6) << ",\n";
        report << "    \"noisyNonFinite\": " << noisyStats.nonFinite << ",\n";
        report << "    \"rrNonFinite\": " << (rrRan ? rrStats.nonFinite : int64_t(-1)) << "\n";
        report << "  },\n";
        report << "  \"artifacts\": {\n";
        report << "    \"noisyExr\": \"" << jsonEscape(noisyExr) << "\",\n";
        report << "    \"rrExr\": \"" << jsonEscape(rrExr) << "\",\n";
        report << "    \"noisyMd5\": \"" << jsonEscape(noisyExr.empty() ? std::string() : md5OfFile(noisyExr)) << "\",\n";
        report << "    \"rrMd5\": \"" << jsonEscape(rrExr.empty() ? std::string() : md5OfFile(rrExr)) << "\"\n";
        report << "  },\n";
        report << "  \"timing\": {\n";
        report << "    \"wallClockPerFrameMs\": " << roundTo(averageMilliseconds, 3) << ",\n";
        report << "    \"wallClockTotalMs\": " << roundTo(totalMilliseconds, 3) << "\n";
        report << "  },\n";
        report << "  \"notes\": [";
        for (size_t i = 0; i < notes.size(); ++i) {
            report << (i ? ",\n    " : "\n    ") << "\"" << jsonEscape(notes[i]) << "\"";
        }
        report << (notes.empty() ? "],\n" : "\n  ],\n");
        report << "  \"outputs\": [";
        for (size_t i = 0; i < writtenFiles.size(); ++i) {
            report << (i ? ",\n    " : "\n    ") << "\"" << jsonEscape(writtenFiles[i]) << "\"";
        }
        report << (writtenFiles.empty() ? "]\n" : "\n  ]\n");
        report << "}\n";

        std::string reportPath = (std::filesystem::path(options.outDir) / "report.json").string();
        {
            std::ofstream out(reportPath, std::ios::binary);
            out << report.str();
        }

        // --- summary --------------------------------------------------------------------
        if (!options.quiet) {
            std::printf("device            : %s (driver %s, Vulkan %s)\n", deviceName.c_str(),
                        driverVersionString(probe.deviceProps.driverVersion).c_str(),
                        apiVersionString(probe.deviceProps.apiVersion).c_str());
            std::printf("shim              : %s\n", ngx.source.c_str());
            std::printf("nvngx_dlssd.dll   : %s\n", dll.exists ? dll.path.c_str() : "(not found)");
            if (dll.exists) {
                std::printf("                    %llu bytes, FileVersion %s\n",
                            static_cast<unsigned long long>(dll.size),
                            dll.fileVersion.empty() ? "(none)" : dll.fileVersion.c_str());
                std::printf("                    SHA-256 %s\n", dll.sha256.c_str());
            }
            std::printf("NGX init          : %s\n", initText.c_str());
            std::printf("RR available      : %s\n",
                        options.noRr ? "n/a (--no-rr)" : (rrAvailable ? "true" : "false"));
            std::printf("trace             : %ux%u -> %ux%u (quality %d, preset %d, %u tris)\n",
                        renderWidthFinal, renderHeightFinal, options.displayWidth,
                        options.displayHeight, options.quality, options.preset,
                        accel.triangleCount);
            if (rrRan) {
                std::printf("RR feature        : created, last evaluate %s\n", evalText.c_str());
            } else {
                std::printf("RR feature        : NOT created (%s)\n", createText.c_str());
            }
            if (haveMetrics) {
                if (rrRan) {
                    std::printf("high-freq energy  : noisy(upscaled) %.6f -> RR %.6f  (ratio %.4f)\n",
                                highFrequencyNoisy, highFrequencyRr, denoiseRatio);
                    std::printf("temporal delta    : noisy %.6f -> RR %.6f  (ratio %.4f)\n",
                                temporalNoisy, temporalRr, temporalRatio);
                    std::printf("RR vs naive upsc. : %.6f  (0 would mean RR changed nothing)\n",
                                rrVersusNaiveUpscale);
                    std::printf("mean luma         : noisy %.6f -> RR %.6f   non-finite %lld/%lld\n",
                                noisyMeanLuma, rrMeanLuma,
                                static_cast<long long>(noisyStats.nonFinite),
                                static_cast<long long>(rrStats.nonFinite));
                } else {
                    std::printf("high-freq energy  : noisy(upscaled) %.6f ; RR n/a (Ray Reconstruction did not run)\n",
                                highFrequencyNoisy);
                    std::printf("temporal delta    : noisy %.6f ; RR n/a\n", temporalNoisy);
                    std::printf("mean luma         : noisy %.6f   non-finite %lld\n",
                                noisyMeanLuma, static_cast<long long>(noisyStats.nonFinite));
                }
            }
            std::printf("timing            : %.2f ms/frame wall clock over %d frames\n",
                        averageMilliseconds, options.frames);
            std::printf("report            : %s\n", reportPath.c_str());
            if (!noisyExr.empty()) {
                std::printf("rt-noisy    md5   : %s\n", md5OfFile(noisyExr).c_str());
            }
            if (!rrExr.empty()) {
                std::printf("rr-output   md5   : %s\n", md5OfFile(rrExr).c_str());
            }
            for (const std::string& file : writtenFiles) {
                std::printf("image             : %s\n", file.c_str());
            }
            for (const std::string& note : notes) {
                std::printf("note              : %s\n", note.c_str());
            }
        }

        // --- teardown -------------------------------------------------------------------
        if (feature && ngx.release) {
            ngx.release(feature);
        }
        if (ngxInitialized && ngx.shutdown) {
            vkDeviceWaitIdle(probe.device);
            ngx.shutdown(probe.device);
        }

        vkDestroyPipeline(probe.device, pipeline, nullptr);
        vkDestroyPipelineLayout(probe.device, pipelineLayout, nullptr);
        vkDestroyShaderModule(probe.device, shader, nullptr);
        vkDestroyDescriptorPool(probe.device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(probe.device, descriptorLayout, nullptr);

        destroyBuffer(probe, uniformBuffer);
        destroyBuffer(probe, normalBuffer);
        destroyBuffer(probe, albedoBuffer);
        destroyBuffer(probe, paramBuffer);
        destroyBuffer(probe, indexBuffer);
        destroyBuffer(probe, emitterPosBuffer);
        destroyBuffer(probe, emitterRadBuffer);

        for (Image* image : {&guides.color, &guides.normals, &guides.albedo, &guides.depth,
                             &guides.motion, &guides.specAlbedo, &guides.specMotion, &guides.rr}) {
            destroyImage(probe, *image);
        }

        if (accel.blas != VK_NULL_HANDLE) {
            probe.destroyAccelerationStructure(probe.device, accel.blas, nullptr);
        }
        if (accel.tlas != VK_NULL_HANDLE) {
            probe.destroyAccelerationStructure(probe.device, accel.tlas, nullptr);
        }
        destroyBuffer(probe, accel.blasBuffer);
        destroyBuffer(probe, accel.tlasBuffer);
        destroyBuffer(probe, accel.scratch);
        destroyBuffer(probe, accel.vertexBuffer);
        destroyBuffer(probe, accel.indexBuffer);
        destroyBuffer(probe, accel.instanceBuffer);

        unloadNgxApi(ngx);
        destroyProbe(probe);

        return (rrRan || options.noRr) ? 0 : 3;
    } catch (const std::exception& error) {
        std::printf("rtprobe failed: %s\n", error.what());
        return 1;
    }
}
