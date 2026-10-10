// matprobe -- an offscreen preview and measurement tool for LabPBR materials.
//
// Point it at a resource pack and a material name. It loads the albedo/_s/_n triplet, runs the
// same LabPBR decode the game runs at pack load, renders the result with a path tracer, and
// writes a PNG next to a report.json of the decoded numbers.
//
// The premise matches rtprobe's: asking "does this material look right" from inside the game
// costs booting Minecraft, loading a world and squinting at a screenshot. Here it costs one
// command, and the answer comes back as both an image and per-channel statistics.
//
// What this deliberately does not do: it does not re-derive the decode from the shader side.
// The pages uploaded to the GPU are already physical channels (see material.cpp), exactly like
// the game's canonical material pages, so the shader samples rather than decodes.

#include "vk.h"
#include "image_io.h"
#include "material.h"

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

using namespace matprobe;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// --------------------------------------------------------------------------------------
// small math
// --------------------------------------------------------------------------------------

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

// --------------------------------------------------------------------------------------
// options
// --------------------------------------------------------------------------------------

struct Options {
    std::string packDir;
    std::vector<std::string> materialNames;   // ns:path under --pack
    std::vector<std::string> albedoFiles;     // explicit albedo PNGs
    std::vector<std::string> specFiles;
    std::vector<std::string> normalFiles;
    std::string mode = "plate";               // plate | sphere
    std::string outDir = "matprobe-out";
    std::string debugChannel = "off";
    std::string shader;
    std::string sunDirection;                 // "x,y,z"; empty means derive from the camera
    uint32_t displayWidth = 1600;
    uint32_t displayHeight = 900;
    int samples = 64;
    float repeat = 1.0f;
    float sun = 3.0f;
    float sky = 1.0f;
    float emission = 1.0f;
    float parallax = 0.25f;   // matches CausticaConfig Rt.Composite.PARALLAX_DEPTH's default
    int layers = 12;          // PARALLAX_LAYERS in closest_hit.rchit.slang
    float minVz = 0.2f;       // PARALLAX_MIN_VZ in closest_hit.rchit.slang
    bool linearPages = false;   // sample the material pages linearly instead of through the atlas sampler
    bool clampSlope = false;    // clamp the march's 1/Z slope to 1, so the reach cannot exceed parallaxDepth
    bool texelSnap = false;     // round the displaced UV onto the sprite's texel grid (iterationRP does this)
    bool stepNormal = false;    // report a vertical wall normal for the band a step exposes
    float tilt = 35.0f;       // camera angle off the plate normal; a parallax march needs one
    float zoom = 1.0f;        // >1 moves the camera in, for looking at a surface close up
    std::string filter = "nearest";   // matches the sampler the game binds its textures with
    float fovY = 40.0f;
    int gpuIndex = -1;
    bool listGpus = false;
    bool listMaterials = false;
    bool validation = false;
    bool quiet = false;
    bool json = false;
};

void printUsage() {
    std::printf(
        "matprobe -- offscreen LabPBR material preview and decoder report\n"
        "\n"
        "  --pack DIR         resource-pack root (containing assets/) or its assets/\n"
        "  --material NAME    logical material name, e.g. minecraft:block/stone (repeatable)\n"
        "  --albedo FILE      explicit albedo PNG; _s/_n are looked for next to it (repeatable)\n"
        "  --spec FILE        explicit _s PNG       (pairs with the matching --albedo)\n"
        "  --normal FILE      explicit _n PNG       (pairs with the matching --albedo)\n"
        "  --list-materials   list every PBR material in --pack and exit\n"
        "  --mode MODE        plate | sphere                          [plate]\n"
        "  --display WxH      output resolution                       [1600x900]\n"
        "  --samples N        iterations per pixel                    [64]\n"
        "  --repeat F         UV tiling of the material on the plate  [1.0]\n"
        "  --sun F            sun intensity                           [3.0]\n"
        "  --sky F            sky intensity                           [1.0]\n"
        "  --emission F       emission scale                          [1.0]\n"
        "  --parallax F       material parallax depth 0..0.5, 0 = off  [0.25]\n"
        "  --layers N         parallax march layers (game uses 12)      [12]\n"
        "  --min-vz F         parallax Z floor   (game uses 0.2)       [0.2]\n"
        "  --clamp-slope      clamp the march's lateral slope to 1, so the reach\n"
        "                     can never exceed --parallax\n"
        "  --texel-snap       round the displaced UV onto the sprite's texel grid;\n"
        "                     iterationRP does this and Caustica does not\n"
        "  --step-normal      light the band a step exposes as a vertical wall\n"
        "                     instead of through the normal map\n"
        "  --linear-pages     sample the material pages with linear filtering\n"
        "                     while albedo keeps NEAREST (experiment)\n"
        "  --tilt DEGREES     camera angle off the plate normal       [35]\n"
        "  --zoom F           camera distance divisor, >1 moves in     [1.0]\n"
        "  --filter MODE      nearest | linear texel filtering    [nearest]\n"
        "  --sun-dir X,Y,Z    sun direction; default derives it from the camera+plate normal\n"
        "  --debug CHANNEL    off | albedo | normal | roughness | metalness | f0 |\n"
        "                     emission | sss | porosity | height | ao        [off]\n"
        "  --out DIR          output directory                        [matprobe-out]\n"
        "  --shader PATH      explicit matprobe.comp.spv\n"
        "  --gpu N            pick a physical device by index\n"
        "  --list-gpus        list devices and exit\n"
        "  --validation       enable the Khronos validation layer\n"
        "  --quiet            suppress the human-readable summary\n"
        "  --json             print the report to stdout as well\n");
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
        } else if (arg == "--pack") {
            options.packDir = next(i);
        } else if (arg == "--material") {
            options.materialNames.push_back(next(i));
        } else if (arg == "--albedo") {
            options.albedoFiles.push_back(next(i));
        } else if (arg == "--spec") {
            options.specFiles.push_back(next(i));
        } else if (arg == "--normal") {
            options.normalFiles.push_back(next(i));
        } else if (arg == "--mode") {
            options.mode = next(i);
        } else if (arg == "--display") {
            if (!parseSize(next(i), options.displayWidth, options.displayHeight)) {
                fail("--display wants WxH");
            }
        } else if (arg == "--samples") {
            options.samples = std::stoi(next(i));
        } else if (arg == "--repeat") {
            options.repeat = std::stof(next(i));
        } else if (arg == "--sun") {
            options.sun = std::stof(next(i));
        } else if (arg == "--sky") {
            options.sky = std::stof(next(i));
        } else if (arg == "--emission") {
            options.emission = std::stof(next(i));
        } else if (arg == "--parallax") {
            options.parallax = std::stof(next(i));
        } else if (arg == "--layers") {
            options.layers = std::stoi(next(i));
        } else if (arg == "--min-vz") {
            options.minVz = std::stof(next(i));
        } else if (arg == "--linear-pages") {
            options.linearPages = true;
        } else if (arg == "--clamp-slope") {
            options.clampSlope = true;
        } else if (arg == "--texel-snap") {
            options.texelSnap = true;
        } else if (arg == "--step-normal") {
            options.stepNormal = true;
        } else if (arg == "--tilt") {
            options.tilt = std::stof(next(i));
        } else if (arg == "--zoom") {
            options.zoom = std::stof(next(i));
        } else if (arg == "--filter") {
            options.filter = next(i);
        } else if (arg == "--sun-dir") {
            options.sunDirection = next(i);
        } else if (arg == "--debug") {
            options.debugChannel = next(i);
        } else if (arg == "--out") {
            options.outDir = next(i);
        } else if (arg == "--shader") {
            options.shader = next(i);
        } else if (arg == "--gpu") {
            options.gpuIndex = std::stoi(next(i));
        } else if (arg == "--list-gpus") {
            options.listGpus = true;
        } else if (arg == "--list-materials") {
            options.listMaterials = true;
        } else if (arg == "--validation") {
            options.validation = true;
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--json") {
            options.json = true;
        } else {
            fail("unknown argument " + arg + " (try --help)");
        }
    }

    if (options.mode != "plate" && options.mode != "sphere") {
        fail("--mode must be plate or sphere");
    }
    if (options.samples < 1) {
        fail("--samples must be at least 1");
    }
    if (options.repeat <= 0.0f) {
        fail("--repeat must be positive");
    }
    if (options.parallax < 0.0f || options.parallax > 0.5f) {
        fail("--parallax must be in 0..0.5, the range the game's Material Parallax setting offers");
    }
    if (options.tilt < 0.0f || options.tilt > 85.0f) {
        fail("--tilt must be in 0..85 degrees");
    }
    if (options.zoom <= 0.0f) {
        fail("--zoom must be positive");
    }
    if (options.layers < 1 || options.layers > 64) {
        fail("--layers must be in 1..64 (the shader's loop bound)");
    }
    if (options.minVz <= 0.0f) {
        fail("--min-vz must be positive");
    }
    if (options.filter != "nearest" && options.filter != "linear") {
        fail("--filter must be nearest or linear");
    }
    if (options.specFiles.size() > options.albedoFiles.size()
            || options.normalFiles.size() > options.albedoFiles.size()) {
        fail("--spec/--normal must not outnumber --albedo");
    }
    return options;
}

// Channel numbers must match the switch in shaders/matprobe.comp's debugValue().
int debugChannelIndex(const std::string& name) {
    static const std::pair<const char*, int> kChannels[] = {
            {"off", 0}, {"albedo", 1}, {"normal", 2}, {"roughness", 3}, {"metalness", 4},
            {"f0", 5}, {"emission", 6}, {"sss", 7}, {"porosity", 8}, {"height", 9}, {"ao", 10}};
    for (const auto& entry : kChannels) {
        if (name == entry.first) {
            return entry.second;
        }
    }
    fail("unknown --debug channel " + name + " (try --help)");
}

// --------------------------------------------------------------------------------------
// material requests
// --------------------------------------------------------------------------------------

struct MaterialRequest {
    std::string name;
    std::string albedoPath;
    std::string specPath;
    std::string normalPath;
};

std::string siblingTexturePath(const std::string& albedoPath, const std::string& suffix) {
    std::filesystem::path path(albedoPath);
    return (path.parent_path() / (path.stem().string() + suffix)).string();
}

std::vector<MaterialRequest> collectRequests(const Options& options) {
    std::vector<MaterialRequest> requests;

    if (!options.materialNames.empty()) {
        if (options.packDir.empty()) {
            fail("--material needs --pack to resolve the material name");
        }
        std::string assetsRoot = findAssetsRoot(options.packDir);
        for (const std::string& name : options.materialNames) {
            MaterialRequest request;
            request.name = name;
            request.albedoPath = resolveTexturePath(assetsRoot, name, ".png");
            request.specPath = resolveTexturePath(assetsRoot, name, "_s.png");
            request.normalPath = resolveTexturePath(assetsRoot, name, "_n.png");
            requests.push_back(request);
        }
    }

    for (size_t i = 0; i < options.albedoFiles.size(); ++i) {
        MaterialRequest request;
        std::filesystem::path albedo(options.albedoFiles[i]);
        request.name = albedo.stem().string();
        request.albedoPath = albedo.string();
        request.specPath = i < options.specFiles.size() ? options.specFiles[i]
                                                        : siblingTexturePath(request.albedoPath, "_s");
        request.normalPath = i < options.normalFiles.size() ? options.normalFiles[i]
                                                            : siblingTexturePath(request.albedoPath, "_n");
        requests.push_back(request);
    }

    for (const MaterialRequest& request : requests) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(request.albedoPath, error)) {
            fail("albedo texture not found: " + request.albedoPath);
        }
    }
    return requests;
}

// --------------------------------------------------------------------------------------
// page packing
// --------------------------------------------------------------------------------------

struct PageLayout {
    uint32_t pageWidth = 0, pageHeight = 0;
    uint32_t cellSize = 0;
    uint32_t gutter = 4;
    uint32_t columns = 1, rows = 1;
    std::vector<float> rects;   // 4 per material: u0, v0, du, dv
};

// Every material gets a square cell, and the cell is padded with a copy of its own border
// texels. Without that gutter a bilinear tap at the edge of one cell would read into its
// neighbour, which is exactly the class of artefact this tool exists to surface rather than
// manufacture.
PageLayout layoutPages(const std::vector<Material>& materials, uint32_t gutter) {
    PageLayout layout;
    layout.gutter = gutter;
    uint32_t content = 1;
    for (const Material& material : materials) {
        content = std::max(content, std::max(material.width, material.height));
    }
    layout.cellSize = content + 2 * gutter;

    const size_t count = std::max<size_t>(1, materials.size());
    layout.columns = uint32_t(std::ceil(std::sqrt(double(count))));
    layout.rows = uint32_t((count + layout.columns - 1) / layout.columns);
    layout.pageWidth = layout.columns * layout.cellSize;
    layout.pageHeight = layout.rows * layout.cellSize;
    if (layout.pageWidth > 8192 || layout.pageHeight > 8192) {
        fail("material pages would exceed 8192x8192; probe fewer or smaller textures");
    }

    // 6 floats per material: the cell's page-UV origin and size, then the cell's own texel count.
    // The texel count is what the optional texel-grid snap rounds a displaced sample onto, so it
    // has to travel with the rect rather than be derived from the page size.
    layout.rects.assign(materials.size() * 6, 0.0f);
    for (size_t index = 0; index < materials.size(); ++index) {
        uint32_t column = uint32_t(index) % layout.columns;
        uint32_t row = uint32_t(index) / layout.columns;
        float originX = float(column * layout.cellSize + gutter);
        float originY = float(row * layout.cellSize + gutter);
        layout.rects[index * 6 + 0] = originX / float(layout.pageWidth);
        layout.rects[index * 6 + 1] = originY / float(layout.pageHeight);
        layout.rects[index * 6 + 2] = float(materials[index].width) / float(layout.pageWidth);
        layout.rects[index * 6 + 3] = float(materials[index].height) / float(layout.pageHeight);
        layout.rects[index * 6 + 4] = float(materials[index].width);
        layout.rects[index * 6 + 5] = float(materials[index].height);
    }
    return layout;
}

struct PageSet {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> albedo, surface0, surface1, normalAo;
};

PageSet buildPages(const std::vector<Material>& materials, const PageLayout& layout) {
    PageSet pages;
    pages.width = layout.pageWidth;
    pages.height = layout.pageHeight;
    const size_t bytes = size_t(pages.width) * pages.height * 4;
    pages.albedo.assign(bytes, 255);
    pages.surface0.assign(bytes, 0);
    pages.surface1.assign(bytes, 0);
    pages.normalAo.assign(bytes, 0);

    for (size_t index = 0; index < materials.size(); ++index) {
        const Material& material = materials[index];
        uint32_t column = uint32_t(index) % layout.columns;
        uint32_t row = uint32_t(index) / layout.columns;

        for (uint32_t cy = 0; cy < layout.cellSize; ++cy) {
            for (uint32_t cx = 0; cx < layout.cellSize; ++cx) {
                // Clamping the read instead of letting it run past the edge is what produces
                // the gutter: the border texel is replicated outward.
                int sx = std::clamp(int(cx) - int(layout.gutter), 0, int(material.width) - 1);
                int sy = std::clamp(int(cy) - int(layout.gutter), 0, int(material.height) - 1);
                size_t source = (size_t(sy) * material.width + sx) * 4;
                size_t destination =
                        (size_t(row * layout.cellSize + cy) * pages.width
                         + (column * layout.cellSize + cx)) * 4;
                for (int c = 0; c < 4; ++c) {
                    pages.albedo[destination + c] = material.textures.albedo[source + c];
                    pages.surface0[destination + c] = material.textures.surface0[source + c];
                    pages.surface1[destination + c] = material.textures.surface1[source + c];
                    pages.normalAo[destination + c] = material.textures.normalAo[source + c];
                }
            }
        }
    }
    return pages;
}

// --------------------------------------------------------------------------------------
// scene
// --------------------------------------------------------------------------------------

struct Scene {
    std::vector<float> positions;      // 3 per vertex
    std::vector<float> normals;        // 3 per vertex
    std::vector<float> uvs;            // 2 per vertex
    std::vector<uint32_t> indices;
    std::vector<uint32_t> primSlots;   // 1 per triangle
    std::vector<float> materialRects;     // 6 per material: uv origin, uv size, texel count
    std::vector<uint32_t> materialFlags;  // 1 per material; bit 0 = _n carries a height field
};

uint32_t addVertex(Scene& scene, Vec3 p, Vec3 n, float u, float v) {
    uint32_t index = uint32_t(scene.positions.size() / 3);
    scene.positions.insert(scene.positions.end(), {p.x, p.y, p.z});
    scene.normals.insert(scene.normals.end(), {n.x, n.y, n.z});
    scene.uvs.insert(scene.uvs.end(), {u, v});
    return index;
}

void addTriangle(Scene& scene, uint32_t a, uint32_t b, uint32_t c, uint32_t slot) {
    scene.indices.insert(scene.indices.end(), {a, b, c});
    scene.primSlots.push_back(slot);
}

// A plate facing +Z. Its UV runs along the same axes as the geometric basis the shader
// builds, which makes that basis exact here rather than merely plausible.
void addPlate(Scene& scene, Vec3 centre, float halfSize, uint32_t slot) {
    Vec3 n{0.0f, 0.0f, 1.0f};
    uint32_t a = addVertex(scene, {centre.x - halfSize, centre.y - halfSize, centre.z}, n, 0.0f, 1.0f);
    uint32_t b = addVertex(scene, {centre.x + halfSize, centre.y - halfSize, centre.z}, n, 1.0f, 1.0f);
    uint32_t c = addVertex(scene, {centre.x + halfSize, centre.y + halfSize, centre.z}, n, 1.0f, 0.0f);
    uint32_t d = addVertex(scene, {centre.x - halfSize, centre.y + halfSize, centre.z}, n, 0.0f, 0.0f);
    addTriangle(scene, a, b, c, slot);
    addTriangle(scene, a, c, d, slot);
}

// Equirectangular UVs; j runs to segments inclusive so the seam vertex is duplicated rather
// than wrapped, which keeps the interpolated UV monotonic across every quad.
void addSphere(Scene& scene, Vec3 centre, float radius, int segments, int rings, uint32_t slot) {
    uint32_t base = uint32_t(scene.positions.size() / 3);
    for (int i = 0; i <= rings; ++i) {
        float theta = kPi * float(i) / float(rings);
        for (int j = 0; j <= segments; ++j) {
            float phi = 2.0f * kPi * float(j) / float(segments);
            Vec3 dir{std::sin(theta) * std::cos(phi), std::cos(theta),
                     std::sin(theta) * std::sin(phi)};
            addVertex(scene, centre + dir * radius, dir, float(j) / float(segments),
                      float(i) / float(rings));
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
                addTriangle(scene, v00, v10, v11, slot);
            }
            if (i != rings - 1) {
                addTriangle(scene, v00, v11, v01, slot);
            }
        }
    }
}

struct Camera {
    Vec3 position;
    Vec3 target;
};

void buildScene(Scene& scene, const std::vector<Material>& materials, const PageLayout& layout,
                const Options& options, Camera& camera) {
    scene.materialRects = layout.rects;
    // Only materials whose _n actually carries a height field get the parallax flag; the shader
    // uses it to skip the march entirely for everything else.
    scene.materialFlags.assign(materials.size(), 0u);
    for (size_t i = 0; i < materials.size(); ++i) {
        if (materials[i].hasHeight) {
            scene.materialFlags[i] |= 1u;
        }
    }

    const size_t count = std::max<size_t>(1, materials.size());
    const float spacing = 1.35f;
    for (size_t index = 0; index < count; ++index) {
        uint32_t column = uint32_t(index) % layout.columns;
        uint32_t row = uint32_t(index) / layout.columns;
        float x = (float(column) - float(layout.columns - 1) * 0.5f) * spacing;
        float y = (float(layout.rows - 1) * 0.5f - float(row)) * spacing;
        Vec3 centre{x, y, 0.0f};
        if (options.mode == "sphere") {
            addSphere(scene, centre, 0.6f, 48, 24, uint32_t(index));
        } else {
            addPlate(scene, centre, 0.55f, uint32_t(index));
        }
    }

    float extent = float(std::max(layout.columns, layout.rows)) * spacing;
    float distance = extent * 1.15f + 1.6f;
    camera.target = {0.0f, 0.0f, 0.0f};
    // The tilt is what makes a parallax march visible at all: viewed head-on the view ray is
    // parallel to the height field's own axis, so the march never moves the sample. A little
    // elevation on top keeps a normal map reading as relief rather than as a flat tint.
    float tilt = options.tilt * kPi / 180.0f;
    // The elevation scales with the distance too, so zooming in magnifies the surface rather
    // than changing the angle it is viewed from.
    distance /= options.zoom;
    camera.position = {std::sin(tilt) * distance, extent * 0.14f / options.zoom,
                       std::cos(tilt) * distance};
}

Vec3 defaultSunDirection(const Camera& camera) {
    // Place the sun near the plate's specular reflection direction so the highlight lands
    // where the camera can see it, then bias it off-axis so surfaces are lit at an angle
    // rather than head-on.
    Vec3 normal{0.0f, 0.0f, 1.0f};
    Vec3 viewDirection = normalize(camera.target - camera.position);
    Vec3 reflected = viewDirection - normal * (2.0f * dot(viewDirection, normal));
    return normalize(reflected + Vec3{0.28f, 0.40f, 0.0f});
}

Vec3 parseSunDirection(const std::string& text, const Camera& camera) {
    if (text.empty()) {
        return defaultSunDirection(camera);
    }
    float x = 0.0f, y = 0.0f, z = 0.0f;
    if (std::sscanf(text.c_str(), "%f,%f,%f", &x, &y, &z) != 3) {
        fail("--sun-dir wants X,Y,Z");
    }
    Vec3 direction{x, y, z};
    if (lengthOf(direction) < 1.0e-6f) {
        fail("--sun-dir must not be the zero vector");
    }
    return normalize(direction);
}

// --------------------------------------------------------------------------------------
// uniforms (std140, mirrors the block in matprobe.comp)
// --------------------------------------------------------------------------------------

struct FrameUniforms {
    float camPos[4];
    float camRight[4];
    float camUp[4];
    float camFwd[4];
    float renderSize[4];
    float sunDir[4];
    float sunColor[4];
    float tuning[4];
    float march[4];
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
        (std::filesystem::path(exeDirectory()) / "matprobe.comp.spv").string(),
        "matprobe.comp.spv",
        (std::filesystem::path(exeDirectory()) / ".." / "matprobe.comp.spv").string(),
        (std::filesystem::path(exeDirectory()) / ".." / ".." / "matprobe.comp.spv").string(),
    };
    for (const std::string& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    fail("cannot find matprobe.comp.spv; pass --shader");
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

std::string md5OfFile(const std::string& path) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_MD5_ALGORITHM, nullptr, 0) < 0) {
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
                static const char* digits = "0123456789abcdef";
                for (unsigned char byte : digest) {
                    result.push_back(digits[byte >> 4]);
                    result.push_back(digits[byte & 0xF]);
                }
            }
        }
        BCryptDestroyHash(hash);
    }
    BCryptCloseAlgorithmProvider(algorithm, 0);
    return result;
}

// --------------------------------------------------------------------------------------
// acceleration structures
// --------------------------------------------------------------------------------------

struct Accel {
    Buffer blasBuffer, tlasBuffer, scratch, vertexBuffer, indexBuffer, instanceBuffer;
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    uint32_t triangleCount = 0;
};

void buildAcceleration(Probe& probe, const Scene& scene, Accel& accel) {
    accel.triangleCount = uint32_t(scene.indices.size() / 3);

    const VkBufferUsageFlags asInput = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                     | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    accel.vertexBuffer = createBuffer(probe, scene.positions.size() * sizeof(float), asInput,
                                      hostVisible, true);
    uploadBuffer(probe, accel.vertexBuffer, scene.positions.data(),
                 scene.positions.size() * sizeof(float));

    accel.indexBuffer = createBuffer(probe, scene.indices.size() * sizeof(uint32_t), asInput,
                                     hostVisible, true);
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

    // The scratch offset carries a device-specific alignment requirement, so over-allocate and
    // round the address up rather than assuming the buffer's own alignment is enough.
    accel.scratch = createBuffer(probe, sizes.buildScratchSize + probe.scratchAlignment,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                         | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = accel.blas;
    VkDeviceAddress blasAddress = probe.getAccelerationStructureDeviceAddress(probe.device, &addressInfo);

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
                                        hostVisible, true);
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

    // Both builds need scratch, and the top-level build must not start before the bottom-level
    // one has finished.
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

// --------------------------------------------------------------------------------------
// readback
// --------------------------------------------------------------------------------------

FloatImage readFloatImage(const Probe& probe, const Image& image) {
    FloatImage out(image.width, image.height);
    VkDeviceSize bytes = VkDeviceSize(image.width) * image.height * 8;

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
            float* destination = out.pixel(x, y);
            for (int c = 0; c < 3; ++c) {
                uint16_t half = 0;
                std::memcpy(&half, raw + pixel * 8 + size_t(c) * 2, sizeof(half));
                destination[c] = halfToFloat(half);
            }
        }
    }
    destroyBuffer(probe, staging);
    return out;
}

// --------------------------------------------------------------------------------------
// report
// --------------------------------------------------------------------------------------

void writeStats(std::ostringstream& out, const char* indent, const char* name,
                const ChannelStats& stats) {
    out << indent << "\"" << name << "\": {\"min\": " << roundTo(stats.min, 5)
        << ", \"mean\": " << roundTo(stats.mean, 5)
        << ", \"max\": " << roundTo(stats.max, 5) << "}";
}

std::string buildReport(const Options& options, const std::vector<Material>& materials,
                        const PageLayout& layout, const std::string& deviceName,
                        const std::string& driver, const std::string& apiVersion,
                        const Vec3& sunDirection, int debugChannel,
                        const std::string& previewName, const std::string& previewMd5,
                        const std::vector<std::string>& notes) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"tool\": \"matprobe\",\n";
    out << "  \"device\": \"" << jsonEscape(deviceName) << "\",\n";
    out << "  \"driver\": \"" << driver << "\",\n";
    out << "  \"apiVersion\": \"" << apiVersion << "\",\n";
    out << "  \"pack\": \"" << jsonEscape(options.packDir) << "\",\n";
    out << "  \"mode\": \"" << options.mode << "\",\n";
    out << "  \"display\": [" << options.displayWidth << ", " << options.displayHeight << "],\n";
    out << "  \"samples\": " << options.samples << ",\n";
    out << "  \"repeat\": " << roundTo(options.repeat, 4) << ",\n";
    out << "  \"sunIntensity\": " << roundTo(options.sun, 4) << ",\n";
    out << "  \"skyIntensity\": " << roundTo(options.sky, 4) << ",\n";
    out << "  \"emissionScale\": " << roundTo(options.emission, 4) << ",\n";
    out << "  \"parallaxDepth\": " << roundTo(options.parallax, 4) << ",\n";
    out << "  \"parallaxLayers\": " << options.layers << ",\n";
    out << "  \"parallaxMinVz\": " << roundTo(options.minVz, 4) << ",\n";
    out << "  \"clampSlope\": " << (options.clampSlope ? "true" : "false") << ",\n";
    out << "  \"texelSnap\": " << (options.texelSnap ? "true" : "false") << ",\n";
    out << "  \"stepNormal\": " << (options.stepNormal ? "true" : "false") << ",\n";
    out << "  \"linearPages\": " << (options.linearPages ? "true" : "false") << ",\n";
    out << "  \"tiltDegrees\": " << roundTo(options.tilt, 2) << ",\n";
    out << "  \"texelFilter\": \"" << jsonEscape(options.filter) << "\",\n";
    out << "  \"zoom\": " << roundTo(options.zoom, 3) << ",\n";
    out << "  \"sunDirection\": [" << roundTo(sunDirection.x, 4) << ", "
        << roundTo(sunDirection.y, 4) << ", " << roundTo(sunDirection.z, 4) << "],\n";
    out << "  \"debugChannel\": \"" << jsonEscape(options.debugChannel) << "\",\n";
    out << "  \"page\": {\"width\": " << layout.pageWidth << ", \"height\": " << layout.pageHeight
        << ", \"cell\": " << layout.cellSize << ", \"gutter\": " << layout.gutter
        << ", \"columns\": " << layout.columns << ", \"rows\": " << layout.rows << "},\n";

    out << "  \"materials\": [\n";
    for (size_t index = 0; index < materials.size(); ++index) {
        const Material& material = materials[index];
        const Diagnostics& diagnostics = material.diagnostics;
        out << "    {\n";
        out << "      \"name\": \"" << jsonEscape(material.name) << "\",\n";
        out << "      \"size\": [" << material.width << ", " << material.height << "],\n";
        out << "      \"hasSpec\": " << (material.hasSpec ? "true" : "false") << ",\n";
        out << "      \"hasNormal\": " << (material.hasNormal ? "true" : "false") << ",\n";
        out << "      \"hasHeight\": " << (material.hasHeight ? "true" : "false") << ",\n";
        out << "      \"paths\": {\"albedo\": \"" << jsonEscape(material.albedoPath)
            << "\", \"spec\": \"" << jsonEscape(material.specPath)
            << "\", \"normal\": \"" << jsonEscape(material.normalPath) << "\"},\n";

        out << "      \"channels\": {\n";
        out << "        ";
        writeStats(out, "", "roughness", diagnostics.roughness);
        out << ",\n        ";
        writeStats(out, "", "metalness", diagnostics.metalness);
        out << ",\n        ";
        writeStats(out, "", "emission", diagnostics.emission);
        out << ",\n        ";
        writeStats(out, "", "sss", diagnostics.sss);
        out << ",\n        ";
        writeStats(out, "", "porosity", diagnostics.porosity);
        out << ",\n        ";
        writeStats(out, "", "height", diagnostics.height);
        out << ",\n        ";
        writeStats(out, "", "f0r", diagnostics.f0r);
        out << ",\n        ";
        writeStats(out, "", "f0g", diagnostics.f0g);
        out << ",\n        ";
        writeStats(out, "", "f0b", diagnostics.f0b);
        out << ",\n        ";
        writeStats(out, "", "albedoLinearR", diagnostics.albedoR);
        out << ",\n        ";
        writeStats(out, "", "albedoLinearG", diagnostics.albedoG);
        out << ",\n        ";
        writeStats(out, "", "albedoLinearB", diagnostics.albedoB);
        out << "\n      },\n";

        out << "      \"classification\": {\n";
        out << "        \"texels\": " << diagnostics.texelCount << ",\n";
        out << "        \"dielectric\": " << diagnostics.dielectricTexels << ",\n";
        out << "        \"metalEnum\": " << diagnostics.metalEnumTexels << ",\n";
        out << "        \"metalAlbedoF0\": " << diagnostics.metalAlbedoF0Texels << ",\n";
        out << "        \"emissive\": " << diagnostics.emissiveTexels << ",\n";
        out << "        \"sss\": " << diagnostics.sssTexels << ",\n";
        out << "        \"porosity\": " << diagnostics.porosityTexels << ",\n";
        out << "        \"nonFlushHeight\": " << diagnostics.nonFlushHeightTexels << "\n";
        out << "      },\n";

        out << "      \"metals\": [";
        for (size_t m = 0; m < diagnostics.metalNames.size(); ++m) {
            out << (m ? ", " : "") << "\"" << diagnostics.metalNames[m] << "\"";
        }
        out << "],\n";
        out << "      \"normalDeviationMean\": "
            << roundTo(diagnostics.normalDeviationMean, 5) << ",\n";

        out << "      \"warnings\": [";
        for (size_t w = 0; w < diagnostics.warnings.size(); ++w) {
            out << (w ? ", " : "") << "\"" << jsonEscape(diagnostics.warnings[w]) << "\"";
        }
        out << "]\n";
        out << "    }" << (index + 1 < materials.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"artifacts\": {\"preview\": \"" << jsonEscape(previewName)
        << "\", \"previewMd5\": \"" << previewMd5 << "\"},\n";
    out << "  \"notes\": [";
    for (size_t i = 0; i < notes.size(); ++i) {
        out << (i ? ", " : "") << "\"" << jsonEscape(notes[i]) << "\"";
    }
    out << "]\n";
    out << "}\n";
    return out.str();
}

std::string statsSummary(const ChannelStats& stats) {
    std::ostringstream out;
    out << roundTo(stats.min, 3) << " / " << roundTo(stats.mean, 3) << " / "
        << roundTo(stats.max, 3);
    return out.str();
}

}  // namespace

// --------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        SetConsoleOutputCP(CP_UTF8);
        Options options = parseArgs(argc, argv);

        if (options.listGpus) {
            std::vector<std::string> devices = enumeratePhysicalDevices();
            for (size_t i = 0; i < devices.size(); ++i) {
                std::printf("[%zu] %s\n", i, devices[i].c_str());
            }
            return 0;
        }

        if (options.listMaterials) {
            if (options.packDir.empty()) {
                fail("--list-materials needs --pack");
            }
            std::string assetsRoot = findAssetsRoot(options.packDir);
            for (const std::string& name : listPbrMaterials(assetsRoot)) {
                std::printf("%s\n", name.c_str());
            }
            return 0;
        }

        std::vector<MaterialRequest> requests = collectRequests(options);
        if (requests.empty()) {
            fail("no material given: use --material ns:path or --albedo file (see --help)");
        }

        std::vector<Material> materials;
        materials.reserve(requests.size());
        for (const MaterialRequest& request : requests) {
            materials.push_back(decodeMaterial(request.name, request.albedoPath,
                                               request.specPath, request.normalPath));
        }

        PageLayout layout = layoutPages(materials, 4);
        PageSet pages = buildPages(materials, layout);

        Camera camera;
        Scene scene;
        buildScene(scene, materials, layout, options, camera);
        Vec3 sunDirection = parseSunDirection(options.sunDirection, camera);
        const int debugChannel = debugChannelIndex(options.debugChannel);

        std::filesystem::create_directories(options.outDir);
        std::string shaderPath = resolveShaderPath(options);

        // --- device ---------------------------------------------------------------------
        Probe probe;
        ProbeConfig config;
        config.gpuIndex = options.gpuIndex;
        config.validation = options.validation;
        std::vector<std::string> notes;
        std::string deviceName = createProbe(probe, config, notes);

        // --- textures -------------------------------------------------------------------
        const VkImageUsageFlags textureUsage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        Image albedoPage = createImage(probe, pages.width, pages.height,
                                       VK_FORMAT_R8G8B8A8_UNORM, textureUsage);
        Image surface0Page = createImage(probe, pages.width, pages.height,
                                         VK_FORMAT_R8G8B8A8_UNORM, textureUsage);
        Image surface1Page = createImage(probe, pages.width, pages.height,
                                         VK_FORMAT_R8G8B8A8_UNORM, textureUsage);
        Image normalAoPage = createImage(probe, pages.width, pages.height,
                                         VK_FORMAT_R8G8B8A8_UNORM, textureUsage);
        // Plates tile, so the sampler repeats; that also means a texel at the page edge is
        // never read, which is why no separate clamp sampler is needed.
        // The game binds its block atlas AND its material pages with one shared NEAREST sampler
        // (RtComposite.atlasSampler: mag/min NEAREST, mipmap LINEAR, REPEAT), so nearest is what
        // reproduces what actually reaches the screen. Linear is kept for reading the decoded
        // fields themselves, where a smooth interpolation is easier to judge than texel steps.
        VkSampler sampler = createSampler(probe, options.filter == "nearest", true);
        // The march compares the height field against a continuously advancing depth. With
        // NEAREST the surface depth jumps between whole texels, so the march stops a whole step
        // early or late depending on where its samples land — which is what shows up as banding
        // at grazing angles. All three canonical pages share one sampler in the game, so this
        // switches all three and leaves the albedo on the atlas sampler.
        VkSampler pageSampler = options.linearPages
                ? createSampler(probe, false, true) : sampler;
        for (const Image* image : {&albedoPage, &surface0Page, &surface1Page, &normalAoPage}) {
            transitionToGeneral(probe, *image);
        }
        const VkDeviceSize pageBytes = VkDeviceSize(pages.width) * pages.height * 4;
        uploadImage(probe, albedoPage, pages.albedo.data(), pageBytes);
        uploadImage(probe, surface0Page, pages.surface0.data(), pageBytes);
        uploadImage(probe, surface1Page, pages.surface1.data(), pageBytes);
        uploadImage(probe, normalAoPage, pages.normalAo.data(), pageBytes);

        // --- output image ---------------------------------------------------------------
        const VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_STORAGE_BIT
                                           | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                           | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        Image color = createImage(probe, options.displayWidth, options.displayHeight,
                                  VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage);
        transitionToGeneral(probe, color);

        // --- acceleration ---------------------------------------------------------------
        Accel accel;
        buildAcceleration(probe, scene, accel);

        // --- descriptors -----------------------------------------------------------------
        VkDescriptorSetLayoutBinding bindings[14]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[2 + i].binding = 2 + i;
            bindings[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[2 + i].descriptorCount = 1;
            bindings[2 + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[6].binding = 6;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t i = 0; i < 7; ++i) {
            bindings[7 + i].binding = 7 + i;
            bindings[7 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[7 + i].descriptorCount = 1;
            bindings[7 + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 14;
        layoutInfo.pBindings = bindings;
        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(probe.device, &layoutInfo, nullptr, &descriptorLayout) != VK_SUCCESS) {
            fail("vkCreateDescriptorSetLayout failed");
        }

        VkDescriptorPoolSize poolSizes[5] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},
        };
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 5;
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
        const VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        Buffer uniformBuffer = createBuffer(probe, sizeof(FrameUniforms),
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostVisible, false);
        auto storageBufferFor = [&](const void* data, size_t bytes, size_t minimumBytes) {
            size_t size = std::max(bytes, minimumBytes);
            Buffer buffer = createBuffer(probe, size,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         hostVisible, false);
            std::vector<unsigned char> padded(size, 0);
            std::memcpy(padded.data(), data, bytes);
            uploadBuffer(probe, buffer, padded.data(), padded.size());
            return buffer;
        };
        Buffer normalBuffer = storageBufferFor(scene.normals.data(),
                                               scene.normals.size() * sizeof(float), 12);
        Buffer uvBuffer = storageBufferFor(scene.uvs.data(), scene.uvs.size() * sizeof(float), 8);
        Buffer slotBuffer = storageBufferFor(scene.primSlots.data(),
                                             scene.primSlots.size() * sizeof(uint32_t), 4);
        Buffer rectBuffer = storageBufferFor(scene.materialRects.data(),
                                             scene.materialRects.size() * sizeof(float), 16);
        Buffer flagBuffer = storageBufferFor(scene.materialFlags.data(),
                                             scene.materialFlags.size() * sizeof(uint32_t), 4);

        auto writeDescriptor = [&](uint32_t binding, VkDescriptorType type, VkBuffer buffer,
                                   VkImageView view, VkSampler imageSampler, VkImageLayout imageLayout) {
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = descriptorSet;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = type;
            VkDescriptorBufferInfo bufferInfo{buffer, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo imageInfo{imageSampler, view, imageLayout};
            // Which union member the driver reads is decided by the descriptor type, not by
            // inspection: image descriptors carry a view+sampler, everything else a buffer
            // range. Getting this backwards leaves the member the driver will read as null.
            if (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    || type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                write.pImageInfo = &imageInfo;
            } else {
                write.pBufferInfo = &bufferInfo;
            }
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
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_NULL_HANDLE, albedoPage.view,
                        sampler, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptor(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_NULL_HANDLE, surface0Page.view,
                        pageSampler, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptor(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_NULL_HANDLE, surface1Page.view,
                        pageSampler, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptor(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_NULL_HANDLE, normalAoPage.view,
                        pageSampler, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptor(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, color.view,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptor(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, accel.vertexBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, normalBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uvBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, accel.indexBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, slotBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rectBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
        writeDescriptor(13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, flagBuffer.buffer, VK_NULL_HANDLE,
                        VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);

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

        // --- uniforms + dispatch --------------------------------------------------------
        FrameUniforms uniforms{};
        Vec3 forward = normalize(camera.target - camera.position);
        Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
        Vec3 up = cross(right, forward);
        float aspect = float(options.displayWidth) / float(options.displayHeight);
        float tanHalfFovY = std::tan(options.fovY * kPi / 360.0f);

        uniforms.camPos[0] = camera.position.x;
        uniforms.camPos[1] = camera.position.y;
        uniforms.camPos[2] = camera.position.z;
        uniforms.camPos[3] = 0.0f;
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
        uniforms.renderSize[0] = float(options.displayWidth);
        uniforms.renderSize[1] = float(options.displayHeight);
        uniforms.renderSize[2] = float(options.samples);
        uniforms.renderSize[3] = float(debugChannel);
        uniforms.sunDir[0] = sunDirection.x;
        uniforms.sunDir[1] = sunDirection.y;
        uniforms.sunDir[2] = sunDirection.z;
        uniforms.sunDir[3] = options.sun;
        uniforms.sunColor[0] = 1.0f;
        uniforms.sunColor[1] = 0.96f;
        uniforms.sunColor[2] = 0.90f;
        uniforms.sunColor[3] = options.sky;
        uniforms.tuning[0] = options.repeat;
        uniforms.tuning[1] = options.emission;
        uniforms.tuning[2] = 1.0f;
        uniforms.tuning[3] = options.parallax;
        uniforms.march[0] = float(options.layers);
        uniforms.march[1] = options.minVz;
        uniforms.march[2] = float((options.texelSnap ? 1 : 0) | (options.stepNormal ? 2 : 0)
                                  | (options.clampSlope ? 4 : 0));
        uniforms.march[3] = 0.0f;
        std::memcpy(uniformBuffer.mapped, &uniforms, sizeof(uniforms));

        auto runStart = std::chrono::steady_clock::now();
        {
            VkCommandBuffer commandBuffer = beginCommands(probe);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                                    1, &descriptorSet, 0, nullptr);
            uint32_t groupCountX = (options.displayWidth + 7) / 8;
            uint32_t groupCountY = (options.displayHeight + 7) / 8;
            vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
            memoryBarrier(commandBuffer);
            submitCommands(probe, commandBuffer);
        }
        double elapsedMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - runStart).count();

        // --- readback + write ----------------------------------------------------------
        FloatImage result = readFloatImage(probe, color);
        std::vector<uint8_t> rgb =
                debugChannel > 0 ? mapToRgb8(result, 0.0f, 1.0f) : tonemapToRgb8(result, 1.0f, 0.18f);

        std::string previewName = debugChannel > 0
                ? "debug-" + options.debugChannel + ".png" : "preview.png";
        std::filesystem::path previewPath = std::filesystem::path(options.outDir) / previewName;
        if (!writePng(previewPath.string(), result.width, result.height, rgb)) {
            fail("cannot write " + previewPath.string());
        }

        std::vector<std::string> reportNotes = notes;
        reportNotes.push_back("decoded with the same LabPBR rule as RtLabPbr.decodeSpec; the shader "
                              "samples these channels rather than re-decoding");
        reportNotes.push_back("preview is scene-linear radiance through a log-average Reinhard "
                              "tonemap; debug views are written with no tone curve");

        std::string report = buildReport(options, materials, layout, deviceName,
                                         driverVersionString(probe.deviceProps.driverVersion),
                                         apiVersionString(probe.deviceProps.apiVersion),
                                         sunDirection, debugChannel, previewName,
                                         md5OfFile(previewPath.string()), reportNotes);
        std::filesystem::path reportPath = std::filesystem::path(options.outDir) / "report.json";
        {
            std::ofstream file(reportPath, std::ios::binary);
            if (!file) {
                fail("cannot write " + reportPath.string());
            }
            file << report;
        }

        if (options.json) {
            std::printf("%s", report.c_str());
        }
        if (!options.quiet) {
            std::printf("matprobe: %s\n", deviceName.c_str());
            std::printf("  materials:");
            for (const Material& material : materials) {
                std::printf(" %s", material.name.c_str());
            }
            std::printf("\n");
            std::printf("  render    %ux%u, %d spp, %s mode, %.0f ms\n", options.displayWidth,
                        options.displayHeight, options.samples, options.mode.c_str(), elapsedMs);
            std::printf("  parallax  %.3f%s, tilt %.0f deg\n", options.parallax,
                        options.parallax > 0.0f ? "" : " (off)", options.tilt);
            for (const Material& material : materials) {
                const Diagnostics& d = material.diagnostics;
                std::printf("  %-32s %ux%u  rough %s  metal %s%s%s\n", material.name.c_str(),
                            material.width, material.height,
                            statsSummary(d.roughness).c_str(), statsSummary(d.metalness).c_str(),
                            d.emissiveTexels ? "  emissive" : "",
                            d.sssTexels ? "  sss" : "");
                for (const std::string& warning : d.warnings) {
                    std::printf("    ! %s\n", warning.c_str());
                }
            }
            std::printf("  wrote     %s\n", previewPath.string().c_str());
            std::printf("  wrote     %s\n", reportPath.string().c_str());
        }

        if (pageSampler != sampler) {
            destroySampler(probe, pageSampler);
        }
        destroySampler(probe, sampler);
        destroyProbe(probe);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "matprobe: %s\n", error.what());
        return 1;
    }
}
