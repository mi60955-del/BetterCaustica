#include "material.h"
#include "vk.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <set>

namespace matprobe {

const char* const kMetalNames[8] = {
        "iron", "gold", "aluminum", "chrome", "copper", "lead", "platinum", "silver"};

namespace {

// LabPBR's conductor table, indexed by the metal enum. Same values and same order as
// RtLabPbr.METAL_N / METAL_K: complex IOR n and k per channel for iron .. silver.
const float kMetalN[8][3] = {
        {2.9114f, 2.9497f, 2.5845f},
        {0.18299f, 0.42108f, 1.3734f},
        {1.3456f, 0.96521f, 0.61722f},
        {3.1071f, 3.1812f, 2.3230f},
        {0.27105f, 0.67693f, 1.3164f},
        {1.9100f, 1.8300f, 1.4400f},
        {2.3757f, 2.0847f, 1.8453f},
        {0.15943f, 0.14512f, 0.13547f}};
const float kMetalK[8][3] = {
        {3.0893f, 2.9318f, 2.7670f},
        {3.4242f, 2.3459f, 1.7704f},
        {7.4746f, 6.3995f, 5.3031f},
        {3.3314f, 3.3291f, 3.1350f},
        {3.6092f, 2.6248f, 2.2921f},
        {3.5100f, 3.4000f, 3.1800f},
        {4.2655f, 3.7153f, 3.1365f},
        {3.9291f, 3.1900f, 2.3808f}};

float metalF0(int metal, int channel) {
    float n = kMetalN[metal][channel];
    float k = kMetalK[metal][channel];
    float nm1 = n - 1.0f;
    float np1 = n + 1.0f;
    return (nm1 * nm1 + k * k) / (np1 * np1 + k * k);
}

uint8_t quantise(float value) {
    return uint8_t(std::lround(clamp01(value) * 255.0f));
}

// Nearest-neighbour fetch that tolerates a _s/_n whose extent differs from the albedo. The
// game requires matching sprite sizes; a mismatched map is reported as a warning and read
// without resampling, so the preview still shows what the texel actually says.
const uint8_t* fetchNearest(const Rgba8Image& image, uint32_t x, uint32_t y,
                            uint32_t width, uint32_t height) {
    uint32_t sx = uint32_t((uint64_t(x) * image.width) / std::max(1u, width));
    uint32_t sy = uint32_t((uint64_t(y) * image.height) / std::max(1u, height));
    return image.pixel(std::min(sx, image.width - 1), std::min(sy, image.height - 1));
}

bool endsWith(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size()
            && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lowerCase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return text;
}

}  // namespace

void ChannelStats::add(float value) {
    if (count_ == 0) {
        min = max = value;
    } else {
        min = std::min(min, value);
        max = std::max(max, value);
    }
    sum_ += double(value);
    ++count_;
    mean = float(sum_ / double(count_));
}

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float srgbToLinear(float value) {
    float c = clamp01(value);
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

Specular decodeLabPbr(float red, float green, float blue, float alpha,
                      float albedoR, float albedoG, float albedoB) {
    Specular out;
    float smoothness = clamp01(red);
    // LabPBR red is perceptual smoothness; squaring it is what turns it into linear GGX alpha.
    // The pipeline stores that value and never squares it again downstream.
    out.roughness = (1.0f - smoothness) * (1.0f - smoothness);

    float g = clamp01(green) * 255.0f;
    if (g < 229.5f) {
        // Dielectric: the green byte is F0 directly, typically ~0.04.
        out.metalness = 0.0f;
        out.f0r = out.f0g = out.f0b = clamp01(green);
    } else if (g < 237.5f) {
        // Conductor enum: the byte names one of eight metals whose real F0 the table supplies.
        out.metalness = 1.0f;
        int metal = std::clamp(int(std::lround(g)) - 230, 0, 7);
        out.f0r = metalF0(metal, 0);
        out.f0g = metalF0(metal, 1);
        out.f0b = metalF0(metal, 2);
    } else {
        // Conductor with an authored F0: the albedo itself is the reflectance.
        out.metalness = 1.0f;
        out.f0r = clamp01(albedoR);
        out.f0g = clamp01(albedoG);
        out.f0b = clamp01(albedoB);
    }

    // Alpha 255 is the author saying "no emission"; 0..254 scales the emission to 0..1.
    float a = clamp01(alpha) * 255.0f;
    out.emission = a < 254.5f ? a / 254.0f : 0.0f;

    // Blue is a single mutually exclusive lane split at byte 64: a texel encodes porosity or
    // subsurface scattering, never both, so at most one of the two is ever non-zero.
    float b = clamp01(blue) * 255.0f;
    out.porosity = b <= 64.5f ? b / 64.0f : 0.0f;
    out.sss = b > 64.5f ? (b - 65.0f) / 190.0f : 0.0f;
    return out;
}

Material decodeMaterial(const std::string& name, const std::string& albedoPath,
                        const std::string& specPath, const std::string& normalPath) {
    Material material;
    material.name = name;
    material.albedoPath = albedoPath;
    material.specPath = specPath;
    material.normalPath = normalPath;

    Rgba8Image albedoImage = decodeImageFile(albedoPath);
    const uint32_t width = albedoImage.width;
    const uint32_t height = albedoImage.height;
    material.width = width;
    material.height = height;

    std::error_code error;
    bool haveSpec = !specPath.empty() && std::filesystem::is_regular_file(specPath, error);
    bool haveNormal = !normalPath.empty() && std::filesystem::is_regular_file(normalPath, error);
    Rgba8Image specImage;
    Rgba8Image normalImage;
    if (haveSpec) {
        specImage = decodeImageFile(specPath);
    }
    if (haveNormal) {
        normalImage = decodeImageFile(normalPath);
    }
    material.hasSpec = haveSpec;
    material.hasNormal = haveNormal;

    Diagnostics& diagnostics = material.diagnostics;
    if (!haveSpec) {
        // RtMaterials.Profile.DEFAULT is the physical surface the game uses for a block with
        // no authored LabPBR map: linear GGX alpha 0.81 and a dielectric F0 of 0.04. Spelling
        // it out here keeps the fallback visible in the report rather than hiding it as a
        // zero-initialised texture.
        diagnostics.warnings.push_back(
                "no _s map: using the game's neutral default surface (roughness 0.81, F0 0.04)");
    }
    if (haveSpec && (specImage.width != width || specImage.height != height)) {
        diagnostics.warnings.push_back("_s extent differs from albedo; sampled nearest-neighbour");
    }
    if (haveNormal && (normalImage.width != width || normalImage.height != height)) {
        diagnostics.warnings.push_back("_n extent differs from albedo; sampled nearest-neighbour");
    }

    const size_t texels = size_t(width) * height;
    Textures& textures = material.textures;
    textures.width = width;
    textures.height = height;
    textures.albedo.assign(texels * 4, 255);
    textures.surface0.assign(texels * 4, 0);
    textures.surface1.assign(texels * 4, 0);
    textures.normalAo.assign(texels * 4, 0);

    diagnostics.texelCount = int(texels);
    bool metalHit[8] = {false, false, false, false, false, false, false, false};
    double normalDeviationSum = 0.0;
    bool anyHeight = false;

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t* albedo = albedoImage.pixel(x, y);
            const size_t i = (size_t(y) * width + x) * 4;

            float albedoR = albedo[0] / 255.0f;
            float albedoG = albedo[1] / 255.0f;
            float albedoB = albedo[2] / 255.0f;
            textures.albedo[i + 0] = albedo[0];
            textures.albedo[i + 1] = albedo[1];
            textures.albedo[i + 2] = albedo[2];

            Specular specular;
            if (haveSpec) {
                const uint8_t* s = fetchNearest(specImage, x, y, width, height);
                specular = decodeLabPbr(s[0] / 255.0f, s[1] / 255.0f, s[2] / 255.0f, s[3] / 255.0f,
                                        srgbToLinear(albedoR), srgbToLinear(albedoG),
                                        srgbToLinear(albedoB));
                // The classification counts re-derive the decode's own branch on the green
                // byte, so the report can state which lane a material used, not just the
                // numbers that came out of it.
                float g = float(s[1]);
                if (g < 229.5f) {
                    diagnostics.dielectricTexels++;
                } else if (g < 237.5f) {
                    diagnostics.metalEnumTexels++;
                    metalHit[std::clamp(int(std::lround(g)) - 230, 0, 7)] = true;
                } else {
                    diagnostics.metalAlbedoF0Texels++;
                }
                if (specular.emission > 0.0f) {
                    diagnostics.emissiveTexels++;
                }
                if (specular.sss > 0.0f) {
                    diagnostics.sssTexels++;
                }
                if (specular.porosity > 0.0f) {
                    diagnostics.porosityTexels++;
                }
            } else {
                specular.roughness = 0.81f;
                specular.metalness = 0.0f;
                specular.f0r = specular.f0g = specular.f0b = 0.04f;
            }

            textures.surface0[i + 0] = quantise(specular.roughness);
            textures.surface0[i + 1] = quantise(specular.metalness);
            textures.surface0[i + 2] = quantise(specular.emission);
            textures.surface0[i + 3] = quantise(specular.sss);
            textures.surface1[i + 0] = quantise(specular.f0r);
            textures.surface1[i + 1] = quantise(specular.f0g);
            textures.surface1[i + 2] = quantise(specular.f0b);
            textures.surface1[i + 3] = quantise(specular.porosity);

            // A material with no _n is flush: height 1.0, a flat normal, no occlusion. That is
            // the same default the game writes (a zero-initialised height would instead push
            // the whole surface a quarter block inward).
            float normalX = 0.5f, normalY = 0.5f, ao = 1.0f, heightByte = 1.0f;
            if (haveNormal) {
                const uint8_t* n = fetchNearest(normalImage, x, y, width, height);
                normalX = n[0] / 255.0f;
                normalY = n[1] / 255.0f;
                ao = n[2] / 255.0f;
                heightByte = n[3] / 255.0f;
                if (heightByte < 1.0f) {
                    diagnostics.nonFlushHeightTexels++;
                    anyHeight = true;
                }
                normalDeviationSum += (std::fabs(normalX * 2.0f - 1.0f)
                                       + std::fabs(normalY * 2.0f - 1.0f)) * 0.5;
            }
            textures.normalAo[i + 0] = quantise(normalX);
            textures.normalAo[i + 1] = quantise(normalY);
            textures.normalAo[i + 2] = quantise(ao);
            textures.normalAo[i + 3] = quantise(heightByte);

            diagnostics.roughness.add(specular.roughness);
            diagnostics.metalness.add(specular.metalness);
            diagnostics.emission.add(specular.emission);
            diagnostics.sss.add(specular.sss);
            diagnostics.porosity.add(specular.porosity);
            diagnostics.height.add(heightByte);
            diagnostics.f0r.add(specular.f0r);
            diagnostics.f0g.add(specular.f0g);
            diagnostics.f0b.add(specular.f0b);
            diagnostics.albedoR.add(srgbToLinear(albedoR));
            diagnostics.albedoG.add(srgbToLinear(albedoG));
            diagnostics.albedoB.add(srgbToLinear(albedoB));
        }
    }

    material.hasHeight = anyHeight;
    diagnostics.normalDeviationMean = texels ? normalDeviationSum / double(texels) : 0.0;
    for (int m = 0; m < 8; ++m) {
        if (metalHit[m]) {
            diagnostics.metalNames.push_back(kMetalNames[m]);
        }
    }
    if (!haveSpec && !haveNormal) {
        diagnostics.warnings.push_back("no LabPBR maps at all: this is a plain albedo texture");
    }
    return material;
}

std::string resolveTexturePath(const std::string& assetsRoot, const std::string& logicalName,
                               const std::string& suffix) {
    std::string namespaceName = "minecraft";
    std::string path = logicalName;
    size_t colon = logicalName.find(':');
    if (colon != std::string::npos) {
        namespaceName = logicalName.substr(0, colon);
        path = logicalName.substr(colon + 1);
    }
    // Accept a name that already carries the textures/ root so both `block/stone` and
    // `textures/block/stone` resolve to the same file.
    if (path.rfind("textures/", 0) == 0) {
        path = path.substr(9);
    }
    std::filesystem::path resolved =
            std::filesystem::path(assetsRoot) / namespaceName / "textures" / (path + suffix);
    return resolved.string();
}

std::string findAssetsRoot(const std::string& packDir) {
    std::filesystem::path root(packDir);
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) {
        fail("--pack is not a directory: " + packDir);
    }
    if (root.filename() == "assets") {
        return root.string();
    }
    std::filesystem::path nested = root / "assets";
    if (std::filesystem::is_directory(nested, error)) {
        return nested.string();
    }
    fail("no assets/ directory under " + packDir
         + "; point --pack at a resource-pack root or at its assets/");
}

std::vector<std::string> listPbrMaterials(const std::string& assetsRoot) {
    std::set<std::string> unique;
    std::error_code error;
    std::filesystem::path root(assetsRoot);
    if (!std::filesystem::is_directory(root, error)) {
        return {};
    }
    const std::string rootText = root.generic_string();

    for (const std::filesystem::directory_entry& entry :
         std::filesystem::recursive_directory_iterator(root, error)) {
        if (error || !entry.is_regular_file()) {
            continue;
        }
        std::string filename = entry.path().filename().string();
        std::string lower = lowerCase(filename);
        // _s and _n are the only suffixes that make a texture a material input; the base
        // albedo has to exist or the map is dead weight, which is also why the game drops it.
        if (!endsWith(lower, "_s.png") && !endsWith(lower, "_n.png")) {
            continue;
        }
        std::filesystem::path basePath =
                entry.path().parent_path() / (filename.substr(0, filename.size() - 6) + ".png");
        if (!std::filesystem::is_regular_file(basePath, error)) {
            continue;
        }

        std::string relative = entry.path().generic_string();
        if (relative.rfind(rootText, 0) != 0) {
            continue;
        }
        relative = relative.substr(rootText.size());
        if (!relative.empty() && relative[0] == '/') {
            relative = relative.substr(1);
        }
        // <namespace>/textures/<path>/<name>{_s|_n}.png
        size_t namespaceEnd = relative.find('/');
        if (namespaceEnd == std::string::npos) {
            continue;
        }
        size_t texturesEnd = relative.find('/', namespaceEnd + 1);
        if (texturesEnd == std::string::npos
                || relative.substr(namespaceEnd + 1, texturesEnd - namespaceEnd - 1) != "textures") {
            continue;
        }
        std::string texturePath = relative.substr(texturesEnd + 1);
        texturePath = texturePath.substr(0, texturePath.size() - 6);
        unique.insert(relative.substr(0, namespaceEnd) + ":" + texturePath);
    }

    return std::vector<std::string>(unique.begin(), unique.end());
}

}  // namespace matprobe
