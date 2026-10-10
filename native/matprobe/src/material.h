// LabPBR 1.3 decode and resource-pack lookup for matprobe.
//
// The decode below is a deliberate re-implementation of RtLabPbr.decodeSpec (Java), which
// RtBlockMaterials runs at pack load to canonicalise materials before upload. That CPU path is
// the single authority for what a LabPBR byte means: closest_hit.rchit.slang no longer decodes
// anything, it samples the physical channels this produces. Keeping one decode rule is what
// makes a result here transfer to the game instead of describing a second, parallel material
// model.
//
// Channel meanings this file implements, all from LabPBR 1.3:
//   _s red    perceptual smoothness -> linear GGX alpha = (1 - smoothness)^2
//   _s green  < 229.5 dielectric F0; 229.5..237.5 one of 8 conductor enums; > 237.5 F0 = albedo
//   _s blue   0..64 porosity (water absorption); 65..255 subsurface scattering (one lane)
//   _s alpha  0..254 emission strength; 255 means "authored no emission"
//   _n rgb    tangent-space normal xy; blue is ambient occlusion
//   _n alpha  height as a depth: 255 is flush, 0 is 25% of a block deep

#pragma once

#include "image_io.h"

#include <cstdint>
#include <string>
#include <vector>

namespace matprobe {

/** LabPBR's conductor enum, indexed by (greenByte - 230). */
extern const char* const kMetalNames[8];

/** One texel's decoded physical surface parameters. Mirrors RtLabPbr.Specular field for field. */
struct Specular {
    float roughness = 0.0f;   // linear GGX alpha, never squared again downstream
    float metalness = 0.0f;
    float f0r = 0.0f, f0g = 0.0f, f0b = 0.0f;
    float emission = 0.0f;    // 0..1 linear
    float sss = 0.0f;         // 0..1, non-zero only when the blue byte is in 65..255
    float porosity = 0.0f;    // 0..1, non-zero only when the blue byte is in 0..64
};

/** The LabPBR decode. `red`/`green`/`blue`/`alpha` and the albedo are all normalised 0..1. */
Specular decodeLabPbr(float red, float green, float blue, float alpha,
                      float albedoR, float albedoG, float albedoB);

float clamp01(float value);

/** sRGB electro-optical transfer, matching the game's srgbToLinear. */
float srgbToLinear(float value);

/** Running min/mean/max for one decoded channel. */
struct ChannelStats {
    float min = 0.0f, mean = 0.0f, max = 0.0f;

    void add(float value);

private:
    double sum_ = 0.0;
    int64_t count_ = 0;
};

/** Everything the report says about one material's decoded channels. */
struct Diagnostics {
    int texelCount = 0;
    ChannelStats roughness, metalness, emission, sss, porosity, height;
    ChannelStats f0r, f0g, f0b;
    ChannelStats albedoR, albedoG, albedoB;
    int dielectricTexels = 0;     // green < 229.5: F0 comes from the green byte
    int metalEnumTexels = 0;      // green in (229.5, 237.5): one of the 8 conductor enums
    int metalAlbedoF0Texels = 0;  // green >= 237.5: F0 is the albedo itself
    int emissiveTexels = 0;
    int sssTexels = 0;
    int porosityTexels = 0;
    int nonFlushHeightTexels = 0;
    double normalDeviationMean = 0.0;  // mean |xy - 0.5| * 2; 0 = a perfectly flat map
    std::vector<std::string> metalNames;  // distinct conductor enums actually hit
    std::vector<std::string> warnings;
};

/** Decoded textures, one texel per source texel, laid out for a straight GPU upload. */
struct Textures {
    uint32_t width = 0, height = 0;
    // Albedo keeps its authored sRGB encoding: the shader applies srgbToLinear, which is what
    // the game does, and quantising it to linear 8-bit first would band the darks for nothing.
    std::vector<uint8_t> albedo;    // RGBA8, .rgb = sRGB, .a = 255
    std::vector<uint8_t> surface0;  // RGBA8: r=roughness g=metalness b=emission a=sss
    std::vector<uint8_t> surface1;  // RGBA8: rgb=F0 (linear) a=porosity
    std::vector<uint8_t> normalAo;  // RGBA8: rg=tangent normal xy b=AO a=height
};

struct Material {
    std::string name;
    std::string albedoPath, specPath, normalPath;
    uint32_t width = 0, height = 0;
    bool hasSpec = false;
    bool hasNormal = false;
    bool hasHeight = false;
    Textures textures;
    Diagnostics diagnostics;
};

/** Loads, decodes and measures one material. A missing _s/_n becomes the neutral default. */
Material decodeMaterial(const std::string& name, const std::string& albedoPath,
                        const std::string& specPath, const std::string& normalPath);

/** Resolves `ns:path` under a pack's assets/ root to a texture path with `suffix` appended. */
std::string resolveTexturePath(const std::string& assetsRoot, const std::string& logicalName,
                               const std::string& suffix);

/** Accepts either a pack root (containing assets/) or an assets/ directory itself. */
std::string findAssetsRoot(const std::string& packDir);

/** Every logical name in the pack that authors a _s or _n map, i.e. a real PBR material. */
std::vector<std::string> listPbrMaterials(const std::string& assetsRoot);

}  // namespace matprobe
