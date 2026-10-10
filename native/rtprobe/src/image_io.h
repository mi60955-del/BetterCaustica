// Image writers and the numeric metrics the report is built on.
//
// PNG is written by hand with stored (uncompressed) deflate blocks so the probe needs no
// zlib dependency. EXR uses the same dialect Caustica's RtOpenExrWriter emits -- single
// part, NO_COMPRESSION, HALF channels named A, B, G, R -- so the project's existing
// exr-to-png.py / inspect_exr.py can read the probe's output unchanged.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Interleaved RGB float image, 3 channels per pixel.
struct FloatImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> data;

    FloatImage() = default;
    FloatImage(uint32_t w, uint32_t h) : width(w), height(h), data(size_t(w) * h * 3, 0.0f) {}

    float* pixel(uint32_t x, uint32_t y) { return &data[(size_t(y) * width + x) * 3]; }
    const float* pixel(uint32_t x, uint32_t y) const { return &data[(size_t(y) * width + x) * 3]; }
};

// Reinhard tonemap driven by the log-average luminance, then a 1/2.2 encode. `bias` is a
// linear multiplier applied before the curve; `keyValue` is the target middle grey.
std::vector<uint8_t> tonemapToRgb8(const FloatImage& image, float bias, float keyValue);

bool writePng(const std::string& path, uint32_t width, uint32_t height,
              const std::vector<uint8_t>& rgb);

bool writeExrHalf(const std::string& path, const FloatImage& image);

// IEEE-754 binary16 to float, for reading back rgba16f / rg16f query results.
float halfToFloat(uint16_t value);

// Linear windowing to 8-bit, with no tone curve. Guide dumps are signed or already
// normalised, so running them through the display tonemap would misrepresent them.
std::vector<uint8_t> mapToRgb8(const FloatImage& image, float lo, float hi);

// --- metrics --------------------------------------------------------------------------

// Mean absolute deviation from a 3x3 box blur, averaged over channels. This is the
// "how much high-frequency noise is on screen" number; a working denoiser drops it.
double highFrequencyEnergy(const FloatImage& image);

// Mean absolute per-channel difference. Both images must have identical dimensions.
double meanAbsDifference(const FloatImage& a, const FloatImage& b);

FloatImage resampleNearest(const FloatImage& src, uint32_t width, uint32_t height);
FloatImage cropTo(const FloatImage& src, uint32_t width, uint32_t height);

struct ImageStats {
    double minLuma = 0.0;
    double meanLuma = 0.0;
    double maxLuma = 0.0;
    int64_t nonFinite = 0;
};

ImageStats computeStats(const FloatImage& image);

// Extracts one channel (0=R, 1=G, 2=B) as a single-channel-valued RGB image, for guide dumps.
FloatImage extractChannel(const FloatImage& image, int channel);
