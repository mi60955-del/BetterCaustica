#include "image_io.h"
#include "vk.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>

namespace {

// --- crc32 / adler32 ------------------------------------------------------------------

uint32_t crc32Of(const uint8_t* data, size_t length, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[n] = c;
        }
        ready = true;
    }
    for (size_t i = 0; i < length; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc;
}

uint32_t adler32Of(const uint8_t* data, size_t length) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void putU32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(uint8_t(value >> 24));
    out.push_back(uint8_t(value >> 16));
    out.push_back(uint8_t(value >> 8));
    out.push_back(uint8_t(value));
}

void putU32Le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(uint8_t(value));
    out.push_back(uint8_t(value >> 8));
    out.push_back(uint8_t(value >> 16));
    out.push_back(uint8_t(value >> 24));
}

void putU64Le(std::vector<uint8_t>& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(uint8_t(value >> (8 * i)));
    }
}

void putF32Le(std::vector<uint8_t>& out, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    putU32Le(out, bits);
}

void pngChunk(std::vector<uint8_t>& out, const char type[5], const std::vector<uint8_t>& payload) {
    putU32(out, static_cast<uint32_t>(payload.size()));
    size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), payload.begin(), payload.end());
    uint32_t crc = crc32Of(out.data() + start, out.size() - start) ^ 0xFFFFFFFFu;
    putU32(out, crc);
}

// zlib stream using only stored (BTYPE=00) deflate blocks: no compression dependency.
std::vector<uint8_t> zlibStored(const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> out;
    out.push_back(0x78);  // CM=8, CINFO=7
    out.push_back(0x01);  // FCHECK so (0x78<<8|0x01) % 31 == 0
    size_t offset = 0;
    const size_t kMaxBlock = 65535;
    do {
        size_t remaining = raw.size() - offset;
        size_t block = remaining < kMaxBlock ? remaining : kMaxBlock;
        bool last = (offset + block) >= raw.size();
        out.push_back(last ? 1 : 0);
        out.push_back(uint8_t(block & 0xFF));
        out.push_back(uint8_t(block >> 8));
        out.push_back(uint8_t(~block & 0xFF));
        out.push_back(uint8_t((~block >> 8) & 0xFF));
        out.insert(out.end(), raw.begin() + offset, raw.begin() + offset + block);
        offset += block;
    } while (offset < raw.size());
    putU32(out, adler32Of(raw.data(), raw.size()));
    return out;
}

uint16_t floatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 127;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent == 128) {                                   // Inf or NaN
        return uint16_t(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    }
    int32_t halfExponent = exponent + 15;
    if (halfExponent >= 31) {
        return uint16_t(sign | 0x7C00u);                     // overflow -> Inf
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return uint16_t(sign);                           // underflow -> zero
        }
        uint32_t m = mantissa | 0x800000u;
        uint32_t shift = uint32_t(14 - halfExponent);
        uint32_t half = m >> shift;
        if ((m >> (shift - 1)) & 1u) {
            ++half;
        }
        return uint16_t(sign | half);
    }
    uint32_t half = (uint32_t(halfExponent) << 10) | (mantissa >> 13);
    if (mantissa & 0x1000u) {
        ++half;                                              // round to nearest
    }
    return uint16_t(sign | half);
}

void putAttribute(std::vector<uint8_t>& out, const char* name, const char* type,
                  const std::vector<uint8_t>& payload) {
    out.insert(out.end(), name, name + std::strlen(name) + 1);
    out.insert(out.end(), type, type + std::strlen(type) + 1);
    putU32Le(out, static_cast<uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
}

}  // namespace

std::vector<uint8_t> tonemapToRgb8(const FloatImage& image, float bias, float keyValue) {
    const size_t pixels = size_t(image.width) * image.height;

    // Log-average luminance (the Reinhard key), ignoring non-positive samples.
    double logSum = 0.0;
    size_t counted = 0;
    for (size_t i = 0; i < pixels; ++i) {
        const float* p = &image.data[i * 3];
        double luma = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        if (std::isfinite(luma) && luma > 1.0e-6) {
            logSum += std::log(luma);
            ++counted;
        }
    }
    double logAverage = counted ? std::exp(logSum / double(counted)) : 1.0;
    if (!(logAverage > 1.0e-8)) {
        logAverage = 1.0;
    }
    double scale = double(keyValue) / logAverage * double(bias);

    std::vector<uint8_t> out(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        const float* p = &image.data[i * 3];
        double r = p[0], g = p[1], b = p[2];
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
            r = g = b = 0.0;
        }
        double luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) * scale;
        // Tone map luminance and carry the chroma ratio through, so hue survives.
        double mapped = luma / (1.0 + std::max(luma, 0.0));
        double gain = luma > 1.0e-9 ? mapped / luma : 0.0;
        for (int c = 0; c < 3; ++c) {
            double v = double(p[c] < 0 ? 0.0f : p[c]) * scale * gain;
            v = std::pow(std::min(std::max(v, 0.0), 1.0), 1.0 / 2.2);
            out[i * 3 + c] = uint8_t(v * 255.0 + 0.5);
        }
    }
    return out;
}

bool writePng(const std::string& path, uint32_t width, uint32_t height,
              const std::vector<uint8_t>& rgb) {
    if (rgb.size() < size_t(width) * height * 3) {
        return false;
    }

    std::vector<uint8_t> raw;
    raw.reserve((size_t(width) * 3 + 1) * height);
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);  // filter type 0 (None)
        const uint8_t* row = rgb.data() + size_t(y) * width * 3;
        raw.insert(raw.end(), row, row + size_t(width) * 3);
    }

    std::vector<uint8_t> png{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    putU32(ihdr, width);
    putU32(ihdr, height);
    ihdr.push_back(8);   // bit depth
    ihdr.push_back(2);   // colour type: truecolour
    ihdr.push_back(0);   // compression
    ihdr.push_back(0);   // filter
    ihdr.push_back(0);   // interlace
    pngChunk(png, "IHDR", ihdr);
    pngChunk(png, "IDAT", zlibStored(raw));
    pngChunk(png, "IEND", {});

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return file.good();
}

bool writeExrHalf(const std::string& path, const FloatImage& image) {
    const uint32_t width = image.width;
    const uint32_t height = image.height;
    if (width == 0 || height == 0) {
        return false;
    }

    std::vector<uint8_t> file{0x76, 0x2F, 0x31, 0x01};
    putU32Le(file, 2);  // version 2, single part, scanline

    // Channel list, written in the lexicographic order the readers expect.
    std::vector<uint8_t> channelList;
    const char* names[4] = {"A", "B", "G", "R"};
    for (const char* name : names) {
        channelList.insert(channelList.end(), name, name + std::strlen(name) + 1);
        putU32Le(channelList, 1);  // pixelType 1 = HALF
        channelList.push_back(0);  // pLinear
        channelList.push_back(0);
        channelList.push_back(0);
        channelList.push_back(0);
        putU32Le(channelList, 1);  // xSampling
        putU32Le(channelList, 1);  // ySampling
    }
    channelList.push_back(0);
    putAttribute(file, "channels", "chlist", channelList);

    putAttribute(file, "compression", "compression", {0});  // NO_COMPRESSION

    std::vector<uint8_t> window;
    putU32Le(window, 0);
    putU32Le(window, 0);
    putU32Le(window, width - 1);
    putU32Le(window, height - 1);
    putAttribute(file, "dataWindow", "box2i", window);
    putAttribute(file, "displayWindow", "box2i", window);

    putAttribute(file, "lineOrder", "lineOrder", {0});  // increasing Y

    std::vector<uint8_t> aspect;
    putF32Le(aspect, 1.0f);
    putAttribute(file, "pixelAspectRatio", "float", aspect);

    std::vector<uint8_t> centre;
    putF32Le(centre, 0.0f);
    putF32Le(centre, 0.0f);
    putAttribute(file, "screenWindowCenter", "v2f", centre);

    std::vector<uint8_t> swWidth;
    putF32Le(swWidth, 1.0f);
    putAttribute(file, "screenWindowWidth", "float", swWidth);

    file.push_back(0);  // end of header

    const size_t planeBytes = size_t(width) * 2;
    const size_t blockBytes = planeBytes * 4;
    const size_t offsetTableStart = file.size();
    file.resize(file.size() + size_t(height) * 8);
    size_t blockStart = offsetTableStart + size_t(height) * 8;

    for (uint32_t y = 0; y < height; ++y) {
        uint64_t blockOffset = blockStart + size_t(y) * (blockBytes + 8);
        size_t patch = offsetTableStart + size_t(y) * 8;
        for (int i = 0; i < 8; ++i) {
            file[patch + i] = uint8_t(blockOffset >> (8 * i));
        }

        putU32Le(file, y);
        putU32Le(file, static_cast<uint32_t>(blockBytes));

        // Alpha plane is a constant 1.0, then B, G, R.
        for (uint32_t x = 0; x < width; ++x) {
            uint16_t half = floatToHalf(1.0f);
            file.push_back(uint8_t(half & 0xFF));
            file.push_back(uint8_t(half >> 8));
        }
        for (int channel : {2, 1, 0}) {
            for (uint32_t x = 0; x < width; ++x) {
                uint16_t half = floatToHalf(image.pixel(x, y)[channel]);
                file.push_back(uint8_t(half & 0xFF));
                file.push_back(uint8_t(half >> 8));
            }
        }
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    return out.good();
}

float halfToFloat(uint16_t value) {
    uint32_t sign = uint32_t(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;                                  // signed zero
        } else {
            // Subnormal: renormalise into a float32 normal.
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3FFu;
            bits = sign | uint32_t(127 - 15 - shift) << 23 | mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | mantissa << 13;        // Inf / NaN
    } else {
        bits = sign | (exponent - 15 + 127) << 23 | mantissa << 13;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<uint8_t> mapToRgb8(const FloatImage& image, float lo, float hi) {
    std::vector<uint8_t> out(image.data.size());
    double span = double(hi) - double(lo);
    if (std::fabs(span) < 1.0e-12) {
        span = 1.0;
    }
    for (size_t i = 0; i < image.data.size(); ++i) {
        double v = (double(image.data[i]) - double(lo)) / span;
        v = std::min(std::max(v, 0.0), 1.0);
        out[i] = uint8_t(v * 255.0 + 0.5);
    }
    return out;
}

double highFrequencyEnergy(const FloatImage& image) {
    const uint32_t w = image.width;
    const uint32_t h = image.height;
    if (w < 3 || h < 3) {
        return 0.0;
    }
    double total = 0.0;
    int64_t counted = 0;
    for (uint32_t y = 1; y + 1 < h; ++y) {
        for (uint32_t x = 1; x + 1 < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        sum += image.pixel(uint32_t(int(x) + dx), uint32_t(int(y) + dy))[c];
                    }
                }
                double centre = image.pixel(x, y)[c];
                double blur = sum / 9.0;
                if (std::isfinite(centre) && std::isfinite(blur)) {
                    total += std::fabs(centre - blur);
                    ++counted;
                }
            }
        }
    }
    return counted ? total / double(counted) : 0.0;
}

double meanAbsDifference(const FloatImage& a, const FloatImage& b) {
    if (a.width != b.width || a.height != b.height || a.data.size() != b.data.size()) {
        return -1.0;
    }
    double total = 0.0;
    int64_t counted = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double x = a.data[i];
        double y = b.data[i];
        if (std::isfinite(x) && std::isfinite(y)) {
            total += std::fabs(x - y);
            ++counted;
        }
    }
    return counted ? total / double(counted) : 0.0;
}

FloatImage resampleNearest(const FloatImage& src, uint32_t width, uint32_t height) {
    FloatImage out(width, height);
    if (src.width == 0 || src.height == 0) {
        return out;
    }
    for (uint32_t y = 0; y < height; ++y) {
        uint32_t sy = uint32_t(uint64_t(y) * src.height / height);
        sy = sy < src.height ? sy : src.height - 1;
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t sx = uint32_t(uint64_t(x) * src.width / width);
            sx = sx < src.width ? sx : src.width - 1;
            const float* s = src.pixel(sx, sy);
            float* d = out.pixel(x, y);
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
    return out;
}

FloatImage cropTo(const FloatImage& src, uint32_t width, uint32_t height) {
    FloatImage out(width, height);
    uint32_t w = width < src.width ? width : src.width;
    uint32_t h = height < src.height ? height : src.height;
    for (uint32_t y = 0; y < h; ++y) {
        std::memcpy(out.pixel(0, y), src.pixel(0, y), size_t(w) * 3 * sizeof(float));
    }
    return out;
}

ImageStats computeStats(const FloatImage& image) {
    ImageStats out;
    double minLuma = 1.0e30;
    double maxLuma = -1.0e30;
    double sum = 0.0;
    int64_t counted = 0;
    for (size_t i = 0; i < image.data.size() / 3; ++i) {
        const float* p = &image.data[i * 3];
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
            ++out.nonFinite;
            continue;
        }
        double luma = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        minLuma = std::min(minLuma, luma);
        maxLuma = std::max(maxLuma, luma);
        sum += luma;
        ++counted;
    }
    out.minLuma = counted ? minLuma : 0.0;
    out.maxLuma = counted ? maxLuma : 0.0;
    out.meanLuma = counted ? sum / double(counted) : 0.0;
    return out;
}

FloatImage extractChannel(const FloatImage& image, int channel) {
    FloatImage out(image.width, image.height);
    for (size_t i = 0; i < image.data.size() / 3; ++i) {
        float v = image.data[i * 3 + channel];
        out.data[i * 3 + 0] = v;
        out.data[i * 3 + 1] = v;
        out.data[i * 3 + 2] = v;
    }
    return out;
}

// --- decode ---------------------------------------------------------------------------

namespace {

std::string hexCode(long value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%08lX", static_cast<unsigned long>(value));
    return std::string(buffer);
}

// WIC is apartment-threaded COM. matprobe is single-threaded, so one process-lifetime
// initialisation suffices; S_FALSE means COM was already up on this thread, and in that case
// this file must not be the one to tear it down.
HRESULT ensureComInitialised() {
    static const HRESULT result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    return result;
}

// Resource-pack paths routinely contain non-ASCII (CJK world/pack names), so the UTF-8 the
// rest of the program speaks has to become UTF-16 before any Win32 call sees it.
std::wstring utf8ToWide(const std::string& text) {
    if (text.empty()) {
        return std::wstring();
    }
    int needed = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), int(text.size()), nullptr, 0);
    if (needed <= 0) {
        fail("cannot convert path to UTF-16: " + text);
    }
    std::wstring wide(size_t(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), int(text.size()), wide.data(), needed);
    return wide;
}

void releaseCom(IUnknown* object) {
    if (object) {
        object->Release();
    }
}

}  // namespace

Rgba8Image decodeImageFile(const std::string& path) {
    HRESULT comResult = ensureComInitialised();
    if (FAILED(comResult)) {
        fail("CoInitializeEx failed (hr=" + hexCode(comResult) + "); WIC is unavailable");
    }
    std::wstring widePath = utf8ToWide(path);

    IWICImagingFactory* factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        fail("cannot create the WIC imaging factory for " + path + " (hr=" + hexCode(hr) + ")");
    }

    IWICBitmapDecoder* decoder = nullptr;
    hr = factory->CreateDecoderFromFilename(widePath.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr) || !decoder) {
        releaseCom(factory);
        fail("cannot open image: " + path);
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr) || !frame) {
        releaseCom(decoder);
        releaseCom(factory);
        fail("image has no readable frame: " + path);
    }

    // A single converter normalises every source form a pack might use — palette, 16-bit,
    // tRNS, interlaced — into the 32bpp RGBA the copy below assumes.
    IWICFormatConverter* converter = nullptr;
    hr = factory->CreateFormatConverter(&converter);
    if (SUCCEEDED(hr) && converter) {
        hr = converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
                                   nullptr, 0.0, WICBitmapPaletteTypeCustom);
    }
    if (FAILED(hr) || !converter) {
        releaseCom(frame);
        releaseCom(decoder);
        releaseCom(factory);
        fail("cannot convert image to 32bpp RGBA: " + path);
    }

    UINT width = 0, height = 0;
    converter->GetSize(&width, &height);
    if (width == 0 || height == 0) {
        releaseCom(converter);
        releaseCom(frame);
        releaseCom(decoder);
        releaseCom(factory);
        fail("image has zero extent: " + path);
    }

    Rgba8Image out;
    out.width = width;
    out.height = height;
    out.data.resize(size_t(width) * height * 4);
    hr = converter->CopyPixels(nullptr, width * 4, UINT(out.data.size()), out.data.data());

    releaseCom(converter);
    releaseCom(frame);
    releaseCom(decoder);
    releaseCom(factory);

    if (FAILED(hr)) {
        fail("cannot read pixels from image: " + path);
    }
    return out;
}
