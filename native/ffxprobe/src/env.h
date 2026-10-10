// Bare Vulkan bring-up for the FSR frame-generation probe.
//
// No VMA, no helper library: every object is created with raw Vulkan calls so that a failure
// points at one obvious place. This is deliberately a separate environment from
// native/rtprobe's, because the two programs need different devices - rtprobe requires ray
// tracing and the FFX path needs none of it, while it does need VK_EXT_descriptor_indexing
// (see `envCreate`).
//
// Every image lives in VK_IMAGE_LAYOUT_GENERAL for its whole lifetime. ffx-api's Vulkan
// backend maps an imported resource's state onto an image layout, and
// FFX_API_RESOURCE_STATE_UNORDERED_ACCESS is the only state that maps to GENERAL, so this is
// what Caustica will be able to hand it: its own targets are already GENERAL.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

[[noreturn]] void fail(const std::string& message);

// Turns any non-success VkResult into the same exception as fail(), with the symbolic name
// attached. Exported because the probe makes its own Vulkan calls outside this file.
void checkOk(VkResult result, const char* what);

// Shared with the probe's own Vulkan calls so that a failure anywhere reports a VkResult name
// rather than a number.
void checkOk(VkResult result, const char* what);

struct Img {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    // Kept because ffx-api's own helper derives its resource description from it.
    VkImageCreateInfo createInfo{};
};

struct Env {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memoryProps{};
    VkDeviceSize bufferImageGranularity = 1;
    bool validation = false;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
};

// Returns the chosen device name. `notes` collects everything that was requested but not
// available, so the report can state what the run did not exercise.
std::string envCreate(Env& env, bool validation, std::vector<std::string>& notes);
void envDestroy(Env& env);

uint32_t findMemoryType(const Env& env, uint32_t typeBits, VkMemoryPropertyFlags wanted);

// Created in GENERAL layout with one barrier at creation time.
Img createImage(Env& env, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage);
void destroyImage(Env& env, Img& img);

VkCommandBuffer beginCommands(const Env& env);
void submitCommands(const Env& env, VkCommandBuffer cmd);

// Compute-write to compute-read, by region, which is what ffx-api's own barriers use.
void memoryBarrier(VkCommandBuffer cmd);

// Copies an image into a staging buffer and returns its tightly packed bytes. `bytesPerPixel`
// is taken from the image format.
std::vector<uint8_t> readImage(const Env& env, const Img& img);

VkShaderModule loadShaderModule(const Env& env, const std::string& path);

std::string vkResultName(VkResult result);

// Errors the validation messenger has seen so far. Reported alongside every FFX return code,
// because the SDK often returns OK for a call that Vulkan itself rejected.
uint32_t validationErrorCount();

// How many device extensions this driver advertises. ffx-api sizes a scratch array of
// VkExtensionProperties from exactly this number, so it is the first thing to know when that
// array appears to be written past.
uint32_t deviceExtensionCount(VkPhysicalDevice physicalDevice);
