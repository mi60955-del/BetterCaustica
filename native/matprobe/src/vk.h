// Minimal Vulkan device bring-up and resource helpers for the probe.
//
// Deliberately small and dependency-free: no VMA, no helper library. Everything the probe
// touches is created with raw Vulkan calls so a failure points at one obvious place.
//
// All guide images live in VK_IMAGE_LAYOUT_GENERAL for their whole lifetime. That is what
// NGX expects for the resources it reads and writes, and it removes every layout-transition
// question from the frame loop.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

[[noreturn]] void fail(const std::string& message);

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    VkDeviceAddress address = 0;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct Probe {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties deviceProps{};
    VkPhysicalDeviceMemoryProperties memoryProps{};
    uint32_t scratchAlignment = 256;
    bool validation = false;

    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getAccelerationStructureBuildSizes = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddress = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
};

struct ProbeConfig {
    std::vector<std::string> instanceExtensions;
    std::vector<std::string> deviceExtensions;
    int gpuIndex = -1;          // -1 selects the first suitable device
    bool validation = false;
};

// Creates instance + device + queue. Extensions that the driver does not offer are dropped
// and reported through `notes` rather than aborting, because NGX's required-extension list
// is broader than what this probe actually needs.
std::string createProbe(Probe& probe, const ProbeConfig& config, std::vector<std::string>& notes);
void destroyProbe(Probe& probe);

// Names of every physical device, indexed the way --gpu wants them.
std::vector<std::string> enumeratePhysicalDevices();

uint32_t findMemoryType(const Probe& probe, uint32_t typeBits, VkMemoryPropertyFlags wanted);

Buffer createBuffer(const Probe& probe, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memory, bool withDeviceAddress);
void destroyBuffer(const Probe& probe, Buffer& buffer);
void uploadBuffer(const Probe& probe, const Buffer& buffer, const void* data, size_t bytes);

Image createImage(const Probe& probe, uint32_t width, uint32_t height, VkFormat format,
                  VkImageUsageFlags usage);
void destroyImage(const Probe& probe, Image& image);

// Copies tightly packed rows into an image. The image must already be in GENERAL layout
// (call transitionToGeneral first); matprobe keeps every texture there, so a sampler can
// read it and a transfer can write it without a layout question in between.
void uploadImage(const Probe& probe, const Image& image, const void* data, VkDeviceSize bytes);

// Material textures are sampled, never written, so they use a plain sampler rather than the
// storage-image path the guide buffers take. `nearest` is what a texel-exact debug view
// wants; `repeat` is what a material plate wants so a tiling texture tiles.
VkSampler createSampler(const Probe& probe, bool nearest, bool repeat);
void destroySampler(const Probe& probe, VkSampler& sampler);

// Moves an image from UNDEFINED to GENERAL once, at creation time.
void transitionToGeneral(const Probe& probe, const Image& image);

// Records a one-shot command buffer, submits it, and waits. Intended for build steps and
// for the frame loop, which is serialised anyway.
VkCommandBuffer beginCommands(const Probe& probe);
void submitCommands(const Probe& probe, VkCommandBuffer commandBuffer);

void memoryBarrier(VkCommandBuffer commandBuffer);
void bufferBarrier(VkCommandBuffer commandBuffer, VkBuffer buffer, VkAccessFlags src,
                   VkAccessFlags dst);

VkShaderModule loadShaderModule(const Probe& probe, const std::string& path);
