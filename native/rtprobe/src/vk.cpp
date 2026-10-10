#include "vk.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        std::ostringstream out;
        out << what << " failed with VkResult " << static_cast<int>(result);
        fail(out.str());
    }
}

bool hasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (const VkExtensionProperties& e : list) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<VkExtensionProperties> instanceExtensionList() {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> list(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, list.data());
    return list;
}

std::vector<VkExtensionProperties> deviceExtensionList(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> list(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, list.data());
    return list;
}

VkInstance createInstance(bool validation, const std::vector<std::string>& requested,
                          std::vector<std::string>& notes, VkDebugUtilsMessengerEXT* messenger) {
    std::vector<const char*> layers;
    bool haveValidation = false;
    if (validation) {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        for (const VkLayerProperties& l : available) {
            if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                haveValidation = true;
            }
        }
        if (haveValidation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        } else {
            notes.push_back("validation layer requested but VK_LAYER_KHRONOS_validation is not installed");
        }
    }

    std::vector<VkExtensionProperties> available = instanceExtensionList();
    std::vector<const char*> extensions;
    for (const std::string& name : requested) {
        if (hasExtension(available, name.c_str())) {
            extensions.push_back(name.c_str());
        } else {
            notes.push_back("dropped unsupported instance extension " + name);
        }
    }
    if (haveValidation && hasExtension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "rtprobe";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "caustica-rtprobe";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");

    if (haveValidation && hasExtension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (createMessenger) {
            VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                 | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            auto callback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                               VkDebugUtilsMessageTypeFlagsEXT,
                               const VkDebugUtilsMessengerCallbackDataEXT* data,
                               void*) -> VkBool32 {
                const char* tag = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                                        ? "validation ERROR" : "validation";
                std::fprintf(stderr, "[%s] %s\n", tag, data->pMessage);
                std::fflush(stderr);
                return VK_FALSE;
            };
            info.pfnUserCallback = callback;
            createMessenger(instance, &info, nullptr, messenger);
        }
    }
    return instance;
}

int scoreDevice(const VkPhysicalDeviceProperties& props) {
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0;
    }
}

}  // namespace

void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::vector<std::string> enumeratePhysicalDevices() {
    std::vector<std::string> names;
    std::vector<std::string> notes;
    VkInstance instance = createInstance(false, {}, notes, nullptr);
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) == VK_SUCCESS && count) {
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(devices[i], &props);
            names.push_back(props.deviceName);
        }
    }
    vkDestroyInstance(instance, nullptr);
    return names;
}

std::string createProbe(Probe& probe, const ProbeConfig& config, std::vector<std::string>& notes) {
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    probe.instance = createInstance(config.validation, config.instanceExtensions, notes, &messenger);
    probe.validation = config.validation;

    uint32_t deviceCount = 0;
    check(vkEnumeratePhysicalDevices(probe.instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
    if (deviceCount == 0) {
        fail("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(probe.instance, &deviceCount, devices.data());

    auto hasComputeQueue = [](VkPhysicalDevice device) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
        for (const VkQueueFamilyProperties& f : families) {
            if (f.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return true;
            }
        }
        return false;
    };

    auto supportsRt = [](VkPhysicalDevice device) {
        std::vector<VkExtensionProperties> exts = deviceExtensionList(device);
        return hasExtension(exts, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            && hasExtension(exts, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    };

    if (config.gpuIndex >= 0) {
        if (static_cast<uint32_t>(config.gpuIndex) >= deviceCount) {
            fail("--gpu index out of range");
        }
        probe.physicalDevice = devices[config.gpuIndex];
    } else {
        int best = -1;
        int bestScore = -1;
        for (uint32_t i = 0; i < deviceCount; ++i) {
            if (!hasComputeQueue(devices[i]) || !supportsRt(devices[i])) {
                continue;
            }
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(devices[i], &props);
            int score = scoreDevice(props);
            if (score > bestScore) {
                bestScore = score;
                best = static_cast<int>(i);
            }
        }
        if (best < 0) {
            fail("no Vulkan device with VK_KHR_acceleration_structure + VK_KHR_ray_query");
        }
        probe.physicalDevice = devices[best];
    }

    vkGetPhysicalDeviceProperties(probe.physicalDevice, &probe.deviceProps);
    vkGetPhysicalDeviceMemoryProperties(probe.physicalDevice, &probe.memoryProps);

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(probe.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(probe.physicalDevice, &familyCount, families.data());
    bool foundQueue = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            probe.queueFamily = i;
            foundQueue = true;
            break;
        }
    }
    if (!foundQueue) {
        fail("no compute-capable queue family");
    }

    // Feature probing: only ask for what the driver actually reports.
    VkPhysicalDeviceFeatures available{};
    vkGetPhysicalDeviceFeatures(probe.physicalDevice, &available);

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rqQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 featuresQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    featuresQuery.pNext = &asQuery;
    asQuery.pNext = &rqQuery;
    rqQuery.pNext = &bdaQuery;
    vkGetPhysicalDeviceFeatures2(probe.physicalDevice, &featuresQuery);

    if (!asQuery.accelerationStructure) {
        fail("device does not support the accelerationStructure feature");
    }
    if (!rqQuery.rayQuery) {
        fail("device does not support the rayQuery feature");
    }
    if (!bdaQuery.bufferDeviceAddress) {
        fail("device does not support bufferDeviceAddress");
    }
    if (!available.shaderStorageImageExtendedFormats) {
        notes.push_back("shaderStorageImageExtendedFormats unavailable; rg16f storage images may fail");
    }

    std::vector<VkExtensionProperties> deviceExtensions = deviceExtensionList(probe.physicalDevice);
    std::vector<std::string> wanted = config.deviceExtensions;
    wanted.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    wanted.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    wanted.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    std::vector<const char*> enabled;
    for (const std::string& name : wanted) {
        if (hasExtension(deviceExtensions, name.c_str())) {
            if (std::find(enabled.begin(), enabled.end(), name.c_str()) == enabled.end()) {
                enabled.push_back(name.c_str());
            }
        } else {
            notes.push_back("dropped unsupported device extension " + name);
        }
    }

    VkPhysicalDeviceFeatures features{};
    features.shaderStorageImageExtendedFormats = available.shaderStorageImageExtendedFormats;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asFeatures.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqFeatures.rayQuery = VK_TRUE;
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.features = features;
    features2.pNext = &asFeatures;
    asFeatures.pNext = &rqFeatures;
    rqFeatures.pNext = &bdaFeatures;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = probe.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features2;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    deviceInfo.ppEnabledExtensionNames = enabled.data();
    check(vkCreateDevice(probe.physicalDevice, &deviceInfo, nullptr, &probe.device), "vkCreateDevice");

    vkGetDeviceQueue(probe.device, probe.queueFamily, 0, &probe.queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = probe.queueFamily;
    check(vkCreateCommandPool(probe.device, &poolInfo, nullptr, &probe.commandPool), "vkCreateCommandPool");

    probe.createAccelerationStructure = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(probe.device, "vkCreateAccelerationStructureKHR"));
    probe.destroyAccelerationStructure = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(probe.device, "vkDestroyAccelerationStructureKHR"));
    probe.getAccelerationStructureBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(probe.device, "vkGetAccelerationStructureBuildSizesKHR"));
    probe.getAccelerationStructureDeviceAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(probe.device, "vkGetAccelerationStructureDeviceAddressKHR"));
    probe.cmdBuildAccelerationStructures = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(probe.device, "vkCmdBuildAccelerationStructuresKHR"));

    if (!probe.createAccelerationStructure || !probe.cmdBuildAccelerationStructures
            || !probe.getAccelerationStructureBuildSizes
            || !probe.getAccelerationStructureDeviceAddress
            || !probe.destroyAccelerationStructure) {
        fail("failed to resolve the VK_KHR_acceleration_structure entry points");
    }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(probe.physicalDevice, &props2);
    probe.scratchAlignment = asProps.minAccelerationStructureScratchOffsetAlignment;
    if (probe.scratchAlignment == 0) {
        probe.scratchAlignment = 256;
    }

    return probe.deviceProps.deviceName;
}

void destroyProbe(Probe& probe) {
    if (probe.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(probe.device);
        if (probe.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(probe.device, probe.commandPool, nullptr);
            probe.commandPool = VK_NULL_HANDLE;
        }
        vkDestroyDevice(probe.device, nullptr);
        probe.device = VK_NULL_HANDLE;
    }
    if (probe.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(probe.instance, nullptr);
        probe.instance = VK_NULL_HANDLE;
    }
}

uint32_t findMemoryType(const Probe& probe, uint32_t typeBits, VkMemoryPropertyFlags wanted) {
    for (uint32_t i = 0; i < probe.memoryProps.memoryTypeCount; ++i) {
        bool typeOk = (typeBits & (1u << i)) != 0;
        bool flagsOk = (probe.memoryProps.memoryTypes[i].propertyFlags & wanted) == wanted;
        if (typeOk && flagsOk) {
            return i;
        }
    }
    fail("no memory type satisfies the requested properties");
}

Buffer createBuffer(const Probe& probe, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memory, bool withDeviceAddress) {
    Buffer out;
    out.size = size;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(probe.device, &bufferInfo, nullptr, &out.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(probe.device, out.buffer, &requirements);

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(probe, requirements.memoryTypeBits, memory);
    if (withDeviceAddress) {
        allocateInfo.pNext = &flagsInfo;
    }
    check(vkAllocateMemory(probe.device, &allocateInfo, nullptr, &out.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(probe.device, out.buffer, out.memory, 0), "vkBindBufferMemory");

    if (withDeviceAddress) {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = out.buffer;
        out.address = vkGetBufferDeviceAddress(probe.device, &info);
    }
    if (memory & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        check(vkMapMemory(probe.device, out.memory, 0, size, 0, &out.mapped), "vkMapMemory");
    }
    return out;
}

void destroyBuffer(const Probe& probe, Buffer& buffer) {
    if (buffer.mapped) {
        vkUnmapMemory(probe.device, buffer.memory);
        buffer.mapped = nullptr;
    }
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(probe.device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(probe.device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
}

void uploadBuffer(const Probe& probe, const Buffer& buffer, const void* data, size_t bytes) {
    if (!buffer.mapped) {
        fail("uploadBuffer needs a host-visible buffer");
    }
    std::memcpy(buffer.mapped, data, bytes);
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    vkFlushMappedMemoryRanges(probe.device, 1, &range);
}

Image createImage(const Probe& probe, uint32_t width, uint32_t height, VkFormat format,
                  VkImageUsageFlags usage) {
    Image out;
    out.format = format;
    out.width = width;
    out.height = height;

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check(vkCreateImage(probe.device, &imageInfo, nullptr, &out.image), "vkCreateImage");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(probe.device, out.image, &requirements);
    VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(probe, requirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(probe.device, &allocateInfo, nullptr, &out.memory), "vkAllocateMemory(image)");
    check(vkBindImageMemory(probe.device, out.image, out.memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = out.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    check(vkCreateImageView(probe.device, &viewInfo, nullptr, &out.view), "vkCreateImageView");
    return out;
}

void destroyImage(const Probe& probe, Image& image) {
    if (image.view != VK_NULL_HANDLE) {
        vkDestroyImageView(probe.device, image.view, nullptr);
        image.view = VK_NULL_HANDLE;
    }
    if (image.image != VK_NULL_HANDLE) {
        vkDestroyImage(probe.device, image.image, nullptr);
        image.image = VK_NULL_HANDLE;
    }
    if (image.memory != VK_NULL_HANDLE) {
        vkFreeMemory(probe.device, image.memory, nullptr);
        image.memory = VK_NULL_HANDLE;
    }
}

VkCommandBuffer beginCommands(const Probe& probe) {
    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = probe.commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(probe.device, &allocateInfo, &commandBuffer), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    return commandBuffer;
}

void submitCommands(const Probe& probe, VkCommandBuffer commandBuffer) {
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(probe.device, &fenceInfo, nullptr, &fence), "vkCreateFence");
    check(vkQueueSubmit(probe.queue, 1, &submitInfo, fence), "vkQueueSubmit");
    check(vkWaitForFences(probe.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkDestroyFence(probe.device, fence, nullptr);
    vkFreeCommandBuffers(probe.device, probe.commandPool, 1, &commandBuffer);
}

void memoryBarrier(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                          | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void bufferBarrier(VkCommandBuffer commandBuffer, VkBuffer buffer, VkAccessFlags src,
                   VkAccessFlags dst) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void transitionToGeneral(const Probe& probe, const Image& image) {
    VkCommandBuffer commandBuffer = beginCommands(probe);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                          | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    submitCommands(probe, commandBuffer);
}

VkShaderModule loadShaderModule(const Probe& probe, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail("cannot open shader module " + path);
    }
    std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        fail("shader module " + path + " is empty or not a whole number of SPIR-V words");
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> words(static_cast<size_t>(size));
    if (!file.read(words.data(), size)) {
        fail("failed to read " + path);
    }

    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = static_cast<size_t>(size);
    createInfo.pCode = reinterpret_cast<const uint32_t*>(words.data());
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(probe.device, &createInfo, nullptr, &module), "vkCreateShaderModule");
    return module;
}
