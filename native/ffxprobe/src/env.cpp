#include "env.h"

#include <cstdio>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

bool hasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (const VkExtensionProperties& e : list) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

uint32_t formatBytesPerPixel(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R16G16_SFLOAT:
            return 4;
        default:
            return 0;
    }
}

std::vector<VkExtensionProperties> deviceExtensionList(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> list(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, list.data());
    return list;
}

PFN_vkCreateDebugUtilsMessengerEXT g_createMessenger = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT g_destroyMessenger = nullptr;
uint32_t g_validationMessages = 0;

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ++g_validationMessages;
        std::fprintf(stderr, "[validation ERROR] %s\n", data->pMessage);
    } else {
        std::fprintf(stderr, "[validation] %s\n", data->pMessage);
    }
    std::fflush(stderr);
    return VK_FALSE;
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

void checkOk(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        std::ostringstream out;
        out << what << " failed with VkResult " << vkResultName(result);
        fail(out.str());
    }
}

std::string vkResultName(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: {
            std::ostringstream out;
            out << "VkResult(" << static_cast<int>(result) << ")";
            return out.str();
        }
    }
}

std::string envCreate(Env& env, bool validation, std::vector<std::string>& notes) {
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
            notes.push_back("validation requested but VK_LAYER_KHRONOS_validation is not installed");
        }
    }

    uint32_t instanceExtCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, nullptr);
    std::vector<VkExtensionProperties> instanceExts(instanceExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, instanceExts.data());

    std::vector<const char*> instanceExtensions;
    // Only validation needs this here. The SDK's debug-label entry points resolve without it,
    // and a run with them answered and a run without them fault in the same place, so the
    // extension is not what ffx-api's start-up is actually waiting for.
    if (haveValidation && hasExtension(instanceExts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else if (haveValidation) {
        notes.push_back("validation layer present but VK_EXT_debug_utils is not");
        haveValidation = false;
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "ffxprobe";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "caustica-ffxprobe";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instanceInfo.ppEnabledLayerNames = layers.data();
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();

    // Lives for the whole function: VkInstanceCreateInfo carries a pointer to it, and the
    // instance has to be created before a messenger can be made from it.
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                              | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messengerInfo.pfnUserCallback = debugCallback;

    if (haveValidation) {
        // Chaining it here is what makes the messenger cover instance and device creation,
        // not only whatever comes after them.
        instanceInfo.pNext = &messengerInfo;
    }
    checkOk(vkCreateInstance(&instanceInfo, nullptr, &env.instance), "vkCreateInstance");
    env.validation = haveValidation;

    if (haveValidation) {
        g_createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(env.instance, "vkCreateDebugUtilsMessengerEXT"));
        g_destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(env.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (g_createMessenger) {
            g_createMessenger(env.instance, &messengerInfo, nullptr, &env.messenger);
        }
    }

    uint32_t deviceCount = 0;
    checkOk(vkEnumeratePhysicalDevices(env.instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
    if (deviceCount == 0) {
        fail("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(env.instance, &deviceCount, devices.data());

    // The probe needs a compute queue and nothing else. The descriptor-indexing feature it
    // then has to turn on is what ffx-api's bindless layouts ask for, so a device without it
    // cannot run the SDK at all and is not a usable candidate.
    VkPhysicalDeviceDescriptorIndexingFeatures diFeatures{};
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    int bestScore = -1;
    for (uint32_t i = 0; i < deviceCount; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, families.data());
        bool compute = false;
        for (const VkQueueFamilyProperties& f : families) {
            if (f.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute = true;
            }
        }

        std::vector<VkExtensionProperties> exts = deviceExtensionList(devices[i]);
        if (!compute || !hasExtension(exts, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            continue;
        }

        VkPhysicalDeviceDescriptorIndexingFeatures di{};
        di.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &di;
        vkGetPhysicalDeviceFeatures2(devices[i], &f2);
        if (!di.descriptorBindingPartiallyBound) {
            continue;
        }

        int score = scoreDevice(props);
        if (score > bestScore) {
            bestScore = score;
            chosen = devices[i];
            diFeatures = di;
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        fail("no Vulkan device with a compute queue and VK_EXT_descriptor_indexing"
             " descriptorBindingPartiallyBound");
    }
    env.physicalDevice = chosen;
    vkGetPhysicalDeviceProperties(env.physicalDevice, &env.props);
    vkGetPhysicalDeviceMemoryProperties(env.physicalDevice, &env.memoryProps);
    env.bufferImageGranularity = env.props.limits.bufferImageGranularity;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(env.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(env.physicalDevice, &familyCount, families.data());
    bool foundQueue = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            env.queueFamily = i;
            foundQueue = true;
            break;
        }
    }
    if (!foundQueue) {
        fail("no compute-capable queue family");
    }

    std::vector<VkExtensionProperties> deviceExts = deviceExtensionList(env.physicalDevice);
    std::vector<const char*> enabled;
    auto want = [&](const char* name) {
        if (hasExtension(deviceExts, name)) {
            enabled.push_back(name);
        } else {
            notes.push_back(std::string("dropped unsupported device extension ") + name);
        }
    };
    want(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    // ffx-api looks these up through vkGetDeviceProcAddr by their KHR-suffixed names. Per spec
    // that answers NULL unless the extension was enabled at device creation, even though both
    // are core since 1.1, and the SDK then calls the null pointer. This is the same gating
    // Caustica's VulkanBackendMixin works around for the game's own device.
    want(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    want(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    want(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    want(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(env.physicalDevice, &features);
    if (!features.shaderStorageImageExtendedFormats) {
        notes.push_back("shaderStorageImageExtendedFormats unavailable; rg16f storage images may fail");
    }
    features.shaderStorageImageExtendedFormats = VK_TRUE;

    // Only what the driver reports: a layout carrying
    // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT is invalid unless the matching feature is on.
    VkPhysicalDeviceDescriptorIndexingFeatures diEnable = diFeatures;

    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.features = features;
    features2.pNext = &diEnable;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = env.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features2;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    deviceInfo.ppEnabledExtensionNames = enabled.data();
    checkOk(vkCreateDevice(env.physicalDevice, &deviceInfo, nullptr, &env.device), "vkCreateDevice");

    vkGetDeviceQueue(env.device, env.queueFamily, 0, &env.queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = env.queueFamily;
    checkOk(vkCreateCommandPool(env.device, &poolInfo, nullptr, &env.commandPool), "vkCreateCommandPool");

    return env.props.deviceName;
}

void envDestroy(Env& env) {
    if (env.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(env.device);
        if (env.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(env.device, env.commandPool, nullptr);
            env.commandPool = VK_NULL_HANDLE;
        }
        vkDestroyDevice(env.device, nullptr);
        env.device = VK_NULL_HANDLE;
    }
    if (env.instance != VK_NULL_HANDLE) {
        if (env.messenger != VK_NULL_HANDLE && g_destroyMessenger) {
            g_destroyMessenger(env.instance, env.messenger, nullptr);
            env.messenger = VK_NULL_HANDLE;
        }
        vkDestroyInstance(env.instance, nullptr);
        env.instance = VK_NULL_HANDLE;
    }
}

uint32_t findMemoryType(const Env& env, uint32_t typeBits, VkMemoryPropertyFlags wanted) {
    for (uint32_t i = 0; i < env.memoryProps.memoryTypeCount; ++i) {
        bool typeOk = (typeBits & (1u << i)) != 0;
        bool flagsOk = (env.memoryProps.memoryTypes[i].propertyFlags & wanted) == wanted;
        if (typeOk && flagsOk) {
            return i;
        }
    }
    fail("no memory type satisfies the requested properties");
}

Img createImage(Env& env, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage) {
    Img out;
    out.format = format;
    out.width = width;
    out.height = height;

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    out.createInfo = info;
    checkOk(vkCreateImage(env.device, &info, nullptr, &out.image), "vkCreateImage");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(env.device, out.image, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(env, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    checkOk(vkAllocateMemory(env.device, &alloc, nullptr, &out.memory), "vkAllocateMemory(image)");
    checkOk(vkBindImageMemory(env.device, out.image, out.memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = out.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    checkOk(vkCreateImageView(env.device, &view, nullptr, &out.view), "vkCreateImageView");

    // Undefined to GENERAL once, so nothing later has to reason about layouts.
    VkCommandBuffer cmd = beginCommands(env);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = out.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                          | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &barrier);
    submitCommands(env, cmd);
    return out;
}

void destroyImage(Env& env, Img& img) {
    if (img.view != VK_NULL_HANDLE) {
        vkDestroyImageView(env.device, img.view, nullptr);
        img.view = VK_NULL_HANDLE;
    }
    if (img.image != VK_NULL_HANDLE) {
        vkDestroyImage(env.device, img.image, nullptr);
        img.image = VK_NULL_HANDLE;
    }
    if (img.memory != VK_NULL_HANDLE) {
        vkFreeMemory(env.device, img.memory, nullptr);
        img.memory = VK_NULL_HANDLE;
    }
}

VkCommandBuffer beginCommands(const Env& env) {
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = env.commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    checkOk(vkAllocateCommandBuffers(env.device, &alloc, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkOk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
    return cmd;
}

void submitCommands(const Env& env, VkCommandBuffer cmd) {
    checkOk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    checkOk(vkCreateFence(env.device, &fenceInfo, nullptr, &fence), "vkCreateFence");
    checkOk(vkQueueSubmit(env.queue, 1, &submit, fence), "vkQueueSubmit");
    checkOk(vkWaitForFences(env.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkDestroyFence(env.device, fence, nullptr);
    vkFreeCommandBuffers(env.device, env.commandPool, 1, &cmd);
}

void memoryBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                          | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_DEPENDENCY_BY_REGION_BIT, 1, &barrier, 0, nullptr, 0, nullptr);
}

std::vector<uint8_t> readImage(const Env& env, const Img& img) {
    const uint32_t bytesPerPixel = 4;
    if (formatBytesPerPixel(img.format) != bytesPerPixel) {
        fail("readImage only handles 4-byte formats; the probe reads colour and motion guides");
    }
    VkDeviceSize size = VkDeviceSize(img.width) * img.height * bytesPerPixel;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    checkOk(vkCreateBuffer(env.device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer(readback)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(env.device, buffer, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(env, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                         | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    checkOk(vkAllocateMemory(env.device, &alloc, nullptr, &memory), "vkAllocateMemory(readback)");
    checkOk(vkBindBufferMemory(env.device, buffer, memory, 0), "vkBindBufferMemory(readback)");

    VkCommandBuffer cmd = beginCommands(env);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {img.width, img.height, 1};
    vkCmdCopyImageToBuffer(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
    memoryBarrier(cmd);
    submitCommands(env, cmd);

    void* mapped = nullptr;
    checkOk(vkMapMemory(env.device, memory, 0, size, 0, &mapped), "vkMapMemory(readback)");
    std::vector<uint8_t> out(static_cast<const uint8_t*>(mapped),
                             static_cast<const uint8_t*>(mapped) + size);
    vkUnmapMemory(env.device, memory);

    vkDestroyBuffer(env.device, buffer, nullptr);
    vkFreeMemory(env.device, memory, nullptr);
    return out;
}

VkShaderModule loadShaderModule(const Env& env, const std::string& path) {
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

    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = static_cast<size_t>(size);
    info.pCode = reinterpret_cast<const uint32_t*>(words.data());
    VkShaderModule module = VK_NULL_HANDLE;
    checkOk(vkCreateShaderModule(env.device, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

uint32_t validationErrorCount() {
    return g_validationMessages;
}

uint32_t deviceExtensionCount(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    return count;
}
