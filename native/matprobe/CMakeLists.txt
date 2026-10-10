cmake_minimum_required(VERSION 3.20)
project(matprobe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Only the Vulkan SDK is needed: matprobe deliberately has no NGX/DLSS dependency, because
# the question it answers (what does this LabPBR material look like) has nothing to do with
# upscaling or denoising.
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK environment variable is not set")
endif()

get_filename_component(MATPROBE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MATPROBE_OUTPUT_ROOT "${MATPROBE_REPO_ROOT}/build/native/matprobe"
    CACHE PATH "Directory for built matprobe artifacts")
set(MATPROBE_OUTPUT_DIR "${MATPROBE_OUTPUT_ROOT}/$<LOWER_CASE:$<CONFIG>>")

set(VULKAN_SDK_DIR "$ENV{VULKAN_SDK}")

# --- shader -------------------------------------------------------------------------
# Compiled to a config-independent path and copied next to the executable, so the probe
# never has to guess where its SPIR-V lives.
set(MATPROBE_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/matprobe.comp")
set(MATPROBE_SHADER_SPV "${MATPROBE_OUTPUT_ROOT}/matprobe.comp.spv")
set(GLSLC "${VULKAN_SDK_DIR}/Bin/glslc.exe")

if(NOT EXISTS "${GLSLC}")
  message(FATAL_ERROR "glslc not found at ${GLSLC} (is VULKAN_SDK correct?)")
endif()

add_custom_command(
    OUTPUT "${MATPROBE_SHADER_SPV}"
    COMMAND "${GLSLC}" --target-env=vulkan1.3 -O -o "${MATPROBE_SHADER_SPV}" "${MATPROBE_SHADER_SRC}"
    DEPENDS "${MATPROBE_SHADER_SRC}"
    COMMENT "Compiling matprobe.comp -> matprobe.comp.spv"
    VERBATIM)

add_custom_target(matprobe_shaders ALL DEPENDS "${MATPROBE_SHADER_SPV}")

# --- executable ---------------------------------------------------------------------
add_executable(matprobe
    src/main.cpp
    src/vk.cpp
    src/image_io.cpp
    src/material.cpp)

target_include_directories(matprobe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${VULKAN_SDK_DIR}/Include")

# windowscodecs+ole32 are the PNG decode path (see image_io.cpp): WIC ships with Windows,
# so reading resource-pack textures costs no third-party dependency.
target_link_libraries(matprobe PRIVATE
    "${VULKAN_SDK_DIR}/Lib/vulkan-1.lib"
    windowscodecs.lib
    ole32.lib
    bcrypt.lib)

# Source files are UTF-8. Without /utf-8 MSVC on a CP936 host decodes them as GBK, which
# silently corrupts any non-ASCII byte in a string literal.
if(MSVC)
  target_compile_options(matprobe PRIVATE /utf-8)
endif()

add_dependencies(matprobe matprobe_shaders)

add_custom_command(TARGET matprobe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MATPROBE_SHADER_SPV}" "$<TARGET_FILE_DIR:matprobe>/matprobe.comp.spv"
    COMMENT "Staging matprobe.comp.spv next to the executable"
    VERBATIM)

set_target_properties(matprobe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MATPROBE_OUTPUT_DIR}"
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
