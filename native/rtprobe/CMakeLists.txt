cmake_minimum_required(VERSION 3.20)
project(rtprobe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Same prerequisites as the ngx_shim target: the NGX static lib and the Vulkan SDK.
if(NOT DEFINED ENV{DLSS_SDK})
  message(FATAL_ERROR "DLSS_SDK environment variable is not set")
endif()
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK environment variable is not set")
endif()

get_filename_component(RTPROBE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(RTPROBE_OUTPUT_ROOT "${RTPROBE_REPO_ROOT}/build/native/rtprobe"
    CACHE PATH "Directory for built rtprobe artifacts")
set(RTPROBE_OUTPUT_DIR "${RTPROBE_OUTPUT_ROOT}/$<LOWER_CASE:$<CONFIG>>")

set(DLSS_SDK_DIR "$ENV{DLSS_SDK}")
set(VULKAN_SDK_DIR "$ENV{VULKAN_SDK}")

# --- shader -------------------------------------------------------------------------
# Compiled to a config-independent path and copied next to the executable, so the probe
# never has to guess where its SPIR-V lives.
set(RTPROBE_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/probe.comp")
set(RTPROBE_SHADER_SPV "${RTPROBE_OUTPUT_ROOT}/probe.comp.spv")
set(GLSLC "${VULKAN_SDK_DIR}/Bin/glslc.exe")

if(NOT EXISTS "${GLSLC}")
  message(FATAL_ERROR "glslc not found at ${GLSLC} (is VULKAN_SDK correct?)")
endif()

add_custom_command(
    OUTPUT "${RTPROBE_SHADER_SPV}"
    COMMAND "${GLSLC}" --target-env=vulkan1.3 -O -o "${RTPROBE_SHADER_SPV}" "${RTPROBE_SHADER_SRC}"
    DEPENDS "${RTPROBE_SHADER_SRC}"
    COMMENT "Compiling probe.comp -> probe.comp.spv"
    VERBATIM)

add_custom_target(rtprobe_shaders ALL DEPENDS "${RTPROBE_SHADER_SPV}")

# --- executable ---------------------------------------------------------------------
# ngx_shim.cpp is compiled into ngx_bridge.cpp (which #includes it) rather than listed here:
# that puts the shim's definitions in the same translation unit as the declarations in
# ngx_bridge.h, so any signature drift is a compile error instead of a runtime crash.
add_executable(rtprobe
    src/main.cpp
    src/vk.cpp
    src/image_io.cpp
    src/ngx_bridge.cpp)

target_include_directories(rtprobe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${DLSS_SDK_DIR}/include"
    "${VULKAN_SDK_DIR}/Include")

target_link_libraries(rtprobe PRIVATE
    "${DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib"
    "${VULKAN_SDK_DIR}/Lib/vulkan-1.lib"
    bcrypt.lib
    version.lib)

# Source files are UTF-8. Without /utf-8 MSVC on a CP936 host decodes them as GBK, which
# silently corrupts any non-ASCII byte in a string literal.
if(MSVC)
  target_compile_options(rtprobe PRIVATE /utf-8)
endif()

add_dependencies(rtprobe rtprobe_shaders)

add_custom_command(TARGET rtprobe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RTPROBE_SHADER_SPV}" "$<TARGET_FILE_DIR:rtprobe>/probe.comp.spv"
    COMMENT "Staging probe.comp.spv next to the executable"
    VERBATIM)

set_target_properties(rtprobe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${RTPROBE_OUTPUT_DIR}"
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
