cmake_minimum_required(VERSION 3.20)
project(ffxprobe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Same convention as native/ngx_shim: an SDK root comes in through the environment and the
# build refuses to guess a location for it.
if(NOT DEFINED ENV{FFX_SDK})
  message(FATAL_ERROR "FFX_SDK environment variable is not set")
endif()
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK environment variable is not set")
endif()

set(FFX_SDK_DIR "$ENV{FFX_SDK}")
set(VULKAN_SDK_DIR "$ENV{VULKAN_SDK}")

# The ffx-api headers include each other by relative path ("../ffx_api.h"), so the include
# directory handed to the compiler is the one that *contains* the ffx_api folder. Both the
# official SDK layout (include/ffx_api) and a flat checkout (ffx_api) are accepted.
set(FFX_API_PARENT_CANDIDATES "${FFX_SDK_DIR}/include" "${FFX_SDK_DIR}" "${FFX_SDK_DIR}/ffx-api/include")
set(FFX_API_PARENT "")
foreach(candidate ${FFX_API_PARENT_CANDIDATES})
  if(EXISTS "${candidate}/ffx_api/ffx_framegeneration.h")
    set(FFX_API_PARENT "${candidate}")
    break()
  endif()
endforeach()
if(NOT FFX_API_PARENT)
  message(FATAL_ERROR "no ffx_api/ffx_framegeneration.h under: ${FFX_API_PARENT_CANDIDATES}")
endif()
message(STATUS "ffx-api headers: ${FFX_API_PARENT}/ffx_api")

# Linked at run time through LoadLibrary, so no import library is required.
set(FFX_DLL_CANDIDATES
    "${FFX_SDK_DIR}/bin/Windows_x86_64/retail/amd_fidelityfx_vk.dll"
    "${FFX_SDK_DIR}/bin/Windows_x86_64/release/amd_fidelityfx_vk.dll"
    "${FFX_SDK_DIR}/bin/Windows_x86_64/relwithdebinfo/amd_fidelityfx_vk.dll"
    "${FFX_SDK_DIR}/amd_fidelityfx_vk.dll")
set(FFX_VK_DLL "")
foreach(candidate ${FFX_DLL_CANDIDATES})
  if(EXISTS "${candidate}")
    set(FFX_VK_DLL "${candidate}")
    break()
  endif()
endforeach()
if(NOT FFX_VK_DLL)
  message(FATAL_ERROR "no amd_fidelityfx_vk.dll under: ${FFX_DLL_CANDIDATES}")
endif()
message(STATUS "ffx vk dll: ${FFX_VK_DLL}")

get_filename_component(FFXPROBE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(FFXPROBE_OUTPUT_ROOT "${FFXPROBE_REPO_ROOT}/build/native/ffxprobe"
    CACHE PATH "Directory for built ffxprobe artifacts")
set(FFXPROBE_OUTPUT_DIR "${FFXPROBE_OUTPUT_ROOT}/$<LOWER_CASE:$<CONFIG>>")

# --- shaders --------------------------------------------------------------------------
# glslc rather than slangc: the probe's scene generator is throwaway GLSL that has to match
# what the FFX backend expects byte for byte, not what Caustica's shader pipeline produces.
set(FFXPROBE_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene.comp")
set(FFXPROBE_SHADER_SPV "${FFXPROBE_OUTPUT_ROOT}/scene.comp.spv")
set(GLSLC "${VULKAN_SDK_DIR}/Bin/glslc.exe")

if(NOT EXISTS "${GLSLC}")
  message(FATAL_ERROR "glslc not found at ${GLSLC} (is VULKAN_SDK correct?)")
endif()

add_custom_command(
    OUTPUT "${FFXPROBE_SHADER_SPV}"
    COMMAND "${GLSLC}" --target-env=vulkan1.3 -O -o "${FFXPROBE_SHADER_SPV}" "${FFXPROBE_SHADER_SRC}"
    DEPENDS "${FFXPROBE_SHADER_SRC}"
    COMMENT "Compiling scene.comp -> scene.comp.spv"
    VERBATIM)

add_custom_target(ffxprobe_shaders ALL DEPENDS "${FFXPROBE_SHADER_SPV}")

# --- executable -----------------------------------------------------------------------
# image_io.cpp is compiled out of native/rtprobe: the probe needs the same stored-block PNG
# writer and there is no reason to keep a second copy of it. The Vulkan device this program
# creates turns on VK_EXT_descriptor_indexing because that is what ffx-api's own bindless
# descriptor layouts ask for.
add_executable(ffxprobe
    src/main.cpp
    src/env.cpp
    src/ffx_host.cpp
    ../rtprobe/src/image_io.cpp)

target_include_directories(ffxprobe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rtprobe/src"
    "${FFX_API_PARENT}"
    "${VULKAN_SDK_DIR}/Include")

target_link_libraries(ffxprobe PRIVATE "${VULKAN_SDK_DIR}/Lib/vulkan-1.lib")

if(MSVC)
  # Without /utf-8 MSVC decodes these UTF-8 sources as CP936 on a Chinese host, corrupting
  # every non-ASCII byte that ends up in a string literal.
  target_compile_options(ffxprobe PRIVATE /utf-8)
endif()

add_dependencies(ffxprobe ffxprobe_shaders)

# The SDK is loaded with LoadLibrary by full path, but staging it next to the executable
# keeps a plain `ffxprobe --help` working and mirrors how the mod will ship it.
add_custom_command(TARGET ffxprobe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFX_VK_DLL}" "$<TARGET_FILE_DIR:ffxprobe>/amd_fidelityfx_vk.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFXPROBE_SHADER_SPV}" "$<TARGET_FILE_DIR:ffxprobe>/scene.comp.spv"
    COMMENT "Staging amd_fidelityfx_vk.dll and scene.comp.spv next to the executable"
    VERBATIM)

set_target_properties(ffxprobe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FFXPROBE_OUTPUT_DIR}"
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
