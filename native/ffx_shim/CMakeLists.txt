cmake_minimum_required(VERSION 3.20)
project(ffxshim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

get_filename_component(FFX_SHIM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(FFX_SHIM_OUTPUT_ROOT "${FFX_SHIM_REPO_ROOT}/build/native/ffx_shim"
    CACHE PATH "Directory for built FFX shim artifacts")
set(FFX_SHIM_OUTPUT_DIR "${FFX_SHIM_OUTPUT_ROOT}/$<LOWER_CASE:$<CONFIG>>")

function(ffxshim_set_artifact_dirs target)
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${FFX_SHIM_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${FFX_SHIM_OUTPUT_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${FFX_SHIM_OUTPUT_DIR}")
endfunction()

if(NOT DEFINED ENV{FFX_SDK})
  message(FATAL_ERROR "FFX_SDK environment variable is not set")
endif()
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK environment variable is not set")
endif()

set(FFX_SDK_DIR "$ENV{FFX_SDK}")

# The ffx-api headers include one another by relative path (vk/ffx_api_vk.h reaches "../ffx_api.h"),
# so what the compiler needs is the directory that *contains* the ffx_api folder. Both the official
# SDK layout (include/ffx_api) and a flat vendored copy (ffx_api) are accepted.
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

add_library(ffxshim SHARED ffx_shim.cpp)

target_include_directories(ffxshim PRIVATE "${FFX_API_PARENT}")
target_include_directories(ffxshim PRIVATE "$ENV{VULKAN_SDK}/Include")

if(WIN32)
  set_property(TARGET ffxshim PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

set_target_properties(ffxshim PROPERTIES
    OUTPUT_NAME "ffxshim")
ffxshim_set_artifact_dirs(ffxshim)
