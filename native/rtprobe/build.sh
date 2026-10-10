#!/usr/bin/env bash
# Builds rtprobe without vcvars64 / cmd.exe.
#
# Why this exists in addition to build.bat: the VS generator cannot detect this machine's
# Visual Studio Build Tools instance, and cmd.exe is not reachable from every shell here, so
# the MSVC environment is assembled explicitly and CMake is driven with Ninja.
#
# Usage:  bash build.sh [Release|Debug]

set -euo pipefail

CONFIG="${1:-Release}"

MSVC='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
SDK='C:\Program Files (x86)\Windows Kits\10'
SDK_VERSION='10.0.26100.0'

MSVC_UNIX='/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207'
NINJA_UNIX='/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja'
SDK_BIN_UNIX='/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64'
CMAKE='/c/Program Files/CMake/bin/cmake.exe'

export INCLUDE="${MSVC}\\include;${SDK}\\Include\\${SDK_VERSION}\\ucrt;${SDK}\\Include\\${SDK_VERSION}\\um;${SDK}\\Include\\${SDK_VERSION}\\shared"
export LIB="${MSVC}\\lib\\x64;${SDK}\\Lib\\${SDK_VERSION}\\ucrt\\x64;${SDK}\\Lib\\${SDK_VERSION}\\um\\x64"
export PATH="${MSVC_UNIX}/bin/Hostx64/x64:${NINJA_UNIX}:${SDK_BIN_UNIX}:${PATH}"

if [ -z "${DLSS_SDK:-}" ]; then
    echo "[rtprobe] DLSS_SDK is not set" >&2
    exit 1
fi
if [ -z "${VULKAN_SDK:-}" ]; then
    echo "[rtprobe] VULKAN_SDK is not set" >&2
    exit 1
fi

# Resolved with bash parameter expansion rather than dirname(1): this machine's Git Bash
# ships without coreutils, so any external binary here is a coin flip.
SOURCE_DIR="${BASH_SOURCE[0]%/*}"
if [ "${SOURCE_DIR}" = "${BASH_SOURCE[0]}" ]; then
    SOURCE_DIR="."
fi
BUILD_DIR="${SOURCE_DIR}/../../build/native/rtprobe/cmake"

echo "[rtprobe] configuring ${SOURCE_DIR} -> ${BUILD_DIR} (${CONFIG})"
"${CMAKE}" -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE="${CONFIG}"

echo "[rtprobe] building"
"${CMAKE}" --build "${BUILD_DIR}"

echo "[rtprobe] done"
