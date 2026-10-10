@echo off
REM Builds ffxprobe with Ninja + the MSVC toolchain from Visual Studio Build Tools.
REM Mirrors native/rtprobe/build.bat.
REM
REM Usage:  build.bat [Release|Debug]
REM Env:    FFX_SDK (directory holding ffx_api/ and amd_fidelityfx_vk.dll) and VULKAN_SDK
REM         must be set.

setlocal enabledelayedexpansion

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Release"

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"

if not exist "%VCVARS%" (
  echo [ffxprobe] vcvars64.bat not found at "%VCVARS%"
  exit /b 1
)
if not exist "%CMAKE%" (
  echo [ffxprobe] cmake not found at "%CMAKE%"
  exit /b 1
)
if "%FFX_SDK%"=="" (
  echo [ffxprobe] FFX_SDK is not set
  exit /b 1
)
if "%VULKAN_SDK%"=="" (
  echo [ffxprobe] VULKAN_SDK is not set
  exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 (
  echo [ffxprobe] vcvars64 failed
  exit /b 1
)

set "PATH=%PATH%;%NINJA_DIR%"

set "SOURCE=%~dp0"
set "SOURCE=%SOURCE:~0,-1%"
set "BUILD=%~dp0..\..\build\native\ffxprobe\cmake"

echo [ffxprobe] configuring %SOURCE% -> %BUILD% (%CONFIG%)
"%CMAKE%" -S "%SOURCE%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG%
if errorlevel 1 (
  echo [ffxprobe] configure failed
  exit /b 1
)

echo [ffxprobe] building
"%CMAKE%" --build "%BUILD%"
if errorlevel 1 (
  echo [ffxprobe] build failed
  exit /b 1
)

echo [ffxprobe] done
endlocal
