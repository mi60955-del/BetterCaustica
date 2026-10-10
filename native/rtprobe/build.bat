@echo off
REM Builds rtprobe with Ninja + the MSVC toolchain from Visual Studio Build Tools.
REM
REM The VS generator is not used on purpose: the installed Build Tools instance is not
REM detected by CMake's Visual Studio generator on this machine, while Ninja plus a
REM vcvars64 environment works and is what the project's own flake.nix uses.
REM
REM Usage:  build.bat [Release|Debug]
REM Env:    DLSS_SDK and VULKAN_SDK must be set.

setlocal enabledelayedexpansion

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Release"

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"

if not exist "%VCVARS%" (
  echo [rtprobe] vcvars64.bat not found at "%VCVARS%"
  exit /b 1
)
if not exist "%CMAKE%" (
  echo [rtprobe] cmake not found at "%CMAKE%"
  exit /b 1
)
if "%DLSS_SDK%"=="" (
  echo [rtprobe] DLSS_SDK is not set
  exit /b 1
)
if "%VULKAN_SDK%"=="" (
  echo [rtprobe] VULKAN_SDK is not set
  exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 (
  echo [rtprobe] vcvars64 failed
  exit /b 1
)

set "PATH=%PATH%;%NINJA_DIR%"

set "SOURCE=%~dp0"
set "SOURCE=%SOURCE:~0,-1%"
set "BUILD=%~dp0..\..\build\native\rtprobe\cmake"

echo [rtprobe] configuring %SOURCE% -> %BUILD% (%CONFIG%)
"%CMAKE%" -S "%SOURCE%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG%
if errorlevel 1 (
  echo [rtprobe] configure failed
  exit /b 1
)

echo [rtprobe] building
"%CMAKE%" --build "%BUILD%"
if errorlevel 1 (
  echo [rtprobe] build failed
  exit /b 1
)

echo [rtprobe] done
endlocal
