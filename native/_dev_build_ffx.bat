@echo off
REM Developer-only helper: builds the FFX native pieces against the vendored SDK spike in
REM third_party\ffx-vk-spike. Not part of the Gradle build; see native/ffxprobe/README.md.
REM
REM Usage:  _dev_build_ffx.bat [shim|probe|both] [Release|Debug]

setlocal enabledelayedexpansion

set "WHAT=%~1"
if "%WHAT%"=="" set "WHAT=both"
set "CONFIG=%~2"
if "%CONFIG%"=="" set "CONFIG=Release"

REM %~dp0 already ends in a backslash, so ..\ here is the repository root.
set "FFX_SDK=%~dp0..\third_party\ffx-vk-spike"
for %%I in ("%FFX_SDK%") do set "FFX_SDK=%%~fI"
if not exist "%FFX_SDK%\ffx_api\ffx_framegeneration.h" (
  echo [ffx] no ffx_api headers under "%FFX_SDK%"
  exit /b 1
)
echo [ffx] FFX_SDK=%FFX_SDK%

if "%WHAT%"=="shim" goto :shim
if "%WHAT%"=="probe" goto :probe

:shim
call "%~dp0ffx_shim\build.bat" %CONFIG%
if errorlevel 1 exit /b 1
if "%WHAT%"=="shim" goto :done

:probe
call "%~dp0ffxprobe\build.bat" %CONFIG%
if errorlevel 1 exit /b 1

:done
endlocal
