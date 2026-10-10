# rtprobe

An offscreen Vulkan ray tracer plus a DLSS Ray Reconstruction pass, built to answer one
question fast: **is RT + DLSS-RR actually working, and how well?**

Validating that inside the mod means booting Minecraft, loading a world and reading a log.
`rtprobe` runs the same NGX call sequence against a synthetic scene in a few seconds and
reports numbers instead of impressions. Nothing here is wired into Gradle, so it cannot
affect the mod build.

## Quick start

```bat
REM prerequisites: DLSS_SDK and VULKAN_SDK set, Visual Studio Build Tools installed
native\rtprobe\build.bat
build\native\rtprobe\release\rtprobe.exe
```

Outputs land in `rtprobe-out\`: `rt-noisy.png`, `rr-output.png`, the matching `.exr`
files and `report.json`.

Useful invocations:

```bat
rtprobe.exe --list-gpus
rtprobe.exe --display 1280x720 --frames 24 --dump-guides --out out\guides
rtprobe.exe --dll-dir "E:\Minecraft\26.2\caustica-ngx\natives\windows-x64"   REM default
rtprobe.exe --shim-dll "...\ngxshim.dll"    REM test a deployed shim binary instead
rtprobe.exe --no-rr                          REM baseline trace without NGX
rtprobe.exe --preset 0                       REM A/B a render preset
```

`--help` lists everything.

## Why the build looks unusual

Two machine-specific facts, both already hit once:

* **The CMake Visual Studio generator cannot see this Build Tools install** — it finds the
  Windows SDK but reports `No CMAKE_CXX_COMPILER could be found`. Ninja plus a `vcvars64`
  environment works, and matches what the project's own `flake.nix` uses
  (`CMAKE_GENERATOR = "Ninja"`).
* **Sources are UTF-8 on a CP936 host.** Without `-utf-8` MSVC decodes them as GBK and
  silently corrupts non-ASCII bytes in string literals. `CMakeLists.txt` sets it.

`build.bat` is the documented path. `build.sh` does the same thing without `cmd.exe`, for
shells where `cmd.exe` is unavailable; it also avoids `dirname(1)`, which this machine's
Git Bash does not ship.

## What it validates

### The scene

30,464 triangles: a checkerboard floor (high-frequency detail a denoiser must not smear), an
18-sphere grid spanning roughness 0.0003 (mirror) to 0.95 x metalness 0.0/0.5/1.0, and two
emissive spheres. The camera orbits by default so motion vectors and disocclusion are
exercised; the final `--hold` frames freeze the pose so temporal convergence can be measured
separately from camera motion.

### Guide-buffer contract

This is the part that has to be right for a result here to transfer to the game. Every buffer
matches Caustica's conventions (`shaders/pipelines/world/guides.slang`, `bindings.slang`,
`RtDlssRr.java`):

| Buffer | Format | Contents |
|---|---|---|
| color | `R16G16B16A16_SFLOAT` | scene-linear HDR, 1 spp, not tonemapped |
| normals | `R16G16B16A16_SFLOAT` | world normal.xyz, GGX roughness in `.w` |
| diffuse albedo | `R16G16B16A16_SFLOAT` | `base * (1 - metalness)`, `.a` = emissive flag |
| depth | `R32_SFLOAT` | HW depth from a reversed-Z projection: `clip.z / clip.w`, near = 1, far -> 0 |
| motion | `R16G16_SFLOAT` | `(prevNdc - jitteredNdc) * 0.5 * renderSize` — render-pixel units, jitter-free |
| specular albedo | `R16G16B16A16_SFLOAT` | F0, demodulated only for near-mirror opaque non-emissive reflectors |
| specular motion | `R16G16_SFLOAT` | reflection motion; zero above roughness 0.25 |

Sky follows the same miss handling (normal 0, roughness 1, depth 0, albedo 0.5, specular
albedo 0), and its motion reprojects the ray direction as a point at infinity so a pure
camera rotation still produces correct background vectors.

### The NGX contract

* Feature flags `75` = `IsHDR | MVLowRes | DepthInverted | AutoExposure` — identical to
  `RtDlssRr.FEATURE_FLAGS`.
* `mvScaleX/Y = 1.0` (motion is already in render-pixel units).
* Jitter is Halton(2,3) minus 0.5 over `max(32, ceil(8 * (display/render)^2))` phases,
  copied from `CausticaJitter`, and reported to NGX **negated**, matching
  `RtDlssRr.evaluate(-jitterX, -jitterY, ...)`.
* `worldToView` and `viewToClip` are passed **separately** and as the transpose of the
  GLSL/column-vector convention, i.e. plain row-major storage.
* DLSSD is created with the **shared capability parameter block**, not a fresh
  `AllocateParameters()` block — a fresh block fails with `FAIL_InvalidParameter`.
* Success is `0x1`, not `0`. Failure is `0xBAD00000 | code`. Anything that tests
  `result != 0` reports every success as a failure.

`native/ngx_shim/ngx_shim.cpp` is `#include`d by `src/ngx_bridge.cpp` rather than linked as a
separate object, so the declarations in `ngx_bridge.h` are conflict-checked against the real
definitions by the compiler and the probe cannot drift from the mod's call sequence. Passing
`--shim-dll` switches to `LoadLibrary` on a deployed `ngxshim.dll` instead, which is how the
shipped binary itself gets exercised.

### The resolution dance

`NGX_DLSSD_GET_OPTIMAL_SETTINGS` only works after the DLSSD feature library has been loaded,
which happens on the first create. So the probe creates 1:1 (the shim maps that to DLAA), asks
for the real size, then rebuilds at it. Same sequence as `RtDlssRr`. Pass `--render WxH` to
skip the query and create once at a known size.

## Reading `report.json`

| Field | Meaning |
|---|---|
| `ngx.dllFileVersion`, `ngx.dllSha256`, `ngx.dllBytes` | which DLSS-RR DLL was actually loaded |
| `ngx.initResult`, `ngx.rayReconstructionAvailable` | NGX init and the RR capability query |
| `dlssd.createResult`, `lastEvaluateResult` | every VkResult in the chain, spelled out |
| `frame.renderWidth/Height` | the internal resolution NGX asked for; compares against `displayWidth/Height` |
| `metrics.highFrequencyRatio` | RR high-frequency energy / noisy upsampled. Lower = more denoised |
| `metrics.temporalRatio` | RR frame-to-frame delta / noisy frame-to-frame delta over the frozen-camera frames. Lower = better temporal accumulation |
| `metrics.rrVsNaiveUpscale` | distance from a plain nearest upscale of the noisy image. `0` would mean RR did nothing |
| `artifacts.rrMd5` | hash of the RR EXR — an A/B of two runs reduces to comparing one line |
| `timing.wallClockPerFrameMs` | wall clock, sensitive to other GPU load; a sanity signal, not a benchmark |

Fields read `null` when the corresponding work did not run, so a skipped stage can never be
mistaken for a perfect score.

## Measured on this machine

RTX 3060 Laptop 6 GB, driver 616.92, Vulkan 1.4.351, `nvngx_dlssd.dll` FileVersion
`310.9.1.0` (48,343,664 bytes,
SHA-256 `4BC7EA5FCB2F32CF86BC2CB072E8D2860914A0BE511CBDCB2FC206C1D9D83B80`).

| Run | Result |
|---|---|
| `--display 1920x1080` (default, 48 frames) | render `1280x720` (Quality 1.5x, sharpness 0.35), init/evaluate `success (0x1)` |
| high-frequency energy | `0.3975` -> `0.0195` (ratio `0.049`) |
| temporal delta | `0.6343` -> `0.0695` (ratio `0.1095`) |
| RR vs naive upscale | `0.4686` |
| mean luma | `1.9495` -> `1.9347`, non-finite `0 / 0` |
| timing | ~80 ms/frame at 1080p, ~35 ms/frame at 720p; ~2.5-4 s for a default run |

Guide dumps verified separately: motion vectors span roughly -5.7..+3.2 px on a moving frame
and drop to exactly 0 when the camera is frozen; depth matches `near / -z_view`; metalness 1.0
spheres have zero diffuse albedo. The RR image shows no ghosting under camera motion, which is
the end-to-end proof that the motion-vector and depth conventions are right.

## Open finding: `--preset` has no measurable effect

Presets `0, 1, 2, 3, 6` produce **byte-identical** RR output (same MD5) at the same
resolution and frame count, both through the DLAA-precreate-then-rebuild path and through a
single create at the final render size. So on this DLL/driver combination the
`RayReconstruction_Hint_Render_Preset_*` parameters the shim sets do not change what RR
produces.

What has been ruled out:

* the rebuild path — a single create at the real render size behaves the same way;
* process-level latching — each run is a fresh process, and the first create already carries
  the requested preset;
* the parameter names — the shim compiles against `nvsdk_ngx_defs_dlssd.h`, and it uses the
  capability block, which is the block DLSSD requires.

Root cause is **not** determined. This is worth knowing because the working assumption in
`caustica-dlss-diagnostics` is that Preset F (`preset = 6`) must be set for the DLSS 4.5
network to be used at all; the probe suggests the knob may have no effect on the rendering
path here. Verifying it further would need NGX-side instrumentation the probe does not have.

## Limitations

* Compute-shader `VK_KHR_ray_query`, not a ray-tracing pipeline. Hardware traversal either
  way, and NGX does not care how the color was traced — but this is not a byte-for-byte
  replica of Caustica's pipeline topology.
* Reflection motion uses Caustica's mirror-across-the-surface-plane formula, which is exact
  for a planar reflector and an approximation over the spheres here. A static camera still
  yields exactly zero reflection motion, which is the property that matters.
* Depth uses a near-plane-only reversed-Z projection (no far plane), so `ndc.z` is small
  (~0.009 at 8-11 units). Consistent with the matrices handed to NGX, and RR linearizes from
  them.
* Timing is wall clock and includes a fence wait per frame; it is not a throughput benchmark.
* One scene. Adding another means editing `buildScene()`.

## Layout

```
CMakeLists.txt          Ninja/MSVC build, compiles the shader, stages the SPIR-V next to the exe
build.bat / build.sh    two ways to get the MSVC environment in place
shaders/probe.comp      the path tracer and every guide buffer write
src/main.cpp            CLI, scene, frame loop, metrics, report
src/vk.{h,cpp}          device bring-up and resource helpers (no VMA, no helper library)
src/ngx_bridge.{h,cpp}  shim access, statically linked or dlopen'd
src/image_io.{h,cpp}    PNG (hand-rolled stored-deflate) and EXR writers, metrics
```
