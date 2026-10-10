# ffxprobe

A standalone program that answers one question: **can the stock FidelityFX SDK Vulkan backend
generate a frame on this machine?** Not "does the API return OK" - the calls return OK for
several wrong inputs too - but "does a frame come out the other end that is in the right
place".

This exists because DLSS Frame Generation cannot run on the author's RTX 3060 Laptop, and
before any of Caustica's frame loop is rewired around AMD's SDK, someone has to know whether
AMD's SDK runs at all on an NVIDIA Vulkan driver. AMD's own docs do not say, and the third
party that ships FSR frame generation on Vulkan (Minecraft Vulkan Renderer) ships a *forked*
SDK, so "unmodified FFX works" had no witness.

## Building

```
native\_dev_build_ffxprobe.bat Release
```

which is `native\ffxprobe\build.bat` with `FFX_SDK` pointed at
`third_party\ffx-vk-spike`. `build.bat` needs `FFX_SDK` and `VULKAN_SDK` in the environment,
like `native\ngx_shim` needs `DLSS_SDK`.

`FFX_SDK` may be either an official SDK root (`include/ffx_api`,
`bin/Windows_x86_64/retail`) or the flat spike layout (`ffx_api/`, with
`amd_fidelityfx_vk.dll` alongside). The DLL is loaded at run time with `LoadLibraryExA`, so no
import library is needed, and CMake stages it next to the executable.

## Running

```
ffxprobe                       # all four arms, PNGs in .
ffxprobe --control             # create only an upscaler context, then stop
ffxprobe --arm pixels_s1       # one arm
ffxprobe --validation          # with the Vulkan validation layer
```

Exit code 0 means at least one arm produced a frame with the moving bar at the midpoint.

## What the probe measures

The scene (`shaders/scene.comp`) is one red bar over a striped world that pans at a different
rate, plus a green square fixed to the screen. All three move by amounts the CPU knows exactly,
so the correct interpolated frame is the one where the bar's centroid sits halfway between its
previous and current centres. The measured quantity is that distance:

| column | meaning |
| --- | --- |
| `errMid` | bar centroid in the generated frame, minus the true midpoint |
| `errPrev` | minus the centroid measured in the previous rendered frame |
| `errCur` | minus the centroid measured in the current rendered frame |

`errPrev` and `errCur` are the two ways to be wrong in a way a real user would notice - showing
an old frame, or repeating the current one - so an arm only counts as passing if it is closer to
the midpoint than to either of those.

`hudFill` is separate: the fraction of the fixed square's rectangle that still reads as
saturated green in the generated frame. `mad(against a plain blend)` compares the output with a
straight 50/50 average of the two sources.

## What it found

### 1. Frame generation runs on this driver, unmodified

`ffxCreateContext` -> `ffxConfigure` -> `ffxDispatch(prepare)` -> `ffxDispatch(generate)` all
return OK on a GeForce RTX 3060 Laptop GPU, `HUDLessColor` filled, no swapchain handed to AMD
(`FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY`), and a frame comes out that tracks the
midpoint to 0.14 px at 512x288. The provider reports itself as `0xf600000000401003 "1.1.3"` and
asks for 11 512 336 B of device memory. Four contexts were created and destroyed in one process
without leaking into each other, which is what a resolution change or a settings toggle needs.

### 2. The blocker was two alias extension names, and this repo already knew

With a plain device, `ffxCreateContext` faults by calling address 0. The probe prints the
call site as `amd_fidelityfx_vk.dll + 0x1e5b0`, which is the exact offset
`src/main/java/dev/comfyfluffy/caustica/mixin/VulkanBackendMixin.java` already documents:
ffx-api resolves `vkGetImageMemoryRequirements2KHR` and friends through the
`PFN_vkGetDeviceProcAddr` it is handed, using the **KHR-suffixed** names. Per spec that answers
`NULL` for an extension that is core but not *enabled*, even though the functionality has been
core since 1.1, and the SDK then calls the null pointer.

The fix on the probe's side is to enable `VK_KHR_get_memory_requirements2` and
`VK_KHR_dedicated_allocation` on the device it creates. In the game, `VulkanBackendMixin`
already does this, so **Caustica's own device should already satisfy FFX's requirement** - but
this has to be re-checked against the real device rather than assumed, because the mixin adds
those names only when the driver advertises them.

Two theories this killed on the way, both worth recording so nobody re-litigates them:

- *The app's `vkGetDeviceProcAddr` must fall back to `vkGetInstanceProcAddr`.* Measured with
  `--shim fallback|strict|none`: all three fault identically. The fallback is harmless and the
  probe keeps it as a diagnostic, but it is not what makes the difference. What the six
  unanswered names actually are: `vkSetDebugUtilsObjectNameEXT`,
  `vkCmdWriteBufferMarkerAMD`, `vkCmdWriteBufferMarker2AMD`,
  `vkCmdBeginDebugUtilsLabelEXT`, `vkCmdEndDebugUtilsLabelEXT`,
  `vkGetBufferMemoryRequirements2KHR` - and only the last one is on the path that runs.
- *`VK_EXT_debug_utils` is required.* With debug labels resolving to real pointers and with them
  resolving to null, the fault is the same, at the same offset.

### 3. Motion vector conventions: partly answered

`ffx_frameinterpolation.cpp:848` computes the shader's scale as
`motionVectorScale / motionVectorsTargetSize`, and `ffx_frameinterpolation.h:46` then does
`reprojectedUv = uv + motionVector`. So the value the shader works in is UV, and the only
self-consistent input is a raw vector in **render pixels with `motionVectorScale = (1, 1)`**.

The measurement agrees about magnitude and is silent about sign:

- `pixels_sW` (vectors in pixels, scale set to the render size, i.e. 512x too large) is caught
  clearly: `errPrev = +0.00`, so the generated frame is a copy of the previous frame.
- `pixels_s1` scores best on `errMid`, as predicted.
- `pixels_neg` (negated) and `uv_s1` (vectors 512x too small, i.e. effectively zero) *also* land
  on the midpoint, within 0.4 px.

That last line is a limit of the instrument, not a result about FSR. Frame generation builds an
**optical flow** vector field from the two backbuffers and blends it against the game field by
colour similarity (`ffx_frameinterpolation.h:102-106`, and the `Compute interpolation` step in
`docs/techniques/frame-interpolation.md`). In a scene this regular, flow recovers the true
motion on its own, so a wrong game vector costs quality but does not move the bar. **A bar
centroid cannot distinguish the sign of the game vectors.** Settling that needs a scene the
flow term cannot match - the obvious one is the same geometry with per-frame uncorrelated
appearance, so the flow confidence collapses while the geometric vector stays correct.

For the integration itself the sign is already fixed twice over: FSR2's documentation states
that vectors "encode the motion from a pixel in the current frame to the position of that same
pixel in the previous frame", and Caustica's `gMotion` is `prev - cur` in render pixels, which
is what step 1's own interpolation used and the author judged free of reverse drag.

### 4. `HUDLessColor` solves the GUI problem for free

`hudFill = 1.000` in every arm: the screen-fixed square is recomposed onto the generated frame
without smearing, while the world underneath it interpolates. Step 1's one defect - the
interface being interpolated along with the world - is what this input exists for.

## Conventions that had to be read out of the source rather than the docs

FSR2's documentation says to set `motionVectorScale` to the render resolution for NDC-space
vectors. Given `ffx_frameinterpolation.cpp:848` and `ffx_frameinterpolation.h:46` that is wrong
by a factor of two for NDC and wrong by the resolution for pixels; the same table appears twice
in `super-resolution-temporal.md`. Trust the shader and the host code, not the table.

| input | what the code actually wants |
| --- | --- |
| `motionVectors` | render resolution, `R16G16_SFLOAT`, value = previous - current, in **pixels**, no jitter |
| `motionVectorScale` | `(1, 1)` for pixel vectors; the runtime divides it by the vector target size |
| `depth` | render resolution, read with `texelFetch`, so `VK_IMAGE_USAGE_SAMPLED_BIT`; `FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED` for reversed-Z |
| `jitterOffset` | `(0, 0)` unless `ENABLE_MOTION_VECTORS_JITTER_CANCELLATION` is set |
| resource handles | `FfxApiResource::resource` is the `VkImage` **itself**, cast to `void*`; ffx-api makes its own image views |
| resource state | `FFX_API_RESOURCE_STATE_UNORDERED_ACCESS` - it is the only state `getVKImageLayoutFromResourceState` maps to `VK_IMAGE_LAYOUT_GENERAL`, and Caustica's targets are all GENERAL |
| `description.mipCount` | must be 1; the helper's value feeds `VkImageViewCreateInfo::levelCount`, where 0 is invalid |
| device extensions | `VK_EXT_descriptor_indexing` (bindless layouts use `PARTIALLY_BOUND`), plus the two alias names in section 2 |
| `frameID` | increments by **exactly one per rendered frame**; any gap resets the logic |

## Diagnostic switches

| flag | why it exists |
| --- | --- |
| `--shim fallback\|strict\|none` | three ways to answer `vkGetDeviceProcAddr`, to tell an integration bug from a bug in the DLL |
| `--control` | brings up a *different* effect on the same device, to tell a shared start-up failure from an effect-specific one |
| `--arm NAME` | one convention per process, so an arm cannot inherit another's history |
| `--guard` | allocates ffx-api's memory through callbacks whose tail is poisoned, and prints the call site. Off by default: the SDK releases some blocks without calling `dealloc`, so the registry goes stale. It is what proved no linear overrun of the scratch buffer |
| `--debug-view` | `FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW` plus `ENABLE_DEBUG_CHECKING` |
| crash handler | prints the faulting module and offset, the return address read off the stack, and the bytes of the call instruction. Section 2 was found this way: no symbols, no PDB, just `amd_fidelityfx_vk.dll + 0x1e5b0` |
