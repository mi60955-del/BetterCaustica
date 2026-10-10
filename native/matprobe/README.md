# matprobe

An offscreen preview and measurement tool for LabPBR materials. Point it at a resource pack
and name a material; it loads the albedo / `_s` / `_n` triplet, runs the same LabPBR decode the
game runs at pack load, renders the result with a path tracer, and writes a PNG next to a
`report.json` of the numbers the image is based on.

Validating a material from inside the game costs booting Minecraft, loading a world, finding the
right block and squinting at a screenshot — and the answer is still "it looks off". This is the
same trade rtprobe makes for RT + DLSS-RR, applied to materials: a couple of seconds, and the
answer comes back as both an image and per-channel statistics.

Nothing here is wired into Gradle, so it cannot affect the mod build.

## Quick start

```bat
REM build (see the build section; needs VULKAN_SDK)
native\matprobe\build.bat

REM make a synthetic pack that exercises every decode branch, then look at it
python native\matprobe\testpack\generate_testpack.py
build\native\matprobe\release\matprobe.exe --pack native\matprobe\testpack\pack --list-materials
build\native\matprobe\release\matprobe.exe --pack native\matprobe\testpack\pack ^
    --material matprobe_test:block/gold_metal --out out

REM Material Parallax: the same knob as the game setting, 0 = off
build\native\matprobe\release\matprobe.exe --pack "C:\path\to\pack" ^
    --material minecraft:block/bricks --parallax 0.5 --tilt 55 --out out

REM the real case: a material out of a resource pack
build\native\matprobe\release\matprobe.exe --pack "C:\path\to\pack" ^
    --material minecraft:block/stone --material minecraft:block/polished_deepslate --out out

REM or skip the pack and hand it files directly
build\native\matprobe\release\matprobe.exe --albedo textures\block\stone.png --out out
```

`--help` lists every option.

## Two viewing modes

| Mode | What it is for |
|---|---|
| `plate` (default) | One flat quad per material, viewed at a slight angle. The texture is shown un-warped, so this is the mode for judging texel detail, tiling and a normal map's relief. |
| `sphere` | One sphere per material with equirectangular UVs. This is the mode for judging the BRDF — a metal reads as a metal because it reflects the sky gradient, and roughness shows up as how sharp that reflection is. |

Multiple materials lay out in a grid in either mode, so an A/B of two packs is one command.

## Texel filtering

`--filter nearest` (the default) is what reproduces what the game puts on screen. Caustica binds
its block atlas **and** its canonical material pages with one shared sampler
(`RtComposite.atlasSampler`): `magFilter`/`minFilter` `NEAREST`, `mipmapMode` `LINEAR`,
`REPEAT`. `NEAREST` magnification is why a 16×16 texture reads as hard pixels in the game rather
than as a smooth blur — judging a material from a linearly filtered preview would be judging
something the game never shows.

`--filter linear` is offered for inspecting the decoded fields themselves (roughness, F0,
height), where smooth interpolation is easier to read than texel steps. It is *not* what the
game does.

## Material Parallax

`--parallax F` is the same knob as the game's **Material Parallax** setting
(`caustica.rt.parallax-depth`, 0–0.5, default 0.25): how far a material's height map may sink
its surface, as a fraction of the 25%-of-a-block depth LabPBR authors. `0` turns the march off,
which is the quickest way to tell a parallax artefact from anything else on a surface.

The march is a port of `parallaxLocalUv` in `closest_hit.rchit.slang` — 12 layers, a bisection
of the last interval, the same Z floor and the same clamp — so the two agree by construction
rather than by approximation. It runs only for materials whose `_n` actually carries a height
field (reported as `hasHeight`), which is what keeps it from costing every surface sixteen extra
fetches for nothing.

`--tilt DEGREES` sets the camera's angle off the plate normal. It defaults to 35° because
parallax is **invisible head-on**: the view ray is then parallel to the height field's own axis,
so the march never moves the sample. Raise it to see more displacement, lower it to see the
texture flatter.

### March variants (what the game does, and what it could do)

Three switches change only what the march hands back, never the march itself. They exist so a
candidate can be rendered here before anything is baked into the mod; `--clamp-slope` has not
been taken into the game, the other two have.

| flag | effect |
|---|---|
| `--texel-snap` | round the displaced UV onto the material sprite's own texel grid |
| `--step-normal` | blend a vertical wall normal into the band a step exposes, weighted by the height gradient across one texel (`WALL_SLOPE_REFERENCE = 0.35`) |
| `--clamp-slope` | bound the march's lateral slope to 1, so the reach can never exceed `--parallax` |

Without the first two, a several-texel displacement is rendered as a continuous sub-texel warp:
the contour between "moved" and "not moved" follows the height field's interpolated ramp instead
of the texel grid, so a brick loses its corners to a rounded erosion and a mortar channel's flat
floor sharpens into a V — and the band itself, lit by the smooth normal map, reads as a stretched
patch of texture rather than as a wall. With them it reads as blocks.

`--clamp-slope` is a bound, not a fix: the slope exceeds 1 from **45°** onward, so it flattens
the relief across 45–80°, where the physically correct offset is up to 1.73× (60°) or 5.7× (80°)
the material's depth. That is why it is not in the game.

### Reproducing a grazing-angle artefact

Pass the **same material several times** (`--material minecraft:block/bricks` ×6) and matprobe
lays the copies out in a grid, which is the only way to get a "looking along a whole brick floor"
framing. A single plate at `--tilt 80` is too small on screen for its artefacts to be legible.

## What it decodes

The decode is a re-implementation of `RtLabPbr.decodeSpec`, which `RtBlockMaterials` runs at
pack load to canonicalise materials before upload. That CPU path is the single authority for
what a LabPBR byte means: `closest_hit.rchit.slang` no longer decodes anything, it samples the
physical channels this produces. Keeping one decode rule is what makes a result here transfer
to the game.

| LabPBR input | Meaning | Decoded to |
|---|---|---|
| `_s` red | perceptual smoothness | linear GGX alpha `(1 - smoothness)²` |
| `_s` green | `< 229.5` dielectric F0; `229.5..237.5` one of 8 conductor enums; `> 237.5` F0 = albedo | metalness + `f0.rgb` |
| `_s` blue | `0..64` porosity, `65..255` subsurface scattering (one lane, never both) | `porosity` / `sss` |
| `_s` alpha | `0..254` emission strength, `255` = none authored | `emission` |
| `_n` rgb | tangent-space normal xy; blue is ambient occlusion | world normal, `ao` |
| `_n` alpha | height as a depth: 255 flush, 0 is 25% of a block deep | `height` |

A material with no `_s` falls back to the game's neutral default surface
(`RtMaterials.Profile.DEFAULT`: linear alpha 0.81, dielectric F0 0.04), and one with no `_n` is
flush with no occlusion. Both are reported as warnings rather than silently applied.

The four GPU pages are the decoded result, not the raw bytes:

| Page | Format | Contents |
|---|---|---|
| albedo | `RGBA8` | authored sRGB albedo in `.rgb` (the shader applies `srgbToLinear`) |
| surface0 | `RGBA8` | `r`=roughness `g`=metalness `b`=emission `a`=sss |
| surface1 | `RGBA8` | `rgb`=F0 (linear) `a`=porosity |
| normalAo | `RGBA8` | `rg`=tangent normal xy `b`=AO `a`=height |

Materials are packed into those pages in a grid, each cell padded with a copy of its own border
texels so a bilinear tap at a cell edge cannot read into its neighbour.

## Reading `report.json`

| Field | Meaning |
|---|---|
| `materials[].channels.*` | min/mean/max of every decoded channel, over the material's texels |
| `materials[].classification` | how many texels took each decode branch: `dielectric`, `metalEnum`, `metalAlbedoF0`, `emissive`, `sss`, `porosity`, `nonFlushHeight` |
| `materials[].metals[]` | which conductor enums were actually hit, by name |
| `materials[].normalDeviationMean` | mean `|xy - 0.5| * 2`; `0` means a flat map, higher means more relief |
| `materials[].warnings[]` | missing maps, extent mismatches, plain-albedo textures |
| `artifacts.previewMd5` | hash of the PNG — an A/B of two runs reduces to one line |
| `parallaxDepth`, `tiltDegrees` | the march depth this run used, and the camera angle it was viewed at |
| `page` | the packed page size, cell size, gutter and grid — what actually went to the GPU |

Sanity checks that have held so far: a dielectric's `f0` mean is ~0.04; a conductor enum decodes
to a saturated F0 (gold from byte 231 is `(0.944, 0.776, 0.373)`); with no `_s` the roughness is
exactly 0.81; with no `_n` the height is exactly 1.0 everywhere.

## Building

Only the Vulkan SDK is required — matprobe has no NGX/DLSS dependency, because "what does this
material look like" has nothing to do with upscaling.

```bash
cd native/matprobe
source ./build.sh Release        # bash is not on PATH here, so `source`, not `bash build.sh`
```

`build.sh` assembles the MSVC environment explicitly and drives CMake with Ninja. Two facts
about this machine, both already hit once:

* **The CMake Visual Studio generator cannot see this Build Tools install.** Use Ninja.
* **Sources are UTF-8 on a CP936 host.** Without `-utf-8` MSVC decodes them as GBK and silently
  corrupts non-ASCII bytes. `CMakeLists.txt` sets it.

`windowscodecs.lib` + `ole32.lib` are the PNG decode path (WIC, below); `bcrypt.lib` is the MD5.

## PNG decoding, and where it does not work

Textures are decoded with WIC, which ships with Windows, so reading a resource pack costs no
third-party dependency. **WIC is COM**: it needs `CoInitializeEx` and a COM activation to
succeed.

In a sandboxed environment that blocks COM activation for the executable, that activation fails
with `E_ACCESSDENIED` (`0x80070005`) and matprobe reports
`cannot create the WIC imaging factory`. Under this repository's WorkBuddy sandbox that is what
happens when the exe is run from inside the workspace; running it from a normal terminal, or
copying the exe outside the workspace, works. The failure is reported with its HRESULT rather
than as a crash, so it is visible rather than mysterious.

## Known differences from the game

* **Working space.** The preview shades in linear sRGB. The game converts authored BT.709
  albedo and F0 into ACEScg before transport. This affects the hue of a very saturated material,
  not its decoded parameters, which are what this tool exists to report.
* **No subsurface transport.** The `sss` channel is decoded, previewed and reported, but the
  tracer does not implement a scattering lobe, so a scattering material renders as an opaque
  diffuse one. The number is right; the look is not modelled.
* **No wetness.** Porosity is decoded and reported; the game's wet-surface response needs a
  wetness value this tool has no way to know.
* **Single mip level.** Textures are uploaded without a mip chain, so an obliquely viewed surface
  can alias. The game's material pages carry an explicit semantic mip chain (`MAX_VALID_LOD` = 3)
  that this does not build. Head-on at 1:1 it does not show.
* **One sky.** The environment is a gradient sky plus an analytic sun. Metal reflects the sky
  gradient and nothing else, so this shows what a material does with the light there is, not how
  it would look in a built-up scene.

## Layout

```
CMakeLists.txt              Ninja/MSVC build, compiles the shader, stages the SPIR-V next to the exe
build.sh                    gets the MSVC environment in place without cmd.exe
shaders/matprobe.comp       the path tracer: samples the decoded pages, shades with GGX
src/main.cpp                CLI, page packing, scene, camera, report
src/material.{h,cpp}        the LabPBR decode, pack lookup, per-channel diagnostics
src/vk.{h,cpp}              device bring-up, buffers, images, textures, sampler
src/image_io.{h,cpp}        PNG/EXR writers, WIC PNG reader, metrics
testpack/generate_testpack.py   writes a synthetic pack covering every decode branch
```
