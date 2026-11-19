# Changelog

## 0.1.3 — 2026-11-18

### Added

- **Rain.** Falling rain is drawn as streaks after the denoiser and before the
  tone curve. Each drop is one quad whose vertices are generated from the vertex
  index alone: a vertical strip along the fall direction and the camera's
  horizontal right, lit from the sky directly overhead, carried sideways by the
  wind over the fall so that its lean and its drift are one motion, softened at
  both ends, and faded out where it would touch the eye or the edge of the
  disc around it. The pass has no depth attachment; the fragment shader reads
  the traced depth guide and runs the comparison itself, and a streak narrower
  than a pixel fades by its own coverage instead of shimmering. Because the
  streaks land in the reconstructed image, the exposure meter, bloom and the
  tone curve treat them as scene light. A still render leaves them out, since
  the mean of falling rain is a smear of it.
- **Weather darkening.** **Rain Sky Darkening** mirrors the curve vanilla's own
  lightmap follows (each weather level takes up to 5/16 of the sky light, with a
  20% floor) and applies it to the sky's radiometric inputs, so the sun, the
  atmosphere and every surface lit by them agree on the dim. Auto-exposure
  latches the same shift at the tone curve rather than in the scene, because a
  uniform dim would otherwise be metered straight back out.
- **Wetting and puddles.** **Rain Wetness** soaks every surface while it rains
  and dries slowly afterwards, so a gap in the shower does not snap stone and
  soil back to dry. **Puddle Coverage** then collects standing water on level
  ground: a puddle is a surface of its own, flattening the normal map, taking
  the water's roughness and Fresnel and darkening what lies beneath it, placed
  by a world-anchored lattice so it stays put as you walk. Emissive grounds,
  glass, foliage cutouts and entities never pool.
- **Water body medium.** Water now absorbs by channel instead of fading as one
  colour: red is swallowed about four times as fast as blue, so depth reads as a
  shift toward blue-green before the light is gone, and the biome tint only
  bends the ratio between channels. **Water Opacity** scales the absorption and
  the underwater sun shafts run on the same scattering, so beams and body agree.
  With **Water Fog** off the world is rendered exactly as before.
- **Cloud out-scatter.** Rays that terminate on geometry no longer see the hole
  in the sky behind a cloud sheet: a sheet's scattered radiance and its
  transmittance are shaded for rays that end on terrain or on a reflection, not
  only for rays that escape to the sky. **Cloud Steps** is the march count for
  the volumetric deck, separate from every other cloud dial.
- **Cloud quality dials.** Three new controls on the Sky page's Cloud detail
  group, each defaulting to the frame the procedural stack shipped with.
  **Cloud Light Steps** (1-8) is the sample count of the volumetric tier's sun
  leg through its own deck: 1 is the single midpoint tap, more resolve a crown
  shadowing its own base. **Cloud Detail** switches off the close view's
  erosion cut and the cirrus sheet's filament curl for a smooth,
  several-fetches-cheaper body — the shape the ground shadow field and the light
  legs already read — while coverage, crown, base and the cast shadow stay.
  **Cloud
  Shadow Steps** (1-4) averages the ground shadow field's sun leg through each
  deck instead of pricing the whole slab at one mid-plane crossing, sharpening
  a broken deck's shadow gaps; every NEE site and fog sun leg pays it, so the
  cost spreads across the frame rather than per cloud pixel.
- **Distant Horizons LOD.** With Distant Horizons installed, its far terrain is
  meshed into the traced scene beyond the edge of the traced vanilla world, so
  the horizon follows DH's render distance instead of the vanilla one. Rings at
  16-, 32- and 64-block cells cover 256-, 1024- and 4096-block regions, coarse
  cells are masked where a finer ring covers them, and every region is meshed
  into the same packed section format the traced terrain uses, so LOD geometry
  costs the terrain pipeline nothing extra. **LOD max distance** 0 follows DH's
  own chunk render distance, **LOD emissive light** lets distant windows, lava
  and caves glow at night, and **LOD fetch budget** bounds the render-thread
  microseconds spent reading DH's data. DH's own rasterized LOD is switched off
  while this is on.
- **Indirect radiance cache.** **Indirect Radiance Cache** ends each indirect
  path at its first bounce and reads the light beyond it out of a camera-centred
  grid of 48×16×48 probes two blocks apart, each holding an L1 spherical
  harmonic and refreshed by the screen pixel that owns it. Probes carry an age
  and a confidence, so a stale or missing one falls back to tracing and a light
  that changes propagates within a couple of seconds.
- **Indirect light subsampling.** **Indirect Light Subsampling** traces the
  indirect pass for two of every three pixels per axis and rebuilds the rest
  from a depth-aware upsample; it saves about a fifth of the indirect ray work
  and the result reads softer and slightly darker.
- **Endless still render.** **Endless Still Render** keeps summing past the
  sample count until the still-render key is pressed again, and that press both
  resolves and saves the run. F2 takes the accumulated picture at any moment.
- **Still render resolution.** **Still Render Resolution** resizes the window to
  **Still Render Width** and **Still Render Height** for the duration of a run
  and puts it back afterwards, so a picture can be accumulated larger than the
  desktop; a fullscreen run keeps the display's own size and says so.
- **Still render denoise.** **Still Render Denoise** filters the accumulated
  mean before it is tone mapped, using the normal, albedo and depth the tracer
  already wrote: it removes the part of the noise neighbouring pixels saw from
  the same surface and leaves textures, silhouettes and fine geometry alone.
  **Still Render Denoise Strength** blends it against the raw mean, and a run
  with DLSS-RR on skips the filter because it is already reconstructed.
- **Cat-eye and swirly bokeh.** **Cat-Eye Bokeh** clips the aperture toward the
  frame's edges the way a lens barrel does, leaving the crescent a wide-open
  lens puts on corner highlights, and renormalises the clip so the frame does
  not darken. **Swirly Bokeh** stretches highlights tangentially around the
  centre with the square of the field angle, for the swirl of a Petzval-type
  lens. Both work in the traced lens and in the post pass.
- **RR Input Spark Clamp.** Caps isolated single-texel spikes in the traced
  frame before Ray Reconstruction sees it, so one stray sample cannot become a
  bright dot in the reconstruction; a real highlight covers several texels and
  survives untouched.
- **Emitter Shadow Ray Depth Limit.** Stops the emitter shadow ray at vertices
  deeper than the given number of bounces and treats them as unoccluded, which
  is where emitters are dim and indirect anyway.
- **Pipeline dump.** `-Dcaustica.rt.pipelineStats=true` dumps what the driver
  compiled for each pipeline — executable statistics, register use, spills and
  the internal representations — through `VK_KHR_pipeline_executable_properties`.

### Changed

- **Clouds are now a four-deck procedural stack.** The nine baked VDB volumes,
  their atlas, the shape slider and the random-shape switch are gone. A low
  cumulus deck, a towering cumulus deck above it, an altostratus sheet and a
  curl-warped cirrus sheet are all cut from one 512-square tileable noise atlas
  generated at device bring-up. Each body is a solid isosurface: a biased
  coverage threshold, a flat base with a domed crown, a fourth-power erosion cut
  and a downwind shear of the crown. Lighting uses a four-lobe powder phase and a
  multi-scatter gain, so thin rims keep a silver lining while deep cores stay lit.
  Cloud shadows are priced per deck; the high sheets ride a longer fade than the
  cumulus.
- **Cloud settings.** Coverage is now a 0-200% shape threshold (70% nearly
  clear, 100% broken cumulus, 120% two-thirds covered, 180% solid overcast), density is a multiplier on the
  authored per-deck extinctions, size scales the noise tiles, and flow scrolls
  only the fine detail. Rain and thunder raise the coverage threshold and the
  density together. Cloud Type and Random Cloud Shapes were removed with the
  baked volumes.
- **Smaller jar.** Nine 2 MB cloud volumes and their baking tool are no longer
  shipped; the shape atlas is generated in memory instead.
- **Exposure metering.** The metering histogram now averages a 4×4 block per
  sample before binning. Log is concave, so noise in a one-sample frame biases
  the metered average upward; with DLSS-RR off nothing else was averaging it.
- **Focus pull.** The focal plane now travels on an exponential in dioptres: it
  covers 99% of the defocus it started with over **Focus Time**, at a constant
  rate in 1/s, in either direction, and cannot overshoot. A near surface and a
  far one take the same time.
- **Still render sums light.** A run accumulates scene-linear radiance before
  exposure instead of the mapped frame, and the running mean becomes the source
  the display chain sees, so bloom, flare and the tone curve treat it as the
  picture rather than as a frame that happened to be averaged.
- **Settings pages.** An option can now name the switch it depends on: an off
  parent greys its children where they are instead of hiding them, while the
  ones the machine cannot use at all (no HDR swapchain) are the ones that
  disappear. The **Volumetric Quality** row reads back the tier the dials
  currently spell and says Custom when they spell none.
- **Opacity micromaps retired.** The device no longer asks for
  `VK_EXT_opacity_micromap`: the cutout triangulation still produced artefacts
  with them, so the any-hit cutout test is the only path now.
- **Retuned defaults.** The shipped look now runs eight path bounces, a
  0.0078/block fog that blues out over 280 blocks and reddens out over 1.1 km,
  shafts at 0.02 / 0.2 anisotropy / 512-block reach, a four-deck cloud layer at
  70% coverage, 2× density, 1088-block tiles and eight march steps, rain streaks
  at 2000 drops, half-length and 80% brightness, 1.1 Purkinje strength, and a
  2.3-block depth-of-field focus that racks in 0.02 s with both blur caps off
  and swirl at 0.7. Lens flare, sharpening, clouds, fog, shafts, water fog and
  Distant Horizons LOD all ship off; the still render ships endless at 3840×2160
  with its denoise off.

### Fixed

- **Boat water patch.** The water patch vanilla draws inside a boat was captured
  into the acceleration structure as a quad and read as an opaque lid over the
  interior; it is now a raster-only mask.
- **Xaero's Minimap.** Waypoint markers kept the rotation of the last pass that
  went through `LevelRenderer`, which a traced frame skips. The hook now hands
  the minimap the world model-view matrix of the pass the ray tracer replaces,
  and is inert when the mod is absent.
- **DLSS-RR specular hit distance.** The shim passed a null specular hit
  distance on every frame; it now passes the mirror layer's depth from the guide
  probe wherever the renderer has one, so a reflection the tracer resolved to a
  mirror surface carries its own distance instead of the layer's.

### Performance

- **One raygen record per frame.** Besides the volume-free and no-march twins,
  the world raygen now has no-probe and no-RIS twins, and the composite picks
  the record that matches the frame's switches: a sky with the radiance cache
  off, or with emitter light samples at zero, no longer carries the code for
  either, and none of it costs registers in the paths that do not use it.


## 0.1.1 — 2026-10-31

### Added

- **Sun shafts.** Tyndall rays traced separately from the height fog, so the air
  can stay clear while the beams stay visible. Eight sun-visibility probes per
  pixel gate the light on the shadow structure along the ray, and intensity,
  density, forward scatter and reach are independent controls. Underwater beams
  run the same integrator on their own medium: a separate density and a ceiling
  on the forward lobe keep the surface you look up at from turning white.
- **Sharpening.** A post-tonemap unsharp mask that restores the texel-scale detail
  DLSS-RR's upscale averages away, with a master switch and an amount dial.
- **Lens flare.** Three artifacts gathered from the exposed image: an
  aperture-blade starburst whose blade count, ray length and strength are
  adjustable, elliptical halos with radial chromatic dispersion, and out-of-focus
  dust on the front element lit by the scene's own bloom. The flare image renders
  at the display size divided by a resolution dial (a sixteenth to a quarter; the
  default quarter keeps rings and rays at their finest), and one sampling dial
  scales both gathers' tap counts. Each artifact has its own switch and every
  control is live: with the starburst and the halos off the chain skips its source
  pass entirely, and one master switch turns the whole chain off.
- **Parallax styles.** The height-field march now runs in one of two modes. Sharp
  snaps a displaced texel onto the shading grid and turns the band a step exposes
  into a wall, so authored hard edges — plank grooves, brick bevels, cut stone —
  read as cut geometry. Smooth leaves the offset continuous and reads the field a
  mip coarser, so the same relief reads as a rounded bevel carried by the material's
  own normal map. Chosen on the Materials page.
- **Parallax self-shadow.** A displaced texel walks toward the sun through the same
  height field, so relief the light cannot reach into darkens. Only surfaces that
  carry a height field pay for it, and only on texels the march displaced; its own
  switch turns it off.
- **AgX Contrast and AgX Highlight Range**, plus a **Bloom Strength** multiplier,
  on the tone-mapping page.
- **Day/Night Fog Falloff**, so the height fog can hold its density around the
  clock or live only around sunrise and sunset.
- **F9 ray-tracing toggle** in the Caustica controls category, with a HUD message
  and a restart hint when the device was created without RT. F3+A clears the
  failure latch and retries the RT path as well.
- **Nether portal emission.** The portal sprite is a 250 cd/m² emitter in the
  material package, so its surface reads as a light source instead of a dark sheet.
- **Per-pass GPU timestamps** with `-Dcaustica.rt.gpuTiming=true`, written for the
  sky LUT, both trace passes, DLSS-RR, exposure, bloom, lens, display and copy into
  `rt-frame-stats/gpu.csv`.

### Changed

- **Clouds rebuilt on baked volumetric volumes** (the UE "clouds_unreal" scheme:
  Houdini-authored VDB clouds raymarched with the Ryan Brucks volumetric
  raymarcher). The procedural streak/coverage field is gone; the layer is now an
  infinite wind-drifted grid of instances, each rendering one of nine baked cloud
  shapes (ships in the jar, ~18 MB) that the march walks with Beer-Lambert
  extinction, a short light march and the existing multi-scatter grading. The
  Cloud Type slider is now **Cloud Shape** and picks the baked shape; coverage
  places instances, a new **Cloud Size** dial sets the instance footprint,
  patchiness clusters them into weather systems and breathes their size, and
  thickness still sets the slab the optical depth scales with. All existing
  switches keep their meaning: the planar tier crosses the mid plane once, the
  volumetric tier walks it, NEE probes and continuation rays keep the mean
  extinction, and a cloud-off frame stays bit-identical to the pre-cloud
  renderer.
- **Water scattering is replaced by the underwater shafts.** The Water Scattering
  slider (`composite.water-turbidity`) is gone; underwater beams now come from the
  Sun Shafts page's Underwater Shaft Density and Scatter Ceiling, which light the
  beams without changing the water body's opacity.
- **Retuned defaults.** The shipped look now uses a quarter-resolution lens flare
  (three blades, 5% strength, halos and dust off, doubled sampling), 50%
  sharpening, parallax disabled (depth 0, eight-layer budget, self-shadow and
  entity parallax off), AgX at 16 stops / 1.05 contrast / 1.4 Purkinje, a
  312-block cloud deck of 400-block instances, 208 blocks thick, at 40% coverage /
  60% density / full patchiness, fading at 5120 blocks on the 0.2 shape, and water
  opacity 1.0 with a 1.3 wave-size scale. Sun shafts start off to keep the frame
  budget for the other effects (when enabled they use 0.55% extinction, isotropic
  phase and 72-block reach); water shafts, fog and clouds are off as well.

### Fixed

- **Entity capture over budget.** When more entities are renderable than the
  capture budget, the capture now walks a nearest-first ranking refreshed at most
  twice a second, so the entities around the camera make it into the traced scene
  instead of whichever ones the world iterator happened to reach first. The budget
  reports how often it truncated and how many entity-frames it dropped, at most
  once per ten seconds.
- **DLSS-RR transparency.** Fully transparent entity texels are now holes in the
  radiance and guide passes alike; the mismatch let RR reconstruct a sharp ghost
  plane over the hole in dark scenes. Fog-dominated pixels blend their guides
  toward the medium (normal, roughness, albedo and specular motion), water specular
  motion follows the same lifted reflection direction as the radiance, thin
  dielectric sheets carry their refracted motion, and the shafts stay out of the
  albedo guide so RR denoises them as light rather than material.
- **Enchantment glint.** Vanilla submits the same model a second time with a
  scrolling glint texture; the duplicate used to sit in the BVH as a coincident
  black sheet. The repeated prims now carry the glint texture slot and UV scale,
  and the hit shader adds the overlay to the face underneath.
- **Hurt and spawn-flash overlays.** The OverlayTexture texel is packed into the
  prim flags, so the red hurt flash and the spawn fade appear on traced entities
  while capes, eyes and item frames keep opting out like vanilla.
- **Entity texture slots.** The bindless array now recycles the least-recently-used
  slot not touched in the current frame and falls back to a reserved white texture,
  so a few hundred maps, paintings or banners no longer starve mob textures into
  atlas blocks or transparent shells. Per-frame energy-swirl render types no longer
  leak a slot each frame.
- **Thin translucent blocks.** Pane-like blocks thinner than a quarter block are
  meshed as filters rather than volume boundaries, so a face vanilla culls can no
  longer read as an exit and total-internal-reflect a hard disc of background onto
  the pane.
- **Water reflection.** Wave-perturbed normals can aim the mirror below its own
  plane; the reflection is now lifted to a grazing floor on the side the ray
  arrived from, so the sea no longer leaks an unattenuated view through its floor.
- **Sky-less dimensions.** A dimension with a black sky colour and no sky light
  (the Nether) now uses the fog colour as its dome and applies the DimensionType
  ambient floor in every branch, so unlit surfaces are no longer pure black.
- **Block sprite gutters.** Material pages fill their gutters with the sprite's
  wrapped continuation instead of a replicated border, removing doubled texels at
  sprite edges under bilinear filtering and the parallax march.
- **FSR frame generation size.** The generated frame and its interpolation source
  are described with the HDR backbuffer's extent rather than the swapchain's, so a
  one-pixel disagreement no longer scales the generated frame vertically.
- **Frame generation with RT off.** Generated frames are skipped while the RT
  master switch is off, so a stale generated frame cannot be presented over the
  vanilla image.
- **RT output mode logging.** The mode line now reports what produced the frame:
  `rt` only with a live Vulkan RT context, otherwise vanilla with the reason (the
  switch is off, no context yet, or an OpenGL backend).

### Performance

- **Volume-free world raygen.** Both world raygen passes compile volume-free twins;
  with fog, clouds, shafts and the volume debug view all off, the composite traces
  those records instead. The fog and shaft integrals never executed in that state,
  but carrying their code cost 5.6 ms: 37.7 → 47.2 fps at a fixed pose on a 6 GB
  laptop. Wave caustics stay in the lean path.
- **Block material page sizing.** Atlases are laid out by a search that minimizes
  page count times page area across the pack's sprite extents instead of always
  using one large page. Vanilla 16 px atlases land in small pages, while 128 px
  packs climb to a page their cells can tile.
- **Entity texture and capture reuse.** Texture slots are recycled per frame, and
  over-budget entity capture reads the cached nearest-first ranking instead of
  re-sorting every frame.
- **No UI blit with RT off.** The UI redirect is skipped when the RT master switch
  is off, removing a full-screen blit that had nothing to composite.
- **Fog noise LOD.** The marched tier skips its FBM where the analytic profile is
  below the noise floor, so steps above the significant band stop paying for noise.

## 0.1.0-bettercaustica — 2026-10-10

First public BetterCaustica build. Based on upstream Caustica `0.1.0`
(`330acd2d`, "Add ACES 2.0 HDR display pipeline").

### Added

- **Volumetric fog.** Analytic height fog integrated inside the path tracer, with
  density, height scale, base height, anisotropy, albedo, sky light and sun-shadow
  controls, plus an optional marched mode that gathers the fog into wind-drifted
  billows. Off by default.
- **Clouds.** A planar cloud layer with coverage, patchiness, altitude and
  thickness controls, weather-driven thickening and an evolution phase. The same
  field shades the sky, casts cloud shadows and dims the fog's sun shafts.
- **Water scattering** inside the volumetric medium.
- **FSR frame generation.** AMD FidelityFX frame generation as a backend that runs
  on any Vulkan 1.3 device, selectable next to DLSS frame generation, self
  interpolation and frame duplication. Bundled only when `FFX_SDK` is set at build
  time; see the developer guide.
- **Fog and cloud debug views** (Fog Transmittance, Fog In-Scatter, Cloud Density,
  Cloud Shadow) on the existing debug-view dial.
- **Mod Menu and Sodium Config API** entry points, so the settings screen is
  reachable from either host.
- **Simplified and Traditional Chinese** option translations.
- Native probes (`ffxprobe`, `matprobe`, `rtprobe`), the FFX shim, and offline
  tooling for LUT baking, cloud-field preview and screenshot capture.

### Changed

- Mod id, resource namespace and jar resource root renamed from `caustica` to
  `bettercaustica`.
- Configuration now lives in `config/bettercaustica.toml`.
- Gallery screenshots re-encoded as JPEG.

### Unchanged on purpose

- The `-Dcaustica.*` system-property overrides, the `text.caustica.*` translation
  keys and the `dev.comfyfluffy.caustica` Java package keep upstream's names.
