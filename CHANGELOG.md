# Changelog

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
