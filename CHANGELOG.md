# Changelog

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
