# Repository Guidance

This directory is **the single source of the renderer and the package that ships**. As of 2026-10-23
the separate development copy under `../Caustica` was merged into this tree; that directory no longer
holds source (see its own `AGENTS.md`). Build, edit, and release here.

## Identity rules (do not drift)

The release package carries its own identity, and the merge must not erode it:

- Mod id `bettercaustica` — `src/main/resources/fabric.mod.json` and `CausticaMod.MOD_ID`.
- Resource namespace `assets/bettercaustica/`, shader root `/bettercaustica/shaders/`, natives
  `/bettercaustica/natives/`, look package `/bettercaustica/color/`.
- Config file `config/bettercaustica.toml`.
- Unchanged on purpose: the Java package `dev.comfyfluffy.caustica`, the system-property namespace
  `caustica.*`, and the translation keys `caustica.options.*`.

Copying code in from another checkout? Do not bring `caustica` paths, `caustica.toml`, or
`/caustica/shaders/` string literals with it — they resolve to nothing in this package.

## Code rules

- Keep planning documents and `todos.md` local-only. Do not add `docs/*PLAN*.md` or `todos.md` to Git.
- Comments and Javadocs must describe the current implementation and its active invariants only.
- Do not preserve implementation history, migration notes, completed phases, or superseded behavior in
  source comments. Git history owns that context.
- Do not reference internal plan steps, phase labels, milestone IDs, or numbered design-document
  sections from source comments.
- Prefer direct explanations of why the current code is required, especially API contracts,
  synchronization rules, units, and non-obvious constraints.
- Until release, modify only the English locale (`en_us.json`); leave every other locale unchanged.

## Project Information

- This project is a modified version of https://github.com/ComfyFluffy/Caustica, adding visual effect
  optimizations.
- Game directory: `E:\Minecraft\26.2`; launch with
  `C:\Users\mi609\Documents\JAVAMCMOD\mc-mcp\tools\mcbox.py`.
- The local game has built-in MCP tools. Skill documentation:
  `C:\Users\mi609\Documents\JAVAMCMOD\mc-mcp\skill\mc-mcp\SKILL.md`.

## Build and measure

```
./gradlew build -x test                 # jars land in build/libs/
./gradlew build -x test -PshaderDefines=FOO   # A/B build, same compiler and flags
```

`compileShaders` also emits the volume-free twins of the two world raygen passes
(`primary_lean.rgen.spv` / `indirect_lean.rgen.spv`, and the SER variants) from
`-DCAUSTICA_NO_VOLUME_PATH`. They exist so a frame with every volumetric switch off can trace a raygen
that carries no fog/water code — that code costs registers, and registers cost about 5 ms of the frame
on a 6 GB laptop. Keep them compiling when the world shaders change.

`RtGpuTiming` (`-Dcaustica.rt.gpuTiming=true`) writes per-pass GPU timestamps to
`rt-frame-stats/gpu.csv`. The CPU-side stages in `frame.csv` say nothing about a GPU-bound pass.

Before believing any frame-rate number, confirm both log lines:

```
Using graphics backend Vulkan
DLSS-RR feature created: <render> -> <display>
```

Under OpenGL the mod loads and reports `RT output mode: rt` while the ray tracer does nothing at all,
which reads as a suspiciously high frame rate rather than as a failure.
