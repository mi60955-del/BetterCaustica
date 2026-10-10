package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaConfig.BooleanSetting;
import dev.comfyfluffy.caustica.CausticaConfig.FloatSetting;
import dev.comfyfluffy.caustica.CausticaConfig.IntSetting;
import dev.comfyfluffy.caustica.CausticaConfig.RuntimeSetting;
import dev.comfyfluffy.caustica.CausticaConfig.StringSetting;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Supplier;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/**
 * Every Caustica option a player can reach, declared once.
 *
 * <p>Two hosts publish them. With Sodium installed it replaces the vanilla Video Settings screen, and
 * {@link CausticaSodiumConfig} mounts these options inside Sodium's own screen; without Sodium,
 * {@link RtVideoOptions} builds vanilla widgets for the same list and {@code VideoSettingsScreenMixin}
 * puts them in the vanilla screen. Each entry here owns the label, the tooltip, the range, the step and
 * the unit, so the two hosts cannot disagree about what a control <em>is</em> and only decide how to draw
 * it. Before this list both hosts carried their own copy of every option and they had already drifted:
 * gamma ran 0.50..1.50 in one screen and 0.10..5.00 in the other, and the same exposure EV printed as
 * "-3.0 EV" beside "-3.00 EV".
 *
 * <p>Adding an option means adding one line to {@link #ALL}. Nothing else.
 */
public final class CausticaSettings {
    private CausticaSettings() {
    }

    /** One page of the settings surface; both hosts show pages as their own screen or tab. */
    public enum Page {
        RAY_TRACING("raytracing"),
        VOLUME("volume"),
        MATERIALS("materials"),
        ENTITIES("entities"),
        IMAGE("image"),
        WATER("water"),
        DISPLAY("display"),
        DIAGNOSTICS("diagnostics");

        private final String key;

        Page(String key) {
            this.key = key;
        }

        public String titleKey() {
            return "caustica.sodium.page." + key;
        }
    }

    /** How loud Sodium's badge on the option is. The vanilla screen has no equivalent and ignores it. */
    public enum Impact {
        LOW, MEDIUM, HIGH
    }

    /**
     * How a value is written down. One place formats it, so a control reads identically in both hosts;
     * {@code value} is always in the setting's own units, not in the widget's scaled integers.
     */
    public enum Unit {
        COUNT, PERCENT, MULTIPLE, EV, SECONDS, NITS;

        public String format(double value) {
            return switch (this) {
                case COUNT -> Integer.toString((int) Math.round(value));
                case PERCENT -> (int) Math.round(value * 100.0) + "%";
                case MULTIPLE -> twoDecimals(value) + "×";
                case EV -> twoDecimals(value) + " EV";
                case SECONDS -> twoDecimals(value) + " s";
                case NITS -> Integer.toString((int) Math.round(value)) + " nits";
            };
        }

        private static String twoDecimals(double value) {
            return String.format(Locale.ROOT, "%.2f", value);
        }
    }

    /**
     * The control a player drags, and the numbers that bound it. {@link Scaled} carries the widget in
     * hundredths (or whatever {@code scale} says) so the stored float keeps its own precision and the
     * slider keeps a sane step count.
     *
     * <p>The {@code step} these controls carry is enforced natively by Sodium's slider; vanilla's
     * {@code IntRange} has no step, so {@link RtVideoOptions} implements it by snapping values onto
     * the grid instead.
     */
    public sealed interface Control {
        record Toggle() implements Control {
        }

        record Count(int min, int max, int step, Unit unit, String valueKeys) implements Control {
            Count(int min, int max, int step, Unit unit) {
                this(min, max, step, unit, null);
            }

            /**
             * Some integer settings have names worth showing: the debug views are numbered because the
             * renderer switches on the number, but "Exposure EV map" tells a player more than "8". When
             * this is non-null both hosts translate {@code valueKeys + value} instead of printing it.
             */
            boolean named() {
                return valueKeys != null;
            }
        }

        record Scaled(int min, int max, int step, double scale, Unit unit) implements Control {
        }

        /**
         * A choice list. Both hosts cycle it, and each keeps its own value mapping because the stored
         * numbers are not indices: the DLSS steps are NGX enums and the HDR peaks are the mastering
         * targets an ACES output transform was baked for.
         */
        record Cycled(Cycle cycle) implements Control {
        }
    }

    public enum Cycle {
        EXPOSURE_MODE, DLSS_QUALITY, HDR_PEAK
    }

    /**
     * @param path      the dotted location inside {@code config/bettercaustica.toml}, and Sodium's option id
     * @param nameKey   the option's translation key; its tooltip is always {@code nameKey + ".tooltip"}
     * @param available false hides the option from both hosts, for controls that cannot do anything on
     *                  this session's hardware
     */
    public record Option(Page page, String group, String path, String nameKey, Control control,
                         RuntimeSetting<?> setting, Impact impact, boolean restart,
                         Supplier<Boolean> available) {
        public String tooltipKey() {
            return nameKey + ".tooltip";
        }

        public boolean availableNow() {
            return available.get();
        }

        BooleanSetting bool() {
            return (BooleanSetting) setting;
        }

        IntSetting integer() {
            return (IntSetting) setting;
        }

        FloatSetting floating() {
            return (FloatSetting) setting;
        }

        StringSetting text() {
            return (StringSetting) setting;
        }

        /**
         * How the widget shows one of its own values. Both hosts call this and nothing else formats a
         * value, so a control reads identically on either screen: the same unit, the same decimal places,
         * and the same name where the values have names.
         */
        public Component valueLabel(int widgetValue) {
            return switch (control) {
                case Control.Toggle ignored -> Component.empty();
                case Control.Count c when c.named() -> Component.translatable(c.valueKeys() + widgetValue);
                case Control.Count c -> Component.literal(c.unit().format(widgetValue));
                case Control.Scaled c -> Component.literal(c.unit().format(widgetValue / c.scale()));
                case Control.Cycled ignored -> Component.empty();
            };
        }
    }

    /**
     * The options whose value is captured while a device, buffer pool or acceleration structure is built,
     * so changing them only takes effect on the next launch. Named here rather than at each entry because
     * the set is a property of the renderer's lifecycle, not of the control: Sodium shows these with its
     * own restart badge and the vanilla screen appends {@code caustica.options.rt.restartNeeded} to the
     * value, since it has no badge.
     */
    private static final Set<String> RESTART_PATHS = Set.of(
            "worker-threads",
            "omm.subdivision",
            "entities.max-ordinary-entities",
            "entities.block-entities.max-entities",
            "entities.textures.max-textures",
            "entities.textures.pbr",
            "particles.max-particles");

    private static final String RT = "caustica.options.rt.";
    private static final String SETTING = "caustica.setting.";
    private static final Supplier<Boolean> ALWAYS = () -> Boolean.TRUE;
    // The Purkinje shift adapts to the metered absolute scene luminance, which the exposure
    // controller only refreshes in auto mode; in manual mode it has no signal to work from.
    private static final Supplier<Boolean> AUTO_EXPOSURE =
            () -> "auto".equalsIgnoreCase(CausticaConfig.Rt.Exposure.MODE.get());
    private static final Supplier<Boolean> HDR_CAPABLE = CausticaConfig.Rt.Hdr::swapchainPqAvailable;

    public static final List<Option> ALL = build();

    /** The entries in display order; {@link #groups} preserves it within each page. */
    private static List<Option> build() {
        List<Option> out = new ArrayList<>(96);

        // ---- Ray tracing ---------------------------------------------------------------------------
        toggle(out, Page.RAY_TRACING, "core", "enabled", SETTING + "enabled", CausticaConfig.Rt.ENABLED);
        count(out, Page.RAY_TRACING, "core", "composite.spp", RT + "spp", CausticaConfig.Rt.Composite.SPP,
                1, 16, 1, Impact.HIGH);
        count(out, Page.RAY_TRACING, "core", "composite.max-bounces", RT + "maxBounces",
                CausticaConfig.Rt.Composite.MAX_BOUNCES, 2, 8, 1, Impact.HIGH);
        count(out, Page.RAY_TRACING, "core", "worker-threads", SETTING + "worker-threads",
                CausticaConfig.Rt.WORKER_THREADS, 1, 16, 1, Impact.MEDIUM);
        count(out, Page.RAY_TRACING, "core", "composite.debug-view", RT + "debugView",
                CausticaConfig.Rt.Composite.DEBUG_VIEW, 0, 13, 1, Impact.LOW, RT + "debugView.");
        count(out, Page.RAY_TRACING, "core", "omm.subdivision", SETTING + "ommSubdivision",
                CausticaConfig.Rt.Omm.SUBDIVISION, 0, 6, 1, Impact.MEDIUM);
        toggle(out, Page.RAY_TRACING, "core", "omm.stats", SETTING + "ommStats",
                CausticaConfig.Rt.Omm.STATS);
        count(out, Page.RAY_TRACING, "terrain", "terrain.async-dispatch-per-tick",
                SETTING + "asyncDispatch", CausticaConfig.Rt.Terrain.ASYNC_DISPATCH_PER_PASS,
                0, 256, 1, Impact.MEDIUM);
        count(out, Page.RAY_TRACING, "terrain", "terrain.section-results-per-tick",
                SETTING + "completionResults", CausticaConfig.Rt.Terrain.COMPLETION_RESULTS_PER_PASS,
                0, 256, 1, Impact.MEDIUM);
        count(out, Page.RAY_TRACING, "terrain", "terrain.max-inflight-sections", SETTING + "maxInflight",
                CausticaConfig.Rt.Terrain.MAX_INFLIGHT_SECTIONS, 0, 256, 1, Impact.MEDIUM);
        count(out, Page.RAY_TRACING, "terrain", "terrain.section-table-initial-capacity",
                SETTING + "sectionTableCapacity", CausticaConfig.Rt.Terrain.SECTION_TABLE_INITIAL_CAPACITY,
                1, 8192, 1, Impact.LOW);
        count(out, Page.RAY_TRACING, "terrain", "terrain.rebase-distance-blocks", SETTING + "rebaseDistance",
                CausticaConfig.Rt.Terrain.REBASE_DISTANCE_BLOCKS, 0, 1024, 16, Impact.LOW);
        toggle(out, Page.RAY_TRACING, "terrain", "terrain.blas-compaction", SETTING + "blasCompaction",
                CausticaConfig.Rt.Terrain.BLAS_COMPACTION);
        count(out, Page.RAY_TRACING, "lights", "lights.ris-candidates", RT + "risCandidates",
                CausticaConfig.Rt.Lights.RIS_CANDIDATES, 0, 16, 1, Impact.MEDIUM);
        scaled(out, Page.RAY_TRACING, "lights", "lights.min-fill-ratio", SETTING + "minFillRatio",
                CausticaConfig.Rt.Lights.MIN_FILL_RATIO, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW);
        toggle(out, Page.RAY_TRACING, "lights", "lights.temporal-reuse", SETTING + "lightTemporalReuse",
                CausticaConfig.Rt.Lights.TEMPORAL_REUSE);
        count(out, Page.RAY_TRACING, "lights", "lights.temporal-history", SETTING + "lightTemporalHistory",
                CausticaConfig.Rt.Lights.TEMPORAL_HISTORY, 8, 256, 8, Impact.LOW);
        toggle(out, Page.RAY_TRACING, "lights", "lights.stats", SETTING + "lightStats",
                CausticaConfig.Rt.Lights.STATS);
        toggle(out, Page.RAY_TRACING, "lights", "lights.dump", SETTING + "lightDump",
                CausticaConfig.Rt.Lights.DUMP);
        count(out, Page.RAY_TRACING, "lights", "lights.dump-radius", SETTING + "lightDumpRadius",
                CausticaConfig.Rt.Lights.DUMP_RADIUS, 1, 64, 1, Impact.LOW);

        // ---- Volumetrics: the air around the terrain ------------------------------------------------
        // The analytic height fog. Every control here feeds the WorldPush volume block the volume
        // shaders read; the master switch defaults to off, so the group is inert until a player opens
        // it. It owns its own page: the group outgrew the Ray Tracing page's tail and the density model
        // (day curve, shadow culling) keeps growing around it.

        // The tier preset heads the group: it is the one control here that writes the others, so a player
        // who touches nothing else can read the whole system's cost ladder off it. A count with value keys
        // rather than a cycle list, the same shape the debug view uses for its named views.
        count(out, Page.VOLUME, "volumetrics", "volume.quality", RT + "volumeQuality",
                CausticaConfig.Rt.Volume.QUALITY, 0, 4, 1, Impact.HIGH, RT + "volumeQuality.");
        toggle(out, Page.VOLUME, "volumetrics", "volume.fog", RT + "fog",
                CausticaConfig.Rt.Volume.FOG_ENABLED);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-density", RT + "fogDensity",
                CausticaConfig.Rt.Volume.FOG_DENSITY, 0, 250, 5, 1000.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-height", RT + "fogHeight",
                CausticaConfig.Rt.Volume.FOG_HEIGHT, 2, 128, 2, 1.0, Unit.COUNT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-base-y", RT + "fogBaseY",
                CausticaConfig.Rt.Volume.FOG_BASE_Y, -64, 320, 2, 1.0, Unit.COUNT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-anisotropy", RT + "fogAnisotropy",
                CausticaConfig.Rt.Volume.FOG_ANISOTROPY, -90, 90, 5, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-albedo", RT + "fogAlbedo",
                CausticaConfig.Rt.Volume.FOG_ALBEDO, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-ambient", RT + "fogAmbient",
                CausticaConfig.Rt.Volume.FOG_AMBIENT, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);
        toggle(out, Page.VOLUME, "volumetrics", "volume.fog-shadow", RT + "fogShadow",
                CausticaConfig.Rt.Volume.FOG_SHADOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-shadow-floor", RT + "fogShadowFloor",
                CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR, 0, 95, 5, 100.0, Unit.PERCENT, Impact.LOW);
        toggle(out, Page.VOLUME, "volumetrics", "volume.fog-march", RT + "fogMarch",
                CausticaConfig.Rt.Volume.FOG_MARCH);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-noise", RT + "fogNoise",
                CausticaConfig.Rt.Volume.FOG_NOISE, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-wind-speed", RT + "fogWindSpeed",
                CausticaConfig.Rt.Volume.FOG_WIND_SPEED, 0, 32, 1, 1.0, Unit.COUNT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-wind-angle", RT + "fogWindAngle",
                CausticaConfig.Rt.Volume.FOG_WIND_ANGLE, 0, 360, 15, 1.0, Unit.COUNT, Impact.LOW);
        count(out, Page.VOLUME, "volumetrics", "volume.fog-march-steps", RT + "fogMarchSteps",
                CausticaConfig.Rt.Volume.FOG_MARCH_STEPS, 4, 64, 4, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-jitter", RT + "fogJitter",
                CausticaConfig.Rt.Volume.FOG_JITTER, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.fog-rain-boost", RT + "fogRainBoost",
                CausticaConfig.Rt.Volume.FOG_RAIN_BOOST, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);

        // ---- Clouds: the planar sheet and its shadow field ------------------------------------------
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-coverage", RT + "cloudCoverage",
                CausticaConfig.Rt.Volume.CLOUD_COVERAGE, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-patchiness", RT + "cloudPatchiness",
                CausticaConfig.Rt.Volume.CLOUD_PATCHINESS, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-altitude", RT + "cloudAltitude",
                CausticaConfig.Rt.Volume.CLOUD_ALTITUDE, 0, 512, 8, 1.0, Unit.COUNT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-thickness", RT + "cloudThickness",
                CausticaConfig.Rt.Volume.CLOUD_THICKNESS, 1, 256, 1, 1.0, Unit.COUNT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-density", RT + "cloudDensity",
                CausticaConfig.Rt.Volume.CLOUD_DENSITY, 0, 300, 10, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-wind-speed", RT + "cloudWindSpeed",
                CausticaConfig.Rt.Volume.CLOUD_WIND_SPEED, 0, 64, 1, 1.0, Unit.COUNT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-wind-angle", RT + "cloudWindAngle",
                CausticaConfig.Rt.Volume.CLOUD_WIND_ANGLE, 0, 360, 15, 1.0, Unit.COUNT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-fade-distance", RT + "cloudFadeDistance",
                CausticaConfig.Rt.Volume.CLOUD_FADE_DISTANCE, 512, 16384, 256, 1.0, Unit.COUNT, Impact.LOW);
        toggle(out, Page.VOLUME, "volumetrics", "volume.cloud", RT + "cloud",
                CausticaConfig.Rt.Volume.CLOUD_ENABLED);
        toggle(out, Page.VOLUME, "volumetrics", "volume.cloud-volumetric", RT + "cloudVolumetric",
                CausticaConfig.Rt.Volume.CLOUD_VOLUMETRIC);
        count(out, Page.VOLUME, "volumetrics", "volume.cloud-steps", RT + "cloudSteps",
                CausticaConfig.Rt.Volume.CLOUD_STEPS, 8, 48, 4, Impact.HIGH);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-type", RT + "cloudType",
                CausticaConfig.Rt.Volume.CLOUD_TYPE, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.VOLUME, "volumetrics", "volume.cloud-rain-boost", RT + "cloudRainBoost",
                CausticaConfig.Rt.Volume.CLOUD_RAIN_BOOST, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);

        // ---- Materials: what a surface is made of ---------------------------------------------------
        // Everything here describes the surface itself rather than the world around it, which is why the
        // parallax relief and the surface wetness sit here instead of inside the terrain streaming group or
        // the water wave group. The relief group shapes the height-field march; the PBR group decides which
        // authored channels of a material the renderer honours, one switch per channel so a player can
        // isolate any single one against the rest.
        scaled(out, Page.MATERIALS, "parallax", "composite.parallax-depth", RT + "parallaxDepth",
                CausticaConfig.Rt.Composite.PARALLAX_DEPTH, 0, 50, 1, 100.0, Unit.PERCENT, Impact.MEDIUM);
        count(out, Page.MATERIALS, "parallax", "materials.parallax-layers", RT + "parallaxLayers",
                CausticaConfig.Rt.Materials.PARALLAX_LAYERS, 8, 64, 4, Impact.MEDIUM);
        toggle(out, Page.MATERIALS, "parallax", "materials.entity-parallax", RT + "entityParallax",
                CausticaConfig.Rt.Materials.ENTITY_PARALLAX);
        toggle(out, Page.MATERIALS, "labpbr", "materials.texture-emission", RT + "textureEmission",
                CausticaConfig.Rt.Materials.TEXTURE_EMISSION);
        toggle(out, Page.MATERIALS, "labpbr", "materials.block-emission", RT + "blockEmission",
                CausticaConfig.Rt.Materials.BLOCK_EMISSION);
        toggle(out, Page.MATERIALS, "labpbr", "materials.subsurface", RT + "subsurface",
                CausticaConfig.Rt.Materials.SUBSURFACE);
        toggle(out, Page.MATERIALS, "labpbr", "materials.porosity", RT + "porosity",
                CausticaConfig.Rt.Materials.POROSITY);
        scaled(out, Page.MATERIALS, "labpbr", "composite.wetness", RT + "wetness",
                CausticaConfig.Rt.Composite.WETNESS, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);

        // ---- Entities, particles and overlays ------------------------------------------------------
        toggle(out, Page.ENTITIES, "entities", "entities.enabled", RT + "entities",
                CausticaConfig.Rt.Entities.ENABLED);
        toggle(out, Page.ENTITIES, "entities", "entities.trace-self.enabled", RT + "traceSelf",
                CausticaConfig.Rt.Entities.SELF_ENABLED);
        toggle(out, Page.ENTITIES, "entities", "entities.glow.enabled", RT + "glow",
                CausticaConfig.Rt.Entities.GLOW_ENABLED);
        toggle(out, Page.ENTITIES, "entities", "entities.name-tags.enabled", SETTING + "nameTags",
                CausticaConfig.Rt.Entities.NAME_TAGS_ENABLED);
        toggle(out, Page.ENTITIES, "entities", "entities.refit.enabled", SETTING + "entityRefit",
                CausticaConfig.Rt.Entities.REFIT_ENABLED);
        count(out, Page.ENTITIES, "entities", "entities.max-ordinary-entities", SETTING + "maxOrdinary",
                CausticaConfig.Rt.Entities.MAX_ORDINARY_ENTITIES, 0, 4096, 16, Impact.MEDIUM);
        count(out, Page.ENTITIES, "entities", "entities.block-entities.max-entities",
                SETTING + "maxBlockEntities", CausticaConfig.Rt.Entities.MAX_BLOCK_ENTITIES,
                0, 4096, 16, Impact.MEDIUM);
        count(out, Page.ENTITIES, "entities", "entities.block-entities.view-chunks",
                SETTING + "beViewChunks", CausticaConfig.Rt.Entities.BE_VIEW_CHUNKS, 0, 64, 1,
                Impact.MEDIUM);
        count(out, Page.ENTITIES, "entities", "entities.block-entities.builds-per-frame",
                SETTING + "beBuilds", CausticaConfig.Rt.Entities.BE_BUILDS_PER_FRAME, 0, 512, 8,
                Impact.MEDIUM);
        count(out, Page.ENTITIES, "entities", "entities.textures.max-textures",
                SETTING + "maxEntityTextures", CausticaConfig.Rt.EntityTextures.MAX_TEXTURES, 1, 4096, 1,
                Impact.LOW);
        toggle(out, Page.ENTITIES, "entities", "entities.textures.pbr", SETTING + "entityPbr",
                CausticaConfig.Rt.EntityTextures.PBR);
        toggle(out, Page.ENTITIES, "entities", "entities.debug.capture-parity", SETTING + "captureParity",
                CausticaConfig.Rt.Entities.CAPTURE_PARITY);
        toggle(out, Page.ENTITIES, "particles", "particles.enabled", RT + "particles",
                CausticaConfig.Rt.Entities.PARTICLES_ENABLED);
        count(out, Page.ENTITIES, "particles", "particles.max-particles", SETTING + "maxParticles",
                CausticaConfig.Rt.Entities.MAX_PARTICLES, 0, 8192, 32, Impact.MEDIUM);
        toggle(out, Page.ENTITIES, "overlay", "overlay.block-outline.enabled", SETTING + "blockOutline",
                CausticaConfig.Rt.Overlay.BLOCK_OUTLINE_ENABLED);

        // ---- Image: exposure, grading, sky ---------------------------------------------------------
        cycled(out, Page.IMAGE, "exposure", "exposure.mode", RT + "exposureMode", Cycle.EXPOSURE_MODE,
                CausticaConfig.Rt.Exposure.MODE, Impact.MEDIUM);
        scaled(out, Page.IMAGE, "exposure", "exposure.manual-ev", RT + "manualEv",
                CausticaConfig.Rt.Exposure.MANUAL_EV, -1500, 1500, 5, 100.0, Unit.EV, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.key", SETTING + "exposureKey",
                CausticaConfig.Rt.Exposure.KEY, 5, 500, 1, 100.0, Unit.MULTIPLE, Impact.MEDIUM);
        scaled(out, Page.IMAGE, "exposure", "exposure.night-lift", RT + "nightLift",
                CausticaConfig.Rt.Exposure.NIGHT_LIFT, -600, 600, 5, 100.0, Unit.EV, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.adapt-darken", SETTING + "adaptDarken",
                CausticaConfig.Rt.Exposure.ADAPT_DARKEN, 0, 1000, 5, 100.0, Unit.SECONDS, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.adapt-brighten", SETTING + "adaptBrighten",
                CausticaConfig.Rt.Exposure.ADAPT_BRIGHTEN, 0, 1000, 5, 100.0, Unit.SECONDS, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.low-percentile", SETTING + "lowPercentile",
                CausticaConfig.Rt.Exposure.LOW_PERCENTILE, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.high-percentile", SETTING + "highPercentile",
                CausticaConfig.Rt.Exposure.HIGH_PERCENTILE, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW);
        count(out, Page.IMAGE, "exposure", "exposure.stride", SETTING + "exposureStride",
                CausticaConfig.Rt.Exposure.STRIDE, 1, 8, 1, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.center-weight-sigma", SETTING + "centerSigma",
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_SIGMA, 1, 200, 1, 100.0, Unit.MULTIPLE,
                Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.center-weight-floor", SETTING + "centerFloor",
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_FLOOR, 0, 100, 1, 100.0, Unit.PERCENT,
                Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.sky-weight-cap", SETTING + "skyWeightCap",
                CausticaConfig.Rt.Exposure.SKY_WEIGHT_CAP, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.IMAGE, "exposure", "exposure.emissive-weight-cap", SETTING + "emissiveWeightCap",
                CausticaConfig.Rt.Exposure.EMISSIVE_WEIGHT_CAP, 0, 100, 1, 100.0, Unit.PERCENT,
                Impact.LOW);
        toggle(out, Page.IMAGE, "exposure", "exposure.pre-exposure", SETTING + "preExposure",
                CausticaConfig.Rt.Exposure.PRE_EXPOSURE);
        scaled(out, Page.IMAGE, "tonemap", "tonemap.gamma", RT + "gamma", CausticaConfig.Rt.Tonemap.GAMMA,
                10, 500, 5, 100.0, Unit.MULTIPLE, Impact.LOW);
        toggle(out, Page.IMAGE, "tonemap", "tonemap.purkinje", RT + "purkinje",
                CausticaConfig.Rt.Tonemap.PURKINJE, AUTO_EXPOSURE);
        scaled(out, Page.IMAGE, "tonemap", "tonemap.purkinje-strength", RT + "purkinjeStrength",
                CausticaConfig.Rt.Tonemap.PURKINJE_STRENGTH, 0, 200, 5, 100.0, Unit.MULTIPLE,
                Impact.LOW, AUTO_EXPOSURE);
        toggle(out, Page.IMAGE, "tonemap", "tonemap.advanced-grading", RT + "advancedGrading",
                CausticaConfig.Rt.Tonemap.ADVANCED_GRADING);
        scaled(out, Page.IMAGE, "tonemap", "tonemap.agx-gain", RT + "agxExposure",
                CausticaConfig.Rt.Tonemap.AGX_GAIN, 2, 800, 1, 100.0, Unit.MULTIPLE, Impact.LOW);
        scaled(out, Page.IMAGE, "tonemap", "tonemap.agx-saturation", RT + "agxSaturation",
                CausticaConfig.Rt.Tonemap.AGX_SATURATION, 0, 300, 1, 100.0, Unit.MULTIPLE, Impact.LOW);
        toggle(out, Page.IMAGE, "tonemap", "tonemap.look-enabled", RT + "look",
                CausticaConfig.Rt.Tonemap.LOOK_ENABLED);
        toggle(out, Page.IMAGE, "sky", "sky.vanilla-sun", RT + "vanillaSun",
                CausticaConfig.Rt.Sky.VANILLA_SUN);
        toggle(out, Page.IMAGE, "sky", "sky.vanilla-moon", RT + "vanillaMoon",
                CausticaConfig.Rt.Sky.VANILLA_MOON);

        // ---- Water ---------------------------------------------------------------------------------
        toggle(out, Page.WATER, "water", "composite.water-waves", RT + "waterWaves",
                CausticaConfig.Rt.Composite.WATER_WAVES);
        scaled(out, Page.WATER, "water", "composite.water-opacity", RT + "waterOpacity",
                CausticaConfig.Rt.Composite.WATER_OPACITY, 0, 400, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.WATER, "water", "composite.water-turbidity", RT + "waterTurbidity",
                CausticaConfig.Rt.Composite.WATER_TURBIDITY, 0, 100, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM);
        scaled(out, Page.WATER, "water", "composite.water-spray", RT + "waterSpray",
                CausticaConfig.Rt.Composite.WATER_SPRAY, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.WATER, "water", "composite.water-wave-height", RT + "waterWaveHeight",
                CausticaConfig.Rt.Composite.WATER_WAVE_HEIGHT, 0, 300, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM);
        scaled(out, Page.WATER, "water", "composite.water-wave-scale", RT + "waterWaveSize",
                CausticaConfig.Rt.Composite.WATER_WAVE_SCALE, 25, 400, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM);
        scaled(out, Page.WATER, "water", "composite.water-wave-speed", RT + "waterWaveSpeed",
                CausticaConfig.Rt.Composite.WATER_WAVE_SPEED, 0, 300, 5, 100.0, Unit.PERCENT, Impact.LOW);
        scaled(out, Page.WATER, "water", "composite.water-reflection", RT + "waterReflection",
                CausticaConfig.Rt.Composite.WATER_REFLECTION, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);

        // ---- Display: upscaling, frame generation, latency, HDR ------------------------------------
        toggle(out, Page.DISPLAY, "upscaling", "dlss-rr.enabled", SETTING + "dlssRr",
                CausticaConfig.Rt.DlssRr.ENABLED);
        cycled(out, Page.DISPLAY, "upscaling", "dlss-rr.quality", RT + "dlssQuality", Cycle.DLSS_QUALITY,
                CausticaConfig.Rt.DlssRr.QUALITY, Impact.HIGH);
        count(out, Page.DISPLAY, "upscaling", "dlss-rr.preset", SETTING + "dlssPreset",
                CausticaConfig.Rt.DlssRr.PRESET, 0, 8, 1, Impact.LOW);
        toggle(out, Page.DISPLAY, "frame-generation", "frame-generation.enabled",
                SETTING + "frameGeneration", CausticaConfig.Rt.Fg.ENABLED);
        count(out, Page.DISPLAY, "frame-generation", "frame-generation.multi-frame-count",
                SETTING + "multiFrameCount", CausticaConfig.Rt.Fg.MULTI_FRAME_COUNT, 1, 3, 1,
                Impact.MEDIUM);
        toggle(out, Page.DISPLAY, "reflex", "reflex.enabled", SETTING + "reflex",
                CausticaConfig.Rt.Reflex.ENABLED);
        toggle(out, Page.DISPLAY, "reflex", "reflex.low-latency-boost", SETTING + "reflexBoost",
                CausticaConfig.Rt.Reflex.LOW_LATENCY_BOOST);
        count(out, Page.DISPLAY, "reflex", "reflex.minimum-interval-us", SETTING + "reflexInterval",
                CausticaConfig.Rt.Reflex.MINIMUM_INTERVAL_US, 0, 100000, 500, Impact.LOW);
        toggle(out, Page.DISPLAY, "hdr", "hdr.enabled", RT + "hdr", CausticaConfig.Rt.Hdr.ENABLED,
                HDR_CAPABLE);
        scaled(out, Page.DISPLAY, "hdr", "hdr.ui-nits", RT + "hdrUiBrightness",
                CausticaConfig.Rt.Hdr.UI_NITS, 80, 500, 5, 1.0, Unit.NITS, Impact.LOW, HDR_CAPABLE);
        cycled(out, Page.DISPLAY, "hdr", "hdr.peak-nits", RT + "hdrPeak", Cycle.HDR_PEAK,
                CausticaConfig.Rt.Hdr.PEAK_NITS, Impact.LOW, HDR_CAPABLE);

        // ---- Diagnostics ---------------------------------------------------------------------------
        scaled(out, Page.DIAGNOSTICS, "jitter", "composite.jitter-sign-x", SETTING + "jitterSignX",
                CausticaConfig.Rt.Composite.JITTER_SIGN_X, -100, 100, 100, 100.0, Unit.MULTIPLE,
                Impact.HIGH);
        scaled(out, Page.DIAGNOSTICS, "jitter", "composite.jitter-sign-y", SETTING + "jitterSignY",
                CausticaConfig.Rt.Composite.JITTER_SIGN_Y, -100, 100, 100, 100.0, Unit.MULTIPLE,
                Impact.HIGH);
        toggle(out, Page.DIAGNOSTICS, "logging", "frame-stats.enabled", SETTING + "frameStats",
                CausticaConfig.Rt.FrameStats.ENABLED);
        toggle(out, Page.DIAGNOSTICS, "logging", "screenshots.exr-enabled", SETTING + "exr",
                CausticaConfig.Rt.Screenshots.EXR_ENABLED);
        toggle(out, Page.DIAGNOSTICS, "logging", "diagnostics.heavy-crash-diagnostics",
                SETTING + "heavyDiagnostics", CausticaConfig.Rt.Diagnostics.HEAVY_CRASH_DIAGNOSTICS);

        return List.copyOf(out);
    }

    /** The entries for one page, in declaration order. */
    public static List<Option> options(Page page) {
        return ALL.stream().filter(o -> o.page() == page && o.availableNow()).toList();
    }

    /** The group keys used by one page, in first-use order. */
    public static List<String> groups(Page page) {
        List<String> order = new ArrayList<>();
        for (Option option : options(page)) {
            if (!order.contains(option.group())) {
                order.add(option.group());
            }
        }
        return order;
    }

    /**
     * The stored exposure-mode word, normalized to one of the two legal values. The TOML is a supported
     * interface, so a hand-edited value has to reach both hosts' widgets as something they can show.
     */
    public static String normalizeExposureMode(String stored) {
        return "manual".equalsIgnoreCase(stored) ? "manual" : "auto";
    }

    public static String groupKey(String group) {
        return "caustica.sodium.group." + group;
    }

    /**
     * The options whose new value cannot reach the screen until the swapchain is rebuilt — HDR/PQ is a
     * property of the surface, not of a per-frame push constant.
     */
    private static final Set<String> SURFACE_RECREATE_PATHS = Set.of("hdr.enabled");

    /**
     * Anything a committed change has to do beyond writing the setting. Both hosts call this after every
     * write, so a side effect belongs to the option rather than to whichever screen happened to be open:
     * changing {@code hdr.enabled} from Sodium's screen used to leave the swapchain in SDR until the next
     * resize, while the same change from the vanilla screen rebuilt it.
     */
    public static void afterChange(Option option) {
        // The quality preset is a write of several settings disguised as one control, so its side effect
        // lives here rather than in whichever screen happened to be open — the same reasoning the surface
        // rebuild below follows.
        if (CausticaConfig.Rt.Volume.QUALITY.tomlPath().equals(option.path())) {
            CausticaConfig.Rt.Volume.applyQuality(option.integer().get());
        }
        if (SURFACE_RECREATE_PATHS.contains(option.path())) {
            // GpuSurface refuses configure() while an image is acquired, so this asks for the rebuild at
            // the next safe frame boundary instead of doing it here.
            Minecraft.getInstance().invalidateSurfaceConfiguration();
        }
    }

    // ---- Declaration helpers -----------------------------------------------------------------------

    private static Option toggle(List<Option> out, Page page, String group, String path, String nameKey,
                                 BooleanSetting setting) {
        return add(out, page, group, path, nameKey, new Control.Toggle(), setting, Impact.LOW, ALWAYS);
    }

    private static Option toggle(List<Option> out, Page page, String group, String path, String nameKey,
                                 BooleanSetting setting, Supplier<Boolean> available) {
        return add(out, page, group, path, nameKey, new Control.Toggle(), setting, Impact.LOW, available);
    }

    private static Option count(List<Option> out, Page page, String group, String path, String nameKey,
                                IntSetting setting, int min, int max, int step, Impact impact) {
        return add(out, page, group, path, nameKey, new Control.Count(min, max, step, Unit.COUNT),
                setting, impact, ALWAYS);
    }

    /** As above, for an integer setting whose values have names worth showing instead of numbers. */
    private static Option count(List<Option> out, Page page, String group, String path, String nameKey,
                                IntSetting setting, int min, int max, int step, Impact impact,
                                String valueKeys) {
        return add(out, page, group, path, nameKey,
                new Control.Count(min, max, step, Unit.COUNT, valueKeys), setting, impact, ALWAYS);
    }

    private static Option scaled(List<Option> out, Page page, String group, String path, String nameKey,
                                 FloatSetting setting, int min, int max, int step, double scale, Unit unit,
                                 Impact impact) {
        return scaled(out, page, group, path, nameKey, setting, min, max, step, scale, unit, impact,
                ALWAYS);
    }

    private static Option scaled(List<Option> out, Page page, String group, String path, String nameKey,
                                 FloatSetting setting, int min, int max, int step, double scale, Unit unit,
                                 Impact impact, Supplier<Boolean> available) {
        return add(out, page, group, path, nameKey, new Control.Scaled(min, max, step, scale, unit),
                setting, impact, available);
    }

    private static Option cycled(List<Option> out, Page page, String group, String path, String nameKey,
                                 Cycle cycle, RuntimeSetting<?> setting, Impact impact) {
        return cycled(out, page, group, path, nameKey, cycle, setting, impact, ALWAYS);
    }

    private static Option cycled(List<Option> out, Page page, String group, String path, String nameKey,
                                 Cycle cycle, RuntimeSetting<?> setting, Impact impact,
                                 Supplier<Boolean> available) {
        return add(out, page, group, path, nameKey, new Control.Cycled(cycle), setting, impact, available);
    }

    private static Option add(List<Option> out, Page page, String group, String path, String nameKey,
                              Control control, RuntimeSetting<?> setting, Impact impact,
                              Supplier<Boolean> available) {
        Option option = new Option(page, group, path, nameKey, control, setting, impact,
                RESTART_PATHS.contains(path), available);
        out.add(option);
        return option;
    }

}
