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

    /** A landing-screen band, grouping pages by what the player is trying to do. */
    public enum Section {
        RENDERING("rendering"),
        IMAGE("image"),
        OUTPUT("output");

        private final String key;

        Section(String key) {
            this.key = key;
        }

        public String titleKey() {
            return "caustica.sodium.section." + key;
        }
    }

    /** One page of the settings surface; both hosts show pages as their own screen or tab. */
    public enum Page {
        RAY_TRACING("raytracing", Section.RENDERING),
        VOLUME("volume", Section.RENDERING),
        SKY("sky", Section.RENDERING),
        MATERIALS("materials", Section.RENDERING),
        ENTITIES("entities", Section.RENDERING),
        WATER("water", Section.RENDERING),
        EXPOSURE("exposure", Section.IMAGE),
        COLOR("color", Section.IMAGE),
        LENS("lens", Section.IMAGE),
        DOF("dof", Section.IMAGE),
        OUTPUT("display", Section.OUTPUT),
        STILL("still", Section.OUTPUT),
        DIAGNOSTICS("diagnostics", Section.OUTPUT);

        private final String key;
        private final Section section;

        Page(String key, Section section) {
            this.key = key;
            this.section = section;
        }

        public String titleKey() {
            return "caustica.sodium.page." + key;
        }

        public Section section() {
            return section;
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
        COUNT, PERCENT, MULTIPLE, EV, SECONDS, NITS, CENTIMETRES, DEGREES, BLOCKS, PER_BLOCK;

        public String format(double value) {
            return switch (this) {
                case COUNT -> Integer.toString((int) Math.round(value));
                case PERCENT -> (int) Math.round(value * 100.0) + "%";
                case MULTIPLE -> twoDecimals(value) + "×";
                case EV -> twoDecimals(value) + " EV";
                case SECONDS -> twoDecimals(value) + " s";
                case NITS -> Integer.toString((int) Math.round(value)) + " nits";
                // A length the renderer keeps in blocks, read out in the unit the thing it describes is
                // actually measured in: one block is one metre, so the lens radius is a centimetre count.
                case CENTIMETRES -> Integer.toString((int) Math.round(value * 100.0)) + " cm";
                // An angle the player dials in whole degrees, which is the unit an aperture's blade
                // rotation is read by eye.
                case DEGREES -> Integer.toString((int) Math.round(value)) + "°";
                // A distance the player dials in tenths of a block: focus is judged by eye against
                // surfaces only a fraction of a block apart, where whole-block steps skip past the
                // sharp plane. The block is the renderer's own length unit, so no suffix is needed.
                case BLOCKS -> oneDecimal(value);
                // A medium's extinction per block, whose useful values all sit below one percent: the
                // fog's Rayleigh density reads 0.00050 here and would round to "0%" in percent.
                case PER_BLOCK -> String.format(Locale.ROOT, "%.5f", value) + " /block";
            };
        }

        private static String twoDecimals(double value) {
            return String.format(Locale.ROOT, "%.2f", value);
        }

        private static String oneDecimal(double value) {
            return String.format(Locale.ROOT, "%.1f", value);
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
        EXPOSURE_MODE, DLSS_QUALITY, HDR_PEAK, FRAME_GENERATION_BACKEND, DOF_MODE, PARALLAX_STYLE
    }

    /**
     * A parent option this one only means something under. {@code parentPath} is the parent's TOML path,
     * which is also its Sodium id: Sodium re-evaluates the row when that option's value changes, and the
     * vanilla screen looks the parent up by the same string.
     *
     * <p>The parent being off greys the row rather than removing it. A control that vanishes takes the
     * relationship with it — the player cannot see that there is anything to turn on first.
     */
    public record Dependency(String parentPath, Supplier<Boolean> parentOn) {
        public boolean satisfied() {
            return parentOn.get();
        }
    }

    /**
     * Why an option may be unavailable, in the two senses that used to be one field.
     *
     * <p>{@code hardware} is about the machine: no PQ swapchain, no OMM support. The row is not drawn at
     * all, because there is nothing the player could do with it. {@code dependsOn} is about another
     * option: the row stays and is greyed, because the switch that wakes it is on the same page.
     */
    public record Gate(Supplier<Boolean> hardware, Dependency dependsOn) {
        public static final Gate OPEN = new Gate(() -> Boolean.TRUE, null);

        /** Not offered on this session's hardware; the row is absent. */
        public static Gate requires(Supplier<Boolean> capability) {
            return new Gate(capability, null);
        }

        /** Offered, but greyed until {@code parentPath} is on. */
        public static Gate under(String parentPath, Supplier<Boolean> parentOn) {
            return new Gate(() -> Boolean.TRUE, new Dependency(parentPath, parentOn));
        }
    }

    /**
     * @param path      the dotted location inside {@code config/bettercaustica.toml}, and Sodium's option id
     * @param nameKey   the option's translation key; its tooltip is always {@code nameKey + ".tooltip"}
     * @param gate      whether the row exists at all, and what greys it
     * @param readout   non-null when the row reports something other than its own stored number; the
     *                  quality preset uses it to name the tier the individual dials currently spell
     */
    public record Option(Page page, String group, String path, String nameKey, Control control,
                         RuntimeSetting<?> setting, Impact impact, boolean restart,
                         Gate gate, Supplier<Component> readout) {
        public String tooltipKey() {
            return nameKey + ".tooltip";
        }

        public boolean offeredNow() {
            return gate.hardware().get();
        }

        /** Whether the row is operative: hardware present and its parent switch on. */
        public boolean enabledNow() {
            return offeredNow() && (gate.dependsOn() == null || gate.dependsOn().satisfied());
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
            if (readout != null) {
                return readout.get();
            }
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
     * The options whose value is captured while a device, buffer pool, acceleration structure or vendor
     * frame-generation feature is built, so changing them only takes effect on the next launch. Named here
     * rather than at each entry because the set is a property of the renderer's lifecycle, not of the
     * control: Sodium shows these with its own restart badge and the vanilla screen appends
     * {@code caustica.options.rt.restartNeeded} to the value, since it has no badge.
     */
    private static final Set<String> RESTART_PATHS = Set.of(
            "worker-threads",
            "entities.max-ordinary-entities",
            "entities.block-entities.max-entities",
            "entities.textures.max-textures",
            "entities.textures.pbr",
            "particles.max-particles",
            // NGX and FidelityFX build their frame-generation object against the live swapchain and the
            // in-flight submits; releasing it because the player picked the other generator loses the
            // Vulkan device.
            "frame-generation.backend");

    private static final String RT = "caustica.options.rt.";
    private static final String SETTING = "caustica.setting.";

    // ---- Parent state the gates read ---------------------------------------------------------------
    // Each one is a live read, so a page that stays open follows the switch it depends on instead of
    // freezing the state it was built with.

    // Emitter light sampling is the only consumer of the shadow-ray depth limit: with no emitters
    // sampled there is no sun or emitter ray to bound.
    private static final Supplier<Boolean> RIS_ON =
            () -> CausticaConfig.Rt.Lights.RIS_CANDIDATES.value() > 0;
    // The Purkinje shift adapts to the metered absolute scene luminance, which the exposure controller
    // only refreshes in auto mode; in manual mode it has no signal to work from.
    private static final Supplier<Boolean> AUTO_EXPOSURE =
            () -> "auto".equalsIgnoreCase(CausticaConfig.Rt.Exposure.MODE.get());
    private static final Supplier<Boolean> MANUAL_EXPOSURE = () -> !AUTO_EXPOSURE.get();
    private static final Supplier<Boolean> HDR_CAPABLE = CausticaConfig.Rt.Hdr::swapchainPqAvailable;
    // The three lens artifacts share one chain and one set of textures, so the shape and gain
    // entries only do anything while the chain itself runs.
    private static final Supplier<Boolean> LENS_ENABLED = () -> CausticaConfig.Rt.Lens.ENABLED.value();
    private static final Supplier<Boolean> LENS_STARBURST = () -> LENS_ENABLED.get()
            && CausticaConfig.Rt.Lens.STARBURST_ENABLED.value();
    private static final Supplier<Boolean> LENS_HALO = () -> LENS_ENABLED.get()
            && CausticaConfig.Rt.Lens.HALO_ENABLED.value();
    private static final Supplier<Boolean> LENS_DIRT = () -> LENS_ENABLED.get()
            && CausticaConfig.Rt.Lens.DIRT_ENABLED.value();
    // The lens's radius is what the far field blurs by, so the blur budget is only read while the lens
    // exists at all; the focus dial is only read while nothing is probing the view axis for it.
    private static final Supplier<Boolean> DOF_ENABLED =
            () -> CausticaConfig.Rt.DepthOfField.MODE.value() != 0;
    private static final Supplier<Boolean> DOF_MANUAL_FOCUS =
            () -> DOF_ENABLED.get() && !CausticaConfig.Rt.DepthOfField.AUTO_FOCUS.value();
    // Reflection focus is a rule the meter applies, so it only means anything while a meter is running.
    private static final Supplier<Boolean> DOF_AUTO_FOCUS =
            () -> DOF_ENABLED.get() && CausticaConfig.Rt.DepthOfField.AUTO_FOCUS.value();
    // The cap's dial only does anything while the cap itself is on, and the cap only binds while a lens is.
    private static final Supplier<Boolean> DOF_TRACED =
            () -> CausticaConfig.Rt.DepthOfField.MODE.value() == 1;
    private static final Supplier<Boolean> DOF_BLUR_CAP =
            () -> DOF_TRACED.get() && CausticaConfig.Rt.DepthOfField.BLUR_CAP.value();
    // The post-process gather carries its own aperture shape, and the rotation only shows on a polygon.
    private static final Supplier<Boolean> DOF_POST =
            () -> CausticaConfig.Rt.DepthOfField.MODE.value() == 2;
    private static final Supplier<Boolean> DOF_POST_BLADED =
            () -> DOF_POST.get() && CausticaConfig.Rt.DepthOfField.BLADES.value() >= 3;
    // The widest blur is bounded two ways: the traced lens by its cap switch, the post gather by its
    // kernel always. Either way the dial names the bound.
    private static final Supplier<Boolean> DOF_MAX_BLUR =
            () -> DOF_BLUR_CAP.get() || DOF_POST.get();

    // Volumetrics: the fog's own tree, and the two media that only exist alongside it.
    private static final Supplier<Boolean> FOG_ENABLED = () -> CausticaConfig.Rt.Volume.FOG_ENABLED.value();
    private static final Supplier<Boolean> FOG_SHADOW = () -> FOG_ENABLED.get()
            && CausticaConfig.Rt.Volume.FOG_SHADOW.value();
    private static final Supplier<Boolean> FOG_MARCH = () -> FOG_ENABLED.get()
            && CausticaConfig.Rt.Volume.FOG_MARCH.value();
    private static final Supplier<Boolean> SHAFTS = () -> CausticaConfig.Rt.Volume.SHAFT_ENABLED.value();
    private static final Supplier<Boolean> BOUNDARY_FOG =
            () -> CausticaConfig.Rt.Volume.BOUNDARY_FOG_ENABLED.value();
    private static final Supplier<Boolean> EDGE_FADE =
            () -> CausticaConfig.Rt.Volume.EDGE_FADE_ENABLED.value();
    private static final Supplier<Boolean> CLOUD_ENABLED = () -> CausticaConfig.Rt.Volume.CLOUD_ENABLED.value();
    private static final Supplier<Boolean> CLOUD_VOLUMETRIC = () -> CLOUD_ENABLED.get()
            && CausticaConfig.Rt.Volume.CLOUD_VOLUMETRIC.value();
    private static final Supplier<Boolean> CLOUD_SHADOW_FIELD = () -> CLOUD_ENABLED.get()
            && CausticaConfig.Rt.Volume.CLOUD_SHADOW.value();
    // Rain streaks: the weather's own switch, whose dials only mean anything while it is up.
    private static final Supplier<Boolean> RAIN_STREAKS = () -> CausticaConfig.Rt.Sky.RAIN_STREAKS.value();

    // Distant Horizons LOD field.
    private static final Supplier<Boolean> DH_LOD = () -> CausticaConfig.Rt.Dh.LOD_ENABLED.value();

    // Water: the surface's own dials, and the body's medium.
    private static final Supplier<Boolean> WATER_WAVES = () -> CausticaConfig.Rt.Composite.WATER_WAVES.value();
    private static final Supplier<Boolean> WATER_FOG = () -> CausticaConfig.Rt.Composite.WATER_FOG.value();

    // Materials, entities and the display chain.
    private static final Supplier<Boolean> PARALLAX_ON =
            () -> CausticaConfig.Rt.Composite.PARALLAX_DEPTH.value() > 0.0f;
    private static final Supplier<Boolean> ENTITIES_ON = () -> CausticaConfig.Rt.Entities.ENABLED.value();
    private static final Supplier<Boolean> PARTICLES_ON = () -> CausticaConfig.Rt.Entities.PARTICLES_ENABLED.value();
    private static final Supplier<Boolean> DLSS_RR = () -> CausticaConfig.Rt.DlssRr.ENABLED.value();
    private static final Supplier<Boolean> FRAME_GENERATION = () -> CausticaConfig.Rt.Fg.ENABLED.value();
    private static final Supplier<Boolean> REFLEX = () -> CausticaConfig.Rt.Reflex.ENABLED.value();
    private static final Supplier<Boolean> SHARPEN = () -> CausticaConfig.Rt.Sharpen.ENABLED.value();
    private static final Supplier<Boolean> STILL_RESOLUTION = () -> CausticaConfig.Rt.Still.RESOLUTION_SWITCH.value();
    private static final Supplier<Boolean> STILL_DENOISE = () -> CausticaConfig.Rt.Still.DENOISE.value();

    // Tone mapping: the two grading paths are exclusive branches of one output transform.
    private static final Supplier<Boolean> ADVANCED_GRADING = () -> CausticaConfig.Rt.Tonemap.ADVANCED_GRADING.value();
    private static final Supplier<Boolean> LOOK_TRANSFORM = () -> !ADVANCED_GRADING.get();
    private static final Supplier<Boolean> PURKINJE = () -> AUTO_EXPOSURE.get()
            && CausticaConfig.Rt.Tonemap.PURKINJE.value();

    // Diagnostics: the probes and dumps that only exist alongside their switch.
    private static final Supplier<Boolean> TEMPORAL_REUSE = () -> CausticaConfig.Rt.Lights.TEMPORAL_REUSE.value();
    private static final Supplier<Boolean> LIGHT_DUMP = () -> CausticaConfig.Rt.Lights.DUMP.value();

    public static final List<Option> ALL = build();

    /** The entries in display order; {@link #groups} preserves it within each page. */
    private static List<Option> build() {
        List<Option> out = new ArrayList<>(192);

        // ---- Ray tracing ---------------------------------------------------------------------------
        toggle(out, Page.RAY_TRACING, "core", "enabled", SETTING + "enabled", CausticaConfig.Rt.ENABLED);
        count(out, Page.RAY_TRACING, "core", "composite.spp", RT + "spp", CausticaConfig.Rt.Composite.SPP,
                1, 16, 1, Impact.HIGH);
        count(out, Page.RAY_TRACING, "core", "composite.max-bounces", RT + "maxBounces",
                CausticaConfig.Rt.Composite.MAX_BOUNCES, 2, 8, 1, Impact.HIGH);
        toggle(out, Page.RAY_TRACING, "core", "composite.russian-roulette", RT + "rrEnabled",
                CausticaConfig.Rt.Composite.RR_ENABLED);
        count(out, Page.RAY_TRACING, "core", "worker-threads", SETTING + "worker-threads",
                CausticaConfig.Rt.WORKER_THREADS, 1, 16, 1, Impact.MEDIUM);

        // The denoiser is part of the ray tracer, not of the display chain: it consumes the guides the
        // trace writes and replaces the accumulation the path budget would otherwise pay for.
        toggle(out, Page.RAY_TRACING, "denoise", "dlss-rr.enabled", SETTING + "dlssRr",
                CausticaConfig.Rt.DlssRr.ENABLED);
        toggle(out, Page.RAY_TRACING, "denoise", "dlss-rr.firefly-clamp", SETTING + "dlssRrFireflyClamp",
                CausticaConfig.Rt.DlssRr.FIREFLY_CLAMP, Gate.under("dlss-rr.enabled", DLSS_RR));
        cycled(out, Page.RAY_TRACING, "denoise", "dlss-rr.quality", RT + "dlssQuality", Cycle.DLSS_QUALITY,
                CausticaConfig.Rt.DlssRr.QUALITY, Impact.HIGH, Gate.under("dlss-rr.enabled", DLSS_RR));
        count(out, Page.RAY_TRACING, "denoise", "dlss-rr.preset", SETTING + "dlssPreset",
                CausticaConfig.Rt.DlssRr.PRESET, 0, 8, 1, Impact.LOW,
                Gate.under("dlss-rr.enabled", DLSS_RR));

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

        // Distant Horizons LOD field: the far terrain the ray tracer draws from DH's columns. Inert
        // without the DH mod, so the whole group hides behind its master switch.
        toggle(out, Page.RAY_TRACING, "terrain", "dh.lod", SETTING + "dhLod",
                CausticaConfig.Rt.Dh.LOD_ENABLED);
        count(out, Page.RAY_TRACING, "terrain", "dh.lod-max-distance", SETTING + "dhLodMaxDistance",
                CausticaConfig.Rt.Dh.MAX_DISTANCE, 0, 32768, 256, Impact.MEDIUM,
                Gate.under("dh.lod", DH_LOD));
        toggle(out, Page.RAY_TRACING, "terrain", "dh.lod-emissive", SETTING + "dhLodEmissive",
                CausticaConfig.Rt.Dh.EMISSIVE, Gate.under("dh.lod", DH_LOD));
        count(out, Page.RAY_TRACING, "terrain", "dh.lod-fetch-budget", SETTING + "dhLodFetchBudget",
                CausticaConfig.Rt.Dh.FETCH_BUDGET_MICROS, 0, 20000, 250, Impact.LOW,
                Gate.under("dh.lod", DH_LOD));

        count(out, Page.RAY_TRACING, "lights", "lights.ris-candidates", RT + "risCandidates",
                CausticaConfig.Rt.Lights.RIS_CANDIDATES, 0, 16, 1, Impact.MEDIUM);
        count(out, Page.RAY_TRACING, "lights", "lights.shadow-skip-depth", SETTING + "shadowSkipDepth",
                CausticaConfig.Rt.Lights.SHADOW_SKIP_DEPTH, 0, 16, 1, Impact.MEDIUM,
                Gate.under("lights.ris-candidates", RIS_ON));
        count(out, Page.RAY_TRACING, "lights", "lights.indirect-subsample", SETTING + "indirectSubsample",
                CausticaConfig.Rt.Lights.INDIRECT_SUBSAMPLE, 0, 3, 3, Impact.MEDIUM,
                SETTING + "indirectSubsample.");
        toggle(out, Page.RAY_TRACING, "lights", "lights.radiance-cache", SETTING + "radianceCache",
                CausticaConfig.Rt.Lights.RADIANCE_CACHE);
        scaled(out, Page.RAY_TRACING, "lights", "lights.min-fill-ratio", SETTING + "minFillRatio",
                CausticaConfig.Rt.Lights.MIN_FILL_RATIO, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW);
        toggle(out, Page.RAY_TRACING, "lights", "lights.temporal-reuse", SETTING + "lightTemporalReuse",
                CausticaConfig.Rt.Lights.TEMPORAL_REUSE);
        count(out, Page.RAY_TRACING, "lights", "lights.temporal-history", SETTING + "lightTemporalHistory",
                CausticaConfig.Rt.Lights.TEMPORAL_HISTORY, 8, 256, 8, Impact.LOW,
                Gate.under("lights.temporal-reuse", TEMPORAL_REUSE));

        // ---- Volumetrics: the air around the terrain ------------------------------------------------
        // The tier preset heads the page: it is the one control here that writes the others, so a player
        // who touches nothing else can read the whole system's cost ladder off it. Its label is a
        // readback rather than a memory — see CausticaConfig.Rt.Volume.currentTier.
        countWithReadout(out, Page.VOLUME, "preset", "volume.quality", RT + "volumeQuality",
                CausticaConfig.Rt.Volume.QUALITY, 0, 4, 1, Impact.HIGH, Gate.OPEN,
                () -> qualityLabel(CausticaConfig.Rt.Volume.currentTier()));

        toggle(out, Page.VOLUME, "fog", "volume.fog", RT + "fog", CausticaConfig.Rt.Volume.FOG_ENABLED);
        scaled(out, Page.VOLUME, "fog", "volume.fog-density", RT + "fogDensity",
                CausticaConfig.Rt.Volume.FOG_DENSITY, 0, 2000, 10, 100000.0, Unit.PER_BLOCK,
                Impact.MEDIUM, Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog", "volume.fog-day-decay", RT + "fogDayDecay",
                CausticaConfig.Rt.Volume.FOG_DAY_DECAY, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog", "volume.fog-height", RT + "fogHeight",
                CausticaConfig.Rt.Volume.FOG_HEIGHT, 2, 128, 1, 1.0, Unit.COUNT, Impact.MEDIUM,
                Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog", "volume.fog-base-y", RT + "fogBaseY",
                CausticaConfig.Rt.Volume.FOG_BASE_Y, -64, 320, 1, 1.0, Unit.COUNT, Impact.MEDIUM,
                Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog", "volume.fog-anisotropy", RT + "fogAnisotropy",
                CausticaConfig.Rt.Volume.FOG_ANISOTROPY, -90, 90, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog", "volume.fog-albedo", RT + "fogAlbedo",
                CausticaConfig.Rt.Volume.FOG_ALBEDO, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog", "volume.fog-ambient", RT + "fogAmbient",
                CausticaConfig.Rt.Volume.FOG_AMBIENT, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog", FOG_ENABLED));

        toggle(out, Page.VOLUME, "fog-detail", "volume.fog-shadow", RT + "fogShadow",
                CausticaConfig.Rt.Volume.FOG_SHADOW, Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog-detail", "volume.fog-shadow-floor", RT + "fogShadowFloor",
                CausticaConfig.Rt.Volume.FOG_SHADOW_FLOOR, 0, 95, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog-shadow", FOG_SHADOW));
        toggle(out, Page.VOLUME, "fog-detail", "volume.fog-march", RT + "fogMarch",
                CausticaConfig.Rt.Volume.FOG_MARCH, Gate.under("volume.fog", FOG_ENABLED));
        scaled(out, Page.VOLUME, "fog-detail", "volume.fog-noise", RT + "fogNoise",
                CausticaConfig.Rt.Volume.FOG_NOISE, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("volume.fog-march", FOG_MARCH));
        count(out, Page.VOLUME, "fog-detail", "volume.fog-march-steps", RT + "fogMarchSteps",
                CausticaConfig.Rt.Volume.FOG_MARCH_STEPS, 4, 64, 4, Impact.MEDIUM,
                Gate.under("volume.fog-march", FOG_MARCH));
        scaled(out, Page.VOLUME, "fog-detail", "volume.fog-jitter", RT + "fogJitter",
                CausticaConfig.Rt.Volume.FOG_JITTER, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog-march", FOG_MARCH));
        scaled(out, Page.VOLUME, "fog-detail", "volume.fog-wind-speed", RT + "fogWindSpeed",
                CausticaConfig.Rt.Volume.FOG_WIND_SPEED, 0, 32, 1, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.fog-march", FOG_MARCH));
        scaled(out, Page.VOLUME, "fog-detail", "volume.fog-wind-angle", RT + "fogWindAngle",
                CausticaConfig.Rt.Volume.FOG_WIND_ANGLE, 0, 360, 15, 1.0, Unit.DEGREES, Impact.LOW,
                Gate.under("volume.fog-march", FOG_MARCH));
        scaled(out, Page.VOLUME, "fog-detail", "volume.fog-rain-boost", RT + "fogRainBoost",
                CausticaConfig.Rt.Volume.FOG_RAIN_BOOST, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.fog", FOG_ENABLED));

        // The world border's haze. Its own three controls because it is its own ramp: it joins the fog's
        // medium, so it borrows that medium's albedo, phase and lighting dials above, but it stays out of
        // the height profile, the day curve and the weather lift entirely. It sits with the render edge's
        // fade rather than with the fog, because both of them are what makes the far field end.
        toggle(out, Page.VOLUME, "boundary", "volume.boundary-fog", RT + "boundaryFog",
                CausticaConfig.Rt.Volume.BOUNDARY_FOG_ENABLED);
        scaled(out, Page.VOLUME, "boundary", "volume.boundary-fog-density", RT + "boundaryFogDensity",
                CausticaConfig.Rt.Volume.BOUNDARY_FOG_DENSITY, 0, 250, 5, 1000.0, Unit.PERCENT,
                Impact.MEDIUM, Gate.under("volume.boundary-fog", BOUNDARY_FOG));
        scaled(out, Page.VOLUME, "boundary", "volume.boundary-fog-ramp", RT + "boundaryFogRamp",
                CausticaConfig.Rt.Volume.BOUNDARY_FOG_RAMP, 8, 1024, 8, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.boundary-fog", BOUNDARY_FOG));

        // Clear-air sun shafts: the god-ray layer that does not need the fog. Its own group because it is
        // its own medium — every control here works with the fog off, and the fog's own controls have no
        // effect on it.
        toggle(out, Page.VOLUME, "shafts", "volume.shafts", RT + "shafts",
                CausticaConfig.Rt.Volume.SHAFT_ENABLED);
        scaled(out, Page.VOLUME, "shafts", "volume.shaft-intensity", RT + "shaftIntensity",
                CausticaConfig.Rt.Volume.SHAFT_INTENSITY, 0, 400, 5, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("volume.shafts", SHAFTS));
        scaled(out, Page.VOLUME, "shafts", "volume.shaft-density", RT + "shaftDensity",
                CausticaConfig.Rt.Volume.SHAFT_DENSITY, 0, 2000, 10, 100000.0, Unit.PER_BLOCK, Impact.LOW,
                Gate.under("volume.shafts", SHAFTS));
        scaled(out, Page.VOLUME, "shafts", "volume.shaft-anisotropy", RT + "shaftAnisotropy",
                CausticaConfig.Rt.Volume.SHAFT_ANISOTROPY, 0, 95, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.shafts", SHAFTS));
        scaled(out, Page.VOLUME, "shafts", "volume.shaft-distance", RT + "shaftDistance",
                CausticaConfig.Rt.Volume.SHAFT_DISTANCE, 16, 512, 8, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.shafts", SHAFTS));
        scaled(out, Page.VOLUME, "shafts", "volume.water-shaft-density", RT + "waterShaftDensity",
                CausticaConfig.Rt.Volume.WATER_SHAFT_DENSITY, 0, 500, 5, 1000.0, Unit.PER_BLOCK,
                Impact.LOW, Gate.under("volume.shafts", SHAFTS));
        scaled(out, Page.VOLUME, "shafts", "volume.water-shaft-ceiling", RT + "waterShaftCeiling",
                CausticaConfig.Rt.Volume.WATER_SHAFT_CEILING, 0, 60, 1, 100.0, Unit.PERCENT,
                Impact.LOW, Gate.under("volume.shafts", SHAFTS));

        // ---- Sky & clouds: what is above the terrain -------------------------------------------------
        toggle(out, Page.SKY, "clouds", "volume.cloud", RT + "cloud", CausticaConfig.Rt.Volume.CLOUD_ENABLED);
        toggle(out, Page.SKY, "clouds", "volume.cloud-shadow", RT + "cloudShadow",
                CausticaConfig.Rt.Volume.CLOUD_SHADOW, Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-coverage", RT + "cloudCoverage",
                CausticaConfig.Rt.Volume.CLOUD_COVERAGE, 0, 200, 5, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-patchiness", RT + "cloudPatchiness",
                CausticaConfig.Rt.Volume.CLOUD_PATCHINESS, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-altitude", RT + "cloudAltitude",
                CausticaConfig.Rt.Volume.CLOUD_ALTITUDE, 0, 512, 8, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-thickness", RT + "cloudThickness",
                CausticaConfig.Rt.Volume.CLOUD_THICKNESS, 8, 1024, 8, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-size", RT + "cloudSize",
                CausticaConfig.Rt.Volume.CLOUD_SIZE, 32, 2048, 16, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-density", RT + "cloudDensity",
                CausticaConfig.Rt.Volume.CLOUD_DENSITY, 0, 300, 10, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "clouds", "volume.cloud-fade-distance", RT + "cloudFadeDistance",
                CausticaConfig.Rt.Volume.CLOUD_FADE_DISTANCE, 512, 16384, 256, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));

        toggle(out, Page.SKY, "cloud-detail", "volume.cloud-volumetric", RT + "cloudVolumetric",
                CausticaConfig.Rt.Volume.CLOUD_VOLUMETRIC, Gate.under("volume.cloud", CLOUD_ENABLED));
        count(out, Page.SKY, "cloud-detail", "volume.cloud-steps", RT + "cloudSteps",
                CausticaConfig.Rt.Volume.CLOUD_STEPS, 8, 48, 4, Impact.HIGH,
                Gate.under("volume.cloud-volumetric", CLOUD_VOLUMETRIC));
        count(out, Page.SKY, "cloud-detail", "volume.cloud-light-steps", RT + "cloudLightSteps",
                CausticaConfig.Rt.Volume.CLOUD_LIGHT_STEPS, 1, 8, 1, Impact.HIGH,
                Gate.under("volume.cloud-volumetric", CLOUD_VOLUMETRIC));
        toggle(out, Page.SKY, "cloud-detail", "volume.cloud-detail", RT + "cloudDetail",
                CausticaConfig.Rt.Volume.CLOUD_DETAIL, Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "cloud-detail", "volume.cloud-flow", RT + "cloudFlow",
                CausticaConfig.Rt.Volume.CLOUD_FLOW, 0, 200, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        toggle(out, Page.SKY, "cloud-detail", "volume.cloud-out-scatter", RT + "cloudOutScatter",
                CausticaConfig.Rt.Volume.CLOUD_OUT_SCATTER, Gate.under("volume.cloud", CLOUD_ENABLED));
        count(out, Page.SKY, "cloud-detail", "volume.cloud-shadow-steps", RT + "cloudShadowSteps",
                CausticaConfig.Rt.Volume.CLOUD_SHADOW_STEPS, 1, 4, 1, Impact.HIGH,
                Gate.under("volume.cloud-shadow", CLOUD_SHADOW_FIELD));
        scaled(out, Page.SKY, "cloud-detail", "volume.cloud-wind-speed", RT + "cloudWindSpeed",
                CausticaConfig.Rt.Volume.CLOUD_WIND_SPEED, 0, 64, 1, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "cloud-detail", "volume.cloud-wind-angle", RT + "cloudWindAngle",
                CausticaConfig.Rt.Volume.CLOUD_WIND_ANGLE, 0, 360, 15, 1.0, Unit.DEGREES, Impact.LOW,
                Gate.under("volume.cloud", CLOUD_ENABLED));
        scaled(out, Page.SKY, "cloud-detail", "volume.cloud-rain-boost", RT + "cloudRainBoost",
                CausticaConfig.Rt.Volume.CLOUD_RAIN_BOOST, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("volume.cloud", CLOUD_ENABLED));

        // The render edge's fade. It needs no density dial because there is no density: the coverage is a
        // blend weight toward the sky, so the opacity at the edge is the whole shape of the effect and the
        // render distance only decides where that edge sits.
        toggle(out, Page.SKY, "horizon", "volume.edge-fade", RT + "edgeFade",
                CausticaConfig.Rt.Volume.EDGE_FADE_ENABLED);
        scaled(out, Page.SKY, "horizon", "volume.edge-fade-opacity", RT + "edgeFadeOpacity",
                CausticaConfig.Rt.Volume.EDGE_FADE_OPACITY, 0, 100, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM, Gate.under("volume.edge-fade", EDGE_FADE));
        scaled(out, Page.SKY, "horizon", "volume.edge-fade-start", RT + "edgeFadeStart",
                CausticaConfig.Rt.Volume.EDGE_FADE_START, 0, 95, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("volume.edge-fade", EDGE_FADE));

        toggle(out, Page.SKY, "sky", "sky.vanilla-sun", RT + "vanillaSun",
                CausticaConfig.Rt.Sky.VANILLA_SUN);
        toggle(out, Page.SKY, "sky", "sky.vanilla-moon", RT + "vanillaMoon",
                CausticaConfig.Rt.Sky.VANILLA_MOON);
        scaled(out, Page.SKY, "sky", "sky.rain-darkening", RT + "rainSkyDarkening",
                CausticaConfig.Rt.Sky.RAIN_SKY_DARKENING, 0, 100, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM);

        // The weather itself: what falls out of the sky this page has already darkened. Vanilla's own
        // rain is cancelled with the rest of its weather rendering, so this is the only thing that
        // draws it. Every dial is gated on the switch because a length or a brightness with no streaks
        // on screen has nothing to describe.
        toggle(out, Page.SKY, "weather", "sky.rain-streaks", RT + "rainStreaks",
                CausticaConfig.Rt.Sky.RAIN_STREAKS);
        count(out, Page.SKY, "weather", "sky.rain-streak-density", RT + "rainStreakDensity",
                CausticaConfig.Rt.Sky.RAIN_STREAK_DENSITY, 0, 12000, 200, Impact.MEDIUM,
                Gate.under("sky.rain-streaks", RAIN_STREAKS));
        scaled(out, Page.SKY, "weather", "sky.rain-streak-length", RT + "rainStreakLength",
                CausticaConfig.Rt.Sky.RAIN_STREAK_LENGTH, 20, 400, 10, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("sky.rain-streaks", RAIN_STREAKS));
        scaled(out, Page.SKY, "weather", "sky.rain-streak-width", RT + "rainStreakWidth",
                CausticaConfig.Rt.Sky.RAIN_STREAK_WIDTH, 4, 80, 1, 1000.0, Unit.CENTIMETRES, Impact.LOW,
                Gate.under("sky.rain-streaks", RAIN_STREAKS));
        scaled(out, Page.SKY, "weather", "sky.rain-streak-wind-speed", RT + "rainStreakWindSpeed",
                CausticaConfig.Rt.Sky.RAIN_STREAK_WIND_SPEED, 0, 24, 1, 1.0, Unit.COUNT, Impact.LOW,
                Gate.under("sky.rain-streaks", RAIN_STREAKS));
        scaled(out, Page.SKY, "weather", "sky.rain-streak-wind-angle", RT + "rainStreakWindAngle",
                CausticaConfig.Rt.Sky.RAIN_STREAK_WIND_ANGLE, 0, 360, 15, 1.0, Unit.DEGREES, Impact.LOW,
                Gate.under("sky.rain-streaks", RAIN_STREAKS));
        scaled(out, Page.SKY, "weather", "sky.rain-streak-brightness", RT + "rainStreakBrightness",
                CausticaConfig.Rt.Sky.RAIN_STREAK_BRIGHTNESS, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("sky.rain-streaks", RAIN_STREAKS));

        // ---- Materials: what a surface is made of ---------------------------------------------------
        // Everything here describes the surface itself rather than the world around it, which is why the
        // parallax relief and the surface wetness sit here instead of inside the terrain streaming group or
        // the water group. 0 disables the march entirely, which is also the quickest way to tell a parallax
        // artifact apart from everything else on a surface.
        scaled(out, Page.MATERIALS, "parallax", "composite.parallax-depth", RT + "parallaxDepth",
                CausticaConfig.Rt.Composite.PARALLAX_DEPTH, 0, 50, 1, 100.0, Unit.PERCENT, Impact.MEDIUM);
        cycled(out, Page.MATERIALS, "parallax", "materials.parallax-style", RT + "parallaxStyle",
                Cycle.PARALLAX_STYLE, CausticaConfig.Rt.Materials.PARALLAX_STYLE, Impact.LOW,
                Gate.under("composite.parallax-depth", PARALLAX_ON));
        toggle(out, Page.MATERIALS, "parallax", "materials.parallax-shadow", RT + "parallaxShadow",
                CausticaConfig.Rt.Materials.PARALLAX_SHADOW, Gate.under("composite.parallax-depth", PARALLAX_ON));
        count(out, Page.MATERIALS, "parallax", "materials.parallax-layers", RT + "parallaxLayers",
                CausticaConfig.Rt.Materials.PARALLAX_LAYERS, 8, 64, 4, Impact.MEDIUM,
                Gate.under("composite.parallax-depth", PARALLAX_ON));
        toggle(out, Page.MATERIALS, "parallax", "materials.entity-parallax", RT + "entityParallax",
                CausticaConfig.Rt.Materials.ENTITY_PARALLAX, Gate.under("composite.parallax-depth", PARALLAX_ON));

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
        scaled(out, Page.MATERIALS, "labpbr", "composite.rain-wetness", RT + "rainWetness",
                CausticaConfig.Rt.Composite.RAIN_WETNESS, 0, 100, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        scaled(out, Page.MATERIALS, "labpbr", "composite.puddle-coverage", RT + "puddleCoverage",
                CausticaConfig.Rt.Composite.PUDDLE_COVERAGE, 0, 100, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM);

        // ---- Entities, particles and overlays ------------------------------------------------------
        toggle(out, Page.ENTITIES, "entities", "entities.enabled", RT + "entities",
                CausticaConfig.Rt.Entities.ENABLED);
        toggle(out, Page.ENTITIES, "entities", "entities.trace-self.enabled", RT + "traceSelf",
                CausticaConfig.Rt.Entities.SELF_ENABLED, Gate.under("entities.enabled", ENTITIES_ON));
        toggle(out, Page.ENTITIES, "entities", "entities.glow.enabled", RT + "glow",
                CausticaConfig.Rt.Entities.GLOW_ENABLED, Gate.under("entities.enabled", ENTITIES_ON));
        toggle(out, Page.ENTITIES, "entities", "entities.name-tags.enabled", SETTING + "nameTags",
                CausticaConfig.Rt.Entities.NAME_TAGS_ENABLED, Gate.under("entities.enabled", ENTITIES_ON));
        toggle(out, Page.ENTITIES, "entities", "entities.refit.enabled", SETTING + "entityRefit",
                CausticaConfig.Rt.Entities.REFIT_ENABLED, Gate.under("entities.enabled", ENTITIES_ON));
        count(out, Page.ENTITIES, "entities", "entities.block-entities.view-chunks",
                SETTING + "beViewChunks", CausticaConfig.Rt.Entities.BE_VIEW_CHUNKS, 0, 64, 1,
                Impact.MEDIUM, Gate.under("entities.enabled", ENTITIES_ON));
        count(out, Page.ENTITIES, "entities", "entities.block-entities.builds-per-frame",
                SETTING + "beBuilds", CausticaConfig.Rt.Entities.BE_BUILDS_PER_FRAME, 0, 512, 8,
                Impact.MEDIUM, Gate.under("entities.enabled", ENTITIES_ON));
        // The four restart-bound caps close the group: a player who changes one is changing a buffer
        // budget, and having them adjacent is what makes that legible.
        count(out, Page.ENTITIES, "entities", "entities.max-ordinary-entities", SETTING + "maxOrdinary",
                CausticaConfig.Rt.Entities.MAX_ORDINARY_ENTITIES, 0, 4096, 16, Impact.MEDIUM,
                Gate.under("entities.enabled", ENTITIES_ON));
        count(out, Page.ENTITIES, "entities", "entities.block-entities.max-entities",
                SETTING + "maxBlockEntities", CausticaConfig.Rt.Entities.MAX_BLOCK_ENTITIES,
                0, 4096, 16, Impact.MEDIUM, Gate.under("entities.enabled", ENTITIES_ON));
        count(out, Page.ENTITIES, "entities", "entities.textures.max-textures",
                SETTING + "maxEntityTextures", CausticaConfig.Rt.EntityTextures.MAX_TEXTURES, 1, 4096, 1,
                Impact.LOW, Gate.under("entities.enabled", ENTITIES_ON));
        toggle(out, Page.ENTITIES, "entities", "entities.textures.pbr", SETTING + "entityPbr",
                CausticaConfig.Rt.EntityTextures.PBR, Gate.under("entities.enabled", ENTITIES_ON));

        toggle(out, Page.ENTITIES, "particles", "particles.enabled", RT + "particles",
                CausticaConfig.Rt.Entities.PARTICLES_ENABLED);
        count(out, Page.ENTITIES, "particles", "particles.max-particles", SETTING + "maxParticles",
                CausticaConfig.Rt.Entities.MAX_PARTICLES, 0, 8192, 32, Impact.MEDIUM,
                Gate.under("particles.enabled", PARTICLES_ON));

        toggle(out, Page.ENTITIES, "overlay", "overlay.block-outline.enabled", SETTING + "blockOutline",
                CausticaConfig.Rt.Overlay.BLOCK_OUTLINE_ENABLED);

        // ---- Water: the surface, then the body's own medium -----------------------------------------
        toggle(out, Page.WATER, "water", "composite.water-waves", RT + "waterWaves",
                CausticaConfig.Rt.Composite.WATER_WAVES);
        scaled(out, Page.WATER, "water", "composite.water-wave-height", RT + "waterWaveHeight",
                CausticaConfig.Rt.Composite.WATER_WAVE_HEIGHT, 0, 300, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM, Gate.under("composite.water-waves", WATER_WAVES));
        scaled(out, Page.WATER, "water", "composite.water-wave-scale", RT + "waterWaveSize",
                CausticaConfig.Rt.Composite.WATER_WAVE_SCALE, 25, 400, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM, Gate.under("composite.water-waves", WATER_WAVES));
        scaled(out, Page.WATER, "water", "composite.water-wave-speed", RT + "waterWaveSpeed",
                CausticaConfig.Rt.Composite.WATER_WAVE_SPEED, 0, 300, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("composite.water-waves", WATER_WAVES));
        scaled(out, Page.WATER, "water", "composite.water-reflection", RT + "waterReflection",
                CausticaConfig.Rt.Composite.WATER_REFLECTION, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW);

        // The body's own medium: how much light it absorbs, and how much of that it re-scatters as fog.
        // Both are per-block properties of the water, so they sit together rather than with the surface.
        scaled(out, Page.WATER, "water-body", "composite.water-opacity", RT + "waterOpacity",
                CausticaConfig.Rt.Composite.WATER_OPACITY, 0, 400, 5, 100.0, Unit.PERCENT, Impact.MEDIUM);
        toggle(out, Page.WATER, "water-body", "composite.water-fog", RT + "waterFog",
                CausticaConfig.Rt.Composite.WATER_FOG);
        scaled(out, Page.WATER, "water-body", "composite.water-fog-density", RT + "waterFogDensity",
                CausticaConfig.Rt.Composite.WATER_FOG_DENSITY, 0, 100, 5, 100.0, Unit.PERCENT,
                Impact.MEDIUM, Gate.under("composite.water-fog", WATER_FOG));
        scaled(out, Page.WATER, "water-body", "composite.water-fog-anisotropy", RT + "waterFogAnisotropy",
                CausticaConfig.Rt.Composite.WATER_FOG_ANISOTROPY, 0, 95, 5, 100.0, Unit.PERCENT,
                Impact.LOW, Gate.under("composite.water-fog", WATER_FOG));
        scaled(out, Page.WATER, "water-body", "composite.water-fog-ambient", RT + "waterFogAmbient",
                CausticaConfig.Rt.Composite.WATER_FOG_AMBIENT, 0, 200, 5, 100.0, Unit.PERCENT,
                Impact.LOW, Gate.under("composite.water-fog", WATER_FOG));

        // ---- Exposure: how the frame decides what mid grey is ----------------------------------------
        cycled(out, Page.EXPOSURE, "exposure", "exposure.mode", RT + "exposureMode", Cycle.EXPOSURE_MODE,
                CausticaConfig.Rt.Exposure.MODE, Impact.MEDIUM);
        scaled(out, Page.EXPOSURE, "exposure", "exposure.manual-ev", RT + "manualEv",
                CausticaConfig.Rt.Exposure.MANUAL_EV, -1500, 1500, 5, 100.0, Unit.EV, Impact.LOW,
                Gate.under("exposure.mode", MANUAL_EXPOSURE));
        // The metering ladder is what auto mode runs on; the two switches above it pick the branch.
        scaled(out, Page.EXPOSURE, "exposure", "exposure.key", SETTING + "exposureKey",
                CausticaConfig.Rt.Exposure.KEY, 5, 500, 1, 100.0, Unit.MULTIPLE, Impact.MEDIUM,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.night-lift", RT + "nightLift",
                CausticaConfig.Rt.Exposure.NIGHT_LIFT, -600, 600, 5, 100.0, Unit.EV, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.adapt-darken", SETTING + "adaptDarken",
                CausticaConfig.Rt.Exposure.ADAPT_DARKEN, 0, 1000, 5, 100.0, Unit.SECONDS, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.adapt-brighten", SETTING + "adaptBrighten",
                CausticaConfig.Rt.Exposure.ADAPT_BRIGHTEN, 0, 1000, 5, 100.0, Unit.SECONDS, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.low-percentile", SETTING + "lowPercentile",
                CausticaConfig.Rt.Exposure.LOW_PERCENTILE, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.high-percentile", SETTING + "highPercentile",
                CausticaConfig.Rt.Exposure.HIGH_PERCENTILE, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        count(out, Page.EXPOSURE, "exposure", "exposure.stride", SETTING + "exposureStride",
                CausticaConfig.Rt.Exposure.STRIDE, 1, 8, 1, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.center-weight-sigma", SETTING + "centerSigma",
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_SIGMA, 1, 200, 1, 100.0, Unit.MULTIPLE,
                Impact.LOW, Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.center-weight-floor", SETTING + "centerFloor",
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_FLOOR, 0, 100, 1, 100.0, Unit.PERCENT,
                Impact.LOW, Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.sky-weight-cap", SETTING + "skyWeightCap",
                CausticaConfig.Rt.Exposure.SKY_WEIGHT_CAP, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.EXPOSURE, "exposure", "exposure.emissive-weight-cap", SETTING + "emissiveWeightCap",
                CausticaConfig.Rt.Exposure.EMISSIVE_WEIGHT_CAP, 0, 100, 1, 100.0, Unit.PERCENT,
                Impact.LOW, Gate.under("exposure.mode", AUTO_EXPOSURE));
        // Pre-exposure multiplies and divides by the same number, so it applies in both branches.
        toggle(out, Page.EXPOSURE, "exposure", "exposure.pre-exposure", SETTING + "preExposure",
                CausticaConfig.Rt.Exposure.PRE_EXPOSURE);

        // ---- Colour & tone --------------------------------------------------------------------------
        scaled(out, Page.COLOR, "tonemap", "tonemap.gamma", RT + "gamma", CausticaConfig.Rt.Tonemap.GAMMA,
                10, 500, 5, 100.0, Unit.MULTIPLE, Impact.LOW);
        toggle(out, Page.COLOR, "tonemap", "tonemap.purkinje", RT + "purkinje",
                CausticaConfig.Rt.Tonemap.PURKINJE, Gate.under("exposure.mode", AUTO_EXPOSURE));
        scaled(out, Page.COLOR, "tonemap", "tonemap.purkinje-strength", RT + "purkinjeStrength",
                CausticaConfig.Rt.Tonemap.PURKINJE_STRENGTH, 0, 200, 5, 100.0, Unit.MULTIPLE,
                Impact.LOW, Gate.under("tonemap.purkinje", PURKINJE));
        toggle(out, Page.COLOR, "tonemap", "tonemap.advanced-grading", RT + "advancedGrading",
                CausticaConfig.Rt.Tonemap.ADVANCED_GRADING);
        scaled(out, Page.COLOR, "tonemap", "tonemap.agx-gain", RT + "agxExposure",
                CausticaConfig.Rt.Tonemap.AGX_GAIN, 2, 800, 1, 100.0, Unit.MULTIPLE, Impact.LOW,
                Gate.under("tonemap.advanced-grading", ADVANCED_GRADING));
        scaled(out, Page.COLOR, "tonemap", "tonemap.agx-saturation", RT + "agxSaturation",
                CausticaConfig.Rt.Tonemap.AGX_SATURATION, 0, 300, 1, 100.0, Unit.MULTIPLE, Impact.LOW,
                Gate.under("tonemap.advanced-grading", ADVANCED_GRADING));
        scaled(out, Page.COLOR, "tonemap", "tonemap.agx-contrast", RT + "agxContrast",
                CausticaConfig.Rt.Tonemap.AGX_CONTRAST, 0, 200, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("tonemap.advanced-grading", ADVANCED_GRADING));
        scaled(out, Page.COLOR, "tonemap", "tonemap.agx-range", RT + "agxRange",
                CausticaConfig.Rt.Tonemap.AGX_RANGE, 10, 20, 1, 1.0, Unit.EV, Impact.LOW,
                Gate.under("tonemap.advanced-grading", ADVANCED_GRADING));
        // The look transform is the OTHER branch of the same choice: the display shader runs it only while
        // advanced grading is off. It sits last on the page for that reason.
        toggle(out, Page.COLOR, "tonemap", "tonemap.look-enabled", RT + "look",
                CausticaConfig.Rt.Tonemap.LOOK_ENABLED, Gate.under("tonemap.advanced-grading", LOOK_TRANSFORM));
        scaled(out, Page.COLOR, "tonemap", "tonemap.bloom-strength", RT + "bloomStrength",
                CausticaConfig.Rt.Tonemap.BLOOM_STRENGTH, 0, 300, 5, 100.0, Unit.PERCENT, Impact.LOW);

        // ---- Lens flare -----------------------------------------------------------------------------
        // One chain, one set of textures: the three artifacts are switches inside it, and each owns the
        // dials that shape it. The order is the chain's own — resolution and sampling first, then the
        // starburst, the halo and the dust.
        toggle(out, Page.LENS, "lens", "lens.enabled", RT + "lens", CausticaConfig.Rt.Lens.ENABLED);
        count(out, Page.LENS, "lens", "lens.resolution", RT + "lensResolution",
                CausticaConfig.Rt.Lens.RESOLUTION, 4, 16, 2, Impact.MEDIUM,
                Gate.under("lens.enabled", LENS_ENABLED), "caustica.options.rt.lensResolution.");
        scaled(out, Page.LENS, "lens", "lens.sample-rate", RT + "lensSampleRate",
                CausticaConfig.Rt.Lens.SAMPLE_RATE, 25, 200, 5, 100.0, Unit.PERCENT, Impact.MEDIUM,
                Gate.under("lens.enabled", LENS_ENABLED));
        toggle(out, Page.LENS, "lens", "lens.starburst-enabled", RT + "lensStarburstEnabled",
                CausticaConfig.Rt.Lens.STARBURST_ENABLED, Gate.under("lens.enabled", LENS_ENABLED));
        scaled(out, Page.LENS, "lens", "lens.starburst", RT + "lensStarburst",
                CausticaConfig.Rt.Lens.STARBURST, 0, 300, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("lens.starburst-enabled", LENS_STARBURST));
        count(out, Page.LENS, "lens", "lens.spikes", RT + "lensSpikes",
                CausticaConfig.Rt.Lens.SPIKES, 3, 12, 1, Impact.LOW,
                Gate.under("lens.starburst-enabled", LENS_STARBURST));
        scaled(out, Page.LENS, "lens", "lens.spike-length", RT + "lensSpikeLength",
                CausticaConfig.Rt.Lens.SPIKE_LENGTH, 2, 80, 1, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("lens.starburst-enabled", LENS_STARBURST));
        count(out, Page.LENS, "lens", "lens.samples", RT + "lensSamples",
                CausticaConfig.Rt.Lens.SAMPLES, 8, 32, 4, Impact.MEDIUM,
                Gate.under("lens.starburst-enabled", LENS_STARBURST));
        toggle(out, Page.LENS, "lens", "lens.halo-enabled", RT + "lensHaloEnabled",
                CausticaConfig.Rt.Lens.HALO_ENABLED, Gate.under("lens.enabled", LENS_ENABLED));
        scaled(out, Page.LENS, "lens", "lens.halo", RT + "lensHalo", CausticaConfig.Rt.Lens.HALO,
                0, 300, 5, 100.0, Unit.PERCENT, Impact.LOW, Gate.under("lens.halo-enabled", LENS_HALO));
        scaled(out, Page.LENS, "lens", "lens.halo-radius", RT + "lensHaloRadius",
                CausticaConfig.Rt.Lens.HALO_RADIUS, 1, 60, 1, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("lens.halo-enabled", LENS_HALO));
        scaled(out, Page.LENS, "lens", "lens.dispersion", RT + "lensDispersion",
                CausticaConfig.Rt.Lens.DISPERSION, 0, 100, 1, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("lens.halo-enabled", LENS_HALO));
        count(out, Page.LENS, "lens", "lens.halo-taps", RT + "lensHaloTaps",
                CausticaConfig.Rt.Lens.HALO_TAPS, 8, 48, 8, Impact.MEDIUM,
                Gate.under("lens.halo-enabled", LENS_HALO));
        toggle(out, Page.LENS, "lens", "lens.dirt-enabled", RT + "lensDirtEnabled",
                CausticaConfig.Rt.Lens.DIRT_ENABLED, Gate.under("lens.enabled", LENS_ENABLED));
        scaled(out, Page.LENS, "lens", "lens.dirt", RT + "lensDirt", CausticaConfig.Rt.Lens.DIRT,
                0, 300, 5, 100.0, Unit.PERCENT, Impact.LOW, Gate.under("lens.dirt-enabled", LENS_DIRT));
        scaled(out, Page.LENS, "lens", "lens.dirt-scale", RT + "lensDirtScale",
                CausticaConfig.Rt.Lens.DIRT_SCALE, 25, 400, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("lens.dirt-enabled", LENS_DIRT));

        // ---- Depth of field: the traced camera's own lens, or a post-process stand-in ---------------
        // The traced mode sits beside the flare rather than with the post effects because it is not
        // one: the blur is sampled by the primary ray, so the light it softens has already reflected,
        // refracted and scattered on the way in, and no pass downstream of the trace could produce it.
        // The post mode is exactly such a pass: cheaper, denoiser-free, and an approximation of the
        // same defocus, which is why both exist behind one mode switch.
        cycled(out, Page.DOF, "depth-of-field", "dof.mode", RT + "dofMode", Cycle.DOF_MODE,
                CausticaConfig.Rt.DepthOfField.MODE, Impact.MEDIUM);
        toggle(out, Page.DOF, "depth-of-field", "dof.auto-focus", RT + "dofAutoFocus",
                CausticaConfig.Rt.DepthOfField.AUTO_FOCUS, Gate.under("dof.mode", DOF_ENABLED));
        toggle(out, Page.DOF, "depth-of-field", "dof.reflection-focus", RT + "dofReflectionFocus",
                CausticaConfig.Rt.DepthOfField.REFLECTION_FOCUS, Gate.under("dof.auto-focus", DOF_AUTO_FOCUS));
        // Tenths of a block: the widget carries value * 10, so the slider still steps once per click
        // while the focus plane can sit anywhere the renderer's 0.5..512 clamp allows a tenth of.
        scaled(out, Page.DOF, "depth-of-field", "dof.focus-distance", RT + "dofFocusDistance",
                CausticaConfig.Rt.DepthOfField.FOCUS_BLOCKS, 5, 2560, 1, 10.0, Unit.BLOCKS, Impact.LOW,
                Gate.under("dof.auto-focus", DOF_MANUAL_FOCUS));
        scaled(out, Page.DOF, "depth-of-field", "dof.rack-time", RT + "dofFocusTime",
                CausticaConfig.Rt.DepthOfField.RACK_SECONDS, 2, 100, 1, 100.0, Unit.SECONDS, Impact.LOW,
                Gate.under("dof.mode", DOF_ENABLED));
        scaled(out, Page.DOF, "depth-of-field", "dof.aperture", RT + "dofAperture",
                CausticaConfig.Rt.DepthOfField.APERTURE, 0, 10, 1, 100.0, Unit.CENTIMETRES,
                Impact.MEDIUM, Gate.under("dof.mode", DOF_ENABLED));
        toggle(out, Page.DOF, "depth-of-field", "dof.blur-cap", RT + "dofBlurCap",
                CausticaConfig.Rt.DepthOfField.BLUR_CAP, Gate.under("dof.mode", DOF_TRACED));
        scaled(out, Page.DOF, "depth-of-field", "dof.max-blur", RT + "dofMaxBlur",
                CausticaConfig.Rt.DepthOfField.MAX_BLUR_PIXELS, 2, 48, 1, 1.0, Unit.COUNT, Impact.MEDIUM,
                Gate.under("dof.mode", DOF_MAX_BLUR));
        toggle(out, Page.DOF, "depth-of-field", "dof.per-pixel-cap", RT + "dofPerPixelCap",
                CausticaConfig.Rt.DepthOfField.PER_PIXEL_CAP, Gate.under("dof.blur-cap", DOF_BLUR_CAP));
        toggle(out, Page.DOF, "depth-of-field", "dof.pinhole-guides", RT + "dofPinholeGuides",
                CausticaConfig.Rt.DepthOfField.PINHOLE_GUIDES, Gate.under("dof.mode", DOF_TRACED));
        count(out, Page.DOF, "depth-of-field", "dof.blades", RT + "dofBlades",
                CausticaConfig.Rt.DepthOfField.BLADES, 0, 12, 1, Impact.LOW,
                Gate.under("dof.mode", DOF_POST));
        scaled(out, Page.DOF, "depth-of-field", "dof.blade-rotation", RT + "dofBladeRotation",
                CausticaConfig.Rt.DepthOfField.BLADE_ROTATION, 0, 180, 5, 1.0, Unit.DEGREES, Impact.LOW,
                Gate.under("dof.blades", DOF_POST_BLADED));
        scaled(out, Page.DOF, "depth-of-field", "dof.cat-eye", RT + "dofCatEye",
                CausticaConfig.Rt.DepthOfField.CAT_EYE, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("dof.mode", DOF_ENABLED));
        scaled(out, Page.DOF, "depth-of-field", "dof.swirl", RT + "dofSwirl",
                CausticaConfig.Rt.DepthOfField.SWIRL, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("dof.mode", DOF_ENABLED));

        // ---- Output: what happens to the finished frame ---------------------------------------------
        toggle(out, Page.OUTPUT, "sharpen", "sharpen.enabled", RT + "sharpen",
                CausticaConfig.Rt.Sharpen.ENABLED);
        scaled(out, Page.OUTPUT, "sharpen", "sharpen.amount", RT + "sharpenAmount",
                CausticaConfig.Rt.Sharpen.AMOUNT, 0, 200, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("sharpen.enabled", SHARPEN));
        toggle(out, Page.OUTPUT, "frame-generation", "frame-generation.enabled",
                SETTING + "frameGeneration", CausticaConfig.Rt.Fg.ENABLED);
        count(out, Page.OUTPUT, "frame-generation", "frame-generation.multi-frame-count",
                SETTING + "multiFrameCount", CausticaConfig.Rt.Fg.MULTI_FRAME_COUNT, 1, 3, 1,
                Impact.MEDIUM, Gate.under("frame-generation.enabled", FRAME_GENERATION));
        cycled(out, Page.OUTPUT, "frame-generation", "frame-generation.backend",
                SETTING + "frameGenerationBackend", Cycle.FRAME_GENERATION_BACKEND,
                CausticaConfig.Rt.Fg.BACKEND, Impact.MEDIUM,
                Gate.under("frame-generation.enabled", FRAME_GENERATION));
        toggle(out, Page.OUTPUT, "reflex", "reflex.enabled", SETTING + "reflex",
                CausticaConfig.Rt.Reflex.ENABLED);
        toggle(out, Page.OUTPUT, "reflex", "reflex.low-latency-boost", SETTING + "reflexBoost",
                CausticaConfig.Rt.Reflex.LOW_LATENCY_BOOST, Gate.under("reflex.enabled", REFLEX));
        count(out, Page.OUTPUT, "reflex", "reflex.minimum-interval-us", SETTING + "reflexInterval",
                CausticaConfig.Rt.Reflex.MINIMUM_INTERVAL_US, 0, 100000, 500, Impact.LOW,
                Gate.under("reflex.enabled", REFLEX));
        toggle(out, Page.OUTPUT, "hdr", "hdr.enabled", RT + "hdr", CausticaConfig.Rt.Hdr.ENABLED,
                Gate.requires(HDR_CAPABLE));
        scaled(out, Page.OUTPUT, "hdr", "hdr.ui-nits", RT + "hdrUiBrightness",
                CausticaConfig.Rt.Hdr.UI_NITS, 80, 500, 5, 1.0, Unit.NITS, Impact.LOW,
                Gate.requires(HDR_CAPABLE));
        cycled(out, Page.OUTPUT, "hdr", "hdr.peak-nits", RT + "hdrPeak", Cycle.HDR_PEAK,
                CausticaConfig.Rt.Hdr.PEAK_NITS, Impact.LOW, Gate.requires(HDR_CAPABLE));

        // ---- Still render: the offline frame, unrelated to the live one ------------------------------
        count(out, Page.STILL, "still", "still.samples", SETTING + "stillSamples",
                CausticaConfig.Rt.Still.SAMPLES, 4, 1024, 4, Impact.LOW);
        toggle(out, Page.STILL, "still", "still.infinite", SETTING + "stillInfinite",
                CausticaConfig.Rt.Still.INFINITE);
        toggle(out, Page.STILL, "still", "still.resolution-switch", SETTING + "stillResolutionSwitch",
                CausticaConfig.Rt.Still.RESOLUTION_SWITCH);
        count(out, Page.STILL, "still", "still.width", SETTING + "stillWidth",
                CausticaConfig.Rt.Still.WIDTH, 0, 7680, 8, Impact.LOW,
                Gate.under("still.resolution-switch", STILL_RESOLUTION));
        count(out, Page.STILL, "still", "still.height", SETTING + "stillHeight",
                CausticaConfig.Rt.Still.HEIGHT, 0, 4320, 8, Impact.LOW,
                Gate.under("still.resolution-switch", STILL_RESOLUTION));
        toggle(out, Page.STILL, "still", "still.denoise", SETTING + "stillDenoise",
                CausticaConfig.Rt.Still.DENOISE);
        scaled(out, Page.STILL, "still", "still.denoise-strength", SETTING + "stillDenoiseStrength",
                CausticaConfig.Rt.Still.DENOISE_STRENGTH, 0, 100, 5, 100.0, Unit.PERCENT, Impact.LOW,
                Gate.under("still.denoise", STILL_DENOISE));
        toggle(out, Page.STILL, "still", "still.legacy-accumulation", SETTING + "stillLegacyAccumulation",
                CausticaConfig.Rt.Still.LEGACY_ACCUMULATION);

        // ---- Diagnostics: what only exists to look at a buffer or a log -------------------------------
        count(out, Page.DIAGNOSTICS, "debug", "composite.debug-view", RT + "debugView",
                CausticaConfig.Rt.Composite.DEBUG_VIEW, 0, 13, 1, Impact.LOW, RT + "debugView.");
        toggle(out, Page.DIAGNOSTICS, "debug", "lights.stats", SETTING + "lightStats",
                CausticaConfig.Rt.Lights.STATS);
        toggle(out, Page.DIAGNOSTICS, "debug", "lights.dump", SETTING + "lightDump",
                CausticaConfig.Rt.Lights.DUMP);
        count(out, Page.DIAGNOSTICS, "debug", "lights.dump-radius", SETTING + "lightDumpRadius",
                CausticaConfig.Rt.Lights.DUMP_RADIUS, 1, 64, 1, Impact.LOW,
                Gate.under("lights.dump", LIGHT_DUMP));
        toggle(out, Page.DIAGNOSTICS, "debug", "entities.debug.capture-parity", SETTING + "captureParity",
                CausticaConfig.Rt.Entities.CAPTURE_PARITY);

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

    /** The entries for one page that this session can show, in declaration order. */
    public static List<Option> options(Page page) {
        return ALL.stream().filter(o -> o.page() == page && o.offeredNow()).toList();
    }

    /** One option by its TOML path, or null when nothing declares it. */
    public static Option byPath(String path) {
        for (Option option : ALL) {
            if (option.path().equals(path)) {
                return option;
            }
        }
        return null;
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
     * The tier name the preset row shows. {@code tier} is a readback of the individual dials rather than
     * the stored preset value, so a player who turns one switch off sees "Custom" instead of the tier they
     * last clicked.
     */
    private static Component qualityLabel(int tier) {
        if (tier < 0) {
            return Component.translatable(RT + "volumeQuality.custom");
        }
        return Component.translatable(RT + "volumeQuality." + tier);
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
        if (CausticaConfig.Rt.ENABLED.tomlPath().equals(option.path()) && option.bool().value()) {
            // Switching ray tracing back on is the player asking for a retry, so the fallback latches
            // recorded by earlier frames must not keep the renderer on vanilla.
            VanillaRenderController.requestRtRetry();
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
        return add(out, page, group, path, nameKey, new Control.Toggle(), setting, Impact.LOW, Gate.OPEN, null);
    }

    private static Option toggle(List<Option> out, Page page, String group, String path, String nameKey,
                                 BooleanSetting setting, Gate gate) {
        return add(out, page, group, path, nameKey, new Control.Toggle(), setting, Impact.LOW, gate, null);
    }

    private static Option count(List<Option> out, Page page, String group, String path, String nameKey,
                                IntSetting setting, int min, int max, int step, Impact impact) {
        return count(out, page, group, path, nameKey, setting, min, max, step, impact, Gate.OPEN);
    }

    private static Option count(List<Option> out, Page page, String group, String path, String nameKey,
                                IntSetting setting, int min, int max, int step, Impact impact, Gate gate) {
        return add(out, page, group, path, nameKey, new Control.Count(min, max, step, Unit.COUNT),
                setting, impact, gate, null);
    }

    /** An integer setting whose values have names worth showing instead of numbers. */
    private static Option count(List<Option> out, Page page, String group, String path, String nameKey,
                                IntSetting setting, int min, int max, int step, Impact impact,
                                String valueKeys) {
        return add(out, page, group, path, nameKey,
                new Control.Count(min, max, step, Unit.COUNT, valueKeys), setting, impact, Gate.OPEN, null);
    }

    /** The named-value form, gated like any other option that depends on a switch above it. */
    private static Option count(List<Option> out, Page page, String group, String path, String nameKey,
                                IntSetting setting, int min, int max, int step, Impact impact, Gate gate,
                                String valueKeys) {
        return add(out, page, group, path, nameKey,
                new Control.Count(min, max, step, Unit.COUNT, valueKeys), setting, impact, gate, null);
    }

    /** A count whose caption is read back from other settings rather than from its own stored number. */
    private static Option countWithReadout(List<Option> out, Page page, String group, String path,
                                           String nameKey, IntSetting setting, int min, int max, int step,
                                           Impact impact, Gate gate, Supplier<Component> readout) {
        return add(out, page, group, path, nameKey, new Control.Count(min, max, step, Unit.COUNT),
                setting, impact, gate, readout);
    }

    private static Option scaled(List<Option> out, Page page, String group, String path, String nameKey,
                                 FloatSetting setting, int min, int max, int step, double scale, Unit unit,
                                 Impact impact) {
        return scaled(out, page, group, path, nameKey, setting, min, max, step, scale, unit, impact, Gate.OPEN);
    }

    private static Option scaled(List<Option> out, Page page, String group, String path, String nameKey,
                                 FloatSetting setting, int min, int max, int step, double scale, Unit unit,
                                 Impact impact, Gate gate) {
        return add(out, page, group, path, nameKey, new Control.Scaled(min, max, step, scale, unit),
                setting, impact, gate, null);
    }

    private static Option cycled(List<Option> out, Page page, String group, String path, String nameKey,
                                 Cycle cycle, RuntimeSetting<?> setting, Impact impact) {
        return cycled(out, page, group, path, nameKey, cycle, setting, impact, Gate.OPEN);
    }

    private static Option cycled(List<Option> out, Page page, String group, String path, String nameKey,
                                 Cycle cycle, RuntimeSetting<?> setting, Impact impact, Gate gate) {
        return add(out, page, group, path, nameKey, new Control.Cycled(cycle), setting, impact, gate, null);
    }

    private static Option add(List<Option> out, Page page, String group, String path, String nameKey,
                              Control control, RuntimeSetting<?> setting, Impact impact, Gate gate,
                              Supplier<Component> readout) {
        Option option = new Option(page, group, path, nameKey, control, setting, impact,
                RESTART_PATHS.contains(path), gate, readout);
        out.add(option);
        return option;
    }

}
