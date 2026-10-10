package dev.comfyfluffy.caustica;

import com.electronwill.nightconfig.core.CommentedConfig;
import com.electronwill.nightconfig.core.file.CommentedFileConfig;
import com.electronwill.nightconfig.core.file.FileNotFoundAction;
import com.electronwill.nightconfig.toml.TomlFormat;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.function.UnaryOperator;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Central mutable runtime configuration. Each setting resolves its value, in order of precedence, from a
 * {@code -Dcaustica.*} system property, then the {@code config/bettercaustica.toml} file, then a hardcoded
 * default. The settings UI and any other code call the same {@code set(...)} methods, and {@link #save()}
 * writes the current values back to the TOML file.
 *
 * <p>The system property namespace ({@code caustica.rt.foo}) and the TOML layout are independent: the file
 * uses real nested tables (e.g. {@code [omm]} with a {@code subdivision} key) grouped for readability, while
 * the property namespace stays flat and dotted for convenient one-off {@code -D} overrides.
 */
public final class CausticaConfig {
    private static final Logger LOGGER = LoggerFactory.getLogger("Caustica");
    private static final List<RuntimeSetting<?>> SETTINGS = new CopyOnWriteArrayList<>();

    private static final Path CONFIG_PATH = resolveConfigPath();
    private static final CommentedFileConfig FILE = loadFile(CONFIG_PATH);

    private CausticaConfig() {
    }

    public static List<RuntimeSetting<?>> settings() {
        return List.copyOf(SETTINGS);
    }

    public static Path configPath() {
        return CONFIG_PATH;
    }

    /**
     * Forces every settings holder to class-initialize so all settings are registered (and have applied
     * their file values). Call before {@link #save()} to write a complete file, and once at startup so the
     * file round-trips the full surface even for settings the renderer has not touched yet. Holders are
     * discovered reflectively rather than listed by hand: the list is exactly the kind of thing a new
     * holder forgets to join, and a missed holder silently drops its whole TOML section from first-run
     * files.
     */
    public static void ensureRegistered() {
        ensureRegistered(CausticaConfig.class);
    }

    private static void ensureRegistered(Class<?> root) {
        for (Class<?> holder : root.getDeclaredClasses()) {
            try {
                Class.forName(holder.getName(), true, CausticaConfig.class.getClassLoader());
            } catch (Throwable t) {
                LOGGER.warn("Could not initialize settings holder {}", holder.getName(), t);
            }
            ensureRegistered(holder);
        }
    }

    /** Writes the default config file if it does not exist yet. */
    public static void saveIfMissing() {
        ensureRegistered();
        if (FILE.valueMap().isEmpty()) {
            save();
        }
    }

    /** Serializes all registered settings to the TOML config file. */
    public static synchronized void save() {
        ensureRegistered();
        writeComments();
        for (RuntimeSetting<?> setting : SETTINGS) {
            setting.writeToFile(FILE);
        }
        FILE.save();
    }

    private static void writeComments() {
        FILE.setComment("enabled",
                " BetterCaustica ray-tracing settings. A matching -Dcaustica.* system property overrides a value here.");
        FILE.setComment("volume",
                " Volumetric media in the air around the terrain. fog enables the analytic height fog;\n"
                        + " fog-density is extinction at the base in 1/block, fog-height the e-folding\n"
                        + " altitude, fog-base-y the absolute world height where the constant-density region\n"
                        + " ends. fog-anisotropy forward-scatters sunlight into beams, fog-albedo is the\n"
                        + " single-scatter albedo, fog-ambient the sky-light term's weight, and fog-shadow\n"
                        + " gates the per-pixel sun shadow ray. Everything defaults to off.");
        FILE.setComment("terrain",
                " Controls terrain loading. Higher limits can load terrain faster but use more CPU and GPU time.");
        FILE.setComment("frame-generation",
                " Presents each rendered frame plus extra generated frames, so a 40 fps render shows as 80.\n"
                        + " backend picks who generates them: dlss is NVIDIA's (needs supported hardware),\n"
                        + " fsr is AMD FidelityFX (runs on any Vulkan 1.3 device), self interpolates from\n"
                        + " this mod's own motion vectors, and duplicate presents the rendered frame twice\n"
                        + " without interpolating anything, which measures the present path alone.\n"
                        + " multi-frame-count sets generated frames per rendered frame (1 = 2x, 2 = 3x).\n"
                        + " FSR generates one per frame, so it caps itself at 1.\n"
                        + " debug-view selects what the self backend writes: 0 the interpolated frame,\n"
                        + " 1 motion magnitude, 2 whether a pixel had a history sample, 3 history only.");
        FILE.setComment("reflex",
                " NVIDIA Reflex. Requires supported NVIDIA hardware and drivers.\n"
                        + " minimum-interval-us controls frame limiting; 0 disables the limit.");
        FILE.setComment("lights",
                " Controls direct lighting from glowing blocks such as torches, glowstone, and lava.\n"
                        + " ris-candidates is how many emitters each surface tests per frame (0-16); it\n"
                        + " takes effect on the next frame, but crossing 0 only restamps terrain that is\n"
                        + " re-meshed afterwards. stats, dump, and dump-radius are debugging options.");
        FILE.setComment("tonemap",
                " Controls the final image. gamma applies either way: 1 is neutral, lower brightens midtones.\n"
                        + " advanced-grading on (default) uses AgX, with agx-gain and agx-saturation\n"
                        + " tuned to land near the authored look's brightness and colorfulness.\n"
                        + " advanced-grading off uses the authored look, where look-enabled\n"
                        + " applies the look package's scene-referred LMT - turn that off to see the\n"
                        + " exposed image straight through ACES 2.0 - and look-override names a different\n"
                        + " .bin LMT beside the package's look.json (authoring aid; no rebuild needed, but\n"
                        + " the TOML is read once at startup, so a change applies on the next launch.\n"
                        + " purkinje is the dark-adaptation colour shift (reds dim, blue-green holds),\n"
                        + " needs auto exposure, and purkinje-strength 1 is the CIE mesopic response.");
        FILE.setComment("exposure",
                " Controls automatic exposure. manual-ev sets exposure in manual mode and adjusts it in auto mode.\n"
                        + " adapt-darken and adapt-brighten control adjustment speed in seconds.\n"
                        + " sky-weight-cap and emissive-weight-cap limit how much bright areas affect exposure.");
        FILE.setComment("hdr",
                " HDR display output. Requires operating system and display support.\n"
                        + " ui-nits controls UI brightness; peak-nits must be one of "
                        + joinNumbers(Rt.Hdr.PEAK_NITS_STEPS) + ".");
        FILE.setComment("screenshots",
                " exr-enabled saves an ACEScg EXR beside the normal F2 PNG while ray tracing is active.");
        FILE.setComment("composite",
                " Per-frame render budget and the water surface: spp is rays per pixel, max-bounces the\n"
                        + " bounce depth, water-waves/-opacity/-wave-height/-wave-size/-wave-speed/-reflection\n"
                        + " shape and tint the animated water. debug-view is a diagnostic switch.");
        FILE.setComment("omm",
                " Opacity micromaps (alpha-tested geometry acceleration). subdivision 0 disables them.\n"
                        + " stats prints per-build micromap counters; enabling it costs a little perf.");
        FILE.setComment("entities",
                " Ray-traced entities: max-ordinary-entities and block-entities.max-entities cap the\n"
                        + " traced entity count, textures.max-textures the entity albedo atlas, and pbr\n"
                        + " toggles LabPBR shading for entities. Changing the caps needs a restart.");
        FILE.setComment("particles",
                " Ray-traced particles. max-particles caps how many are traced; needs a restart.");
        FILE.setComment("overlay",
                " block-outline.enabled renders the targeted block's outline through the RT overlay path.");
        FILE.setComment("sky",
                " Vanilla sun and moon discs drawn by the RT sky. Turning them off leaves only the\n"
                        + " physically shaded atmosphere.");
        FILE.setComment("dlss-rr",
                " NVIDIA DLSS Ray Reconstruction: denoises and upscales the path-traced image.\n"
                        + " quality picks the render-percentage mode, preset forces a driver preset (0 =\n"
                        + " let the driver choose). Requires supported NVIDIA hardware and drivers.");
        FILE.setComment("frame-stats",
                " frame-stats.enabled writes per-frame counters to rt-frame-stats/frame.csv for tuning\n"
                        + " and bug reports.");
        FILE.setComment("diagnostics",
                " heavy-crash-diagnostics enables vendor Vulkan diagnostics and a full crash dump on\n"
                        + " device loss: much slower acceleration structure builds and a measurable\n"
                        + " framerate cost, so only while chasing a live crash.");
        FILE.setComment("ngx",
                " path points at a custom ngxshim.dll/directory when debugging the shim itself.\n"
                        + " Leave empty to use the bundled natives.");
        FILE.setComment("ffx",
                " path points at a custom ffxshim.dll/directory when debugging the shim itself.\n"
                        + " Leave empty to use the bundled natives.");
    }

    private static String joinNumbers(List<Integer> values) {
        StringBuilder out = new StringBuilder();
        for (Integer value : values) {
            if (out.length() > 0) {
                out.append(", ");
            }
            out.append(value);
        }
        return out.toString();
    }

    private static Path resolveConfigPath() {
        try {
            return FabricLoader.getInstance().getConfigDir().resolve("bettercaustica.toml");
        } catch (Throwable t) {
            return Path.of("config", "bettercaustica.toml");
        }
    }

    private static CommentedFileConfig loadFile(Path path) {
        CommentedFileConfig config = CommentedFileConfig.builder(path, TomlFormat.instance())
                .onFileNotFound(FileNotFoundAction.CREATE_EMPTY)
                .preserveInsertionOrder()
                .sync()
                .build();
        try {
            config.load();
        } catch (Exception e) {
            LOGGER.warn("Failed to read Caustica config {}: {}", path, e.toString());
            // Whatever parsed is unreliable and saveIfMissing would rebuild the file from defaults;
            // keep the original bytes under a .broken name so the user's values are not lost.
            preserveBrokenConfig(path);
            config.clear();
        }
        return config;
    }

    private static void preserveBrokenConfig(Path path) {
        try {
            Path broken = path.resolveSibling(path.getFileName() + ".broken");
            java.nio.file.Files.move(path, broken, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            LOGGER.warn("Preserved the unparseable config as {}", broken);
        } catch (Exception moveFailure) {
            LOGGER.warn("Could not preserve the unparseable config", moveFailure);
        }
    }

    private static Boolean fileBoolean(String tomlPath) {
        Object value = FILE.contains(tomlPath) ? FILE.get(tomlPath) : null;
        if (value instanceof Boolean b) {
            return b;
        }
        // A hand-edited TOML can hold any type for any key. The generic cast these accessors relied on
        // would explode inside a holder's static initializer; an unexpected type falls back to the
        // setting's default instead.
        if (value != null) {
            LOGGER.warn("Caustica config '{}' is a {}, expected a boolean; using the default",
                    tomlPath, value.getClass().getSimpleName());
        }
        return null;
    }

    private static Number fileNumber(String tomlPath) {
        Object value = FILE.contains(tomlPath) ? FILE.get(tomlPath) : null;
        if (value instanceof Number n) {
            return n;
        }
        if (value != null) {
            LOGGER.warn("Caustica config '{}' is a {}, expected a number; using the default",
                    tomlPath, value.getClass().getSimpleName());
        }
        return null;
    }

    private static String fileString(String tomlPath) {
        Object value = FILE.contains(tomlPath) ? FILE.get(tomlPath) : null;
        if (value instanceof String str) {
            return str;
        }
        if (value != null) {
            LOGGER.warn("Caustica config '{}' is a {}, expected a string; using the default",
                    tomlPath, value.getClass().getSimpleName());
        }
        return null;
    }

    public interface RuntimeSetting<T> {
        /** The {@code -Dcaustica.*} system property name that overrides this setting. */
        String key();

        /** The dotted path of this setting inside the nested {@code config/bettercaustica.toml} tables. */
        String tomlPath();

        T defaultValue();

        T get();

        void set(T value);

        /** Writes this setting's current value into the given config at {@link #tomlPath()}. */
        void writeToFile(CommentedConfig config);
    }

    public static final class BooleanSetting implements RuntimeSetting<Boolean> {
        private final String key;
        private final String tomlPath;
        private final boolean defaultValue;
        private volatile boolean value;

        private BooleanSetting(String key, String tomlPath, boolean defaultValue) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.defaultValue = defaultValue;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public Boolean defaultValue() {
            return defaultValue;
        }

        @Override
        public Boolean get() {
            return value;
        }

        public boolean value() {
            return value;
        }

        @Override
        public void set(Boolean value) {
            this.value = value != null ? value : defaultValue;
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            config.set(tomlPath, value);
        }

        private boolean resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                return Boolean.parseBoolean(prop.trim());
            }
            Boolean fromFile = fileBoolean(tomlPath);
            return fromFile != null ? fromFile : defaultValue;
        }
    }

    public static final class IntSetting implements RuntimeSetting<Integer> {
        private final String key;
        private final String tomlPath;
        private final int defaultValue;
        private final IntUnaryOperator sanitize;
        private volatile int value;

        private IntSetting(String key, String tomlPath, int defaultValue, IntUnaryOperator sanitize) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.defaultValue = sanitize.applyAsInt(defaultValue);
            this.sanitize = sanitize;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public Integer defaultValue() {
            return defaultValue;
        }

        @Override
        public Integer get() {
            return value;
        }

        public int value() {
            return value;
        }

        @Override
        public void set(Integer value) {
            this.value = sanitize.applyAsInt(value != null ? value : defaultValue);
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            config.set(tomlPath, value);
        }

        private int resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                try {
                    return sanitize.applyAsInt(Integer.parseInt(prop.trim()));
                } catch (NumberFormatException e) {
                    return defaultValue;
                }
            }
            Number fromFile = fileNumber(tomlPath);
            return fromFile != null ? sanitize.applyAsInt(fromFile.intValue()) : defaultValue;
        }
    }

    public static final class FloatSetting implements RuntimeSetting<Float> {
        private final String key;
        private final String tomlPath;
        private final float defaultValue;
        // Maps a raw external number (system property, file, or the constructor's raw default) into the
        // stored value domain, e.g. degrees -> radians.
        private final DoubleUnaryOperator inputTransform;
        // Inverse of inputTransform: maps the stored value domain back to the raw external domain (e.g.
        // radians -> degrees) for writeToFile, so a value round-trips through the file unchanged instead
        // of having inputTransform re-applied to an already-transformed number on the next load.
        private final DoubleUnaryOperator outputTransform;
        // Idempotent guard on a value-domain number (clamp / finite check); safe to apply to any source.
        private final DoubleUnaryOperator valueClamp;
        private volatile float value;

        private FloatSetting(String key, String tomlPath, float rawDefault, DoubleUnaryOperator inputTransform,
                             DoubleUnaryOperator outputTransform, DoubleUnaryOperator valueClamp) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.inputTransform = inputTransform;
            this.outputTransform = outputTransform;
            this.valueClamp = valueClamp;
            this.defaultValue = (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(rawDefault));
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public Float defaultValue() {
            return defaultValue;
        }

        @Override
        public Float get() {
            return value;
        }

        public float value() {
            return value;
        }

        @Override
        public void set(Float value) {
            if (value == null) {
                this.value = defaultValue;
            } else {
                this.value = (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(value));
            }
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            // Round-trip through Float.toString() so the file gets the shortest decimal that reproduces
            // this float (e.g. "0.6"), not outputTransform's raw double with float's binary noise spelled
            // out to 17 digits (e.g. 0.6000000487130328).
            float raw = (float) outputTransform.applyAsDouble(value);
            config.set(tomlPath, Double.parseDouble(Float.toString(raw)));
        }

        private float resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                try {
                    return (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(Double.parseDouble(prop.trim())));
                } catch (NumberFormatException e) {
                    return defaultValue;
                }
            }
            Number fromFile = fileNumber(tomlPath);
            if (fromFile == null) {
                return defaultValue;
            }
            return (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(fromFile.doubleValue()));
        }
    }

    public static final class StringSetting implements RuntimeSetting<String> {
        private final String key;
        private final String tomlPath;
        private final String defaultValue;
        private final UnaryOperator<String> sanitize;
        private volatile String value;

        private StringSetting(String key, String tomlPath, String defaultValue, UnaryOperator<String> sanitize) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.defaultValue = sanitize.apply(defaultValue);
            this.sanitize = sanitize;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public String defaultValue() {
            return defaultValue;
        }

        @Override
        public String get() {
            return value;
        }

        @Override
        public void set(String value) {
            this.value = sanitize.apply(value != null ? value : defaultValue);
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            config.set(tomlPath, value);
        }

        private String resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                return sanitize.apply(prop);
            }
            String fromFile = fileString(tomlPath);
            return sanitize.apply(fromFile != null ? fromFile : defaultValue);
        }
    }

    public static final class OptionalStringSetting implements RuntimeSetting<String> {
        private final String key;
        private final String tomlPath;
        private volatile String value;

        private OptionalStringSetting(String key, String tomlPath) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public String defaultValue() {
            return null;
        }

        @Override
        public String get() {
            return value;
        }

        @Override
        public void set(String value) {
            this.value = value;
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            if (value != null) {
                config.set(tomlPath, value);
            } else {
                config.remove(tomlPath);
            }
        }

        private String resolveInitial() {
            String prop = System.getProperty(key);
            return prop != null ? prop : fileString(tomlPath);
        }
    }

    public static final class Rt {
        public static final BooleanSetting ENABLED = bool("caustica.rt", "enabled", true);
        public static final IntSetting WORKER_THREADS =
                intAtLeast("caustica.rt.workerThreads", "worker-threads", defaultWorkerThreads(), 1);

        private Rt() {
        }

        public static final class Composite {
            public static final IntSetting DEBUG_VIEW = intValue("caustica.rt.debugView", "composite.debug-view", 0);
            public static final IntSetting SPP = intAtLeast("caustica.rt.spp", "composite.spp", 1, 1);
            public static final IntSetting MAX_BOUNCES =
                    clampedInt("caustica.rt.maxBounces", "composite.max-bounces", 4, 2, 8);
            public static final BooleanSetting WATER_WAVES =
                    bool("caustica.rt.waterWaves", "composite.water-waves", true);
            /**
             * Multiplier on water's Beer-Lambert extinction: 1 keeps the authored look, 0 is fully
             * clear water, up to 4 for deliberately murky bodies. Scales the floor together with the
             * tint-driven term so 0 leaves no residual absorption.
             */
            public static final FloatSetting WATER_OPACITY =
                    clampedFloat("caustica.rt.waterOpacity", "composite.water-opacity", 4.0f, 0.0f, 4.0f);
            /**
             * The water's own single-scatter fraction sigmaS/sigmaT: how much of the light the water
             * extinguishes it re-scatters instead of absorbing. Drives the underwater light shafts —
             * 0 is the previous behaviour (pure absorption, no in-scatter), higher values brighten
             * the water body and let sun beams read inside it. The scattering inherits the water's
             * tint by construction, so tinted bodies scatter tinted light.
             */
            public static final FloatSetting WATER_TURBIDITY =
                    clampedFloat("caustica.rt.waterTurbidity", "composite.water-turbidity",
                            0.35f, 0.0f, 1.0f);
            /**
             * Strength of the spray band hugging the water surface: mist whose density follows the
             * wave crests' steepness. Rides the marched fog integrator, so it needs volumetric fog
             * with the march tier on; 0 keeps it off entirely. Rain pushes it up on top of this base.
             */
            public static final FloatSetting WATER_SPRAY =
                    clampedFloat("caustica.rt.waterSpray", "composite.water-spray", 0.0f, 0.0f, 1.0f);
            /**
             * Wave height as a multiple of the authored spectrum: 1 is the tuned look, 0 flattens the
             * surface to a mirror, 3 is deliberately stormy. It scales the wave gradient, so reflection
             * break-up and caustic contrast move with it.
             */
            public static final FloatSetting WATER_WAVE_HEIGHT =
                    clampedFloat("caustica.rt.waterWaveHeight", "composite.water-wave-height",
                            0.8f, 0.0f, 3.0f);
            /**
             * Rate the wave clock advances at, as a multiple of the authored one. Relative speeds inside the
             * spectrum stay physical because deep-water dispersion still sets them; only the whole sea
             * slows down or speeds up. 0 freezes it.
             */
            public static final FloatSetting WATER_WAVE_SPEED =
                    clampedFloat("caustica.rt.waterWaveSpeed", "composite.water-wave-speed",
                            0.5f, 0.0f, 3.0f);
            /**
             * Wavelength multiplier. Every component scales together, so the spectrum keeps its shape and
             * only its size changes: 0.25 is ripple-sized chop, 4 is long ocean swell. Also scales the
             * caustic pattern, which differentiates the same field.
             */
            public static final FloatSetting WATER_WAVE_SCALE =
                    clampedFloat("caustica.rt.waterWaveScale", "composite.water-wave-scale",
                            1.5f, 0.25f, 4.0f);
            /**
             * Fraction of the Fresnel reflection a water surface keeps, so it controls how much of the
             * bright sky the sea mirrors back. 1 is a perfect dielectric; lower values hand the removed
             * energy to transmission, where the water's own absorption accounts for it, so the surface goes
             * darker rather than the light disappearing. Glass and other dielectrics ignore it.
             */
            public static final FloatSetting WATER_REFLECTION =
                    clampedFloat("caustica.rt.waterReflection", "composite.water-reflection",
                            0.8f, 0.0f, 1.0f);
            /**
             * How much water every surface has available, against the porosity its own LabPBR map was
             * authored with: 0 is fully dry, 1 lets a porous material absorb everything it asked for. It
             * scales the response uniformly, so the pack's porosity map alone decides WHICH surfaces react
             * and in what proportion — this only sets how wet the world is. Rain would drive the same
             * quantity; until then this is the whole weather system.
             */
            public static final FloatSetting WETNESS =
                    clampedFloat("caustica.rt.wetness", "composite.wetness", 0.0f, 0.0f, 1.0f);
            /**
             * How far a parallax-displaced texel may sit below its flat position, in sprite-local UV.
             * One sprite spans one block face, so 0.25 is LabPBR's authored 25% of a block and is the
             * value that matches what a pack draws. 0 disables the march entirely, which is also the
             * quickest way to tell a parallax artifact apart from everything else on a surface.
             */
            public static final FloatSetting PARALLAX_DEPTH =
                    clampedFloat("caustica.rt.parallaxDepth", "composite.parallax-depth",
                            0.25f, 0.0f, 0.5f);
            // Sun/moon angular radii and the noon south tilt moved into the versioned look package
            // (look.json "sky"): they shape the sky alongside the exposure curve, the LMT and the
            // photometric anchors that were already authored there, and splitting them across two
            // sources meant a package could not fully describe its own look.
            public static final FloatSetting JITTER_SIGN_X =
                    finiteFloat("caustica.rt.jitterSignX", "composite.jitter-sign-x", 1.0f);
            public static final FloatSetting JITTER_SIGN_Y =
                    finiteFloat("caustica.rt.jitterSignY", "composite.jitter-sign-y", -1.0f);

            private Composite() {
            }
        }

        /**
         * Volumetric media in the air around the terrain: height fog first, with cloud and spray
         * following their own phases. Every switch defaults to off, so a build that ignores this
         * section renders exactly what one without it did — the volume shaders gate on flags the
         * renderer only sets from these values.
         */
        public static final class Volume {
            /**
             * Master switch of the analytic height fog. Requires a dimension with sky light: the fog's
             * ambient term reads the sky irradiance basis, which is only baked where the sky lights the
             * world, so the renderer forces the fog off elsewhere regardless of this value.
             */
            public static final BooleanSetting FOG_ENABLED =
                    bool("caustica.rt.volume.fog", "volume.fog", false);
            /**
             * Fog extinction at the fog base, in 1/block — the same unit as the water medium's
             * extinction. 0.02 puts the e-folding distance at 50 blocks; above ~0.1 visibility inside
             * the fog drops to tens of blocks and the in-scatter term starts dominating the frame.
             */
            public static final FloatSetting FOG_DENSITY =
                    clampedFloat("caustica.rt.volume.fogDensity", "volume.fog-density",
                            0.02f, 0.0f, 0.25f);
            /**
             * Height scale H of the exponential falloff, in blocks: density at the base divided by e
             * every H blocks of altitude. Small values hug the ground, large ones fill valleys.
             */
            public static final FloatSetting FOG_HEIGHT =
                    clampedFloat("caustica.rt.volume.fogHeight", "volume.fog-height",
                            24.0f, 2.0f, 128.0f);
            /**
             * The fog base in absolute world Y. Below it the density is constant at
             * {@link #FOG_DENSITY}; above it the exponential falloff takes over. The renderer rebases
             * this into terrain coordinates per frame, so the profile stays pinned to the world when
             * the terrain origin shifts.
             */
            public static final FloatSetting FOG_BASE_Y =
                    clampedFloat("caustica.rt.volume.fogBaseY", "volume.fog-base-y",
                            62.0f, -64.0f, 320.0f);
            /**
             * Henyey–Greenstein anisotropy of the fog's phase function. Positive forward-scatters:
             * looking toward the light through fog brightens, which is what turns a flat bright haze
             * into a beam. 0 is isotropic; the useful range is roughly 0.3..0.6.
             */
            public static final FloatSetting FOG_ANISOTROPY =
                    clampedFloat("caustica.rt.volume.fogAnisotropy", "volume.fog-anisotropy",
                            0.5f, -0.9f, 0.9f);
            /**
             * Single-scatter albedo of the fog, shared by all channels: the fraction of extinguished
             * light the fog re-scatters rather than absorbs. Also the ratio the closed-form in-scatter
             * integral assumes between scattering and extinction.
             */
            public static final FloatSetting FOG_ALBEDO =
                    clampedFloat("caustica.rt.volume.fogAlbedo", "volume.fog-albedo",
                            0.85f, 0.0f, 1.0f);
            /**
             * Weight of the sky-irradiance ambient term in the fog's in-scatter. This is what keeps
             * night fog from going black: the term is the sky's own SH irradiance, so it fades with the
             * sky instead of adding a constant glow.
             */
            public static final FloatSetting FOG_AMBIENT =
                    clampedFloat("caustica.rt.volume.fogAmbient", "volume.fog-ambient",
                            1.0f, 0.0f, 1.0f);
            /**
             * The fog in-scatter's one-per-pixel geometric shadow ray toward the dominant celestial
             * body. Off leaves the sun term unoccluded — uniform bright fog with no god rays.
             */
            public static final BooleanSetting FOG_SHADOW =
                    bool("caustica.rt.volume.fogShadow", "volume.fog-shadow", true);
            /**
             * The marched fog tier: steps the fog integral instead of using the closed form, so the
             * density can gather into FBM billows ({@link #FOG_NOISE}) and every march sample casts its
             * own sun shadow ray. Costly — steps density evaluations plus up to steps shadow rays per
             * segment — but with noise at zero it converges to the analytic tier's exact result.
             */
            public static final BooleanSetting FOG_MARCH =
                    bool("caustica.rt.volume.fogMarch", "volume.fog-march", false);
            /**
             * Amplitude of the FBM billow modulation, 0..1: the density multiplier spans
             * [1-a, 1+a] around the analytic height profile. 0 reproduces the analytic field exactly,
             * which is what the march-to-analytic convergence test assumes.
             */
            public static final FloatSetting FOG_NOISE =
                    clampedFloat("caustica.rt.volume.fogNoise", "volume.fog-noise",
                            0.5f, 0.0f, 1.0f);
            /**
             * Wind speed in blocks per second: how fast the billow field drifts horizontally. The
             * drift is a monotone translation of a world-anchored field, so cloud shadows and fog
             * shapes move smoothly instead of boiling per frame.
             */
            public static final FloatSetting FOG_WIND_SPEED =
                    clampedFloat("caustica.rt.volume.fogWindSpeed", "volume.fog-wind-speed",
                            2.0f, 0.0f, 32.0f);
            /** Wind azimuth in degrees; 0 blows toward +X (east). */
            public static final FloatSetting FOG_WIND_ANGLE =
                    clampedFloat("caustica.rt.volume.fogWindAngle", "volume.fog-wind-angle",
                            0.0f, 0.0f, 360.0f);
            /**
             * March samples per path segment. The midpoint-quadrature error falls roughly with the
             * square of this; 16 is a balanced default, 8 reads grainy in billows, 64 is the
             * convergence reference.
             */
            public static final IntSetting FOG_MARCH_STEPS =
                    clampedInt("caustica.rt.volume.fogMarchSteps", "volume.fog-march-steps",
                            16, 4, 64);
            /**
             * Blue-noise jitter of the march samples, 0..1. 0 places every sample at its step's exact
             * midpoint (banding shows as the step edges); 1 scatters each sample uniformly inside its
             * step, dithering the edges into a stable grain the denoiser resolves.
             */
            public static final FloatSetting FOG_JITTER =
                    clampedFloat("caustica.rt.volume.fogJitter", "volume.fog-jitter",
                            1.0f, 0.0f, 1.0f);
            /**
             * How strongly rain thickens the fog: the pushed density scales by 1 + boost·rainLevel,
             * on top of (not instead of) the day curve. 0 keeps the fog weather-blind.
             */
            public static final FloatSetting FOG_RAIN_BOOST =
                    clampedFloat("caustica.rt.volume.fogRainBoost", "volume.fog-rain-boost",
                            0.5f, 0.0f, 1.0f);

            /**
             * Density fraction fully-shadowed fog keeps (the shadow cull's floor): 0 hard-culls the
             * fog from shadow (crisp but sterile edges), higher values keep more milk in sealed rooms
             * and shaded valleys. Rides fog1.w into the shaders; both the analytic and the marched
             * tier read the same floor.
             */
            public static final FloatSetting FOG_SHADOW_FLOOR =
                    clampedFloat("caustica.rt.volume.fogShadowFloor", "volume.fog-shadow-floor",
                            0.10f, 0.0f, 0.95f);

            /**
             * The planar cloud sheet and its shadow field (Phase 4). Off keeps every consumer inert:
             * the sheet never renders, the NEE cloud shadow multiplies by exactly 1, and the fog/water
             * sun legs are untouched — the md5 baseline survives.
             */
            public static final BooleanSetting CLOUD_ENABLED =
                    bool("caustica.rt.volume.cloud", "volume.cloud", false);
            /**
             * Fraction of the sky the coverage field claims, 0..1. 0 is an empty sky, 1 washes most
             * of the dome overcast; the field's soft threshold keeps edges cloud-shaped throughout.
             */
            public static final FloatSetting CLOUD_COVERAGE =
                    clampedFloat("caustica.rt.volume.cloudCoverage", "volume.cloud-coverage",
                            0.5f, 0.0f, 1.0f);
            /**
             * Absolute world height of the cloud plane (the sheet and the shadow field share it).
             * Already-rebased on the way into the shader like the fog base.
             */
            public static final FloatSetting CLOUD_ALTITUDE =
                    clampedFloat("caustica.rt.volume.cloudAltitude", "volume.cloud-altitude",
                            192.0f, -64.0f, 1024.0f);
            /**
             * Slab thickness in blocks. It means two things at once, and the second is why it goes this
             * high: the planar tier's optical depth scales with it, and the volumetric tier's CLOUD SIZE
             * is derived from it (both the 3D lobe cell and the cumulus planform cell are fractions of
             * it). A 48-block ceiling could only ever produce a deck of altocumulus elements, because a
             * real cumulus is 200..1000 blocks across and its vertical extent is what the slab has to
             * hold — a thick slab with a proportionally lower density is a sky of towers, a thin one with
             * a higher density is a sheet of lanes, and the column ends up the same either way.
             */
            public static final FloatSetting CLOUD_THICKNESS =
                    clampedFloat("caustica.rt.volume.cloudThickness", "volume.cloud-thickness",
                            12.0f, 0.5f, 256.0f);
            /**
             * Density multiplier on the optical depths: the shadow strength. 0 makes the sheet a
             * thin bright veil with no shadow; 1 casts full water-cloud-order shadows.
             */
            public static final FloatSetting CLOUD_DENSITY =
                    clampedFloat("caustica.rt.volume.cloudDensity", "volume.cloud-density",
                            1.0f, 0.0f, 3.0f);
            /**
             * How strongly the cloud layer gathers into localized systems with clear sky between
             * them, 0..1. 0 spreads the streaks uniformly over the whole dome (a woven mat); 1
             * breaks them into distinct cloud fields with real gaps. Fades out near full coverage,
             * so 100% coverage still delivers a solid overcast.
             */
            public static final FloatSetting CLOUD_PATCHINESS =
                    clampedFloat("caustica.rt.volume.cloudPatchiness", "volume.cloud-patchiness",
                            0.7f, 0.0f, 1.0f);
            /** Cloud field drift speed, blocks per second, on the wave clock like the fog wind. */
            public static final FloatSetting CLOUD_WIND_SPEED =
                    clampedFloat("caustica.rt.volume.cloudWindSpeed", "volume.cloud-wind-speed",
                            3.0f, 0.0f, 64.0f);
            /** Wind azimuth in degrees; 0 blows toward +X (east). */
            public static final FloatSetting CLOUD_WIND_ANGLE =
                    clampedFloat("caustica.rt.volume.cloudWindAngle", "volume.cloud-wind-angle",
                            0.0f, 0.0f, 360.0f);
            /**
             * Distance in blocks where the cloud sheet fully dissipates: crossings past it are not
             * rendered at all, and the fade begins at 60% of this range. Sheet-only — the shadow
             * field keeps working beyond it, since the layer still exists overhead.
             */
            public static final FloatSetting CLOUD_FADE_DISTANCE =
                    clampedFloat("caustica.rt.volume.cloudFadeDistance", "volume.cloud-fade-distance",
                            8000.0f, 512.0f, 16384.0f);
            /**
             * Render the cloud layer as a slab that is walked through instead of crossed once (Phase 5).
             * Off is the planar sheet and the md5 baseline; on replaces the sheet with the volumetric
             * tier, which spends a step budget per primary-ray pixel to gain real vertical structure,
             * parallax and a lit crown over a dark underbelly. Continuation rays keep the tier's mean
             * extinction either way, so reflections and bounces do not pay the walk.
             */
            public static final BooleanSetting CLOUD_VOLUMETRIC =
                    bool("caustica.rt.volume.cloudVolumetric", "volume.cloud-volumetric", false);
            /**
             * Steps the volumetric tier walks across the slab per primary-ray pixel. The cost is linear
             * in this and the quality of the dithering the denoiser has to resolve is inversely so:
             * below the shader's floor (8) the slab's rims alias visibly, above ~48 the extra samples
             * stop changing the image. Only read while {@link #CLOUD_VOLUMETRIC} is on.
             */
            public static final IntSetting CLOUD_STEPS =
                    clampedInt("caustica.rt.volume.cloudSteps", "volume.cloud-steps", 16, 8, 48);
            /**
             * Character of the volumetric deck, 0..1. It drives two things together, because a cumulus
             * is not a stratus with a different silhouette — it is a different planform AND a different
             * vertical profile: 0 is a stratus sheet (the 2D field's own streak lanes, sampled at their
             * own scale, in a board with softened rims), 1 a field of cumulus (the same lattice sampled
             * half as fine, so the lanes widen into cloud-sized patches, under the classic heap profile
             * of a flat base and a rounded crown). Both ends carry the same column depth, so this dial
             * changes the shape the light falls on and never how much cloud is overhead. Ignored while
             * {@link #CLOUD_VOLUMETRIC} is off.
             */
            public static final FloatSetting CLOUD_TYPE =
                    clampedFloat("caustica.rt.volume.cloudType", "volume.cloud-type", 0.6f, 0.0f, 1.0f);

            /**
             * How much weather thickens the deck, 0..1. Rain lifts the coverage toward solid overcast and
             * multiplies the density (darker shadows under a storm); thunder piles on top of rain. 0
             * holds the sky exactly to what the coverage and density sliders say whatever the weather.
             */
            public static final FloatSetting CLOUD_RAIN_BOOST =
                    clampedFloat("caustica.rt.volume.cloudRainBoost", "volume.cloud-rain-boost",
                            0.6f, 0.0f, 1.0f);

            /**
             * Weather strength the deck responds to: rain, with thunder adding a third again because a
             * thunderstorm decks over harder than the same rain without it.
             */
            public static float stormLevel(float rain, float thunder) {
                return Math.clamp(rain + 0.35f * thunder, 0.0f, 1.0f);
            }

            /** The coverage pushed for a weather: the slider's clear-sky value lifted toward overcast. */
            public static float rainCoverage(float coverage, float storm, float boost) {
                return Math.clamp(coverage + 0.45f * boost * storm, 0.0f, 1.0f);
            }

            /** The density pushed for a weather: a thicker deck also shadows harder. */
            public static float rainDensity(float density, float storm, float boost) {
                return density * (1.0f + 1.6f * boost * storm);
            }

            /**
             * The volumetric quality preset: 0..4 is Off / Low / Medium / High / Ultra. Reading it does
             * nothing on its own, and it is never applied at load — a hand-tuned configuration must not be
             * rewritten by an update. Its default is Off because that is what every volumetric switch
             * already defaults to, so on a fresh install the label states the truth. Selecting a tier WRITES the settings below, because a preset that only
             * overrode them inside the renderer would leave the switches on screen describing a frame
             * that is not the one being drawn. Tier 0 is the one tier with a hard requirement: it has to
             * render exactly like the renderer did before any of this existed.
             */
            private static final List<Integer> QUALITY_TIERS = List.of(0, 1, 2, 3, 4);

            public static final IntSetting QUALITY =
                    intChoice("caustica.rt.volume.quality", "volume.quality", 0, QUALITY_TIERS);

            /**
             * What one tier turns on, and the numbers it writes where it differs from the defaults. Pure
             * and separate from {@link #applyQuality} so the table can be reasoned about (and tested)
             * without a settings store behind it.
             *
             * <p>The thresholds climb one effect at a time. Above Off the clouds come back, above Low the
             * deck becomes volumetric rather than a sheet, above Medium the fog layer returns, and Ultra
             * additionally marches it. Water scattering follows the clouds rather than the fog because it
             * is what the underwater in-scatter is gated on, and Off zeroes it for the same reason it
             * zeroes everything else: a frame that still carried a volumetric term would not be the
             * pre-volumetric frame the tier promises.
             */
            public record QualityTier(boolean cloud, boolean cloudVolumetric, int cloudSteps,
                                      boolean fog, boolean fogShadow, boolean fogMarch, int fogMarchSteps,
                                      float waterTurbidity, float waterSpray) {
            }

            public static QualityTier qualityTier(int tier) {
                int t = Math.clamp(tier, 0, 4);
                return new QualityTier(
                        t >= 1,                       // cloud
                        t >= 2,                       // cloudVolumetric
                        switch (t) {
                            case 2 -> 12;
                            case 4 -> 40;
                            default -> 24;
                        },
                        t >= 3,                       // fog
                        t >= 2,                       // fogShadow
                        t >= 4,                       // fogMarch
                        32,                           // fogMarchSteps (only read when marching)
                        t >= 1 ? WATER_TURBIDITY_STOCK : 0.0f,
                        t >= 4 ? 0.5f : 0.0f);        // waterSpray (rides the marched fog tier)
            }

            /** The turbidity a tier above Off restores: the value the water page ships with. */
            private static final float WATER_TURBIDITY_STOCK = 0.35f;

            /**
             * The one option whose value means "write these other ones". It is a write rather than a
             * multiplier: afterwards the individual controls hold the tier's numbers and are the truth,
             * so picking a tier and then turning fog off simply leaves fog off.
             */
            public static void applyQuality(int tier) {
                QualityTier q = qualityTier(tier);
                CLOUD_ENABLED.set(q.cloud());
                CLOUD_VOLUMETRIC.set(q.cloudVolumetric());
                CLOUD_STEPS.set(q.cloudSteps());
                FOG_ENABLED.set(q.fog());
                FOG_SHADOW.set(q.fogShadow());
                FOG_MARCH.set(q.fogMarch());
                FOG_MARCH_STEPS.set(q.fogMarchSteps());
                CausticaConfig.Rt.Composite.WATER_TURBIDITY.set(q.waterTurbidity());
                CausticaConfig.Rt.Composite.WATER_SPRAY.set(q.waterSpray());
            }

            private Volume() {
            }
        }

        /**
         * Material response: how much of what a surface's own textures say about it the renderer honours.
         *
         * <p>These sit above the individual LabPBR channels so a player can isolate one channel against the
         * rest — a pack's emissive map against the block light around it, or porosity against wetness —
         * without editing the pack. Every switch defaults to on, so a build that ignores this section
         * renders exactly what one without it did.
         */
        public static final class Materials {
            /**
             * Ceiling on the per-hit parallax march's layer budget, in steps. The march derives its own
             * count from how far the displaced texel has to travel and caps it here, so this is the height
             * fetch count a frame pays for: higher resolves steeper views, lower is cheaper.
             */
            public static final IntSetting PARALLAX_LAYERS =
                    clampedInt("caustica.rt.materials.parallaxLayers", "materials.parallax-layers",
                            32, 8, 64);
            /** Run height-field parallax on entities as well as on terrain. */
            public static final BooleanSetting ENTITY_PARALLAX =
                    bool("caustica.rt.materials.entityParallax", "materials.entity-parallax", true);
            /** Honour the emissiveness a pack authored in its own specular map (_s alpha). */
            public static final BooleanSetting TEXTURE_EMISSION =
                    bool("caustica.rt.materials.textureEmission", "materials.texture-emission", true);
            /**
             * Honour emission authored by block state and block light instead of by a texture: a glowing
             * block state, and the light level a block model's cutout/translucent quads carry.
             */
            public static final BooleanSetting BLOCK_EMISSION =
                    bool("caustica.rt.materials.blockEmission", "materials.block-emission", true);
            /** Honour labPBR subsurface scattering (_s blue above the porosity split). */
            public static final BooleanSetting SUBSURFACE =
                    bool("caustica.rt.materials.subsurface", "materials.subsurface", true);
            /**
             * Honour labPBR porosity (_s blue at or below the porosity split). Porosity is what lets a wet
             * surface absorb water and darken, so switching it off keeps the wet sheen and drops the
             * absorption — the difference between "the world is wet" and "this material drinks it".
             */
            public static final BooleanSetting POROSITY =
                    bool("caustica.rt.materials.porosity", "materials.porosity", true);

            // The push-constant bit each switch occupies, mirroring MATERIAL_TOGGLE_* in world_common.slang.
            // RtMaterialLayoutTest checks these against the shader source, since the shader's copy is the
            // one that interprets them.
            public static final int TOGGLE_TEXTURE_EMISSION = 1;
            public static final int TOGGLE_BLOCK_EMISSION = 2;
            public static final int TOGGLE_SUBSURFACE = 4;
            public static final int TOGGLE_POROSITY = 8;
            public static final int TOGGLE_ENTITY_PARALLAX = 16;

            private Materials() {
            }
        }

        public static final class Sky {
            /**
             * Draw the sun the way vanilla does: the resource pack's own sprite, painted across a gnomonic
             * square about 33 degrees wide. Off substitutes a circular disc at the angular radius the
             * lighting already uses, so the drawn body and the penumbra it casts stop disagreeing.
             */
            public static final BooleanSetting VANILLA_SUN =
                    bool("caustica.rt.vanillaSun", "sky.vanilla-sun", false);
            /**
             * The same choice for the moon. Off renders a sphere rather than a sprite: the terminator, and
             * therefore the phase, comes out of the sun direction instead of being read from a texture.
             */
            public static final BooleanSetting VANILLA_MOON =
                    bool("caustica.rt.vanillaMoon", "sky.vanilla-moon", false);

            private Sky() {
            }
        }

        public static final class Terrain {
            // External keys retain their historical "per-tick" names for config compatibility; terrain
            // streaming is render-pass driven and these Java names reflect the actual scheduling unit.
            public static final IntSetting ASYNC_DISPATCH_PER_PASS =
                    intAtLeast("caustica.rt.asyncDispatchPerTick", "terrain.async-dispatch-per-tick", 32, 0);
            public static final IntSetting COMPLETION_RESULTS_PER_PASS =
                    intAtLeast("caustica.rt.sectionResultsPerTick", "terrain.section-results-per-tick", 32, 0);
            public static final IntSetting MAX_INFLIGHT_SECTIONS =
                    intAtLeast("caustica.rt.maxInflightSections", "terrain.max-inflight-sections", 32, 0);
            public static final IntSetting SECTION_TABLE_INITIAL_CAPACITY =
                    intAtLeast("caustica.rt.sectionTableInitialCapacity", "terrain.section-table-initial-capacity", 512, 1);
            public static final IntSetting REBASE_DISTANCE_BLOCKS =
                    intAtLeast("caustica.rt.rebaseDistanceBlocks", "terrain.rebase-distance-blocks", 128, 0);
            public static final BooleanSetting BLAS_COMPACTION =
                    bool("caustica.rt.blasCompaction", "terrain.blas-compaction", false);

            private Terrain() {
            }
        }

        /** RIS block-emitter lights. {@code ris-candidates = 0}, the default, disables everything. */
        public static final class Lights {
            /**
             * RIS candidate count M per diffuse vertex. Read per frame from the push constants, so a
             * change takes effect on the next frame — EXCEPT across zero: the terrain mesher stamps
             * each section with its NEE-emitter membership at meshing time, so existing sections keep
             * whatever they were stamped with until they are re-meshed. That is benign in both
             * directions rather than a correctness bug: when the stamp and {@code ris-candidates}
             * disagree, {@code gateEmitter} in indirect.rgen.slang is false, so the emitter's
             * direct-hit emission term is gathered normally — the surface is lit either way, it just
             * loses the RIS variance reduction on already-resident sections.
             */
            public static final IntSetting RIS_CANDIDATES =
                    clampedInt("caustica.rt.risCandidates", "lights.ris-candidates", 0, 0, 16);
            public static final FloatSetting MIN_FILL_RATIO =
                    finiteFloat("caustica.rt.lightMinFillRatio", "lights.min-fill-ratio", 0.25f);
            public static final BooleanSetting TEMPORAL_REUSE =
                    bool("caustica.rt.lightTemporalReuse", "lights.temporal-reuse", false);
            /**
             * How many effective samples one bounce-0 reservoir may accumulate before it stops admitting
             * history. This is the bias/variance dial of the temporal merge: at the candidate count the
             * merge contributes nothing and shading is single-frame only, and every step above narrows the
             * weight estimate while keeping the survivor's single shadow-ray test, which is what decides
             * whether reuse pays for itself against a temporal upscaler.
             */
            public static final IntSetting TEMPORAL_HISTORY =
                    clampedInt("caustica.rt.lightTemporalHistory", "lights.temporal-history", 8, 8, 256);
            public static final BooleanSetting STATS = bool("caustica.rt.lightStats", "lights.stats", false);
            public static final BooleanSetting DUMP = bool("caustica.rt.lightDump", "lights.dump", false);
            public static final IntSetting DUMP_RADIUS =
                    intAtLeast("caustica.rt.lightDumpRadius", "lights.dump-radius", 12, 1);

            private Lights() {
            }
        }

        public static final class Omm {
            public static final BooleanSetting ENABLED = bool("caustica.rt.omm", "omm.enabled", false);
            public static final IntSetting SUBDIVISION =
                    clampedInt("caustica.rt.ommSubdivision", "omm.subdivision", 4, 0, 6);
            public static final BooleanSetting STATS = bool("caustica.rt.ommStats", "omm.stats", false);

            private Omm() {
            }
        }

        public static final class Entities {
            public static final BooleanSetting ENABLED = bool("caustica.rt.entities", "entities.enabled", true);
            public static final BooleanSetting PARTICLES_ENABLED =
                    bool("caustica.rt.particles", "particles.enabled", true);
            public static final BooleanSetting GLOW_ENABLED =
                    bool("caustica.rt.glow", "entities.glow.enabled", true);
            public static final BooleanSetting NAME_TAGS_ENABLED =
                    bool("caustica.rt.nameTags", "entities.name-tags.enabled", true);
            /**
             * Traces the first-person camera owner for secondary rays (reflections/shadows/GI) so the
             * player sees themselves in water as others would; off excludes the body from every ray.
             * Primary rays never include it in first person either way.
             */
            public static final BooleanSetting SELF_ENABLED =
                    bool("caustica.rt.traceSelf", "entities.trace-self.enabled", false);
            /** Debug-only: render each model submission twice and require bitwise-identical CPU captures. */
            public static final BooleanSetting CAPTURE_PARITY =
                    bool("caustica.rt.entityCaptureParity", "entities.debug.capture-parity", false);
            public static final IntSetting MAX_ORDINARY_ENTITIES =
                    intAtLeast("caustica.rt.maxOrdinaryEntities", "entities.max-ordinary-entities", 1024, 0);
            public static final IntSetting MAX_BLOCK_ENTITIES =
                    intAtLeast("caustica.rt.maxBlockEntities", "entities.block-entities.max-entities", 1024, 0);
            public static final IntSetting MAX_PARTICLES =
                    intAtLeast("caustica.rt.maxParticles", "particles.max-particles", 1024, 0);
            public static final IntSetting BE_VIEW_CHUNKS =
                    intAtLeast("caustica.rt.beViewChunks", "entities.block-entities.view-chunks", 8, 0);
            public static final IntSetting BE_BUILDS_PER_FRAME =
                    intAtLeast("caustica.rt.beBuildsPerFrame", "entities.block-entities.builds-per-frame", 64, 0);
            public static final BooleanSetting REFIT_ENABLED =
                    bool("caustica.rt.entityRefit", "entities.refit.enabled", true);

            private Entities() {
            }

            public static int maxEntities() {
                // Saturating, not exact: the three settings are bounded individually, but their sum must
                // survive a hand-edited TOML with huge values, where addExact would throw on the render
                // path the first time it summed them.
                long total = (long) MAX_ORDINARY_ENTITIES.value() + MAX_BLOCK_ENTITIES.value()
                        + MAX_PARTICLES.value();
                return (int) Math.min(total, Integer.MAX_VALUE);
            }

            public static int entityListCapacity() {
                return Math.max(16, maxEntities());
            }

            public static int entityMapCapacity() {
                // Fastutil expected-size constructors apply their own load-factor headroom.
                return Math.max(16, MAX_ORDINARY_ENTITIES.value());
            }
        }

        public static final class EntityTextures {
            public static final IntSetting MAX_TEXTURES =
                    intAtLeast("caustica.rt.maxEntityTextures", "entities.textures.max-textures", 256, 1);
            public static final BooleanSetting PBR = bool("caustica.rt.entityPbr", "entities.textures.pbr", true);

            private EntityTextures() {
            }
        }

        public static final class Overlay {
            public static final BooleanSetting BLOCK_OUTLINE_ENABLED =
                    bool("caustica.rt.blockOutline", "overlay.block-outline.enabled", true);

            private Overlay() {
            }
        }

        public static final class DlssRr {
            public static final BooleanSetting ENABLED = bool("caustica.rt.dlssRr", "dlss-rr.enabled", true);
            public static final IntSetting PRESET = intValue("caustica.rt.dlssRr.preset", "dlss-rr.preset", 6);

            // NVSDK_NGX_PerfQuality_Value. Per NVIDIA's DLSS-RR programming guide, Ray Reconstruction only
            // supports Performance(0), Balanced(1), Quality(2), Ultra-Performance(3), and DLAA(5) —
            // Ultra Quality(4) is not a valid PerfQualityValue for RR (its optimal-settings query returns a
            // zeroed render size for it) and is deliberately excluded here.
            public static final List<Integer> QUALITY_STEPS = List.of(3, 0, 1, 2, 5);
            public static final IntSetting QUALITY =
                    intChoice("caustica.rt.dlssRr.quality", "dlss-rr.quality", 0, QUALITY_STEPS);

            private DlssRr() {
            }
        }

        /**
         * Frame generation: the extra-present loop plus whichever backend fills the generated frame.
         * Default off. A backend that this hardware or driver cannot run falls back to duplicating
         * the rendered frame rather than disabling the feature, so a bad choice costs smoothness and
         * not the whole frame.
         */
        public static final class Fg {
            public static final String DLSS = "dlss";
            public static final String FSR = "fsr";
            public static final String SELF = "self";
            public static final String DUPLICATE = "duplicate";
            public static final List<String> BACKENDS = List.of(DLSS, FSR, SELF, DUPLICATE);

            public static final BooleanSetting ENABLED = bool("caustica.rt.fg", "frame-generation.enabled", false);
            public static final IntSetting MULTI_FRAME_COUNT =
                    intAtLeast("caustica.rt.fg.multiFrameCount", "frame-generation.multi-frame-count", 1, 1);
            public static final StringSetting BACKEND = string("caustica.rt.fg.backend",
                    "frame-generation.backend", DLSS,
                    v -> BACKENDS.contains(v) ? v : DLSS);
            public static final IntSetting DEBUG_VIEW =
                    intAtLeast("caustica.rt.fg.debugView", "frame-generation.debug-view", 0, 0);

            private Fg() {
            }
        }

        /**
         * NVIDIA Reflex ({@code VK_NV_low_latency2}). Default on; gated additionally by device support.
         * The renderer configures the swapchain latency mode, paces frames with {@code vkLatencySleepNV},
         * and emits simulation, render-submit, and present latency markers.
         */
        public static final class Reflex {
            public static final BooleanSetting ENABLED = bool("caustica.rt.reflex", "reflex.enabled", true);
            public static final BooleanSetting LOW_LATENCY_BOOST =
                    bool("caustica.rt.reflex.boost", "reflex.low-latency-boost", false);
            public static final IntSetting MINIMUM_INTERVAL_US =
                    intAtLeast("caustica.rt.reflex.minIntervalUs", "reflex.minimum-interval-us", 0, 0);

            private Reflex() {
            }
        }

        public static final class Exposure {
            // Control points are measured-EV100 : compensation-EV, and compensation is exactly where the
            // frame's metered median lands: rendered = key * 2^comp, i.e. comp is EV from mid grey.
            //
            // Four points and only four (RtLookPackage rejects any other count), flat outside the ends, so
            // "-2:-3" doubles as the deepest-night lift and "15:1.0" as the brightest-day ceiling. The
            // segments compress 17 EV of scene range (-2..15) into 4 EV of rendered difference; the
            // diagnostics print 1 - the local slope, so 0.75 / 0.67 / 0.86.
            //
            // The absolute multiplier that lands a frame on that rendered level is
            //     log2(exposure) = log2(key) + comp - (evScene - 3) = 0.53 + comp - evScene,
            // because a scene value is cd/m2 and EV100 = log2(8L). Measured in game: a night frame
            // metering EV100 2.9 asks for -4.05 EV, a noon frame metering 15.4 asks for -13.9 EV.
            //
            // That is what minEv/maxEv below bound, and it makes max-ev = -2 a night-sky-only guard: it
            // engages once the metered median drops below about EV100 -0.5, which stops a frame of nearly
            // nothing but sky from being boosted into grey mush. A night frame with any lit ground in it
            // never reaches it.
            //
            // Because the curve sets the MEDIAN, adding scene light cannot brighten a surface: the
            // controller cancels it. Forcing a full moon over a crescent one lifted the night sky 3 EV and
            // the sand it over 0.7 EV. Only the curve or the metering moves how bright night reads.
            //
            // Daylight shade and a lit interior at night measure the SAME (~EV 7), so no luminance-only
            // curve can separate them -- what does is the asymmetric temporal adaptation below, which
            // holds a low exposure when you step from noon sun into shade. That is a real limit of this
            // controller, not a tuning miss.
            public static final StringSetting MODE =
                    string("caustica.rt.exposure.mode", "exposure.mode", "auto", Exposure::sanitizeMode);
            public static final FloatSetting MANUAL_EV =
                    clampedFloat("caustica.rt.exposure.manualEv", "exposure.manual-ev",
                            0.0f, -15.0f, 15.0f);
            public static final FloatSetting KEY = exposureScale("caustica.rt.exposure.key", "exposure.key", 0.18f);
            /**
             * How far night is moved off where the curve would place it, in EV. Negative darkens, positive
             * brightens, and 0 is the authored night.
             *
             * <p>Gate this on the world rather than on the metered scene level, and the reason is the
             * curve's own shape: compensation is a single frame-global number set by the metered median, so
             * a night frame that happens to hold a lit window or a lantern meters bright and the curve asks
             * for nothing at all -- exactly the frame the control exists for. The gate is the world's own
             * clock, so it says what the sky looks like regardless of where the camera points, and a
             * dimension with no sky light is never moved.
             *
             * <p>The offset is folded into the curve's two night control points, so it is full at and below
             * the first one and tapers to nothing at EV100 8, which leaves midday exactly as the look
             * package set it.
             */
            public static final FloatSetting NIGHT_LIFT =
                    clampedFloat("caustica.rt.exposure.nightLift", "exposure.night-lift",
                            -1.5f, -6.0f, 6.0f);
            /**
             * Adaptation time constants in seconds, applied in EV space by the resolve. Named for what
             * the SCENE did: walking into a dark cave is "darken" (exposure has to rise), stepping back
             * out is "brighten".
             *
             * <p>Asymmetric on purpose, and in the direction human vision actually works — light
             * adaptation takes seconds, dark adaptation takes minutes. Every shipping game compresses
             * that, but keeping the sign right is what makes a sunrise read as a sunrise instead of as a
             * lens. The names describe the scene change, not the inverse movement of the exposure multiplier.
             */
            public static final FloatSetting ADAPT_DARKEN =
                    exposureScale("caustica.rt.exposure.adaptDarken", "exposure.adapt-darken", 2.01f);
            public static final FloatSetting ADAPT_BRIGHTEN =
                    exposureScale("caustica.rt.exposure.adaptBrighten", "exposure.adapt-brighten", 0.4f);
            public static final FloatSetting LOW_PERCENTILE =
                    clampedFloat("caustica.rt.exposure.lowPercentile", "exposure.low-percentile", 0.50f, 0.0f, 1.0f);
            public static final FloatSetting HIGH_PERCENTILE =
                    clampedFloat("caustica.rt.exposure.highPercentile", "exposure.high-percentile", 0.95f, 0.0f, 1.0f);
            public static final IntSetting STRIDE =
                    clampedInt("caustica.rt.exposure.stride", "exposure.stride", 2, 1, 8);
            public static final FloatSetting CENTER_WEIGHT_SIGMA =
                    clampedFloat("caustica.rt.exposure.centerWeightSigma",
                            "exposure.center-weight-sigma", 0.35f, 0.01f, 2.0f);
            public static final FloatSetting CENTER_WEIGHT_FLOOR =
                    clampedFloat("caustica.rt.exposure.centerWeightFloor",
                            "exposure.center-weight-floor", 0.15f, 0.0f, 1.0f);
            public static final FloatSetting SKY_WEIGHT_CAP =
                    clampedFloat("caustica.rt.exposure.skyWeightCap",
                            "exposure.sky-weight-cap", 0.25f, 0.0f, 1.0f);
            public static final FloatSetting EMISSIVE_WEIGHT_CAP =
                    clampedFloat("caustica.rt.exposure.emissiveWeightCap",
                            "exposure.emissive-weight-cap", 0.10f, 0.0f, 1.0f);
            /**
             * Pre-exposure: raygen multiplies scene radiance by the previous frame's exposure before
             * the fp16 write, and the display pass divides it back out, so stored values sit near
             * {@code key} instead of spanning the ~26 EV physical photometric units require. The two
             * cancel algebraically, so <b>toggling this must not change the
             * image</b>; it exists as an A/B switch for exactly that check, and as an escape hatch
             * if DLSS-RR ever proves sensitive to its history being at the previous frame's scale.
             */
            public static final BooleanSetting PRE_EXPOSURE =
                    bool("caustica.rt.exposure.preExposure", "exposure.pre-exposure", true);

            private Exposure() {
            }

            /**
             * Bounds on the ABSOLUTE exposure multiplier, authored alongside the curve in the look package.
             * Measured demand runs from -13.9 EV on a noon frame to -4.1 EV on a night frame carrying lit
             * ground, so max-ev = -2 engages only once the metered median drops below about EV100 -0.5 --
             * a frame of nearly nothing but night sky -- where it stops the controller from boosting the
             * sky into grey mush.
             *
             * <p>min-ev deliberately does not cover a zoomed-in sun, which asks for about -20.8: letting
             * the whole frame go black because the sun is in shot is worse than clamping it, and the sky
             * metering cap already bounds the sun's share.
             */
            public static float minEv() {
                return dev.comfyfluffy.caustica.rt.RtLookPackage.current().exposure().minEv();
            }

            public static float maxEv() {
                return dev.comfyfluffy.caustica.rt.RtLookPackage.current().exposure().maxEv();
            }

            public static String curve() {
                return dev.comfyfluffy.caustica.rt.RtLookPackage.current().exposure().curve();
            }

            /**
             * Sanity bound on an exposure multiplier, not an artistic one. It must remain below the
             * 3.8e-6 multiplier requested by {@code -18 EV}; min-ev/max-ev provides the artistic bound.
             */
            public static float clampScale(float value) {
                return Math.clamp(value, 1.0e-8f, 1.0e8f);
            }

            private static String sanitizeMode(String value) {
                if ("auto".equalsIgnoreCase(value)) {
                    return "auto";
                }
                if ("manual".equalsIgnoreCase(value)) {
                    return "manual";
                }
                return "auto";
            }

        }

        /** Scene-referred look transform and baked SDR/HDR ACES display transforms. */
        public static final class Tonemap {
            public static final FloatSetting GAMMA =
                    clampedFloat("caustica.rt.tonemap.gamma", "tonemap.gamma", 1.0f, 0.1f, 5.0f);

            /**
             * The Purkinje shift: as the eye adapts to dim light its sensitivity peak moves from
             * 555 nm to about 507 nm, so saturated red loses nearly all of its apparent brightness
             * while blue-green holds. The display pass applies the CIE 191:2010 mesopic re-weighting
             * to the exposed scene-linear image, before either display transform, so it reaches the
             * authored look, AgX and the HDR image alike. The scotopic basis it needs is baked from
             * CIE 018:2019 tables by {@code tools/bake_scotopic_basis.py}.
             *
             * <p>Needs auto exposure. The shift adapts to the metered ABSOLUTE scene luminance
             * ({@code ExposureState.evScene}), and the meter only refreshes that in auto mode — with
             * {@link Exposure#MODE} on manual there is no luminance to adapt to and this is skipped.
             * Read per frame, so it is safe to expose as a live toggle; it rebuilds nothing.
             */
            public static final BooleanSetting PURKINJE =
                    bool("caustica.rt.tonemap.purkinje", "tonemap.purkinje", true);

            /**
             * Scale on the Purkinje deviation from the working-space neutral, which always keeps its
             * brightness. 1 is the CIE mesopic response itself; below 1 weakens the red collapse, above
             * 1 deepens it. Only read while {@link #PURKINJE} is on.
             */
            public static final FloatSetting PURKINJE_STRENGTH =
                    clampedFloat("caustica.rt.tonemap.purkinjeStrength", "tonemap.purkinje-strength",
                            1.0f, 0.0f, 2.0f);

            /**
             * Selects which grading the display pass applies. True — the default — is the advanced
             * grading: AgX on the exposed image, with no look package involved and its own
             * exposure/saturation controls. False is the authored grading: the look package's LMT
             * followed by the baked ACES 2.0 output transform, which is what {@link #LOOK_ENABLED} and
             * {@link #LOOK_OVERRIDE} shape.
             *
             * <p>Read per frame, so it is safe to expose as a live toggle; it rebuilds nothing.
             */
            public static final BooleanSetting ADVANCED_GRADING =
                    bool("caustica.rt.tonemap.advancedGrading", "tonemap.advanced-grading", true);

            /**
             * AgX incident-exposure gain: the multiplier on scene-linear ACEScg before AgX's log2
             * encoding step around its 0.18 mid-grey pivot. Required because the authored LMT contributes
             * a large exposure offset that AgX has no equivalent of, so the same auto-exposure curve
             * leaves the image far too bright without it. Only read while {@link #ADVANCED_GRADING} is on.
             */
            public static final FloatSetting AGX_GAIN =
                    clampedFloat("caustica.rt.tonemap.agxGain", "tonemap.agx-gain", 0.7f, 0.02f, 8.0f);

            /**
             * Chroma scale applied to AgX's linear BT.709 output. AgX spends most of its 14-stop range on
             * the highlight shoulder, which flattens chroma, so on a photometric scene it lands well below
             * the authored look's saturation and needs this to be brought back up. 1 is AgX's own output.
             * Only read while {@link #ADVANCED_GRADING} is on.
             */
            public static final FloatSetting AGX_SATURATION =
                    clampedFloat("caustica.rt.tonemap.agxSaturation", "tonemap.agx-saturation", 1.15f, 0.0f, 3.0f);

            /**
             * Scene-referred look transform (LMT), part of the authored grading. False skips the
             * {@code applyLook} sample entirely, so the exposed ACEScg image goes straight to the ACES
             * 2.0 display transform. This is the quickest way to tell whether a given look is what is
             * flattening the image — the LMT is a baked 65³ CLUT and iterating on it otherwise means a
             * full rebuild.
             *
             * <p>Do not read this as "the look package is optional": exposure, the LMT and the
             * photometric anchors are one authored look. This switch exists for authoring and A/B
             * diagnosis, and the default remains true so a shipped build is unchanged.
             */
            public static final BooleanSetting LOOK_ENABLED =
                    bool("caustica.rt.tonemap.lookEnabled", "tonemap.look-enabled", true);

            /**
             * Which LMT inside the active look package to sample, by bare file name of a
             * {@code .bin} CLUT that sits beside {@code look.json}. Empty uses the
             * package's own {@code lmt.resource}, and the shipped default names
             * {@code lmt_toe_mid.bin}. The file must still satisfy {@link
             * dev.comfyfluffy.caustica.rt.pipeline.RtToneLut}'s header + shaper contract, and a bad
             * name falls back to the packaged LMT with a warning rather than failing the frame.
             */
            public static final StringSetting LOOK_OVERRIDE =
                    string("caustica.rt.tonemap.lookOverride", "tonemap.look-override", "lmt_toe_mid.bin",
                            Tonemap::sanitizeLmtName);

            private Tonemap() {
            }

            /**
             * Rejects anything that could escape the look package's directory when concatenated onto it.
             * A rejected value is swallowed to empty (the packaged LMT) instead of throwing, because this
             * is read while building the display pipeline and a typo should not take the renderer down.
             */
            private static String sanitizeLmtName(String value) {
                if (value == null) {
                    return "";
                }
                String trimmed = value.trim();
                if (trimmed.isEmpty() || trimmed.contains("/") || trimmed.contains("\\")
                        || ".".equals(trimmed) || "..".equals(trimmed)) {
                    return "";
                }
                return trimmed.endsWith(".bin") ? trimmed : trimmed + ".bin";
            }
        }

        /** Render-frame timing + hitch logging. See {@code RtFrameStats}. */
        public static final class FrameStats {
            public static final BooleanSetting ENABLED = bool("caustica.rt.frameStats", "frame-stats.enabled", false);

            private FrameStats() {
            }
        }

        /** Optional high-dynamic-range screenshot output paired with vanilla's F2 PNG. */
        public static final class Screenshots {
            public static final BooleanSetting EXR_ENABLED =
                    bool("caustica.rt.screenshots.exr", "screenshots.exr-enabled", false);

            private Screenshots() {
            }
        }

        /** Startup Vulkan inventory + {@code VK_EXT_device_fault} reporting on device loss. See {@code VulkanDiagnostics}. */
        public static final class Diagnostics {
            /** Heavy driver-side crash diagnostics: vendor diagnostics-config extensions (shader debug
             * info, resource tracking, automatic checkpoints, shader error reporting) and the
             * {@code deviceFaultVendorBinary} feature (vendor-format crash dump on device loss). Off by
             * default: measured ~10x BLAS build time / -20% fps when enabled. Plain {@code deviceFault}
             * reporting (fault addresses + vendor records) is always on and unaffected. Turn on only
             * while chasing a live device-loss crash. */
            public static final BooleanSetting HEAVY_CRASH_DIAGNOSTICS =
                    bool("caustica.rt.heavyCrashDiagnostics", "diagnostics.heavy-crash-diagnostics", false);

            private Diagnostics() {
            }
        }

        /**
         * HDR display output. When enabled the swapchain is created in PQ (ST.2084/HDR10 — the display-ready
         * encoding both HDR10 swapchains and DLSS Frame Generation require; whatever pixel format the surface
         * pairs with that color space, commonly a 10-bit UNORM), falling back to SDR if the surface doesn't
         * advertise it. The ACES LUT owns scene-to-display mapping; {@code uiNits} places SDR-authored UI
         * in that PQ output, while {@code peakNits} selects the LUT's mastering target.
         */
        public static final class Hdr {
            public static final BooleanSetting ENABLED = bool("caustica.rt.hdr", "hdr.enabled", false);
            public static final FloatSetting UI_NITS =
                    clampedFloat("caustica.rt.hdr.uiNits", "hdr.ui-nits", 200.0f, 80.0f, 500.0f);

            // ACES HDR LUTs are available only for these mastering targets.
            public static final List<Integer> PEAK_NITS_STEPS = List.of(500, 1000, 2000, 4000);
            public static final IntSetting PEAK_NITS =
                    intChoice("caustica.rt.hdr.peakNits", "hdr.peak-nits", 1000, PEAK_NITS_STEPS);

            // Surface capability and current swapchain state are separate: HDR controls remain available
            // while the swapchain is native SDR, so enabling HDR can recreate it in PQ.
            private static volatile boolean SWAPCHAIN_PQ_AVAILABLE = false;
            private static volatile boolean SWAPCHAIN_PQ_ACTIVE = false;

            private Hdr() {
            }

            public static void setSwapchainPqAvailable(boolean available) {
                SWAPCHAIN_PQ_AVAILABLE = available;
            }

            public static void setSwapchainPqActive(boolean active) {
                SWAPCHAIN_PQ_ACTIVE = active;
            }

            /**
             * Whether this session's surface can create a PQ swapchain, independent of which format the
             * current swapchain uses.
             */
            public static boolean swapchainPqAvailable() {
                return SWAPCHAIN_PQ_AVAILABLE;
            }

            /** Whether the currently configured swapchain is HDR10/PQ rather than native SDR. */
            public static boolean swapchainPqActive() {
                return SWAPCHAIN_PQ_ACTIVE;
            }

            /**
             * Whether the HDR display path (world HDR + PQ swapchain + UI overlay) should be active this
             * frame. The option invalidates the surface configuration after changing {@link #ENABLED};
             * the ordinary resize/configure path recreates the swapchain in SDR or PQ.
             */
            public static boolean enabled() {
                return SWAPCHAIN_PQ_ACTIVE && ENABLED.value();
            }

            /** Absolute brightness assigned to SDR-authored UI in the PQ output. */
            public static float uiNits() {
                return UI_NITS.value();
            }

        }
    }

    public static final class Ngx {
        public static final OptionalStringSetting PATH = optionalString("caustica.ngx.path", "ngx.path");

        private Ngx() {
        }
    }

    /**
     * FidelityFX. {@code path} points at {@code ffxshim.dll} or a directory holding it and
     * {@code amd_fidelityfx_vk.dll}, which is how a locally built SDK drop is tested without
     * rebuilding the mod. Unset uses the natives bundled in the jar.
     */
    public static final class Ffx {
        public static final OptionalStringSetting PATH = optionalString("caustica.ffx.path", "ffx.path");

        private Ffx() {
        }
    }

    private static BooleanSetting bool(String key, String tomlPath, boolean fallback) {
        return new BooleanSetting(key, tomlPath, fallback);
    }

    private static StringSetting string(String key, String tomlPath, String fallback, UnaryOperator<String> sanitize) {
        return new StringSetting(key, tomlPath, fallback, sanitize);
    }

    private static OptionalStringSetting optionalString(String key, String tomlPath) {
        return new OptionalStringSetting(key, tomlPath);
    }

    private static IntSetting intValue(String key, String tomlPath, int fallback) {
        return new IntSetting(key, tomlPath, fallback, v -> v);
    }

    private static IntSetting intAtLeast(String key, String tomlPath, int fallback, int min) {
        return new IntSetting(key, tomlPath, fallback, v -> Math.max(min, v));
    }

    private static IntSetting intChoice(String key, String tomlPath, int fallback, List<Integer> choices) {
        return new IntSetting(key, tomlPath, fallback, v -> choices.contains(v) ? v : fallback);
    }

    private static IntSetting clampedInt(String key, String tomlPath, int fallback, int min, int max) {
        return new IntSetting(key, tomlPath, fallback, v -> Math.clamp(v, min, max));
    }

    private static FloatSetting finiteFloat(String key, String tomlPath, float fallback) {
        return new FloatSetting(key, tomlPath, fallback, v -> v, v -> v, v -> Double.isFinite(v) ? v : fallback);
    }

    private static FloatSetting exposureScale(String key, String tomlPath, float fallback) {
        // Math.clamp passes NaN through, and parseDouble accepts "nan"; a non-finite value must fall
        // back to the default rather than poison the stored setting and round-trip through the file.
        return new FloatSetting(key, tomlPath, fallback, v -> v, v -> v,
                v -> Double.isFinite(v) ? Math.clamp(v, 1.0e-4, 1.0e4) : fallback);
    }

    private static FloatSetting clampedFloat(String key, String tomlPath, float fallback, float min, float max) {
        return new FloatSetting(key, tomlPath, fallback, v -> v, v -> v,
                v -> Double.isFinite(v) ? Math.clamp(v, min, max) : fallback);
    }

    private static int defaultWorkerThreads() {
        return Math.clamp(Runtime.getRuntime().availableProcessors() / 2, 1, 4);
    }
}
