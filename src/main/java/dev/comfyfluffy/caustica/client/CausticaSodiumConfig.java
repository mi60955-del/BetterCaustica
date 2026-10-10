package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.CausticaSettings.Control;
import dev.comfyfluffy.caustica.client.CausticaSettings.Cycle;
import dev.comfyfluffy.caustica.client.CausticaSettings.Option;
import dev.comfyfluffy.caustica.client.CausticaSettings.Page;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import net.caffeinemc.mods.sodium.api.config.ConfigEntryPoint;
import net.caffeinemc.mods.sodium.api.config.StorageEventHandler;
import net.caffeinemc.mods.sodium.api.config.option.OptionFlag;
import net.caffeinemc.mods.sodium.api.config.option.OptionImpact;
import net.caffeinemc.mods.sodium.api.config.structure.BooleanOptionBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.ConfigBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.EnumOptionBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.IntegerOptionBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.ModOptionsBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.OptionBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.OptionGroupBuilder;
import net.caffeinemc.mods.sodium.api.config.structure.OptionPageBuilder;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;

/**
 * Publishes {@link CausticaSettings} inside Sodium's video settings screen.
 *
 * <p>Sodium replaces the vanilla {@code VideoSettingsScreen}, so {@code VideoSettingsScreenMixin} never
 * runs when Sodium is installed and the RT controls would otherwise be unreachable from the menus. Sodium
 * exposes a public config API reached through the {@code sodium:config_api_user} Fabric entry point, which
 * is what this class implements; it is compiled against but never depends on Sodium, and loads only when
 * Sodium is present.
 *
 * <p>Every option binds straight to its {@code RuntimeSetting}, so this screen, the vanilla screen and
 * {@code config/bettercaustica.toml} stay one value rather than three that can disagree.
 *
 * <p>Sodium's own controls are boolean, integer, enum and external-option; there is no floating-point or
 * text control, so a {@link Control.Scaled} becomes an integer widget carrying {@code value * scale} with
 * a formatter that puts the decimal point back — the same approach {@link RtVideoOptions} takes for
 * vanilla's percent sliders. The two free-text settings ({@code ngx.path} and
 * {@code tonemap.look-override}) stay TOML-only.
 *
 * <p>Sodium validates every option when the group is built, not when the screen is opened: a missing
 * tooltip, binding, storage handler or default value aborts resource reload and crashes the client. An
 * integer option additionally needs both a range and a value formatter. Every builder here supplies them
 * unconditionally, and {@code CausticaSodiumConfigTest} builds this whole tree headlessly to keep it that
 * way.
 */
public final class CausticaSodiumConfig implements ConfigEntryPoint {
    /** Persists the whole setting set once Sodium commits a change, matching the vanilla screen. */
    private static final StorageEventHandler SAVE = CausticaConfig::save;

    enum ExposureMode {
        AUTO, MANUAL;

        static ExposureMode of(String stored) {
            return CausticaSettings.normalizeExposureMode(stored).equals("manual") ? MANUAL : AUTO;
        }

        String stored() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    // The enum order must match CausticaConfig.Rt.DlssRr.QUALITY_STEPS, which owns the actual
    // NVSDK_NGX_PerfQuality_Value numbers; the index is only a UI ordering. Package-private so the
    // test can pin the two together.
    enum DlssQuality {
        ULTRA_PERFORMANCE, PERFORMANCE, BALANCED, QUALITY, DLAA;

        int value() {
            return CausticaConfig.Rt.DlssRr.QUALITY_STEPS.get(ordinal());
        }
    }

    // Likewise ordered against CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS: the ACES HDR display transform is
    // baked only for these mastering targets. Package-private so the test can pin the two together.
    enum HdrPeak {
        NITS_500, NITS_1000, NITS_2000, NITS_4000;

        int value() {
            return CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS.get(ordinal());
        }
    }

    @Override
    public void registerConfigLate(ConfigBuilder config) {
        ModOptionsBuilder mod = config.registerOwnModOptions();
        // Pages with nothing to offer on this session's hardware are skipped, the same gate the vanilla
        // screen applies, rather than registered as an empty page.
        for (Page page : Page.values()) {
            if (!CausticaSettings.options(page).isEmpty()) {
                mod.addPage(buildPage(config, page));
            }
        }
    }

    /** One page per {@link Page}, with the groups in the order the catalog first uses each of them. */
    private static OptionPageBuilder buildPage(ConfigBuilder config, Page page) {
        Map<String, OptionGroupBuilder> groups = new LinkedHashMap<>();
        for (Option option : CausticaSettings.options(page)) {
            groups.computeIfAbsent(option.group(), name -> group(config, name))
                    .addOption(buildOption(config, option));
        }
        OptionPageBuilder built = config.createOptionPage()
                .setName(Component.translatable(page.titleKey()));
        for (OptionGroupBuilder group : groups.values()) {
            built.addOptionGroup(group);
        }
        return built;
    }

    private static OptionBuilder buildOption(ConfigBuilder config, Option option) {
        return switch (option.control()) {
            case Control.Toggle ignored -> toggle(config, option);
            case Control.Count count -> count(config, option, count);
            case Control.Scaled scaled -> scaled(config, option, scaled);
            case Control.Cycled cycled -> cycled(config, option, cycled.cycle());
        };
    }

    private static OptionBuilder cycled(ConfigBuilder config, Option option, Cycle cycle) {
        return switch (cycle) {
            case EXPOSURE_MODE -> exposureMode(config, option);
            case DLSS_QUALITY -> dlssQuality(config, option);
            case HDR_PEAK -> hdrPeak(config, option);
        };
    }

    private static OptionGroupBuilder group(ConfigBuilder config, String name) {
        return config.createOptionGroup().setName(Component.translatable(CausticaSettings.groupKey(name)));
    }

    private static Identifier id(String path) {
        return Identifier.parse("bettercaustica:" + path);
    }

    private static OptionImpact impact(Option option) {
        return switch (option.impact()) {
            case LOW -> OptionImpact.LOW;
            case MEDIUM -> OptionImpact.MEDIUM;
            case HIGH -> OptionImpact.HIGH;
        };
    }

    private static OptionFlag[] flags(Option option) {
        return option.restart() ? new OptionFlag[]{OptionFlag.REQUIRES_GAME_RESTART} : new OptionFlag[0];
    }

    // ---- Controls ----------------------------------------------------------------------------------

    private static BooleanOptionBuilder toggle(ConfigBuilder config, Option option) {
        BooleanOptionBuilder built = config.createBooleanOption(id(option.path()))
                .setName(Component.translatable(option.nameKey()))
                .setTooltip(Component.translatable(option.tooltipKey()))
                .setDefaultValue(option.bool().defaultValue())
                .setBinding(v -> { option.bool().set(v); CausticaSettings.afterChange(option); }, option.bool()::get)
                .setStorageHandler(SAVE);
        OptionFlag[] flags = flags(option);
        if (flags.length > 0) {
            built.setFlags(flags);
        }
        return built;
    }

    private static IntegerOptionBuilder count(ConfigBuilder config, Option option, Control.Count control) {
        IntegerOptionBuilder built = config.createIntegerOption(id(option.path()))
                .setName(Component.translatable(option.nameKey()))
                .setTooltip(Component.translatable(option.tooltipKey()))
                .setRange(control.min(), control.max(), control.step())
                .setValueFormatter(v -> option.valueLabel(v))
                .setDefaultValue(option.integer().defaultValue())
                .setBinding(v -> { option.integer().set(v); CausticaSettings.afterChange(option); },
                        option.integer()::get)
                .setStorageHandler(SAVE)
                .setImpact(impact(option));
        OptionFlag[] flags = flags(option);
        if (flags.length > 0) {
            built.setFlags(flags);
        }
        return built;
    }

    /**
     * A float setting on an integer control: the widget carries {@code value * scale}, so the slider step
     * and the stored precision stay independent of each other. The stored value is re-read through the same
     * rounding the widget shows, so dragging and releasing cannot disagree.
     */
    private static IntegerOptionBuilder scaled(ConfigBuilder config, Option option, Control.Scaled control) {
        double scale = control.scale();
        IntegerOptionBuilder built = config.createIntegerOption(id(option.path()))
                .setName(Component.translatable(option.nameKey()))
                .setTooltip(Component.translatable(option.tooltipKey()))
                .setRange(control.min(), control.max(), control.step())
                .setValueFormatter(v -> option.valueLabel(v))
                .setDefaultValue((int) Math.round(option.floating().defaultValue() * scale))
                .setBinding(v -> { option.floating().set((float) (v / scale)); CausticaSettings.afterChange(option); },
                        () -> (int) Math.round(option.floating().get() * scale))
                .setStorageHandler(SAVE)
                .setImpact(impact(option));
        OptionFlag[] flags = flags(option);
        if (flags.length > 0) {
            built.setFlags(flags);
        }
        return built;
    }

    private static EnumOptionBuilder<ExposureMode> exposureMode(ConfigBuilder config, Option option) {
        return config.createEnumOption(id(option.path()), ExposureMode.class)
                .setName(Component.translatable(option.nameKey()))
                .setTooltip(Component.translatable(option.tooltipKey()))
                .setElementNameProvider(mode ->
                        Component.translatable(option.nameKey() + "." + mode.stored()))
                .setDefaultValue(ExposureMode.of(option.text().defaultValue()))
                .setBinding(mode -> { option.text().set(mode.stored()); CausticaSettings.afterChange(option); },
                        () -> ExposureMode.of(option.text().get()))
                .setStorageHandler(SAVE)
                .setImpact(impact(option));
    }

    private static EnumOptionBuilder<DlssQuality> dlssQuality(ConfigBuilder config, Option option) {
        return config.createEnumOption(id(option.path()), DlssQuality.class)
                .setName(Component.translatable(option.nameKey()))
                .setTooltip(Component.translatable(option.tooltipKey()))
                .setElementNameProvider(quality ->
                        Component.translatable(option.nameKey() + "." + quality.value()))
                .setDefaultValue(qualityOf(option.integer().defaultValue()))
                .setBinding(quality -> { option.integer().set(quality.value()); CausticaSettings.afterChange(option); },
                        () -> qualityOf(option.integer().get()))
                .setStorageHandler(SAVE)
                .setImpact(impact(option));
    }

    private static EnumOptionBuilder<HdrPeak> hdrPeak(ConfigBuilder config, Option option) {
        return config.createEnumOption(id(option.path()), HdrPeak.class)
                .setName(Component.translatable(option.nameKey()))
                .setTooltip(Component.translatable(option.tooltipKey()))
                .setElementNameProvider(peak -> Component.translatable(option.nameKey() + ".nits",
                        peak.value()))
                .setDefaultValue(peakOf(option.integer().defaultValue()))
                .setBinding(peak -> { option.integer().set(peak.value()); CausticaSettings.afterChange(option); },
                        () -> peakOf(option.integer().get()))
                .setStorageHandler(SAVE)
                .setImpact(impact(option));
    }

    private static DlssQuality qualityOf(int stored) {
        for (DlssQuality quality : DlssQuality.values()) {
            if (quality.value() == stored) {
                return quality;
            }
        }
        return DlssQuality.PERFORMANCE;
    }

    private static HdrPeak peakOf(int stored) {
        for (HdrPeak peak : HdrPeak.values()) {
            if (peak.value() == stored) {
                return peak;
            }
        }
        return HdrPeak.NITS_1000;
    }

}
