package dev.comfyfluffy.caustica.client;

import com.mojang.serialization.Codec;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.CausticaSettings.Control;
import dev.comfyfluffy.caustica.client.CausticaSettings.Cycle;
import dev.comfyfluffy.caustica.client.CausticaSettings.Option;
import dev.comfyfluffy.caustica.client.CausticaSettings.Page;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.Minecraft;
import net.minecraft.client.OptionInstance;
import net.minecraft.client.Options;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.OptionsList;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

/**
 * Draws {@link CausticaSettings} with vanilla widgets, for the two hosts Sodium does not own: the Ray
 * Tracing section of the vanilla Video Settings screen (injected by {@code VideoSettingsScreenMixin}) and
 * the Mod Menu landing screen ({@link CausticaRtSettingsScreen}).
 *
 * <p>Each option binds straight to its {@code RuntimeSetting}, so this screen, Sodium's screen and
 * {@code config/bettercaustica.toml} stay one value rather than three that can disagree. The ranges, steps and
 * value spellings come from the catalog rather than from here, which is what keeps the two screens from
 * drifting into offering different things under the same name.
 *
 * <p>Vanilla has no floating-point option and no unit of its own, so a {@link Control.Scaled} is drawn as
 * an integer slider over the scaled range whose label is formatted by
 * {@link Option#valueLabel(int)} — the same call Sodium's formatter makes. Vanilla also has no restart
 * badge, so those options say it in their value instead. Vanilla's {@code IntRange} carries no step
 * either, so the step the catalog declares — which Sodium's slider enforces natively — is implemented
 * here by snapping the stored value and the initial value onto the grid.
 */
public final class RtVideoOptions {
    private RtVideoOptions() {
    }

    /**
     * The Ray Tracing section as one unit: the section header and the page buttons. Used where the RT
     * settings sit among other sections, i.e. the vanilla Video Settings screen.
     */
    public static void addRuntimeSection(OptionsList list) {
        list.addHeader(Component.translatable("caustica.options.rt.header"));
        addRuntimeControls(list);
    }

    /**
     * The page buttons without a header, for a host that titles its own screen. One button per page that
     * has anything to offer on this session's hardware, in catalog order, so the vanilla and Sodium
     * surfaces present the same list.
     */
    public static void addRuntimeControls(OptionsList list) {
        for (Page page : Page.values()) {
            if (!CausticaSettings.options(page).isEmpty()) {
                list.addBig(pageButton(page));
            }
        }
    }

    /** The title of one catalog group, for the header the page screen puts above it. */
    public static Component groupTitle(String group) {
        return Component.translatable(CausticaSettings.groupKey(group));
    }

    /** Every option of one group of one page, as vanilla widgets, in catalog order. */
    public static OptionInstance<?>[] groupOptions(Page page, String group) {
        List<OptionInstance<?>> built = new ArrayList<>();
        for (Option option : CausticaSettings.options(page)) {
            if (option.group().equals(group)) {
                built.add(widget(option));
            }
        }
        return built.toArray(OptionInstance<?>[]::new);
    }

    private static Button pageButton(Page page) {
        return Button.builder(Component.translatable(page.titleKey()), button -> open(page)).build();
    }

    /**
     * Opens a page over whichever screen launched it. The parent is read at click time rather than captured
     * at build time, so the row returns to the right place from both the vanilla screen and Mod Menu.
     */
    private static void open(Page page) {
        Minecraft mc = Minecraft.getInstance();
        Screen parent = mc.gui.screen();
        mc.gui.setScreen(new CausticaPageScreen(parent, mc.options, page));
    }

    private static OptionInstance<?> widget(Option option) {
        return switch (option.control()) {
            case Control.Toggle ignored -> toggle(option);
            case Control.Count count -> count(option, count);
            case Control.Scaled scaled -> scaled(option, scaled);
            case Control.Cycled cycled -> cycled(option, cycled.cycle());
        };
    }

    /** The value as the catalog spells it, with the restart notice vanilla has no badge for. */
    private static Component label(Option option, int widgetValue) {
        Component value = option.valueLabel(widgetValue);
        if (!option.restart()) {
            return value;
        }
        return value.copy().append(Component.literal(" "))
                .append(Component.translatable("caustica.options.rt.restartNeeded"));
    }

    private static OptionInstance<Boolean> toggle(Option option) {
        // An Enum range rather than createBoolean, so the value passes through the caption getter:
        // createBoolean's fixed on/off label has nowhere to carry the restart notice.
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                (caption, value) -> toggleValueLabel(value, option.restart()),
                new OptionInstance.Enum<>(List.of(Boolean.FALSE, Boolean.TRUE), Codec.BOOL),
                option.bool().value(),
                value -> {
                    option.bool().set(value);
                    CausticaSettings.afterChange(option);
                });
    }

    /**
     * The on/off word the widget shows after the caption — CycleButton prepends "caption: " itself —
     * with the restart notice appended where the option owes one.
     */
    private static Component toggleValueLabel(boolean value, boolean restart) {
        Component state = Component.translatable(value ? "options.on" : "options.off");
        if (!restart) {
            return state;
        }
        return state.copy().append(" ").append(Component.translatable("caustica.options.rt.restartNeeded"));
    }

    /** The nearest value on the catalog's step grid, clamped into the range. */
    private static int snap(int value, int min, int max, int step) {
        if (step <= 1) {
            return Math.clamp(value, min, max);
        }
        return Math.clamp(min + Math.round((value - min) / (float) step) * step, min, max);
    }

    private static OptionInstance<Integer> count(Option option, Control.Count control) {
        int min = control.min();
        int max = control.max();
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                (caption, value) -> Options.genericValueLabel(caption, label(option, value)),
                new OptionInstance.IntRange(min, max),
                snap(Math.clamp(option.integer().value(), min, max), min, max, control.step()),
                value -> {
                    option.integer().set(snap(value, min, max, control.step()));
                    CausticaSettings.afterChange(option);
                });
    }

    private static OptionInstance<Integer> scaled(Option option, Control.Scaled control) {
        double scale = control.scale();
        int min = control.min();
        int max = control.max();
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                (caption, value) -> Options.genericValueLabel(caption, label(option, value)),
                new OptionInstance.IntRange(min, max),
                snap(Math.clamp(Math.round(option.floating().value() * scale), min, max),
                        min, max, control.step()),
                value -> {
                    option.floating().set((float) (snap(value, min, max, control.step()) / scale));
                    CausticaSettings.afterChange(option);
                });
    }

    private static OptionInstance<?> cycled(Option option, Cycle cycle) {
        return switch (cycle) {
            case EXPOSURE_MODE -> exposureMode(option);
            case DLSS_QUALITY -> steppedList(option, CausticaConfig.Rt.DlssRr.QUALITY_STEPS, true);
            case HDR_PEAK -> steppedList(option, CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS, false);
        };
    }

    /**
     * Auto or manual. Stored as a word rather than an index because the TOML is a supported interface of
     * its own, and {@code mode = "auto"} has to keep meaning something to whoever edits it by hand.
     */
    private static OptionInstance<String> exposureMode(Option option) {
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                // CycleButton already prepends "caption: " itself (DisplayState.NAME_AND_VALUE), so this
                // must return only the value's text, not caption + value again.
                (caption, value) -> Component.translatable(option.nameKey() + "." + value),
                new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
                CausticaSettings.normalizeExposureMode(option.text().get()),
                value -> {
                    option.text().set(value);
                    CausticaSettings.afterChange(option);
                });
    }

    /**
     * An option stored as one of a fixed list of numbers rather than as an index into them.
     *
     * @param named true when the steps have translation keys of their own (DLSS's quality modes), false when
     *              the number is the name (an HDR mastering target, which is nits and nothing else)
     */
    private static OptionInstance<Integer> steppedList(Option option, List<Integer> steps, boolean named) {
        int initial = steps.contains(option.integer().value()) ? option.integer().value() : steps.get(0);
        int initialPosition = Math.max(steps.indexOf(initial), 0);
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                (caption, position) -> Options.genericValueLabel(caption, named
                        ? Component.translatable(option.nameKey() + "." + steps.get(position))
                        : Component.translatable("caustica.options.rt.hdrPeak.nits", steps.get(position))),
                new OptionInstance.IntRange(0, steps.size() - 1),
                initialPosition,
                position -> {
                    option.integer().set(steps.get(position));
                    CausticaSettings.afterChange(option);
                });
    }
}
