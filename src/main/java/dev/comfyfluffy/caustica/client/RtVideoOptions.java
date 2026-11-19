package dev.comfyfluffy.caustica.client;

import com.mojang.serialization.Codec;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.CausticaSettings.Control;
import dev.comfyfluffy.caustica.client.CausticaSettings.Cycle;
import dev.comfyfluffy.caustica.client.CausticaSettings.Option;
import dev.comfyfluffy.caustica.client.CausticaSettings.Page;
import dev.comfyfluffy.caustica.client.CausticaSettings.Section;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.Minecraft;
import net.minecraft.client.OptionInstance;
import net.minecraft.client.Options;
import net.minecraft.client.gui.components.AbstractWidget;
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
 *
 * <p>An option whose {@code Gate} names a parent is built like any other and then deactivated: vanilla
 * draws an inactive widget greyed, which is the same thing Sodium's {@code setEnabledProvider} does. The
 * rows are handed back to the screen so it can re-evaluate them when the switch that woke them moves.
 */
public final class RtVideoOptions {
    private RtVideoOptions() {
    }

    /** One catalog entry with the widget that draws it, so the screen can re-grey it after any write. */
    public record Row(Option option, AbstractWidget widget) {
    }

    /**
     * The Ray Tracing section as one unit: the section header, the master switch and the way into the
     * hub. Used where the RT settings sit among other sections, i.e. the vanilla Video Settings screen.
     */
    public static void addRuntimeSection(OptionsList list) {
        list.addHeader(Component.translatable("caustica.options.rt.header"));
        Option enabled = CausticaSettings.byPath("enabled");
        if (enabled != null) {
            list.addBig(widget(enabled, () -> { }));
        }
        list.addBig(Button.builder(Component.translatable("caustica.options.rt.openHub"), button -> openHub())
                .build());
    }

    /**
     * The page buttons, banded by {@link Section}, for the screen that titles itself. One button per page
     * that has anything to offer on this session's hardware, in catalog order, so the vanilla and Sodium
     * surfaces present the same list.
     */
    public static void addRuntimeControls(OptionsList list) {
        for (Section section : Section.values()) {
            boolean titled = false;
            for (Page page : Page.values()) {
                if (page.section() != section || CausticaSettings.options(page).isEmpty()) {
                    continue;
                }
                if (!titled) {
                    // A band with no pages on it (every one of them gated off this machine) would be a
                    // heading over nothing, so the header follows the first page that survives.
                    list.addHeader(Component.translatable(section.titleKey()));
                    titled = true;
                }
                list.addBig(pageButton(page));
            }
        }
    }

    /** The title of one catalog group, for the header the page screen puts above it. */
    public static Component groupTitle(String group) {
        return Component.translatable(CausticaSettings.groupKey(group));
    }

    /**
     * Every option of one group of one page, as vanilla widgets, in catalog order. {@code onChanged} runs
     * after any write so the caller can re-evaluate the rows whose parent just moved.
     */
    public static List<Row> groupRows(Page page, String group, Runnable onChanged) {
        List<Row> rows = new ArrayList<>();
        for (Option option : CausticaSettings.options(page)) {
            if (option.group().equals(group)) {
                rows.add(new Row(option, widget(option, onChanged)));
            }
        }
        return rows;
    }

    /**
     * The controls of one group before they become widgets. Exposed so the catalog can be built without a
     * client: {@link #groupRows} needs one to size the buttons, this does not.
     */
    static List<OptionInstance<?>> instances(Page page, String group, Runnable onChanged) {
        List<OptionInstance<?>> built = new ArrayList<>();
        for (Option option : CausticaSettings.options(page)) {
            if (option.group().equals(group)) {
                built.add(instance(option, onChanged));
            }
        }
        return built;
    }

    /** Re-evaluates every row's parent gate. Cheap: one supplier read per row. */
    public static void refresh(List<Row> rows) {
        for (Row row : rows) {
            row.widget().active = row.option().enabledNow();
        }
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

    private static void openHub() {
        Minecraft mc = Minecraft.getInstance();
        mc.gui.setScreen(new CausticaRtSettingsScreen(mc.gui.screen(), mc.options));
    }

    private static AbstractWidget widget(Option option, Runnable onChanged) {
        // The default 150-wide construction is the column width OptionsList lays a small row out at; a
        // big row widens it to 310 itself.
        AbstractWidget built = instance(option, onChanged).createButton(Minecraft.getInstance().options);
        built.active = option.enabledNow();
        return built;
    }

    private static OptionInstance<?> instance(Option option, Runnable onChanged) {
        return switch (option.control()) {
            case Control.Toggle ignored -> toggle(option, onChanged);
            case Control.Count count -> count(option, count, onChanged);
            case Control.Scaled scaled -> scaled(option, scaled, onChanged);
            case Control.Cycled cycled -> cycled(option, cycled.cycle(), onChanged);
        };
    }

    /** The value as the catalog spells it, with the restart notice vanilla has no badge for. */
    private static Component label(Option option, int widgetValue) {
        return withRestartNotice(option, option.valueLabel(widgetValue));
    }

    /**
     * The restart notice for a value the widget already formatted. Both the catalog-formatted controls and
     * the cycled ones, whose value text is built from a translation key rather than a number, need it.
     */
    private static Component withRestartNotice(Option option, Component value) {
        if (!option.restart()) {
            return value;
        }
        return value.copy().append(" ").append(Component.translatable("caustica.options.rt.restartNeeded"));
    }

    private static OptionInstance<Boolean> toggle(Option option, Runnable onChanged) {
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
                    onChanged.run();
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

    private static OptionInstance<Integer> count(Option option, Control.Count control, Runnable onChanged) {
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
                    onChanged.run();
                });
    }

    private static OptionInstance<Integer> scaled(Option option, Control.Scaled control, Runnable onChanged) {
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
                    onChanged.run();
                });
    }

    private static OptionInstance<?> cycled(Option option, Cycle cycle, Runnable onChanged) {
        return switch (cycle) {
            case EXPOSURE_MODE -> exposureMode(option, onChanged);
            case DLSS_QUALITY -> steppedList(option, CausticaConfig.Rt.DlssRr.QUALITY_STEPS,
                    value -> Component.translatable(option.nameKey() + "." + value), onChanged);
            case HDR_PEAK -> steppedList(option, CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS,
                    value -> Component.translatable(option.nameKey() + ".nits", value), onChanged);
            case FRAME_GENERATION_BACKEND -> frameGenerationBackend(option, onChanged);
            case DOF_MODE -> steppedList(option, CausticaConfig.Rt.DepthOfField.MODES,
                    value -> Component.translatable(option.nameKey() + "." + value), onChanged);
            case PARALLAX_STYLE -> steppedList(option, CausticaConfig.Rt.Materials.PARALLAX_STYLES,
                    value -> Component.translatable(option.nameKey() + "." + value), onChanged);
        };
    }

    /**
     * One of {@link CausticaConfig.Rt.Fg#BACKENDS}, stored as its TOML word. An unrecognised stored value
     * shows as the default rather than leaving the widget without a selected entry.
     */
    private static OptionInstance<String> frameGenerationBackend(Option option, Runnable onChanged) {
        List<String> backends = CausticaConfig.Rt.Fg.BACKENDS;
        String stored = option.text().get();
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                // An OptionInstance.Enum cycles with DisplayState.NAME_AND_VALUE, so the caption is
                // already prepended and this must return only the value's text.
                (caption, value) -> withRestartNotice(option,
                        Component.translatable(option.nameKey() + "." + value)),
                new OptionInstance.Enum<>(backends, Codec.STRING),
                backends.contains(stored) ? stored : CausticaConfig.Rt.Fg.DLSS,
                value -> {
                    option.text().set(value);
                    CausticaSettings.afterChange(option);
                    onChanged.run();
                });
    }

    /**
     * Auto or manual. Stored as a word rather than an index because the TOML is a supported interface of
     * its own, and {@code mode = "auto"} has to keep meaning something to whoever edits it by hand.
     */
    private static OptionInstance<String> exposureMode(Option option, Runnable onChanged) {
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
                    onChanged.run();
                });
    }

    /**
     * An option stored as one of a fixed list of numbers rather than as an index into them.
     *
     * @param valueLabel how one of those numbers is written down; the DLSS and mode lists name their own
     *                   values, an HDR mastering target is the number and nothing else
     */
    private static OptionInstance<Integer> steppedList(Option option, List<Integer> steps,
                                                       java.util.function.IntFunction<Component> valueLabel,
                                                       Runnable onChanged) {
        int initial = steps.contains(option.integer().value()) ? option.integer().value() : steps.get(0);
        int initialPosition = Math.max(steps.indexOf(initial), 0);
        return new OptionInstance<>(option.nameKey(),
                OptionInstance.cachedConstantTooltip(Component.translatable(option.tooltipKey())),
                (caption, position) -> Options.genericValueLabel(caption, valueLabel.apply(steps.get(position))),
                new OptionInstance.IntRange(0, steps.size() - 1),
                initialPosition,
                position -> {
                    option.integer().set(steps.get(position));
                    CausticaSettings.afterChange(option);
                    onChanged.run();
                });
    }
}
