package dev.comfyfluffy.caustica.client;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.CausticaSettings.Control;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import net.caffeinemc.mods.sodium.client.config.ConfigManager;
import net.caffeinemc.mods.sodium.client.config.builder.ConfigBuilderImpl;
import net.caffeinemc.mods.sodium.client.config.structure.Option;
import net.caffeinemc.mods.sodium.client.config.structure.OptionGroup;
import net.caffeinemc.mods.sodium.client.config.structure.OptionPage;
import net.caffeinemc.mods.sodium.client.config.structure.Page;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.contents.TranslatableContents;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards {@link CausticaSettings} and the two screens that draw it against the ways it can be wrong that
 * only surface in the player's game.
 *
 * <p>Sodium validates each option as it is added to its group and aborts resource reload when a required
 * field is missing, so one incomplete option takes the whole client down. Building the tree here runs that
 * same validation without launching.
 *
 * <p>Labels are checked against the shipped locale because both hosts render whatever component they are
 * handed, so an unregistered key appears in the menu verbatim. The catalog is checked directly rather than
 * only through Sodium's tree, because an option the current hardware hides is never added to that tree and
 * would otherwise go unexamined.
 */
class CausticaSodiumConfigTest {

    @Test
    void everyLabelHasAnEnglishTranslation() {
        Map<String, String> locale = englishLocale();
        Set<String> headings = new LinkedHashSet<>();
        Set<String> controls = new LinkedHashSet<>();
        collectKeys(headings, controls);
        List<String> missing = new ArrayList<>();
        for (String key : headings) {
            if (!locale.containsKey(key)) {
                missing.add(key);
            }
        }
        for (String key : controls) {
            if (!locale.containsKey(key)) {
                missing.add(key);
            }
            // A tooltip cannot be read off a built option outside Sodium's own config, but every control
            // names its tooltip after itself, so the control's own key stands in for it.
            if (!locale.containsKey(key + ".tooltip")) {
                missing.add(key + ".tooltip");
            }
        }
        assertTrue(missing.isEmpty(), "missing from en_us.json: " + missing);
    }

    /**
     * The catalog is drawn by two hosts, and an option whose control does not match its setting type would
     * only show up as a cast failure the first time a player opens the screen. The widget range is checked
     * at the same time because a default outside it is silently unreadable: the slider sits at an end and
     * shows a value the setting does not hold.
     */
    @Test
    void everyOptionBindsItsControlToASettingThatFits() {
        List<String> problems = new ArrayList<>();
        for (CausticaSettings.Option option : CausticaSettings.ALL) {
            switch (option.control()) {
                case Control.Toggle ignored -> {
                    if (!(option.setting() instanceof CausticaConfig.BooleanSetting)) {
                        problems.add(option.path() + " is a toggle on " + option.setting().getClass());
                    }
                }
                case Control.Count count -> {
                    if (!(option.setting() instanceof CausticaConfig.IntSetting integer)) {
                        problems.add(option.path() + " is a count on " + option.setting().getClass());
                    } else if (!fits(count.min(), count.max(), integer.defaultValue())) {
                        problems.add(option.path() + " defaults to " + integer.defaultValue()
                                + ", outside " + count.min() + ".." + count.max());
                    }
                }
                case Control.Scaled scaled -> {
                    if (!(option.setting() instanceof CausticaConfig.FloatSetting floating)) {
                        problems.add(option.path() + " is scaled on " + option.setting().getClass());
                    } else if (!fits(scaled.min(), scaled.max(),
                            Math.round(floating.defaultValue() * scaled.scale()))) {
                        problems.add(option.path() + " defaults to " + floating.defaultValue()
                                + ", outside the widget range " + scaled.min() + ".." + scaled.max());
                    }
                }
                case Control.Cycled cycled -> {
                    if (!cycles(cycled.cycle(), option.setting())) {
                        problems.add(option.path() + " cycles " + option.setting().getClass());
                    }
                }
            }
        }
        assertTrue(problems.isEmpty(), "bad catalog entries: " + problems);
    }

    /** Options the vanilla screen can only describe in words, since it has no restart badge. */
    @Test
    void restartNoticeIsTranslated() {
        assertTrue(englishLocale().containsKey("caustica.options.rt.restartNeeded"),
                "RtVideoOptions appends this key to every restart-only option");
    }

    /**
     * The value labels of a cycled control come from per-value keys the option name does not contain, so
     * the control-key walk above never sees them. A missing one renders as the raw key in both hosts.
     */
    @Test
    void everyCycledValueHasAnEnglishTranslation() {
        Map<String, String> locale = englishLocale();
        List<String> keys = new ArrayList<>();
        for (CausticaSettings.Option option : CausticaSettings.ALL) {
            if (!(option.control() instanceof Control.Cycled cycled)) {
                continue;
            }
            keys.addAll(cycledValueKeys(option, cycled.cycle()));
        }
        List<String> missing = keys.stream().filter(key -> !locale.containsKey(key)).toList();
        assertTrue(missing.isEmpty(), "missing from en_us.json: " + missing);
    }

    private static List<String> cycledValueKeys(CausticaSettings.Option option, CausticaSettings.Cycle cycle) {
        return switch (cycle) {
            case EXPOSURE_MODE -> List.of(option.nameKey() + ".auto", option.nameKey() + ".manual");
            case DLSS_QUALITY -> CausticaConfig.Rt.DlssRr.QUALITY_STEPS.stream()
                    .map(step -> option.nameKey() + "." + step).toList();
            case HDR_PEAK -> List.of(option.nameKey() + ".nits");
        };
    }

    /**
     * The enum hosts cycle by ordinal into the step lists, so a list or enum reordered on one side only
     * would silently attach the wrong value to each name.
     */
    @Test
    void cycledEnumOrderMatchesTheStepLists() {
        CausticaSodiumConfig.DlssQuality[] qualities = CausticaSodiumConfig.DlssQuality.values();
        assertEquals(CausticaConfig.Rt.DlssRr.QUALITY_STEPS.size(), qualities.length,
                "DlssQuality and QUALITY_STEPS disagree in length");
        for (int i = 0; i < qualities.length; i++) {
            assertEquals(CausticaConfig.Rt.DlssRr.QUALITY_STEPS.get(i), qualities[i].value(),
                    "DlssQuality ordinal " + i);
        }
        CausticaSodiumConfig.HdrPeak[] peaks = CausticaSodiumConfig.HdrPeak.values();
        assertEquals(CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS.size(), peaks.length,
                "HdrPeak and PEAK_NITS_STEPS disagree in length");
        for (int i = 0; i < peaks.length; i++) {
            assertEquals(CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS.get(i), peaks[i].value(),
                    "HdrPeak ordinal " + i);
        }
    }

    /**
     * The second host has to be able to draw the same list. Neither screen can be opened from a test, so
     * this builds every vanilla widget the catalog describes: a control whose type does not match its
     * setting, or a range that cannot hold the current value, fails here instead of the first time a player
     * without Sodium opens Video Settings.
     */
    @Test
    void vanillaWidgetsCoverEveryCatalogOption() {
        for (CausticaSettings.Page page : CausticaSettings.Page.values()) {
            int built = 0;
            for (String group : CausticaSettings.groups(page)) {
                int expected = CausticaSettings.options(page).stream()
                        .filter(option -> option.group().equals(group))
                        .mapToInt(option -> 1)
                        .sum();
                assertEquals(expected, RtVideoOptions.groupOptions(page, group).length,
                        page + "/" + group + " built a different number of rows than the catalog lists");
                built += expected;
            }
            assertEquals(CausticaSettings.options(page).size(), built,
                    page + " dropped options between the catalog and the screen");
        }
    }

    private static boolean cycles(CausticaSettings.Cycle cycle, Object setting) {
        return switch (cycle) {
            case EXPOSURE_MODE -> setting instanceof CausticaConfig.StringSetting;
            case DLSS_QUALITY, HDR_PEAK -> setting instanceof CausticaConfig.IntSetting;
        };
    }

    private static boolean fits(int min, int max, double value) {
        return value >= min && value <= max;
    }

    /** Builds the tree the way Sodium does at startup; the walk itself is the assertion. */
    @Test
    void sodiumAcceptsEveryRegisteredOption() {
        List<Option> options = new ArrayList<>();
        for (Page page : buildPages()) {
            for (OptionGroup group : ((OptionPage) page).groups()) {
                options.addAll(group.options());
            }
        }
        assertTrue(options.size() > 50, "expected the full option set, got " + options.size());
    }

    private static List<Page> buildPages() {
        ConfigBuilderImpl config = new ConfigBuilderImpl(
                modId -> new ConfigManager.ModMetadata(modId, "test"), "bettercaustica");
        new CausticaSodiumConfig().registerConfigLate(config);
        List<Page> pages = new ArrayList<>();
        config.build().forEach(mod -> pages.addAll(mod.pages()));
        return pages;
    }

    private static void collectKeys(Set<String> headings, Set<String> controls) {
        for (Page page : buildPages()) {
            OptionPage optionPage = assertInstanceOf(OptionPage.class, page);
            collect(optionPage.name(), headings);
            for (OptionGroup group : optionPage.groups()) {
                collect(group.name(), headings);
                for (Option option : group.options()) {
                    collect(option.getName(), controls);
                }
            }
        }
    }

    private static void collect(Component component, Set<String> keys) {
        if (component.getContents() instanceof TranslatableContents translatable) {
            keys.add(translatable.getKey());
        }
    }

    private static Map<String, String> englishLocale() {
        try (InputStream resource = Objects.requireNonNull(
                CausticaSodiumConfigTest.class.getResourceAsStream("/assets/bettercaustica/lang/en_us.json"))) {
            return new Gson().fromJson(new InputStreamReader(resource, StandardCharsets.UTF_8),
                    new TypeToken<Map<String, String>>() { }.getType());
        } catch (Exception e) {
            throw new AssertionError("Could not read en_us.json", e);
        }
    }
}
