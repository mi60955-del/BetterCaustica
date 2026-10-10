package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.CausticaSettings.Page;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.options.OptionsSubScreen;
import net.minecraft.network.chat.Component;

/**
 * One page of {@link CausticaSettings} drawn with vanilla widgets, reached from the Ray Tracing section of
 * the Video Settings screen or from Mod Menu. Its counterpart when Sodium is installed is the tab of the
 * same name in Sodium's own screen; both take their pages, groups, labels and value spellings from the
 * catalog, so neither can offer something the other does not.
 *
 * <p>Extends {@link OptionsSubScreen} so the row, header/footer layout and Done button match the vanilla
 * option screens. Nothing here touches {@code options.txt}: the widgets write {@link CausticaConfig}, which
 * is persisted from {@link #removed()}.
 */
public final class CausticaPageScreen extends OptionsSubScreen {
    private final Page page;

    public CausticaPageScreen(Screen lastScreen, Options options, Page page) {
        super(lastScreen, options, Component.translatable(page.titleKey()));
        this.page = page;
    }

    @Override
    protected void addOptions() {
        for (String group : CausticaSettings.groups(page)) {
            this.list.addHeader(RtVideoOptions.groupTitle(group));
            this.list.addSmall(RtVideoOptions.groupOptions(page, group));
        }
    }

    @Override
    public void removed() {
        super.removed();
        // The parent screen also saves on exit, but Done here must not depend on the player then leaving
        // that screen too.
        CausticaConfig.save();
    }
}
