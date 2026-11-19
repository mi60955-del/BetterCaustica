package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.CausticaSettings.Page;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.options.OptionsSubScreen;
import net.minecraft.network.chat.Component;

/**
 * One page of {@link CausticaSettings} drawn with vanilla widgets, reached from the Ray Tracing hub or
 * from Mod Menu. Its counterpart when Sodium is installed is the tab of the same name in Sodium's own
 * screen; both take their pages, groups, labels and value spellings from the catalog, so neither can
 * offer something the other does not.
 *
 * <p>Extends {@link OptionsSubScreen} so the row, header/footer layout and Done button match the vanilla
 * option screens. Nothing here touches {@code options.txt}: the widgets write {@link CausticaConfig}, which
 * is persisted from {@link #removed()}.
 *
 * <p>The rows are kept so a write can re-evaluate their gates: turning a group's master switch on wakes
 * its children without leaving the page, which is the one thing a screen that builds its list once cannot
 * do on its own.
 */
public final class CausticaPageScreen extends OptionsSubScreen {
    private final Page page;
    private final List<RtVideoOptions.Row> rows = new ArrayList<>();

    public CausticaPageScreen(Screen lastScreen, Options options, Page page) {
        super(lastScreen, options, Component.translatable(page.titleKey()));
        this.page = page;
    }

    @Override
    protected void addOptions() {
        // The list is rebuilt from scratch on every init (a resize does it too), so the row references
        // from the previous build have to go with it.
        this.rows.clear();
        List<String> groups = CausticaSettings.groups(page);
        // A page with one group has no structure to announce: its title already says which thing this is,
        // and a header repeating it one line lower is the same words twice.
        boolean headed = groups.size() > 1;
        for (String group : groups) {
            if (headed) {
                this.list.addHeader(RtVideoOptions.groupTitle(group));
            }
            List<RtVideoOptions.Row> built = RtVideoOptions.groupRows(page, group, this::refreshGates);
            this.rows.addAll(built);
            this.list.addSmall(built.stream().map(RtVideoOptions.Row::widget).toList());
        }
        RtVideoOptions.refresh(this.rows);
    }

    private void refreshGates() {
        RtVideoOptions.refresh(this.rows);
    }

    @Override
    public void removed() {
        super.removed();
        // The parent screen also saves on exit, but Done here must not depend on the player then leaving
        // that screen too.
        CausticaConfig.save();
    }
}
