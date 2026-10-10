package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaConfig;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.options.OptionsSubScreen;
import net.minecraft.network.chat.Component;

/**
 * Caustica's settings landing screen, opened from Mod Menu. It carries the same Ray Tracing controls the
 * vanilla Video Settings screen shows, so a player who never opens the vanilla menus still has a route to
 * them.
 *
 * <p>Reached through {@link CausticaModMenu} rather than a mixin, so nothing here depends on which screen
 * the host mod replaces.
 */
public final class CausticaRtSettingsScreen extends OptionsSubScreen {
    private static final Component TITLE = Component.translatable("caustica.options.rt.header");

    public CausticaRtSettingsScreen(Screen lastScreen, Options options) {
        super(lastScreen, options, TITLE);
    }

    @Override
    protected void addOptions() {
        // No section header: the screen title already names it.
        RtVideoOptions.addRuntimeControls(this.list);
    }

    @Override
    public void removed() {
        super.removed();
        CausticaConfig.save();
    }
}
