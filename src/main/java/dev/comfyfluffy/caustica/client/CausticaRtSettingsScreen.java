package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaConfig;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.options.OptionsSubScreen;
import net.minecraft.network.chat.Component;

/**
 * Caustica's settings hub, opened from Mod Menu and from the Ray Tracing section of Video Settings. It
 * carries one button per page, banded by {@link CausticaSettings.Section}, so the twelve subsystems are
 * three groups a player can scan instead of one column of buttons.
 *
 * <p>Reached through {@link CausticaModMenu} or the hub button rather than a mixin, so nothing here depends
 * on which screen the host mod replaces.
 */
public final class CausticaRtSettingsScreen extends OptionsSubScreen {
    private static final Component TITLE = Component.translatable("caustica.options.rt.hub");

    public CausticaRtSettingsScreen(Screen lastScreen, Options options) {
        super(lastScreen, options, TITLE);
    }

    @Override
    protected void addOptions() {
        RtVideoOptions.addRuntimeControls(this.list);
    }

    @Override
    public void removed() {
        super.removed();
        CausticaConfig.save();
    }
}
