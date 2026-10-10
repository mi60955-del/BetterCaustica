package dev.comfyfluffy.caustica.client;

import com.terraformersmc.modmenu.api.ConfigScreenFactory;
import com.terraformersmc.modmenu.api.ModMenuApi;
import net.minecraft.client.Minecraft;

/**
 * Gives Mod Menu a Settings button for Caustica. Reached only through the {@code modmenu} Fabric entry
 * point, so Mod Menu is a compile-time target and not a dependency: with it uninstalled nothing here loads
 * and the button simply is not offered.
 */
public final class CausticaModMenu implements ModMenuApi {
    @Override
    public ConfigScreenFactory<?> getModConfigScreenFactory() {
        return parent -> new CausticaRtSettingsScreen(parent, Minecraft.getInstance().options);
    }
}
