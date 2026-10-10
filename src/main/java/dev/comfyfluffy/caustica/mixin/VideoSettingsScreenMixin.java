package dev.comfyfluffy.caustica.mixin;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.client.RtVideoOptions;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.OptionInstance;
import net.minecraft.client.Options;
import net.minecraft.client.gui.components.OptionsList;
import net.minecraft.client.gui.screens.options.VideoSettingsScreen;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Surfaces the runtime-tunable RT settings inside the vanilla Video Settings screen when the RT renderer is
 * enabled. Two changes, both gated on {@link CausticaConfig.Rt#ENABLED}:
 *
 * <ul>
 *   <li>The Quality section drops the vanilla options the path tracer supersedes (Ambient Occlusion and
 *       Entity Shadows are computed by RT global illumination / RT shadows).</li>
 *   <li>A "Ray Tracing" section adds the controls from {@link RtVideoOptions#addRuntimeSection}, which is
 *       the same section the Mod Menu landing screen shows.</li>
 * </ul>
 *
 * When RT is disabled the screen is left exactly as vanilla built it. When Sodium is installed it replaces
 * this screen entirely and nothing here runs; Caustica's Sodium integration publishes the same settings
 * through Sodium's own config API instead.
 */
@Mixin(VideoSettingsScreen.class)
public abstract class VideoSettingsScreenMixin {
    @Shadow
    private static OptionInstance<?>[] qualityOptions(Options options) {
        throw new AssertionError("mixin stub");
    }

    @Redirect(
        method = "addOptions",
        at = @At(
            value = "INVOKE",
            target = "Lnet/minecraft/client/gui/screens/options/VideoSettingsScreen;qualityOptions(Lnet/minecraft/client/Options;)[Lnet/minecraft/client/OptionInstance;"))
    private OptionInstance<?>[] caustica$filterQualityOptions(Options options) {
        OptionInstance<?>[] base = qualityOptions(options);
        if (!CausticaConfig.Rt.ENABLED.value()) {
            return base;
        }
        List<OptionInstance<?>> kept = new ArrayList<>(base.length);
        for (OptionInstance<?> option : base) {
            // Path-traced GI + RT shadows make these vanilla raster controls inert under RT.
            if (option == options.ambientOcclusion() || option == options.entityShadows()) {
                continue;
            }
            kept.add(option);
        }
        return kept.toArray(OptionInstance<?>[]::new);
    }

    @Inject(method = "addOptions", at = @At("HEAD"))
    private void caustica$addRtOptions(CallbackInfo ci) {
        if (!CausticaConfig.Rt.ENABLED.value()) {
            return;
        }
        OptionsList list = ((OptionsSubScreenAccessor) (Object) this).getList();
        if (list == null) {
            return;
        }
        RtVideoOptions.addRuntimeSection(list);
    }

    @Inject(method = "removed", at = @At("TAIL"))
    private void caustica$saveConfig(CallbackInfo ci) {
        // Persist any RT settings the player changed in this screen to the TOML config. Unconditional:
        // the file is small and vanilla itself closes this screen far less often than it rewrites its
        // own options.txt, so a dirty flag would not buy anything measurable.
        CausticaConfig.save();
    }
}
