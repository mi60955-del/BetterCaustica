package dev.comfyfluffy.caustica.client;

import com.mojang.blaze3d.platform.InputConstants;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import dev.comfyfluffy.caustica.rt.RtStillRender;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;

/**
 * Renderer controls a player reaches for while looking at the world instead of from a screen.
 *
 * <p>The ray-tracing master switch is the comparison a player makes most often — a traced frame
 * against the vanilla rasterizer — and Video Settings cannot serve it: opening the screen replaces
 * the frame being judged, and returning to the world rebuilds the state the two frames should
 * share. The binding writes the same {@code caustica.rt.enabled} value the screen writes, so the
 * two controls cannot disagree, and the swap applies on the next frame.
 *
 * <p>Switching off leaves the RT runtime alive (context, worker pool, terrain residency) because a
 * hot switch is meant for repeated A/B; tearing residency down would put the world into the
 * waiting-for-terrain state on every flip. Releasing those resources belongs to client shutdown.
 *
 * <p>The still-render binding is the other half of the same idea: the picture worth keeping is the one in
 * front of the player, so averaging it cannot ask them to stop the world, leave the world, or rebuild it
 * through a settings screen. Counting frames, restarting on a camera move and saving live in
 * {@link RtStillRender}; the press only starts a run or drops the one in flight, and the ticks give the run
 * the moments it needs to resolve.
 */
public final class CausticaHotkeys {
	private static final String TOGGLE_RT_TRANSLATION = "key.caustica.toggle-rt";
	private static final String STILL_TRANSLATION = "key.caustica.render-still";
	private static final KeyMapping.Category CATEGORY =
			KeyMapping.Category.register(Identifier.fromNamespaceAndPath("caustica", "caustica"));
	private static KeyMapping toggleRayTracing;
	private static KeyMapping renderStill;

	private CausticaHotkeys() {
	}

	/** Registers the bindings; called once from the client initializer. */
	public static void register() {
		toggleRayTracing = KeyMappingHelper.registerKeyMapping(new KeyMapping(TOGGLE_RT_TRANSLATION,
				InputConstants.Type.KEYSYM, InputConstants.KEY_F9, CATEGORY));
		renderStill = KeyMappingHelper.registerKeyMapping(new KeyMapping(STILL_TRANSLATION,
				InputConstants.Type.KEYSYM, InputConstants.KEY_F10, CATEGORY));
		ClientTickEvents.END_CLIENT_TICK.register(client -> {
			// consumeClick() yields one event per press, so a held key cannot oscillate the renderer.
			while (toggleRayTracing.consumeClick()) {
				toggleRayTracing(client, !CausticaConfig.Rt.ENABLED.value());
			}
			while (renderStill.consumeClick()) {
				RtStillRender.INSTANCE.toggle();
			}
			// A still run finishes on a tick rather than on the frame that filled it: the last sample is in
			// a submission this tick is past, and a tick still runs once the world is gone.
			if (client.level == null) {
				RtStillRender.INSTANCE.cancel("the world was left");
			}
			RtStillRender.INSTANCE.service();
		});
	}

	private static void toggleRayTracing(Minecraft client, boolean enabled) {
		CausticaConfig.Rt.ENABLED.set(enabled);
		CausticaConfig.save();
		if (enabled) {
			// A latched failure records a frame that could not be replaced, not a hardware condition, so
			// an explicit re-enable retries RT rather than inheriting that frame's fallback.
			VanillaRenderController.requestRtRetry();
		}
		// The RT extensions are requested while the GPU device is created, so a session that started
		// with the switch off has no RT entry points to bring up now: the saved value applies on the
		// next launch instead.
		boolean needsRestart = enabled && !RtDeviceBringup.rtRequested();
		String message = needsRestart
				? "caustica.hotkey.rt.restart"
				: enabled ? "caustica.hotkey.rt.on" : "caustica.hotkey.rt.off";
		client.gui.hud.setOverlayMessage(Component.translatable(message), true);
		CausticaMod.LOGGER.info("Ray tracing {} by key binding{}", enabled ? "enabled" : "disabled",
				needsRestart ? " (needs a restart: the device was created without RT)" : "");
	}
}
