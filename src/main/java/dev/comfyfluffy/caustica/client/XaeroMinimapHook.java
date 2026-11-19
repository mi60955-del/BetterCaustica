package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaMod;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import net.fabricmc.loader.api.FabricLoader;
import org.joml.Matrix4fc;

/**
 * Hands Xaero's Minimap the world model-view matrix of the pass the ray tracer replaces.
 *
 * <p>The minimap places its in-world waypoint markers by projecting each marker's camera-relative
 * offset through a model-view matrix it copies from the head of {@code LevelRenderer.render}. A
 * traced frame skips that pass, so the copy is never refreshed and keeps the value it had when the
 * pass last ran. Position offsets are recomputed from the camera every frame and enter the marker
 * transform separately, but the camera's rotation lives only in that matrix: with a stale copy the
 * markers keep the screen position of the old view and follow the player's movement alone, which
 * puts them nowhere near the block they mark as soon as the camera turns.
 *
 * <p>The minimap is optional, so the hook is resolved by name and dropped when the class or method
 * is absent; a missing hook must never fail a frame.
 */
public final class XaeroMinimapHook {
	private static final String MINIMAP_MOD_ID = "xaerominimap";
	private static final String CORE_CLASS = "xaero.common.core.XaeroMinimapCore";
	private static final String MODEL_VIEW_HOOK = "onWorldModelViewMatrix";

	private static MethodHandle modelViewHook;
	private static boolean resolved;

	private XaeroMinimapHook() {
	}

	/**
	 * Publishes the matrix the skipped level pass would have been given, exactly once per frame that
	 * skips it. Calling this on a frame that still renders the level would only repeat what the
	 * minimap already copies from the pass itself.
	 */
	public static void publishWorldModelView(Matrix4fc modelViewMatrix) {
		MethodHandle hook = modelViewHook();
		if (hook == null) {
			return;
		}

		try {
			hook.invokeExact(modelViewMatrix);
		} catch (Throwable t) {
			modelViewHook = null;
			CausticaMod.LOGGER.warn(
					"Xaero's Minimap rejected the world matrix; its waypoint markers will not follow the camera", t);
		}
	}

	private static MethodHandle modelViewHook() {
		if (resolved) {
			return modelViewHook;
		}

		resolved = true;
		if (!FabricLoader.getInstance().isModLoaded(MINIMAP_MOD_ID)) {
			return null;
		}

		try {
			Class<?> core = Class.forName(CORE_CLASS);
			modelViewHook = MethodHandles.publicLookup()
					.findStatic(core, MODEL_VIEW_HOOK, MethodType.methodType(void.class, Matrix4fc.class));
		} catch (Throwable t) {
			CausticaMod.LOGGER.warn(
					"Xaero's Minimap world matrix hook is missing; its waypoint markers will not follow the camera", t);
		}

		return modelViewHook;
	}
}
