package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import dev.comfyfluffy.caustica.rt.RtComposite;
import dev.comfyfluffy.caustica.rt.RtFrameStats;
import dev.comfyfluffy.caustica.rt.RtUiOverlay;
import dev.comfyfluffy.caustica.rt.entity.RtEntities;
import dev.comfyfluffy.caustica.rt.entity.RtEntityTextures;
import dev.comfyfluffy.caustica.rt.material.RtBlockMaterials;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;
import dev.comfyfluffy.caustica.rt.terrain.RtWorkerPool;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.InvalidateRenderStateCallback;

public final class CausticaClient implements ClientModInitializer {
	private static boolean rtInitDone = false;
	// Guards the native teardown so it runs exactly once, at Minecraft.close(). CLIENT_STOPPING can no
	// longer reset rtInitDone before this point, so the guard cannot live on rtInitDone itself.
	private static boolean nativeDestroyed = false;

	@Override
	public void onInitializeClient() {
		CausticaMod.LOGGER.info("Caustica client initialized");

		// Class-init runs DebugScreenEntries.register(...) via its ID field; touching the class here
		// makes the entry discoverable in F3's entry list. Off by default -- the player opts in the
		// same way as any other optional vanilla entry (e.g. GPU utilization).
		@SuppressWarnings("unused")
		Object registerExposureDebugEntry = RtExposureDebugEntry.ID;

		// The GpuDevice exists well before the first tick, so a one-shot at tick start
		// runs on the render thread with the device idle between frames.
		ClientTickEvents.START_CLIENT_TICK.register(client -> {
			if (!VanillaRenderController.rtRuntimeWorkRequested()) {
				if (rtInitDone) {
					shutdownRt();
				}
				return;
			}

			// Bring up the RT device/context once; terrain residency + the composite follow below.
			if (!rtInitDone && RtDeviceBringup.rtRequested()) {
				RtContext ctx = RtContext.get();
				if (ctx != null) {
					rtInitDone = true;
				}
			}

			// Once RT is up, keep section residency synced to vanilla's loaded chunks around
			// the player — builds newly-in-range sections, frees out-of-range ones, per tick.
			if (rtInitDone) {
				RtContext ctx = RtContext.currentOrNull();
				if (ctx != null) {
					RtFrameStats.FRAME.beginIfInactive();
					// Bring the world pipeline + LabPBR atlases up before terrain tessellates, so per-prim
					// material flags resolve from the first section (PBR on join, no re-extract). No-op
					// until we're in a world with the block atlas loaded, or once already created.
					RtComposite.INSTANCE.ensureResourcesReady(ctx);
					RtTerrain.update(ctx);
					// Log each backend's availability once when frame generation is enabled (capability
					// query only; the present-loop integration that consumes it is built separately). Only
					// the selected backend is probed, because probing the other would load a vendor runtime
					// that this session has no use for.
					if (dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.enabled()) {
						if (dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg.selected()) {
							dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg.INSTANCE.probeAvailabilityOnce();
						} else {
							dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.INSTANCE.probeAvailabilityOnce();
						}
						// A backend switch in the settings leaves the previous backend's GPU state
						// alive; release it here so the switch does not stack two contexts or
						// features worth of device memory.
						dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg.INSTANCE.destroyUnlessSelected();
						dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.INSTANCE.destroyUnlessSelected();
					}
				}
			}
		});

		// Vanilla's full render-state invalidation (LevelExtractor.allChanged(): dimension change via
		// setLevel, render-distance change, F3+A) — drop RT terrain residency so it rebuilds for the new
		// world. Fixes stale geometry persisting across an End→Overworld switch (coords alone aren't
		// world-unique). Resource reloads do NOT fire this; that path is handled separately.
		InvalidateRenderStateCallback.EVENT.register(() -> {
			RtTerrain.requestFullClear();
			RtComposite.INSTANCE.resetExposureHistory();
			RtComposite.INSTANCE.resetFailureLatch(); // F3+A doubles as manual RT recovery after a latched failure
		});

		ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
			shutdownRt();
		});

		// The NGX natives extraction and shim dlopen are file IO that would otherwise happen inside
		// the first render-frame acquire when DLSS is wanted; doing it here keeps that frame smooth.
		// Never initializes NGX, and does nothing when DLSS is not wanted or the platform has no bundle.
		if (dev.comfyfluffy.caustica.rt.pipeline.RtDlssRr.enabled()
				|| dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.enabled()) {
			dev.comfyfluffy.caustica.ngx.NgxRuntime.INSTANCE.prewarm();
		}
	}

	private static void shutdownRt() {
		WorldRenderScaler.INSTANCE.destroy();
		RtUiOverlay.destroy(); // GUI redirect is not gated by rtInitDone; always release its TextureTarget
		if (!rtInitDone) {
			RtWorkerPool.INSTANCE.shutdown();
			return;
		}
		// CLIENT_STOPPING fires while the client is still rendering: FastQuit keeps presenting its save
		// screen and the title screen keeps baking font glyph atlases after this callback, and both
		// allocate images on the shared Vulkan device through Blaze3D's VMA. Any native teardown here
		// (RT pipelines, the RT VMA allocator, FFX/NGX release) races those live allocations, so the
		// full destroy chain runs at Minecraft.close() instead, once the render loop has stopped. The
		// process is exiting either way; only the Java-side workers need an orderly stop.
		RtWorkerPool.INSTANCE.shutdown();
		RtEntities.INSTANCE.shutdown();
		rtInitDone = false;
	}

	/**
	 * Destroy every native RT resource. Called from Minecraft.close(), after the render loop has
	 * stopped and no further frames can allocate on the device, but before Blaze3D shuts the device
	 * down, so the RT-owned Vulkan objects still have a live parent to be destroyed against.
	 */
	public static void destroyRtNative() {
		if (nativeDestroyed) {
			return;
		}
		nativeDestroyed = true;
		RtContext ctx = RtContext.currentOrNull();
		if (ctx == null) {
			return; // RT never came up in this session
		}
		try {
			// Let the terrain epoch cancel queued work and release every active-task token before
			// shutdownNow(): discarded worker Runnables cannot deliver their terminal callbacks.
			RtTerrain.shutdown(ctx);
		} catch (Throwable t) {
			CausticaMod.LOGGER.warn("RT terrain shutdown failed during exit", t);
		}
		RtWorkerPool.INSTANCE.shutdown();
		RtEntities.INSTANCE.shutdown();
		RtComposite.INSTANCE.destroy();
		RtEntityTextures.INSTANCE.reset();
		RtBlockMaterials.INSTANCE.destroy();
		dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.INSTANCE.destroy();
		dev.comfyfluffy.caustica.rt.pipeline.RtFsrFg.INSTANCE.destroy();
		dev.comfyfluffy.caustica.rt.RtFramePresenter.INSTANCE.destroy(ctx.device());
		dev.comfyfluffy.caustica.rt.RtReflex.INSTANCE.destroy(ctx.device().vkDevice());
		// Shut NGX down once, after every feature (RR + FG) has been released above.
		dev.comfyfluffy.caustica.ngx.NgxRuntime.INSTANCE.shutdown();
		// Same for FFX: only after the FSR context has been released above.
		dev.comfyfluffy.caustica.ffx.FfxRuntime.INSTANCE.shutdown();
		ctx.destroy();
	}
}
