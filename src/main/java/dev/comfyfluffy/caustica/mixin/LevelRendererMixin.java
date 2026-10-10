package dev.comfyfluffy.caustica.mixin;

import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.resource.GraphicsResourceAllocator;
import dev.comfyfluffy.caustica.client.VanillaRenderController;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.joml.Matrix4fc;
import org.joml.Vector4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(LevelRenderer.class)
public abstract class LevelRendererMixin {
	@Shadow
	@Final
	private LevelRenderState levelRenderState;

	@Inject(method = "render", at = @At("HEAD"), cancellable = true)
	private void caustica$cancelVanillaWorld(
			GraphicsResourceAllocator resourceAllocator,
			DeltaTracker deltaTracker,
			boolean renderOutline,
			CameraRenderState cameraState,
			Matrix4fc modelViewMatrix,
			GpuBufferSlice terrainFog,
			Vector4f fogColor,
			boolean shouldRenderSky,
			CallbackInfo ci) {
		Runnable playerCompiledSectionCallback = this.levelRenderState.playerCompiledSectionCallback;
		boolean waitingForRtPlayerSection = false;
		if (VanillaRenderController.rtRuntimeWorkRequested() && playerCompiledSectionCallback != null) {
			if (RtTerrain.isSectionReady(cameraState.blockPos)) {
				playerCompiledSectionCallback.run();
				VanillaRenderController.INSTANCE.markRtPlayerSectionReady();
			} else {
				waitingForRtPlayerSection = true;
			}
		}

		if (!VanillaRenderController.INSTANCE.shouldCancelLevelRenderer(waitingForRtPlayerSection)) {
			return;
		}

		VanillaRenderController.INSTANCE.markWorldSkipped();
		ci.cancel();
	}

	/**
	 * The entity-outline pass blends {@code entityOutlineTarget} into the main target's colour and depth, and
	 * that target is only ever refreshed by the level pass. While the ray-traced composite replaces the world
	 * the level pass never runs, so the blend would paste a stale frame back over the traced image — and the
	 * traced path draws its own glow and block outlines instead.
	 */
	@Inject(method = "doEntityOutline", at = @At("HEAD"), cancellable = true)
	private void caustica$skipEntityOutline(CallbackInfo ci) {
		if (VanillaRenderController.INSTANCE.wasWorldSkippedThisFrame()) {
			ci.cancel();
		}
	}
}
