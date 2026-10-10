package dev.comfyfluffy.caustica.mixin;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.rt.RtReflex;
import dev.comfyfluffy.caustica.rt.RtUiOverlay;

import net.minecraft.client.Minecraft;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * The Reflex per-frame sleep call must run at the very start of the frame, before input
 * sampling/simulation. {@link Minecraft#runTick} is Minecraft's per-loop-iteration entry point (called once
 * per {@code while (running)} iteration in {@link Minecraft#run()}, right after
 * {@code RenderSystem.pollEvents()}), so its HEAD is the earliest hookable point in this codebase for that
 * purpose — the alternative (hooking inside {@code run()}'s loop body directly) isn't a clean Mixin target
 * since it's inline in a loop, not a call to a named method.
 *
 * <p>SIMULATION_START/END bracket the tick/extract work that happens before rendering (from here through
 * just before {@code renderFrame} is called); RENDERSUBMIT/PRESENT markers are set from
 * {@code GameRendererMixin}/{@code VulkanGpuSurfaceMixin} respectively. No-ops entirely unless Reflex is
 * enabled AND {@link RtReflex#applySleepMode} has already succeeded for the current swapchain.
 */
@Mixin(Minecraft.class)
public abstract class MinecraftMixin {
	@Inject(method = "close", at = @At("HEAD"))
	private void caustica$destroyUiOverlayBeforeRendererShutdown(CallbackInfo ci) {
		RtUiOverlay.destroy();
		// The render loop has stopped by close(); this is the first point where every native RT
		// resource can be destroyed without racing frames that still allocate on the shared device
		// (FastQuit's save screen, title-screen font atlases). Blaze3D shuts the Vulkan device down
		// later inside close(), so the RT-owned objects still have a live parent here.
		dev.comfyfluffy.caustica.client.CausticaClient.destroyRtNative();
	}

	@Inject(method = "runTick", at = @At("HEAD"))
	private void caustica$reflexSleepAndSimStart(boolean advanceGameTime, CallbackInfo ci) {
		VulkanDevice device = caustica$reflexDevice();
		long swapchain = RtReflex.INSTANCE.appliedSwapchain();
		if (device == null || swapchain == 0L) {
			return;
		}
		RtReflex.INSTANCE.sleep(device.vkDevice(), swapchain);
		RtReflex.INSTANCE.marker(device.vkDevice(), swapchain, RtReflex.MARKER_SIMULATION_START,
				RtReflex.INSTANCE.currentSimFrameId());
	}

	@Inject(method = "runTick",
			at = @At(value = "INVOKE", target = "Lnet/minecraft/client/Minecraft;renderFrame(Z)V"))
	private void caustica$reflexSimEnd(boolean advanceGameTime, CallbackInfo ci) {
		VulkanDevice device = caustica$reflexDevice();
		long swapchain = RtReflex.INSTANCE.appliedSwapchain();
		if (device == null || swapchain == 0L) {
			return;
		}
		RtReflex.INSTANCE.marker(device.vkDevice(), swapchain, RtReflex.MARKER_SIMULATION_END,
				RtReflex.INSTANCE.currentSimFrameId());
	}

	private static VulkanDevice caustica$reflexDevice() {
		if (!RtReflex.enabled()) {
			return null;
		}
		return ((GpuDeviceAccessor) RenderSystem.getDevice()).caustica$getBackend() instanceof VulkanDevice device
				? device : null;
	}
}
