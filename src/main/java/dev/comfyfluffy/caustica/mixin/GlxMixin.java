package dev.comfyfluffy.caustica.mixin;

import com.mojang.blaze3d.platform.GLX;
import dev.comfyfluffy.caustica.CausticaMod;
import java.util.Locale;
import java.util.function.LongSupplier;
import org.lwjgl.glfw.GLFW;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Always selects the native Wayland window system on Linux, unconditionally — not gated on whether HDR
 * is currently enabled. GLFW's platform is chosen once, before any window/surface exists, which is before
 * {@code CausticaConfig.Rt.Hdr.enabled()} can mean anything actionable and long before
 * {@code VulkanGpuSurfaceMixin} can know whether the resulting surface is PQ-capable. Since HDR is a
 * live runtime toggle (see {@code CausticaConfig.Rt.Hdr.swapchainPqAvailable}), the window has to already
 * be running on whatever backend can expose an HDR-capable surface before the toggle is ever flipped —
 * there is no "switch to Wayland later" once GLFW has initialized on X11. {@code -Dcaustica.hdr.x11=true}
 * restores vanilla's X11 choice for sessions where Wayland misbehaves; it also gives up Linux HDR.
 */
@Mixin(GLX.class)
public abstract class GlxMixin {
    @Inject(
            method = "_initGlfw",
            at = @At(
                    value = "INVOKE",
                    target = "Lorg/lwjgl/glfw/GLFW;glfwInit()Z",
                    shift = At.Shift.BEFORE))
    private static void caustica$preferWaylandForHdr(CallbackInfoReturnable<LongSupplier> cir) {
        if (!caustica$isLinux()) {
            return;
        }

        if (Boolean.getBoolean("caustica.hdr.x11")) {
            CausticaMod.LOGGER.info("HDR: caustica.hdr.x11 is set; keeping vanilla's X11 window system");
            return;
        }

        if (!GLFW.glfwPlatformSupported(GLFW.GLFW_PLATFORM_WAYLAND)) {
            CausticaMod.LOGGER.warn(
                    "HDR: this GLFW build has no Wayland support; Linux HDR presentation is unavailable");
            return;
        }

        String waylandDisplay = System.getenv("WAYLAND_DISPLAY");
        String sessionType = System.getenv("XDG_SESSION_TYPE");
        boolean waylandSession = waylandDisplay != null && !waylandDisplay.isBlank()
                || "wayland".equalsIgnoreCase(sessionType);
        if (!waylandSession) {
            CausticaMod.LOGGER.warn(
                    "HDR: no Wayland session detected; Linux HDR requires launching Minecraft in a native Wayland session");
            return;
        }

        // Minecraft prefers X11 when both backends exist. This later hint replaces that choice for HDR.
        GLFW.glfwInitHint(GLFW.GLFW_PLATFORM, GLFW.GLFW_PLATFORM_WAYLAND);
        CausticaMod.LOGGER.info("HDR: selecting GLFW's native Wayland backend for Linux HDR presentation");
    }

    @Inject(method = "_initGlfw", at = @At("RETURN"))
    private static void caustica$logHdrWindowSystem(CallbackInfoReturnable<LongSupplier> cir) {
        if (!caustica$isLinux()) {
            return;
        }

        int platform = GLFW.glfwGetPlatform();
        if (platform == GLFW.GLFW_PLATFORM_WAYLAND) {
            CausticaMod.LOGGER.info("HDR: GLFW initialized with the native Wayland backend");
        } else {
            CausticaMod.LOGGER.warn(
                    "HDR: GLFW initialized with platform {} instead of Wayland; the Vulkan surface is unlikely to expose HDR formats",
                    caustica$platformName(platform));
        }
    }

    @Unique
    private static boolean caustica$isLinux() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux");
    }

    @Unique
    private static String caustica$platformName(int platform) {
        return switch (platform) {
            case GLFW.GLFW_PLATFORM_WAYLAND -> "Wayland";
            case GLFW.GLFW_PLATFORM_X11 -> "X11";
            case GLFW.GLFW_PLATFORM_WIN32 -> "Win32";
            case GLFW.GLFW_PLATFORM_COCOA -> "Cocoa";
            case GLFW.GLFW_PLATFORM_NULL -> "Null";
            default -> "unknown (" + platform + ")";
        };
    }
}
