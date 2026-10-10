package dev.comfyfluffy.caustica.ffx;

import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;

import net.fabricmc.loader.api.FabricLoader;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkInstance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Loads the FidelityFX shim and its AMD runtime for the mod's Vulkan device.
 *
 * <p>{@code amd_fidelityfx_vk.dll} is a dependency of {@code ffxshim.dll} rather than something
 * Java calls, so the two are staged in the same directory and the shim is told to look there.
 * Loading it from the mod's own directory matters: the JVM's library path contains LWJGL's
 * extracted natives, and a system-wide FidelityFX DLL from a game install would otherwise be
 * free to satisfy the import.
 *
 * <p>Failure latches for the session, because a missing or mismatched SDK is not something a
 * frame can fix, and retrying per frame would flood the log. {@link #shutdown()} clears the latch
 * so a new device gets a fresh attempt.
 */
public final class FfxRuntime {
    public static final FfxRuntime INSTANCE = new FfxRuntime();

    private static final PlatformNatives PLATFORM_NATIVES = PlatformNatives.current();

    // Volatile because probe and render paths read it from the render thread while a concurrent
    // tick path may still be inside acquire(); all mutations themselves stay synchronized.
    private volatile FfxLibrary lib;
    private boolean initialized;
    private boolean failed;

    private FfxRuntime() {
    }

    /**
     * Ensure the shim is loaded and its SDK resolved for {@code device}, or {@code null} if it is
     * unavailable. Idempotent.
     */
    public synchronized FfxLibrary acquire(VulkanDevice device) {
        if (initialized) {
            return lib;
        }
        if (failed) {
            return null;
        }
        try {
            init(device);
            initialized = true;
            return lib;
        } catch (Throwable t) {
            failed = true;
            lib = null;
            CausticaMod.LOGGER.error("FidelityFX init failed; FSR frame generation disabled", t);
            return null;
        }
    }

    /** The shared library once {@link #acquire} has succeeded, else {@code null}. */
    public FfxLibrary library() {
        return lib;
    }

    /**
     * vkGetDeviceProcAddr as a raw address, for the backend description the shim builds. Resolved
     * through the instance LWJGL is using, so the SDK and the engine see the same dispatch chain
     * and any layers in it.
     */
    public static long deviceProcAddr(VulkanDevice device) {
        VkInstance instance = device.vkDevice().getPhysicalDevice().getInstance();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            return VK10.vkGetInstanceProcAddr(instance, stack.ASCII("vkGetDeviceProcAddr"));
        }
    }

    /** Tear down the context and release the SDK. Call at device teardown only. */
    public synchronized void shutdown() {
        if (lib != null) {
            try {
                lib.destroy();
                lib.unload();
            } catch (Throwable t) {
                CausticaMod.LOGGER.warn("FidelityFX shutdown failed", t);
            }
        }
        lib = null;
        initialized = false;
        failed = false;
    }

    private void init(VulkanDevice device) {
        if (!PLATFORM_NATIVES.supported()) {
            throw new IllegalStateException("FidelityFX natives are not bundled for "
                    + PLATFORM_NATIVES.platformDir());
        }
        Path shim = locateShim();
        if (shim == null) {
            throw new IllegalStateException(PLATFORM_NATIVES.shimName()
                    + " not found (bundled natives or -Dcaustica.ffx.path)");
        }
        lib = FfxLibrary.load(shim);
        // Passing the directory rather than a file name is what the shim's own default would do;
        // naming it keeps the two DLLs together regardless of how the loader resolves imports.
        int rc = lib.load(shim.getParent() == null ? null : shim.getParent().toString());
        if (rc != FfxLibrary.OK || !lib.available()) {
            throw new IllegalStateException("ffxshim_load failed: rc=" + rc + " " + lib.lastMessage());
        }
        // No provider version is reported here: that is a property of the effect context, which
        // does not exist until RtFsrFg creates one.
        CausticaMod.LOGGER.info("FidelityFX loaded from {} (sdk next to it)", shim);
    }

    private static Path locateShim() {
        String override = CausticaConfig.Ffx.PATH.get();
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            if (Files.isDirectory(p)) {
                p = p.resolve(PLATFORM_NATIVES.shimName());
            }
            return Files.isRegularFile(p) ? p : null;
        }
        return extractBundledNatives();
    }

    /**
     * Write both DLLs into the game directory, leaving an already-correct copy alone so a launch
     * does not rewrite 9 MB every time.
     */
    private static Path extractBundledNatives() {
        Path dir = FabricLoader.getInstance().getGameDir().resolve("bettercaustica-ffx")
                .resolve("natives").resolve(PLATFORM_NATIVES.platformDir());
        try {
            Files.createDirectories(dir);
            for (String name : PLATFORM_NATIVES.fileNames()) {
                extractBundledNative(name, dir.resolve(name));
            }
            Path shim = dir.resolve(PLATFORM_NATIVES.shimName());
            return Files.isRegularFile(shim) ? shim : null;
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("Could not extract bundled FidelityFX natives to {}", dir, e);
            return null;
        }
    }

    private static void extractBundledNative(String name, Path dst) throws IOException {
        String resource = "/bettercaustica/natives/" + PLATFORM_NATIVES.platformDir() + "/" + name;
        try (InputStream in = FfxRuntime.class.getResourceAsStream(resource)) {
            if (in == null) {
                return;
            }
            byte[] bytes = in.readAllBytes();
            if (!sameBytes(dst, bytes)) {
                Files.write(dst, bytes);
            }
        }
    }

    private static boolean sameBytes(Path path, byte[] bytes) throws IOException {
        return Files.isRegularFile(path)
                && Files.size(path) == bytes.length
                && Arrays.equals(Files.readAllBytes(path), bytes);
    }

    private record PlatformNatives(String platformDir, String shimName, String sdkName, boolean supported) {
        private static PlatformNatives current() {
            String os = System.getProperty("os.name", "").toLowerCase();
            String arch = System.getProperty("os.arch", "").toLowerCase();
            boolean x64 = arch.equals("x86_64") || arch.equals("amd64");
            if (os.contains("win") && x64) {
                return new PlatformNatives("windows-x64", "ffxshim.dll", "amd_fidelityfx_vk.dll", true);
            }
            // AMD ships no Linux build of the Vulkan backend, so there is nothing to bundle and
            // the loader path stays off rather than failing at launch.
            return new PlatformNatives(os + "/" + arch, System.mapLibraryName("ffxshim"), "", false);
        }

        private String[] fileNames() {
            return supported ? new String[]{shimName, sdkName} : new String[]{shimName};
        }
    }
}
