package dev.comfyfluffy.caustica.ngx;

import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;

import net.fabricmc.loader.api.FabricLoader;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkInstance;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * Shared NVIDIA NGX lifetime for the mod. Loads the native shim, extracts the bundled NGX feature DLLs,
 * and runs {@code ngxshim_init} / {@code ngxshim_shutdown} exactly once per Vulkan device. Multiple NGX
 * features (DLSS Ray Reconstruction and DLSS Frame Generation) share this single initialized
 * {@link NgxLibrary}; each feature owns only its own create/evaluate/release. NGX is shut down only at
 * device teardown (so releasing one feature can't tear NGX down while another still holds a handle).
 */
public final class NgxRuntime {
    public static final NgxRuntime INSTANCE = new NgxRuntime();

    private static final PlatformNatives PLATFORM_NATIVES = PlatformNatives.current();

    // One shim load per resolved path, reused across init/shutdown cycles: the global-arena lookup
    // keeps the DLL loaded for the JVM's life anyway, so a second load would only grow the OS
    // reference count and strand the previous method handles. This also keeps the natives
    // re-extraction contract intact — an already-correct file on disk is never rewritten, so the
    // loader never has to open a file another module has mapped.
    private static volatile NgxLibrary cachedLibrary;
    private static volatile Path cachedLibraryPath;

    private NgxLibrary lib;
    private boolean initialized;
    private boolean failed;
    private long initializedDevice;

    private NgxRuntime() {
    }

    /**
     * Ensure NGX is loaded and initialized for {@code device}, returning the shared {@link NgxLibrary}, or
     * {@code null} if it is unavailable. Idempotent; latches failure so it isn't retried every frame
     * (cleared by {@link #shutdown()} so a fresh device can re-init). A device other than the one NGX was
     * initialized against first shuts the old state down and re-initializes, so a rebuilt render device
     * never receives a library still bound to the previous {@code VkDevice}.
     */
    public synchronized NgxLibrary acquire(VulkanDevice device) {
        if (initialized) {
            if (initializedDevice == device.vkDevice().address()) {
                return lib;
            }
            CausticaMod.LOGGER.info("NGX device changed; re-initializing for the new Vulkan device");
            shutdown();
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
            CausticaMod.LOGGER.error("NGX init failed; DLSS features disabled", t);
            return null;
        }
    }

    public synchronized boolean isInitialized() {
        return initialized;
    }

    /** The shared library once {@link #acquire} has succeeded, else {@code null}. */
    public NgxLibrary library() {
        return lib;
    }

    /**
     * Shut down NGX. Call only at device teardown, after every feature has been released. Uses the device
     * handle recorded at init time rather than resolving one from the render backend, which may already be
     * gone when this runs. No-op if NGX was never initialized.
     */
    public synchronized void shutdown() {
        if (lib != null && initialized && initializedDevice != 0L) {
            try {
                lib.shutdown(initializedDevice);
            } catch (Throwable t) {
                CausticaMod.LOGGER.warn("NGX shutdown failed", t);
            }
        }
        initialized = false;
        failed = false;
        lib = null;
        initializedDevice = 0L;
    }

    /**
     * Extract the bundled natives and load the shim ahead of the first frame that needs it, so the file
     * IO and dlopen do not land on the render thread inside an {@link #acquire} call. Never initializes
     * NGX; safe to call before a device exists, and cheap to repeat.
     */
    public void prewarm() {
        if (!PLATFORM_NATIVES.supported()) {
            return;
        }
        Path shim = locateShim();
        if (shim == null) {
            return;
        }
        try {
            loadShim(shim);
        } catch (Throwable t) {
            CausticaMod.LOGGER.debug("NGX shim prewarm did not complete ({})", t.toString());
        }
    }

    /** NVSDK_NGX_Result: failure when the top 12 bits == 0xBAD. Shared by all NGX feature wrappers. */
    public static boolean ngxFailed(int result) {
        // Negative values are shim-side sentinels rather than NVSDK_NGX_Result codes, and their top
        // bits do not spell 0xBAD; every negative is a failure.
        return result < 0 || (result & 0xFFF00000) == 0xBAD00000;
    }

    private void init(VulkanDevice device) {
        if (!PLATFORM_NATIVES.supported()) {
            throw new IllegalStateException("NGX natives are not bundled for " + PLATFORM_NATIVES.platformDir());
        }
        Path shim = locateShim();
        if (shim == null) {
            throw new IllegalStateException(PLATFORM_NATIVES.shimName()
                    + " not found (bundled natives or -Dcaustica.ngx.path)");
        }
        Path nativesDir = shim.getParent();
        if (nativesDir != null) {
            List<String> missingFeatures = missingFeatureLibraries(nativesDir);
            if (!missingFeatures.isEmpty()) {
                CausticaMod.LOGGER.warn("NGX feature libraries {} not found next to {}; those features will be unavailable",
                        missingFeatures, PLATFORM_NATIVES.shimName());
            }
        }

        lib = loadShim(shim);

        Path dataPath = FabricLoader.getInstance().getGameDir().resolve("bettercaustica-ngx");
        try {
            Files.createDirectories(dataPath);
        } catch (Exception e) {
            CausticaMod.LOGGER.warn("Could not create NGX data path {}", dataPath, e);
        }

        VkInstance instance = device.vkDevice().getPhysicalDevice().getInstance();
        try (Arena arena = Arena.ofConfined()) {
            long gdpa;
            try (MemoryStack stack = MemoryStack.stackPush()) {
                gdpa = VK10.vkGetInstanceProcAddr(instance, stack.ASCII("vkGetDeviceProcAddr"));
            }
            int rc = lib.init(0L, wideString(arena, dataPath.toString()),
                    instance.address(), device.vkDevice().getPhysicalDevice().address(), device.vkDevice().address(),
                    0L, gdpa, wideString(arena, nativesDir == null ? "" : nativesDir.toString()));
            if (ngxFailed(rc)) {
                throw new IllegalStateException("ngxshim_init failed: 0x" + Integer.toHexString(rc)
                        + " last=0x" + Integer.toHexString(lib.lastResult()));
            }
        }
        initializedDevice = device.vkDevice().address();
        CausticaMod.LOGGER.info("NGX initialized (shim {})", shim);
    }

    private static NgxLibrary loadShim(Path shim) {
        NgxLibrary cached = cachedLibrary;
        if (cached != null && shim.equals(cachedLibraryPath)) {
            return cached;
        }
        NgxLibrary loaded = NgxLibrary.load(shim);
        cachedLibrary = loaded;
        cachedLibraryPath = shim;
        return loaded;
    }

    private static Path locateShim() {
        String override = CausticaConfig.Ngx.PATH.get();
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            if (Files.isDirectory(p)) {
                p = p.resolve(PLATFORM_NATIVES.shimName());
            }
            return Files.isRegularFile(p) ? p : null;
        }
        return extractBundledNatives();
    }

    private static Path extractBundledNatives() {
        Path dir = FabricLoader.getInstance().getGameDir().resolve("bettercaustica-ngx")
                .resolve("natives").resolve(PLATFORM_NATIVES.platformDir());
        try {
            Files.createDirectories(dir);
            boolean hasShim = extractBundledNative(PLATFORM_NATIVES.shimName(), dir.resolve(PLATFORM_NATIVES.shimName()));
            extractBundledFeatureLibraries(dir);
            return hasShim && Files.isRegularFile(dir.resolve(PLATFORM_NATIVES.shimName()))
                    ? dir.resolve(PLATFORM_NATIVES.shimName()) : null;
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("Could not extract bundled NGX natives to {}", dir, e);
            return null;
        }
    }

    private static boolean extractBundledNative(String name, Path dst) throws IOException {
        String resource = PLATFORM_NATIVES.resourceDir() + name;
        try (InputStream in = NgxRuntime.class.getResourceAsStream(resource)) {
            if (in == null) {
                return false;
            }
            byte[] bytes = in.readAllBytes();
            if (!sameBytes(dst, bytes)) {
                Files.write(dst, bytes);
            }
            return true;
        }
    }

    private static void extractBundledFeatureLibraries(Path dir) throws IOException {
        for (String name : PLATFORM_NATIVES.exactFeatureNames()) {
            extractBundledNative(name, dir.resolve(name));
        }
        for (String name : bundledFeatureLibraryNames()) {
            extractBundledNative(name, dir.resolve(name));
        }
    }

    private static List<String> bundledFeatureLibraryNames() {
        List<String> names = new ArrayList<>();
        FabricLoader.getInstance().getModContainer("bettercaustica").ifPresent(container -> {
            String nativeDir = "bettercaustica/natives/" + PLATFORM_NATIVES.platformDir();
            for (Path root : container.getRootPaths()) {
                Path dir = root.resolve(nativeDir);
                if (!Files.isDirectory(dir)) {
                    continue;
                }
                try (Stream<Path> files = Files.list(dir)) {
                    files.map(path -> path.getFileName().toString())
                            .filter(PLATFORM_NATIVES::isFeatureLibrary)
                            .forEach(names::add);
                } catch (IOException e) {
                    CausticaMod.LOGGER.warn("Could not list bundled NGX natives in {}", dir, e);
                }
            }
        });
        return names;
    }

    private static List<String> missingFeatureLibraries(Path dir) {
        List<String> missing = new ArrayList<>();
        for (String name : PLATFORM_NATIVES.exactFeatureNames()) {
            if (!Files.isRegularFile(dir.resolve(name))) {
                missing.add(name);
            }
        }
        List<String> names;
        try (Stream<Path> files = Files.list(dir)) {
            names = files.map(path -> path.getFileName().toString()).toList();
        } catch (IOException e) {
            return PLATFORM_NATIVES.featureDescriptions();
        }
        for (String prefix : PLATFORM_NATIVES.featureNamePrefixes()) {
            if (names.stream().noneMatch(name -> name.startsWith(prefix))) {
                missing.add(prefix + "*");
            }
        }
        return missing;
    }

    private static boolean sameBytes(Path path, byte[] bytes) throws IOException {
        try {
            return Files.size(path) == bytes.length && Arrays.equals(Files.readAllBytes(path), bytes);
        } catch (NoSuchFileException e) {
            return false;
        }
    }

    // Native wchar_t width differs by platform: 2 bytes (UTF-16) on Windows, 4 bytes (UTF-32)
    // on Linux. Encode paths to the platform width expected by the NGX C ABI, per the same probe
    // that selected the natives bundle.
    private static final Charset WCHAR_CHARSET =
            PLATFORM_NATIVES.wcharUtf16() ? StandardCharsets.UTF_16LE : Charset.forName("UTF-32LE");
    private static final int WCHAR_SIZE = PLATFORM_NATIVES.wcharUtf16() ? 2 : 4;

    private static MemorySegment wideString(Arena arena, String s) {
        byte[] data = s.getBytes(WCHAR_CHARSET);
        MemorySegment seg = arena.allocate((long) data.length + WCHAR_SIZE);
        MemorySegment.copy(data, 0, seg, ValueLayout.JAVA_BYTE, 0, data.length);
        for (int i = 0; i < WCHAR_SIZE; i++) {
            seg.set(ValueLayout.JAVA_BYTE, data.length + i, (byte) 0);
        }
        return seg;
    }

    private record PlatformNatives(String platformDir, String shimName, List<String> exactFeatureNames,
                                   List<String> featureNamePrefixes, boolean wcharUtf16, boolean supported) {
        private static PlatformNatives current() {
            String os = System.getProperty("os.name", "").toLowerCase();
            String arch = System.getProperty("os.arch", "").toLowerCase();
            boolean x64 = arch.equals("x86_64") || arch.equals("amd64");
            if (os.contains("win") && x64) {
                return new PlatformNatives("windows-x64", "ngxshim.dll",
                        List.of("nvngx_dlssd.dll", "nvngx_dlssg.dll"), List.of(), true, true);
            }
            if (os.contains("linux") && x64) {
                return new PlatformNatives("linux-x64", "libngxshim.so", List.of(),
                        List.of("libnvidia-ngx-dlssd.so", "libnvidia-ngx-dlssg.so"), false, true);
            }
            return new PlatformNatives(os + "/" + arch, System.mapLibraryName("ngxshim"), List.of(), List.of(),
                    false, false);
        }

        private String resourceDir() {
            return "/bettercaustica/natives/" + platformDir + "/";
        }

        private boolean isFeatureLibrary(String name) {
            return exactFeatureNames.contains(name)
                    || featureNamePrefixes.stream().anyMatch(name::startsWith);
        }

        private List<String> featureDescriptions() {
            List<String> descriptions = new ArrayList<>(exactFeatureNames);
            featureNamePrefixes.stream()
                    .map(prefix -> prefix + "*")
                    .forEach(descriptions::add);
            return descriptions;
        }
    }
}
