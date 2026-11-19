package dev.comfyfluffy.caustica.rt.dh;

import net.fabricmc.loader.api.FabricLoader;

/**
 * Answers whether Distant Horizons is installed without resolving a single DH class.
 *
 * <p>{@link DhBridge} names DH API types in its field and method descriptors, so the JVM cannot link
 * it at all when DH is absent: the first touch throws {@code NoClassDefFoundError} — a
 * {@code LinkageError}, not an exception the bridge's own {@code available()} guard can catch,
 * because verification happens before any statement of the class runs. Every caller therefore
 * consults this gate first and skips its DhBridge calls when {@link #present()} is false.
 *
 * <p>The answer is fixed for the session: Fabric's mod list is final once the client entry points
 * have run, which is the earliest point any caller can execute.
 */
public final class DhPresence {
    private static final boolean PRESENT = FabricLoader.getInstance().isModLoaded("distanthorizons");

    private DhPresence() {
    }

    public static boolean present() {
        return PRESENT;
    }
}
