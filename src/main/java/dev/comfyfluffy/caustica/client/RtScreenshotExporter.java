package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtComposite;
import net.minecraft.ChatFormatting;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Util;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

/** Adds a residual-exposed scene-linear ACEScg EXR beside vanilla's ordinary F2 PNG. */
public final class RtScreenshotExporter {
    private RtScreenshotExporter() {
    }

    /**
     * Exports the RT image and returns the exact filename vanilla should use for the paired PNG.
     * Returns {@code null} only when a filename cannot be reserved, allowing the caller to fall back to
     * vanilla's ordinary auto-naming path.
     */
    public static String exportPaired(File workDir, Consumer<Component> callback) {
        Path screenshotDirectory = workDir.toPath().resolve("screenshots");
        try {
            Files.createDirectories(screenshotDirectory);
            Path output = nextPairedPath(screenshotDirectory);
            if (!RtComposite.INSTANCE.exportLatestResidualExposureExr(output)) {
                // Without a line here a missing EXR is indistinguishable from a lost write, and the
                // usual cause is simply that RT was not producing frames when F2 was pressed.
                CausticaMod.LOGGER.info("No residual-exposure EXR written: the RT composite had no frame to export");
                callback.accept(Component.translatable(
                        "caustica.options.rt.screenshot.exrUnavailable").withStyle(ChatFormatting.YELLOW));
                return pngName(output);
            }
            File file = output.toFile().getAbsoluteFile();
            Component link = Component.literal(file.getName())
                    .withStyle(ChatFormatting.UNDERLINE)
                    .withStyle(style -> style.withClickEvent(new ClickEvent.OpenFile(file)));
            callback.accept(Component.translatable("caustica.options.rt.screenshot.exrSaved", link));
            CausticaMod.LOGGER.info("Saved residual-exposure ACEScg screenshot to {}", file);
            return pngName(output);
        } catch (Exception e) {
            CausticaMod.LOGGER.warn("Couldn't save residual-exposure ACEScg screenshot", e);
            callback.accept(Component.translatable("caustica.options.rt.screenshot.exrFailed", e.getMessage())
                    .withStyle(ChatFormatting.RED));
            return null;
        }
    }

    private static String pngName(Path exrPath) {
        String name = exrPath.getFileName().toString();
        return name.substring(0, name.length() - ".exr".length()) + ".png";
    }

    private static Path nextPairedPath(Path directory) {
        String base = Util.getFilenameFormattedDateTime();
        int count = 1;
        while (true) {
            String suffix = count == 1 ? "" : "_" + count;
            Path candidate = directory.resolve(base + suffix + ".exr");
            Path vanillaPng = directory.resolve(base + suffix + ".png");
            if (!Files.exists(candidate) && !Files.exists(vanillaPng)) {
                return candidate;
            }
            count++;
        }
    }
}
