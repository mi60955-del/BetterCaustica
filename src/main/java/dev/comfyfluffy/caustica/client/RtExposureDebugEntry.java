package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.rt.RtComposite;
import dev.comfyfluffy.caustica.rt.pipeline.RtExposure;
import net.minecraft.client.gui.components.debug.DebugScreenDisplayer;
import net.minecraft.client.gui.components.debug.DebugScreenEntries;
import net.minecraft.client.gui.components.debug.DebugScreenEntry;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.chunk.LevelChunk;
import org.jspecify.annotations.Nullable;

/**
 * F3 line for the RT auto-exposure controller. Off by default like any other optional entry
 * (Vanilla's own per-player {@code debug-profile.json}, toggled through the F3 entry list) --
 * registration only makes it available, it does not turn it on.
 *
 * <p>{@code RtExposure.debugSummaryLine()} owns the displayed controller values.
 */
public final class RtExposureDebugEntry implements DebugScreenEntry {
    public static final Identifier ID = DebugScreenEntries.register(
            Identifier.fromNamespaceAndPath("caustica", "rt_exposure"), new RtExposureDebugEntry());

    @Override
    public void display(DebugScreenDisplayer displayer, @Nullable Level serverOrClientLevel,
                        @Nullable LevelChunk clientChunk, @Nullable LevelChunk serverChunk) {
        RtComposite composite = RtComposite.INSTANCE;
        if (composite.hasFailed()) {
            return; // vanilla is rendering this frame; the exposure state is stale/irrelevant.
        }
        RtExposure exposure = composite.exposure();
        if (!exposure.ready()) {
            return; // RT hasn't produced an exposure value yet (no world, or still bringing up).
        }
        String line = exposure.debugSummaryLine();
        if (line != null) {
            displayer.addLine(line);
        }
    }

    @Override
    public boolean isAllowed(boolean reducedDebugInfo) {
        return !reducedDebugInfo;
    }
}
