package dev.comfyfluffy.caustica.rt.material;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the canonical page-size search: the footprint that matters is page count times page area, and a
 * fixed large page is not the cheapest layout for ordinary 16x16 atlases.
 */
final class RtBlockMaterialsPageSizeTest {
    private static final int PAGE_CAPACITY = 1024;

    @Test
    void noAuthoredSpritesStillReserveTheFallbackPage() {
        int[] plan = RtBlockMaterials.planPages(new int[0][], PAGE_CAPACITY);
        assertEquals(32, plan[0]);
        assertEquals(1, plan[1]);
    }

    @Test
    void vanillaSizedAtlasPacksIntoSmallPages() {
        int[][] extents = new int[147][];
        for (int i = 0; i < extents.length; i++) {
            extents[i] = new int[]{16, 16};
        }
        int[] plan = RtBlockMaterials.planPages(extents, PAGE_CAPACITY);
        long area = (long) plan[1] * plan[0] * plan[0];
        System.out.println("vanilla plan: size=" + plan[0] + " pages=" + plan[1] + " area=" + area);
        assertTrue(area <= 4L * 256 * 256,
                "147 vanilla sprites must not need more than a few small pages, got " + plan[0] + "x" + plan[1]);
    }

    @Test
    void highResolutionPackKeepsAPageItsCellsCanTile() {
        int[][] extents = new int[1300][];
        for (int i = 0; i < extents.length; i++) {
            extents[i] = new int[]{128, 128};
        }
        int[] plan = RtBlockMaterials.planPages(extents, PAGE_CAPACITY);
        long area = (long) plan[1] * plan[0] * plan[0];
        System.out.println("128px plan: size=" + plan[0] + " pages=" + plan[1] + " area=" + area);
        // A 144-pixel cell (128 + 2x8 gutter, aligned to 8) fits once per 256 page, so small pages would
        // cost one page per sprite. The search has to climb to a page that tiles.
        assertTrue(plan[0] >= 512, "128px cells cannot tile a small page, chose " + plan[0]);
        // Efficiency bound: the chosen footprint must stay within half again of the packed cell area,
        // which rules out both the one-cell-per-page cliff and a page so large that shelf rounding and
        // gutter replication dominate.
        long cells = (long) extents.length * 144 * 144;
        assertTrue(area < cells * 3 / 2, "packing efficiency lost: area=" + area + " cells=" + cells);
    }

    @Test
    void largePagesAreOnlyChosenWhenTheyActuallyPackTighter() {
        // 260 sprites at 16x16: three pages of 256 hold 192 cells, so the search must either add a page
        // or step up a size — whichever costs fewer texels.
        int[][] extents = new int[260][];
        for (int i = 0; i < extents.length; i++) {
            extents[i] = new int[]{16, 16};
        }
        int[] plan = RtBlockMaterials.planPages(extents, PAGE_CAPACITY);
        long area = (long) plan[1] * plan[0] * plan[0];
        System.out.println("260x16 plan: size=" + plan[0] + " pages=" + plan[1] + " area=" + area);
        assertTrue(area <= 2048L * 2048L,
                "the chosen layout must not exceed the footprint of one legacy page, got " + area);
    }
}
