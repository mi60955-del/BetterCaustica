package dev.comfyfluffy.caustica.rt.material;

import dev.comfyfluffy.caustica.CausticaMod;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.block.dispatch.BlockStateModel;
import net.minecraft.client.renderer.block.dispatch.BlockStateModelPart;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.client.resources.model.geometry.BakedQuad;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.RenderShape;
import net.minecraft.world.level.block.state.BlockState;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Resource-epoch proof that an atlas sprite is used by at least one light-emitting block state. */
public final class RtEmissionSemantics {
    private static final Direction[] DIRECTIONS = Direction.values();
    /**
     * Random variants sampled per emissive state. A weighted multipart part that can appear with
     * probability under 1/{@code VARIANT_PROBES} can escape every probe and lose its emission mask;
     * raising this trades reload time for coverage of ever-rarer variants.
     */
    private static final int VARIANT_PROBES = 8;

    /**
     * Sprites that emit uniformly across their whole surface, so their block light IS the emission and no
     * spatial mask should be inferred for them. The heuristic asks "which part of this texture glows" and
     * answers with the cube of the texel's linear luminance, so it gives up on a dark, fully saturated
     * emitter outright: the nether portal's purple averages 0.16 linear luminance, which masks down to
     * about 0.003 — a block vanilla lights at level 11 would render almost black.
     */
    private static final Set<String> UNIFORM_EMITTER_SPRITES = Set.of("block/nether_portal");

    private final Map<TextureAtlasSprite, Integer> maxStateLight;

    private RtEmissionSemantics(Map<TextureAtlasSprite, Integer> maxStateLight) {
        this.maxStateLight = maxStateLight;
    }

    public static RtEmissionSemantics analyze() {
        IdentityHashMap<TextureAtlasSprite, Integer> sprites = new IdentityHashMap<>();
        int states = 0;
        int failures = 0;
        for (Block block : BuiltInRegistries.BLOCK) {
            for (BlockState state : block.getStateDefinition().getPossibleStates()) {
                int light = state.getLightEmission();
                if (light <= 0 || state.getRenderShape() != RenderShape.MODEL) continue;
                states++;
                try {
                    collectState(state, light, sprites);
                } catch (Throwable throwable) {
                    failures++;
                    CausticaMod.LOGGER.debug("Could not analyze emissive material sprites for {}", state, throwable);
                }
            }
        }
        Map<TextureAtlasSprite, Integer> frozen = Collections.unmodifiableMap(new IdentityHashMap<>(sprites));
        CausticaMod.LOGGER.info("RT emission semantics: emittingStates={}, sprites={}, failedStates={}",
                states, frozen.size(), failures);
        return new RtEmissionSemantics(frozen);
    }

    /**
     * True when {@code sprite} should compile an albedo-derived emission mask. A sprite that emits
     * uniformly is deliberately excluded: its block light is already the whole answer, and masking it by
     * texture luminance only dims it.
     */
    public boolean permits(TextureAtlasSprite sprite) {
        if (!maxStateLight.containsKey(sprite)) return false;
        return !UNIFORM_EMITTER_SPRITES.contains(sprite.contents().name().getPath());
    }

    /**
     * Whether some light-emitting block state's model can resolve {@code sprite}, i.e. the sprite can be
     * shaded with a state-uniform emission source and therefore needs a uniform emission summary and
     * albedo grid compiled. Sprites the variant probes missed degrade here the same way their heuristic
     * masks already do (see {@link #permits}).
     */
    public boolean mayResolveAsStateUniform(TextureAtlasSprite sprite) {
        return maxStateLight.containsKey(sprite);
    }

    private static void collectState(BlockState state, int light,
                                     IdentityHashMap<TextureAtlasSprite, Integer> sprites) {
        BlockStateModel model = Minecraft.getInstance().getModelManager().getBlockStateModelSet().get(state);
        if (model == null) return;
        for (int probe = 0; probe < VARIANT_PROBES; probe++) {
            List<BlockStateModelPart> parts = new ArrayList<>();
            model.collectParts(RandomSource.create(mixSeed(state.hashCode(), probe)), parts);
            for (BlockStateModelPart part : parts) {
                collectQuads(part.getQuads(null), light, sprites);
                for (Direction direction : DIRECTIONS) collectQuads(part.getQuads(direction), light, sprites);
            }
        }
    }

    private static void collectQuads(List<? extends BakedQuad> quads,
                                     int light, IdentityHashMap<TextureAtlasSprite, Integer> sprites) {
        for (var quad : quads) {
            TextureAtlasSprite sprite = quad.materialInfo().sprite();
            if (sprite != null) sprites.merge(sprite, light, Math::max);
        }
    }

    private static long mixSeed(int stateHash, int probe) {
        long seed = Integer.toUnsignedLong(stateHash) ^ (0x9E3779B97F4A7C15L * (probe + 1L));
        seed ^= seed >>> 30;
        seed *= 0xBF58476D1CE4E5B9L;
        seed ^= seed >>> 27;
        return seed;
    }
}
