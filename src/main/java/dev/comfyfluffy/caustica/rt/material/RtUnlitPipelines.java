package dev.comfyfluffy.caustica.rt.material;

import com.mojang.blaze3d.pipeline.RenderPipeline;
import net.minecraft.resources.Identifier;

import java.util.Set;

/**
 * Vanilla pipelines that draw with no lighting term at all.
 *
 * <p>Their fragment shader outputs texture x vertex colour x colour-modulator and stops there, so what
 * reaches the screen is already the final display colour. Capturing such a submission as an ordinary
 * diffuse surface makes the tracer light something that was never lit: the beacon beam comes back as a
 * sun-shaded column instead of the flat ribbon the game authored.
 *
 * <p>Keyed on the fragment shader rather than on the render type or pipeline name, because one shader
 * backs every variant of it — the beacon beam has an opaque and a blended layer, and both are unlit.
 * A pipeline that is not listed here is treated as lit, which is the safe default.
 */
public final class RtUnlitPipelines {
    private static final Set<String> UNLIT_FRAGMENT_SHADERS = Set.of(
            "core/rendertype_beacon_beam",
            // The lightning bolt is a vertex-coloured quad list — RenderTypes.LIGHTNING binds no texture at
            // all — drawn as vertexColor x ColorModulator and nothing else, so every bit of its brightness
            // is authored rather than lit.
            "core/rendertype_lightning",
            // The end portal and the end gateway sum texture layers against a fixed colour table and stop;
            // nothing in the pipeline is lit, so tracing them as a diffuse surface would light a colour
            // vanilla never lit. Their composite itself is reproduced by ENTITY_PRIM_END_PORTAL.
            "core/rendertype_end_portal");

    /** Vanilla's beacon-beam fragment shader, keyed the same way (one shader backs both its layers). */
    private static final String BEACON_BEAM_FRAGMENT_SHADER = "core/rendertype_beacon_beam";

    private RtUnlitPipelines() {
    }

    /**
     * The beacon beam specifically, for the caller that has to scale its emission: within the unlit set
     * the beam is the one surface authored as a dim additive glow ribbon rather than as a bright object,
     * so it cannot share the glowstone-face baseline the others are anchored on.
     */
    public static boolean beaconBeam(RenderPipeline pipeline) {
        if (pipeline == null) {
            return false;
        }
        Identifier fragment = pipeline.getFragmentShader();
        return fragment != null && BEACON_BEAM_FRAGMENT_SHADER.equals(fragment.getPath());
    }

    public static boolean unlit(RenderPipeline pipeline) {
        if (pipeline == null) {
            return false;
        }
        Identifier fragment = pipeline.getFragmentShader();
        return fragment != null && UNLIT_FRAGMENT_SHADERS.contains(fragment.getPath());
    }
}
