package dev.comfyfluffy.caustica.rt.material;

/** Immutable physical material description produced at the resource-epoch boundary. */
public record RtMaterialDesc(
        int model,
        Source source,
        int features,
        float roughness,
        float metalness,
        float ior,
        float transmission,
        EmissionSource emissionSource,
        /**
         * Final HDR emitting-surface luminance in cd/m²: the look-package block baseline, replaced by
         * a resource-pack {@code emission.strength_cd_m2} value when present.
         * 0 when {@code emissionSource == NONE}. Applied uniformly regardless of source — LabPBR,
         * heuristic-mask, or state-uniform all get the same baseline unless absolutely overridden.
         */
        float emissionStrength,
        EmissionSummary emissionSummary
) {
    public enum Source {
        OVERRIDE,
        LAB_PBR,
        HEURISTIC,
        NEUTRAL
    }

    public enum EmissionSource {
        NONE,
        LAB_PBR,
        HEURISTIC_MASK,
        STATE_UNIFORM
    }

    /** Normalized compiler output; per-primitive state light multiplies it when the source is state-gated. */
    public record EmissionSummary(float averageR, float averageG, float averageB,
                                  float integratedLuminance, float coverage) {
        public static final EmissionSummary NONE = new EmissionSummary(0, 0, 0, 0, 0);

        public boolean emissive() {
            return integratedLuminance > 0.0f && coverage > 0.0f;
        }
    }

    public RtMaterialDesc {
        if (source == null || emissionSource == null || emissionSummary == null) {
            throw new IllegalArgumentException("Material description enums/summary must be present");
        }
        // Shader-side models include particle (2), which the CPU compiler never produces; keeping the
        // CPU surface closed to it catches a mis-mapped constant at compile time rather than as noise
        // in the traced image.
        if (model != 0 && model != 1 && model != 3) {
            throw new IllegalArgumentException("RtMaterialDesc model must be opaque (0), water (1) or dielectric (3): "
                    + model);
        }
        if (!finite01(roughness) || !finite01(metalness) || !Float.isFinite(ior) || ior <= 0.0f
                || !finite01(transmission) || !Float.isFinite(emissionStrength) || emissionStrength < 0.0f) {
            throw new IllegalArgumentException("Invalid physical material parameters");
        }
    }

    private static boolean finite01(float value) {
        return Float.isFinite(value) && value >= 0.0f && value <= 1.0f;
    }
}
