package dev.comfyfluffy.caustica.rt;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import dev.comfyfluffy.caustica.rt.pipeline.RtBloomPipeline;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Immutable, versioned calibration package for the scene-to-display pipeline.
 *
 * <p>The default package is deliberately a classpath asset rather than mutable TOML configuration:
 * exposure shaping, its scene-referred LMT, and the photometric light anchors are one authored look
 * and must move together. A future package selector can load another directory with the same schema
 * without reintroducing independent knobs.
 */
public record RtLookPackage(
        int schemaVersion,
        String id,
        int packageVersion,
        Exposure exposure,
        String lmtResource,
        Bloom bloom,
        Lighting lighting,
        Sky sky,
        End end) {
    public static final int SCHEMA_VERSION = 4;
    /** Shared with the bloom pipeline so a bad package fails at load, not at resize. */
    private static final int MAX_BLOOM_LEVELS = RtBloomPipeline.MAX_LEVELS;
    public static final String DEFAULT_ID = "default";
    public static final String DEFAULT_JSON = "/bettercaustica/color/looks/default/look.json";
    private static final RtLookPackage DEFAULT = load(DEFAULT_JSON);

    public static RtLookPackage current() {
        return DEFAULT;
    }

    static RtLookPackage parse(JsonObject root, String jsonResource) {
        int schemaVersion = requiredInt(root, "schemaVersion");
        if (schemaVersion != SCHEMA_VERSION) {
            throw new IllegalArgumentException(jsonResource + ": unsupported look-package schema "
                    + schemaVersion);
        }
        String id = requiredString(root, "id");
        if (id.isBlank()) {
            throw new IllegalArgumentException(jsonResource + ": id must not be blank");
        }
        int packageVersion = requiredInt(root, "packageVersion");
        if (packageVersion < 1) {
            throw new IllegalArgumentException(jsonResource + ": packageVersion must be at least 1");
        }

        JsonObject exposureJson = requiredObject(root, "exposure");
        Exposure exposure = new Exposure(
                requiredFinite(exposureJson, "minEv"),
                requiredFinite(exposureJson, "maxEv"),
                requiredString(exposureJson, "curve"));
        if (exposure.minEv() > exposure.maxEv()) {
            throw new IllegalArgumentException(jsonResource + ": exposure.minEv must be <= maxEv");
        }
        if (exposure.curve().isBlank()) {
            throw new IllegalArgumentException(jsonResource + ": exposure.curve must not be blank");
        }
        validateCurve(exposure.curve(), jsonResource);

        JsonObject lmt = requiredObject(root, "lmt");
        String lmtFile = requiredString(lmt, "resource");
        if (lmtFile.isBlank() || lmtFile.contains("/") || lmtFile.contains("\\")
                || ".".equals(lmtFile) || "..".equals(lmtFile)) {
            throw new IllegalArgumentException(jsonResource + ": lmt.resource must be a local file name");
        }
        int slash = jsonResource.lastIndexOf('/');
        if (slash < 0) {
            throw new IllegalArgumentException("look-package resource must be absolute: " + jsonResource);
        }
        String lmtResource = jsonResource.substring(0, slash + 1) + lmtFile;

        JsonObject bloomJson = requiredObject(root, "bloom");
        Bloom bloom = new Bloom(
                requiredFinite(bloomJson, "strength"),
                requiredFinite(bloomJson, "thresholdSceneLinear"),
                requiredFinite(bloomJson, "softKneeFraction"),
                requiredFinite(bloomJson, "radius"),
                requiredInt(bloomJson, "levels"));
        requireRange(bloom.strength(), 0.0f, 2.0f, jsonResource, "bloom.strength");
        requireRange(bloom.thresholdSceneLinear(), 0.0f, 65504.0f,
                jsonResource, "bloom.thresholdSceneLinear");
        requireRange(bloom.softKneeFraction(), 0.0f, 1.0f,
                jsonResource, "bloom.softKneeFraction");
        requireRange(bloom.radius(), 0.25f, 4.0f, jsonResource, "bloom.radius");
        if (bloom.levels() < 1 || bloom.levels() > MAX_BLOOM_LEVELS) {
            throw new IllegalArgumentException(jsonResource + ": bloom.levels must be in [1,"
                    + MAX_BLOOM_LEVELS + "]");
        }

        JsonObject lightingJson = requiredObject(root, "lighting");
        Lighting lighting = new Lighting(
                positive(lightingJson, "sunIlluminanceLux", jsonResource),
                positive(lightingJson, "moonIlluminanceLux", jsonResource),
                positive(lightingJson, "blockEmissionLuminanceCdM2", jsonResource),
                nonNegative(lightingJson, "nightAirglowLuminanceCdM2", jsonResource),
                nonNegative(lightingJson, "starLuminanceCdM2", jsonResource),
                nonNegative(lightingJson, "moonPhaseFixedFraction", jsonResource));
        if (lighting.moonPhaseFixedFraction() > 1.0f) {
            throw new IllegalArgumentException(jsonResource
                    + ": lighting.moonPhaseFixedFraction must be in [0,1]");
        }

        JsonObject skyJson = requiredObject(root, "sky");
        Sky sky = new Sky(
                requiredFinite(skyJson, "sunNoonSouthTiltDegrees"),
                nonNegative(skyJson, "sunAngularRadiusDegrees", jsonResource, "sky"),
                nonNegative(skyJson, "moonAngularRadiusDegrees", jsonResource, "sky"),
                positive(skyJson, "sunDiscHalfAngleDegrees", jsonResource, "sky"),
                positive(skyJson, "moonDiscHalfAngleDegrees", jsonResource, "sky"),
                nonNegative(skyJson, "groundAlbedo", jsonResource, "sky"),
                nonNegative(skyJson, "horizonSoftenDegrees", jsonResource, "sky"));
        requireRange(sky.sunNoonSouthTiltDegrees(), -89.0f, 89.0f,
                jsonResource, "sky.sunNoonSouthTiltDegrees");
        // The NEE radius only jitters the shadow ray, so it sets penumbra softness; the disc half-angle is
        // how large the body is DRAWN, matching vanilla's quads (which are ~60x the real sun). Both are
        // angles on the sky, so both stay well inside a quarter turn.
        requireRange(sky.sunAngularRadiusDegrees(), 0.0f, 20.0f,
                jsonResource, "sky.sunAngularRadiusDegrees");
        requireRange(sky.moonAngularRadiusDegrees(), 0.0f, 20.0f,
                jsonResource, "sky.moonAngularRadiusDegrees");
        requireRange(sky.sunDiscHalfAngleDegrees(), 0.0f, 45.0f,
                jsonResource, "sky.sunDiscHalfAngleDegrees");
        requireRange(sky.moonDiscHalfAngleDegrees(), 0.0f, 45.0f,
                jsonResource, "sky.moonDiscHalfAngleDegrees");
        requireRange(sky.groundAlbedo(), 0.0f, 1.0f, jsonResource, "sky.groundAlbedo");
        // Beyond a quarter turn the fade would still be running at the nadir, leaving the lower hemisphere
        // with no settled colour at all.
        requireRange(sky.horizonSoftenDegrees(), 0.0f, 90.0f,
                jsonResource, "sky.horizonSoftenDegrees");

        JsonObject endJson = requiredObject(root, "end");
        End end = new End(
                positive(endJson, "flashIlluminanceLux", jsonResource, "end"),
                nonNegative(endJson, "flashAngularRadiusDegrees", jsonResource, "end"),
                nonNegative(endJson, "flashQuadLuminanceCdM2", jsonResource, "end"),
                nonNegative(endJson, "ambientIlluminanceLux", jsonResource, "end"),
                nonNegative(endJson, "skyBoxGain", jsonResource, "end"),
                nonNegative(endJson, "skyGradientGain", jsonResource, "end"));
        // The flash glow is tens of degrees across; beyond a quarter turn the NEE square would wrap the
        // light's own horizon.
        requireRange(end.flashAngularRadiusDegrees(), 0.0f, 45.0f,
                jsonResource, "end.flashAngularRadiusDegrees");

        return new RtLookPackage(schemaVersion, id, packageVersion, exposure, lmtResource, bloom,
                lighting, sky, end);
    }

    private static RtLookPackage load(String resource) {
        try (InputStream stream = RtLookPackage.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IllegalStateException("missing look package " + resource);
            }
            try (InputStreamReader reader = new InputStreamReader(stream, StandardCharsets.UTF_8)) {
                RtLookPackage value = parse(JsonParser.parseReader(reader).getAsJsonObject(), resource);
                if (RtLookPackage.class.getResource(value.lmtResource()) == null) {
                    throw new IllegalStateException(resource + ": missing LMT " + value.lmtResource());
                }
                return value;
            }
        } catch (IOException | RuntimeException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private static JsonObject requiredObject(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonObject()) {
            throw new IllegalArgumentException("missing object " + name);
        }
        return value.getAsJsonObject();
    }

    private static String requiredString(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new IllegalArgumentException("missing string " + name);
        }
        return value.getAsString();
    }

    private static int requiredInt(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("missing integer " + name);
        }
        return value.getAsInt();
    }

    private static float requiredFinite(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("missing number " + name);
        }
        float result = value.getAsFloat();
        if (!Float.isFinite(result)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        return result;
    }

    private static float positive(JsonObject object, String name, String resource) {
        return positive(object, name, resource, "lighting");
    }

    private static float positive(JsonObject object, String name, String resource, String section) {
        float value = requiredFinite(object, name);
        if (value <= 0.0f) {
            throw new IllegalArgumentException(resource + ": " + section + "." + name
                    + " must be positive");
        }
        return value;
    }

    private static float nonNegative(JsonObject object, String name, String resource) {
        return nonNegative(object, name, resource, "lighting");
    }

    private static float nonNegative(JsonObject object, String name, String resource, String section) {
        float value = requiredFinite(object, name);
        if (value < 0.0f) {
            throw new IllegalArgumentException(resource + ": " + section + "." + name
                    + " must be non-negative");
        }
        return value;
    }

    private static void requireRange(float value, float min, float max, String resource, String name) {
        if (value < min || value > max) {
            throw new IllegalArgumentException(resource + ": " + name + " must be in ["
                    + min + "," + max + "]");
        }
    }

    private static void validateCurve(String spec, String resource) {
        String[] points = spec.split(",");
        if (points.length != 4) {
            throw new IllegalArgumentException(resource
                    + ": exposure.curve must contain exactly four sceneEv:compensationEv points");
        }
        float previousSceneEv = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < points.length; i++) {
            String[] pair = points[i].trim().split(":", -1);
            if (pair.length != 2) {
                throw new IllegalArgumentException(resource + ": exposure.curve point " + (i + 1)
                        + " is not sceneEv:compensationEv");
            }
            float sceneEv;
            float compensationEv;
            try {
                sceneEv = Float.parseFloat(pair[0].trim());
                compensationEv = Float.parseFloat(pair[1].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(resource + ": exposure.curve point " + (i + 1)
                        + " contains a non-number", e);
            }
            if (!Float.isFinite(sceneEv) || !Float.isFinite(compensationEv)) {
                throw new IllegalArgumentException(resource + ": exposure.curve point " + (i + 1)
                        + " must be finite");
            }
            if (sceneEv - previousSceneEv < 1.0e-4f) {
                throw new IllegalArgumentException(resource
                        + ": exposure.curve scene EV points must be strictly increasing");
            }
            previousSceneEv = sceneEv;
        }
    }

    public record Exposure(float minEv, float maxEv, String curve) {
    }

    /**
     * Bloom pyramid (see {@code RtBloomPipeline}). {@code radius} is the upsample tent radius in SOURCE
     * texels, so it needs no resolution scaling; {@code levels} is how many octaves of skirt the effect
     * reaches over, which is what sets its width.
     */
    public record Bloom(float strength, float thresholdSceneLinear, float softKneeFraction, float radius,
                        int levels) {
    }

    /**
     * Photometric anchors. {@code nightAirglowLuminanceCdM2} is airglow plus unresolved starlight—the
     * physical floor of a moonless night, approximately 1e-3 cd/m². Atmospheric multiple scattering is
     * evaluated separately.
     */
    public record Lighting(
            float sunIlluminanceLux,
            float moonIlluminanceLux,
            float blockEmissionLuminanceCdM2,
            float nightAirglowLuminanceCdM2,
            float starLuminanceCdM2,
            float moonPhaseFixedFraction) {
        public float moonPhaseFraction() {
            return 1.0f - moonPhaseFixedFraction;
        }
    }

    /**
     * Sky geometry. These were {@code caustica.rt.*} system properties, which left the shape of the sky
     * outside the versioned package that owns every other photometric decision; they belong with the
     * exposure curve, the LMT and the light anchors already authored here.
     */
    public record Sky(
            float sunNoonSouthTiltDegrees,
            /** Half-angle the NEE shadow ray samples about the body: sets penumbra softness only. */
            float sunAngularRadiusDegrees,
            float moonAngularRadiusDegrees,
            /** Half-angle the body is DRAWN at, matching vanilla's quads: atan(0.30) and atan(0.20). */
            float sunDiscHalfAngleDegrees,
            float moonDiscHalfAngleDegrees,
            float groundAlbedo,
            /**
             * Dip over which the atmosphere's ground fades in below the horizon. The surface is a real
             * discontinuity in the model — ~20 EV per degree of elevation at sea level under a high sun,
             * and up to ~50 at a low one — and Minecraft never shows the terrain that would justify it, so
             * without this the horizon reads as a hard grey line. Zero restores the hard ground.
             */
            float horizonSoftenDegrees) {
    }

    /**
     * The End. The flash carries two decoupled scales for the same reason the sun's drawn size and its
     * illuminance are decoupled: {@code flashIlluminanceLux} is the world lighting (NEE, tinted by the
     * dimension's SKY_LIGHT_COLOR), {@code flashQuadLuminanceCdM2} is how bright the drawn quad's white
     * point renders (the sprite's own radial falloff shapes everything under it). The ambient floor is
     * the isotropic irradiance the dimension's {@code ambient_light} scalar multiplies (the End ships
     * 0.25), tinted by SKY_LIGHT_COLOR + AMBIENT_LIGHT_COLOR; it is what keeps unlit End stone readable
     * between flashes. The gains scale the sampled end_sky box and the attribute-colour dome gradient
     * against the exposure the rest of the scene settles at.
     */
    public record End(
            float flashIlluminanceLux,
            float flashAngularRadiusDegrees,
            float flashQuadLuminanceCdM2,
            float ambientIlluminanceLux,
            float skyBoxGain,
            float skyGradientGain) {
    }
}
