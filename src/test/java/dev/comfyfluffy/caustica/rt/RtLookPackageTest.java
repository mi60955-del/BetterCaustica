package dev.comfyfluffy.caustica.rt;

import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class RtLookPackageTest {
    private static final String VALID = """
            {"schemaVersion":4,"id":"test","packageVersion":1,
             "exposure":{"minEv":-15,"maxEv":-2,"curve":"-2:-3,2:-2,8:0,15:1"},
             "lmt":{"resource":"lmt.bin"},
             "bloom":{"strength":0.08,"thresholdSceneLinear":1,"softKneeFraction":0.5,"radius":1,
             "levels":6},
             "lighting":{"sunIlluminanceLux":128000,"moonIlluminanceLux":5,
             "blockEmissionLuminanceCdM2":2000,"nightAirglowLuminanceCdM2":0.002,
             "starLuminanceCdM2":10,"moonPhaseFixedFraction":0.1},
             "sky":{"sunNoonSouthTiltDegrees":30,"sunAngularRadiusDegrees":0.6,
             "moonAngularRadiusDegrees":1.5,"sunDiscHalfAngleDegrees":16.7,
             "moonDiscHalfAngleDegrees":11.31,"groundAlbedo":0.1,"horizonSoftenDegrees":15},
             "end":{"flashIlluminanceLux":3,"flashAngularRadiusDegrees":20,
             "flashQuadLuminanceCdM2":25,"ambientIlluminanceLux":1,
             "skyBoxGain":1,"skyGradientGain":1}}
            """;

    @Test
    void acceptsACompleteCurrentSchemaPackage() {
        RtLookPackage look = parse(VALID);
        assertEquals(6, look.bloom().levels());
        assertEquals(30.0f, look.sky().sunNoonSouthTiltDegrees());
        assertEquals(0.1f, look.sky().groundAlbedo());
        assertEquals(3.0f, look.end().flashIlluminanceLux());
        assertEquals(1.0f, look.end().skyBoxGain());
    }

    @Test
    void rejectsUnknownSchema() {
        assertThrows(IllegalArgumentException.class, () -> parse(VALID.replace("\"schemaVersion\":4",
                "\"schemaVersion\":3")));
    }

    @Test
    void rejectsInvalidPhysicalRanges() {
        assertThrows(IllegalArgumentException.class, () -> parse(VALID.replace("\"minEv\":-15",
                "\"minEv\":2")));
        // A pyramid deeper than the bloom pipeline allocates would index past its descriptor sets.
        assertThrows(IllegalArgumentException.class, () -> parse(VALID.replace("\"levels\":6",
                "\"levels\":9")));
        assertThrows(IllegalArgumentException.class, () -> parse(VALID.replace("\"groundAlbedo\":0.1",
                "\"groundAlbedo\":1.5")));
        // A fade still running at the nadir leaves the lower hemisphere with no settled colour.
        assertThrows(IllegalArgumentException.class, () -> parse(
                VALID.replace("\"horizonSoftenDegrees\":15", "\"horizonSoftenDegrees\":120")));
        // Past a quarter turn the flash's NEE square would wrap the light's own horizon.
        assertThrows(IllegalArgumentException.class, () -> parse(
                VALID.replace("\"flashAngularRadiusDegrees\":20", "\"flashAngularRadiusDegrees\":60")));
    }

    @Test
    void rejectsAMissingSkySection() {
        assertThrows(IllegalArgumentException.class, () -> parse(VALID.replace("\"sky\":", "\"nope\":")));
    }

    @Test
    void rejectsAMissingEndSection() {
        assertThrows(IllegalArgumentException.class, () -> parse(VALID.replace("\"end\":", "\"nope\":")));
    }

    private static RtLookPackage parse(String json) {
        return RtLookPackage.parse(JsonParser.parseString(json).getAsJsonObject(),
                "/bettercaustica/color/looks/test/look.json");
    }
}
