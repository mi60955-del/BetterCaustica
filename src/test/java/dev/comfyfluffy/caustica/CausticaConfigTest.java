package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class CausticaConfigTest {
    @Test
    void invalidPeakNitsFallsBackToDefault() {
        CausticaConfig.IntSetting setting = CausticaConfig.Rt.Hdr.PEAK_NITS;
        int previous = setting.value();
        try {
            setting.set(2000);
            assertEquals(2000, setting.value());

            setting.set(900);
            assertEquals(1000, setting.value());
        } finally {
            setting.set(previous);
        }
    }

    @Test
    void fogDayFactorFollowsTheDecaySlider() {
        // Twilight owns the peak regardless of the slider.
        assertEquals(1.0f, CausticaConfig.Rt.Volume.fogDayFactor(0.0f, 0.8f), 1e-4f);
        // The shipped default: noon keeps 20% of the peak, midnight 25%.
        assertEquals(0.20f, CausticaConfig.Rt.Volume.fogDayFactor(1.0f, 0.8f), 1e-4f);
        assertEquals(0.25f, CausticaConfig.Rt.Volume.fogDayFactor(-1.0f, 0.8f), 1e-4f);
        // No decay holds the density flat around the clock.
        assertEquals(1.0f, CausticaConfig.Rt.Volume.fogDayFactor(1.0f, 0.0f), 1e-4f);
        assertEquals(1.0f, CausticaConfig.Rt.Volume.fogDayFactor(-1.0f, 0.0f), 1e-4f);
        // Full decay leaves fog only around sunrise and sunset.
        assertEquals(0.0f, CausticaConfig.Rt.Volume.fogDayFactor(1.0f, 1.0f), 1e-4f);
        assertEquals(0.0f, CausticaConfig.Rt.Volume.fogDayFactor(-1.0f, 1.0f), 1e-4f);
    }

    @Test
    void boundaryExtinctionRampsDownToTheBandEdge() {
        float ramp = 96f;
        float peak = 0.06f;
        // Full density at the wall — and outside the border, where the distance goes negative, it stays
        // there rather than running the ramp backwards.
        assertEquals(peak, CausticaConfig.Rt.Volume.boundaryExtinction(0.0, ramp, peak), 0.0f);
        assertEquals(peak, CausticaConfig.Rt.Volume.boundaryExtinction(-12.0, ramp, peak), 0.0f);
        // Nothing at or past the band's edge. The ordinary case is a default overworld border about 30
        // million blocks out, which must land on exactly zero and not on a rounding residue.
        assertEquals(0.0f, CausticaConfig.Rt.Volume.boundaryExtinction(ramp, ramp, peak), 0.0f);
        assertEquals(0.0f, CausticaConfig.Rt.Volume.boundaryExtinction(2.99e7, ramp, peak), 0.0f);
        // Every step toward the border adds density, and the square keeps the band's outer half below a
        // quarter of the peak: the veil announces itself late instead of dimming the whole approach.
        float previous = 0.0f;
        for (double d = ramp; d >= 0.0; d -= 3.0) {
            float sigma = CausticaConfig.Rt.Volume.boundaryExtinction(d, ramp, peak);
            assertTrue(sigma >= previous, "density must grow as the border gets closer at d=" + d);
            previous = sigma;
        }
        assertEquals(peak * 0.25f, CausticaConfig.Rt.Volume.boundaryExtinction(ramp * 0.5, ramp, peak),
                1.0e-6f);
        // Either dial at zero zeroes the feature, including the ramp: no division by the band width
        // reaches the shader as an infinite density.
        assertEquals(0.0f, CausticaConfig.Rt.Volume.boundaryExtinction(0.0, 0.0f, peak), 0.0f);
        assertEquals(0.0f, CausticaConfig.Rt.Volume.boundaryExtinction(0.0, ramp, 0.0f), 0.0f);
    }
}
