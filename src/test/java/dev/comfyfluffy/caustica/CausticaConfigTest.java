package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
}
