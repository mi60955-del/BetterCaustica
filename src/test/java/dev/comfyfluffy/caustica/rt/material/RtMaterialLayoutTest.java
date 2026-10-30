package dev.comfyfluffy.caustica.rt.material;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.gen.MaterialHeaderData;
import dev.comfyfluffy.caustica.rt.gen.MaterialHeaderData.Float4;
import dev.comfyfluffy.caustica.rt.gen.WorldPushConstantsData;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtMaterialLayoutTest {
    @Test
    void reflectedMaterialHeaderMatchesHotAbi() {
        assertEquals(80, MaterialHeaderData.BYTE_SIZE);
        ByteBuffer data = ByteBuffer.allocateDirect(MaterialHeaderData.BYTE_SIZE)
                .order(ByteOrder.nativeOrder());
        new MaterialHeaderData(3, 5, 7, 11,
                new Float4(0.01f, 0.02f, 0.03f, 0.04f),
                new Float4(0.05f, 0.06f, 7.0f, 8.0f),
                new Float4(0.1f, 0.2f, 1.52f, 1.0f),
                new Float4(0.3f, 0.4f, 0.5f, 0.6f)).write(data);
        assertEquals(3, data.getInt(0));
        assertEquals(5, data.getInt(4));
        assertEquals(7, data.getInt(8));
        assertEquals(11, data.getInt(12));
        assertEquals(0.01f, data.getFloat(16));
        assertEquals(7.0f, data.getFloat(40));
        assertEquals(0.1f, data.getFloat(48));
        assertEquals(1.52f, data.getFloat(56));
        assertEquals(0.6f, data.getFloat(76));
    }

    @Test
    void reflectedWorldPushConstantsIncludeLightBuffersFrameIndexAndMaterialScalars() {
        // 11 uint64_t addresses (world/table/material, 6 light buffers, path queue), frameIndex, and the
        // frame-constant scalars the material path reads: surface wetness, the parallax depth cap, the
        // parallax layer budget, the packed material switches, and the parallax style. The last one took
        // the trailing padding, so the block stays at 112 and still lands on its 8-byte alignment.
        assertEquals(112, WorldPushConstantsData.BYTE_SIZE);
        ByteBuffer data = ByteBuffer.allocateDirect(WorldPushConstantsData.BYTE_SIZE)
                .order(ByteOrder.nativeOrder());
        new WorldPushConstantsData(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12, 0.25f, 0.5f, 13, 15,
                1).write(data);
        assertEquals(4L, data.getLong(24));  // materialTableAddr
        assertEquals(5L, data.getLong(32));  // lightBufAddr
        assertEquals(9L, data.getLong(64));  // lightGridSpanAddr (last of the grid addresses)
        assertEquals(10L, data.getLong(72)); // lightTranslationAddr
        assertEquals(11L, data.getLong(80)); // pathQueueAddr
        assertEquals(12, data.getInt(88));     // frameIndex
        assertEquals(0.25f, data.getFloat(92)); // wetness
        assertEquals(0.5f, data.getFloat(96));  // parallaxDepth
        assertEquals(13, data.getInt(100));     // parallaxLayers
        assertEquals(15, data.getInt(104));     // materialToggles
        assertEquals(1, data.getInt(108));      // parallaxStyle
    }

    /**
     * The shader owns what each materialToggles bit means (MATERIAL_TOGGLE_* in world_common.slang) and
     * {@link CausticaConfig.Rt.Materials} mirrors it when it builds the mask. Nothing else compares the two,
     * and a drifting bit would silently gate the wrong channel instead of failing, so the shader source is
     * the oracle here.
     */
    @Test
    void materialToggleBitsMatchTheShader() throws IOException {
        String source = worldCommonSource();
        assertEquals(CausticaConfig.Rt.Materials.TOGGLE_TEXTURE_EMISSION,
                shaderBit(source, "MATERIAL_TOGGLE_TEXTURE_EMISSION"));
        assertEquals(CausticaConfig.Rt.Materials.TOGGLE_BLOCK_EMISSION,
                shaderBit(source, "MATERIAL_TOGGLE_BLOCK_EMISSION"));
        assertEquals(CausticaConfig.Rt.Materials.TOGGLE_SUBSURFACE,
                shaderBit(source, "MATERIAL_TOGGLE_SUBSURFACE"));
        assertEquals(CausticaConfig.Rt.Materials.TOGGLE_POROSITY,
                shaderBit(source, "MATERIAL_TOGGLE_POROSITY"));
        assertEquals(CausticaConfig.Rt.Materials.TOGGLE_ENTITY_PARALLAX,
                shaderBit(source, "MATERIAL_TOGGLE_ENTITY_PARALLAX"));
        assertEquals(CausticaConfig.Rt.Materials.TOGGLE_PARALLAX_SHADOW,
                shaderBit(source, "MATERIAL_TOGGLE_PARALLAX_SHADOW"));
    }

    /**
     * The two parallax styles are an enum the shader switches on (PARALLAX_STYLE_* in world_common.slang)
     * and {@link CausticaConfig.Rt.Materials} mirrors when it writes pc.parallaxStyle. A value that drifted
     * would silently select the other style rather than fail, so the shader source is the oracle here too.
     */
    @Test
    void parallaxStyleValuesMatchTheShader() throws IOException {
        String source = worldCommonSource();
        assertEquals(CausticaConfig.Rt.Materials.PARALLAX_STYLE_SMOOTH,
                shaderInt(source, "PARALLAX_STYLE_SMOOTH"));
        assertEquals(CausticaConfig.Rt.Materials.PARALLAX_STYLE_SHARP,
                shaderInt(source, "PARALLAX_STYLE_SHARP"));
    }

    private static String worldCommonSource() throws IOException {
        for (String candidate : new String[]{
                "shaders/pipelines/world/world_common.slang",
                "../shaders/pipelines/world/world_common.slang"}) {
            if (Files.isRegularFile(Path.of(candidate))) {
                return Files.readString(Path.of(candidate));
            }
        }
        assertNotNull(null, "world_common.slang not found from " + Path.of("").toAbsolutePath());
        throw new IllegalStateException();
    }

    private static int shaderBit(String source, String name) {
        Matcher matcher = Pattern.compile(name + "\\s*=\\s*(\\d+)u").matcher(source);
        assertTrue(matcher.find(), name + " is not defined in world_common.slang");
        return Integer.parseInt(matcher.group(1));
    }

    private static int shaderInt(String source, String name) {
        Matcher matcher = Pattern.compile(name + "\\s*=\\s*(-?\\d+)").matcher(source);
        assertTrue(matcher.find(), name + " is not defined in world_common.slang");
        return Integer.parseInt(matcher.group(1));
    }
}
