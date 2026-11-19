package dev.comfyfluffy.caustica.mixin;

import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.particle.Particle;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

/**
 * Exposes the light coordinates a particle draws with and the state the level-lightmap baseline needs.
 * {@code SingleQuadParticle.extract} folds {@code getLightCoords} — where glowing particle types inject
 * their own block emission (see RtEntities.particleSelfEmission) — into the render state, and the
 * position/level fields are protected, so the injected-emission margin cannot be computed without these.
 */
@Mixin(Particle.class)
public interface ParticleAccessor {
    @Invoker("getLightCoords")
    int caustica$getLightCoords(float partialTick);

    @Accessor("level")
    ClientLevel caustica$getLevel();

    @Accessor("x")
    double caustica$getX();

    @Accessor("y")
    double caustica$getY();

    @Accessor("z")
    double caustica$getZ();
}
