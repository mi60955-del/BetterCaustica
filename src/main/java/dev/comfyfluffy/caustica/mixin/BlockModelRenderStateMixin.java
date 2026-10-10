package dev.comfyfluffy.caustica.mixin;

import com.mojang.blaze3d.vertex.PoseStack;
import dev.comfyfluffy.caustica.rt.entity.ContainedBlockSource;
import dev.comfyfluffy.caustica.rt.entity.RtEntityCollector;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.block.BlockModelRenderState;
import net.minecraft.client.renderer.special.SpecialModelRenderer;
import net.minecraft.util.LightCoordsUtil;
import net.minecraft.world.level.block.state.BlockState;
import org.joml.Matrix4fc;
import org.jspecify.annotations.Nullable;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Lets the RT entity capture pick up a contained block-model display (the sulfur cube's swallowed block,
 * item-frame map blocks, …). These resolve through {@code BlockModelResolver} into the <em>display</em>
 * block-model set, which can wrap a block in a {@code SpecialBlockModelWrapper} (special renderer) — a path
 * {@link RtEntityCollector} doesn't capture, so the block submitted nothing. We record the resolved
 * blockState ({@link ContainedBlockSource}, set by {@code BlockModelResolverMixin}); then, only when {@code
 * submit} is driven by our collector (i.e. an RT entity capture), we re-mesh that block from the world
 * model set through the collector and cancel the original (broken) submit. Normal rendering is untouched —
 * the collector check fails for the vanilla submit collector.
 */
@Mixin(BlockModelRenderState.class)
public abstract class BlockModelRenderStateMixin implements ContainedBlockSource {
    @Shadow @Nullable private Matrix4fc transformation;
    @Shadow @Nullable private Matrix4fc specialRendererTransformation;
    @Shadow @Nullable private SpecialModelRenderer<?> specialRenderer;
    @Shadow public int blockLightCoords;

    @Unique private BlockState caustica$containedState;

    @Override
    public @Nullable BlockState caustica$containedBlock() {
        return this.caustica$containedState;
    }

    @Override
    public void caustica$setContainedBlock(@Nullable BlockState state) {
        this.caustica$containedState = state;
    }

    @Inject(method = "clear", at = @At("HEAD"))
    private void caustica$clearContained(CallbackInfo ci) {
        this.caustica$containedState = null;
    }

    @Inject(method = "submit", at = @At("HEAD"), cancellable = true)
    private void caustica$captureForRt(PoseStack poseStack, SubmitNodeCollector submitNodeCollector,
                                       int externalLightCoords, int overlayCoords, int outlineColor, CallbackInfo ci) {
        if (this.caustica$containedState == null || !(submitNodeCollector instanceof RtEntityCollector rt)) {
            return;
        }
        // Mirror vanilla's submit(), which a composite display model populates on both paths: the
        // world-model mesh always goes out under the state-model wrapper's transform, and the special
        // renderer, when the resolve set one, is submitted on top under its own transform. Choosing
        // either-or would silently drop the special-renderer half of a composite.
        rt.captureBlockState(this.caustica$containedState, this.transformation, poseStack);
        if (this.specialRenderer != null) {
            poseStack.pushPose();
            if (this.specialRendererTransformation != null) {
                poseStack.mulPose(this.specialRendererTransformation);
            }
            try {
                this.specialRenderer.submit(null, poseStack, rt,
                        LightCoordsUtil.max(externalLightCoords, this.blockLightCoords),
                        overlayCoords, false, outlineColor);
            } finally {
                poseStack.popPose();
            }
        }
        ci.cancel();
    }
}
