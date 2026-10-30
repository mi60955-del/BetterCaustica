"""Air -> glass -> partial block, all flush: does the guide trace see the block?

Reproduces the mesher/hit-shader geometry numerically instead of rendering it:

  * a translucent terrain face is recessed into its own block by TRANSLUCENT_INSET;
  * the transmitted ray restarts INSET_TRANSMIT_BIAS into the air gap it leaves;
  * the radiance trace uses tmin 0, while the guide probe used RAY_TMIN.

With the partial block flush against the glass, its face sits ~1e-4 from the ray origin.
RAY_TMIN (1e-3) skips it, so the guide describes the surface behind while the color is the
partial block. DLSS-RR demodulates the color with that wrong surface and turns the block
into a translucent ghost. Full blocks are immune because vanilla culls the glass face that
touches them, so no glass gap exists in front of them.

Run:  python tools/rr-refraction-probe/flush_stack_probe.py
"""

import numpy as np

TRANSLUCENT_INSET = 2.0e-4   # RtTerrainMesher: translucent face recessed into its block
INSET_TRANSMIT_BIAS = 1.0e-4 # world_core.slang: transmitted-ray origin offset
RAY_TMIN = 1.0e-3            # world_core.slang: old guide-probe tmin
BLOCK = 1.0

# Camera in air looks along +x. Glass occupies [0, 1]; its exit face at x = 1 is recessed
# into the glass by TRANSLUCENT_INSET. The partial block [1, 2] keeps its face at x = 1
# (flush against the glass), e.g. a slab/stair/block-entity side. A wall sits behind at x=4.
glass_exit_face = BLOCK - TRANSLUCENT_INSET
origin = glass_exit_face + INSET_TRANSMIT_BIAS      # offset toward the outgoing (+x) side
partial_face = BLOCK
wall_face = 4.0

partial_t = partial_face - origin
wall_t = wall_face - origin


def first_hit(tmin):
    """Nearest surface along +x with t >= tmin: (name, distance)."""
    hits = [("partial-block", partial_t), ("wall", wall_t)]
    for name, t in hits:
        if t >= tmin:
            return name, t
    return "sky", float("inf")


def report():
    print(f"glass exit face      x = {glass_exit_face:.6f}")
    print(f"transmitted origin   x = {origin:.6f}  (+{INSET_TRANSMIT_BIAS:g} into the gap)")
    print(f"partial block face   x = {partial_face:.6f}  -> t = {partial_t:.6f}")
    print(f"wall face            x = {wall_face:.6f}  -> t = {wall_t:.6f}\n")

    rows = []
    for label, tmin in [
        ("radiance trace (tmin 0)", 0.0),
        (f"guide trace, old (tmin {RAY_TMIN:g})", RAY_TMIN),
        ("guide trace, fixed (tmin 0)", 0.0),
    ]:
        name, t = first_hit(tmin)
        rows.append((label, name, t))
        print(f"{label:30s} -> {name:14s} at t = {t:.6f}")

    radiance = first_hit(0.0)[0]
    old_guide = first_hit(RAY_TMIN)[0]
    fixed_guide = first_hit(0.0)[0]
    assert radiance == "partial-block"
    assert old_guide != radiance, "old tmin was supposed to skip the flush face"
    assert fixed_guide == radiance, "fixed guide must see the same surface as radiance"

    # DLSS-RR demodulates this pixel by the guide's albedo and re-modulates by it. The visible
    # surface is the partial block; the old guide names the wall behind it. The block's texel
    # therefore enters and leaves the denoiser at the wrong albedo, and its shading is averaged
    # against neighbours that are actually the wall -- the block reads as a translucent ghost.
    block_albedo = 0.55          # e.g. the white top of a quartz stair
    wall_albedo = 0.20
    guide_old = wall_albedo
    guide_fixed = block_albedo
    print(f"\nvisible surface albedo   {block_albedo:.2f} (partial block)")
    print(f"old guide albedo         {guide_old:.2f} (wall behind)  <- mismatch")
    print(f"fixed guide albedo       {guide_fixed:.2f} (partial block)  <- match")
    assert guide_old != block_albedo and guide_fixed == block_albedo

    print("\nPASS: the old guide probe skipped the flush partial block; tmin 0 resolves it "
          "exactly as the radiance ray does.")


if __name__ == "__main__":
    report()
