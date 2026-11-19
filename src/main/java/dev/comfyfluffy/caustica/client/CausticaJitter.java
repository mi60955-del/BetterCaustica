package dev.comfyfluffy.caustica.client;

/**
 * Sub-pixel camera jitter for DLSS Ray Reconstruction, and for a still run that has no reconstructor.
 *
 * <p>Generates a Halton(2,3) low-discrepancy sequence in render-pixel space, with the DLSS phase-count
 * rule {@code ceil(8 * (display/render)^2)} and RR's recommended floor of 32 phases. A still run with RR
 * off passes a larger floor because nothing else constrains the sequence, and 32 phases repeat four times
 * across a 128-sample exposure. {@link dev.comfyfluffy.caustica.rt.RtComposite} reads the per-frame offset,
 * applies it to the primary ray in the path-tracing shader, and reports it to DLSS-RR's evaluate when RR is
 * the one reconstructing.
 */
public final class CausticaJitter {
	public static final CausticaJitter INSTANCE = new CausticaJitter();

	private int frameIndex;
	private float pixelsX;
	private float pixelsY;

	private CausticaJitter() {
	}

	/**
	 * Advance one frame. Call once per frame before the level projection is built. {@code floorPhases}
	 * raises the phase count for a caller that has no reconstructor to satisfy; it never lowers the DLSS
	 * rule's count.
	 */
	public void prepare(int renderWidth, int renderHeight, int displayWidth, int floorPhases) {
		int phaseCount = Math.max(floorPhases, jitterPhaseCount(renderWidth, displayWidth));
		// floorMod keeps the index in range for the life of the process: a plain % would go negative
		// once the frame counter wraps, and Halton(<=0) is degenerate.
		int index = Math.floorMod(this.frameIndex++, phaseCount) + 1;
		this.pixelsX = halton(index, 2) - 0.5f;
		this.pixelsY = halton(index, 3) - 0.5f;
	}

	/** Jitter offset in render-pixel space, applied to the primary ray and reported to RR evaluate. */
	public float jitterPixelsX() {
		return this.pixelsX;
	}

	public float jitterPixelsY() {
		return this.pixelsY;
	}

	private static int jitterPhaseCount(int renderWidth, int displayWidth) {
		float ratio = (float) displayWidth / Math.max(1, renderWidth);
		return Math.max(32, (int) Math.ceil(8.0f * ratio * ratio));
	}

	private static float halton(int index, int base) {
		float f = 1.0f;
		float result = 0.0f;
		int i = index;
		while (i > 0) {
			f /= base;
			result += f * (i % base);
			i /= base;
		}
		return result;
	}
}
