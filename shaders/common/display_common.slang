// Push-constant structs shared between the display-chain compute passes and
// layout/layout_probe.slang's build-time reflection probe (see GenerateShaderRecords.groovy).
// Each struct is imported into exactly one production shader AND probed here so its Java writer is
// generated from the compiler's reflected offsets, preventing Java/shader ABI drift when fields move.
//
// Each struct here needs a distinct name because reflection JSON parameters are named by struct type across the whole
// probe module, and per-entry-point SPIR-V allows only one push-constant block reachable from any
// single entry point, so the probe needs one small dedicated entry point per struct.

public struct ExposureHistPush {
    public uint stride;
    public float centerWeightSigma;
    public float centerWeightFloor;
};

// The lens's own autofocus meter. It reads the centre window of the depth the frame just traced, so
// everything here is in that image's own texels and in blocks.
public struct DofFocusPush {
    public int   originX;      // top-left texel of the window
    public int   originY;
    public int   extentX;      // window size in texels
    public int   extentY;
    public float nearBlock;    // the closest distance a tap may answer: the projection's near plane
    public float farBlock;     // the far stop. Sky and everything past it answer this
};

public struct ExposureResolvePush {
    public float key;
    public float minEv;
    public float maxEv;
    // Adaptation time constants, in seconds, applied in EV space (see exposure_resolve's smoothing
    // step). Named for what the SCENE did, not for which way the exposure multiplier moved: a darkening
    // scene needs MORE exposure, so adaptDarken is the branch where target > prev.
    public float adaptDarken;
    public float adaptBrighten;
    public float frameTimeSeconds;
    public float evBias;
    public float lowPercentile;
    public float highPercentile;
    public float skyWeightCap;
    public float curveScene0;
    public float curveCompensation0;
    public float curveScene1;
    public float curveCompensation1;
    public float curveScene2;
    public float curveCompensation2;
    public float curveScene3;
    public float curveCompensation3;
    public float emissiveWeightCap;
    // evOffset takes log2(metered stored luminance) to EV100:
    // RtSceneUnits.EV100_OFFSET - log2(preExposure). preExposure is the scalar raygen already
    // multiplied into the stored radiance (1.0 when pre-exposure is disabled), so the metered
    // buffer holds L*preExposure and the resolve divides it back out to recover absolute scene EV --
    // and to keep minEv/maxEv bounding ABSOLUTE exposure rather than the residual.
    public float evOffset;
    public float preExposure;
    // resetSeq invalidates GPU exposure history without a host write or fence.
    public uint resetSeq;
};

// Persistent auto-exposure controller state. This shared definition is also reflected into
// ExposureStateData.java so host reads and initialization follow Slang's Std430 layout.
public struct ExposureState {
    public float previous;
    public uint initialized;
    public float evScene;
    public float evTarget;
    public float evApplied;
    public float clipLowFrac;
    public float clipHighFrac;
    public uint resetSeq;
    public float meteringSkyScale;
    public float meteringSkyFrac;
    public float curveCompensation;
    public float effectiveSlope;
    public float meteringEmissiveScale;
    public float meteringEmissiveFrac;
};

public struct DisplayPush {
    public int hdrEnabled;      // 0 = SDR only, 1 = also write the PQ HDR image
    public float lutSize;       // toneLut/hdrToneLut texels per axis; both LUTs share one size (asserted at load)
    public float gamma;         // post-display artistic gamma; below 1 brightens midtones
    public float hdrPeakNits;   // mastering peak baked into the currently bound HDR ACES LUT
    public int lookEnabled;     // 0 = identity, 1 = apply the scene-referred ACES look LUT
    public float lookLutSize;   // lookLut texels per axis (independent of the output-transform LUT size)
    // Scene-referred blurred highlight signal added before the LMT. The bloom pyramid's level 0 holds the
    // SUM of every band, so RtComposite folds the 1/levelCount normalisation into this value: the authored
    // look-package strength then means the same thing whichever level count the resolution supports.
    public float bloomStrength;
    // 0 = the default grading: the look package's scene-referred LMT followed by the baked ACES 2.0
    // output transform. 1 = the advanced grading: AgX on the exposed image, with no look package.
    public int advancedGrading;
    // AgX incident-exposure gain: the scale applied to scene-linear ACEScg before the log2 encoding step.
    // It has to replace the exposure offset the authored LMT supplies, so it is paired with the
    // auto-exposure key/EV-compensation curve rather than being a general brightness control.
    public float agxGain;
    // Chroma scale about BT.709 luminance, applied to AgX's linear output. AgX spends its range on the
    // highlight shoulder, so it lands well below the authored look's saturation on a photometric scene.
    public float agxSaturation;
    // Contrast blend on AgX's log-encoded value: 0 replaces the published contrast curve with the
    // identity (a flat, low-contrast render), 1 is AgX's own curve, and above 1 extrapolates the
    // deviation for a punchier render. Read per frame.
    public float agxContrast;
    // Width in stops of AgX's log2 encoding window around mid grey; the published transform uses 14.
    // A wider window spends more of the curve on the highlight shoulder, softening the rolloff and
    // lowering contrast; a narrower one compresses the same scene into fewer stops. Read per frame.
    public float agxRange;
    // 0 = leave the image alone; 1 = apply the Purkinje (dark-adaptation) brightness shift to the
    // exposed scene-linear image, before either display transform. Read per frame, so it is a live
    // toggle and rebuilds nothing.
    public int purkinjeEnabled;
    // Scale on the shift's per-pixel brightness deviation: 0 leaves every pixel untouched, 1 is the
    // CIE mesopic response itself, and above 1 deepens the shift. See purkinjeGain() in the display
    // pass.
    public float purkinjeStrength;
    // Master scale on the lens flare image. 0 skips the fetch entirely instead of multiplying it: the
    // flare image is only written by the lens pass, so with the effect off its contents are whatever was
    // last there (or uninitialised), and a multiply would not be enough to keep that out of the frame.
    public float lensStrength;
};

// One dispatch of the lens chain, reduced from the full-resolution exposed image to the field of light
// that overflows the lens. The threshold and the ceiling it applies are shader constants because they
// describe the lens, not the look; only the box's size has to come from the host, because it is the one
// part of the reduction expressed in display pixels.
public struct LensSourcePush {
    // 1 / display size in pixels. The box the pass averages is expressed in display pixels so its
    // footprint does not change with the texture resolution.
    public float2 displayTexel;
};

// The flare gather: diffraction spikes, the aperture ring, and the dust field, summed into one image.
// The three gains are the authored strengths; the shape constants live in the shader and in
// tools/lens_preview.py, which test_lens_preview.py keeps in step.
public struct LensFlarePush {
    public float starburst;     // gain on the spikes; 0 skips the gather
    public float halo;          // gain on the aperture ring; 0 skips the gather
    public float dirt;          // gain on the dust field; 0 skips the procedural field
    public int spikes;          // spike count, also the rotational symmetry of the star
    public int samples;         // taps per spike
    public int haloTaps;        // taps around the ring, per colour channel
    public float spikeLength;   // spike length as a fraction of the display height
    public float haloRadius;    // ring radius as a fraction of the display height
    public float dispersion;    // ring radius spread between channels, i.e. the chromatic fringe
    public float dirtScale;     // grain frequency scale: above 1 makes the dust finer
};

public struct BloomPush {
    public int mode;            // 0 prefilter+downsample, 1 downsample, 2 tent upsample (accumulate)
    public float threshold;     // exposed scene-linear ACEScg luminance
    public float softKnee;      // absolute exposed scene-linear knee width
    public float radius;        // upsample tent radius in SOURCE texels; resolution-independent
};

public struct DebugPresentPush {
    public uint debugView;     // 1..9; this pass is only dispatched for debugView != 0
    public float centerWeightSigma;
    public float centerWeightFloor;
};

public struct SharpenPush {
    public int hdrEnabled;     // 0 = sharpen the SDR image only, 1 = also the PQ HDR image
    // Detail gain on the 3x3 high pass: 0 reproduces the input, 1 adds the whole high-pass term back.
    // One gain serves both images because sRGB and PQ both encode into the same [0,1] code-value range.
    public float amount;
};

public struct StillPush {
    // 0 = add this frame into the accumulator, 1 = store it as the run's first frame, 2 = divide the
    // accumulator out into the 8-bit image. One dispatch carries both halves so the accumulator's layout
    // has exactly one owner.
    public int mode;
    // 1 / the run's sample count. Only mode 2 reads it.
    public float weight;
};

// One dispatch of the post-process depth of field's gather: the near and far accumulation for both
// display channels, from the display mapper's compose pair. The lens here is a bookkeeping lens -- the
// frame was already traced with a pinhole camera, and this pass defocuses its result -- so the fields
// are the four numbers the circle-of-confusion math needs, not a camera state.
public struct DofPostGatherPush {
    public int hdrEnabled;      // 0 = fill the SDR pair only, 1 = also the PQ HDR pair
    // The lens radius in blocks and the focal plane's axial distance in blocks: a surface at range s
    // spreads over aperture * |1/s - 1/f| of the frame's tangent span, which is the same thin-lens
    // answer the traced lens walks its rays with.
    public float aperture;
    public float focus;
    // The tangent span one display pixel covers, 2 / (|m11| * displayHeight). Dividing the tangent-space
    // blur by it turns every radius into display pixels, the unit the kernel and the cap work in.
    public float tangentPerPixel;
    // The widest circle of confusion, in display pixels of radius. The gather kernel is bounded by it:
    // a post-process pass has no temporal accumulation to lean on, so a disc wider than the kernel can
    // fill comes back under-sampled instead of coming back as noise.
    public float maxCoc;
    // The aperture's blade count, 0 = a circle, and its rotation in radians. Blades reshape the bokeh
    // highlights; the coverage function renormalises the polygon to the circle's area, so the blades
    // change the shape of the defocus without changing its brightness.
    public int blades;
    public float bladeRotation;
    // focusDepth's texel count (render resolution) and the dispatch's own (half the display
    // resolution). The depth lookup walks from this pass's half-res coordinates to the render-res
    // depth image through both.
    public float2 renderTexels;
    public float2 halfTexels;
};

// The post-process depth of field's recombine: scales the gather's half-res images back up over the
// compose pair, in place. It needs no lens state -- the gather already folded each pixel's defocus
// into the alpha it stores -- only the sizes that walk between the two resolutions.
public struct DofPostRecombinePush {
    public int hdrEnabled;      // 0 = compose the SDR image only, 1 = also the PQ HDR image
    public float2 halfTexels;   // the gather's dispatch size, half the display resolution
};
