#!/usr/bin/env python3
"""Bakes the scotopic (rod) luminance basis behind the renderer's Purkinje shift.

The display pass re-weights each pixel's brightness by the CIE mesopic response, which needs two
linear functionals on the stored ACEScg triple: the photopic one the renderer already uses
(``ACESCG_LUMA`` in the shaders) and a scotopic one derived here:

    V'(lambda) ~= c . [x_bar, y_bar, z_bar](lambda)      (least squares over 380-780 nm)
    SCOTOPIC_LUMA = c . (AP1/D60 -> XYZ rows)

Both CIE functions are peak-normalised, so the composed functional returns integral(Phi*V') in the
same units that ``ACESCG_LUMA`` returns integral(Phi*V), and the absolute maxima (683 vs 1700.13
lm/W) cancel: only their RATIO is ever consumed.

WHAT THIS MODEL CANNOT DO, and why the error table below is printed on every run: V'(lambda) is not
in the span of the CIE 1931 colour-matching functions. Rod response is not determined by photopic
colourimetry, so two metameric spectra of identical RGB can differ in scotopic terms by many percent
and nothing in the renderer's colour triple says which is which. Consequences that are handled in
the shader rather than hidden:

  * The fit's red coefficient is negative, i.e. the model extrapolates deep red to a negative rod
    response. Rods genuinely do not see saturated red, so the shader clamps the predicted scotopic
    response at zero; a clamped pixel loses all of its rod signal, and that collapse IS the effect.
  * A non-negativity-constrained fit was tried and rejected: positivity can only be bought by giving
    the red primary ~0.9 of its photopic rod response, which removes the collapse above.
  * Saturated blue-green is under-boosted (the fit cannot reach the 440 nm ratio), so cool highlights
    shift less than physiology says they should. Broadband scene colours are the accurate case.

The table is the shader's own gain formula applied to test spectra, alongside a 'no effect' control
arm (rod response == photopic everywhere) so the reported errors mean something: a fit that merely
predicted no Purkinje shift would score well on an absolute-error metric.

Prints the Slang constant block for ``shaders/pipelines/display/main.comp.slang`` plus the
diagnostics that justify it. Paste the printed numbers; nothing at runtime reads this script.

Data provenance (vendored under tools/data/cie/, SHA-256 verified against each metadata file):
  CIE 1931 colour-matching functions, 2 deg observer, 1 nm, CIE 018:2019 Table 6
      https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv  DOI 10.25039/CIE.DS.xvudnb9b
  CIE spectral luminous efficiency for scotopic vision V'(lambda), 1 nm, CIE 018:2019 Table 2
      https://files.cie.co.at/Publications-datasets/CIE_sle_scotopic.csv  DOI 10.25039/CIE.DS.gr6w4b5g
Both CC BY-SA 4.0, International Commission on Illumination (CIE), Vienna.

The mesopic adaptation coefficient m (the photopic weight of the blend) and its limits come from CIE
191:2010 as reproduced in BIPM, 'Principles Governing Photometry' 2nd ed. (Metrologia 56, 2019),
eqs (1) and (11):

    V_mes;m(lambda) = [ m V(lambda) + (1 - m) V'(lambda) ] / M(m)
    m = a + b * log10(L_mes),  a = 0.7670, b = 0.3334, clamped to [0, 1]

M(m), the normalisation that keeps the blend peak-normalised, cancels between a pixel and the
adaptation field, so it is not baked. The script checks the constants against the standard's own
limits: m must reach 1 at 5 cd/m**2 and 0 at 0.005 cd/m**2.

Requires: numpy (tested with 2.5). Python 3.14.

Usage:
    python tools/bake_scotopic_basis.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "cie"

# CIE 1951 V'(lambda) is tabulated 380-780 nm and the CMFs run wider (360-830 nm). The fit uses the
# overlap; it is insensitive to the window (restricting it to 430-660 nm moves the coefficients by
# under 0.5%), so the whole published range is used rather than a tuned one.
FIT_LO_NM, FIT_HI_NM = 380, 780

# Evaluation sources. Monochromatic light below 440 nm or above 660 nm carries so little of either
# response that a tristimulus model cannot be judged on it, so the sweep stops there.
MONO_LO_NM, MONO_HI_NM, MONO_STEP_NM = 440, 660, 10.0

# AP1 primaries and the ACES white point, SMPTE ST 2065-3 (ACEScg) / ST 2065-1 (ACES), as published
# at https://docs.acescentral.com/encodings/acescg/ . Blue's y is +0.044, which sits below the
# spectral locus at that x (imaginary primary, deep violet): the Y-row check in main() is what pins
# the sign, because a negative y there gives a completely different luminance row.
AP1_XY = np.array([[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]], dtype=np.float64)
ACES_WHITE_XY = np.array([0.32168, 0.33767], dtype=np.float64)

# The photopic basis the running shaders already meter and grade with, from ACESCG_LUMA in
# shaders/pipelines/world/*.slang and shaders/pipelines/debug_present/main.comp.slang. The derived
# matrix's Y row must reproduce it: that is the independent check that the AP1 chromaticities above
# describe the same space the renderer stores radiance in.
EXPECTED_PHOTOPIC_LUMA = np.array([0.27222872, 0.67408177, 0.05368952], dtype=np.float64)

# BIPM G1-2019 eq (11), from CIE 191:2010: photopic weight of the mesopic blend.
MESOPIC_A, MESOPIC_B = 0.7670, 0.3334
MESOPIC_PHOTOPTIC_LIMIT = 5.0  # cd/m**2, above which vision is photopic (m = 1)
MESOPIC_SCOTOPIC_LIMIT = 0.005  # cd/m**2, below which vision is scotopic (m = 0)
# Adaptation levels the error table is reported at: deep twilight, moonless night, and the scotopic
# limit. Each m corresponds to log10(L) = (m - a) / b.
EVALUATION_M = (0.5, 0.2, 0.0)

PLANCK_H = 6.62607015e-34  # J s, exact (SI 2019)
PLANCK_C = 299792458.0  # m s-1, exact (SI 2019)
PLANCK_K = 1.380649e-23  # J K-1, exact (SI 2019)

BROADBAND_SPECTRA = 60
BROADBAND_SEED = 20260101


def load_table(csv_path: Path, columns: int) -> np.ndarray:
    """Read a CIE ``wavelength,value...`` CSV, ignoring the trailing metadata-only blank lines."""
    rows = []
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(",")
        if len(fields) != columns:
            continue
        rows.append([float(field) for field in fields])
    array = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{csv_path}: non-numeric row")
    return array[np.argsort(array[:, 0])]


def verify_checksums() -> None:
    """Fail unless every vendored table still matches the checksum CIE published with it."""
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        meta = json.loads((csv_path.with_suffix(csv_path.suffix + "_metadata.json"))
                          .read_text(encoding="utf-8"))
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        published = {c["hashMethod"]: c["checksum"] for c in meta["checksums"]}
        if published.get("sha256") != digest:
            raise ValueError(
                f"{csv_path.name}: sha256 {digest} != CIE metadata {published.get('sha256')}; "
                "re-download the table from the DOI in this module's docstring")
        print(f"verified {csv_path.name} (sha256 ok) DOI {meta['identifier']['identifier']} "
              f"{meta['titles'][0]['title']}")


def primaries_to_xyz(ap1_xy: np.ndarray, white_xy: np.ndarray) -> np.ndarray:
    """Rows map linear AP1/D60 RGB to CIE XYZ, normalised so Y(1,1,1) == 1 (standard derivation).

    Each primary's tristimulus is (x/y, 1, z/y) up to a scale factor; the factors solve the 3x3
    system whose columns sum to the white point's tristimulus.
    """
    x_over_y = ap1_xy[:, 0] / ap1_xy[:, 1]
    z_over_y = (1.0 - ap1_xy[:, 0] - ap1_xy[:, 1]) / ap1_xy[:, 1]
    xyz_of_primaries = np.stack([x_over_y, np.ones(3), z_over_y], axis=1)
    white = np.array([white_xy[0] / white_xy[1], 1.0,
                      (1.0 - white_xy[0] - white_xy[1]) / white_xy[1]])
    scales = np.linalg.solve(xyz_of_primaries.T, white)
    # scales multiply per PRIMARY (the row axis of xyz_of_primaries), so the transposed product has
    # one column per primary: XYZ = matrix @ rgb.
    return (xyz_of_primaries * scales[:, None]).T


def planck(wavelength_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    """Relative spectral power of a Planck radiator; constants are SI-exact, see module docstring."""
    wavelength_m = wavelength_nm * 1e-9
    exponent = PLANCK_H * PLANCK_C / (wavelength_m * PLANCK_K * temperature_k)
    return wavelength_m**-5 / np.expm1(exponent)


def smooth_random_spectra(wavelength: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Non-negative smooth SPDs: sums of a few broad Gaussians, the shape class a rendered surface,
    foliage, sky gradient or torch glow lands in. Deterministic, so the error report reproduces."""
    rng = np.random.default_rng(seed)
    centre = rng.uniform(400.0, 700.0, size=(count, 3))
    width = rng.uniform(25.0, 130.0, size=(count, 3))
    height = rng.uniform(0.05, 1.0, size=(count, 3))
    bump = np.exp(-0.5 * ((wavelength[None, None, :] - centre[:, :, None]) / width[:, :, None]) ** 2)
    return (height[:, :, None] * bump).sum(axis=1)


def build_test_spectra(wavelength: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (SPD rows, monochromatic flag per row): radiators, an equal-energy reference, smooth
    broadband spectra, and the monochromatic hue anchors that bound what a tristimulus model can do."""
    mono_wavelengths = np.arange(MONO_LO_NM, MONO_HI_NM + MONO_STEP_NM / 2, MONO_STEP_NM)
    mono = np.zeros((len(mono_wavelengths), wavelength.size))
    for row, nm in enumerate(mono_wavelengths):
        mono[row, int(np.searchsorted(wavelength, nm))] = 1.0
    broadband = np.vstack([
        np.array([planck(wavelength, t) for t in (1500.0, 2856.0, 4000.0, 6500.0)]),
        np.ones((1, wavelength.size)),
        smooth_random_spectra(wavelength, BROADBAND_SPECTRA, seed=BROADBAND_SEED),
    ])
    return np.vstack([mono, broadband]), np.concatenate(
        [np.ones(len(mono), dtype=bool), np.zeros(len(broadband), dtype=bool)])


def luminance_for_m(m: float) -> float:
    """Adaptation luminance the CIE iteration assigns to a mesopic coefficient."""
    return float(10.0 ** ((m - MESOPIC_A) / MESOPIC_B))


def pixel_gain(ratio: np.ndarray, m: float) -> np.ndarray:
    """The shader's per-pixel brightness multiplier at adaptation coefficient m.

    The neutral normalisation the shader applies cancels between predicted and true gain, so it is
    left out: this metric is about colour-dependent error, and the overall level is the exposure
    control's business. m = 0 is quoted as an absolute difference rather than a ratio because a
    saturated red's true gain is zero there, and both 'no rod signal' and '1% rod signal' render as
    black -- which is the behaviour the effect is after, not an error.
    """
    return m + (1.0 - m) * ratio


def gain_error(c: np.ndarray, xyz: np.ndarray, photopic: np.ndarray, scotopic: np.ndarray,
               m: float) -> np.ndarray:
    """Absolute error in the per-pixel brightness multiplier: what a viewer actually gets wrong."""
    predicted = np.maximum((xyz @ c) / photopic, 0.0)  # the shader's clamp, included deliberately
    return np.abs(pixel_gain(predicted, m) - pixel_gain(scotopic / photopic, m))


def load_cie_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wavelength, CIE 1931 colour-matching functions, V') over the fitting range.

    The anchor checks are the normalisations both tables are defined by, so a wrong file, a
    re-download that changed the columns, or a transposed axis cannot get past them.
    """
    cmf_table = load_table(DATA_DIR / "CIE_xyz_1931_2deg.csv", 4)
    scot_table = load_table(DATA_DIR / "CIE_sle_scotopic.csv", 2)
    wavelength_all, cmf_all = cmf_table[:, 0], cmf_table[:, 1:4]

    peak = wavelength_all[np.argmax(cmf_all[:, 1])]
    if abs(peak - 555.0) > 0.5 or abs(cmf_all[:, 1].max() - 1.0) > 1e-9:
        raise ValueError(f"y_bar peak at {peak} nm with value {cmf_all[:, 1].max()}; "
                         "expected 555 nm / 1.0")
    scotopic_by_nm = dict(zip(scot_table[:, 0].astype(int), scot_table[:, 1]))
    scotopic_peak_nm = max(scotopic_by_nm, key=scotopic_by_nm.get)
    if not 500.0 <= scotopic_peak_nm <= 515.0 or abs(scotopic_by_nm[scotopic_peak_nm] - 1.0) > 1e-9:
        raise ValueError(f"V' peak at {scotopic_peak_nm} nm; expected 507 nm with value 1.0")
    print(f"anchors: y_bar peak {peak:.0f} nm == 1, V' peak {scotopic_peak_nm} nm == 1 "
          f"(Purkinje shift {peak - scotopic_peak_nm:.0f} nm)")

    in_range = (wavelength_all >= FIT_LO_NM) & (wavelength_all <= FIT_HI_NM)
    wavelength = wavelength_all[in_range]
    return wavelength, cmf_all[in_range], np.array([scotopic_by_nm[int(w)] for w in wavelength])


def scotopic_coefficients(cmf: np.ndarray, v_scotopic: np.ndarray,
                          weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares c with V'(lambda) ~= c . [x_bar, y_bar, z_bar]; unweighted by default.

    Weighting by either sensitivity was tried and is worse on the shader's own error (see main's
    table): the photopic-weighted fits chase the bright end of the curve and misplace the colours
    the shift is about.
    """
    if weights is None:
        weights = np.ones(cmf.shape[0])
    c, *_ = np.linalg.lstsq(cmf * weights[:, None], v_scotopic * weights, rcond=None)
    return c


@dataclass(frozen=True)
class Baked:
    """The constants shaders/pipelines/display/main.comp.slang pastes."""

    photopic_luma: np.ndarray
    scotopic_luma: np.ndarray
    neutral_ratio: float
    ev_intercept: float
    ev_slope: float

    def pixel_gain(self, acescg: np.ndarray, ev_scene: float, strength: float = 1.0) -> float:
        """The shader's per-pixel brightness multiplier, for checking what the constants do."""
        weight = photopic_weight(ev_scene)
        photopic = float(self.photopic_luma @ acescg)
        scotopic = max(float(self.scotopic_luma @ acescg), 0.0)
        ratio = scotopic / max(photopic, 1.0e-6)
        scotopic_weight = 1.0 - weight
        gain = (weight + scotopic_weight * ratio) / (weight + scotopic_weight * self.neutral_ratio)
        return max(1.0 + (gain - 1.0) * max(strength, 0.0), 0.0)


def photopic_weight(ev_scene: float) -> float:
    """CIE 191:2010's mesopic photopic weight on the EV100 scale ExposureState reports.

    The standard states it in log10 of luminance; the scene's absolute luminance is already on the
    renderer's EV100 scale (RtSceneUnits: EV100 = log2(8 L), a scene value of 1.0 being 1 cd/m2), so
    the restatement below keeps the shader from converting units. m == 1 at the standard's 5 cd/m2
    photopic limit and 0 at its 0.005 cd/m2 scotopic limit, which is what leaves daylight untouched.
    """
    return float(np.clip(EV_PHOTOPIC_WEIGHT_INTERCEPT + EV_PHOTOPIC_WEIGHT_SLOPE * ev_scene, 0.0, 1.0))


EV_PHOTOPIC_WEIGHT_INTERCEPT = MESOPIC_A - np.log2(8.0) * MESOPIC_B / np.log2(10.0)
EV_PHOTOPIC_WEIGHT_SLOPE = MESOPIC_B / np.log2(10.0)


def bake(wavelength: np.ndarray, cmf: np.ndarray, v_scotopic: np.ndarray) -> Baked:
    matrix = primaries_to_xyz(AP1_XY, ACES_WHITE_XY)
    photopic_error = float(np.max(np.abs(matrix[1] - EXPECTED_PHOTOPIC_LUMA)))
    print(f"AP1/D60 -> XYZ Y row = {matrix[1]}")
    print(f"  vs shader ACESCG_LUMA: max abs error {photopic_error:.3e} "
          f"({'OK' if photopic_error < 5e-8 else 'MISMATCH'})")
    if photopic_error >= 5e-8:
        raise SystemExit("derived ACEScg luminance row does not match the committed shader constant")
    scotopic_luma = matrix.T @ scotopic_coefficients(cmf, v_scotopic)
    return Baked(
        photopic_luma=matrix[1],
        scotopic_luma=scotopic_luma,
        neutral_ratio=float(scotopic_luma.sum() / matrix[1].sum()),
        ev_intercept=float(EV_PHOTOPIC_WEIGHT_INTERCEPT),
        ev_slope=float(EV_PHOTOPIC_WEIGHT_SLOPE),
    )


def report(label: str, c: np.ndarray, matrix: np.ndarray, xyz: np.ndarray, photopic: np.ndarray,
           scotopic: np.ndarray, is_mono: np.ndarray) -> None:
    scotopic_luma = matrix.T @ c
    neutral = float(scotopic_luma.sum() / matrix[1].sum())
    red_ratio = scotopic_luma[0] / matrix[1][0]
    print(f"{label}: scotopic luma = {scotopic_luma}")
    print(f"   neutral ratio {neutral:.4f}, red primary ratio {red_ratio:+.4f}"
          f"{' (clamped to 0 in the shader)' if red_ratio < 0.0 else ''}")
    for m in EVALUATION_M:
        error = gain_error(c, xyz, photopic, scotopic, m)
        for name, mask in (("broadband", ~is_mono), ("monochromatic", is_mono)):
            spread = error[mask]
            print(f"   m={m:.1f} (L={luminance_for_m(m):.3f} cd/m2) {name}: "
                  f"median {np.median(spread):.3f} p95 {np.percentile(spread, 95):.3f} "
                  f"max {spread.max():.3f}")


def main() -> int:
    np.set_printoptions(precision=8, suppress=True)
    verify_checksums()
    wavelength, cmf, v_scotopic = load_cie_tables()
    v_photopic = cmf[:, 1]
    baked = bake(wavelength, cmf, v_scotopic)

    step = np.gradient(wavelength)
    spd, is_mono = build_test_spectra(wavelength)
    xyz = (spd[:, None, :] * cmf.T * step).sum(axis=-1)
    photopic_integral = (spd * v_photopic * step).sum(axis=-1)
    scotopic_integral = (spd * v_scotopic * step).sum(axis=-1)
    matrix = primaries_to_xyz(AP1_XY, ACES_WHITE_XY)

    # The baked row is the unweighted fit; the alternatives are reported so the choice stays
    # checkable, and a control arm (c = [0,1,0]: rod response == photopic, i.e. no shift at all)
    # gives those errors something to be measured against.
    for label, c in (("baked   (unweighted)", scotopic_coefficients(cmf, v_scotopic)),
                     ("alt     (y-weighted)", scotopic_coefficients(cmf, v_scotopic, v_photopic)),
                     ("alt     (V'-weighted)", scotopic_coefficients(cmf, v_scotopic, v_scotopic)),
                     ("alt     (sqrt(y V') )", scotopic_coefficients(
                         cmf, v_scotopic, np.sqrt(v_photopic * v_scotopic))),
                     ("control (no shift)", np.array([0.0, 1.0, 0.0]))):
        report(label, c, matrix, xyz, photopic_integral, scotopic_integral, is_mono)
        print()

    print(f"mesopic m = {MESOPIC_A} + {MESOPIC_B} * log10(L): m({MESOPIC_PHOTOPTIC_LIMIT} cd/m2) = "
          f"{MESOPIC_A + MESOPIC_B * np.log10(MESOPIC_PHOTOPTIC_LIMIT):.4f}, "
          f"m({MESOPIC_SCOTOPIC_LIMIT} cd/m2) = "
          f"{MESOPIC_A + MESOPIC_B * np.log10(MESOPIC_SCOTOPIC_LIMIT):.4f} "
          "(both land on the standard's own 1 and 0 limits)")
    for ev in (15.0, 5.32, 0.0, -3.0, -6.0):
        print(f"  m(EV100 = {ev:+.2f}, L = {2.0 ** ev / 8.0:.6f} cd/m2) = {photopic_weight(ev):.4f}")

    print("\n--- paste into shaders/pipelines/display/main.comp.slang ---")
    print(f"static const float3 ACESCG_LUMA = "
          f"float3({', '.join(f'{v:.8f}' for v in baked.photopic_luma)});")
    print("static const float3 ACESCG_SCOTOPIC_LUMA = "
          f"float3({', '.join(f'{v:.8f}' for v in baked.scotopic_luma)});")
    print(f"static const float PURKINJE_NEUTRAL_RATIO = {baked.neutral_ratio:.8f};")
    print(f"static const float MESOPIC_EV_INTERCEPT = {baked.ev_intercept:.8f};")
    print(f"static const float MESOPIC_EV_SLOPE = {baked.ev_slope:.8f};")
    return 0


if __name__ == "__main__":
    sys.exit(main())
