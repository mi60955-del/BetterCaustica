# Third-Party Notices

Caustica's project-owned code is licensed under `LGPL-3.0-or-later`. This file
documents third-party components and license boundaries that are not changed by
Caustica's license.

## NVIDIA DLSS / NGX SDK

Caustica can build and distribute release artifacts that include NVIDIA DLSS/NGX
SDK runtime components, including DLSS Ray Reconstruction and Frame Generation
libraries. These NVIDIA components are proprietary third-party software and are
not licensed under the LGPL.

The NVIDIA SDK components remain subject to the NVIDIA RTX SDKs license:

<https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt>

The LGPL license grant for Caustica does not grant rights to NVIDIA SDK
components. Redistribution and use of those components must comply with
NVIDIA's license terms.

This software contains source code provided by NVIDIA Corporation.

Bundled NVIDIA SDK runtime libraries may include files matching:

- `caustica/natives/windows-x64/nvngx_dlssd.dll`
- `caustica/natives/windows-x64/nvngx_dlssg.dll`
- `caustica/natives/linux-x64/libnvidia-ngx-dlssd.so*`
- `caustica/natives/linux-x64/libnvidia-ngx-dlssg.so*`

Caustica's `ngxshim` native library is project-owned glue code and follows
Caustica's project license unless otherwise noted.

## CIE photometric and colorimetric data

Caustica's dark-adaptation (Purkinje) shift is baked from two CIE data tables vendored under
`tools/data/cie/`:

- Colour-matching functions of the CIE 1931 standard colorimetric observer (2 degree observer,
  1 nm, CIE 018:2019 Table 6) — <https://doi.org/10.25039/CIE.DS.xvudnb9b>
- CIE spectral luminous efficiency for scotopic vision, V'(lambda) (1 nm, CIE 018:2019 Table 2) —
  <https://doi.org/10.25039/CIE.DS.gr6w4b5g>

Both are published by the International Commission on Illumination (CIE), Vienna, under the
Creative Commons Attribution-ShareAlike 4.0 International licence
(<https://creativecommons.org/licenses/by-sa/4.0/>). `tools/bake_scotopic_basis.py` verifies each
vendored file against the SHA-256 published with it, and the scotopic luminance basis derived from
them carries its provenance in `shaders/pipelines/display/main.comp.slang`.

The mesopic adaptation-coefficient rule used by the same shift is quoted from BIPM,
`Principles Governing Photometry`, 2nd edition (Metrologia 56, 2019), equations (1) and (11),
which reproduce CIE 191:2010.
