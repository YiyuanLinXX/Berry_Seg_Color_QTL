# Third-party software and model terms

## SAM-CLIP

- Source: <https://github.com/YiyuanLinXX/SAM-CLIP>
- Pinned commit: `7f5a7a21ea00863ca52a96c1c96acf631109cf95`
- License: MIT (`LICENSES/SAM-CLIP-MIT.txt`)
- Distribution approach: external checkout; no SAM-CLIP source or weights are
  vendored here.

SAM-CLIP itself builds on SAM, CLIP, and other research code. Users must also
follow the licenses attached to those upstream packages and weights.

## FoundationStereo

- Source: <https://github.com/NVlabs/FoundationStereo>
- Pinned commit: `6e8806816b533e4d13ddbb95ffa907b797060a62`
- License: NVIDIA FoundationStereo license
  (`LICENSES/FoundationStereo-LICENSE`)
- Important restriction: FoundationStereo and derivative works are limited to
  non-commercial research use under the current upstream license.
- Distribution approach: external checkout and separately obtained weights.

`scripts/depth/run_foundation_stereo_pairs.py` is derived from the upstream
demo and is covered by the FoundationStereo license. All other original code
in this repository is covered by the top-level MIT license unless a file says
otherwise.

## Citation responsibility

Publications and derivative repositories should cite the ASABE workflow,
SAM-CLIP, Segment Anything, CLIP, FoundationStereo, and R/qtl as applicable.
