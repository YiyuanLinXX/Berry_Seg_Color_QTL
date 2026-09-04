# Data

Compact processed inputs are versioned under `processed/`; see
`../docs/DATA.md` for inclusion policy and `../docs/DATA_DICTIONARY.md` for
column semantics. `manifest.sha256` covers every file under `processed/`.

Do not add raw imagery, depth arrays, masks, checkpoints, or trained model
files here. The top-level `.gitignore` excludes their usual locations and
extensions.
