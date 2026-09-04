# GitHub release checklist

- [ ] Obtain coauthor/data-owner approval for every file under `data/processed/`.
- [ ] Decide and add an explicit data license; the code license is not enough.
- [ ] Decide whether precise vineyard coordinates and acquisition timestamps
      should be public or de-identified.
- [ ] Deposit raw imagery and redistributable checkpoints externally and add
      the DOI/URL plus checksums to `docs/DATA.md`.
- [ ] Confirm the berry SAM-CLIP checkpoint has a redistribution statement, or
      publish only download/request instructions.
- [ ] Run `make verify` and `make test`.
- [ ] Recreate the QTL environment and run a two-permutation smoke test.
- [ ] Review author order, affiliations, repository name, and DOI in
      `CITATION.cff`.
- [ ] Remove any draft notes that should not be public.
- [ ] Only after all checks pass, run `git init` manually and inspect
      `git status` before the first commit.

Suggested first GitHub repository name: `grape-berry-color-qtl`.
