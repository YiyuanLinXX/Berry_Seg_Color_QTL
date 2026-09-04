# Migration notes

The research working directory was about 32 GB and mixed source code, caches,
raw data, derived data, trained models, figures, package libraries, and a 13 GB
Git object database. This public-repository candidate is about 13 MB.

## Retained

- three vision post-processing modules;
- FoundationStereo batch integration without vendoring the upstream model;
- frame-to-vine, instance-to-vine, and vine-aggregation modules from the
  GeoReference workflow;
- continuous and categorical QTL scripts;
- QTL inputs, spatial inputs, checksums, study parameters, and selected paper
  outputs;
- environment files, tests, citations, license notices, and release checks.

## Excluded

- old Git metadata and the nested QTL Git pointer/history;
- raw RGB, depth arrays, predicted/instance masks, and masked crops;
- 3–4 GB random-forest files and exploratory color-classification outputs;
- copied R libraries, temporary plots, and full repeated QTL result trees;
- exploratory 2025 classifiers and t-SNE utilities not required for the
  paper's continuous color-trait workflow.

The existing source directories were not modified or deleted. The new folder
does not contain `.git`; repository initialization is intentionally left to
the maintainer after the data-permission checklist is resolved.
