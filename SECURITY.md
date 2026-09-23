# Security and Repository Hygiene

This repository is an educational and portfolio project. It should contain source code, documentation, tests and reproducibility configuration only.

## Do not commit

- passwords, API keys, access tokens or private keys;
- `.env` files or local credentials;
- private or personally identifiable datasets;
- virtual environments, caches or local IDE configuration;
- trained model artefacts that are intentionally excluded by `.gitignore`;
- large source datasets or GloVe embedding archives.

The root `.gitignore` excludes environment files, model files, datasets, embeddings, caches and common local development artefacts.

## Reporting

If you identify a security issue or accidentally exposed credential, report it privately to the repository owner rather than opening a public issue containing the sensitive value.
