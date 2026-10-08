# Preserved local files (2026-10-08)

Before removing this local checkout, personal notes/configuration and unpublished source changes were preserved in `local-archive-20261008.tar.gz`. Its `ARCHIVE_MANIFEST.json` records each preserved path, size and SHA-256, plus original Git refs.

Extract from the repository root:

```sh
tar -xzf docs/local-archive-20261008.tar.gz
```

Re-downloadable datasets/models, training checkpoints and generated caches are intentionally excluded. Source scripts inside model directories are preserved. Source-only patches of unpublished local branches, if present, live under `historical-branches/`; they omit model/checkpoint binaries.
