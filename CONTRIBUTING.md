# Contributing

- **Branches:** `feature/<task>` (e.g. `feature/webdev`), merged to `main` by pull request.
- **Before opening a PR:** `python -m pytest -m "not slow"`; regenerate the map with `python tools/make_codebase_map.py`;
  check new code against the existing modules (reuse `common/` helpers, do not duplicate them).
- **Docs:** add an activity-log line and any decision (DEC-nnn) to `docs/PROJECT_LOG.md` in the same PR.
- **Never commit:** `work/`, `keys/`, `*.sqlite`, `*.npy`, photos of people.
- **AI assistance:** disclose AI-assisted code in the PR description as required.
