# Release preparation checks

Prepared from the current development source on 2026-11-02 in a separate directory. The original working tree and previous review archive were not modified.

- Source-only copy; no previous Git history or remotes imported.
- Excluded datasets, model weights, checkpoints, logs, environment files, caches and generated experimental reports.
- Replaced 28 private endpoint occurrences with an explicit placeholder.
- Removed shell W&B credential defaults; credentials come from environment variables.
- Known credential-prefix, private-endpoint and personal-path pattern rescan: no matches.
- Parsed 622 Python files and checked 92 agent_rubric2 shell launchers: passed.
- GPU training and external judge connectivity have not been tested in this export.
- Pattern scanning is not a guarantee that every possible secret format is recognized. Gitleaks 8.30.1 was also run; three upstream code/prose false positives were reviewed and narrowly allowlisted.
- Upstream license and attribution files retained.

Prepared for publication to AnnaXChun/ORSD.
