# Contributing to dinoseg

`dinoseg` is a research reference implementation, not a packaged product. Small, focused contributions are welcome — bug fixes, documentation corrections, and a headless path that exports masks are the most useful things right now.

## Before you start

- Open an issue for anything larger than a bug fix or a docs correction, so the approach can be agreed before code is written.
- Do not add benchmark or accuracy claims without a reproducible script and dataset that anyone can rerun.
- Keep the on-disk sidecar contract in [`README.md`](README.md#on-disk-contract) compatible, or version it explicitly. Existing label files must not be silently invalidated.

## Development setup

Follow the [quick start](README.md#quick-start). In short: Python 3.12, a local DINOv3 checkout, and the ViT-S/16 LVD-1689M checkpoint. The scripts are run directly from the repository root:

```bash
python3 -m py_compile dinoseg3.py dinoseg2.py dinoseg.py
python3 dinoseg3.py img_0001.jpg img_0002.jpg
```

There is no test suite and no CI. A pull request that changes behaviour should say how it was checked by hand, with which images and settings. `dinoseg3.py` is the canonical script; changes to the older snapshots must keep them clearly marked as historical.

## Ground rules

- Keep the change scoped. Do not refactor unrelated code or reformat whole files in the same pull request.
- Do not commit model weights, embedding caches, label sidecars, a local `dinov3` checkout, or editor/build artifacts. The existing [`.gitignore`](.gitignore) covers them.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
- By contributing, you agree that your contribution is licensed under the repository's GNU AGPL-3.0-only license; see [`LICENSE`](LICENSE).
