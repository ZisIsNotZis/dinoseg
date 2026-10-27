English | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/logo.svg" alt="dinoseg — click-to-label segmentation" width="480">
</p>

# dinoseg

**Click a few pixels. Get masks across the whole batch.**

dinoseg is an interactive segmentation labeller built on **frozen DINOv3 ViT-S/16 patch features**. Point at a handful of pixels in a few images and a classifier over those features paints the mask on every image in the set — no fine-tuning, no labelled dataset, no GPU cluster.

> **Status: research reference implementation.** It runs from source on Python 3.12, is GUI-only, and has no packaging, tests, CI, or mask export. The full inventory of what is and is not here lives in [`docs/project-status.md`](docs/project-status.md).

## Why it exists

Industrial inspection usually gives you many images and almost no masks. Fine-tuning a segmentation network needs labels you do not have; colour thresholds break the moment the lighting or the part changes. Frozen DINOv3 patch tokens are good enough that a linear probe or a nearest-neighbour vote over a few clicked patches generalises across a batch, so the human effort is a few clicks instead of an annotation campaign. `dinoseg` is the small OpenCV GUI that tests that idea.

## Quick start

You need Python 3.12 and the usual scientific stack:

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

Two external pieces are required, because the scripts load DINOv3 through `torch.hub` in `source='local'` mode:

1. **A local DINOv3 checkout** — a clone of
   [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) whose
   `hubconf.py` exposes `dinov3_vits16`. The scripts look for a directory named
   `dinov3` in the working directory unless you point `DINO_HUB` elsewhere.
2. **The ViT-S/16 LVD-1689M checkpoint**
   (`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`), downloaded from the
   official DINOv3 model release.

```bash
git clone https://github.com/facebookresearch/dinov3.git

export DINO_HUB="$PWD/dinov3"
export DINO_WEIGHTS="$PWD/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

python3 dinoseg3.py img_0001.jpg img_0002.jpg img_0003.jpg
```

The first run extracts patch tokens for each image and caches them next to the image as sidecar files. Later runs reuse the cache until the image or the weights change.

### Controls

The window is a grid of up to `CVSH × CVSW` images.

| Input | Action |
| --- | --- |
| `1`–`4` | pick the class to paint (modulo `CLS`) |
| left click | label the patch under the cursor with the current class |
| right click | label the patch as background/ignore |
| `v` | cycle overlay: off → kNN prediction → linear probe |
| `f` | sort images by current-prediction confidence |
| `w` / `s` / `a` / `d` | shrink/grow the grid (rows/columns) |
| `q` / `e` | page back/forward one screen; `Tab` / `r` two screens |
| `Esc` | quit |

## What it can do now

- **Frozen DINOv3 features** — one forward pass per image, resized to 640×640, producing a 40×40×384 patch-token map.
- **Click-to-label** — up to 99 labelled patches per image, stored as `(y, x, class)` triples beside the image.
- **Two predictors** over the frozen features:
  - **kNN vote** — cosine similarity against every labelled patch, with a 0.8 margin to background.
  - **Linear probe** — `torch.nn.Linear(384, CLS)` trained with Adam on the clicked patches. Background is a padded logit fixed at `0`, so a patch falls back to background whenever all `CLS` class logits come out negative.
- **Batch review** — grid navigation, class switching, confidence sorting and live overlay toggling.
- **Incremental caches** — embeddings, labels and predictions are invalidated by the `mtime` of their inputs, so a click retrains only what depends on it.
- **No masks on disk** — predictions are drawn to an OpenCV window and that is all; export is the biggest missing piece.

## Architecture

```
 image ──resize 640×640──► frozen DINOv3 ViT-S/16 ──► 40×40×384 patch tokens
                                                              │
                                                     sidecar cache (.40,40,384e)
                                                              │
 clicks ──► sidecar labels (.99,3B) ──► kNN vote / linear probe ──► mask overlay
                                                              │
                                                   OpenCV grid window (no export)
```

Everything runs in-process from a single script. `dinoseg3.py` owns the `Lazy`/`Cache` graph: each value carries the timestamp of its inputs and recomputes only when one of them changes.

### The three scripts

| File | Role | State |
| --- | --- | --- |
| `dinoseg3.py` | **Canonical.** Lazy `mtime`-invalidated cache graph, self-contained, kNN **and** linear-probe predictors, no retrain on every redraw. | Usable |
| `dinoseg2.py` | Second iteration: split helpers into `util.py`, float16 disk cache, added the kNN predictor, matplotlib colour map. | **Superseded; does not run as archived** (`util.py` is missing) |
| `dinoseg.py` | First iteration: everything inline, in-memory cache, one linear layer retrained on every click, `ctypes` `mmap` label files. | **Superseded** |

They are three successive snapshots of the same program, not three interoperable tools. `dinoseg3.py` is the one to run.

### On-disk contract

| File | Contents |
| --- | --- |
| `<image>.40,40,384e` | float16 patch tokens, `40 × 40 × 384` |
| `<image>.99,3B` | up to 99 labelled pixels, `(y, x, class)` as `uint8` |

Both are written next to the source image. Delete them to start over. Nothing else is written.

## How to develop

There is no package and no test suite. Run the script from the repository root, keep `DINO_HUB` and `DINO_WEIGHTS` exported, and use a small set of images you can label by hand in a minute.

- **Syntax check:** `python3 -m py_compile dinoseg3.py dinoseg2.py dinoseg.py`
- **Configuration:** every upper-case global can be overridden by an environment variable of the same name (`globals().update(...)` at the top of each script) — for example `CLS=7` for the number of classes, `EMBH=40` / `EMBW=40` for the patch grid, or `CVSH` / `CVSW` for the window grid.
- **Writable data directory:** the scripts write sidecar files next to the input images, so a read-only dataset mount will fail.
- **Adding a predictor:** follow `KnnPrd` / `LinPrd` in `dinoseg3.py`; return a class-id map at patch resolution, then let the existing renderer resize it.

## How to contribute

Small, focused pull requests are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first, and open an issue before starting anything larger than a bug fix or a docs correction. The most valuable contribution is a headless path that exports masks; the second is retiring the broken `dinoseg2.py` snapshot or restoring its missing `util.py`.

## Roadmap

### Implemented

- DINOv3 ViT-S/16 patch-token extraction (`640×640 → 40×40×384`) with a float16 sidecar cache.
- Click-to-label annotation stored in fixed-size sidecar files (max 99 points per image).
- kNN cosine vote and linear-probe predictors over the frozen features.
- Interactive grid navigation, class switching, confidence sorting and overlay toggling.
- `mtime`-driven incremental recomputation in the canonical script.

### Planned

- **Mask export and a headless CLI** — the single biggest gap; write masks as PNG/NumPy instead of only drawing them.
- **Packaging** — `pyproject.toml` and an installable entry point.
- **Tests** — a small synthetic-image smoke test around the cache graph and the two predictors, plus a reproducible IoU/Dice measurement instead of an unmeasured quality claim.
- **Cleanup** — retire `dinoseg2.py`/`dinoseg.py` or restore their missing helper so every file in the repository runs.
- **Annotation workflow** — import/export click labels, multi-class defaults that make the first run useful (`CLS` currently defaults to `1`).

### Future vision

The useful version of this idea is a batch labeller: click a few representative patches, propagate the mask across an inspection set, correct the failures, and export the masks without ever training a segmentation network. `dinoseg` is the reference prototype for that loop, not a promise that this repository will become the product.

## Known rough edges

- `dinoseg2.py` imports `from util import vcache, chunk`, and `util.py` is absent from the archive.
- `dinoseg.py` used `ctypes`/`libc` `mmap`, so it only worked on Linux/glibc.
- The linear path in `dinoseg2.py` applies its confidence cut as `cnf < 0` on raw logits, which is effectively no cut. Fixed in `dinoseg3.py`.
- Accuracy is not measured anywhere: no IoU/Dice, no dataset, no benchmark. The quality claim is "usable for interactive bootstrapping", nothing more.

## Related work

Same author, same checkpoint lineage as the `dino.py` project (DINO features for machine vision), but **no shared code and no runtime dependency**. `dino.py` runs DINOv2/v3 forward passes in pure `ggml`/GGUF; `dinoseg` runs DINOv3 through PyTorch `torch.hub` and builds an interactive labeller on top.

## License

GNU AGPL-3.0-only. See [`LICENSE`](LICENSE).
