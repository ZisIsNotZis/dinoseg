# dinoseg — click-to-label segmentation on top of frozen DINOv3 features

An interactive, dependency-light tool for **industrial machine-vision
segmentation / region masking**: you point at a handful of pixels in a few
images, and a classifier over frozen DINOv3 patch tokens paints the mask on
every image — no training loop, no labelled dataset, no GPU cluster.

The problem it solves: in a factory / inspection setting you usually have many
images and almost no masks. Fine-tuning a segmentation network needs labels you
do not have; classical colour/threshold segmentation breaks the moment
lighting or the part changes. DINOv3's frozen patch tokens are good enough that
a linear probe or a nearest-neighbour vote over a few clicked pixels generalises
across the batch, so the human effort is a few clicks instead of a full
annotation campaign. The tool is a small OpenCV GUI that does exactly that and
nothing else.

> This is a **private-archive salvage**, published as-is for the record. It is
> not a product, has no packaging or tests, and was never released. See
> [`docs/project-status.md`](docs/project-status.md) for the honest status.

## How to run

Requirements (all three scripts):

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

Two external pieces must exist, because the scripts load DINOv3 through
`torch.hub` in `source='local'` mode:

1. **A local DINOv3 checkout** — a clone of
   [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) whose
   `hubconf.py` exposes the `dinov3_vits16` entry point. The scripts look for a
   directory literally named `dinov3` (relative to the working directory);
   point `DINO_HUB` at another path if yours lives elsewhere.
2. **The ViT-S/16 LVD-1689M checkpoint**
   `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` — the same checkpoint the
   author's sibling `dino.py` repo converts to GGUF. The original code
   hard-coded an absolute path (`/home/z/gguf/...`); that has been genericised
   to the `DINO_WEIGHTS` environment variable.

Then:

```bash
export DINO_HUB=/path/to/dinov3                              # default: ./dinov3
export DINO_WEIGHTS=/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# canonical script; pass one or more images
python3 dinoseg3.py img_0001.jpg img_0002.jpg img_0003.jpg
```

Embeddings and labels are cached **next to each image** as sidecar files
(`<image>.40,40,384e` for the float16 patch tokens, `<image>.99,3B` for up to
99 labelled pixels). Delete them to start over. Nothing else is written.

### GUI

The window is a grid of up to `CVSH × CVSW` images.

| Input | Action |
| --- | --- |
| `1`–`6` | pick the class id to paint (modulo `CLS`) |
| left click | label the patch under the cursor with the current class |
| right click | label the patch as background/ignore |
| `v` | cycle overlay: off → kNN prediction → linear probe |
| `f` | sort images by current-prediction confidence |
| `w`/`s`/`a`/`d` | shrink/grow the grid (rows/columns) |
| `q`/`e` | page back/forward one screen; `Tab`/`r` two screens |
| `Esc` | quit |

### Environment variables

Every upper-case global can be overridden by an environment variable of the
same name (`globals().update(...)` at the top of each script), so besides the
two above you can set e.g. `CLS=7` (number of classes, default 1),
`EMBH=40` / `EMBW=40` (patch grid), `CVSH` / `CVSW` (grid size).

## The three files

| File | Role | State |
| --- | --- | --- |
| `dinoseg3.py` | **Canonical / current.** Lazy `mtime`-invalidated cache graph, self-contained, kNN **and** linear-probe predictors, no retrain on every redraw. | Usable |
| `dinoseg2.py` | Second iteration: split helpers into `util.py`, float16 disk cache, added the kNN predictor, matplotlib colour map. | **Superseded by `dinoseg3.py`; also does not run as-archived** |
| `dinoseg.py` | First iteration: everything inline, in-memory cache, single linear layer retrained on every click, `ctypes` `mmap` label files. | **Superseded** |

They are three successive snapshots of the same program, not three
interoperable tools. `dinoseg3.py` is the one to run.

## Implemented vs. not

**Implemented (works in the code as written):**

- DINOv3 ViT-S/16 patch-token extraction (640×640 → 40×40×384) with a
  file-backed float16 embedding cache.
- Click-to-label annotation stored as fixed-size sidecar files (max 99 points
  per image, `(y, x, class)` triples).
- Two predictors over the frozen features:
  - **kNN / nearest-neighbour vote** — cosine similarity against every labelled
    patch, threshold 0.8 (`KnnPrd` in `dinoseg3.py`; `Fulbl` in `dinoseg2.py`).
  - **Linear probe** — `torch.nn.Linear(384, CLS)` trained with Adam
    (`LinPrd`/`LinModel` in `dinoseg3.py`; `Prd` in `dinoseg2.py`).
- Interactive grid navigation, class switching, confidence sorting, overlay
  toggle.
- Incremental recomputation: in `dinoseg3.py` every cached value is invalidated
  by the `mtime` of its inputs, so editing a label file retrains only what
  depends on it.

**Not implemented / rough edges (do not oversell this):**

- **No CLI, no headless mode, no batch export.** The tool only draws to an
  OpenCV window; there is no way to write masks to disk other than
  screen-reading. This is the single biggest gap.
- **No packaging** (`setup.py`/`pyproject.toml`), **no tests**, **no CI**.
- `dinoseg2.py` imports `from util import vcache, chunk`, but **`util.py` is not
  present in the archive** — that version cannot run as-archived. (The same two
  helpers are inlined in `dinoseg.py`, so the module was dropped between
  snapshots, not lost from the archive.)
- `dinoseg.py` additionally used `ctypes`/`libc` `mmap`, so it only worked on
  Linux/glibc.
- `CLS` defaults to **1**, so a first run has a single class and no useful
  background handling until you set `CLS` or right-click.
- Accuracy is not measured anywhere: **no IoU/Dice, no benchmark, no dataset,
  no numbers.** The quality claim is "usable for interactive bootstrapping",
  nothing more.
- `dinoseg2.py`'s `Prd` applies its confidence cut with `cnf < 0` on raw logits
  (line 105), which is effectively no cut — a leftover threshold bug. Fixed in
  `dinoseg3.py`, which uses the kNN margin `< .8`.
- Assumes NVIDIA-style float16 support only for storage, not compute; the
  forward pass runs whatever `torch.hub` gives you (CPU works, slowly).
- Absolute paths and the hard-coded `dinov3` hub name were genericised to
  `DINO_WEIGHTS` / `DINO_HUB` for this publication. The env-override line in
  `dinoseg.py` also had a no-op bug (`i is os.environ`) that was corrected to
  `i in os.environ`, otherwise the override never fired.

## Relationship to the author's `dino.py` repo

Same author, same line of work (DINO features for machine vision), and
demonstrably the same checkpoint lineage: `dinoseg*.py` load
`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, the exact filename `dino.py`'s
`convert_dinov3_vit_to_hf.py` treats as the ViT-S/16 LVD-1689M HF source, and
`dino.py`'s own tools default to the same `/home/z/gguf/` weights directory the
original `dinoseg3.py` hard-coded.

That said, **the two share no code**: there is no import, no common library, and
no runtime dependency between them. They are siblings on opposite ends of the
same pipeline — `dino.py` runs DINOv2/v3 forward passes in pure `ggml`/GGUF
(`dino2.py`, `dino3.py`) and adds smart-crop / keyframe apps, whereas `dinoseg`
runs DINOv3 through PyTorch `torch.hub` and builds an interactive segmentation
labeller on top. Whether one was meant to replace the other is not stated
anywhere in the code; do not claim it.

Status: **archived salvage, do not publish as a maintained tool.** Unreleased,
unmeasured, GUI-only, no tests. Kept as provenance/portfolio evidence of the
DINO-for-inspection line of work.

License: GNU AGPL-3.0-only; see [LICENSE](LICENSE).
