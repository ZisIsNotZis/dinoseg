# dinoseg project status

## Classification

Private-archive salvage: interactive segmentation / region-masking tooling built
on **frozen DINOv3 ViT-S/16 patch tokens** for industrial machine vision.
Three successive snapshots of one Python GUI app (`dinoseg.py` → `dinoseg2.py`
→ `dinoseg3.py`); researched and written, never released, no package, no tests.

## Status

**Archived / superseded snapshots — not maintained.** `dinoseg3.py` is the
canonical, runnable version: it loads DINOv3 via `torch.hub`, extracts 40×40
patch tokens, lets the user click labels into sidecar files, and renders kNN and
linear-probe masks in an OpenCV grid GUI. `dinoseg2.py` and `dinoseg.py` are
earlier iterations, kept for provenance only. `dinoseg2.py` cannot run
as-archived (missing `util.py`). No accuracy metric, no benchmark and no export
path were ever implemented — the tool draws to screen and that is all.

## Evidence

- `dinoseg3.py` — canonical. `Lazy`/`Cache` mtime-invalidated cache graph;
  `EmbModel()` (`torch.hub.load(DINO_HUB, 'dinov3_vits16', source='local',
  weights=EMBF)`); `KnnPrd` (cosine vote, `< .8` → background) and `LinPrd`
  (`torch.nn.Linear(384, CLS)`, Adam 10×200).
- `dinoseg2.py` — second iteration; `Fulbl`/`Prd` predictors, float16 disk
  cache (`.40,40,384e`) with migration from float32, matplotlib `rainbow`
  overlay. Imports `from util import vcache, chunk`, and **`util.py` is absent
  from the archive** → not runnable as copied.
- `dinoseg.py` — first iteration; everything inline, in-memory `vcache`,
  `ctypes` `mmap` of `libc.so.6` for label files, one linear layer retrained on
  every click.
- Shared on-disk contract across all three: `<image>.<H>,<W>,<EMB><dtype>` for
  cached patch tokens and `<image>.99,3B` for up to 99 `(y, x, class)` labels.
- The load uses the exact checkpoint
  `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, the same file the author's
  `dino.py` repo converts to GGUF (`convert_dinov3_vit_to_hf.py`), and the
  original `dinoseg3.py` defaulted to the same `/home/z/gguf/` weights
  directory as `dino.py`'s own tools.
- Genericisation applied for publication: the hard-coded weights path became
  `DINO_WEIGHTS` and the hard-coded hub dir became `DINO_HUB`
  (`os.getenv`/`getenv`, default `dinov3`); the no-op env-override predicate in
  `dinoseg.py` (`i is os.environ`) was corrected to `i in os.environ`.
- `python3 -m py_compile` passes for all three files (Python 3.12.3).

## Boundaries

- No CLI, no headless mode, no mask export; the only output is an OpenCV window.
- No tests, no CI, no packaging, no `LICENSE` (all-rights-reserved by default —
  a license must be chosen before publishing).
- No quantitative result of any kind: no IoU/Dice, no dataset, no comparison to
  fine-tuned segmentation models. Do not cite an accuracy figure.
- No code sharing with `dino.py`: no import, no common library, no runtime
  dependency. They are siblings (PyTorch interactive labeller vs. ggml/GGUF
  extractor), not one a successor of the other.
- `CLS` defaults to 1 and the model is loaded through a local `dinov3`
  checkout, so the tool is environment-sensitive; `dinoseg2.py`'s linear path
  carries a `cnf < 0` threshold bug that makes its confidence cut a no-op.
- The script writes sidecar cache/label files next to the input images; on a
  read-only or shared dataset directory it fails.
