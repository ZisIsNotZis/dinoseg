[English](README.md) | 简体中文

<p align="center">
  <img src="assets/logo.svg" alt="dinoseg — 点击标注分割" width="480">
</p>

# dinoseg

**点几个像素，整批图像就能出掩码。**

dinoseg 是一个建立在**冻结的 DINOv3 ViT-S/16 图块特征**之上的交互式分割标注工具。你只需要在少数几张图里点中几块像素，特征之上的分类器就会把掩码铺到整批图像上——不用微调，不用标好的数据集，也不用 GPU 集群。

> **状态：研究用参考实现。** 它只能在 Python 3.12 下从源码运行，只有图形界面，没有打包、测试、CI，也不能导出掩码。完整而诚实的状态清单见 [`docs/project-status.md`](docs/project-status.md)。

## 为什么做它

工业质检的日常是图像很多、掩码几乎没有。微调分割网络需要你手上没有的标注；颜色阈值又会在光线或零件一变时立刻失效。冻结的 DINOv3 图块特征已经足够好：在少量点击过的图块上做线性探针或最近邻投票，就能泛化到一整批图像。于是人的工作量从一轮标注工程变成几次点击。`dinoseg` 就是验证这个想法的小型 OpenCV 界面。

## 快速开始

需要 Python 3.12 和常见的科学计算依赖：

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

还缺两样外部材料，因为脚本通过 `torch.hub` 的 `source='local'` 模式加载 DINOv3：

1. **本地的 DINOv3 代码仓库** — 克隆
   [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)，其
   `hubconf.py` 需要暴露 `dinov3_vits16`。脚本默认在当前工作目录里找名为
   `dinov3` 的目录；放在别处就用 `DINO_HUB` 指过去。
2. **ViT-S/16 LVD-1689M 权重**
   （`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`），从 DINOv3 官方模型发布页下载。

```bash
git clone https://github.com/facebookresearch/dinov3.git

export DINO_HUB="$PWD/dinov3"
export DINO_WEIGHTS="$PWD/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

python3 dinoseg3.py img_0001.jpg img_0002.jpg img_0003.jpg
```

第一次运行会为每张图提取图块特征，并以 sidecar 文件缓存在图像旁边。之后只要图像和权重没变，就直接复用缓存。

### 操作

窗口是 `CVSH × CVSW` 的图像网格。

| 输入 | 动作 |
| --- | --- |
| `1`–`4` | 选择当前要涂的类别（对 `CLS` 取模） |
| 左键 | 把光标下的图块标成当前类别 |
| 右键 | 把光标下的图块标成背景/忽略 |
| `v` | 切换叠加层：关 → kNN 预测 → 线性探针 |
| `f` | 按当前预测置信度给图像排序 |
| `w` / `s` / `a` / `d` | 缩小/放大网格（行/列） |
| `q` / `e` | 向前/向后翻一屏；`Tab` / `r` 翻两屏 |
| `Esc` | 退出 |

## 目前能做什么

- **冻结的 DINOv3 特征** — 每张图做一次前向，缩放到 640×640，得到 40×40×384 的图块特征图。
- **点击标注** — 每张图最多 99 个已标注图块，以 `(y, x, class)` 三元组存在图像旁边。
- **两种预测器**，都跑在冻结特征上：
  - **kNN 投票** — 对每个已标注图块算余弦相似度，低于 0.8 的边距判为背景。
  - **线性探针** — `torch.nn.Linear(384, CLS)`，用 Adam 在点击过的图块上训练。背景是补出来的一列固定为 `0` 的 logit，所以只要 `CLS` 个类别 logit 全为负，该图块就判为背景。
- **批量检查** — 网格翻页、类别切换、按置信度排序、实时切换叠加层。
- **增量缓存** — 特征、标注和预测都按输入的 `mtime` 失效，一次点击只会重算依赖它的部分。
- **不落盘掩码** — 预测只画在 OpenCV 窗口里，仅此而已；导出是当前最大的缺口。

## 架构

```
 图像 ──缩放到 640×640──► 冻结的 DINOv3 ViT-S/16 ──► 40×40×384 图块特征
                                                              │
                                                     sidecar 缓存 (.40,40,384e)
                                                              │
 点击 ──► sidecar 标注 (.99,3B) ──► kNN 投票 / 线性探针 ──► 掩码叠加层
                                                              │
                                                   OpenCV 网格窗口（无导出）
```

所有逻辑都在同一个进程、同一个脚本里。`dinoseg3.py` 维护 `Lazy`/`Cache` 图：每个值都记着自己输入的时戳，只有输入变化时才重算。

### 三个脚本

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `dinoseg3.py` | **当前版本。** 按 `mtime` 失效的惰性缓存图，自包含，同时有 kNN **和** 线性探针，重绘时不重复训练。 | 可用 |
| `dinoseg2.py` | 第二版：把辅助函数拆进 `util.py`，加入 float16 磁盘缓存和 kNN 预测器、matplotlib 配色。 | **已被取代；按存档无法运行**（缺 `util.py`） |
| `dinoseg.py` | 第一版：全部内联，内存缓存，每次点击都重训一个线性层，用 `ctypes` `mmap` 读写标注文件。 | **已被取代** |

它们是同一个程序的三个连续快照，不是三个可以互相配合的工具。要跑就跑 `dinoseg3.py`。

### 磁盘上的约定

| 文件 | 内容 |
| --- | --- |
| `<image>.40,40,384e` | float16 图块特征，`40 × 40 × 384` |
| `<image>.99,3B` | 最多 99 个已标注像素，`(y, x, class)`，`uint8` |

两者都写在源图像旁边。想从头开始就删掉它们。除此之外不写任何东西。

## 如何开发

这里没有安装包，也没有测试套件。从仓库根目录运行脚本，保持 `DINO_HUB` 和 `DINO_WEIGHTS` 已导出，再准备一小批你能在一分钟内手工标完的图像。

- **语法检查：** `python3 -m py_compile dinoseg3.py dinoseg2.py dinoseg.py`
- **配置：** 每个全大写全局变量都可以用同名环境变量覆盖（脚本顶部的 `globals().update(...)`），例如 `CLS=7` 设置类别数、`EMBH=40` / `EMBW=40` 设置图块网格、`CVSH` / `CVSW` 设置窗口网格。
- **可写的数据目录：** 脚本会把 sidecar 文件写在输入图像旁边，只读的数据挂载会失败。
- **新增预测器：** 参照 `dinoseg3.py` 里的 `KnnPrd` / `LinPrd`，返回图块分辨率的类别编号图，剩下的交给现有渲染器缩放。

## 如何贡献

欢迎小而专注的拉取请求。动手前先读 [`CONTRIBUTING.md`](CONTRIBUTING.md)；超出修 bug 或改文档范围的改动，请先开 issue 讨论。最有价值的贡献是加一条无界面的掩码导出路径；其次是清理已经跑不起来的 `dinoseg2.py`，或者把它缺的 `util.py` 补回来。

## 路线图

### 已实现

- DINOv3 ViT-S/16 图块特征提取（`640×640 → 40×40×384`），带 float16 sidecar 缓存。
- 点击标注，存进固定大小的 sidecar 文件（每张图最多 99 个点）。
- 冻结特征上的 kNN 余弦投票和线性探针两种预测器。
- 网格翻页、类别切换、置信度排序、叠加层切换。
- 当前版本按 `mtime` 做增量重算。

### 计划中

- **掩码导出和无界面 CLI** — 最大的缺口；把掩码写成 PNG/NumPy，而不是只画在窗口里。
- **打包** — `pyproject.toml` 和可安装的命令行入口。
- **测试** — 围绕缓存图和两个预测器加一个合成图像冒烟测试，再用可复现的 IoU/Dice 测量取代“没有测过”的质量声明。
- **清理** — 退役 `dinoseg2.py`/`dinoseg.py`，或者补回缺失的辅助模块，让仓库里每个文件都能跑。
- **标注流程** — 导入/导出点击标签；让首次运行就可用（`CLS` 现在默认是 `1`）。

### 未来愿景

这个想法真正有用的形态是一个批量标注器：点几块有代表性的图块，把掩码传播到整个质检批次，修正失败的少数几张，然后导出掩码——全程不训练分割网络。`dinoseg` 是这条回路上的参考原型，不承诺这个仓库一定会变成那个产品。

## 已知粗糙之处

- `dinoseg2.py` 里有 `from util import vcache, chunk`，而存档里没有 `util.py`。
- `dinoseg.py` 用了 `ctypes`/`libc` 的 `mmap`，因此只能在 Linux/glibc 上工作。
- `dinoseg2.py` 的线性路径在原始 logits 上用 `cnf < 0` 做置信度截断，实际上等于没截。`dinoseg3.py` 已修复。
- 精度从未被测量：没有 IoU/Dice，没有数据集，没有基准。质量声明只有“可用于交互式冷启动”，仅此而已。

## 相关工作

与同一作者的 `dino.py` 项目同源、使用同一份权重（都是 DINO 特征做机器视觉），但**没有共享代码，也没有运行依赖**。`dino.py` 用纯 `ggml`/GGUF 跑 DINOv2/v3 前向；`dinoseg` 用 PyTorch `torch.hub` 跑 DINOv3，并在上面搭交互式标注器。

## 许可证

GNU AGPL-3.0-only。见 [`LICENSE`](LICENSE)。
