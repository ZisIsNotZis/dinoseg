#!/bin/env python3.12
from os import environ, getenv, stat
from sys import argv  # noqa
from typing import Generic, TypeVar, Callable, ParamSpec
from dataclasses import dataclass

from matplotlib.cm import rainbow  # noqa
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CVSI = CVSC = 0
CVS = CLS = CVSH = CVSW = 1
EMBF = getenv('DINO_WEIGHTS', 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
DINO_HUB = getenv('DINO_HUB', 'dinov3')
EMBH = EMBW = 40
globals().update((i, type(j)(environ[i])) for i, j in globals().items() if i.isupper() and i in environ)
T = TypeVar('T')
P = ParamSpec('P')


@dataclass
class Lazy(Generic[T]):
    t: Callable[[], float]
    f: Callable[[], T]
    _t = -1
    _ = None

    def __call__(self) -> T:
        if self._t < (_ := self.t()):
            self._ = self.f()
            self._t = _
        return self._


def Cache(f: Callable[P, T]) -> Callable[P, T]:
    def _(*arg: P.args, cache={}, **kwarg: P.kwargs) -> T:
        if (_ := cache.get(arg)) is None:
            _ = cache[arg] = f(*arg, **kwarg)
        return _

    return _


def mtime(f: str) -> float:
    try:
        return stat(f).st_mtime
    except FileNotFoundError:
        return 0


def color(a: torch.Tensor) -> torch.Tensor:
    return torch.where(a[..., None] < CLS, torch.from_numpy(rainbow(a / CLS)[..., 3]).mul_(128).byte(), 0)


@Cache
def mmap(f: str) -> torch.Tensor:
    return torch.from_file(f, True, (_ := np.array(f.split('.')[-1][:-1].split(','), int)).prod(), dtype=torch.from_numpy(np.empty((), f[-1])).dtype).reshape(_.tolist())


def fromfile(f: str) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(f, f.split('.')[-1])[0])


def tofile(a: torch.Tensor, f: str) -> torch.Tensor:
    (_ := a.numpy()).tofile(f + '.' + ','.join(map(str, a.shape)) + _.dtype.char)
    return a


def append(a: torch.Tensor, *i: int) -> None:
    a[a.any(1).numpy().argmin()] = torch.as_tensor(i)


def strip(a: torch.Tensor) -> torch.Tensor:
    return a[:a.any(1).numpy().argmin()]


@Cache
def Img(f: str) -> Lazy[torch.Tensor]:
    return Lazy(lambda: mtime(f), lambda: TF.resize(torch.from_numpy(cv2.imread(f)).permute(2, 0, 1), [EMBH * 16, EMBW * 16]))


@Cache
def EmbModel() -> Lazy[torch.nn.Module]:
    return Lazy(lambda: mtime(EMBF), lambda: torch.hub.load(DINO_HUB, EMBF.split('_pretrain')[0], source='local', weights=EMBF).requires_grad_(False).eval())


@Cache
def Emb(f: str) -> Lazy[torch.Tensor]:
    return Lazy(lambda: max(Img(f).t(), EmbModel().t()), lambda: F.normalize((tofile(EmbModel()().forward_features(TF.normalize(Img(f)() / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[None])['x_norm_patchtokens'][0].half().unflatten(0, (EMBH, -1)), f) if mtime(f'{f}.{EMBH},{EMBW},384e') < max(Img(f).t(), EmbModel().t()) else fromfile(f'{f}.{EMBH},{EMBW},384e')).float(), 2, 2))


@Cache
def Lbl(f: str) -> Lazy[torch.Tensor]:
    return Lazy(lambda: mtime(f'{f}.99,3B'), lambda: mmap(f'{f}.99,3B'))


@Cache
def Xy() -> Lazy[tuple[torch.Tensor, torch.Tensor]]:
    return Lazy(lambda: max(Lbl(f).t() for f in argv), lambda: (torch.cat([Emb(f)()[*strip(Lbl(f)()[:, :2]).T.long()] for f in argv if Lbl(f).t()]), torch.cat([strip(Lbl(f)())[:, 2] for f in argv if Lbl(f).t()]).long()))


@Cache
def LinModel() -> Lazy[torch.nn.Module]:
    def _(_) -> torch.nn.Module:
        x, y = Xy()()
        m = torch.nn.Linear(384, CLS)
        o = torch.optim.Adam(m.parameters(), .2, weight_decay=1e-3)
        for _ in range(10):
            for _ in range(200):
                (_ := torch.nn.functional.cross_entropy(torch.constant_pad_nd(m(x), (0, 1)), y)).backward()
                o.step()
                o.zero_grad()
            print('los', _)
        return m.requires_grad_(False).eval()

    return Lazy(Xy().t, _)


@Cache
def KnnLogit(f: str) -> Lazy[torch.Tensor]:
    return Lazy(lambda: max(Emb(f).t(), Xy().t()), lambda: Emb(f)() @ Xy()()[0].T)


@Cache
def KnnPrd(f: str, resize=False) -> Lazy[torch.Tensor]:
    return Lazy(KnnLogit(f).t, lambda: Xy()()[1][(_ := TF.resize(KnnLogit(f)().permute(2, 0, 1), [EMBH * 16, EMBW * 16] if resize else [EMBH, EMBW]).max(0))[1]].masked_fill_(_[0] < .8, CLS))


@Cache
def LinLogit(f: str) -> Lazy[torch.Tensor]:
    return Lazy(lambda: max(Emb(f).t(), LinModel().t()), lambda: LinModel()()(Emb(f)()))


@Cache
def LinPrd(f: str, resize=False) -> Lazy[torch.Tensor]:
    return Lazy(LinLogit(f).t, lambda: TF.resize(torch.constant_pad_nd(LinLogit(f)(), (0, 1)).permute(2, 0, 1), [EMBH * 16, EMBW * 16] if resize else [EMBH, EMBW]).argmax(0))


def draw() -> None:
    img = torch.stack([Img(f)() for f in argv[CVSI:CVSI + CVSH * CVSW]]).permute(0, 2, 3, 1)
    match CVS:
        case 1:
            for i, f in enumerate(argv[CVSI:CVSI + CVSH * CVSW]):
                img[i] = (img[i] >> 1) & color(KnnPrd(f, True)())
        case 2:
            for i, f in enumerate(argv[CVSI:CVSI + CVSH * CVSW]):
                img[i] = (img[i] >> 1) & color(LinPrd(f, True)())
    cv2.imshow('lb', img.unflatten(0, (CVSH, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2).numpy())


def cb(i: str | int, x=0, y=0, *_) -> None:
    global CVSI, CVSH, CVSW, CVSC, CVS
    match i:
        case '1':
            CVSC = 0 % CLS
        case '2':
            CVSC = 1 % CLS
        case '3':
            CVSC = 2 % CLS
        case '4':
            CVSC = 3 % CLS
        case 'w':
            if CVSH > 1:
                CVSH -= 1
                draw()
        case 's':
            if (CVSH + 1) * CVSW <= len(argv):
                CVSH += 1
                CVSI = min(CVSI, len(argv) - CVSH * CVSW)
                draw()
        case 'a':
            if CVSW > 1:
                CVSW -= 1
                draw()
        case 'd':
            if CVSH * (CVSW + 1) <= len(argv):
                CVSW += 1
                CVSI = min(CVSI, len(argv) - CVSH * CVSW)
                draw()
        case '\t':
            if CVSI > 0:
                CVSI = max(CVSI - 2 * CVSH * CVSW, 0)
                draw()
        case 'q':
            if CVSI > 0:
                CVSI = max(CVSI - CVSH * CVSW, 0)
                draw()
        case 'e':
            if CVSI < len(argv) - CVSH * CVSW:
                CVSI = min(CVSI + CVSH * CVSW, len(argv) - CVSH * CVSW)
                draw()
        case 'r':
            if CVSI < len(argv) - CVSH * CVSW:
                CVSI = min(CVSI + 2 * CVSH * CVSW, len(argv) - CVSH * CVSW)
                draw()
        case '\x1b':
            exit()
        case 'f':
            argv[:] = [argv[i] for i in torch.cat([LinLogit(f)().log_softmax(3).max(3)[0].mean((1, 2)) for f in argv]).argsort()]
            CVSI = 0
            draw()
        case 'v':
            CVS = (CVS + 1) % 3
            draw()
        case cv2.EVENT_LBUTTONDOWN:
            append(Lbl(argv[CVSI + y // 16 // EMBH * CVSW + x // 16 // EMBW])(), y // 16 % EMBH, x // 16 % EMBW, CVSC)
            draw()
        case cv2.EVENT_RBUTTONDOWN:
            append(Lbl(argv[CVSI + y // 16 // EMBH * CVSW + x // 16 // EMBW])(), y // 16 % EMBH, x // 16 % EMBW, CLS)
            draw()


del argv[0]
cv2.namedWindow('lb', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('lb', cb)
draw()
while True:
    cb(chr(cv2.waitKey()))
