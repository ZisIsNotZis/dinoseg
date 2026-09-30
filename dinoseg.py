#!/bin/env python3.12
import os
import sys
import typing
import functools
from ctypes import CDLL, c_char, c_int, c_long

import numpy as np
import cv2
import torch
import torchvision

P = typing.ParamSpec('P')
T = typing.TypeVar('T')
CVSI = CVSC = 0
CVS = CLS = CVSH = CVSW = 1
EMBF = os.getenv('DINO_WEIGHTS', 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
DINO_HUB = os.getenv('DINO_HUB', 'dinov3')
EMBM = None
EMBH = EMBW = 40
EMB = 384
EMBQ = 'e'
LBLN = 99
TRN = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
globals().update((i, type(j)(os.environ[i])) for i, j in globals().items() if i.isupper() and i in os.environ)


def cdll(name: str) -> typing.Callable[[typing.Callable[P, T]], typing.Callable[P, int]]:
    def cdll(f: typing.Callable[P, T]) -> typing.Callable[P, int]:
        g = getattr(CDLL(name), f.__name__)
        g.argtypes = [f.__annotations__.get(i, c_int) for i in f.__code__.co_varnames[:f.__code__.co_argcount]]
        g.restype = f.__annotations__.get('return', c_int)
        return g

    return cdll


def vcache(f: typing.Callable[P, typing.Iterator[T]]) -> typing.Callable[P, list[T]]:
    @functools.wraps(f)
    def _(*args: P.args, _={}, **kwargs: P.kwargs) -> list[T]:
        for i in chunk([i for i in args if i not in _]):
            for i, j in zip(i, f(*i, **kwargs)):
                if j is not None:
                    _[i] = j
        return [_.get(i) for i in args]

    return _


@cdll('libc.so.6')
def mmap(ptr: c_long, length: c_long, flag, prot, fd, offset: c_long) -> c_long: ...


def fromfile(f: str) -> np.ndarray:
    return np.fromfile(f, f.split('.')[-1])[0]


def memmap(f: str, creat=False) -> np.ndarray:
    t = np.dtype(f.split('.')[-1])
    f = os.open(f, os.O_RDWR | creat * os.O_CREAT)
    os.ftruncate(f, t.itemsize)
    m = mmap(0, t.itemsize, 0x3, 0x01, f, 0)
    os.close(f)
    return np.frombuffer((c_char * t.itemsize).from_address(m), t)[0]


def chunk(a: list[T], n=8) -> typing.Iterator[list[T]]:
    for i in range(0, len(a), n):
        yield a[i:i + n]


@vcache
def Img(*f: str) -> typing.Iterator[torch.Tensor]:
    for i in f:
        yield torch.from_numpy(cv2.resize(cv2.imread(i), (EMBW * 16, EMBH * 16)))


@vcache
def Emb(*f: str) -> typing.Iterator[torch.Tensor]:
    global EMBM
    t = []
    for i in f:
        try:
            os.stat(f'{i}.{EMBH},{EMBW},{EMB}{EMBQ}')
        except FileNotFoundError:
            t.append(i)
    if t:
        print('emb', t)
        EMBM = EMBM or torch.hub.load(DINO_HUB, EMBF.split('_pretrain')[0], source='local', weights=EMBF)
        with torch.no_grad():
            for t, i in zip(t, EMBM.forward_features(TRN(torch.stack(Img(*t)).permute(0, 3, 1, 2) / 255))['x_norm_patchtokens']):
                i.numpy().astype(EMBQ, copy=0).tofile(f'{t}.{EMBH},{EMBW},{EMB}{EMBQ}')
    for i in f:
        yield torch.from_numpy(fromfile(f'{i}.{EMBH},{EMBW},{EMB}{EMBQ}').astype('f', copy=False))


@vcache
def Lbl(*f: str, creat=False) -> typing.Iterator[None | torch.Tensor]:
    for i in f:
        try:
            yield torch.from_numpy(memmap(f'{i}.{LBLN},3B', creat))
        except FileNotFoundError:
            yield


def color(i: torch.Tensor) -> torch.Tensor:
    return torch.where(i[..., None] < CLS, torch.from_numpy(cv2.cvtColor(torch.constant_pad_nd(180 // CLS * i[..., None], (0, 2), 255).flatten(0, 1).numpy(), cv2.COLOR_HSV2BGR)).unflatten(0, (len(i), -1)), 0)


def fit() -> None:
    global M
    M = torch.nn.Linear(EMB, CLS)
    if _ := Lbl.__kwdefaults__['_']:
        emb = torch.cat([i[*j[:j.any(1).argmin(), :2].T.long()] for i, j in zip(Emb(*_.keys()), _.values())])
        lbl = torch.cat([j[:j.any(1).argmin(), 2].long() for j in _.values()])
        o = torch.optim.Adam(M.parameters(), .2)
        _ = 1
        while _ > 1e-3:
            for _ in range(100):
                (_ := torch.nn.functional.cross_entropy(torch.constant_pad_nd(M(emb), (0, 1)), lbl)).backward()
                o.step()
                o.zero_grad()
            print('los', _)
    predict()


def predict() -> None:
    global PRD
    with torch.no_grad():
        _, PRD = torch.nn.functional.interpolate(M(torch.stack(Emb(*sys.argv[CVSI:CVSI + CVSH * CVSW]))).permute(0, 3, 1, 2), None, 16, 'bilinear', False, None, True).max(1)
    PRD = torch.where(_ > 0, PRD.byte(), CLS)
    draw()


def draw() -> None:
    img = torch.stack(Img(*sys.argv[CVSI:CVSI + CVSH * CVSW]))
    if CVS:
        img >>= 1
        img += color(PRD) >> 1
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
        case '5':
            CVSC = 4 % CLS
        case '6':
            CVSC = 5 % CLS
        case 'w':
            if CVSH > 1:
                CVSH -= 1
                predict()
        case 's':
            if (CVSH + 1) * CVSW <= len(sys.argv):
                CVSH += 1
                CVSI = min(CVSI, len(sys.argv) - CVSH * CVSW)
                predict()
        case 'a':
            if CVSW > 1:
                CVSW -= 1
                predict()
        case 'd':
            if CVSH * (CVSW + 1) <= len(sys.argv):
                CVSW += 1
                CVSI = min(CVSI, len(sys.argv) - CVSH * CVSW)
                predict()
        case '\t':
            if CVSI > 0:
                CVSI = max(CVSI - 2 * CVSH * CVSW, 0)
                predict()
        case 'q':
            if CVSI > 0:
                CVSI = max(CVSI - CVSH * CVSW, 0)
                predict()
        case 'e':
            if CVSI < len(sys.argv) - CVSH * CVSW:
                CVSI = min(CVSI + CVSH * CVSW, len(sys.argv) - CVSH * CVSW)
                predict()
        case 'r':
            if CVSI < len(sys.argv) - CVSH * CVSW:
                CVSI = min(CVSI + 2 * CVSH * CVSW, len(sys.argv) - CVSH * CVSW)
                predict()
        case '\x1b':
            exit()
        case 'f':
            sys.argv = [sys.argv[i] for i in torch.cat([torch.constant_pad_nd(M(torch.stack(i)), (0, 1)).log_softmax(3).max(3)[0].mean((1, 2)) for i in chunk(Emb(*sys.argv))]).exp().argsort()]
            CVSI = 0
            predict()
        case 'x':
            CVSC = 'x'
        case 'v':
            CVS = 1 - CVS
            draw()
        case cv2.EVENT_LBUTTONDOWN:
            i, y, x = CVSI + y // 16 // EMBH * CVSW + x // 16 // EMBW, y // 16 % EMBH, x // 16 % EMBW
            if CVSC == 'x':
                CVSC = 0
                try:
                    Lbl(sys.argv[i])[0].zero_()
                    fit()
                except FileNotFoundError:
                    pass
            else:
                a = Lbl(sys.argv[i], creat=True)[0]
                a[_[1] if (_ := (a[:, :2] == torch.tensor((y, x))).all(1).max(0))[0] else a.any(1).argmin()] = torch.tensor((y, x, CVSC))
                fit()
        case cv2.EVENT_RBUTTONDOWN:
            i, y, x = CVSI + y // 16 // EMBH * CVSW + x // 16 // EMBW, y // 16 % EMBH, x // 16 % EMBW
            if CVSC == 'x':
                CVSC = 0
                os.remove(sys.argv.pop(i))
                if CVSH * CVSW > len(sys.argv):
                    if CVSH > CVSW:
                        CVSH -= 1
                    else:
                        CVSW -= 1
                CVSI = min(CVSI, len(sys.argv) - CVSH * CVSW)
                draw()
            else:
                a = Lbl(sys.argv[i], creat=True)[0]
                a[_[1] if (_ := (a[:, :2] == torch.tensor((y, x))).all(1).max(0))[0] else a.any(1).argmin()] = torch.tensor((y, x, CLS))
                fit()


del sys.argv[0]
Lbl(*sys.argv)
cv2.namedWindow('lb', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('lb', cb)
fit()
while True:
    cb(chr(cv2.waitKey()))
