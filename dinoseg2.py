#!/bin/env python3.12
from os import environ, getenv, remove
from sys import argv  # noqa

from matplotlib.cm import rainbow  # noqa
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from util import vcache, chunk

CVSI = CVSC = 0
CVS = CLS = CVSH = CVSW = 1
EMBF = getenv('DINO_WEIGHTS', 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
DINO_HUB = getenv('DINO_HUB', 'dinov3')
EMBH = EMBW = 40
EMB = 384
globals().update((i, type(j)(environ[i])) for i, j in globals().items() if i.isupper() and i in environ)


def append(a, *i):
    print('append', *i)
    a[a.any(1).numpy().argmin()] = torch.as_tensor(i)


def strip(a):
    return a[:a.any(1).numpy().argmin()]


@vcache
def Img(*f: str) -> list[torch.Tensor]:
    return [TF.resize(torch.from_numpy(cv2.imread(i)).permute(2, 0, 1), [EMBH * 16, EMBW * 16]) for i in f]


@vcache
def Emb(*f: str, m=[None]) -> list[torch.Tensor]:
    global E
    _ = []
    for i in f:
        try:
            open(f'{i}.{EMBH},{EMBW},{EMB}e')
        except FileNotFoundError:
            try:
                np.fromfile(f'{i}.{EMBH},{EMBW},{EMB}f', f'{EMBH},{EMBW},{EMB}f')[0].astype('e').tofile(f'{i}.{EMBH},{EMBW},{EMB}e')
                remove(f'{i}.{EMBH},{EMBW},{EMB}f')
            except RuntimeError:
                _.append(i)
    if _:
        print('emb', _, end='...', flush=True)
        m[0] = m[0] or torch.hub.load(DINO_HUB, EMBF.split('_pretrain')[0], source='local', weights=EMBF)
        with torch.inference_mode():
            for _, i in zip(_, m[0].forward_features(TF.normalize(torch.stack(Img(*_)) / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))['x_norm_patchtokens']):
                i.half().numpy().tofile(f'{_}.{EMBH},{EMBW},{EMB}e')
        print('ok', flush=True)
    return [F.normalize(torch.from_numpy(np.fromfile(f'{i}.{EMBH},{EMBW},{EMB}e', f'{EMBH},{EMBW},{EMB}e')[0].astype('f')), 2, 2) for i in f]


@vcache
def Lbl(*f: str, force=False) -> list[torch.Tensor]:
    _ = []
    for i in f:
        try:
            open(f'{i}.99,3B')
        except FileNotFoundError:
            if not force:
                _.append(None)
                continue
        Lbl.__kwdefaults__['_'][i] = i = torch.from_file(f'{i}.99,3B', True, 99 * 3, dtype=torch.uint8).view(-1, 3)
        _.append(i)
    return _


def Fulbl(*f: str, resize=False) -> list[torch.Tensor]:
    emb, lbl = map(torch.cat, zip(*[[i[*strip(j)[:, :2].T.long()], strip(j)[:, 2]] for i, j in zip(Emb(*Lbl.__kwdefaults__['_']), Lbl(*Lbl.__kwdefaults__['_']))]))
    _ = []
    for i in Emb(*f):
        fulbl = (i @ emb.T).permute(2, 0, 1)
        if resize:
            fulbl = TF.resize(fulbl, [EMBH * 16, EMBW * 16])
        cnf, fulbl = fulbl.max(0)
        fulbl = lbl[fulbl]
        fulbl[cnf < .8] = CLS
        _.append(fulbl)
    return _


def Prd(*f: str, resize=False) -> list[torch.Tensor]:
    global M
    emb, lbl = map(torch.cat, zip(*[[i[*strip(j)[:, :2].T.long()], strip(j)[:, 2]] for i, j in zip(Emb(*Lbl.__kwdefaults__['_']), Lbl(*Lbl.__kwdefaults__['_']))]))
    lbl = lbl.long()
    M = torch.nn.Linear(EMB, CLS)
    o = torch.optim.Adam(M.parameters(), .3, weight_decay=1e-3)
    for _ in range(5):
        for _ in range(200):
            (_ := torch.nn.functional.cross_entropy(torch.constant_pad_nd(M(emb), (0, 1)), lbl)).backward()
            o.step()
            o.zero_grad()
        print('los', _)
    with torch.inference_mode():
        prd = M(torch.stack(Emb(*f))).permute(0, 3, 1, 2)
    if resize:
        prd = TF.resize(prd, [EMBH * 16, EMBW * 16])
    cnf, prd = prd.max(1)
    prd[cnf < 0] = CLS
    return prd


def draw() -> None:
    img = torch.stack(Img(*argv[CVSI:CVSI + CVSH * CVSW])).permute(0, 2, 3, 1)
    if CVS and Lbl.__kwdefaults__['_']:
        img >>= 1
        match CVS:
            case 1:
                for i, lbl in zip(img, Fulbl(*argv[CVSI:CVSI + CVSH * CVSW], resize=True)):
                    i += torch.where(lbl[..., None] < CLS, torch.from_numpy(rainbow(lbl / CLS)[..., :3]).mul_(128).byte(), 0)
            case 2:
                for i, prd in zip(img, Prd(*argv[CVSI:CVSI + CVSH * CVSW], resize=True)):
                    i += torch.where(prd[..., None] < CLS, torch.from_numpy(rainbow(prd / CLS)[..., :3]).mul_(128).byte(), 0)
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
            argv[:] = [argv[i] for i in torch.cat([torch.constant_pad_nd(M(torch.stack(i)), (0, 1)).log_softmax(3).max(3)[0].mean((1, 2)) for i in chunk(Emb(*argv))]).exp().argsort()]
            CVSI = 0
            draw()
        case 'v':
            CVS = (CVS + 1) % 3
            draw()
        case cv2.EVENT_LBUTTONDOWN:
            append(Lbl(argv[CVSI + y // 16 // EMBH * CVSW + x // 16 // EMBW], force=True)[0], y // 16 % EMBH, x // 16 % EMBW, CVSC)
            draw()
        case cv2.EVENT_RBUTTONDOWN:
            append(Lbl(argv[CVSI + y // 16 // EMBH * CVSW + x // 16 // EMBW], force=True)[0], y // 16 % EMBH, x // 16 % EMBW, CLS)
            draw()


del argv[0]
Lbl(*argv)
cv2.namedWindow('lb', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('lb', cb)
draw()
while True:
    cb(chr(cv2.waitKey()))
