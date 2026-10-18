"""The MSA network and its training loss, shared by the coordinator and every contributor.

Same architecture, inputs, targets and augmentation as the single-machine training that produced v1-v8: a 1D residual
net (secondary structure) feeding a 2D dilated residual net (18-bin Cα distogram). Weights travel as plain .npz files
(float32 arrays keyed by parameter name), never as pickles.
"""
import io
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

EDGES = np.arange(4, 21, dtype=np.float32)          # bins: <4, [4,5), ... [19,20), >=20  -> 18
NB = len(EDGES) + 1
P1 = 16
DIL1 = [1, 2, 4]
DIL2 = [1, 2, 4, 8] * 3
CROPS = [10, 14, 20, 28, 36, 48, 64, 96]
DEPTH_W = {'full': 0.5, 'd64': 0.2, 'd8': 0.15, 'd1': 0.15}


class Res1(nn.Module):
    def __init__(s, c, d):
        super().__init__()
        s.n1, s.c1 = nn.InstanceNorm1d(c, affine=True), nn.Conv1d(c, c, 3, padding=d, dilation=d)
        s.n2, s.c2 = nn.InstanceNorm1d(c, affine=True), nn.Conv1d(c, c, 3, padding=d, dilation=d)

    def forward(s, x):
        return x + s.c2(Fn.elu(s.n2(s.c1(Fn.elu(s.n1(x))))))


class Res2(nn.Module):
    def __init__(s, c, d):
        super().__init__()
        s.n1, s.c1 = nn.InstanceNorm2d(c, affine=True), nn.Conv2d(c, c, 3, padding=d, dilation=d)
        s.n2, s.c2 = nn.InstanceNorm2d(c, affine=True), nn.Conv2d(c, c, 3, padding=d, dilation=d)

    def forward(s, x):
        return x + s.c2(Fn.elu(s.n2(s.c1(Fn.elu(s.n1(x))))))


class Net(nn.Module):
    def __init__(s, c1=64, c2=48):
        super().__init__()
        s.in1 = nn.Conv1d(43, c1, 1)
        s.r1 = nn.ModuleList([Res1(c1, d) for d in DIL1])
        s.n1 = nn.InstanceNorm1d(c1, affine=True)
        s.ss = nn.Conv1d(c1, 3, 1)
        s.proj = nn.Conv1d(c1, P1, 1)
        s.in2 = nn.Conv2d(2 * P1 + 9, c2, 1)
        s.r2 = nn.ModuleList([Res2(c2, d) for d in DIL2])
        s.n2 = nn.InstanceNorm2d(c2, affine=True)
        s.out = nn.Conv2d(c2, NB, 1)

    def forward(s, f1, f2):
        h = s.in1(f1)
        for b in s.r1: h = b(h)
        h = Fn.elu(s.n1(h))
        ss = s.ss(h)
        p = s.proj(h)
        L = f1.shape[-1]
        pi = p[:, :, :, None].expand(-1, -1, L, L)
        pj = p[:, :, None, :].expand(-1, -1, L, L)
        g = s.in2(torch.cat([pi, pj, f2], 1))
        for b in s.r2: g = b(g)
        g = s.out(Fn.elu(s.n2(g)))
        g = 0.5 * (g + g.transpose(2, 3))
        return ss, g


def inputs(z, tag):
    f1 = z[tag + '_f1'].astype(np.float32)
    f2 = z[tag + '_f2'].astype(np.float32)
    L = f1.shape[0]
    ln = math.log(float(z[tag + '_neff']) + 1) / 5
    f1 = np.concatenate([f1, np.full((L, 1), ln, np.float32)], 1)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None]).astype(np.float32)
    extra = np.stack([np.full((L, L), ln, np.float32), np.log1p(sep) / 5, sep < 6, sep < 12, sep < 24], -1).astype(np.float32)
    f2 = np.concatenate([f2, extra], -1)
    return torch.from_numpy(f1.T[None].copy()), torch.from_numpy(f2.transpose(2, 0, 1)[None].copy())


def targets(z):
    D = z['dist'].astype(np.float32)
    L = D.shape[0]
    valid = ~np.isnan(D)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None])
    mask = valid & (sep >= 3)
    b = np.digitize(np.nan_to_num(D, nan=0), EDGES)
    return torch.from_numpy(b.astype(np.int64))[None], torch.from_numpy(mask)[None], torch.from_numpy(z['ss'].astype(np.int64))[None]


def tags_of(z):
    return [t for t in ('full', 'd64', 'd8', 'd1') if t + '_f1' in z]


def example(z, rng, crop_p=0.35, maxl=256):
    """One augmented training example: a random alignment depth, then a random window (always when longer than maxl)."""
    tg = tags_of(z)
    tag = rng.choices(tg, weights=[DEPTH_W[t] for t in tg])[0]
    f1, f2 = inputs(z, tag)
    b, m, ss = targets(z)
    L = f1.shape[-1]
    w = maxl if L > maxl else (rng.choice([c for c in CROPS if c < L]) if L > 14 and rng.random() < crop_p else None)
    if w:
        s0 = rng.randint(0, L - w)
        f1, f2 = f1[..., s0:s0 + w], f2[..., s0:s0 + w, s0:s0 + w]
        b, m, ss = b[:, s0:s0 + w, s0:s0 + w], m[:, s0:s0 + w, s0:s0 + w], ss[:, s0:s0 + w]
    return f1, f2, b, m, ss


def loss_of(net, ex, dev):
    f1, f2, b, m, ss = ex
    pss, g = net(f1.to(dev), f2.to(dev))
    ld = Fn.cross_entropy(g, b.to(dev), reduction='none')[m.to(dev)].mean()
    ls = Fn.cross_entropy(pss, ss.to(dev), ignore_index=-1)
    return ld + 0.5 * ls, ld.item(), ls.item()


def unit_gradient(net, chains, dev, seed, clip=1.0, crop_p=0.35, maxl=256):
    """Average of per-chain gradients, each clipped to norm `clip` as in single-machine training.
    chains: list of loaded .npz dicts. Returns ({name: float32 array}, mean distance loss, mean SS loss)."""
    rng = random.Random(seed)
    params = [p for p in net.parameters()]
    acc = [torch.zeros_like(p, device='cpu') for p in params]
    lds, lss, used = [], [], 0
    net.train()
    for z in chains:
        loss, ld, ls = loss_of(net, example(z, rng, crop_p, maxl), dev)
        if not math.isfinite(loss.item()):
            continue
        net.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, clip)
        for a, p in zip(acc, params):
            a += p.grad.detach().to('cpu')
        lds.append(ld); lss.append(ls); used += 1
    if not used:
        raise RuntimeError('no usable chain in this unit')
    names = [n for n, _ in net.named_parameters()]
    return {n: (a / used).numpy() for n, a in zip(names, acc)}, float(np.mean(lds)), float(np.mean(lss))


def evaluate(net, chains, dev):
    """Held-out metrics: distogram loss, top-L/2 long-range (|i-j| >= 24) contact precision at 8 Å, 3-state SS accuracy."""
    net.eval()
    out = {'loss': [], 'lr_p': [], 'q3': []}
    with torch.no_grad():
        for z in chains:
            f1, f2 = inputs(z, 'full')
            b, m, ss = targets(z)
            pss, g = net(f1.to(dev), f2.to(dev))
            g, pss = g.cpu(), pss.cpu()
            out['loss'].append(Fn.cross_entropy(g, b, reduction='none')[m].mean().item())
            pr = torch.softmax(g, 1)[0, :5].sum(0).numpy()
            D = z['dist'].astype(np.float32); L = D.shape[0]
            cand = [(pr[i, j], i, j) for i in range(L) for j in range(i + 24, L) if not np.isnan(D[i, j])]
            cand.sort(reverse=True)
            top = cand[:max(1, L // 2)]
            if top: out['lr_p'].append(np.mean([D[i, j] < 8 for _, i, j in top]))
            ok = ss[0] >= 0
            out['q3'].append((pss[0].argmax(0)[ok] == ss[0][ok]).float().mean().item())
    net.train()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


# ---- transport: .npz bytes, no pickle ----
def to_npz(arrays, dtype=np.float32):
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v, dtype=dtype) for k, v in arrays.items()})
    return buf.getvalue()


def from_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_weights(net, arrays):
    sd = net.state_dict()
    missing = [k for k in sd if k not in arrays]
    if missing:
        raise ValueError('weights are missing ' + ', '.join(missing[:3]))
    net.load_state_dict({k: torch.from_numpy(np.asarray(arrays[k], dtype=np.float32)).reshape(sd[k].shape) for k in sd})


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
