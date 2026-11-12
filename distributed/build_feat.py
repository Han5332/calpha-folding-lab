"""Training features for one chain, computed the way the lab computes them at prediction time.

A numpy port of evo.js (process -> weights -> features): the same alignment filter, sequence weights, mean-field DCA,
mutual information and profile. `python build_feat.py --check rows.json` compares it with numbers dumped from evo.js.

example(query, rows, dist, ss) returns the arrays of one training .npz: features at four alignment depths (everything
kept, 64, 8 and 1 sequence), the Cα distance matrix (NaN where a residue has no coordinates) and 3-state secondary
structure (0 helix, 1 strand, 2 coil, -1 unknown). When the columns are not one continuous stretch of the chain, idx
holds each column's residue number, so the network is told the real sequence separations.
"""
import json
import random
import sys

import numpy as np
import torch

ALPH, Q, GAP = 'ARNDCQEGHILKMFPSTWYV', 21, 20
MAX_SEQS, MIN_COV, MAX_ID, W_ID, LAM_DCA, LAM_MI = 1500, 0.5, 0.9, 0.8, 0.5, 0.05
IDX = np.full(256, GAP, np.uint8)
for _k, _c in enumerate(ALPH):
    IDX[ord(_c)] = _k


def encode(s):
    return IDX[np.frombuffer(s.encode('latin-1', 'replace'), np.uint8)]


def process(query, rows, max_seqs=MAX_SEQS):
    """evo.js process(): drop duplicates, rows covering < 50% of the query and rows >= 90% identical to one already
    kept, stop at max_seqs. Returns the kept rows as an (N, L) array of residue codes, query first."""
    L, seen = len(query), {query}
    K = np.empty((max_seqs, L), np.uint8)
    K[0] = encode(query)
    Kn, n = np.zeros(max_seqs, np.int64), 1
    Kn[0] = (K[0] != GAP).sum()
    for r in rows:
        if n >= max_seqs:
            break
        if len(r) != L or r in seen:
            continue
        seen.add(r)
        if L - r.count('-') < MIN_COV * L:
            continue
        x = encode(r)
        real = x != GAP
        same = ((K[:n] == x) & real).sum(1)
        if (same >= MAX_ID * np.minimum(Kn[:n], real.sum())).any():
            continue
        K[n], Kn[n] = x, real.sum()
        n += 1
    return K[:n].copy()


def features(X):
    """evo.js features(): (f1 (L, 42), f2 (L, L, 4), Neff) of an encoded alignment."""
    N, L = X.shape
    oh = np.zeros((N, L * Q), np.float32)
    oh[np.repeat(np.arange(N), L), np.tile(np.arange(L) * Q, N) + X.ravel()] = 1
    w = 1 / ((oh @ oh.T) >= W_ID * L).sum(1)                 # a sequence counts itself, as in evo.js
    neff = w.sum()
    oh = oh.astype(np.float64)
    Fi = (w / neff) @ oh
    Fij = (oh.T * (w / neff)) @ oh
    # mean-field DCA on the first 20 states, pseudocount LAM_DCA
    fi = (1 - LAM_DCA) * Fi + LAM_DCA / Q
    F = ((1 - LAM_DCA) * Fij + LAM_DCA / (Q * Q)).reshape(L, Q, L, Q)
    F[np.arange(L), :, np.arange(L), :] = np.eye(Q) * fi.reshape(L, Q, 1)
    C = (F - np.multiply.outer(fi.reshape(L, Q), fi.reshape(L, Q)))[:, :Q - 1, :, :Q - 1].reshape(L * (Q - 1), -1)
    del F
    chol = torch.linalg.cholesky(torch.from_numpy(np.ascontiguousarray(C)))    # raises when not positive definite
    e = -torch.cholesky_inverse(chol).numpy().reshape(L, Q - 1, L, Q - 1).transpose(0, 2, 1, 3)
    ra, cb, al = e.sum(3, keepdims=True) / Q, e.sum(2, keepdims=True) / Q, e.sum((2, 3), keepdims=True) / (Q * Q)
    dca = np.sqrt(((e - ra - cb + al) ** 2).sum((2, 3)))
    del e, C, chol
    # mutual information over all 21 states, pseudocount LAM_MI
    pi = (1 - LAM_MI) * Fi + LAM_MI / Q
    P = (1 - LAM_MI) * Fij + LAM_MI / (Q * Q)
    mi = (P * (np.log(P) - np.log(np.multiply.outer(pi, pi)))).reshape(L, Q, L, Q).sum((1, 3))
    del P, Fij
    f2 = np.zeros((L, L, 4), np.float32)
    off = ~np.eye(L, dtype=bool)
    for c, S in enumerate((dca, mi)):
        S = S * off
        m = S.sum(1) / (L - 1)
        M = (S - np.multiply.outer(m, m) / (S.sum() / (L * (L - 1)))) * off      # average product correction
        f2[..., c] = M
        f2[..., 2 + c] = (M - M.mean()) / (M.std() + 1e-6) * off
    f1 = np.zeros((L, 42), np.float32)
    q = X[0]
    f1[np.arange(L)[q < 20], q[q < 20]] = 1
    f1[:, 20:41] = Fi.reshape(L, Q)
    p = pi.reshape(L, Q)
    f1[:, 41] = -(p * np.log(p)).sum(1) / np.log(Q)
    return f1, f2, float(neff)


def example(query, rows, dist, ss, seed=0, idx=None, depths=(64, 8, 1)):
    """Arrays of one training .npz. rows: aligned sequences as long as the query (the query itself may be among them)."""
    X = process(query, rows)
    rng = random.Random(seed)
    out = {'dist': np.asarray(dist, np.float16), 'ss': np.asarray(ss, np.int8), 'nseq': np.int64(len(X))}
    if idx is not None:
        out['idx'] = np.asarray(idx, np.int32)
    for tag, depth in [('full', len(X))] + [(f'd{d}', d) for d in depths]:
        if tag != 'full' and depth >= len(X):
            continue                                    # the alignment is no deeper: the trainer skips missing depths
        pick = [0] + sorted(rng.sample(range(1, len(X)), depth - 1)) if depth < len(X) else slice(None)
        f1, f2, neff = features(X[pick])
        out[tag + '_f1'], out[tag + '_f2'], out[tag + '_neff'] = f1.astype(np.float16), f2.astype(np.float16), np.float32(neff)
    return out


if __name__ == '__main__':
    # self-check against evo.js: {"query", "rows", "kept": [row strings], "f1": [...], "f2": [...], "neff"} from node
    if len(sys.argv) != 3 or sys.argv[1] != '--check':
        sys.exit(__doc__)
    ref = json.load(open(sys.argv[2]))
    X = process(ref['query'], ref['rows'])
    kept = [''.join((ALPH + '-')[v] for v in r) for r in X]
    assert kept == ref['kept'], f'filter differs: {len(kept)} rows kept here, {len(ref["kept"])} in evo.js'
    f1, f2, neff = features(X)
    d1 = np.abs(f1.ravel() - np.array(ref['f1'], np.float32)).max()
    d2 = np.abs(f2.ravel() - np.array(ref['f2'], np.float32)).max()
    assert abs(neff - ref['neff']) < 1e-6 * neff and d1 < 1e-4 and d2 < 1e-3, (neff, ref['neff'], d1, d2)
    print(f'ok: {len(kept)} rows kept, Neff {neff:.2f}, largest difference from evo.js: f1 {d1:.1e}, f2 {d2:.1e}')
