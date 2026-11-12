"""Build training .npz files for new chains: alignment from OpenProteinSet, structure from the RCSB.

usage: python make_data.py CHAINS.txt OUT_DIR [--near 2 --far 4 | --tiles] [--shard K/N]

CHAINS.txt has one chain per line: NAME FOLDER LENGTH RESOLUTION SEQUENCE (written by pick_chains.py). FOLDER is the
OpenProteinSet folder whose alignment has exactly this sequence. Nothing but the finished .npz files is written to
disk. A chain whose files disagree with its sequence is skipped and named in OUT_DIR/skipped.txt. As in the existing
training files, a chain keeps only the residues that have coordinates, and helix and strand come from the entry's
own HELIX and SHEET records.

With no option each chain becomes one file with features of the whole chain, as the lab computes them for proteins
of up to 300 residues. Longer proteins are read 128 columns at a time (evo.js predictLong), with features computed
from those columns of the alignment alone, and the other options build training files the same way:
  --near N   N files NAME_wSTART: a window of 128 consecutive residues
  --far M    M files NAME_pA_B: the 64 residues from A next to the 64 from B, further along the chain. Half of them
             are chosen among segment pairs that touch (some Cα pair under 12 Å), the rest at random.
  --tiles    every pair of 64-residue segments on the lab's grid, at full alignment depth only: for testing
"""
import argparse
import gzip
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_feat as B  # noqa: E402

OPS = 'https://openfold.s3.amazonaws.com/pdb/{}/a3m/{}_hits.a3m'
RCSB = 'https://files.rcsb.org/download/{}.cif.gz'
WIN = 128                                       # evo.js WIN
THREE = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), B.ALPH))
TOKEN = re.compile(r"""'((?:[^']|'(?!\s|$))*)'|"((?:[^"]|"(?!\s|$))*)"|(\S+)""")


def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except OSError as e:
            err = e
        time.sleep(2 * (k + 1))
    raise err


def cif_tables(text, wanted):
    """The mmCIF categories in `wanted` as {category: [row dict, ...]}, from loops and from single key-value records."""
    toks, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(';'):                              # a text block runs to the next line starting with ;
            j = i + 1
            while j < len(lines) and not lines[j].startswith(';'):
                j += 1
            toks.append((' '.join([line[1:]] + lines[i + 1:j]), True))
            i = j + 1
            continue
        if line.startswith(('ATOM', 'HETATM')) and ' CA ' not in line:
            i += 1                                            # only Cα rows are needed, and atoms are most of the file
            continue
        if not line.startswith('#'):
            for m in TOKEN.finditer(line):
                if m.group(3) is None:
                    toks.append((m.group(1) if m.group(1) is not None else m.group(2), True))
                elif m.group(3).startswith('#'):
                    break
                else:
                    toks.append((m.group(3), False))
        i += 1
    out, k, n = {c: [] for c in wanted}, 0, len(toks)
    name = lambda t: not t[1] and t[0].startswith('_')
    while k < n:
        t, quoted = toks[k]
        if not quoted and t == 'loop_':
            k += 1
            cols = []
            while k < n and name(toks[k]):
                cols.append(toks[k][0]); k += 1
            start = k
            while k < n and not (not toks[k][1] and (toks[k][0].startswith(('_', 'data_')) or toks[k][0] == 'loop_')):
                k += 1
            cat = cols[0].split('.')[0][1:] if cols else ''
            if cat in out:
                keys = [c.split('.', 1)[1] for c in cols]
                vals = [v for v, _ in toks[start:k]]
                out[cat] += [dict(zip(keys, vals[r:r + len(keys)])) for r in range(0, len(vals) - len(keys) + 1, len(keys))]
        elif name(toks[k]) and k + 1 < n:
            cat, _, key = t[1:].partition('.')
            if cat in out:
                if not out[cat]:
                    out[cat].append({})
                out[cat][0][key] = toks[k + 1][0]
            k += 2
        else:
            k += 1
    return out


def structure(text, chain):
    """(sequence, Cα coordinates (L, 3) with NaN where missing, secondary structure) of one author chain. The sequence
    is the full one the experiment used (SEQRES), so residues without coordinates stay in, as NaN."""
    T = cif_tables(text, ('atom_site', 'entity_poly_seq', 'struct_conf', 'struct_sheet_range'))
    atoms = [a for a in T['atom_site'] if a['auth_asym_id'] == chain and a['label_atom_id'] == 'CA' and a['label_seq_id'].isdigit()]
    if not atoms:
        return None
    atoms = [a for a in atoms if a.get('pdbx_PDB_model_num', '1') == atoms[0].get('pdbx_PDB_model_num', '1')]
    mons = {}
    for r in T['entity_poly_seq']:
        if r['entity_id'] == atoms[0]['label_entity_id']:
            mons.setdefault(int(r['num']), r['mon_id'])
    L = max(mons) if mons else 0
    seq = ''.join(THREE.get(mons.get(i + 1, ''), 'X') for i in range(L))
    xyz = np.full((L, 3), np.nan, np.float32)
    for a in reversed(atoms):                                  # reversed: the first alternate location wins
        i = int(a['label_seq_id']) - 1
        if i < L:
            xyz[i] = float(a['Cartn_x']), float(a['Cartn_y']), float(a['Cartn_z'])
    ss = np.full(L, 2, np.int8)
    ss[np.isnan(xyz[:, 0])] = -1
    for rows, code in ((T['struct_sheet_range'], 1), (T['struct_conf'], 0)):
        for r in rows:
            if r.get('beg_auth_asym_id') != chain or (code == 0 and not r.get('conf_type_id', '').startswith('HELX')):
                continue
            if r['beg_label_seq_id'].isdigit() and r['end_label_seq_id'].isdigit():
                seg = ss[int(r['beg_label_seq_id']) - 1:int(r['end_label_seq_id'])]
                seg[seg >= 0] = code
    return seq, xyz, ss


def distances(xyz):
    return np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1))


def a3m_rows(data):
    """Aligned rows of an A3M file as evo.js parseAlignment reads them: insertions (lowercase) removed. Query first."""
    rows, cur = [], []
    for line in data.decode('latin-1').split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if cur: rows.append(''.join(cur))
            cur = []
        elif line and not line.startswith('#'):
            cur.append(re.sub(r'[a-z.\s]', '', line).upper())
    if cur: rows.append(''.join(cur))
    return rows


def load(name, folder, seq):
    """(sequence, alignment rows as an (N, L) byte array, Cα distances, secondary structure) of one chain, kept to the
    residues that have coordinates, or a string saying why the chain is skipped."""
    cif = fetch(RCSB.format(name.split('_')[0]))
    if cif is None:
        return 'no structure file'
    got = structure(gzip.decompress(cif).decode('latin-1'), name.split('_', 1)[1])
    if got is None or len(got[0]) != len(seq):
        return 'structure does not match the sequence length'
    sseq, xyz, ss = got
    if sum(a != b for a, b in zip(sseq, seq) if a != 'X') > 0.02 * len(seq) or np.isfinite(xyz[:, 0]).mean() < 0.7:
        return 'structure disagrees with the sequence or is mostly missing'
    rows = []
    for db in ('bfd_uniclust', 'uniref90'):
        a3m = fetch(OPS.format(folder, db))
        r = a3m_rows(a3m) if a3m else []
        if r and (len(r[0]) != len(seq) or sum(x != y for x, y in zip(r[0], seq)) > 0.02 * len(seq)):
            return f'the {db} alignment is for another sequence'
        if r and not rows:
            seq = r[0]                           # a few modified residues are spelled differently: the alignment's query rules
        rows += r[1:]
        if len(B.process(seq, rows, 201)) > 200:
            break                                # deep enough without the second, bigger file
    if not rows:
        return 'no alignment'
    # like the existing training files and the lab's own PDB reader: only residues that have coordinates
    keep = np.flatnonzero(np.isfinite(xyz[:, 0]))
    A = np.frombuffer(''.join(r for r in rows if len(r) == len(seq)).encode('latin-1'), np.uint8).reshape(-1, len(seq))[:, keep]
    return ''.join(seq[i] for i in keep), A, distances(xyz[keep]), ss[keep]


def build(name, folder, seq, near, far, tiles, seed):
    """{file name: arrays} for one chain, or a string saying why it was skipped."""
    got = load(name, folder, seq)
    if isinstance(got, str):
        return got
    seq, A, D, ss = got
    L, H = len(seq), WIN // 2
    if not (near or far or tiles):
        return {name: B.example(seq, [bytes(r).decode('latin-1') for r in A], D, ss, seed)} if L >= 40 else 'too few residues with coordinates'
    if L < 2 * WIN:
        return 'too few residues with coordinates'

    def cut(a, b, depths=(64, 8, 1)):
        """Training arrays for the 64 residues from a joined to the 64 from b."""
        cols = np.r_[a:a + H, b:b + H]
        return B.example(''.join(seq[i] for i in cols), [bytes(r).decode('latin-1') for r in A[:, cols]], D[np.ix_(cols, cols)],
                         ss[cols], seed, None if b == a + H else cols, depths)

    rng, out = random.Random(seed), {}
    grid = list(range(0, L - H, H)) + [L - H]                                   # segment starts: every 64, and the last 64
    windows = grid[:-2] + [L - WIN]                                             # window starts, as predictLong lays them
    if tiles:
        out = {f'{name}_w{s}': cut(s, s + H, ()) for s in windows}
        out.update({f'{name}_p{a}_{b}': cut(a, b, ()) for a in grid for b in grid if b - a > H})
        return out
    for s in rng.sample(windows, min(near, len(windows))):
        out[f'{name}_w{s}'] = cut(s, s + H)
    pairs = [(a, b) for a, b in ((rng.randrange(L - H + 1), rng.randrange(L - H + 1)) for _ in range(60)) if b - a > H]
    picked = [p for p in pairs if D[p[0]:p[0] + H, p[1]:p[1] + H].min() < 12][:far // 2]
    picked += [p for p in pairs if p not in picked][:far - len(picked)]
    for a, b in picked:
        out[f'{name}_p{a}_{b}'] = cut(a, b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chains'); ap.add_argument('out')
    ap.add_argument('--near', type=int, default=0); ap.add_argument('--far', type=int, default=0)
    ap.add_argument('--tiles', action='store_true'); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--limit', type=int, default=0, help='stop after this many chains of the list (0 = all)')
    a = ap.parse_args()
    torch.set_num_threads(2)                     # several of these run side by side
    k, n = map(int, a.shard.split('/'))
    os.makedirs(a.out, exist_ok=True)
    chains = [l.split() for l in open(a.chains) if l.strip()]
    chains = (chains[:a.limit] if a.limit else chains)[k::n]
    done = {re.sub(r'(_w\d+|_p\d+_\d+)?\.npz$', '', f) for f in os.listdir(a.out) if f.endswith('.npz')}
    skipped = os.path.join(a.out, 'skipped.txt')
    done |= {l.split()[0] for l in open(skipped)} if os.path.exists(skipped) else set()
    for i, (name, folder, _, _, seq) in enumerate(chains):
        if name in done:
            continue
        t = time.time()
        try:
            got = build(name, folder, seq, a.near, a.far, a.tiles, i)
        except Exception as e:                    # one bad chain must not stop a ten-hour run
            got = f'{type(e).__name__}: {e}'
        if isinstance(got, str):
            open(skipped, 'a').write(f'{name} {got}\n')
        else:
            for fn, arrays in got.items():
                tmp = os.path.join(a.out, fn + '.part')
                with open(tmp, 'wb') as f:
                    np.savez_compressed(f, **arrays)
                os.replace(tmp, os.path.join(a.out, fn + '.npz'))
        print(f'{i + 1}/{len(chains)} {name} L={len(seq)} {got if isinstance(got, str) else "ok"} {time.time() - t:.1f}s', flush=True)


if __name__ == '__main__':
    main()
