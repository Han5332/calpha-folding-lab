"""Turn weights from a volunteer run into the lab's evo-model.js.

usage: python export_app.py RUN/best.npz ../evo-model.js [--note "volunteer run 1"]

The current evo-model.js is the template: its distance reference table is kept, and the weights (float16, base64), the
network's size and the training note change. Reload the lab with a new ?v= tag afterwards.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evonet as E  # noqa: E402

PREFIX = 'var FOLD_EVO_MODEL = '


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('weights'); ap.add_argument('model_js'); ap.add_argument('--note', default='volunteer training run')
    a = ap.parse_args()
    text = open(a.model_js, encoding='utf-8').read()
    i = text.index(PREFIX) + len(PREFIX)
    spec = json.loads(text[i:text.rindex(';')])
    W = E.from_npz(open(a.weights, 'rb').read())
    # the network's size is read from the weights, so a grown network exports too
    cfg = spec['config']
    cycles = sum(k.startswith('r2.') and k.endswith('.c1.weight') for k in W) // 4
    cfg['c1'], cfg['c2'], cfg['dil2'] = int(W['in1.weight'].shape[0]), int(W['in2.weight'].shape[0]), E.DIL2[:4] * cycles
    order = list(E.Net(cfg['c1'], cfg['c2'], cycles).state_dict())
    if set(order) != set(W):
        sys.exit('these weights are not for the MSA network')
    tensors, chunks, off = [], [], 0
    for name in order:
        arr = np.asarray(W[name], dtype=np.float16).ravel()
        tensors.append([name, list(np.asarray(W[name]).shape), off, int(arr.size)]); chunks.append(arr); off += arr.size
    spec['tensors'] = tensors
    spec['params'] = int(off)
    spec['data'] = base64.b64encode(np.concatenate(chunks).astype('<f2').tobytes()).decode()
    spec['training'] = dict(spec.get('training') or {}, note=a.note)
    head = text[:text.index(PREFIX)]
    open(a.model_js, 'w', encoding='utf-8').write(head + PREFIX + json.dumps(spec, separators=(',', ':')) + ';\n')
    print(f'wrote {a.model_js}: {off} parameters')


if __name__ == '__main__':
    main()
