"""Turn weights from a volunteer run into the lab's evo-model.js.

usage: python export_app.py RUN/best.npz ../evo-model.js [--note "volunteer run 1"]

The current evo-model.js is the template: its network configuration and distance reference table are kept, and only
the weights (float16, base64) and the training note change. Reload the lab with a new ?v= tag afterwards.
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
    order = [t[0] for t in spec['tensors']]
    if set(order) != set(W):
        sys.exit('these weights do not match the network in ' + a.model_js)
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
