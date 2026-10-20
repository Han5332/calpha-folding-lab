"""Pack a folding trajectory into the scroll-to-fold page.
usage: build_site.py traj.json src.html out.html [artifact.html]
out.html is a complete page for any static host; artifact.html (optional) is the same content without the document
skeleton, for publishing as a claude.ai Artifact."""
import base64, json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
n = d['n']
N = np.array(d['native'], float).reshape(n, 3)
c0 = N.mean(0); N0 = N - c0

def fit(X):
    # Kabsch: rotate X (centred) onto the native, proper rotation only
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ N0)
    dd = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, dd]) @ Vt
    return Xc @ R

xyz, frames = [], []
for f in d['frames']:
    X = fit(np.array(f['x'], float).reshape(n, 3))
    xyz.append(np.round(X * 100).astype('<i2'))
    frames.append([f['step'], f['T'], f['E'], f['rmsd'], f['tm'], f['q'], f['rg'], f['ss']])
blob = base64.b64encode(np.concatenate(xyz).tobytes()).decode()
bench = [
    ('Chignolin', '1UAO', 10, 4.41, 0.26), ('Trp-cage', '1L2Y', 20, 2.15, 0.23), ('Villin headpiece', '1YRF', 35, 2.04, 0.61),
    ('WW domain', '1E0L', 37, 3.77, 0.56), ('Homeodomain', '1ENH', 54, 3.19, 0.64), ('Protein G', '1PGA', 56, 2.89, 0.61),
    ('Ubiquitin', '1UBQ', 76, 2.69, 0.75), ('CheY', '3CHY', 128, 2.81, 0.77), ('T4 lysozyme', '2LZM', 164, 4.21, 0.71),
    ('Adenylate kinase', '4AKE', 214, 4.06, 0.74)]
data = {'n': n, 'seq': d['seq'], 'ssNative': d['ssNative'], 'native': [round(v, 2) for v in N0.ravel().tolist()], 'restraints': d['restraints'],
        'final': d['final'], 'frames': frames, 'xyz': blob,
        'bench': [{'name': a, 'id': b, 'n': c, 'rmsd': r, 'tm': t} for a, b, c, r, t in bench]}
src = open(sys.argv[2]).read()
assert '/*DATA*/null' in src
body = src.replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
i = body.index('</style>') + len('</style>')
open(sys.argv[3], 'w').write(head + body[:i] + '\n</head>\n<body>\n' + body[i:] + '\n</body>\n</html>\n')
if len(sys.argv) > 4:
    open(sys.argv[4], 'w').write(body)
print('frames', len(frames), 'page bytes', len(src) + len(json.dumps(data)))
