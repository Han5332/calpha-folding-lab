"""Coordinator for volunteer training of the MSA network.

Holds the weights and the optimizer, hands out work units (a few training chains each), and applies the gradients
contributors send back (asynchronous data-parallel SGD with one AdamW on the server). Serves the training data on
demand, so a contributor only downloads the chains it is given, and a live dashboard at http://HOST:PORT/.

usage:
  python coordinator.py --feat ../path/to/feat --init weights/v8.npz --out run1
  python coordinator.py --feat ../path/to/feat --out run1 --resume          # continue a stopped run
Then on each contributing computer:
  python contribute.py --server http://THIS-COMPUTER:8800 --token TOKEN

By default it listens on 127.0.0.1 only. Use --host 0.0.0.0 to accept computers on your network; anyone who can reach
the port and knows the token can download the training data and submit gradients, so share the token only with
machines you trust.
"""
import argparse
import glob
import hmac
import json
import math
import os
import random
import re
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # evonet.py sits next to this script
import evonet as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_RE = re.compile(r'^[A-Za-z0-9_]{1,40}$')
MAX_BODY = 16 << 20


class Coordinator:
    def __init__(s, a):
        s.a = a
        os.makedirs(a.out, exist_ok=True)
        s.lock = threading.Lock()
        s.started = time.time()
        # same file list and train / held-out split as the single-machine runs, so held-out chains stay unseen
        exclude = set(open(a.exclude).read().split()) if a.exclude else set()
        files = sorted(f for f in glob.glob(os.path.join(a.feat, '*.npz'))
                       if not os.path.basename(f).startswith('nat_') and os.path.basename(f)[:-4] not in exclude)
        if not files:
            raise SystemExit(f'no training .npz files in {a.feat}')
        random.Random(1).shuffle(files)
        nval = max(20, len(files) // 10)
        s.val, s.train = files[:nval], files[nval:]
        s.paths = {os.path.basename(f)[:-4]: f for f in s.train}
        s.val_eval = s.val[:a.val]
        s.net = E.Net(a.c1, a.c2)
        s.opt = torch.optim.AdamW(s.net.parameters(), lr=a.lr, weight_decay=1e-4)
        s.names = [n for n, _ in s.net.named_parameters()]
        s.shapes = {n: tuple(p.shape) for n, p in s.net.named_parameters()}
        s.token = a.token or s._load_or_make_token()
        st = os.path.join(a.out, 'state.json')
        if a.resume and os.path.exists(st):
            s.state = json.load(open(st))
            E.load_weights(s.net, E.from_npz(open(os.path.join(a.out, 'latest.npz'), 'rb').read()))
            op = os.path.join(a.out, 'optimizer.pt')
            if os.path.exists(op):
                s.opt.load_state_dict(torch.load(op, map_location='cpu'))
            print(f"resumed at update {s.state['version']}, epoch {s.state['epoch'] + 1}")
        else:
            if not a.init:
                raise SystemExit('--init weights.npz is needed for a new run')
            E.load_weights(s.net, E.from_npz(open(a.init, 'rb').read()))
            s.state = {'version': 0, 'epoch': 0, 'pos': 0, 'chains': 0, 'loss': [], 'val': [], 'best': None,
                       'rejected': 0, 'stale': 0}
        s.total_chains = len(s.train) * a.epochs
        s.total_updates = math.ceil(s.total_chains / a.unit)   # an estimate: contributors may ask for bigger units
        s.order = s._epoch_order(s.state['epoch'])
        s.outstanding = {}          # unit id -> {files, worker, issued, version}
        s.requeue = []              # files of units that timed out or were rejected
        s.workers = {}              # worker id -> stats
        s.weights_cache = (None, None)
        s.eval_busy = False
        s.finished = s.state['epoch'] >= a.epochs
        print(f'{len(s.train)} training chains, {len(s.val)} held out ({len(s.val_eval)} evaluated), '
              f'{a.unit} chains per unit, {s.total_updates} updates planned')

    # ---- bookkeeping ----
    def _load_or_make_token(s):
        p = os.path.join(s.a.out, 'token.txt')
        if os.path.exists(p):
            return open(p).read().strip()
        t = secrets.token_urlsafe(12)
        open(p, 'w').write(t + '\n')
        return t

    def _epoch_order(s, ep):
        names = sorted(s.paths)
        random.Random(1000 + ep).shuffle(names)
        return names

    def _lr(s):
        # short warm-up, then cosine decay to 10% over the planned updates
        v = s.state['version']
        warm = min(1.0, (v + 1) / 20)
        frac = min(1.0, s.state['chains'] / max(1, s.total_chains))   # schedule by chains seen, whatever the unit sizes
        return s.a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    def _save(s):
        open(os.path.join(s.a.out, 'latest.npz'), 'wb').write(s._weights_bytes())
        torch.save(s.opt.state_dict(), os.path.join(s.a.out, 'optimizer.pt'))
        tmp = os.path.join(s.a.out, 'state.tmp')
        json.dump(s.state, open(tmp, 'w'))
        os.replace(tmp, os.path.join(s.a.out, 'state.json'))

    def _weights_bytes(s):
        v, data = s.weights_cache
        if v != s.state['version']:
            data = E.to_npz({k: t.detach().cpu().numpy() for k, t in s.net.state_dict().items()})
            s.weights_cache = (s.state['version'], data)
        return data

    # ---- work units ----
    def next_unit(s, wid, want=None):
        size = max(1, min(int(want or s.a.unit), s.a.max_unit))
        with s.lock:
            now = time.time()
            for uid, u in list(s.outstanding.items()):
                if now - u['issued'] > s.a.timeout:
                    s.requeue.extend(u['files']); del s.outstanding[uid]
            if s.finished:
                return {'done': True}
            files = []
            while s.requeue and len(files) < size:
                files.append(s.requeue.pop())
            while len(files) < size:
                if s.state['pos'] >= len(s.order):
                    if s.outstanding or s.requeue:
                        break           # finish this epoch's stragglers before starting the next
                    s.state['epoch'] += 1; s.state['pos'] = 0
                    if s.state['epoch'] >= s.a.epochs:
                        s.finished = True; s._save()
                        break
                    s.order = s._epoch_order(s.state['epoch'])
                files.append(s.order[s.state['pos']]); s.state['pos'] += 1
            if not files:
                return {'done': True} if s.finished else {'wait': 5}
            uid = secrets.token_hex(6)
            s.outstanding[uid] = {'files': files, 'worker': wid, 'issued': now, 'version': s.state['version']}
            return {'unit': uid, 'files': files, 'version': s.state['version'], 'seed': random.getrandbits(31)}

    def submit(s, wid, uid, used_version, grads, ld, lsv, secs):
        with s.lock:
            u = s.outstanding.pop(uid, None)
            w = s.workers.get(wid)
            if u is None or w is None or u['worker'] != wid:
                return {'ok': False, 'why': 'unknown or expired unit'}
            reason = s._check(grads)
            stale = s.state['version'] - used_version
            if reason is None and stale > s.a.max_stale:
                reason = f'{stale} updates stale'
                s.state['stale'] += 1
            if reason:
                s.requeue.extend(u['files']); s.state['rejected'] += 1; w['rejected'] += 1
                return {'ok': False, 'why': reason}
            for g in s.opt.param_groups:
                g['lr'] = s._lr()
            for n, p in s.net.named_parameters():
                p.grad = torch.from_numpy(grads[n].astype(np.float32)).reshape(p.shape)
            s.opt.step()
            s.opt.zero_grad(set_to_none=True)
            s.state['version'] += 1
            s.state['chains'] += len(u['files'])
            s.state['loss'].append([s.state['version'], round(ld, 4), round(lsv, 4), round(time.time() - s.started, 1)])
            s.state['loss'] = s.state['loss'][-4000:]
            w['units'] += 1; w['chains'] += len(u['files']); w['secs'] += secs
            w['rate'] = 0.8 * w['rate'] + 0.2 * (len(u['files']) / max(secs, 1e-3)) if w['units'] > 1 else len(u['files']) / max(secs, 1e-3)
            v = s.state['version']
            if v % s.a.save_every == 0:
                s._save()
            if v % s.a.eval_every == 0 and not s.eval_busy:
                s.eval_busy = True
                snap = {k: t.detach().clone() for k, t in s.net.state_dict().items()}
                threading.Thread(target=s._evaluate, args=(v, snap), daemon=True).start()
            return {'ok': True, 'version': v}

    def _check(s, grads):
        if set(grads) != set(s.names):
            return 'wrong parameter names'
        sq = 0.0
        for n in s.names:
            g = grads[n]
            if tuple(g.shape) != s.shapes[n] and g.size != int(np.prod(s.shapes[n])):
                return f'wrong shape for {n}'
            if not np.all(np.isfinite(g)):
                return 'non-finite gradient'
            sq += float(np.sum(g.astype(np.float64) ** 2))
        # an average of per-chain gradients clipped to norm 1 cannot exceed norm 1
        if math.sqrt(sq) > s.a.clip * 1.01:
            return f'gradient norm {math.sqrt(sq):.2f} above the clip'
        return None

    def _evaluate(s, v, snap):
        try:
            net = E.Net(s.a.c1, s.a.c2)
            net.load_state_dict(snap)
            dev = E.pick_device(s.a.eval_device)
            net.to(dev)
            chains = [dict(np.load(f)) for f in s.val_eval]
            r = E.evaluate(net, chains, dev)
            r['version'] = v; r['time'] = round(time.time() - s.started, 1)
            with s.lock:
                s.state['val'].append(r)
                if s.state['best'] is None or r['loss'] < s.state['best']['loss']:
                    s.state['best'] = r
                    open(os.path.join(s.a.out, 'best.npz'), 'wb').write(
                        E.to_npz({k: t.cpu().numpy() for k, t in snap.items()}))
                s._save()
            print(f"update {v}: held-out loss {r['loss']:.3f}, long-range contacts {r['lr_p']:.3f}, Q3 {r['q3']:.3f}")
        finally:
            s.eval_busy = False

    def status(s, local):
        with s.lock:
            now = time.time()
            ws = [{'name': w['name'], 'device': w['device'], 'units': w['units'], 'chains': w['chains'],
                   'rate': round(w['rate'], 3), 'rejected': w['rejected'], 'seen': round(now - w['seen'], 1),
                   'busy': any(u['worker'] == wid for u in s.outstanding.values())} for wid, w in s.workers.items()]
            st = s.state
            out = {'version': st['version'], 'epoch': st['epoch'], 'epochs': s.a.epochs, 'pos': st['pos'],
                   'ntrain': len(s.train), 'nval': len(s.val_eval), 'unit': s.a.unit, 'planned': s.total_updates,
                   'chains': st['chains'], 'total_chains': s.total_chains, 'lr': s._lr(), 'loss': st['loss'][-1500:], 'val': st['val'], 'best': st['best'],
                   'rejected': st['rejected'], 'stale': st['stale'], 'outstanding': len(s.outstanding),
                   'workers': ws, 'uptime': round(now - s.started), 'finished': s.finished, 'evalEvery': s.a.eval_every}
            if local:
                out['token'] = s.token
            return out


def make_handler(C):
    page = open(os.path.join(HERE, 'dashboard.html'), 'rb').read()

    class H(BaseHTTPRequestHandler):
        server_version = 'CalphaCoordinator/1'

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            t = self.headers.get('X-Token', '')
            if not hmac.compare_digest(t.encode(), C.token.encode()):
                self._send(403, {'error': 'wrong token'}); return False
            return True

        def _worker(self):
            wid = self.headers.get('X-Worker', '')
            with C.lock:
                w = C.workers.get(wid)
                if w: w['seen'] = time.time()
            if not w:
                self._send(409, {'error': 'join first'})
            return wid if w else None

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ('/', '/index.html'):
                return self._send(200, page, 'text/html; charset=utf-8')
            if u.path == '/api/status':
                return self._send(200, C.status(self.client_address[0] in ('127.0.0.1', '::1')))
            if not self._authed():
                return
            if u.path == '/api/work':
                wid = self._worker()
                if wid:
                    q = parse_qs(u.query)
                    try: want = int(q.get('chains', ['0'])[0]) or None
                    except ValueError: want = None
                    self._send(200, C.next_unit(wid, want))
            elif u.path == '/api/weights':
                if self._worker():
                    with C.lock:
                        data, v = C._weights_bytes(), C.state['version']
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('X-Version', str(v))
                    self.end_headers()
                    self.wfile.write(data)
            elif u.path.startswith('/api/feat/'):
                name = u.path[len('/api/feat/'):]
                if self._worker():
                    p = C.paths.get(name) if NAME_RE.match(name) else None
                    if not p:
                        return self._send(404, {'error': 'no such training chain'})
                    self._send(200, open(p, 'rb').read(), 'application/octet-stream')
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get('Content-Length', 0))
            if n > MAX_BODY:
                return self._send(413, {'error': 'too large'})
            body = self.rfile.read(n)
            if not self._authed():
                return
            if u.path == '/api/join':
                try: info = json.loads(body or b'{}')
                except ValueError: info = {}
                name = re.sub(r'[^A-Za-z0-9 _.-]', '', str(info.get('name', 'contributor')))[:40] or 'contributor'
                device = re.sub(r'[^A-Za-z0-9 _.:()-]', '', str(info.get('device', '?')))[:60]
                wid = secrets.token_hex(8)
                with C.lock:
                    C.workers[wid] = {'name': name, 'device': device, 'units': 0, 'chains': 0, 'secs': 0.0, 'rate': 0.0,
                                      'rejected': 0, 'seen': time.time(), 'joined': time.time()}
                print(f'joined: {name} ({device})')
                return self._send(200, {'worker': wid, 'config': {'c1': C.a.c1, 'c2': C.a.c2, 'crop_p': C.a.crop_p,
                                                                   'maxl': C.a.maxl, 'clip': C.a.clip}})
            if u.path == '/api/result':
                wid = self._worker()
                if not wid:
                    return
                q = parse_qs(u.query)
                try:
                    # contributors send float16 (current) or float32 (older versions): check and apply in float32
                    grads = {k: v.astype(np.float32) for k, v in E.from_npz(body).items()}
                    r = C.submit(wid, q['unit'][0], int(q['version'][0]), grads, float(q['ld'][0]), float(q['ls'][0]),
                                 float(q['secs'][0]))
                except (KeyError, ValueError, OSError) as e:
                    return self._send(400, {'error': f'bad result: {e}'})
                return self._send(200, r)
            self._send(404, {'error': 'not found'})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--feat', required=True, help='folder of training .npz feature files')
    ap.add_argument('--init', help='starting weights (.npz), e.g. weights/v8.npz')
    ap.add_argument('--out', required=True, help='run folder: checkpoints, state, token')
    ap.add_argument('--resume', action='store_true', help='continue the run saved in --out')
    ap.add_argument('--exclude', help='file listing chain names to leave out (test-protein homologs)')
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8800)
    ap.add_argument('--token', help='shared secret contributors must send (default: generated, saved in --out)')
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--unit', type=int, default=8, help='chains per work unit unless a contributor asks for more')
    ap.add_argument('--max-unit', type=int, default=64, help='largest unit a contributor may ask for')
    ap.add_argument('--lr', type=float, default=2e-4, help='peak AdamW learning rate (2e-4 suits continuing from v8)'); ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--max-stale', type=int, default=8, help='drop gradients computed this many updates ago')
    ap.add_argument('--timeout', type=float, default=900, help='seconds before an unreturned unit is handed out again')
    ap.add_argument('--val', type=int, default=120, help='held-out chains evaluated')
    ap.add_argument('--eval-every', type=int, default=100); ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--eval-device', default='cpu')
    ap.add_argument('--c1', type=int, default=64); ap.add_argument('--c2', type=int, default=48)
    ap.add_argument('--crop-p', type=float, default=0.35); ap.add_argument('--maxl', type=int, default=256)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    C = Coordinator(a)
    if not C.state['val']:
        # score the starting weights first, so the dashboard shows whether the volunteers' updates help
        C.eval_busy = True
        threading.Thread(target=C._evaluate, args=(0, {k: t.detach().clone() for k, t in C.net.state_dict().items()}), daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(C))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # save on kill too
    shown = 'localhost' if a.host in ('127.0.0.1', '::1') else a.host
    print(f'dashboard: http://{shown}:{a.port}/')
    print(f'contributors run:  python contribute.py --server http://<this computer>:{a.port} --token {C.token}')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        with C.lock:
            C._save()
        print('saved; resume with --resume')


if __name__ == '__main__':
    main()
