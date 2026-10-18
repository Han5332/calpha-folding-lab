"""Contribute this computer's GPU or CPU to training the Cα Folding Lab's MSA network.

Repeatedly: ask the coordinator for a work unit (a few training chains), download the current weights and those chains
(cached between units), compute the training gradient, send it back. Stop any time with Ctrl-C.

usage:
  python contribute.py --server http://COORDINATOR:8800 --token TOKEN [--name my-laptop] [--device auto|cuda|mps|cpu]

Needs Python 3.8+ with numpy and PyTorch (pip install numpy torch). Only arrays travel over the network (.npz files
read with allow_pickle=False), so the coordinator cannot run code on this computer.
"""
import argparse
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # evonet.py sits next to this script
import evonet as E  # noqa: E402


class Client:
    def __init__(s, server, token, timeout=120):
        s.server, s.token, s.timeout, s.worker = server.rstrip('/'), token, timeout, ''

    def call(s, path, data=None, ctype='application/json'):
        req = urllib.request.Request(s.server + path, data=data, method='POST' if data is not None else 'GET')
        req.add_header('X-Token', s.token)
        if s.worker:
            req.add_header('X-Worker', s.worker)
        if data is not None:
            req.add_header('Content-Type', ctype)
        with urllib.request.urlopen(req, timeout=s.timeout) as r:
            return r.read(), r.headers

    def json(s, path, payload=None):
        body, _ = s.call(path, None if payload is None else json.dumps(payload).encode())
        return json.loads(body)


def device_label(dev):
    if dev.type == 'cuda':
        return 'cuda: ' + torch.cuda.get_device_name(dev)
    if dev.type == 'mps':
        return 'mps: Apple GPU'
    return f'cpu: {platform.processor() or platform.machine()} x{torch.get_num_threads()}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--server', required=True); ap.add_argument('--token', required=True)
    ap.add_argument('--name', default=socket.gethostname().split('.')[0])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--cache', default=os.path.join(os.path.expanduser('~'), '.calpha-contrib'))
    ap.add_argument('--threads', type=int, default=0, help='CPU threads for torch (0 = torch default)')
    ap.add_argument('--chains', type=int, default=0, help='chains per work unit (0 = coordinator default; 32 suits a fast GPU on Wi-Fi)')
    ap.add_argument('--max-units', type=int, default=0, help='stop after this many units (0 = until training ends)')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = E.pick_device(a.device)
    os.makedirs(a.cache, exist_ok=True)
    cl = Client(a.server, a.token)
    wait = 5
    while True:
        try:
            j = cl.json('/api/join', {'name': a.name, 'device': device_label(dev)})
            break
        except urllib.error.HTTPError as e:
            sys.exit(f'could not join: {e.code} {e.read().decode(errors="replace")[:200]}')
        except OSError as e:
            # the coordinator may not be up yet, or this computer is not on its network: keep trying
            print(f'cannot reach {a.server} ({e}); trying again in {wait} s', flush=True)
            time.sleep(wait); wait = min(wait * 2, 60)
    cl.worker, cfg = j['worker'], j['config']
    net = E.Net(cfg['c1'], cfg['c2']).to(dev)
    print(f'joined as {a.name} on {device_label(dev)}')
    have_v, done, backoff = None, 0, 2
    while True:
        try:
            w = cl.json('/api/work' + (f'?chains={a.chains}' if a.chains else ''))
            if w.get('done'):
                print('training is finished, thank you!'); return
            if 'wait' in w:
                time.sleep(w['wait']); continue
            t0 = time.time()
            if have_v != w['version']:
                data, hdr = cl.call('/api/weights')
                E.load_weights(net, E.from_npz(data))
                net.to(dev)
                have_v = int(hdr.get('X-Version', w['version']))
            chains = []
            for name in w['files']:
                p = os.path.join(a.cache, name + '.npz')
                if not os.path.exists(p):
                    data, _ = cl.call('/api/feat/' + name)
                    tmp = p + '.part'
                    open(tmp, 'wb').write(data)
                    os.replace(tmp, p)
                with np.load(p, allow_pickle=False) as z:
                    chains.append({k: z[k] for k in z.files})
            t1 = time.time()
            grads, ld, ls = E.unit_gradient(net, chains, dev, w['seed'], cfg['clip'], cfg['crop_p'], cfg['maxl'])
            secs = time.time() - t1
            q = f"/api/result?unit={w['unit']}&version={have_v}&ld={ld:.5f}&ls={ls:.5f}&secs={secs:.3f}"
            r = json.loads(cl.call(q, E.to_npz(grads), 'application/octet-stream')[0])
            done += 1
            state = f"update {r['version']}" if r.get('ok') else f"not used ({r.get('why')})"
            print(f'unit {done}: {len(chains)} chains, loss {ld:.3f}, {secs:.1f} s compute + {t1 - t0:.1f} s download, {state}',
                  flush=True)
            backoff = 2
            if a.max_units and done >= a.max_units:
                return
        except KeyboardInterrupt:
            print('\nstopped'); return
        except urllib.error.HTTPError as e:
            if e.code == 409:          # the coordinator restarted: join again
                cl.worker = cl.json('/api/join', {'name': a.name, 'device': device_label(dev)})['worker']
                continue
            print(f'server said {e.code}: {e.read().decode(errors="replace")[:200]}; retrying in {backoff} s')
            time.sleep(backoff); backoff = min(backoff * 2, 120)
        except (OSError, ValueError) as e:
            print(f'{type(e).__name__}: {e}; retrying in {backoff} s')
            time.sleep(backoff); backoff = min(backoff * 2, 120)


if __name__ == '__main__':
    main()
