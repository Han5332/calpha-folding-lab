"""Contribute this computer's GPU or CPU to training the Cα Folding Lab's MSA network.

Repeatedly: ask the coordinator for a work unit (a few training chains), download the current weights and those chains
(cached between units), compute the training gradient, send it back. While one unit computes, a background thread sends
the previous result and fetches the next unit, so the GPU does not wait on the network. Stop any time with Ctrl-C.

usage:
  python contribute.py --server http://COORDINATOR:8800 --token TOKEN [--name my-laptop] [--device auto|cuda|mps|cpu]
                       [--chains 32] [--precision auto|bf16|fp32]

Needs Python 3.9+ with numpy and PyTorch (pip install numpy torch). Only arrays travel over the network (.npz files
read with allow_pickle=False), so the coordinator cannot run code on this computer.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
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


def device_label(dev, precision='fp32'):
    if dev.type == 'cuda':
        base = 'cuda: ' + torch.cuda.get_device_name(dev)
    elif dev.type == 'mps':
        base = 'mps: Apple GPU'
    else:
        base = f'cpu: {platform.processor() or platform.machine()} x{torch.get_num_threads()}'
    return base + ('' if precision == 'fp32' else f' ({precision})')


def fetch_unit(cl, a, have_v):
    """Next work unit with everything it needs: its chains (from the cache or the coordinator) and, when the weights
    moved on since `have_v`, the new weights as arrays. Runs in the background thread."""
    while True:
        w = cl.json('/api/work' + (f'?chains={a.chains}' if a.chains else ''))
        if 'wait' not in w:
            break
        time.sleep(w['wait'])
    if w.get('done'):
        return w, None, None, 0.0
    t0 = time.time()
    weights = None
    if have_v != w['version']:
        data, hdr = cl.call('/api/weights')
        weights = (E.from_npz(data), int(hdr.get('X-Version', w['version'])))
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
    return w, weights, chains, time.time() - t0


def send_result(cl, w, used_v, grads, ld, ls, secs, dtype, log):
    """Upload one unit's gradient (float16 by default: half the bytes; the coordinator upcasts) and print the outcome."""
    q = f"/api/result?unit={w['unit']}&version={used_v}&ld={ld:.5f}&ls={ls:.5f}&secs={secs:.3f}"
    r = json.loads(cl.call(q, E.to_npz(grads, dtype), 'application/octet-stream')[0])
    state = f"update {r['version']}" if r.get('ok') else f"not used ({r.get('why')})"
    log(f'{len(w["files"])} chains, loss {ld:.3f}, {secs:.1f} s compute, {state}')
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--server', required=True); ap.add_argument('--token', required=True)
    ap.add_argument('--name', default=socket.gethostname().split('.')[0])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--cache', default=os.path.join(os.path.expanduser('~'), '.calpha-contrib'))
    ap.add_argument('--threads', type=int, default=0, help='CPU threads for torch (0 = torch default)')
    ap.add_argument('--chains', type=int, default=0, help='chains per work unit (0 = coordinator default; 32 suits a fast GPU on Wi-Fi)')
    ap.add_argument('--precision', default='auto', choices=['auto', 'bf16', 'fp32'],
                    help='auto = bf16 autocast (plus TF32) on CUDA GPUs that support it, fp32 elsewhere')
    ap.add_argument('--bucket', type=int, default=-1,
                    help='trim chains over 64 residues to a multiple of this (-1 = 8 on Apple GPUs, else 0 = off)')
    ap.add_argument('--send', default='fp16', choices=['fp16', 'fp32'], help='gradient upload format (fp16 halves the upload)')
    ap.add_argument('--no-prefetch', action='store_true', help='fetch and send in turn instead of in the background')
    ap.add_argument('--max-units', type=int, default=0, help='stop after this many units (0 = until training ends)')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = E.pick_device(a.device)
    precision = E.resolve_precision(a.precision, dev)
    bucket = a.bucket if a.bucket >= 0 else (8 if dev.type == 'mps' else 0)
    send_dtype = np.float16 if a.send == 'fp16' else np.float32
    os.makedirs(a.cache, exist_ok=True)
    cl = Client(a.server, a.token)
    wait = 5
    while True:
        try:
            j = cl.json('/api/join', {'name': a.name, 'device': device_label(dev, precision)})
            break
        except urllib.error.HTTPError as e:
            sys.exit(f'could not join: {e.code} {e.read().decode(errors="replace")[:200]}')
        except OSError as e:
            # the coordinator may not be up yet, or this computer is not on its network: keep trying
            print(f'cannot reach {a.server} ({e}); trying again in {wait} s', flush=True)
            time.sleep(wait); wait = min(wait * 2, 60)
    cl.worker, cfg = j['worker'], j['config']
    net = E.Net(cfg['c1'], cfg['c2']).to(dev)
    print(f'joined as {a.name} on {device_label(dev, precision)}')
    lock, done, have_v, backoff = threading.Lock(), [0], None, 2

    def log(msg):
        with lock:
            done[0] += 1
            print(f'unit {done[0]}: {msg}', flush=True)

    pool = ThreadPoolExecutor(max_workers=2)      # one thread fetches the next unit, one sends the last result
    nxt, pending, issued = None, None, 0
    while True:
        try:
            got = nxt.result() if nxt else fetch_unit(cl, a, have_v)
            nxt = None
            w, weights, chains, dl = got
            if w.get('done'):
                if pending: pending.result()
                print('training is finished, thank you!'); return
            if weights:
                E.load_weights(net, weights[0]); net.to(dev); have_v = weights[1]
            used_v = have_v
            issued += 1
            if not a.no_prefetch and not (a.max_units and issued >= a.max_units):
                nxt = pool.submit(fetch_unit, cl, a, have_v)       # overlaps with the compute below
            t1 = time.time()
            grads, ld, ls = E.unit_gradient(net, chains, dev, w['seed'], cfg['clip'], cfg['crop_p'], cfg['maxl'], precision, bucket)
            secs = time.time() - t1
            if pending: pending.result()                          # keep at most one upload in flight
            pending = pool.submit(send_result, cl, w, used_v, grads, ld, ls, secs, send_dtype, log)
            if a.no_prefetch: pending.result(); pending = None
            backoff = 2
            if a.max_units and issued >= a.max_units:
                if pending: pending.result()
                return
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print('\nstopped'); return
        except urllib.error.HTTPError as e:
            nxt = pending = None
            if e.code == 409:          # the coordinator restarted: join again
                cl.worker = cl.json('/api/join', {'name': a.name, 'device': device_label(dev, precision)})['worker']
                have_v = None
                continue
            print(f'server said {e.code}: {e.read().decode(errors="replace")[:200]}; retrying in {backoff} s')
            time.sleep(backoff); backoff = min(backoff * 2, 120)
        except (OSError, ValueError, RuntimeError) as e:
            nxt = pending = None
            print(f'{type(e).__name__}: {e}; retrying in {backoff} s')
            time.sleep(backoff); backoff = min(backoff * 2, 120)


if __name__ == '__main__':
    main()
