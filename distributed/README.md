# Volunteer training

Train the lab's MSA network with the GPUs and CPUs of several computers at once.

- **The coordinator** (`coordinator.py`) runs on the computer that has the training data. It keeps the weights and the
  optimizer, hands out work units of a few training chains, applies the gradients that come back, and serves a live
  dashboard.
- **Contributors** (`contribute.py`) run on any other computer with Python and PyTorch: an NVIDIA GPU, an Apple GPU or
  a plain CPU. Each one downloads the current weights and only the chains it is given (cached between units), computes
  the gradient and sends it back.

Gradients from all contributors feed one AdamW optimizer (asynchronous data-parallel training). Each chain's gradient
is clipped to norm 1 before averaging, as in single-machine training. Results computed on weights more than 8 updates
old are dropped and their chains handed out again, as are units that are not returned within 15 minutes.

Only arrays travel over the network (`.npz` read with `allow_pickle=False`), so neither side can run code on the other.
Every request needs the run's token. Anyone with the token can download the training data and send gradients, so share
it only with computers you trust. The coordinator checks every gradient's shape, finiteness and norm before using it.

## 1. Start the coordinator (the Mac with the data)

```bash
pip3 install numpy torch
cd distributed
python3 coordinator.py --feat /path/to/feat --exclude exclude_v8.txt --init weights/v8.npz --out run1 --host 0.0.0.0
```

- `--feat` is the folder of training `.npz` feature files (about 6,300 files, 3.1 GB). `--exclude` lists the chains
  related to the test proteins, so the benchmark stays fair.
- `--host 0.0.0.0` accepts computers on your network. Without it, only this Mac can connect. macOS may ask whether
  Python may accept incoming connections: allow it.
- The token is printed at start and saved in `run1/token.txt`. The dashboard is at http://localhost:8800/ and shows
  the full contributor command, token included, when opened on the Mac itself.
- Find the Mac's address for the other computers with `ipconfig getifaddr en0` (Wi-Fi). Both computers must be on the
  same network.
- Stop with Ctrl-C (it saves), continue later with `--resume` (same `--feat` and `--out`).

Useful options: `--epochs 4` (passes over the data), `--lr 2e-4` (peak learning rate, warm-up then cosine decay),
`--unit 8` (default chains per unit), `--eval-every 100` (held-out check, best weights saved as `run1/best.npz`).

## 2. Add a Windows laptop with an RTX 5060

RTX 50-series GPUs need a PyTorch build for CUDA 12.8 or newer; older builds install but cannot use the GPU.

1. Install Python 3.12 from python.org (3.9 or newer works) (tick "Add python.exe to PATH") and an up-to-date NVIDIA driver.
2. In PowerShell:

   ```powershell
   py -m pip install numpy
   py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
   py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   The last line should print `True NVIDIA GeForce RTX 5060 Laptop GPU`.
3. Copy `contribute.py` and `evonet.py` from this folder to the laptop (clone the repo, or download it as a ZIP from
   GitHub while signed in), then run, with the Mac's address and the token:

   ```powershell
   py contribute.py --server http://192.168.1.23:8800 --token YOUR_TOKEN --name windows-5060 --chains 32
   ```

   `--chains 32` asks for bigger units, so the GPU spends its time computing rather than waiting on Wi-Fi.
   Keep the laptop plugged in and set it not to sleep while it trains.

   On an NVIDIA GPU the contributor runs the network in **bf16** (bfloat16 autocast, with TF32 for the rest), which uses
   the tensor cores of RTX 30/40/50-series cards. Weights, gradients and losses stay float32; on the same chains, bf16
   gradients matched fp32 to a cosine similarity of 0.9999 (measured with the same bf16 arithmetic on a Mac; the CUDA
   speed-up itself was not measured here). Pass `--precision fp32` to turn it off.

### What the contributor does to keep the GPU busy

- **Prefetch:** while one unit computes, a background thread fetches the next unit, its chains and any new weights, and
  another sends the previous result. `--no-prefetch` goes back to doing them in turn. Because the next unit is fetched
  during the current one, its weights are usually one update old, well inside the coordinator's 8-update limit.
- **Half-size uploads:** gradients go up as float16 (1.2 MB instead of 2.3 MB per unit); the coordinator converts them to
  float32 before checking and applying them. `--send fp32` sends full precision. Older contributors that send float32
  keep working with the current coordinator, and the current contributor works with an older coordinator.

The same command works on Linux or another Mac (`python3` instead of `py`, and the CUDA wheel only for NVIDIA GPUs).

## 3. Train on the Mac itself (Apple GPU)

The Mac that runs the coordinator can contribute its GPU at the same time. In a second terminal:

```bash
cd distributed
python3 contribute.py --server http://127.0.0.1:8800 --token TOKEN --name mac-gpu --chains 16
```

- The coordinator's held-out checks run on the CPU (`--eval-device cpu`, the default), so they don't take the GPU away
  from this contributor.
- It runs in fp32 on Apple GPUs: on an M-series Mac, fp16 autocast was no faster and bf16 was half as fast.
- Gradients are summed on the GPU and copied back once per unit, instead of after every chain. Once the lengths are
  compiled, that raised throughput by about 27% (217 → 276 chains per minute on the Apple GPU, measured while other
  jobs shared the machine), with bit-for-bit identical gradients.
- Apple GPUs compile a kernel graph for every new chain length and then reuse it. To keep the number of lengths small,
  the contributor trims chains longer than 64 residues to a multiple of 8 (a random window, losing at most 7 residues;
  `--bucket 0` turns it off, and it is off by default on NVIDIA GPUs and CPUs). From a fresh start this doubled
  end-to-end throughput: 134 and 167 chains per minute against 73 and 68 without it, over 20 units of 16 chains each.
- A CPU contributor on the same Mac adds little (about 45 chains per minute) and competes with the GPU contributor and
  the coordinator for the processor, so leave it off while the GPU trains.

## 4. Use the result in the lab

```bash
python3 export_app.py run1/best.npz ../evo-model.js --note "volunteer run 1"
```

Then change the `?v=` tag on `evo-model.js` in `index.html` so browsers fetch the new weights. Benchmark before
keeping it: a run that only revisits v8's data mostly fine-tunes, and gains need new chains or more epochs.

## Files

| File | Role |
|---|---|
| `coordinator.py` | Work queue, optimizer, data and weight server, dashboard |
| `contribute.py` | Contributor loop: fetch, compute gradient, send |
| `evonet.py` | Network, inputs, augmentation and loss, shared by both sides |
| `dashboard.html` | Live view of contributors, throughput and loss |
| `export_app.py` | Writes trained weights into the lab's `evo-model.js` |
| `weights/v8.npz` | v8 weights, the starting point |
