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
- It prints a **join link** (`http://192.168.x.x:8800/join?token=...`). Open that link on any other computer on the same
  network to add it: section 2. The dashboard at http://localhost:8800/ shows the link too, when opened on the Mac itself.
- The token in the link is saved in `run1/token.txt`, so the link stays the same when the run is resumed.
- Add `--c2 64` or `--cycles 5` to grow v8 into a wider or deeper network before training: see "What was tried" below.
  A resumed run needs the same two options again.
- Stop with Ctrl-C (it saves), continue later with `--resume` (same `--feat` and `--out`).

Useful options: `--epochs 4` (passes over the data), `--lr 2e-4` (peak learning rate, warm-up then cosine decay),
`--unit 8` (default chains per unit), `--eval-every 100` (held-out check, best weights saved as `run1/best.npz`).

## 2. Add a Windows laptop with an RTX 5060

Open the join link in the laptop's browser, download `train.bat` from the page and double-click it. The file already
holds the Mac's address and the run's token. It installs what is missing (Python 3.13 through winget, numpy, and the
CUDA 12.8 build of PyTorch that RTX 50-series GPUs need), downloads `contribute.py` and `evonet.py` from the Mac, and
starts training with units of 32 chains. Closing the window stops it; double-clicking again carries on.

- The window should say `joined as ... on cuda: NVIDIA GeForce RTX 5060 Laptop GPU (bf16)`. If it says `cpu`, update the
  NVIDIA driver and run `py -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`.
- Keep the laptop plugged in and set it not to sleep while it trains.
- By hand instead: copy `contribute.py` and `evonet.py` to the laptop and run
  `py contribute.py --server http://192.168.1.23:8800 --token YOUR_TOKEN --chains 32`.
- `train.bat` was written on a Mac and has not been run on Windows yet.

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

For Linux or another Mac, the join page has a one-line terminal command instead of `train.bat`.

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

Then change the `?v=` tag on `evo-model.js` in `index.html` so browsers fetch the new weights. The exporter reads the
network's size from the weights, so a grown network needs nothing else; the lab runs it more slowly in proportion to
its size. Benchmark before keeping it.

### What was tried on v8's data (November 2026)

One Apple M4 GPU, 5,284 training chains, scored on 467 held-out chains: distogram loss / long-range contact
precision (top L/2) / Q3. v8 scores 1.433 / 0.647 / 0.827.

- Fine-tuning v8 for three epochs changes nothing, whatever the recipe: AdamW 1.433, with a weight average 1.432,
  squeeze-excitation blocks 1.433, Schedule-Free AdamW 1.433, Cautious AdamW 1.433. v8 is at its plateau on this data.
- From scratch for one epoch at a peak rate of 2e-3: AdamW 1.681 / 0.462, Cautious AdamW 1.652 / 0.491, Schedule-Free
  AdamW 1.618 / 0.519. AdamW at 2e-4 (the default) reaches only 1.868 in one epoch, and 6e-3 is past the peak (1.698).
  Schedule-Free reaches AdamW's one-epoch loss in about half the chains.
- Schedule-Free from scratch for ten epochs: 1.451 / 0.636 / 0.819, still falling by about 0.005 per epoch. It does not
  beat v8, so v8 stays in the lab.
- Speed: one group per channel in `evonet.py` (the same weights and gradients as instance norm) gives 843 chains per
  minute against 786. Half precision gives nothing on an Apple GPU.

So a from-scratch run wants `--lr 2e-3`, and better weights need new chains, not a new optimizer.

## 5. More training data, for big proteins

Above 300 residues the lab reads a protein 128 columns of its alignment at a time. Windows of 128 consecutive residues
give the pairs close in sequence; up to 1,024 residues, each pair of far-apart 64-residue segments is then read on its
own, so the network predicts every residue pair. Before, pairs 128 or more apart got no prediction.

- `pick_chains.py` picks new chains: one per RCSB 30%-identity cluster that holds no training, held-out, benchmark or
  built-in chain, with an alignment in OpenProteinSet (public, on Amazon's open-data service).
- `make_data.py` downloads each chain's structure (RCSB) and alignment (OpenProteinSet) and writes training files:
  whole chains, or for long chains `--near 2 --far 4` (windows and far-apart segment pairs, which carry their real
  residue numbers in `idx`). `--tiles` writes every segment pair of a chain, for testing.
- `build_feat.py` computes the features exactly as `evo.js` does (`python build_feat.py --check` compares them).
- `holdout_v8.txt` lists v8's held-out chains; pass it with `--holdout` so added chains never move them into training.
  `--feat` takes several folders.

On 59 held-back proteins of 266 to 767 residues, v8 gets 27% of its top L/5 far contacts (128 or more apart) right
without ever training on them, and a short pilot on the first 1,600 long-chain files raised that to 29% while halving
the far restraints that are wrong. Coevolution computed on the whole chain finds far contacts no better than
coevolution computed two segments at a time (15% against 13% of the top L/5 on 12 chains), so the segment pairs lose
nothing.

## Files

| File | Role |
|---|---|
| `coordinator.py` | Work queue, optimizer, data and weight server, dashboard |
| `contribute.py` | Contributor loop: fetch, compute gradient, send |
| `evonet.py` | Network, inputs, augmentation and loss, shared by both sides |
| `dashboard.html` | Live view of contributors, throughput and loss |
| `join.html`, `join.bat` | The join link's page and the Windows starter it hands out |
| `pick_chains.py`, `make_data.py`, `build_feat.py` | New training chains: choosing them, downloading them, their features |
| `holdout_v8.txt` | The held-out chains every run so far has used |
| `export_app.py` | Writes trained weights into the lab's `evo-model.js` |
| `weights/v8.npz` | v8 weights, the starting point |
