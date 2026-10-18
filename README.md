# Cα Folding Lab

A protein-folding lab that runs entirely in the browser. It folds proteins of 8–300 residues as a chain of Cα beads
with Langevin dynamics and simulated annealing, then compares the result with the experimental structure from the
Protein Data Bank (RMSD, TM-score, GDT-TS, native contacts, secondary structure).

## Methods

- **MSA + AI (v8)**: a multiple sequence alignment of related sequences goes through a small convolutional network
  (a 1D net for secondary structure plus a 2D dilated residual net, 583k parameters) that predicts a Cα distance histogram
  for every residue pair. The predictions become distance restraints on top of the knowledge-based force field.
  The network was trained on about 5,900 PDB chains of 12–396 residues, with no homologs of the ten built-in test proteins.
- **Knowledge-based (v2)**: fragment and contact statistics from 375 unrelated PDB chains.
- **Simple physics (v1)**: Chou–Fasman propensities and hydrophobic attraction.
- **Native contacts (Gō)**: built from the PDB contacts; it knows the answer and shows the route.

Twelve replicas run in parallel Web Workers by default, with optional Monte Carlo restarts when a replica stalls.
Pair interactions use Verlet neighbour lists, which give results identical to the full pair loop and are 2–4× faster on
proteins above ~150 residues. The final model is the lowest-energy large cluster of late frames, scored with an extra
backbone-dihedral term so that register-shifted β-sheets do not win.

## Benchmark (v8)

Cα RMSD to the PDB structure, mean of three seeds × 12 replicas, consensus model:

| Protein | Residues | RMSD (Å) | TM-score |
|---|---|---|---|
| Chignolin (1UAO) | 10 | 4.4 | 0.26 |
| Trp-cage (1L2Y) | 20 | 2.2 | 0.23 |
| Villin headpiece (1YRF) | 35 | 2.0 | 0.61 |
| FBP28 WW domain (1E0L) | 37 | 3.8 | 0.56 |
| Engrailed homeodomain (1ENH) | 54 | 3.2 | 0.64 |
| Protein G B1 (1PGA) | 56 | 2.9 | 0.61 |
| Ubiquitin (1UBQ) | 76 | 2.7 | 0.75 |
| CheY (3CHY) | 128 | 2.8 | 0.77 |
| T4 lysozyme (2LZM) | 164 | 4.2 | 0.71 |
| Adenylate kinase (4AKE) | 214 | 4.1 | 0.74 |

On 8 of the 10 proteins the three seeds land within 0.4 Å of each other. TM-scores are harsh below about 20 residues.

## Running it

Serve the folder with any static web server and open it in a browser:

```bash
python3 -m http.server 8791
```

Then open http://localhost:8791. Opening `index.html` straight from disk won't work, because the Web Workers need a server.

You can upload your own structure (`.pdb`, `.cif`/mmCIF, or either gzipped) or type a sequence. MSA + AI needs an
alignment: the built-in proteins ship with one, and for anything else you upload an A3M or FASTA alignment (for example
from the ColabFold MMseqs2 server). Without one, the protein folds with the knowledge-based force field alone.

The three large built-in proteins (CheY, T4 lysozyme, adenylate kinase) ship with their coevolution maps and network
predictions precomputed in `evo-feat.js`, so they start folding within a few seconds; the stored result is used only when
the sequence, the bundled alignment and the network weights all match it. Anything else runs the full pipeline in the
browser, which for a ~200-residue protein takes 3–4 minutes before folding starts. Folding a large protein runs for tens
of minutes depending on the machine.

## Files

| File | Contents |
|---|---|
| `index.html` | Page layout and styles |
| `app.js` | Run control, analysis, charts, uploads |
| `engine.js` | Force field, Langevin dynamics, restarts, scoring, model selection |
| `worker.js` | Runs one replica in a Web Worker |
| `evo.js`, `evo-worker.js` | Alignment processing, coevolution statistics, network inference |
| `evo-model.js` | Network weights (float16, base64) |
| `evo-msa.js` | Filtered alignments for the built-in proteins |
| `evo-feat.js` | Precomputed coevolution maps and network predictions for the three large built-in proteins |
| `evo-examples.js` | Saved best-prediction runs |
| `kb.js` | Knowledge-based statistics |
| `natives.js` | The ten built-in PDB structures |
| `viewer3d.js` | three.js cartoon / atom viewer |
| `benchmark.js` | Measured RMSD / TM-score per version |
| `monitor/` | Local dashboard used while training the network |
| `distributed/` | Volunteer training: other computers contribute GPU or CPU time ([guide](distributed/README.md)) |

## Data

Structures are from the RCSB Protein Data Bank (1UAO, 1L2Y, 1YRF, 1E0L, 1ENH, 1PGA, 1UBQ, 3CHY, 2LZM, 4AKE). Alignments are from ColabFold /
MMseqs2 (UniRef30 and environmental databases).
