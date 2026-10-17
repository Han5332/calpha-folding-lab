# Cα Folding Lab

A protein-folding lab that runs entirely in the browser. It folds small proteins (8–120 residues) as a chain of Cα beads
with Langevin dynamics and simulated annealing, then compares the result with the experimental structure from the
Protein Data Bank (RMSD, TM-score, GDT-TS, native contacts, secondary structure).

## Methods

- **MSA + AI (v7)**: a multiple sequence alignment of related sequences goes through a small convolutional network
  (a 1D net for secondary structure plus a 2D dilated residual net, 583k parameters) that predicts a Cα distance histogram
  for every residue pair. The predictions become distance restraints on top of the knowledge-based force field.
  The network was trained on about 5,500 PDB chains with no homologs of the built-in test proteins.
- **Knowledge-based (v2)**: fragment and contact statistics from 375 unrelated PDB chains.
- **Simple physics (v1)**: Chou–Fasman propensities and hydrophobic attraction.
- **Native contacts (Gō)**: built from the PDB contacts; it knows the answer and shows the route.

Replicas run in parallel Web Workers, with optional Monte Carlo restarts when a replica stalls. The final model is the
lowest-energy large cluster of late frames, scored with an extra backbone-dihedral term so that register-shifted
β-sheets do not win.

## Running it

Serve the folder with any static web server and open it in a browser:

```bash
python3 -m http.server 8791
```

Then open http://localhost:8791. Opening `index.html` straight from disk won't work, because the Web Workers need a server.

You can upload your own structure (`.pdb`, `.cif`/mmCIF, or either gzipped), type a sequence, or upload an alignment
(A3M or FASTA). Alignments for new sequences come from the public ColabFold MMseqs2 server.

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
| `evo-examples.js` | Saved best-prediction runs |
| `kb.js` | Knowledge-based statistics |
| `natives.js` | The six built-in PDB structures |
| `viewer3d.js` | three.js cartoon / atom viewer |
| `benchmark.js` | Measured RMSD / TM-score per version |
| `monitor/` | Local dashboard used while training the network |

## Data

Structures are from the RCSB Protein Data Bank (1UAO, 1L2Y, 1YRF, 1ENH, 1PGA, 1UBQ). Alignments are from ColabFold /
MMseqs2 (UniRef30 and environmental databases).
