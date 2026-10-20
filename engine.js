/*
 * Cα folding engine.
 *
 * One bead per residue at the Cα position, 3.8 Å virtual bonds, Langevin dynamics (BAOAB).
 * Two force fields:
 *   go  — structure-based (Clementi, Nymeyer & Onuchic 2000). Native contacts attract, everything
 *         else repels. It knows the answer; it shows how a chain gets there.
 *   kb  — knowledge-based, sequence-only. Position-specific angle/dihedral potentials come from sequence-matched
 *         fragments of a non-redundant PDB library (homologs of the test proteins removed); side-chain sites
 *         interact through a statistical contact potential derived from the same library. Never sees the target.
 *   seq — simple physics baseline. Chou–Fasman propensities set local preferences and a hydrophobic/electrostatic
 *         contact potential between virtual side-chain sites drives collapse. Never sees the target.
 * Plus the comparison metrics: optimal superposition, RMSD, TM-score, GDT-TS, contact overlap, Q.
 *
 * Units: Å, energy ε (one native contact), mass 1, kT in ε.
 */
(function (root) {
  'use strict';

  const DEG = Math.PI / 180;
  const BOND = 3.8;

  // Fauchère–Pliska octanol/water hydrophobicity (π).
  const HYDRO = { W: 2.25, I: 1.80, F: 1.79, L: 1.70, C: 1.54, M: 1.23, V: 1.22, Y: 0.96, P: 0.72, A: 0.31,
    T: 0.26, H: 0.13, G: 0.00, S: -0.04, Q: -0.22, N: -0.60, E: -0.64, D: -0.77, K: -0.99, R: -1.01 };
  const CHARGE = { K: 1, R: 1, D: -1, E: -1 };
  // Approximate Cα → side-chain centroid distance (Å).
  const SIDECHAIN = { G: 0, A: 1.5, S: 1.9, C: 2.2, T: 1.9, P: 1.9, V: 1.9, D: 2.4, N: 2.5, I: 2.3, L: 2.6, M: 3.0,
    E: 3.1, Q: 3.1, K: 3.5, H: 3.1, F: 3.4, R: 4.1, Y: 3.8, W: 3.8 };
  // Chou–Fasman (1978) propensities ×100: [helix, strand, turn].
  const CHOU_FASMAN = { A: [142, 83, 66], R: [98, 93, 95], N: [67, 89, 156], D: [101, 54, 146], C: [70, 119, 119],
    Q: [111, 110, 98], E: [151, 37, 74], G: [57, 75, 156], H: [100, 87, 95], I: [108, 160, 47], L: [121, 130, 59],
    K: [114, 74, 101], M: [145, 105, 60], F: [113, 138, 60], P: [57, 55, 152], S: [77, 75, 143], T: [83, 119, 96],
    W: [108, 137, 96], Y: [69, 147, 114], V: [106, 170, 50] };

  // Cα pseudo-bond angle θ (vertex i) and pseudo-dihedral φ (i-1, i, i+1, i+2) of each local state.
  // θ: [mean, sd]; φ: [mean, von Mises κ]. Means measured on the helices and strands of the bundled PDB set.
  const LOCAL = {
    H: { theta: [92 * DEG, 5 * DEG], phi: [50 * DEG, 12] },
    E: { theta: [121 * DEG, 10 * DEG], phi: [-165 * DEG, 3] },
    C: { theta: [108 * DEG, 16 * DEG], phi: [-120 * DEG, 0.7] },
  };

  const DEFAULTS = {
    go: { kBond: 100, kAngle: 20, kDih1: 1, kDih3: 0.5, contactCut: 8, minSep: 4, repel: 4 },
    seq: { kBond: 100, local: 0.7, hydro: 1.5, charge: 1, beta: 1, compact: 1, repel: 4, wellIn: 5, wellOut: 7.5 },
    kb: { kBond: 100, local: 1, contact: 1, burial: 1, beta: 1, compact: 1, repel: 4, wellIn: 5.5, wellOut: 7, siteRepel: 3.8, kSite: 2, evo: 0, hbond: 0 },
  };

  // ---------- random numbers ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u, v, s;
      do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      const m = Math.sqrt(-2 * Math.log(s) / s);
      spare = v * m;
      return u * m;
    };
  }

  // ---------- secondary structure from sequence ----------
  // Sliding-window Chou–Fasman: average propensities over ±3 residues, softmax into P(helix/strand/coil).
  function predictSS(seq) {
    const n = seq.length, probs = [];
    for (let i = 0; i < n; i++) {
      let a = 0, b = 0, t = 0, c = 0;
      for (let k = Math.max(0, i - 3); k <= Math.min(n - 1, i + 3); k++) {
        const p = CHOU_FASMAN[seq[k]] || [100, 100, 100];
        a += p[0]; b += p[1]; t += p[2]; c++;
      }
      a /= 100 * c; b /= 100 * c; t /= 100 * c;
      const s = [12 * (a - 1.03), 12 * (b - 1.05), 12 * (t - 1.0) + 0.6];
      const m = Math.max(s[0], s[1], s[2]);
      const e = s.map(v => Math.exp(v - m)), z = e[0] + e[1] + e[2];
      probs.push({ H: e[0] / z, E: e[1] / z, C: e[2] / z });
    }
    let labels = probs.map(p => (p.H >= p.E && p.H >= p.C ? 'H' : p.E >= p.C ? 'E' : 'C')).join('');
    labels = dropShortRuns(labels, 'H', 4);
    labels = dropShortRuns(labels, 'E', 3);
    return { probs, labels };
  }
  function dropShortRuns(s, ch, minLen) {
    return s.replace(new RegExp(ch + '+', 'g'), run => (run.length >= minLen ? run : 'C'.repeat(run.length)));
  }
  function weightsFromLabels(labels) {
    return labels.split('').map(c => (c === 'H' ? { H: 0.9, E: 0.03, C: 0.07 }
      : c === 'G' ? { H: 0.5, E: 0.03, C: 0.47 }
        : c === 'E' ? { H: 0.03, E: 0.9, C: 0.07 } : { H: 0.05, E: 0.05, C: 0.9 }));
  }

  // ---------- knowledge base ----------
  let KBD = null;
  function kbData() {
    if (KBD) return KBD;
    const kb = root.FOLD_KB;
    if (!kb) return null;
    const dec = str => { const b = atob(str), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
    const fs = kb.frag.seq, fss = kb.frag.ss, n = fs.length;
    const aa = new Uint8Array(n), ss = new Uint8Array(n), idx = {};
    for (let i = 0; i < 20; i++) idx[kb.aa[i]] = i;
    for (let i = 0; i < n; i++) {
      aa[i] = fs[i] === '|' ? 255 : idx[fs[i]];
      const c = fss[i];
      ss[i] = c === 'H' || c === 'G' ? 0 : c === 'E' ? 1 : c === '|' ? 255 : 2;
    }
    const E = new Float64Array(400);
    for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++) E[20 * i + j] = kb.contact[i][j];
    KBD = { n, aa, ss, theta: dec(kb.frag.theta), phi: dec(kb.frag.phi), E, idx, blosum: Int8Array.from(kb.blosum),
      scLen: kb.scLen, backbone: kb.backbone, rotamers: kb.rotamers, chains: kb.chains, residues: kb.residues };
    return KBD;
  }

  // Grids for the tabulated local potentials: θ from 60° in 2.5° steps, φ over the full circle in 10° steps.
  const TH0 = 60, THS = 2.5, THN = 45, PHN = 36;

  // Fragment picking (Rosetta-style, single sequence): for every 9-, 5- and 3-residue window of the target, take the
  // library windows with the best BLOSUM62 score (plus a bonus per matching secondary-structure letter when a
  // PDB record is supplied). Their Cα angles become position-specific potentials; their DSSP-style labels vote
  // for the secondary structure.
  function buildLocalModel(seq, opts) {
    opts = opts || {};
    const D = kbData(), n = seq.length, B = D.blosum;
    const tgt = new Uint8Array(n);
    for (let i = 0; i < n; i++) tgt[i] = D.idx[seq[i]] != null ? D.idx[seq[i]] : 0;
    const hint = opts.ss ? opts.ss.split('').map(c => (c === 'H' || c === 'G' ? 0 : c === 'E' ? 1 : 2)) : null;
    const bonus = opts.ssBonus != null ? opts.ssBonus : 2;
    const sizes = [[9, 25, 1.0], [5, 40, 0.6], [3, 60, 0.35]];
    const thS = [], phS = [], votes = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) { thS.push([]); phS.push([]); }
    // frags[L][p]: library start indices of the fragments picked for window p (kept for fragment insertion)
    const frags = { 9: [], 3: [] };
    let picked = 0;
    for (const [L, K, wgt] of sizes) {
      if (L > n) continue;
      for (let p = 0; p + L <= n; p++) {
        const bestS = new Float64Array(K).fill(-1e9), bestI = new Int32Array(K).fill(-1);
        for (let st = 0; st + L <= D.n; st++) {
          let sc = 0, bad = -1;
          for (let k = 0; k < L; k++) {
            const a = D.aa[st + k];
            if (a === 255) { bad = k; break; }
            sc += B[tgt[p + k] * 20 + a];
            if (hint && D.ss[st + k] === hint[p + k]) sc += bonus;
          }
          if (bad >= 0) { st += bad; continue; }
          if (sc <= bestS[K - 1]) continue;
          let q = K - 1;
          while (q > 0 && bestS[q - 1] < sc) { bestS[q] = bestS[q - 1]; bestI[q] = bestI[q - 1]; q--; }
          bestS[q] = sc; bestI[q] = st;
        }
        if (frags[L]) frags[L][p] = Array.from(bestI).filter(s => s >= 0);
        for (let f = 0; f < K; f++) {
          const st = bestI[f];
          if (st < 0) continue;
          picked++;
          for (let k = 0; k < L; k++) {
            const i = p + k, r = st + k;
            if (i >= 1 && i <= n - 2 && D.theta[r] !== 255) thS[i].push(D.theta[r], wgt);
            if (i >= 1 && i <= n - 3 && D.phi[r] !== 255) phS[i].push(D.phi[r] / 254 * 360 - 180, wgt);
            votes[3 * i + D.ss[r]] += wgt;
          }
        }
      }
    }
    // smoothed densities → −ln p on the grids (0 at the most likely angle)
    const theta = new Float32Array(n * THN), phi = new Float32Array(n * PHN);
    const sig = 6, kap = 6, floor = 0.03;
    for (let i = 0; i < n; i++) {
      const S = thS[i];
      if (S.length) {
        const row = new Float64Array(THN);
        for (let g = 0; g < THN; g++) {
          const t = TH0 + g * THS;
          let d = 0;
          for (let m = 0; m < S.length; m += 2) { const u = (t - S[m]) / sig; d += S[m + 1] * Math.exp(-0.5 * u * u); }
          row[g] = d;
        }
        const mx = Math.max(...row) || 1;
        for (let g = 0; g < THN; g++) theta[i * THN + g] = -Math.log(row[g] / mx + floor) + Math.log(1 + floor);
      }
      const P = phS[i];
      if (P.length) {
        const row = new Float64Array(PHN);
        for (let g = 0; g < PHN; g++) {
          const a = (-180 + g * 10) * DEG;
          let d = 0;
          for (let m = 0; m < P.length; m += 2) d += P[m + 1] * Math.exp(kap * (Math.cos(a - P[m] * DEG) - 1));
          row[g] = d;
        }
        const mx = Math.max(...row) || 1;
        for (let g = 0; g < PHN; g++) phi[i * PHN + g] = -Math.log(row[g] / mx + floor) + Math.log(1 + floor);
      }
    }
    const probs = [];
    for (let i = 0; i < n; i++) {
      const h = votes[3 * i] + 0.2, e = votes[3 * i + 1] + 0.2, c = votes[3 * i + 2] + 0.3, z = h + e + c;
      probs.push({ H: h / z, E: e / z, C: c / z });
    }
    let labels = probs.map(q => (q.H >= q.E && q.H >= q.C ? 'H' : q.E >= q.C ? 'E' : 'C')).join('');
    labels = dropShortRuns(dropShortRuns(labels, 'H', 4), 'E', 3);
    return { local: { theta, phi }, probs, labels, fragments: picked, frags };
  }

  let tabV = 0, tabD = 0;
  // Catmull-Rom interpolation of a tabulated row; x in grid units. Sets tabV and tabD (per grid unit).
  function tableEval(row, off, N, x, periodic) {
    let k = Math.floor(x), t = x - k, i0, i1, i2, i3;
    if (periodic) {
      k = ((k % N) + N) % N;
      i0 = (k + N - 1) % N; i1 = k; i2 = (k + 1) % N; i3 = (k + 2) % N;
    } else {
      if (x <= 0) { tabV = row[off]; tabD = 0; return; }
      if (x >= N - 1) { tabV = row[off + N - 1]; tabD = 0; return; }
      i0 = Math.max(0, k - 1); i1 = k; i2 = k + 1; i3 = Math.min(N - 1, k + 2);
    }
    const p0 = row[off + i0], p1 = row[off + i1], p2 = row[off + i2], p3 = row[off + i3];
    const a = -p0 + 3 * p1 - 3 * p2 + p3, b = 2 * p0 - 5 * p1 + 4 * p2 - p3, c = -p0 + p2;
    tabV = 0.5 * (2 * p1 + c * t + b * t * t + a * t * t * t);
    tabD = 0.5 * (c + 2 * b * t + 3 * a * t * t);
  }

  // ---------- geometry ----------
  function placeAtom(x, a, b, c, d, len, theta, phi) {
    // NeRF: put atom d bonded to c, with angle (b, c, d) = theta and dihedral (a, b, c, d) = phi.
    const bcx = x[3 * c] - x[3 * b], bcy = x[3 * c + 1] - x[3 * b + 1], bcz = x[3 * c + 2] - x[3 * b + 2];
    const lbc = Math.hypot(bcx, bcy, bcz);
    const ux = bcx / lbc, uy = bcy / lbc, uz = bcz / lbc;
    const abx = x[3 * b] - x[3 * a], aby = x[3 * b + 1] - x[3 * a + 1], abz = x[3 * b + 2] - x[3 * a + 2];
    let nx = aby * uz - abz * uy, ny = abz * ux - abx * uz, nz = abx * uy - aby * ux;
    const ln = Math.hypot(nx, ny, nz);
    nx /= ln; ny /= ln; nz /= ln;
    const mx = ny * uz - nz * uy, my = nz * ux - nx * uz, mz = nx * uy - ny * ux;
    const d0 = -len * Math.cos(theta), d1 = len * Math.sin(theta) * Math.cos(phi), d2 = len * Math.sin(theta) * Math.sin(phi);
    x[3 * d] = x[3 * c] + d0 * ux + d1 * mx + d2 * nx;
    x[3 * d + 1] = x[3 * c + 1] + d0 * uy + d1 * my + d2 * ny;
    x[3 * d + 2] = x[3 * c + 2] + d0 * uz + d1 * mz + d2 * nz;
  }

  // Build a chain from virtual-bond angles theta[1..n-2] and dihedrals phi[1..n-3].
  function buildChain(n, theta, phi) {
    const x = new Float64Array(3 * n);
    if (n > 1) x[3] = BOND;
    if (n > 2) { x[6] = BOND - BOND * Math.cos(theta[1]); x[7] = BOND * Math.sin(theta[1]); }
    for (let k = 3; k < n; k++) placeAtom(x, k - 3, k - 2, k - 1, k, BOND, theta[k - 1], phi[k - 2]);
    return x;
  }

  // Self-avoiding random coil, grown bead by bead.
  function randomCoil(n, rand) {
    const theta = new Float64Array(n), phi = new Float64Array(n);
    for (let attempt = 0; attempt < 50; attempt++) {
      const x = new Float64Array(3 * n);
      if (n > 1) x[3] = BOND;
      theta[1] = (105 + 20 * rand()) * DEG;
      if (n > 2) { x[6] = BOND - BOND * Math.cos(theta[1]); x[7] = BOND * Math.sin(theta[1]); }
      let ok = true;
      for (let k = 3; k < n && ok; k++) {
        let placed = false;
        for (let tries = 0; tries < 60 && !placed; tries++) {
          const th = (100 + 30 * rand()) * DEG, ph = (rand() * 360 - 180) * DEG;
          placeAtom(x, k - 3, k - 2, k - 1, k, BOND, th, ph);
          placed = true;
          for (let j = 0; j < k - 2; j++) {
            const dx = x[3 * k] - x[3 * j], dy = x[3 * k + 1] - x[3 * j + 1], dz = x[3 * k + 2] - x[3 * j + 2];
            if (dx * dx + dy * dy + dz * dz < 4.5 * 4.5) { placed = false; break; }
          }
        }
        ok = placed;
      }
      if (ok) return center(x);
    }
    const th = new Float64Array(n).fill(120 * DEG), ph = new Float64Array(n).fill(180 * DEG);
    return center(buildChain(n, th, ph));
  }

  function center(x) {
    const n = x.length / 3;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) { cx += x[3 * i]; cy += x[3 * i + 1]; cz += x[3 * i + 2]; }
    cx /= n; cy /= n; cz /= n;
    for (let i = 0; i < n; i++) { x[3 * i] -= cx; x[3 * i + 1] -= cy; x[3 * i + 2] -= cz; }
    return x;
  }

  function dist(x, i, j) {
    return Math.hypot(x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]);
  }
  function angleAt(x, i, j, k) {
    const ax = x[3 * i] - x[3 * j], ay = x[3 * i + 1] - x[3 * j + 1], az = x[3 * i + 2] - x[3 * j + 2];
    const bx = x[3 * k] - x[3 * j], by = x[3 * k + 1] - x[3 * j + 1], bz = x[3 * k + 2] - x[3 * j + 2];
    const c = (ax * bx + ay * by + az * bz) / (Math.hypot(ax, ay, az) * Math.hypot(bx, by, bz));
    return Math.acos(Math.max(-1, Math.min(1, c)));
  }
  function dihedral(x, i, j, k, l) {
    const rijx = x[3 * i] - x[3 * j], rijy = x[3 * i + 1] - x[3 * j + 1], rijz = x[3 * i + 2] - x[3 * j + 2];
    const rkjx = x[3 * k] - x[3 * j], rkjy = x[3 * k + 1] - x[3 * j + 1], rkjz = x[3 * k + 2] - x[3 * j + 2];
    const rklx = x[3 * k] - x[3 * l], rkly = x[3 * k + 1] - x[3 * l + 1], rklz = x[3 * k + 2] - x[3 * l + 2];
    const mx = rijy * rkjz - rijz * rkjy, my = rijz * rkjx - rijx * rkjz, mz = rijx * rkjy - rijy * rkjx;
    const nx = rkjy * rklz - rkjz * rkly, ny = rkjz * rklx - rkjx * rklz, nz = rkjx * rkly - rkjy * rklx;
    const rkj = Math.hypot(rkjx, rkjy, rkjz);
    return Math.atan2(rkj * (rijx * nx + rijy * ny + rijz * nz), mx * nx + my * ny + mz * nz);
  }
  function radiusOfGyration(x) {
    const n = x.length / 3;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) { cx += x[3 * i]; cy += x[3 * i + 1]; cz += x[3 * i + 2]; }
    cx /= n; cy /= n; cz /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (x[3 * i] - cx) ** 2 + (x[3 * i + 1] - cy) ** 2 + (x[3 * i + 2] - cz) ** 2;
    return Math.sqrt(s / n);
  }
  function wrap(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }

  // ---------- force field ----------
  function createSystem(cfg) {
    const seq = cfg.seq, n = seq.length;
    const model = cfg.model;
    const p = Object.assign({}, DEFAULTS[model], cfg.params || {});
    const sys = { n, seq, model, params: p, native: cfg.native || null };
    if (cfg.restraints) sys.rest = cfg.restraints;

    if (model === 'go') {
      const X = cfg.native;
      sys.r0 = new Float64Array(n);
      for (let i = 0; i < n - 1; i++) sys.r0[i] = dist(X, i, i + 1);
      sys.theta0 = new Float64Array(n);
      for (let j = 1; j < n - 1; j++) sys.theta0[j] = angleAt(X, j - 1, j, j + 1);
      sys.phi0 = new Float64Array(n);
      for (let k = 1; k < n - 2; k++) sys.phi0[k] = dihedral(X, k - 1, k, k + 1, k + 2);
      const nat = [], non = [];
      for (let i = 0; i < n; i++) for (let j = i + 3; j < n; j++) {
        const d = dist(X, i, j);
        if (j - i >= p.minSep && d < p.contactCut) nat.push(i, j, d); else non.push(i, j);
      }
      sys.natI = Int32Array.from(nat.filter((_, k) => k % 3 === 0));
      sys.natJ = Int32Array.from(nat.filter((_, k) => k % 3 === 1));
      sys.natS = Float64Array.from(nat.filter((_, k) => k % 3 === 2));
      sys.nonI = Int32Array.from(non.filter((_, k) => k % 2 === 0));
      sys.nonJ = Int32Array.from(non.filter((_, k) => k % 2 === 1));
    } else {
      const w = cfg.ssWeights;
      sys.ssWeights = w;
      // angle at vertex j uses residue j; dihedral k (k-1..k+2) is helical/extended only if residues k and k+1 both are
      sys.aW = [];
      for (let j = 0; j < n; j++) sys.aW.push(w[j]);
      sys.dW = [];
      for (let k = 0; k < n - 1; k++) {
        const a = w[k], b = w[k + 1];
        sys.dW.push({ H: Math.min(a.H, b.H), E: Math.min(a.E, b.E), C: Math.max(a.C, b.C) });
      }
      const h = seq.split('').map(c => Math.max(0, HYDRO[c] || 0) / 1.8);
      const q = seq.split('').map(c => CHARGE[c] || 0);
      let pairE = (i, j) => -1.0 * p.hydro * h[i] * h[j] + 0.5 * p.charge * q[i] * q[j];
      if (model === 'kb') {
        const D = kbData();
        sys.local = cfg.local;
        const ai = seq.split('').map(c => (D.idx[c] != null ? D.idx[c] : 0));
        const PR = root.FOLD_KB.pairRef;
        pairE = (i, j) => p.contact * PR[ai[i] * 20 + ai[j]];
        sys.aIdx = Int32Array.from(ai);
        sys.burT = Float64Array.from(root.FOLD_KB.burial.table);
        sys.burN = root.FOLD_KB.burial.nMax;
        sys.burCut = root.FOLD_KB.burial.cut;
        sys.nb = new Float64Array(n); sys.dEdn = new Float64Array(n);
      }
      const I = [], J = [], EIJ = [], BIJ = [];
      for (let i = 0; i < n; i++) for (let j = i + 3; j < n; j++) {
        const e = pairE(i, j);
        const b = j - i >= 3 ? -1.0 * p.beta * Math.sqrt(w[i].E * w[j].E) : 0;
        I.push(i); J.push(j); EIJ.push(e); BIJ.push(b);
      }
      sys.pI = Int32Array.from(I); sys.pJ = Int32Array.from(J);
      sys.pE = Float64Array.from(EIJ); sys.pB = Float64Array.from(BIJ);
      if (model === 'kb') {
        sys.pq = new Float64Array(I.length); sys.pw = new Float64Array(I.length);
        sys.nvec = new Float64Array(3 * n); sys.mlen = new Float64Array(n); sys.gm = new Float64Array(3 * n);
        for (let t = 0; t < I.length; t++) sys.pB[t] *= 2;   // directional pairing: two partners per strand residue
        // backbone hydrogen bonds, whatever the predicted secondary structure: the same directional well (4.9 Å, both
        // Cα-zigzag normals along the pair vector) rewards any pair in H-bond geometry — hairpin partners and helix turns alike
        if (p.hbond) for (let t = 0; t < I.length; t++) sys.pB[t] -= p.hbond;
      }
      sys.rg0 = 2.2 * Math.pow(n, 0.38);
      const D = model === 'kb' ? kbData() : null;
      sys.scLen = Float64Array.from(seq.split('').map(c => (D ? D.scLen[c] : SIDECHAIN[c]) || 2));
      if (D && root.FOLD_KB.site) {
        sys.siteC = new Float64Array(3 * n);
        seq.split('').forEach((c, i) => { const k = root.FOLD_KB.site[c] || [0, 0, 0]; sys.siteC[3 * i] = k[0]; sys.siteC[3 * i + 1] = k[1]; sys.siteC[3 * i + 2] = k[2]; });
        sys.hasSite = Uint8Array.from(seq.split('').map(c => (c === 'G' ? 0 : 1)));
      }
      sys.site = new Float64Array(3 * n); sys.siteU = new Float64Array(3 * n);
      sys.siteA = new Float64Array(n); sys.siteF = new Float64Array(3 * n);
    }
    return sys;
  }

  const comp = { bond: 0, angle: 0, dihedral: 0, contact: 0, burial: 0, repulsion: 0, compact: 0, restraint: 0, total: 0 };

  // Knowledge-based nonbonded terms. Cα: excluded volume and β-pairing. Side-chain sites: excluded volume,
  // contact-reference pair preferences, and a burial term on each residue's neighbour count. Two passes,
  // because a burial force needs every residue's count first. Site forces accumulate in sys.siteF.
  // Verlet neighbour lists. A pair is listed when it was within its cutoff + NL_SKIN at the last rebuild; the list is
  // rebuilt once any bead (or side-chain site) has moved more than NL_SKIN / 2 since then, so every pair inside the
  // cutoff is always listed. Listed pairs keep their original order: forces and energies are bit-for-bit those of the
  // full O(n²) loop, which only ever skipped the pairs left out here.
  const NL_SKIN = 2.0;
  function moved(ref, y, n, h2) {
    for (let k = 0; k < 3 * n; k += 3) {
      const dx = y[k] - ref[k], dy = y[k + 1] - ref[k + 1], dz = y[k + 2] - ref[k + 2];
      if (dx * dx + dy * dy + dz * dz > h2) return true;
    }
    return false;
  }
  function pairList(sys, x, S, caCut, siteCut) {
    const n = sys.n, I = sys.pI, J = sys.pJ;
    let L = sys.nl;
    if (!L) L = sys.nl = { list: new Int32Array(I.length), len: 0, xr: new Float64Array(3 * n), sr: new Float64Array(3 * n), valid: false, rebuilds: 0 };
    const sk = sys.nlSkin || NL_SKIN, h2 = (sk / 2) ** 2;
    if (L.valid && !moved(L.xr, x, n, h2) && !moved(L.sr, S, n, h2)) return L;
    const c1 = (caCut + sk) ** 2, c2 = (siteCut + sk) ** 2;
    let len = 0;
    for (let t = 0; t < I.length; t++) {
      const i = I[t], j = J[t];
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const sx = S[3 * j] - S[3 * i], sy = S[3 * j + 1] - S[3 * i + 1], sz = S[3 * j + 2] - S[3 * i + 2];
      if (dx * dx + dy * dy + dz * dz < c1 || sx * sx + sy * sy + sz * sz < c2) L.list[len++] = t;
    }
    L.len = len; L.xr.set(x.subarray ? x.subarray(0, 3 * n) : x.slice(0, 3 * n)); L.sr.set(S.subarray(0, 3 * n)); L.valid = true; L.rebuilds++;
    return L;
  }
  // per restraint and 1 Å segment: p1 and the Catmull-Rom coefficients c, b, a of tableEval (same arithmetic, same values)
  function restCoef(R) {
    if (R.coef) return R.coef;
    const GN = R.gridN, V = R.V, m = R.I.length, C = new Float64Array(m * (GN - 1) * 4);
    for (let t = 0; t < m; t++) for (let k = 0; k < GN - 1; k++) {
      const off = t * GN, p0 = V[off + Math.max(0, k - 1)], p1 = V[off + k], p2 = V[off + k + 1], p3 = V[off + Math.min(GN - 1, k + 2)];
      const o = (t * (GN - 1) + k) * 4;
      C[o] = p1; C[o + 1] = -p0 + p2; C[o + 2] = 2 * p0 - 5 * p1 + 4 * p2 - p3; C[o + 3] = -p0 + 3 * p1 - 3 * p2 + p3;
    }
    return (R.coef = C);
  }
  function restList(sys, x, R, top) {
    const n = sys.n;
    let L = sys.rl;
    if (!L || L.src !== R) L = sys.rl = { list: new Int32Array(R.I.length), len: 0, xr: new Float64Array(3 * n), valid: false, src: R };
    const sk = sys.nlSkin || NL_SKIN;
    if (L.valid && !moved(L.xr, x, n, (sk / 2) ** 2)) return L;
    const c = (top + sk) ** 2, RI = R.I, RJ = R.J;
    let len = 0;
    for (let t = 0; t < RI.length; t++) {
      const i = RI[t], j = RJ[t];
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      if (dx * dx + dy * dy + dz * dz < c) L.list[len++] = t;
    }
    L.len = len; L.xr.set(x.subarray ? x.subarray(0, 3 * n) : x.slice(0, 3 * n)); L.valid = true;
    return L;
  }
  function kbPairs(sys, x, f) {
    const p = sys.params, n = sys.n, S = sys.site, FS = sys.siteF, HS = sys.hasSite;
    const I = sys.pI, J = sys.pJ, PE = sys.pE, PB = sys.pB, PQ = sys.pq, PW = sys.pw, NB = sys.nb, DN = sys.dEdn;
    const sr2 = p.repel * p.repel, r1 = p.wellIn, r2c = p.wellOut, span = r2c - r1, b0 = sys.burCut[0], b1 = sys.burCut[1], bs = b1 - b0;
    let Er = 0, Ec = 0, Eu = 0;
    NB.fill(0);
    // local plane normals: in a β-strand the normal of the Cα zigzag points across the sheet, toward the partner strand
    const NV = sys.nvec, ML = sys.mlen, GM = sys.gm;
    GM.fill(0);
    for (let i = 1; i < n - 1; i++) {
      const a0 = x[3 * i] - x[3 * i - 3], a1 = x[3 * i + 1] - x[3 * i - 2], a2 = x[3 * i + 2] - x[3 * i - 1];
      const c0 = x[3 * i + 3] - x[3 * i], c1 = x[3 * i + 4] - x[3 * i + 1], c2 = x[3 * i + 5] - x[3 * i + 2];
      const m0 = a1 * c2 - a2 * c1, m1 = a2 * c0 - a0 * c2, m2 = a0 * c1 - a1 * c0, l = Math.sqrt(m0 * m0 + m1 * m1 + m2 * m2);
      ML[i] = l > 1e-6 ? l : 0;
      if (ML[i]) { NV[3 * i] = m0 / l; NV[3 * i + 1] = m1 / l; NV[3 * i + 2] = m2 / l; }
    }
    const NLs = pairList(sys, x, S, 6, Math.max(b1, r2c, p.siteRepel)), LS = NLs.list, LN = NLs.len;
    for (let k = 0; k < LN; k++) {
      const t = LS[k], i = I[t], j = J[t];
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < 36) {
        const r = Math.sqrt(r2), s2 = sr2 / r2, s6 = s2 * s2 * s2, s12 = s6 * s6;
        Er += s12;
        let gx = -12 * s12 / r2 * dx, gy = -12 * s12 / r2 * dy, gz = -12 * s12 / r2 * dz;   // dE/d(xj − xi)
        // directional β pairing: well at 4.9 Å, both plane normals along the inter-strand vector
        const bb = PB[t], u = r - 4.9;
        if (bb !== 0 && u > -1 && u < 1 && ML[i] && ML[j]) {
          const rx = dx / r, ry = dy / r, rz = dz / r;
          const nix = NV[3 * i], niy = NV[3 * i + 1], niz = NV[3 * i + 2], njx = NV[3 * j], njy = NV[3 * j + 1], njz = NV[3 * j + 2];
          const ci = nix * rx + niy * ry + niz * rz, cj = njx * rx + njy * ry + njz * rz, Ai = ci * ci, Aj = cj * cj;
          const w = 1 - u * u, fw = w * w, dfw = -4 * u * w;
          Ec += bb * fw * Ai * Aj;
          const k1 = bb * dfw * Ai * Aj, k2 = bb * fw * 2 / r;
          gx += k1 * rx + k2 * (Aj * ci * (nix - ci * rx) + Ai * cj * (njx - cj * rx));
          gy += k1 * ry + k2 * (Aj * ci * (niy - ci * ry) + Ai * cj * (njy - cj * ry));
          gz += k1 * rz + k2 * (Aj * ci * (niz - ci * rz) + Ai * cj * (njz - cj * rz));
          const qi = bb * fw * Aj * 2 * ci / ML[i], qj = bb * fw * Ai * 2 * cj / ML[j];
          GM[3 * i] += qi * (rx - ci * nix); GM[3 * i + 1] += qi * (ry - ci * niy); GM[3 * i + 2] += qi * (rz - ci * niz);
          GM[3 * j] += qj * (rx - cj * njx); GM[3 * j + 1] += qj * (ry - cj * njy); GM[3 * j + 2] += qj * (rz - cj * njz);
        }
        f[3 * j] -= gx; f[3 * j + 1] -= gy; f[3 * j + 2] -= gz;
        f[3 * i] += gx; f[3 * i + 1] += gy; f[3 * i + 2] += gz;
      }
      const sx = S[3 * j] - S[3 * i], sy = S[3 * j + 1] - S[3 * i + 1], sz = S[3 * j + 2] - S[3 * i + 2];
      const q2 = sx * sx + sy * sy + sz * sz;
      if (q2 >= b1 * b1) { PQ[t] = -1; continue; }
      const q = Math.sqrt(q2);
      PQ[t] = q;
      if (q <= b0) { NB[i] += 1; NB[j] += 1; PW[t] = 0; }
      else { const u = (q - b0) / bs, w = 1 - u * u * (3 - 2 * u); NB[i] += w; NB[j] += w; PW[t] = -6 * u * (1 - u) / bs; }
    }
    const T = sys.burT, NM = sys.burN, A = sys.aIdx, wb = p.burial;
    for (let i = 0; i < n; i++) {
      const v = NB[i], row = A[i] * (NM + 1);
      if (v >= NM) { Eu += wb * T[row + NM]; DN[i] = 0; continue; }
      const k = Math.floor(v), fr = v - k;
      Eu += wb * (T[row + k] * (1 - fr) + T[row + k + 1] * fr);
      DN[i] = wb * (T[row + k + 1] - T[row + k]);
    }
    for (let k = 0; k < LN; k++) {
      const t = LS[k], q = PQ[t];
      if (q < 0) continue;
      const i = I[t], j = J[t];
      let dEdq = (DN[i] + DN[j]) * PW[t];
      if (HS[i] && HS[j] && q < p.siteRepel) { const d = p.siteRepel - q; Er += p.kSite * d * d; dEdq -= 2 * p.kSite * d; }
      const e = PE[t];
      if (e !== 0 && q < r2c) {
        if (q <= r1) Ec += e;
        else { const u = (q - r1) / span; Ec += e * (1 - u * u * (3 - 2 * u)); dEdq -= 6 * e * u * (1 - u) / span; }
      }
      if (dEdq === 0) continue;
      const g = -dEdq / q;
      const sx = S[3 * j] - S[3 * i], sy = S[3 * j + 1] - S[3 * i + 1], sz = S[3 * j + 2] - S[3 * i + 2];
      FS[3 * j] += g * sx; FS[3 * j + 1] += g * sy; FS[3 * j + 2] += g * sz;
      FS[3 * i] -= g * sx; FS[3 * i + 1] -= g * sy; FS[3 * i + 2] -= g * sz;
    }
    // plane-normal gradients back to the three Cα that define each normal (m = a × c)
    for (let i = 1; i < n - 1; i++) {
      const g0 = GM[3 * i], g1 = GM[3 * i + 1], g2 = GM[3 * i + 2];
      if (g0 === 0 && g1 === 0 && g2 === 0) continue;
      const a0 = x[3 * i] - x[3 * i - 3], a1 = x[3 * i + 1] - x[3 * i - 2], a2 = x[3 * i + 2] - x[3 * i - 1];
      const c0 = x[3 * i + 3] - x[3 * i], c1 = x[3 * i + 4] - x[3 * i + 1], c2 = x[3 * i + 5] - x[3 * i + 2];
      const ga0 = c1 * g2 - c2 * g1, ga1 = c2 * g0 - c0 * g2, ga2 = c0 * g1 - c1 * g0;   // c × g
      const gc0 = g1 * a2 - g2 * a1, gc1 = g2 * a0 - g0 * a2, gc2 = g0 * a1 - g1 * a0;   // g × a
      f[3 * i] -= ga0 - gc0; f[3 * i + 1] -= ga1 - gc1; f[3 * i + 2] -= ga2 - gc2;
      f[3 * i - 3] += ga0; f[3 * i - 2] += ga1; f[3 * i - 1] += ga2;
      f[3 * i + 3] -= gc0; f[3 * i + 4] -= gc1; f[3 * i + 5] -= gc2;
    }
    return { rep: Er, contact: Ec, burial: Eu };
  }

  // Fills f with forces for configuration x; returns potential energy, components in `comp`.
  function computeForces(sys, x, f) {
    const n = sys.n, p = sys.params;
    f.fill(0);
    let Eb = 0, Ea = 0, Ed = 0, Ec = 0, Er = 0, Eg = 0, Eu = 0;

    // bonds
    for (let i = 0; i < n - 1; i++) {
      const j = i + 1;
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const r0 = sys.model === 'go' ? sys.r0[i] : BOND;
      const dr = r - r0;
      Eb += p.kBond * dr * dr;
      const g = -2 * p.kBond * dr / r;
      f[3 * j] += g * dx; f[3 * j + 1] += g * dy; f[3 * j + 2] += g * dz;
      f[3 * i] -= g * dx; f[3 * i + 1] -= g * dy; f[3 * i + 2] -= g * dz;
    }

    // bond angles
    for (let j = 1; j < n - 1; j++) {
      const i = j - 1, k = j + 1;
      const ax = x[3 * i] - x[3 * j], ay = x[3 * i + 1] - x[3 * j + 1], az = x[3 * i + 2] - x[3 * j + 2];
      const bx = x[3 * k] - x[3 * j], by = x[3 * k + 1] - x[3 * j + 1], bz = x[3 * k + 2] - x[3 * j + 2];
      const la2 = ax * ax + ay * ay + az * az, lb2 = bx * bx + by * by + bz * bz;
      const la = Math.sqrt(la2), lb = Math.sqrt(lb2);
      let c = (ax * bx + ay * by + az * bz) / (la * lb);
      c = Math.max(-0.999999, Math.min(0.999999, c));
      const th = Math.acos(c), s = Math.sqrt(1 - c * c);
      let V, dV;
      if (sys.model === 'go') {
        const d = th - sys.theta0[j];
        V = p.kAngle * d * d; dV = 2 * p.kAngle * d;
      } else if (sys.model === 'kb') {
        tableEval(sys.local.theta, j * THN, THN, (th / DEG - TH0) / THS, false);
        V = p.local * tabV; dV = p.local * tabD / THS / DEG;
        const tmin = 82 * DEG;
        if (th < tmin) { V += 50 * (tmin - th) ** 2; dV -= 100 * (tmin - th); }
      } else {
        const w = sys.aW[j], A = 2.0 * p.local;
        const gH = w.H * Math.exp(-0.5 * ((th - LOCAL.H.theta[0]) / LOCAL.H.theta[1]) ** 2);
        const gE = w.E * Math.exp(-0.5 * ((th - LOCAL.E.theta[0]) / LOCAL.E.theta[1]) ** 2);
        const gC = w.C * Math.exp(-0.5 * ((th - LOCAL.C.theta[0]) / LOCAL.C.theta[1]) ** 2);
        const z = gH + gE + gC + 1e-9;
        V = -A * Math.log(z);
        dV = A * (gH * (th - LOCAL.H.theta[0]) / LOCAL.H.theta[1] ** 2 + gE * (th - LOCAL.E.theta[0]) / LOCAL.E.theta[1] ** 2
          + gC * (th - LOCAL.C.theta[0]) / LOCAL.C.theta[1] ** 2) / z;
        // steric floor: Cα(i-1)–Cα(i+1) cannot come closer than ~5 Å
        const tmin = 82 * DEG;
        if (th < tmin) { V += 50 * (tmin - th) ** 2; dV -= 100 * (tmin - th); }
      }
      Ea += V;
      const g = dV / s;
      const fix = g * (bx / (la * lb) - c * ax / la2), fiy = g * (by / (la * lb) - c * ay / la2), fiz = g * (bz / (la * lb) - c * az / la2);
      const fkx = g * (ax / (la * lb) - c * bx / lb2), fky = g * (ay / (la * lb) - c * by / lb2), fkz = g * (az / (la * lb) - c * bz / lb2);
      f[3 * i] += fix; f[3 * i + 1] += fiy; f[3 * i + 2] += fiz;
      f[3 * k] += fkx; f[3 * k + 1] += fky; f[3 * k + 2] += fkz;
      f[3 * j] -= fix + fkx; f[3 * j + 1] -= fiy + fky; f[3 * j + 2] -= fiz + fkz;
    }

    // dihedrals
    for (let k = 1; k < n - 2; k++) {
      const i = k - 1, j = k, kk = k + 1, l = k + 2;
      const rijx = x[3 * i] - x[3 * j], rijy = x[3 * i + 1] - x[3 * j + 1], rijz = x[3 * i + 2] - x[3 * j + 2];
      const rkjx = x[3 * kk] - x[3 * j], rkjy = x[3 * kk + 1] - x[3 * j + 1], rkjz = x[3 * kk + 2] - x[3 * j + 2];
      const rklx = x[3 * kk] - x[3 * l], rkly = x[3 * kk + 1] - x[3 * l + 1], rklz = x[3 * kk + 2] - x[3 * l + 2];
      const mx = rijy * rkjz - rijz * rkjy, my = rijz * rkjx - rijx * rkjz, mz = rijx * rkjy - rijy * rkjx;
      const nx = rkjy * rklz - rkjz * rkly, ny = rkjz * rklx - rkjx * rklz, nz = rkjx * rkly - rkjy * rklx;
      const m2 = mx * mx + my * my + mz * mz, n2 = nx * nx + ny * ny + nz * nz;
      if (m2 < 1e-8 || n2 < 1e-8) continue;
      const rkj2 = rkjx * rkjx + rkjy * rkjy + rkjz * rkjz, rkj = Math.sqrt(rkj2);
      const phi = Math.atan2(rkj * (rijx * nx + rijy * ny + rijz * nz), mx * nx + my * ny + mz * nz);
      let V, dV;
      if (sys.model === 'go') {
        const d = phi - sys.phi0[k];
        V = p.kDih1 * (1 - Math.cos(d)) + p.kDih3 * (1 - Math.cos(3 * d));
        dV = p.kDih1 * Math.sin(d) + 3 * p.kDih3 * Math.sin(3 * d);
      } else if (sys.model === 'kb') {
        tableEval(sys.local.phi, k * PHN, PHN, (phi / DEG + 180) / 10, true);
        V = p.local * tabV; dV = p.local * tabD / 10 / DEG;
      } else {
        const w = sys.dW[k], A = 2.0 * p.local;
        const gH = w.H * Math.exp(LOCAL.H.phi[1] * (Math.cos(phi - LOCAL.H.phi[0]) - 1));
        const gE = w.E * Math.exp(LOCAL.E.phi[1] * (Math.cos(phi - LOCAL.E.phi[0]) - 1));
        const gC = w.C * Math.exp(LOCAL.C.phi[1] * (Math.cos(phi - LOCAL.C.phi[0]) - 1));
        const z = gH + gE + gC + 1e-9;
        V = -A * Math.log(z);
        dV = A * (gH * LOCAL.H.phi[1] * Math.sin(phi - LOCAL.H.phi[0]) + gE * LOCAL.E.phi[1] * Math.sin(phi - LOCAL.E.phi[0])
          + gC * LOCAL.C.phi[1] * Math.sin(phi - LOCAL.C.phi[0])) / z;
      }
      Ed += V;
      const fi = -dV * rkj / m2, fl = dV * rkj / n2;
      const Fix = fi * mx, Fiy = fi * my, Fiz = fi * mz;
      const Flx = fl * nx, Fly = fl * ny, Flz = fl * nz;
      const a = (rijx * rkjx + rijy * rkjy + rijz * rkjz) / rkj2;
      const b = (rklx * rkjx + rkly * rkjy + rklz * rkjz) / rkj2;
      f[3 * i] += Fix; f[3 * i + 1] += Fiy; f[3 * i + 2] += Fiz;
      f[3 * l] += Flx; f[3 * l + 1] += Fly; f[3 * l + 2] += Flz;
      f[3 * j] += (a - 1) * Fix - b * Flx; f[3 * j + 1] += (a - 1) * Fiy - b * Fly; f[3 * j + 2] += (a - 1) * Fiz - b * Flz;
      f[3 * kk] += (b - 1) * Flx - a * Fix; f[3 * kk + 1] += (b - 1) * Fly - a * Fiy; f[3 * kk + 2] += (b - 1) * Flz - a * Fiz;
    }

    // nonbonded pairs, |i - j| >= 3
    const sr2 = p.repel * p.repel;
    if (sys.model === 'go') {
      const I = sys.natI, J = sys.natJ, S = sys.natS;
      for (let t = 0; t < I.length; t++) {
        const i = I[t], j = J[t];
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        const s2 = S[t] * S[t] / r2;
        if (s2 < 0.16) continue; // beyond 2.5σ
        const s10 = s2 * s2 * s2 * s2 * s2, s12 = s10 * s2;
        Ec += 5 * s12 - 6 * s10;
        const g = 60 * (s12 - s10) / r2; // -dV/dr / r
        f[3 * j] += g * dx; f[3 * j + 1] += g * dy; f[3 * j + 2] += g * dz;
        f[3 * i] -= g * dx; f[3 * i + 1] -= g * dy; f[3 * i + 2] -= g * dz;
      }
      const NI = sys.nonI, NJ = sys.nonJ;
      for (let t = 0; t < NI.length; t++) {
        const i = NI[t], j = NJ[t];
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 36) continue;
        const s2 = sr2 / r2, s6 = s2 * s2 * s2, s12 = s6 * s6;
        Er += s12;
        const g = 12 * s12 / r2;
        f[3 * j] += g * dx; f[3 * j + 1] += g * dy; f[3 * j + 2] += g * dz;
        f[3 * i] -= g * dx; f[3 * i + 1] -= g * dy; f[3 * i + 2] -= g * dz;
      }
    } else {
      // side-chain sites: each Cα pushed outward along its bend bisector by a residue-specific length
      const S = sys.site, U = sys.siteU, AL = sys.siteA, SL = sys.scLen, FS = sys.siteF, KC = sys.siteC, HS = sys.hasSite;
      for (let i = 0; i < n; i++) {
        let ax, ay, az;
        if (i === 0) { ax = x[0] - x[3]; ay = x[1] - x[4]; az = x[2] - x[5]; }
        else if (i === n - 1) { ax = x[3 * i] - x[3 * i - 3]; ay = x[3 * i + 1] - x[3 * i - 2]; az = x[3 * i + 2] - x[3 * i - 1]; }
        else {
          ax = 2 * x[3 * i] - x[3 * i - 3] - x[3 * i + 3];
          ay = 2 * x[3 * i + 1] - x[3 * i - 2] - x[3 * i + 4];
          az = 2 * x[3 * i + 2] - x[3 * i - 1] - x[3 * i + 5];
        }
        const la = Math.sqrt(ax * ax + ay * ay + az * az) || 1e-6;
        U[3 * i] = ax / la; U[3 * i + 1] = ay / la; U[3 * i + 2] = az / la; AL[i] = la;
        S[3 * i] = x[3 * i] + SL[i] * U[3 * i];
        S[3 * i + 1] = x[3 * i + 1] + SL[i] * U[3 * i + 1];
        S[3 * i + 2] = x[3 * i + 2] + SL[i] * U[3 * i + 2];
        if (KC && i > 0 && i < n - 1) {
          // fitted centroid: c0·(a − b) + c1·(a × b) + c2·(a + b)
          const a0 = x[3 * i] - x[3 * i - 3], a1 = x[3 * i + 1] - x[3 * i - 2], a2 = x[3 * i + 2] - x[3 * i - 1];
          const b0 = x[3 * i + 3] - x[3 * i], b1 = x[3 * i + 4] - x[3 * i + 1], b2 = x[3 * i + 5] - x[3 * i + 2];
          const c0 = KC[3 * i], c1 = KC[3 * i + 1], c2 = KC[3 * i + 2];
          S[3 * i] = x[3 * i] + c0 * (a0 - b0) + c1 * (a1 * b2 - a2 * b1) + c2 * (a0 + b0);
          S[3 * i + 1] = x[3 * i + 1] + c0 * (a1 - b1) + c1 * (a2 * b0 - a0 * b2) + c2 * (a1 + b1);
          S[3 * i + 2] = x[3 * i + 2] + c0 * (a2 - b2) + c1 * (a0 * b1 - a1 * b0) + c2 * (a2 + b2);
        }
      }
      FS.fill(0);
      if (sys.model === 'kb') { const e2 = kbPairs(sys, x, f); Er += e2.rep; Ec += e2.contact; Eu += e2.burial; }
      else {
        const I = sys.pI, J = sys.pJ, PE = sys.pE, PB = sys.pB;
        const r1 = p.wellIn, r2c = p.wellOut, span = r2c - r1;
        for (let t = 0; t < I.length; t++) {
          const i = I[t], j = J[t];
          // Cα–Cα: excluded volume and β-pairing
          const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < 36) {
            const r = Math.sqrt(r2);
            const s2 = sr2 / r2, s6 = s2 * s2 * s2, s12 = s6 * s6;
            Er += s12;
            let dVdr = -12 * s12 / r;
            const bb = PB[t];
            if (bb !== 0) {
              const u = (r - 5.0) / 1.0;
              if (u > -1 && u < 1) { const w = 1 - u * u; Ec += bb * w * w; dVdr += bb * (-4 * u * w); }
            }
            const g = -dVdr / r;
            f[3 * j] += g * dx; f[3 * j + 1] += g * dy; f[3 * j + 2] += g * dz;
            f[3 * i] -= g * dx; f[3 * i + 1] -= g * dy; f[3 * i + 2] -= g * dz;
          }
          // side chain–side chain: contact well, flat to r1, smoothstep to zero at r2c; kb also keeps sites apart
          const e = PE[t];
          if (e === 0 && !HS) continue;
          const sx = S[3 * j] - S[3 * i], sy = S[3 * j + 1] - S[3 * i + 1], sz = S[3 * j + 2] - S[3 * i + 2];
          const q2 = sx * sx + sy * sy + sz * sz;
          if (q2 > r2c * r2c) continue;
          const q = Math.sqrt(q2);
          let g = 0;
          if (HS && HS[i] && HS[j] && q < p.siteRepel) {
            const d = p.siteRepel - q;
            Er += p.kSite * d * d;
            g += 2 * p.kSite * d / q;
          }
          if (q <= r1) Ec += e;
          else {
            const u = (q - r1) / span;
            Ec += e * (1 - u * u * (3 - 2 * u));
            g += 6 * e * u * (1 - u) / (span * q);
          }
          if (g === 0) continue;
          FS[3 * j] += g * sx; FS[3 * j + 1] += g * sy; FS[3 * j + 2] += g * sz;
          FS[3 * i] -= g * sx; FS[3 * i + 1] -= g * sy; FS[3 * i + 2] -= g * sz;
        }
      }
      // pass side-chain forces back to the Cα beads that define each site
      for (let i = 0; i < n; i++) {
        const fx = FS[3 * i], fy = FS[3 * i + 1], fz = FS[3 * i + 2];
        if (fx === 0 && fy === 0 && fz === 0) continue;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        if (KC && i > 0 && i < n - 1) {
          const a0 = x[3 * i] - x[3 * i - 3], a1 = x[3 * i + 1] - x[3 * i - 2], a2 = x[3 * i + 2] - x[3 * i - 1];
          const b0 = x[3 * i + 3] - x[3 * i], b1 = x[3 * i + 4] - x[3 * i + 1], b2 = x[3 * i + 5] - x[3 * i + 2];
          const c0 = KC[3 * i], c1 = KC[3 * i + 1], c2 = KC[3 * i + 2];
          // force on a = (c0 + c2)·F + c1·(b × F); on b = (c2 − c0)·F + c1·(F × a)
          const Fa0 = (c0 + c2) * fx + c1 * (b1 * fz - b2 * fy), Fa1 = (c0 + c2) * fy + c1 * (b2 * fx - b0 * fz), Fa2 = (c0 + c2) * fz + c1 * (b0 * fy - b1 * fx);
          const Fb0 = (c2 - c0) * fx + c1 * (fy * a2 - fz * a1), Fb1 = (c2 - c0) * fy + c1 * (fz * a0 - fx * a2), Fb2 = (c2 - c0) * fz + c1 * (fx * a1 - fy * a0);
          f[3 * i] += Fa0 - Fb0; f[3 * i + 1] += Fa1 - Fb1; f[3 * i + 2] += Fa2 - Fb2;
          f[3 * i - 3] -= Fa0; f[3 * i - 2] -= Fa1; f[3 * i - 1] -= Fa2;
          f[3 * i + 3] += Fb0; f[3 * i + 4] += Fb1; f[3 * i + 5] += Fb2;
          continue;
        }
        if (SL[i] === 0) continue;
        const ux = U[3 * i], uy = U[3 * i + 1], uz = U[3 * i + 2], d = fx * ux + fy * uy + fz * uz;
        const c = SL[i] / AL[i];
        const px = c * (fx - d * ux), py = c * (fy - d * uy), pz = c * (fz - d * uz);
        if (i === 0 || i === n - 1) {
          const o = i === 0 ? 1 : n - 2;
          f[3 * i] += px; f[3 * i + 1] += py; f[3 * i + 2] += pz;
          f[3 * o] -= px; f[3 * o + 1] -= py; f[3 * o + 2] -= pz;
        } else {
          f[3 * i] += 2 * px; f[3 * i + 1] += 2 * py; f[3 * i + 2] += 2 * pz;
          f[3 * i - 3] -= px; f[3 * i - 2] -= py; f[3 * i - 1] -= pz;
          f[3 * i + 3] -= px; f[3 * i + 4] -= py; f[3 * i + 5] -= pz;
        }
      }
      // compaction prior: penalise Rg above the size expected for a globular protein of this length
      if (p.compact > 0) {
        let cx = 0, cy = 0, cz = 0;
        for (let i = 0; i < n; i++) { cx += x[3 * i]; cy += x[3 * i + 1]; cz += x[3 * i + 2]; }
        cx /= n; cy /= n; cz /= n;
        let s = 0;
        for (let i = 0; i < n; i++) s += (x[3 * i] - cx) ** 2 + (x[3 * i + 1] - cy) ** 2 + (x[3 * i + 2] - cz) ** 2;
        const rg = Math.sqrt(s / n), ex = rg - sys.rg0;
        if (ex > 0) {
          const k = 0.5 * p.compact;
          Eg = k * ex * ex;
          const g = -2 * k * ex / (n * rg);
          for (let i = 0; i < n; i++) {
            f[3 * i] += g * (x[3 * i] - cx); f[3 * i + 1] += g * (x[3 * i + 1] - cy); f[3 * i + 2] += g * (x[3 * i + 2] - cz);
          }
        }
      }
    }

    // evolutionary distance restraints (from the MSA network): one tabulated curve per pair on a 1 Å grid
    let Ev = 0;
    const R = sys.rest;
    if (R && p.evo > 0) {
      const RI = R.I, RJ = R.J, RV = R.V, G0 = R.grid0, GN = R.gridN, top = G0 + GN - 1, w = p.evo, smax = sys.restSepMax || Infinity;
      const RL = restList(sys, x, R, top), LR = RL.list, NR = RL.len, RC = restCoef(R);
      for (let k = 0; k < NR; k++) {
        const t = LR[k], i = RI[t], j = RJ[t];
        if (j - i > smax) continue;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= top * top) continue;
        const r = Math.sqrt(r2);
        if (r <= G0) { Ev += w * RV[t * GN]; continue; }
        // Catmull-Rom as in tableEval, with each segment's coefficients computed once per restraint set
        const xg = r - G0, kk = Math.floor(xg), tt = xg - kk, o = (t * (GN - 1) + kk) * 4;
        const c1 = RC[o + 1], c2 = RC[o + 2], c3 = RC[o + 3];
        Ev += w * (0.5 * (2 * RC[o] + c1 * tt + c2 * tt * tt + c3 * tt * tt * tt));
        const g = -w * (0.5 * (c1 + 2 * c2 * tt + 3 * c3 * tt * tt)) / r;
        f[3 * j] += g * dx; f[3 * j + 1] += g * dy; f[3 * j + 2] += g * dz;
        f[3 * i] -= g * dx; f[3 * i + 1] -= g * dy; f[3 * i + 2] -= g * dz;
      }
    }

    comp.bond = Eb; comp.angle = Ea; comp.dihedral = Ed; comp.contact = Ec; comp.repulsion = Er; comp.compact = Eg; comp.burial = Eu; comp.restraint = Ev;
    comp.total = Eb + Ea + Ed + Ec + Er + Eg + Eu + Ev;
    return comp.total;
  }

  // Restraints are phased in by sequence separation, as in trRosetta's staged minimisation: local pairs first so the
  // fragments set the handedness of helices and turns, then medium range, then everything. Distances alone cannot tell a
  // fold from its mirror image. frac = fraction of the run done.
  function restraintStage(sys, frac) {
    // never step back: lengthening a run mid-way lowers frac, but restraints already on stay on
    const stage = Math.max(sys.restStage || 0, frac < 0.12 ? 0 : frac < 0.25 ? 1 : 2);
    sys.restStage = stage;
    sys.restSepMax = stage === 0 ? 12 : stage === 1 ? 24 : Infinity;
  }
  // Annealing temperature at `step`. `anchor` ({step, T}) is set when a run is lengthened mid-way: from there the
  // temperature carries on from where it was and cools to tLo by 85% of the new length, instead of jumping back up.
  function scheduleT(step, steps, tHi, tLo, anchor) {
    if (anchor && step >= anchor.step) {
      const span = 0.85 * steps - anchor.step;
      const fr = span > 0 ? Math.min(1, (step - anchor.step) / span) : 1;
      return anchor.T * Math.pow(tLo / anchor.T, fr);
    }
    return tHi * Math.pow(tLo / tHi, Math.min(1, step / (0.85 * steps)));
  }

  // ---------- blind run control: early stopping and replica culling ----------
  // Neither rule looks at the PDB structure. At each check the caller passes the active replicas as
  // { id, x (current structure), E (current energy), dih (its dihedral energy), best (lowest energy so far) }.
  // The reference replica is the one with the lowest E + 3·dihedral, the score the final model selection uses.
  //  - culling (from 40% to 70% of the run): a replica closer to the reference's mirror image than to the reference,
  //    or one whose dihedral energy is far above it, is returned in `cull`; the caller restarts it from the reference.
  //  - early stop (from 40%): at least half of the never-restarted replicas (3 or more) lie within 2.5 + 0.015·n Å of
  //    the reference, and the lowest energy has dropped by less than 0.3% over the last 10% of the run.
  function RunMonitor(opts) {
    this.o = Object.assign({ from: 0.4, cullTo: 0.7, agree: 0.5, minAgree: 3, tol: 0.003, window: 0.1, dihW: 3, dihK: 2, dihAdd: 15 }, opts);
    this.cut = 2.5 + 0.015 * this.o.n;
    this.hist = []; this.restarted = new Set(); this.stopped = null; this.culls = [];
  }
  RunMonitor.prototype.check = function (frac, reps) {
    const o = this.o, out = { ref: -1, cull: [], stop: false, agree: 0, among: 0 };
    if (!reps.length) return out;
    const bestE = Math.min(...reps.map(q => q.best));
    this.hist.push({ frac, E: bestE });
    let ref = 0;
    reps.forEach((q, k) => { if (q.E + o.dihW * q.dih < reps[ref].E + o.dihW * reps[ref].dih) ref = k; });
    out.ref = ref;
    if (frac < o.from) return out;
    const R = reps[ref], mir = mirror(R.x);
    if (o.cull && frac <= o.cullTo) {
      reps.forEach((q, k) => {
        if (k === ref) return;
        const toRef = rmsd(q.x, R.x), toMir = rmsd(q.x, mir);
        const why = toMir < 0.8 * toRef && toMir < this.cut ? 'mirror'
          : q.dih > o.dihK * R.dih + o.dihAdd && q.E > R.E ? 'strained' : null;
        if (!why) return;
        out.cull.push({ k, id: q.id, why });
        this.restarted.add(q.id);
        this.culls.push({ id: q.id, frac, why });
      });
    }
    if (o.stop && !this.stopped) {
      const own = reps.filter((q, k) => k === ref || !this.restarted.has(q.id));
      out.agree = own.filter(q => q === R || rmsd(q.x, R.x) < this.cut).length; out.among = own.length;
      let past = null;
      for (let k = this.hist.length - 1; k >= 0; k--) if (this.hist[k].frac <= frac - o.window) { past = this.hist[k]; break; }
      const stalled = !!past && past.E - bestE < o.tol * Math.abs(bestE);
      if (stalled && out.agree >= Math.max(o.minAgree, o.agree * own.length)) { out.stop = true; this.stopped = { frac }; }
    }
    return out;
  };
  // New run length after an early stop at `step`: cool to the final temperature over `quench`·(original length), then
  // keep the usual last 15% for sampling the final model.
  function earlyStopSteps(step, steps0, every, quench) {
    return Math.ceil((step + (quench || 0.1) * steps0) / 0.85 / every) * every;
  }

  // ---------- dynamics ----------
  function Simulation(sys, opts) {
    opts = opts || {};
    this.sys = sys;
    this.n = sys.n;
    this.seed = opts.seed >>> 0;
    this.rand = mulberry32(this.seed || 1);
    this.gauss = gaussian(this.rand);
    this.dt = opts.dt || 0.01;
    this.gamma = opts.gamma || 0.5;
    this.x = opts.x ? Float64Array.from(opts.x) : randomCoil(sys.n, this.rand);
    this.v = new Float64Array(3 * sys.n);
    this.f = new Float64Array(3 * sys.n);
    this.step = 0;
    this.energy = computeForces(sys, this.x, this.f);
    this.capForces();
  }
  Simulation.prototype.capForces = function () {
    const f = this.f, fmax = 200;
    for (let i = 0; i < this.n; i++) {
      const m = Math.hypot(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
      if (m > fmax) { const s = fmax / m; f[3 * i] *= s; f[3 * i + 1] *= s; f[3 * i + 2] *= s; }
    }
  };
  // Swap in a rebuilt force field (new weights) without restarting the trajectory.
  Simulation.prototype.setSystem = function (sys) {
    this.sys = sys;
    this.energy = computeForces(sys, this.x, this.f);
    this.capForces();
  };
  // Advance nsteps of BAOAB Langevin dynamics at temperature kT.
  Simulation.prototype.run = function (nsteps, kT) {
    const x = this.x, v = this.v, f = this.f, dt = this.dt, h = 0.5 * dt, N = 3 * this.n;
    const c1 = Math.exp(-this.gamma * dt), c2 = Math.sqrt((1 - c1 * c1) * kT);
    for (let s = 0; s < nsteps; s++) {
      for (let a = 0; a < N; a++) { v[a] += h * f[a]; x[a] += h * v[a]; }
      for (let a = 0; a < N; a++) { v[a] = c1 * v[a] + c2 * this.gauss(); x[a] += h * v[a]; }
      this.energy = computeForces(this.sys, x, f);
      this.capForces();
      for (let a = 0; a < N; a++) v[a] += h * f[a];
    }
    this.step += nsteps;
    return this.energy;
  };

  // ---------- stall-triggered Monte Carlo restarts (basin hopping) ----------
  // Watches one replica. When its smoothed energy has not reached a new low for a while, it tries a Monte Carlo
  // "basin hop": perturb the structure with one of three moves (insert library fragments, swing the chain about a
  // pivot residue, or reheat), re-anneal briefly, then keep the result by the Metropolis rule — always if the energy
  // went down, with probability exp(−ΔE / T_acc) if it went up — or go back to where it was. Moves that have worked
  // are tried more often, and a failed move is not repeated next time. Hops happen only between 35% and 80% of the run,
  // so the final stretch, where the model is picked, is ordinary low-temperature sampling.
  const HOP_MOVES = ['fragments', 'pivot', 'reheat'];
  // Severity 1 (gentle) … 5 (drastic); 3 is the benchmarked default.
  const HOP_SEVERITY = {
    inserts: [1, 2, 3, 6, 10], pivots: [1, 1, 1, 2, 3], pivotDeg: [[15, 40], [25, 80], [40, 120], [70, 160], [120, 180]],
    warm: [0.15, 0.3, 0.4, 0.7, 1.0], reheat: [0.3, 0.6, 0.9, 1.4, 2.0], length: [0.025, 0.03, 0.035, 0.045, 0.06],
  };
  function Hopper(sim, opts) {
    this.sim = sim; this.steps = opts.steps; this.every = opts.every; this.frags = opts.frags || null;
    this.rand = mulberry32(((opts.seed || 1) ^ 0x5bd1e995) >>> 0);
    this.setPatience(opts.patienceSteps || (opts.patience || 0.06) * opts.steps);
    this.severity = opts.severity || 3;
    this.tAcc = opts.tAcc || 2;
    this.recent = []; this.best = Infinity; this.since = 0; this.lastFailed = null;
    this.stats = HOP_MOVES.map(name => ({ name, tries: 0, kept: 0 }));
    this.events = [];
  }
  // How many steps without a new energy low before a restart is tried (counted in samples; at least 3).
  Hopper.prototype.sev = function (key) { return HOP_SEVERITY[key][Math.max(1, Math.min(5, Math.round(this.severity))) - 1]; };
  Hopper.prototype.hopLength = function () { return Math.max(2, Math.round(this.sev('length') * this.steps / this.every)) * this.every; };
  Hopper.prototype.setPatience = function (steps) { this.patienceSteps = steps; this.window = Math.max(3, Math.round(steps / this.every)); };
  Hopper.prototype.setSampling = function (every) { this.every = every; this.setPatience(this.patienceSteps); };
  // Call after every sample (sim.step a multiple of `every`) with the scheduled temperature. Returns a hop event or null.
  Hopper.prototype.observe = function (Tbase) {
    const sim = this.sim, f = sim.step / this.steps;
    this.recent.push(sim.energy);
    if (this.recent.length > 3) this.recent.shift();
    const e = this.recent.reduce((a, b) => a + b, 0) / this.recent.length;
    if (e < this.best - 0.5) { this.best = e; this.since = 0; } else this.since++;
    if (f < 0.35 || f + this.hopLength() / this.steps >= 0.8 || this.since < this.window || this.recent.length < 3) return null;
    return this.hop(Tbase, e);
  };
  Hopper.prototype.pick = function () {
    // Thompson-style draw on (kept + 1) / (tries + 2), skipping the move that just failed
    let best = -1, bi = 0;
    this.stats.forEach((m, k) => {
      if (HOP_MOVES[k] === this.lastFailed) return;
      if (HOP_MOVES[k] === 'fragments' && !this.frags) return;
      const score = (m.kept + 1) / (m.tries + 2) + 0.25 * this.rand();
      if (score > best) { best = score; bi = k; }
    });
    return bi;
  };
  Hopper.prototype.perturb = function (kind) {
    const sim = this.sim, n = sim.n, x = sim.x, rand = this.rand;
    if (kind === 'fragments' && n >= 12) {
      const D = kbData(), th = new Float64Array(n), ph = new Float64Array(n);
      for (let i = 1; i < n - 1; i++) th[i] = angleAt(x, i - 1, i, i + 1);
      for (let i = 1; i < n - 2; i++) ph[i] = dihedral(x, i - 1, i, i + 1, i + 2);
      for (let m = 0, M = this.sev('inserts'); m < M; m++) {
        const L = rand() < 0.6 ? 9 : 3, lists = this.frags[L];
        if (!lists || !lists.length) continue;
        const p = Math.floor(rand() * lists.length), cand = lists[p];
        if (!cand || !cand.length) continue;
        const st = cand[Math.floor(rand() * cand.length)];
        for (let k = 0; k < L; k++) {
          const i = p + k, r = st + k;
          if (i >= 1 && i <= n - 2 && D.theta[r] !== 255) th[i] = D.theta[r] * DEG;
          if (i >= 1 && i <= n - 3 && D.phi[r] !== 255) ph[i] = (D.phi[r] / 254 * 360 - 180) * DEG;
        }
      }
      x.set(center(buildChain(n, th, ph)));
    } else if (kind === 'pivot' || kind === 'fragments') {
      // rotate everything after a random residue about a random axis through it (more pivots, wider swings when drastic)
      const [a0, a1] = this.sev('pivotDeg');
      for (let pv = 0, PV = this.sev('pivots'); pv < PV; pv++) {
      const i = 2 + Math.floor(rand() * Math.max(1, n - 4)), ang = (a0 + (a1 - a0) * rand()) * DEG * (rand() < 0.5 ? -1 : 1);
      let ax = gaussian(rand)(), ay = gaussian(rand)(), az = gaussian(rand)();
      const la = Math.hypot(ax, ay, az) || 1; ax /= la; ay /= la; az /= la;
      const c = Math.cos(ang), sn = Math.sin(ang), ox = x[3 * i], oy = x[3 * i + 1], oz = x[3 * i + 2];
      const tail = rand() < 0.5;   // move the C-terminal or the N-terminal side
      for (let j = 0; j < n; j++) {
        if (tail ? j <= i : j >= i) continue;
        const px = x[3 * j] - ox, py = x[3 * j + 1] - oy, pz = x[3 * j + 2] - oz, d = ax * px + ay * py + az * pz;
        x[3 * j] = ox + px * c + (ay * pz - az * py) * sn + ax * d * (1 - c);
        x[3 * j + 1] = oy + py * c + (az * px - ax * pz) * sn + ay * d * (1 - c);
        x[3 * j + 2] = oz + pz * c + (ax * py - ay * px) * sn + az * d * (1 - c);
      }
      }
    }
    sim.v.fill(0);
    sim.energy = computeForces(sim.sys, sim.x, sim.f);
    sim.capForces();
  };
  Hopper.prototype.hop = function (Tbase, E0) {
    const sim = this.sim, k = this.pick(), kind = HOP_MOVES[k], m = this.stats[k], x0 = Float64Array.from(sim.x), step0 = sim.step;
    m.tries++;
    this.perturb(kind);
    // re-anneal from a warmer temperature back down to the scheduled one, then measure the energy there
    const hopLen = this.hopLength(), relax = hopLen - this.every, Thot = Tbase + this.sev(kind === 'reheat' ? 'reheat' : 'warm'), chunks = Math.max(1, Math.round(relax / 500));
    for (let c = 0; c < chunks; c++) sim.run(Math.round(relax / chunks), Thot * Math.pow(Tbase / Thot, (c + 1) / chunks));
    let e = 0;
    const left = hopLen - (sim.step - step0), q = Math.max(1, Math.floor(left / 4));
    for (let c = 0; c < 4; c++) { sim.run(c < 3 ? q : left - 3 * q, Tbase); e += sim.energy / 4; }
    const keep = e < E0 || this.rand() < Math.exp(-(e - E0) / this.tAcc);
    if (keep) { m.kept++; this.lastFailed = null; this.best = Math.min(this.best, e); }
    else {
      this.lastFailed = kind;
      sim.x.set(x0); sim.v.fill(0);
      sim.energy = computeForces(sim.sys, sim.x, sim.f); sim.capForces();
    }
    this.since = 0; this.recent = [];
    const ev = { step: sim.step, from: step0, kind, E0, E1: e, kept: keep, severity: this.severity };
    this.events.push(ev);
    return ev;
  };

  // ---------- fragment assembly ----------
  // Rosetta-style Monte Carlo in internal coordinates: start extended, copy the library θ/φ of a picked fragment into
  // a random window, accept by Metropolis on the full energy while cooling; 9-mers first, then 3-mers.
  function assembleFragments(sys, frags, opts) {
    opts = opts || {};
    const D = kbData(), n = sys.n, rand = opts.rand || mulberry32(opts.seed || 1);
    const theta = new Float64Array(n).fill(120 * DEG), phi = new Float64Array(n).fill(180 * DEG);
    const th = new Float64Array(n), ph = new Float64Array(n), f = new Float64Array(3 * n);
    let x = buildChain(n, theta, phi), E = computeForces(sys, x, f);
    const kT0 = opts.kTHi || 2, kT1 = opts.kTLo || 0.5;
    for (const [L, M] of [[9, opts.moves9 || 40 * n], [3, opts.moves3 || 40 * n]]) {
      const lists = frags[L];
      if (!lists || !lists.length) continue;
      for (let m = 0; m < M; m++) {
        const kT = kT0 * Math.pow(kT1 / kT0, m / M);
        const p = Math.floor(rand() * lists.length), cand = lists[p];
        if (!cand || !cand.length) continue;
        const st = cand[Math.floor(rand() * cand.length)];
        th.set(theta); ph.set(phi);
        for (let k = 0; k < L; k++) {
          const i = p + k, r = st + k;
          if (i >= 1 && i <= n - 2 && D.theta[r] !== 255) th[i] = D.theta[r] * DEG;
          if (i >= 1 && i <= n - 3 && D.phi[r] !== 255) ph[i] = (D.phi[r] / 254 * 360 - 180) * DEG;
        }
        const y = buildChain(n, th, ph), Ey = computeForces(sys, y, f);
        if (Ey <= E || rand() < Math.exp(-(Ey - E) / kT)) { theta.set(th); phi.set(ph); x = y; E = Ey; }
        if (opts.onFrame && opts.every && (m + 1) % opts.every === 0) opts.onFrame(x, E, L, m + 1);
      }
    }
    return { x: center(x), E, theta, phi };
  }

  // ---------- superposition ----------
  function jacobi4(a) {
    const v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = 0;
      for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[4 * p + q] * a[4 * p + q];
      if (off < 1e-20) break;
      for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
        const apq = a[4 * p + q];
        if (Math.abs(apq) < 1e-15) continue;
        const theta = (a[4 * q + q] - a[4 * p + p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[4 * k + p], akq = a[4 * k + q];
          a[4 * k + p] = c * akp - s * akq; a[4 * k + q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[4 * p + k], aqk = a[4 * q + k];
          a[4 * p + k] = c * apk - s * aqk; a[4 * q + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = v[4 * k + p], vkq = v[4 * k + q];
          v[4 * k + p] = c * vkp - s * vkq; v[4 * k + q] = s * vkp + c * vkq;
        }
      }
    }
    return { values: [a[0], a[5], a[10], a[15]], vectors: v };
  }

  // Optimal rigid superposition of P onto Q over residues idx (Horn's quaternion method).
  // Returns { rmsd, R (row-major 3×3), cp, cq }; apply as y = R (x − cp) + cq.
  function superpose(P, Q, idx) {
    const n = P.length / 3;
    if (!idx) { idx = []; for (let i = 0; i < n; i++) idx.push(i); }
    const m = idx.length;
    const cp = [0, 0, 0], cq = [0, 0, 0];
    for (const i of idx) for (let d = 0; d < 3; d++) { cp[d] += P[3 * i + d]; cq[d] += Q[3 * i + d]; }
    for (let d = 0; d < 3; d++) { cp[d] /= m; cq[d] /= m; }
    let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0, G = 0;
    for (const i of idx) {
      const px = P[3 * i] - cp[0], py = P[3 * i + 1] - cp[1], pz = P[3 * i + 2] - cp[2];
      const qx = Q[3 * i] - cq[0], qy = Q[3 * i + 1] - cq[1], qz = Q[3 * i + 2] - cq[2];
      Sxx += px * qx; Sxy += px * qy; Sxz += px * qz;
      Syx += py * qx; Syy += py * qy; Syz += py * qz;
      Szx += pz * qx; Szy += pz * qy; Szz += pz * qz;
      G += px * px + py * py + pz * pz + qx * qx + qy * qy + qz * qz;
    }
    const N = [
      Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
      Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
      Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
      Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz,
    ];
    const { values, vectors } = jacobi4(N);
    let best = 0;
    for (let k = 1; k < 4; k++) if (values[k] > values[best]) best = k;
    const q0 = vectors[best], q1 = vectors[4 + best], q2 = vectors[8 + best], q3 = vectors[12 + best];
    const R = [
      q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
      2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1),
      2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
    ];
    const msd = Math.max(0, (G - 2 * values[best]) / m);
    return { rmsd: Math.sqrt(msd), R, cp, cq };
  }
  function transform(P, sup) {
    const n = P.length / 3, out = new Float64Array(3 * n), R = sup.R;
    for (let i = 0; i < n; i++) {
      const x = P[3 * i] - sup.cp[0], y = P[3 * i + 1] - sup.cp[1], z = P[3 * i + 2] - sup.cp[2];
      out[3 * i] = R[0] * x + R[1] * y + R[2] * z + sup.cq[0];
      out[3 * i + 1] = R[3] * x + R[4] * y + R[5] * z + sup.cq[1];
      out[3 * i + 2] = R[6] * x + R[7] * y + R[8] * z + sup.cq[2];
    }
    return out;
  }
  function deviations(A, B) {
    const n = A.length / 3, d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = Math.hypot(A[3 * i] - B[3 * i], A[3 * i + 1] - B[3 * i + 1], A[3 * i + 2] - B[3 * i + 2]);
    return d;
  }
  function rmsd(P, Q) { return superpose(P, Q).rmsd; }
  function mirror(X) { const Y = Float64Array.from(X); for (let i = 0; i < Y.length; i += 3) Y[i] = -Y[i]; return Y; }

  // TM-score (Zhang & Skolnick 2004) and GDT-TS, each maximised over superpositions grown from
  // fragment seeds, as the TM-score program does. Equal-length, residue-aligned chains.
  function structureScores(P, Q, opts) {
    opts = opts || {};
    const n = P.length / 3;
    const d0 = n > 21 ? Math.max(0.5, 1.24 * Math.cbrt(n - 15) - 1.8) : 0.5;
    const d0s = Math.min(8, Math.max(4.5, d0));
    const cuts = [1, 2, 4, 8];
    let tm = 0, tmSup = null;
    const gdt = [0, 0, 0, 0];
    const lens = [];
    for (let L = n; L >= 4; L = Math.floor(L / 2)) { lens.push(L); if (opts.fast && lens.length >= 2) break; }
    if (!lens.length) lens.push(n);
    const rules = opts.fast ? [d0s] : [d0s, 1, 2, 4, 8];
    const d = new Float64Array(n);
    const score = sup => {
      const Y = transform(P, sup);
      let s = 0; const cnt = [0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        d[i] = Math.hypot(Y[3 * i] - Q[3 * i], Y[3 * i + 1] - Q[3 * i + 1], Y[3 * i + 2] - Q[3 * i + 2]);
        s += 1 / (1 + (d[i] / d0) ** 2);
        for (let c = 0; c < 4; c++) if (d[i] <= cuts[c]) cnt[c]++;
      }
      s /= n;
      if (s > tm) { tm = s; tmSup = sup; }
      for (let c = 0; c < 4; c++) gdt[c] = Math.max(gdt[c], cnt[c] / n);
    };
    for (const L of lens) {
      const step = Math.max(1, Math.floor(L / 2));
      for (let start = 0; start + L <= n; start += step) {
        for (const cut of rules) {
          let idx = [];
          for (let i = start; i < start + L; i++) idx.push(i);
          let key = '';
          for (let it = 0; it < 12; it++) {
            const sup = superpose(P, Q, idx);
            score(sup);
            let c = cut, sel = [];
            while (true) {
              sel = [];
              for (let i = 0; i < n; i++) if (d[i] < c) sel.push(i);
              if (sel.length >= 3 || c > 20) break;
              c += 0.5;
            }
            const k = sel.join(',');
            if (k === key || sel.length < 3) break;
            key = k; idx = sel;
          }
        }
      }
    }
    return { tm, gdt: (gdt[0] + gdt[1] + gdt[2] + gdt[3]) / 4, gdtCuts: gdt, d0, tmSup };
  }

  // ---------- contacts & secondary structure ----------
  function contacts(X, cutoff, minSep) {
    cutoff = cutoff || 8; minSep = minSep || 4;
    const n = X.length / 3, out = [];
    for (let i = 0; i < n; i++) for (let j = i + minSep; j < n; j++) {
      const r = dist(X, i, j);
      if (r < cutoff) out.push([i, j, r]);
    }
    return out;
  }
  // Fraction of native contacts formed: within 1.2× their native distance.
  function fractionNative(X, nativeContacts) {
    if (!nativeContacts.length) return 0;
    let k = 0;
    for (const [i, j, r0] of nativeContacts) if (dist(X, i, j) < 1.2 * r0) k++;
    return k / nativeContacts.length;
  }
  function contactOverlap(X, nativeContacts) {
    const mine = contacts(X);
    const set = new Set(nativeContacts.map(([i, j]) => i * 10000 + j));
    let tp = 0;
    for (const [i, j] of mine) if (set.has(i * 10000 + j)) tp++;
    return { predicted: mine.length, native: nativeContacts.length, shared: tp,
      precision: mine.length ? tp / mine.length : 0, recall: nativeContacts.length ? tp / nativeContacts.length : 0 };
  }

  // Secondary structure from Cα geometry alone (helix: θ≈91°, φ≈50° in runs; strand: extended and paired).
  function assignSS(X) {
    const n = X.length / 3, ss = new Array(n).fill('C');
    const th = new Float64Array(n), ph = new Float64Array(n).fill(NaN);
    for (let j = 1; j < n - 1; j++) th[j] = angleAt(X, j - 1, j, j + 1) / DEG;
    for (let k = 1; k < n - 2; k++) ph[k] = dihedral(X, k - 1, k, k + 1, k + 2) / DEG;
    const helical = k => ph[k] > 25 && ph[k] < 80 && th[k] > 78 && th[k] < 105 && th[k + 1] > 78 && th[k + 1] < 105;
    for (let k = 1; k < n - 3; k++) {
      if (helical(k) && helical(k + 1)) for (let r = k - 1; r <= k + 3; r++) ss[r] = 'H';
    }
    const ext = new Array(n).fill(false);
    for (let k = 1; k < n - 2; k++) {
      if (Math.abs(ph[k]) > 125 && th[k] > 100 && th[k + 1] > 100) { ext[k] = true; ext[k + 1] = true; }
    }
    for (let i = 0; i < n; i++) {
      if (!ext[i] || ss[i] === 'H') continue;
      for (let j = 0; j < n; j++) {
        if (Math.abs(i - j) >= 3 && ext[j] && dist(X, i, j) < 5.6) { ss[i] = 'E'; break; }
      }
    }
    return dropShortRuns(dropShortRuns(ss.join(''), 'H', 4), 'E', 2);
  }

  function q3(a, b) {
    const norm = c => (c === 'H' || c === 'G' ? 'H' : c === 'E' ? 'E' : 'C');
    let k = 0;
    for (let i = 0; i < a.length; i++) if (norm(a[i]) === norm(b[i])) k++;
    return k / a.length;
  }

  // Everything the comparison panel shows, for model X against native N.
  function compare(X, N, nativeContacts, opts) {
    const sup = superpose(X, N);
    const s = structureScores(X, N, opts);
    const fit = transform(X, sup);
    const mirrorRmsd = rmsd(mirror(X), N);
    return {
      rmsd: sup.rmsd, tm: s.tm, gdt: s.gdt, gdtCuts: s.gdtCuts,
      q: fractionNative(X, nativeContacts),
      contacts: contactOverlap(X, nativeContacts),
      rg: radiusOfGyration(X), mirrorRmsd, fitted: fit, perResidue: deviations(fit, N),
    };
  }

  // ---------- all-atom reconstruction from a Cα trace ----------
  // Backbone C, O and N(i+1) come from a table of their average positions in a frame built on four consecutive Cα,
  // binned by the Cα dihedral (fit offline on the PDB library). Side chains use the library's most common χ1
  // rotamer for each residue type, placed on the reconstructed N–Cα–C frame.
  function withVirtualEnds(X) {
    const n = X.length / 3, Y = new Float64Array(3 * (n + 4));
    Y.set(X, 6);
    placeAtom(Y, 4, 3, 2, 1, BOND, 120 * DEG, 180 * DEG);
    placeAtom(Y, 3, 2, 1, 0, BOND, 120 * DEG, 180 * DEG);
    placeAtom(Y, n - 1, n, n + 1, n + 2, BOND, 120 * DEG, 180 * DEG);
    placeAtom(Y, n, n + 1, n + 2, n + 3, BOND, 120 * DEG, 180 * DEG);
    return Y;
  }
  // Ideal trans peptide unit Cα(i)–C(=O)–N–Cα(i+1) in its own plane: x along Cα(i)→Cα(i+1), y in-plane.
  const PEPTIDE = (function () {
    const rad = d => d * DEG, C = [1.525, 0];
    const aN = rad(180 - 116.2), N = [C[0] + 1.329 * Math.cos(aN), C[1] + 1.329 * Math.sin(aN)];
    const aO = aN - rad(123.2), O = [C[0] + 1.231 * Math.cos(aO), C[1] + 1.231 * Math.sin(aO)];
    const back = aN + Math.PI;
    let best = null;
    for (const sgn of [-1, 1]) {
      const a = back + sgn * rad(121.7), A = [N[0] + 1.458 * Math.cos(a), N[1] + 1.458 * Math.sin(a)];
      const d = Math.hypot(A[0], A[1]);
      if (!best || Math.abs(d - 3.8) < Math.abs(best.d - 3.8)) best = { A, d };
    }
    const r = -Math.atan2(best.A[1], best.A[0]), c = Math.cos(r), sn = Math.sin(r);
    const rot = q => [c * q[0] - sn * q[1], sn * q[0] + c * q[1]];
    return { C: rot(C), O: rot(O), N: rot(N), length: best.d };
  })();
  function peptideAngle(phi, t1, t2) {
    const P = root.FOLD_KB.peptide, NP = P.phiBins, NT = P.thetaBins, A = P.alpha;
    const fp = (phi + 180) / 15 - 0.5, f1 = Math.max(0, Math.min(NT - 1.001, (t1 - P.theta0) / 10 - 0.5)), f2 = Math.max(0, Math.min(NT - 1.001, (t2 - P.theta0) / 10 - 0.5));
    const p0 = Math.floor(fp), a0 = Math.floor(f1), b0 = Math.floor(f2), tp = fp - p0, ta = f1 - a0, tb = f2 - b0;
    let c = 0, s = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) {
      const w = (i ? tp : 1 - tp) * (j ? ta : 1 - ta) * (k ? tb : 1 - tb);
      const pb = (((p0 + i) % NP) + NP) % NP, ab = Math.min(NT - 1, a0 + j), bb = Math.min(NT - 1, b0 + k);
      const ang = A[(pb * NT + ab) * NT + bb] * DEG;
      c += w * Math.cos(ang); s += w * Math.sin(ang);
    }
    return Math.atan2(s, c);
  }
  function reconstructBackbone(X) {
    const n = X.length / 3, Y = withVirtualEnds(X);
    const N = new Float64Array(3 * n), C = new Float64Array(3 * n), O = new Float64Array(3 * n);
    const g = (m, d) => Y[3 * m + d];
    for (let r = -1; r < n; r++) {
      const m = r + 2;
      let xx = g(m + 1, 0) - g(m, 0), xy = g(m + 1, 1) - g(m, 1), xz = g(m + 1, 2) - g(m, 2);
      let l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
      let vx = g(m - 1, 0) - g(m, 0) + g(m + 2, 0) - g(m + 1, 0);
      let vy = g(m - 1, 1) - g(m, 1) + g(m + 2, 1) - g(m + 1, 1);
      let vz = g(m - 1, 2) - g(m, 2) + g(m + 2, 2) - g(m + 1, 2);
      const dv = vx * xx + vy * xy + vz * xz;
      vx -= dv * xx; vy -= dv * xy; vz -= dv * xz;
      l = Math.hypot(vx, vy, vz) || 1; vx /= l; vy /= l; vz /= l;
      const zx = xy * vz - xz * vy, zy = xz * vx - xx * vz, zz = xx * vy - xy * vx;
      const ph = dihedral(Y, m - 1, m, m + 1, m + 2) / DEG;
      // orientation of the peptide plane about the Cα–Cα axis: library circular mean, trilinear in (φ, θi, θi+1)
      const al = peptideAngle(ph, angleAt(Y, m - 1, m, m + 1) / DEG, angleAt(Y, m, m + 1, m + 2) / DEG);
      const sgn = PEPTIDE.O[1] < 0 ? -1 : 1, py = sgn * Math.cos(al), pz = sgn * Math.sin(al);
      const dx = py * vx + pz * zx, dy = py * vy + pz * zy, dz = py * vz + pz * zz;
      const scale = Math.hypot(g(m + 1, 0) - g(m, 0), g(m + 1, 1) - g(m, 1), g(m + 1, 2) - g(m, 2)) / PEPTIDE.length;
      const place = (atom, out, idx) => {
        const q = atom === 0 ? PEPTIDE.C : atom === 1 ? PEPTIDE.O : PEPTIDE.N, lx = q[0] * scale, ly = q[1];
        out[3 * idx] = g(m, 0) + lx * xx + ly * dx;
        out[3 * idx + 1] = g(m, 1) + lx * xy + ly * dy;
        out[3 * idx + 2] = g(m, 2) + lx * xz + ly * dz;
      };
      if (r >= 0) { place(0, C, r); place(1, O, r); }
      if (r + 1 < n) place(2, N, r + 1);
    }
    return { N, C, O };
  }
  const topoCache = {};
  function residueTemplate(aa) {
    if (topoCache[aa]) return topoCache[aa];
    const D = kbData(), sc = D.rotamers[aa] || [];
    const loc = [[1.458, 0, 0], [0, 0, 0]].concat(sc.map(a => [a[2], a[3], a[4]]));
    const el = ['N', 'C'].concat(sc.map(a => a[1]));
    const bonds = [];
    for (let i = 0; i < loc.length; i++) for (let j = Math.max(i + 1, 2); j < loc.length; j++) {
      const d = Math.hypot(loc[i][0] - loc[j][0], loc[i][1] - loc[j][1], loc[i][2] - loc[j][2]);
      if (d < (el[i] === 'S' || el[j] === 'S' ? 2.05 : 1.9)) bonds.push(i, j);
    }
    return (topoCache[aa] = { names: sc.map(a => a[0]), el: sc.map(a => a[1]), loc: sc.map(a => [a[2], a[3], a[4]]), bonds });
  }
  // Atom list and bonds for a sequence; atom order per residue: N, CA, C, O, side chain.
  function atomTopology(seq) {
    const names = [], el = [], res = [], bonds = [], start = [];
    for (let i = 0; i < seq.length; i++) {
      const t = residueTemplate(seq[i]), b = names.length;
      start.push(b);
      names.push('N', 'CA', 'C', 'O'); el.push('N', 'C', 'C', 'O'); res.push(i, i, i, i);
      bonds.push(b, b + 1, b + 1, b + 2, b + 2, b + 3);
      if (i > 0) bonds.push(start[i - 1] + 2, b);
      t.names.forEach((nm, k) => { names.push(nm); el.push(t.el[k]); res.push(i); });
      for (let k = 0; k < t.bonds.length; k += 2) {
        const map = q => (q === 0 ? b : q === 1 ? b + 1 : b + 4 + q - 2);
        bonds.push(map(t.bonds[k]), map(t.bonds[k + 1]));
      }
    }
    return { seq, names, el, res: Int32Array.from(res), bonds: Uint32Array.from(bonds), start, count: names.length };
  }
  function atomPositions(X, topo, out) {
    const n = X.length / 3, bb = reconstructBackbone(X);
    out = out || new Float32Array(3 * topo.count);
    for (let i = 0; i < n; i++) {
      const b = topo.start[i];
      for (let d = 0; d < 3; d++) {
        out[3 * b + d] = bb.N[3 * i + d]; out[3 * (b + 1) + d] = X[3 * i + d];
        out[3 * (b + 2) + d] = bb.C[3 * i + d]; out[3 * (b + 3) + d] = bb.O[3 * i + d];
      }
      const t = residueTemplate(topo.seq[i]);
      if (!t.loc.length) continue;
      let ux = bb.N[3 * i] - X[3 * i], uy = bb.N[3 * i + 1] - X[3 * i + 1], uz = bb.N[3 * i + 2] - X[3 * i + 2];
      let l = Math.hypot(ux, uy, uz); ux /= l; uy /= l; uz /= l;
      const cx = bb.C[3 * i] - X[3 * i], cy = bb.C[3 * i + 1] - X[3 * i + 1], cz = bb.C[3 * i + 2] - X[3 * i + 2];
      let wx = uy * cz - uz * cy, wy = uz * cx - ux * cz, wz = ux * cy - uy * cx;
      l = Math.hypot(wx, wy, wz); wx /= l; wy /= l; wz /= l;
      const vx = wy * uz - wz * uy, vy = wz * ux - wx * uz, vz = wx * uy - wy * ux;
      t.loc.forEach((c, k) => {
        const o = 3 * (b + 4 + k);
        out[o] = X[3 * i] + c[0] * ux + c[1] * vx + c[2] * wx;
        out[o + 1] = X[3 * i + 1] + c[0] * uy + c[1] * vy + c[2] * wy;
        out[o + 2] = X[3 * i + 2] + c[0] * uz + c[1] * vz + c[2] * wz;
      });
    }
    return out;
  }
  // Experimental heavy atoms ([name, element, x, y, z] per residue) with bonds by distance.
  function nativeAtomModel(atomsPerRes) {
    const names = [], el = [], res = [], pos = [], bonds = [], start = [];
    atomsPerRes.forEach((list, i) => {
      start.push(names.length);
      for (const a of list) { names.push(a[0]); el.push(a[1]); res.push(i); pos.push(a[2], a[3], a[4]); }
    });
    const P = Float32Array.from(pos), N = names.length;
    const near = (a, b, cut) => Math.hypot(P[3 * a] - P[3 * b], P[3 * a + 1] - P[3 * b + 1], P[3 * a + 2] - P[3 * b + 2]) < cut;
    for (let i = 0; i < atomsPerRes.length; i++) {
      const a0 = start[i], a1 = i + 1 < start.length ? start[i + 1] : N;
      for (let a = a0; a < a1; a++) for (let b = a + 1; b < a1; b++) if (near(a, b, el[a] === 'S' || el[b] === 'S' ? 2.05 : 1.9)) bonds.push(a, b);
      if (i + 1 < start.length) {
        const c = names.indexOf('C', a0), nn = names.indexOf('N', a1);
        if (c >= 0 && c < a1 && nn >= 0 && near(c, nn, 1.9)) bonds.push(c, nn);
      }
    }
    for (let a = 0; a < N; a++) if (names[a] === 'SG') for (let b = a + 1; b < N; b++) if (names[b] === 'SG' && near(a, b, 2.2)) bonds.push(a, b);
    return { names, el, res: Int32Array.from(res), bonds: Uint32Array.from(bonds), count: N, pos: P };
  }

  // ---------- consensus model ----------
  // Cluster low-temperature frames from all replicas (SPICKER-style): grow the RMSD cutoff until one frame has at
  // least a third of all frames as neighbours; that frame (the medoid of the densest cluster) is the consensus model.
  // With opts.weighted, each frame counts with Boltzmann-like weight exp(−(E − Emin)/σE), so a dense cluster of
  // high-energy frames cannot outvote a slightly smaller low-energy one.
  function consensus(structs, energies, opts) {
    opts = opts || {};
    const m = structs.length, Dm = new Float32Array(m * m);
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) Dm[i * m + j] = Dm[j * m + i] = rmsd(structs[i], structs[j]);
    const w = new Float64Array(m).fill(1);
    if (opts.weighted && m > 2) {
      let mean = 0, emin = Infinity;
      for (const e of energies) { mean += e / m; emin = Math.min(emin, e); }
      let sd = 0;
      for (const e of energies) sd += (e - mean) ** 2 / m;
      sd = Math.sqrt(sd) || 1;
      for (let i = 0; i < m; i++) w[i] = Math.exp(-(energies[i] - emin) / sd);
    }
    const wsum = w.reduce((a, b) => a + b, 0);
    let cut = 1.5, best = 0, size = 1;
    for (; cut <= 10; cut += 0.5) {
      let bi = 0, bs = -1, bc = 0;
      for (let i = 0; i < m; i++) {
        let sc = 0, c = 0;
        for (let j = 0; j < m; j++) if (Dm[i * m + j] < cut) { sc += w[j]; c++; }
        if (sc > bs || (sc === bs && energies[i] < energies[bi])) { bs = sc; bi = i; bc = c; }
      }
      best = bi; size = bc;
      if (bs >= wsum / 3) break;
    }
    if (opts.byEnergy) {
      // split the pool into clusters at this cutoff (largest first); among clusters at least a quarter the size of the
      // largest, take the one whose lower half of energies is lowest. Separates a fold from its mirror image.
      const left = new Uint8Array(m).fill(1), clusters = [];
      for (let guard = 0; guard < m; guard++) {
        let bi = -1, bc = 0;
        for (let i = 0; i < m; i++) {
          if (!left[i]) continue;
          let c = 0;
          for (let j = 0; j < m; j++) if (left[j] && Dm[i * m + j] < cut) c++;
          if (c > bc || (c === bc && bi >= 0 && energies[i] < energies[bi])) { bc = c; bi = i; }
        }
        if (bi < 0) break;
        const mem = [];
        for (let j = 0; j < m; j++) if (left[j] && Dm[bi * m + j] < cut) { mem.push(j); left[j] = 0; }
        const es = mem.map(j => energies[j]).sort((a, b) => a - b), half = es.slice(0, Math.max(1, es.length >> 1));
        clusters.push({ center: bi, size: mem.length, score: half.reduce((a, b) => a + b, 0) / half.length });
        if (clusters.length >= 6) break;
      }
      const big = clusters.filter(c => c.size >= Math.max(3, clusters[0].size / 4));
      const pick = big.reduce((a, b) => (b.score < a.score ? b : a), big[0] || clusters[0]);
      if (pick) return { index: pick.center, cutoff: cut, size: pick.size, total: m, clusters: clusters.length, largest: clusters[0].size };
    }
    return { index: best, cutoff: cut, size, total: m };
  }

  const api = {
    DEG, BOND, HYDRO, CHARGE, SIDECHAIN, CHOU_FASMAN, LOCAL, DEFAULTS,
    mulberry32, predictSS, weightsFromLabels, kbData, buildLocalModel, buildChain, randomCoil, center,
    dist, angleAt, dihedral, radiusOfGyration, wrap,
    createSystem, computeForces, comp, Simulation, assembleFragments, restraintStage, scheduleT, Hopper, RunMonitor, earlyStopSteps,
    superpose, transform, deviations, rmsd, mirror, structureScores,
    contacts, fractionNative, contactOverlap, assignSS, q3, compare,
    reconstructBackbone, atomTopology, atomPositions, nativeAtomModel, consensus,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.Fold = api;
})(typeof window !== 'undefined' ? window : this);
