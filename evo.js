/* Evolution module: multiple sequence alignment → coevolution statistics → neural network → Cα distogram and
   secondary structure → distance restraints for the folding engine.

   Pipeline (identical to the offline training code):
     1. parse an alignment (A3M, aligned FASTA, or unaligned homologs that get Smith–Waterman-aligned to the query)
     2. filter: drop duplicates, rows covering < 50% of the query, and rows ≥ 90% identical to one already kept
     3. reweight sequences (1 / number of neighbours at ≥ 80% identity) → effective sequence count Neff
     4. mean-field direct coupling analysis (DCA) and mutual information, both APC-corrected
     5. residual CNN: 1D net (profile → secondary structure), 2D dilated net (pairs → Cα distance histogram)
     6. restraints: trRosetta-style −ln(p(d) / p_ref(d)) potentials, one smooth curve per residue pair */
(function (root) {
  'use strict';
  const ALPH = 'ARNDCQEGHILKMFPSTWYV', Q = 21, GAP = 20;
  const IDX = new Uint8Array(128).fill(GAP);
  for (let k = 0; k < 20; k++) IDX[ALPH.charCodeAt(k)] = k;
  const MAX_SEQS = 1500, MIN_COV = 0.5, MAX_ID = 0.9, W_ID = 0.8, LAM_DCA = 0.5, LAM_MI = 0.05;
  const NB = 18;                                   // distance bins: <4, [4,5) … [19,20), ≥20 Å
  const BIN_LO = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

  // ---------- alignment input ----------
  function records(text) {
    const out = [];
    let cur = null;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line[0] === '#') continue;
      if (line[0] === '>') { cur = { name: line.slice(1).trim(), seq: '' }; out.push(cur); continue; }
      if (!cur) { cur = { name: 'seq' + (out.length + 1), seq: '' }; out.push(cur); }
      cur.seq += line.replace(/\s+/g, '');
      if (!text.includes('>')) cur = null;          // plain list: one sequence per line
    }
    return out.filter(r => r.seq.length);
  }

  // Smith–Waterman (BLOSUM62, affine gaps 11/1) of homolog s against the query; returns the query-anchored row.
  function alignToQuery(query, s) {
    const kb = root.FOLD_KB, ord = kb.aa, B = kb.blosum, L = query.length, M = s.length;
    const ix = c => { const k = ord.indexOf(c); return k < 0 ? -1 : k; };
    const qa = Array.from(query, ix), sa = Array.from(s, ix);
    const sub = (a, b) => (a < 0 || b < 0 ? -1 : B[a * 20 + b]);
    const GO = 11, GE = 1, NEG = -1e9;
    const H = new Float64Array((L + 1) * (M + 1)), E = new Float64Array((L + 1) * (M + 1)).fill(NEG), F = new Float64Array((L + 1) * (M + 1)).fill(NEG);
    const T = new Uint8Array((L + 1) * (M + 1));
    let best = 0, bi = 0, bj = 0;
    for (let i = 1; i <= L; i++) for (let j = 1; j <= M; j++) {
      const k = i * (M + 1) + j;
      E[k] = Math.max(E[k - 1] - GE, H[k - 1] - GO);
      F[k] = Math.max(F[k - M - 1] - GE, H[k - M - 1] - GO);
      const d = H[k - M - 2] + sub(qa[i - 1], sa[j - 1]);
      let h = 0, t = 0;
      if (d > h) { h = d; t = 1; }
      if (E[k] > h) { h = E[k]; t = 2; }
      if (F[k] > h) { h = F[k]; t = 3; }
      H[k] = h; T[k] = t;
      if (h > best) { best = h; bi = i; bj = j; }
    }
    const row = new Array(L).fill('-'), map = new Int32Array(L).fill(-1);
    let i = bi, j = bj;
    while (i > 0 && j > 0) {
      const t = T[i * (M + 1) + j];
      if (t === 0) break;
      if (t === 1) { row[i - 1] = s[j - 1]; map[i - 1] = j - 1; i--; j--; }
      else if (t === 2) j--;
      else i--;
    }
    return { row: row.join(''), map, score: best };
  }

  // Any of: A3M (lowercase insertions), aligned FASTA (equal lengths, gaps), or unaligned homologs.
  // Returns rows anchored to the query (length L, '-' for gaps); the query itself is not included.
  function parseAlignment(text, query) {
    const recs = records(text), L = query.length;
    if (!recs.length) return { rows: [], format: 'empty', dropped: 0 };
    const strip = s => s.replace(/[a-z.]/g, '');
    const upper = recs.map(r => r.seq.toUpperCase().replace(/\./g, '-'));
    const stripped = recs.map(r => strip(r.seq).toUpperCase());
    const allEq = arr => arr.every(s => s.length === arr[0].length);
    let rows, format, ref;
    if (recs.some(r => /[a-z]/.test(r.seq)) && allEq(stripped)) { format = 'A3M'; rows = stripped; ref = rows[0]; }
    else if (recs.some(r => r.seq.includes('-')) && allEq(upper)) {
      format = 'aligned FASTA';
      const cols = [];
      for (let c = 0; c < upper[0].length; c++) if (upper[0][c] !== '-') cols.push(c);
      rows = upper.map(s => cols.map(c => s[c]).join(''));
      ref = rows[0];
    } else {
      format = 'unaligned (Smith–Waterman to query)';
      const out = [];
      let dropped = 0;
      for (const s of upper) {
        const seq = s.replace(/-/g, '');
        if (seq === query) continue;
        const a = alignToQuery(query, seq);
        if (a.score < 25) { dropped++; continue; }
        out.push(a.row);
      }
      return { rows: out, format, dropped };
    }
    // rows are anchored to the first record; map that reference onto the query if they differ
    if (ref.replace(/-/g, '') !== query || ref.length !== L) {
      const refRes = [];
      for (let c = 0; c < ref.length; c++) if (ref[c] !== '-') refRes.push(c);
      const a = alignToQuery(query, ref.replace(/-/g, ''));
      const map = Int32Array.from(a.map, k => (k < 0 ? -1 : refRes[k]));
      rows = rows.map(s => Array.from(map, c => (c < 0 ? '-' : s[c])).join(''));
    }
    rows = rows.filter(s => s !== query);
    return { rows, format, dropped: 0 };
  }

  // ---------- filtering and weights ----------
  function process(query, rows, maxSeqs) {
    maxSeqs = maxSeqs || MAX_SEQS;
    const L = query.length, seen = new Set([query]), keep = [query];
    for (const r of rows) {
      if (r.length !== L || seen.has(r)) continue;
      seen.add(r);
      let cov = 0;
      for (let i = 0; i < L; i++) if (r[i] !== '-') cov++;
      if (cov < MIN_COV * L) continue;
      keep.push(r);
    }
    const enc = s => { const u = new Uint8Array(L); for (let i = 0; i < L; i++) { const c = s.charCodeAt(i); u[i] = c < 128 ? IDX[c] : GAP; } return u; };
    const K = [enc(query)], Kn = [0];
    for (let i = 0; i < L; i++) if (K[0][i] !== GAP) Kn[0]++;
    for (let s = 1; s < keep.length && K.length < maxSeqs; s++) {
      const x = enc(keep[s]);
      let g = 0;
      for (let i = 0; i < L; i++) if (x[i] !== GAP) g++;
      let redundant = false;
      for (let k = 0; k < K.length && !redundant; k++) {
        const y = K[k];
        let same = 0;
        for (let i = 0; i < L; i++) if (x[i] === y[i] && x[i] !== GAP) same++;
        if (same >= MAX_ID * Math.min(Kn[k], g)) redundant = true;
      }
      if (redundant) continue;
      K.push(x); Kn.push(g);
    }
    const N = K.length, X = new Uint8Array(N * L);
    K.forEach((u, s) => X.set(u, s * L));
    return { X, N, L };
  }

  function weights(X, N, L) {
    const cnt = new Float64Array(N).fill(1), thr = W_ID * L;
    for (let s = 0; s < N; s++) for (let t = s + 1; t < N; t++) {
      let same = 0;
      const a = s * L, b = t * L;
      for (let i = 0; i < L; i++) if (X[a + i] === X[b + i]) same++;
      if (same >= thr) { cnt[s]++; cnt[t]++; }
    }
    return Float64Array.from(cnt, c => 1 / c);
  }

  // Dot product of two contiguous runs, four accumulators for instruction-level parallelism.
  function dot(A, ia, B, ib, len) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
    for (; k + 3 < len; k += 4) {
      s0 += A[ia + k] * B[ib + k]; s1 += A[ia + k + 1] * B[ib + k + 1];
      s2 += A[ia + k + 2] * B[ib + k + 2]; s3 += A[ia + k + 3] * B[ib + k + 3];
    }
    for (; k < len; k++) s0 += A[ia + k] * B[ib + k];
    return (s0 + s1) + (s2 + s3);
  }
  // Cholesky inverse of a symmetric positive-definite n×n matrix (row-major, overwritten).
  function spdInverse(A, n, onProgress) {
    for (let j = 0; j < n; j++) {
      const rj = j * n;
      const d2 = A[rj + j] - dot(A, rj, A, rj, j);
      if (!(d2 > 0)) throw new Error('covariance matrix is not positive definite');
      const d = Math.sqrt(d2);
      A[rj + j] = d;
      for (let i = j + 1; i < n; i++) { const ri = i * n; A[ri + j] = (A[ri + j] - dot(A, ri, A, rj, j)) / d; }
      if (onProgress && j % 64 === 0) onProgress(0.4 * j / n);
    }
    // W = L⁻¹ (lower triangular), stored transposed in U so every product runs along rows: U[c·n + r] = W[r][c]
    const U = new Float64Array(n * n);
    for (let c = 0; c < n; c++) {
      const uc = c * n;
      U[uc + c] = 1 / A[uc + c];
      for (let r = c + 1; r < n; r++) { const rr = r * n; U[uc + r] = -dot(A, rr + c, U, uc + c, r - c) / A[rr + r]; }
      if (onProgress && c % 64 === 0) onProgress(0.4 + 0.3 * c / n);
    }
    // A⁻¹ = Wᵀ W:  inv[a][b] = Σ_{r ≥ max(a,b)} U[a][r] U[b][r]
    const inv = A;
    for (let a = 0; a < n; a++) {
      const ua = a * n;
      for (let b = a; b < n; b++) { const v = dot(U, ua + b, U, b * n + b, n - b); inv[a * n + b] = v; inv[b * n + a] = v; }
      if (onProgress && a % 64 === 0) onProgress(0.7 + 0.3 * a / n);
    }
    return inv;
  }

  function apc(S, L) {
    const m = new Float64Array(L);
    let tot = 0;
    for (let i = 0; i < L; i++) { let s = 0; for (let j = 0; j < L; j++) if (j !== i) s += S[i * L + j]; m[i] = s / (L - 1); tot += s; }
    const a = tot / (L * (L - 1)), out = new Float64Array(L * L);
    for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) out[i * L + j] = i === j ? 0 : S[i * L + j] - m[i] * m[j] / a;
    return out;
  }

  // Coevolution and profile features. f1: L×42 (query one-hot, profile incl. gap, entropy); f2: L×L×4
  // (DCA-APC, MI-APC, and their z-scores). onProgress(fraction) is called during the matrix inversion.
  function features(msa, onProgress) {
    const { X, N, L } = msa;
    const w = weights(X, N, L);
    let neff = 0;
    for (let s = 0; s < N; s++) neff += w[s];
    const LQ = L * Q, Fi = new Float64Array(LQ), Fij = new Float64Array(LQ * LQ);
    for (let s = 0; s < N; s++) {
      const ws = w[s] / neff, row = s * L;
      for (let i = 0; i < L; i++) {
        const a = i * Q + X[row + i];
        Fi[a] += ws;
        const base = a * LQ;
        for (let j = 0; j < L; j++) Fij[base + j * Q + X[row + j]] += ws;
      }
    }
    // ---- mean-field DCA ----
    const q1 = Q - 1, n = L * q1, C = new Float64Array(n * n);
    const fi = new Float64Array(LQ);
    for (let k = 0; k < LQ; k++) fi[k] = (1 - LAM_DCA) * Fi[k] + LAM_DCA / Q;
    for (let i = 0; i < L; i++) for (let a = 0; a < q1; a++) {
      const r = (i * q1 + a) * n, fa = fi[i * Q + a], src = (i * Q + a) * LQ;
      for (let j = 0; j < L; j++) for (let b = 0; b < q1; b++) {
        const fb = fi[j * Q + b];
        const fij = i === j ? (a === b ? fa : 0) : (1 - LAM_DCA) * Fij[src + j * Q + b] + LAM_DCA / (Q * Q);
        C[r + j * q1 + b] = fij - fa * fb;
      }
    }
    const inv = spdInverse(C, n, onProgress);
    const fn = new Float64Array(L * L), e = new Float64Array(Q * Q);
    for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) {
      e.fill(0);
      for (let a = 0; a < q1; a++) for (let b = 0; b < q1; b++) e[a * Q + b] = -inv[(i * q1 + a) * n + j * q1 + b];
      const ra = new Float64Array(Q), cb = new Float64Array(Q);
      let all = 0;
      for (let a = 0; a < Q; a++) for (let b = 0; b < Q; b++) { const v = e[a * Q + b]; ra[a] += v / Q; cb[b] += v / Q; all += v; }
      all /= Q * Q;
      let s = 0;
      for (let a = 0; a < q1; a++) for (let b = 0; b < q1; b++) { const v = e[a * Q + b] - ra[a] - cb[b] + all; s += v * v; }
      fn[i * L + j] = fn[j * L + i] = Math.sqrt(s);
    }
    // ---- mutual information ----
    const pi = new Float64Array(LQ);
    for (let k = 0; k < LQ; k++) pi[k] = (1 - LAM_MI) * Fi[k] + LAM_MI / Q;
    const mi = new Float64Array(L * L);
    for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) {
      let s = 0;
      for (let a = 0; a < Q; a++) {
        const src = (i * Q + a) * LQ + j * Q, pa = pi[i * Q + a];
        for (let b = 0; b < Q; b++) { const p = (1 - LAM_MI) * Fij[src + b] + LAM_MI / (Q * Q); s += p * Math.log(p / (pa * pi[j * Q + b])); }
      }
      mi[i * L + j] = mi[j * L + i] = s;
    }
    const maps = [apc(fn, L), apc(mi, L)];
    const f2 = new Float32Array(L * L * 4);
    maps.forEach((M, c) => {
      let mean = 0, sq = 0;
      for (let k = 0; k < L * L; k++) mean += M[k];
      mean /= L * L;
      for (let k = 0; k < L * L; k++) sq += (M[k] - mean) ** 2;
      const sd = Math.sqrt(sq / (L * L)) + 1e-6;
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
        const k = i * L + j;
        f2[k * 4 + c] = M[k];
        f2[k * 4 + 2 + c] = i === j ? 0 : (M[k] - mean) / sd;
      }
    });
    const f1 = new Float32Array(L * 42), cons = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      const x = X[i];
      if (x < 20) f1[i * 42 + x] = 1;
      let ent = 0;
      for (let a = 0; a < Q; a++) { f1[i * 42 + 20 + a] = Fi[i * Q + a]; const p = pi[i * Q + a]; ent -= p * Math.log(p); }
      f1[i * 42 + 41] = ent / Math.log(Q);
      cons[i] = 1 - ent / Math.log(Q);
    }
    return { f1, f2, neff, N, L, dca: maps[0], mi: maps[1], profile: Fi, conservation: cons };
  }

  // ---------- neural network ----------
  function f16(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    if (e === 0) return s * m * Math.pow(2, -24);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  function loadModel(spec) {
    const b = atob(spec.data), u8 = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) u8[i] = b.charCodeAt(i);
    const u16 = new Uint16Array(u8.buffer), W = {};
    for (const [name, shape, off, len] of spec.tensors) {
      const t = new Float32Array(len);
      for (let k = 0; k < len; k++) t[k] = f16(u16[off + k]);
      W[name] = { shape, data: t };
    }
    return { W, cfg: spec.config };
  }
  const elu = v => (v > 0 ? v : Math.exp(v) - 1);
  function instNorm(x, C, P, g, b, act) {
    for (let c = 0; c < C; c++) {
      const o = c * P;
      let m = 0, v = 0;
      for (let k = 0; k < P; k++) m += x[o + k];
      m /= P;
      for (let k = 0; k < P; k++) { const d = x[o + k] - m; v += d * d; }
      const sc = g[c] / Math.sqrt(v / P + 1e-5), sh = b[c] - m * sc;
      const out = new Float32Array(P);
      for (let k = 0; k < P; k++) { const y = x[o + k] * sc + sh; out[k] = act ? elu(y) : y; }
      x.set(out, o);
    }
    return x;
  }
  function conv1d(x, Cin, L, w, bias, Cout, K, d) {
    const y = new Float32Array(Cout * L), pad = d * (K - 1) / 2;
    for (let o = 0; o < Cout; o++) {
      const yo = o * L;
      for (let i = 0; i < L; i++) y[yo + i] = bias[o];
      for (let c = 0; c < Cin; c++) for (let t = 0; t < K; t++) {
        const wv = w[(o * Cin + c) * K + t], sh = t * d - pad, xc = c * L;
        if (wv === 0) continue;
        const i0 = Math.max(0, -sh), i1 = Math.min(L, L - sh);
        for (let i = i0; i < i1; i++) y[yo + i] += wv * x[xc + i + sh];
      }
    }
    return y;
  }
  // 3×3 dilated convolution, zero padding, C → C channels; four output channels per pass over the input.
  function conv2d3(x, C, L, w, bias, d) {
    const P = L * L, y = new Float32Array(C * P);
    for (let o = 0; o < C; o += 4) {
      const m = Math.min(4, C - o), Y = [o * P, (o + 1) * P, (o + 2) * P, (o + 3) * P];
      for (let u = 0; u < m; u++) y.fill(bias[o + u], Y[u], Y[u] + P);
      const y0 = Y[0], y1 = Y[1], y2 = Y[2], y3 = Y[3];
      for (let c = 0; c < C; c++) {
        const xc = c * P;
        for (let ky = 0; ky < 3; ky++) {
          const dy = (ky - 1) * d, r0 = Math.max(0, -dy), r1 = Math.min(L, L - dy);
          for (let kx = 0; kx < 3; kx++) {
            const tap = ky * 3 + kx, dx = (kx - 1) * d, c0 = Math.max(0, -dx), c1 = Math.min(L, L - dx);
            const w0 = w[(o * C + c) * 9 + tap], w1 = m > 1 ? w[((o + 1) * C + c) * 9 + tap] : 0;
            const w2 = m > 2 ? w[((o + 2) * C + c) * 9 + tap] : 0, w3 = m > 3 ? w[((o + 3) * C + c) * 9 + tap] : 0;
            if (m === 4) {
              for (let r = r0; r < r1; r++) {
                const yr = r * L, xr = xc + (r + dy) * L + dx;
                for (let q = c0; q < c1; q++) {
                  const v = x[xr + q], k = yr + q;
                  y[y0 + k] += w0 * v; y[y1 + k] += w1 * v; y[y2 + k] += w2 * v; y[y3 + k] += w3 * v;
                }
              }
            } else {
              const ws = [w0, w1, w2, w3];
              for (let u = 0; u < m; u++) for (let r = r0; r < r1; r++) {
                const yr = Y[u] + r * L, xr = xc + (r + dy) * L + dx;
                for (let q = c0; q < c1; q++) y[yr + q] += ws[u] * x[xr + q];
              }
            }
          }
        }
      }
    }
    return y;
  }

  // Runs the network. Returns { ss: L×3 (H, E, C), dist: L×L×18 probabilities }.
  function predict(model, feat, onProgress) {
    const { W, cfg } = model, L = feat.L, C1 = cfg.c1, C2 = cfg.c2, P1 = cfg.p1;
    const ln = Math.log(feat.neff + 1) / 5;
    // 1D input: channels × L
    let h = new Float32Array(43 * L);
    for (let i = 0; i < L; i++) { for (let c = 0; c < 42; c++) h[c * L + i] = feat.f1[i * 42 + c]; h[42 * L + i] = ln; }
    h = conv1d(h, 43, L, W['in1.weight'].data, W['in1.bias'].data, C1, 1, 1);
    cfg.dil1.forEach((d, k) => {
      const pre = `r1.${k}.`;
      let t = instNorm(Float32Array.from(h), C1, L, W[pre + 'n1.weight'].data, W[pre + 'n1.bias'].data, true);
      t = conv1d(t, C1, L, W[pre + 'c1.weight'].data, W[pre + 'c1.bias'].data, C1, 3, d);
      t = instNorm(t, C1, L, W[pre + 'n2.weight'].data, W[pre + 'n2.bias'].data, true);
      t = conv1d(t, C1, L, W[pre + 'c2.weight'].data, W[pre + 'c2.bias'].data, C1, 3, d);
      for (let k2 = 0; k2 < h.length; k2++) h[k2] += t[k2];
    });
    h = instNorm(h, C1, L, W['n1.weight'].data, W['n1.bias'].data, true);
    const ssl = conv1d(h, C1, L, W['ss.weight'].data, W['ss.bias'].data, 3, 1, 1);
    const ss = new Float32Array(L * 3);
    for (let i = 0; i < L; i++) {
      const v = [ssl[i], ssl[L + i], ssl[2 * L + i]], m = Math.max(...v), e = v.map(z => Math.exp(z - m)), z = e[0] + e[1] + e[2];
      for (let c = 0; c < 3; c++) ss[i * 3 + c] = e[c] / z;
    }
    const p = conv1d(h, C1, L, W['proj.weight'].data, W['proj.bias'].data, P1, 1, 1);
    // 2D input: 1×1 conv over [p_i, p_j, f2(4), ln, log1p(sep)/5, sep<6, sep<12, sep<24]
    const P = L * L, w2 = W['in2.weight'].data, b2 = W['in2.bias'].data, cin = 2 * P1 + 9;
    let g = new Float32Array(C2 * P);
    for (let o = 0; o < C2; o++) {
      const A = new Float32Array(L), B = new Float32Array(L), wo = o * cin;
      for (let i = 0; i < L; i++) { let a = 0, b = 0; for (let c = 0; c < P1; c++) { a += w2[wo + c] * p[c * L + i]; b += w2[wo + P1 + c] * p[c * L + i]; } A[i] = a; B[i] = b; }
      const wf = k => w2[wo + 2 * P1 + k];
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
        const k = i * L + j, f = feat.f2, sep = Math.abs(i - j);
        g[o * P + k] = b2[o] + A[i] + B[j] + wf(0) * f[k * 4] + wf(1) * f[k * 4 + 1] + wf(2) * f[k * 4 + 2] + wf(3) * f[k * 4 + 3]
          + wf(4) * ln + wf(5) * Math.log1p(sep) / 5 + wf(6) * (sep < 6) + wf(7) * (sep < 12) + wf(8) * (sep < 24);
      }
    }
    cfg.dil2.forEach((d, k) => {
      const pre = `r2.${k}.`;
      let t = instNorm(Float32Array.from(g), C2, P, W[pre + 'n1.weight'].data, W[pre + 'n1.bias'].data, true);
      t = conv2d3(t, C2, L, W[pre + 'c1.weight'].data, W[pre + 'c1.bias'].data, d);
      t = instNorm(t, C2, P, W[pre + 'n2.weight'].data, W[pre + 'n2.bias'].data, true);
      t = conv2d3(t, C2, L, W[pre + 'c2.weight'].data, W[pre + 'c2.bias'].data, d);
      for (let q = 0; q < g.length; q++) g[q] += t[q];
      if (onProgress) onProgress((k + 1) / cfg.dil2.length);
    });
    g = instNorm(g, C2, P, W['n2.weight'].data, W['n2.bias'].data, true);
    const wo = W['out.weight'].data, bo = W['out.bias'].data, logit = new Float32Array(NB * P);
    for (let o = 0; o < NB; o++) for (let k = 0; k < P; k++) {
      let s = bo[o];
      for (let c = 0; c < C2; c++) s += wo[o * C2 + c] * g[c * P + k];
      logit[o * P + k] = s;
    }
    const dist = new Float32Array(P * NB), v = new Float64Array(NB);
    for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
      let m = -1e9;
      for (let o = 0; o < NB; o++) { v[o] = 0.5 * (logit[o * P + i * L + j] + logit[o * P + j * L + i]); if (v[o] > m) m = v[o]; }
      let z = 0;
      for (let o = 0; o < NB; o++) { v[o] = Math.exp(v[o] - m); z += v[o]; }
      for (let o = 0; o < NB; o++) dist[(i * L + j) * NB + o] = v[o] / z;
    }
    return { ss, dist, L };
  }

  // ---------- derived quantities ----------
  // P(dCα < cut) from the distogram (cut on a bin edge: 8 → bins <4 … [7,8)).
  function contactMap(pred, cut) {
    const L = pred.L, nb = BIN_LO.findIndex(lo => lo >= (cut || 8)), out = new Float32Array(L * L);
    for (let k = 0; k < L * L; k++) { let s = 0; for (let b = 0; b < nb; b++) s += pred.dist[k * NB + b]; out[k] = s; }
    return out;
  }
  // Expected distance (bins capped at 21 Å), for display.
  function expectedDistance(pred) {
    const L = pred.L, out = new Float32Array(L * L), mid = BIN_LO.map((lo, b) => (b === 0 ? 3.6 : b === NB - 1 ? 21 : lo + 0.5));
    for (let k = 0; k < L * L; k++) { let s = 0; for (let b = 0; b < NB; b++) s += pred.dist[k * NB + b] * mid[b]; out[k] = s; }
    return out;
  }
  // Labels from the network's helix/strand/coil probabilities; with `blend` (an array of {H, E, C}, e.g. the fragment
  // vote) and weight w, from w·network + (1 − w)·blend.
  function ssLabels(pred, blend, w) {
    let s = '';
    const a = blend ? (w != null ? w : 0.5) : 1;
    for (let i = 0; i < pred.L; i++) {
      let h = pred.ss[3 * i], e = pred.ss[3 * i + 1], c = pred.ss[3 * i + 2];
      if (blend) { h = a * h + (1 - a) * blend[i].H; e = a * e + (1 - a) * blend[i].E; c = a * c + (1 - a) * blend[i].C; }
      s += h >= e && h >= c ? 'H' : e >= c ? 'E' : 'C';
    }
    return s.replace(/H+/g, r => (r.length >= 4 ? r : 'C'.repeat(r.length))).replace(/E+/g, r => (r.length >= 3 ? r : 'C'.repeat(r.length)));
  }
  // Restraint curves on a 1 Å grid (3.5 … 19.5 Å), V = 0 beyond. Two reference states:
  //   'dfire' (default): trRosetta's V(d) = −ln[p(d) / p(d_N)] + α ln(d / d_N), which also adds a generic pull inward;
  //   'background': V(d) = −ln[p(d) / p_bg(d)], p_bg the training-set distance distribution for this sequence
  //                separation and chain length, so an uninformative prediction gives a flat curve.
  // Curves are shifted to 0 at the last bin. Only pairs |i−j| ≥ minSep with P(d < 20 Å) ≥ minP are kept.
  function sepClass(s) { return s <= 30 ? s : s <= 40 ? 31 : s <= 60 ? 32 : s <= 100 ? 33 : 34; }
  function restraints(pred, opts) {
    opts = opts || {};
    const L = pred.L, minSep = opts.minSep || 3, minP = opts.minP != null ? opts.minP : 0.3, alpha = 1.57, eps = 1e-4;
    const R = opts.reference || (root.FOLD_EVO_MODEL && root.FOLD_EVO_MODEL.reference), useBg = opts.ref === 'background' && R;
    const g = useBg ? R.groups.filter(x => L >= x).length : 0;
    const I = [], J = [], V = [], conf = [];
    for (let i = 0; i < L; i++) for (let j = i + minSep; j < L; j++) {
      const k = (i * L + j) * NB;
      let p20 = 0;
      for (let b = 0; b < 17; b++) p20 += pred.dist[k + b];
      if (p20 < minP) continue;
      I.push(i); J.push(j); conf.push(p20);
      if (useBg) {
        // a floor of 0.005 keeps bins that are near-impossible for this separation from becoming deep wells
        const r = (g * 35 + sepClass(j - i)) * NB, fl = 0.005, last = -Math.log((pred.dist[k + 16] + fl) / (R.table[r + 16] + fl));
        for (let b = 0; b < 17; b++) V.push(-Math.log((pred.dist[k + b] + fl) / (R.table[r + b] + fl)) - last);
      } else {
        const pN = pred.dist[k + 16];
        for (let b = 0; b < 17; b++) V.push(-Math.log((pred.dist[k + b] + eps) / (pN + eps)) + alpha * Math.log((3.5 + b) / 19.5));
      }
    }
    return { I: Int32Array.from(I), J: Int32Array.from(J), V: Float32Array.from(V), conf: Float32Array.from(conf), grid0: 3.5, gridN: 17 };
  }
  // Precision of the top-k predicted pairs against a native Cα structure (flat xyz), within a separation range.
  function precision(prob, L, X, lo, hi, k, cut) {
    const cand = [];
    for (let i = 0; i < L; i++) for (let j = i + lo; j < Math.min(L, i + hi); j++) cand.push([prob[i * L + j], i, j]);
    cand.sort((a, b) => b[0] - a[0]);
    const top = cand.slice(0, k);
    if (!top.length) return null;
    let hit = 0;
    for (const [, i, j] of top) {
      const dx = X[3 * i] - X[3 * j], dy = X[3 * i + 1] - X[3 * j + 1], dz = X[3 * i + 2] - X[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz < (cut || 8) ** 2) hit++;
    }
    return hit / top.length;
  }

  root.Evo = { ALPH, NB, BIN_LO, MAX_SEQS, parseAlignment, alignToQuery, process, weights, features, loadModel, predict,
    contactMap, expectedDistance, ssLabels, restraints, precision, spdInverse };
})(typeof self !== 'undefined' ? self : this);
