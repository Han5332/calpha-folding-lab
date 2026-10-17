/* Cα Folding Lab — run control, analysis, 3D view and charts.
   Needs natives.js (FOLD_NATIVES), kb.js (FOLD_KB), engine.js (Fold), viewer3d.js (FoldViewer) and three.js. */
(function () {
  'use strict';
  const F = window.Fold;
  const $ = id => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ---------- targets ----------
  function guidesFromAtoms(atoms) {
    const n = atoms.length, g = new Float64Array(3 * n);
    atoms.forEach((list, i) => {
      const c = list.find(a => a[0] === 'C'), o = list.find(a => a[0] === 'O');
      if (c && o) { g[3 * i] = o[2] - c[2]; g[3 * i + 1] = o[3] - c[3]; g[3 * i + 2] = o[4] - c[4]; }
    });
    return g;
  }
  function prepTarget(p) {
    const X = p.ca ? Float64Array.from(p.ca) : null;
    const t = Object.assign({}, p, { X, contacts: X ? F.contacts(X) : [], rg: X ? F.radiusOfGyration(X) : null, cf: F.predictSS(p.seq), lmCache: {} });
    if (p.atoms) { t.atomModel = F.nativeAtomModel(p.atoms); t.guides = guidesFromAtoms(p.atoms); }
    return t;
  }
  const BUILTIN = window.FOLD_NATIVES.map(prepTarget);
  const KB = window.FOLD_KB;
  const AA3 = { A: 'Ala', R: 'Arg', N: 'Asn', D: 'Asp', C: 'Cys', Q: 'Gln', E: 'Glu', G: 'Gly', H: 'His', I: 'Ile',
    L: 'Leu', K: 'Lys', M: 'Met', F: 'Phe', P: 'Pro', S: 'Ser', T: 'Thr', W: 'Trp', Y: 'Tyr', V: 'Val' };
  const SS_NAME = { H: 'helix', G: '3₁₀ helix', E: 'strand', C: 'coil' };

  const MSA = window.FOLD_MSA || {}, EVO_MODEL = window.FOLD_EVO_MODEL || null;
  const METHODS = {
    evo: { name: 'MSA + AI', badge: 'v8', tHi: 2.0, tLo: 0.4,
      desc: 'A neural network reads an alignment of related sequences and predicts how far apart every residue pair is. Those distances steer the v2 force field.',
      sliders: [
        { key: 'evo', label: 'AI distance restraints', hint: 'Weight of the network’s distance potentials, learned from the MSA' },
        { key: 'local', label: 'Local structure (fragments)', hint: 'Fragment-derived angle and dihedral potentials, picked with the AI secondary structure' },
        { key: 'burial', label: 'Burial', hint: 'Nonpolar residues prefer many neighbours, charged ones the surface' },
        { key: 'contact', label: 'Pair preferences', hint: 'Statistical residue–residue contact energies' },
        { key: 'beta', label: 'Strand pairing', hint: 'Extended residues pair like a β-sheet' },
        { key: 'hbond', label: 'Backbone hydrogen bonds', hint: 'Rewards any residue pair in H-bond geometry, whatever the predicted secondary structure' },
        { key: 'compact', label: 'Compactness prior', hint: 'Penalty for Rg above 2.2·N^0.38 Å' },
      ] },
    kb: { name: 'Knowledge-based', badge: 'v2', tHi: 2.0, tLo: 0.4,
      desc: `Fragments and statistics from ${KB.chains} unrelated PDB chains. Never sees this protein's structure.`,
      sliders: [
        { key: 'local', label: 'Local structure (fragments)', hint: 'Weight of the fragment-derived angle and dihedral potentials' },
        { key: 'burial', label: 'Burial', hint: 'Nonpolar residues prefer many neighbours, charged ones the surface' },
        { key: 'contact', label: 'Pair preferences', hint: 'Statistical residue–residue contact energies' },
        { key: 'beta', label: 'Strand pairing', hint: 'Extended residues pair like a β-sheet' },
        { key: 'hbond', label: 'Backbone hydrogen bonds', hint: 'Rewards any residue pair in H-bond geometry, whatever the predicted secondary structure' },
        { key: 'compact', label: 'Compactness prior', hint: 'Penalty for Rg above 2.2·N^0.38 Å' },
      ] },
    seq: { name: 'Simple physics', badge: 'v1', tHi: 2.0, tLo: 0.4,
      desc: 'Chou–Fasman propensities and hydrophobic attraction: the first version of this lab, kept as a baseline.',
      sliders: [
        { key: 'hydro', label: 'Hydrophobic burial', hint: 'Attraction between nonpolar side chains' },
        { key: 'local', label: 'Local structure', hint: 'Pull toward helix or strand geometry' },
        { key: 'charge', label: 'Salt bridges', hint: 'Opposite charges attract, like charges repel' },
        { key: 'beta', label: 'Strand pairing', hint: 'Extended residues pair at 5 Å' },
        { key: 'compact', label: 'Compactness prior', hint: 'Penalty for Rg above 2.2·N^0.38 Å' },
      ] },
    go: { name: 'Native contacts (Gō)', badge: '', tHi: 1.1, tLo: 0.5,
      desc: 'Built from the PDB contacts (Clementi et al., 2000). It knows the destination and shows the route.', sliders: [] },
  };
  const HBOND_DEFAULT = 1;   // from the benchmark (Method tab)
  const DEF_PARAMS = {
    evo: { evo: 1, hbond: HBOND_DEFAULT, local: F.DEFAULTS.kb.local, burial: F.DEFAULTS.kb.burial, contact: F.DEFAULTS.kb.contact, beta: F.DEFAULTS.kb.beta, compact: F.DEFAULTS.kb.compact },
    kb: { hbond: 0, local: F.DEFAULTS.kb.local, burial: F.DEFAULTS.kb.burial, contact: F.DEFAULTS.kb.contact, beta: F.DEFAULTS.kb.beta, compact: F.DEFAULTS.kb.compact },
    seq: { hydro: F.DEFAULTS.seq.hydro, local: F.DEFAULTS.seq.local, charge: 1, beta: 1, compact: 1 },
    go: {},
  };
  const STEP_CHOICES = [50e3, 100e3, 150e3, 200e3, 300e3, 400e3, 500e3, 600e3, 800e3, 1e6, 1.2e6, 1.5e6, 2e6, 3e6, 4e6, 5e6, 6e6, 8e6, 10e6, 12e6, 15e6, 20e6, 25e6, 30e6, 40e6, 50e6, 75e6, 100e6];
  const MAX_RES = 300;   // longest chain the lab accepts (the network was trained on 10–400 residues)
  function autoStepsIdx(n) {
    const want = clamp(15000 * n, 150e3, 1.2e6);
    let best = 0;
    STEP_CHOICES.forEach((v, i) => { if (Math.abs(v - want) < Math.abs(STEP_CHOICES[best] - want)) best = i; });
    return best;
  }

  const HKEY = 'ca-fold-lab/runs/v3';
  const state = {
    target: BUILTIN[1], method: 'evo', ssPref: 'pred', ssSource: 'pred',
    params: JSON.parse(JSON.stringify(DEF_PARAMS)),
    stepsIdx: autoStepsIdx(BUILTIN[1].seq.length), tHi: METHODS.evo.tHi, tLo: METHODS.evo.tLo, seed: 11, replicas: 12, watch: false,
    run: null, focus: 0, scrub: null, frame: 'live', tab: 'traj',
    hopOn: true, hopSeverity: 3, stopAtFold: false, view: { rep: 'cartoon', color: 'rainbow', layout: 'overlay', showModel: true, showPdb: true, spin: false },
    history: loadHistory(), queue: [], replaying: false,
  };

  // ---------- theme ----------
  const TOKENS = ['bg', 'panel', 'panel-2', 'ink', 'ink-2', 'muted', 'line', 'grid', 'model', 'native', 'helix', 'strand', 'coil',
    'heat-lo', 'heat-hi', 'ramp-a', 'ramp-b', 'good', 'warn', 'bad', 'stage', 'accent'];
  const theme = {}, trgb = {};
  const hexToRgb = h => { h = h.replace('#', '').trim(); if (h.length === 3) h = h.split('').map(c => c + c).join(''); const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const css = c => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
  const unit = c => [c[0] / 255, c[1] / 255, c[2] / 255];
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of TOKENS) { theme[k] = cs.getPropertyValue('--' + k).trim(); trgb[k] = hexToRgb(theme[k]); }
    theme.fontUi = cs.getPropertyValue('--font-ui').trim();
    theme.fontMono = cs.getPropertyValue('--font-mono').trim();
    if (viewer) viewer.setBackground(unit(trgb.stage));
    pdbDirty = true;
    dirty.all();
  }
  function hsl(h, s, l) {
    const a = s * Math.min(l, 1 - l), f = n => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [f(0), f(8), f(4)];
  }
  const rainbow = (i, n) => hsl(240 * (1 - i / Math.max(1, n - 1)), 0.72, 0.52);

  // ---------- formatting ----------
  const fmtSteps = n => (n >= 1e6 ? +(n / 1e6).toFixed(n % 1e6 ? 1 : 0) + 'M' : n >= 1e3 ? Math.round(n / 1e3) + 'k' : String(Math.round(n)));
  const fmtInt = n => Math.round(n).toLocaleString('en-US');
  const pct = v => Math.round(v * 100) + '%';
  const f1 = v => (v == null || !isFinite(v) ? '–' : v.toFixed(1));
  const f2 = v => (v == null || !isFinite(v) ? '–' : v.toFixed(2));
  const signed = v => (v == null || !isFinite(v) ? '–' : v < 0 ? '−' + Math.abs(v).toFixed(1) : v.toFixed(1));
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const ICON = {
    good: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M4.6 8.2l2.2 2.2 4.6-4.8" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    warn: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.2l7 12.6H1z" fill="currentColor"/><path d="M8 6v3.6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.8" r="1" fill="#fff"/></svg>',
    bad: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M5.4 5.4l5.2 5.2M10.6 5.4l-5.2 5.2" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>',
    none: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  };
  function verdict(n, m) {
    if (!m) return { tone: 'none', text: 'No reference structure' };
    if (n >= 25 ? m.rmsd <= 2.5 || m.tm >= 0.7 : m.rmsd <= 1.5 + 0.03 * n) return { tone: 'good', text: 'Near-native' };
    if (n >= 25 ? m.tm >= 0.5 || m.rmsd <= 4 : m.rmsd <= 1.5 + 0.1 * n && m.q >= 0.3) return { tone: 'warn', text: 'Same overall fold' };
    return { tone: 'bad', text: 'Different fold' };
  }

  // ---------- history ----------
  function loadHistory() { try { return JSON.parse(localStorage.getItem(HKEY) || '[]'); } catch (e) { return []; } }
  function saveHistory() { try { localStorage.setItem(HKEY, JSON.stringify(state.history.slice(0, 80))); } catch (e) { /* storage unavailable */ } }

  // ---------- run setup ----------
  const totalSteps = () => STEP_CHOICES[state.stepsIdx];
  // Monte Carlo restarts: steps a replica may go without a new energy low before one is tried (0 = off)
  const HOP_PATIENCE = 50e3;
  const DIHEDRAL_SEL = 3;          // weight of the dihedral term in MSA + AI model selection (see finishRun)
  const SEVERITY_NAMES = ['Gentle', 'Mild', 'Default', 'Strong', 'Drastic'];
  const SEVERITY_HINTS = ['1 fragment swap, pivots of 15–40°, mild reheat', '2 fragment swaps, pivots of 25–80°',
    '3 fragment swaps, pivots of 40–120° (the benchmarked setting)', '6 fragment swaps, two pivots of 70–160°, hot reheat',
    '10 fragment swaps, three pivots of 120–180°, very hot reheat and a longer re-anneal'];
  function renderHop() {
    $('hop').checked = state.hopOn;
    const s = $('hop-sev');
    s.value = state.hopSeverity; s.disabled = !state.hopOn;
    $('hop-sev-out').textContent = SEVERITY_NAMES[state.hopSeverity - 1];
    $('hop-sev-hint').textContent = SEVERITY_HINTS[state.hopSeverity - 1];
    $('hop-sev').closest('.slider').classList.toggle('off', !state.hopOn);
  }
  const sampleEvery = steps => Math.max(500, Math.round(steps / 300 / 500) * 500);
  function localModel(t, ss) {
    const key = ss === 'real' && t.ss ? 'real' : 'pred';
    if (!t.lmCache[key]) t.lmCache[key] = F.buildLocalModel(t.seq, { ss: key === 'real' ? t.ss : null });
    return t.lmCache[key];
  }
  const engineModel = m => (m === 'evo' ? 'kb' : m);
  function methodInputs(t, method, ss) {
    if (method === 'evo') {
      const ev = t.evo, real = ss === 'real' && t.ss;
      if (!ev || ev.status !== 'ready') { const lm = localModel(t, ss); return { ssWeights: lm.probs, local: lm.local, pred: lm.labels, restraints: null, needsEvo: true }; }
      if (ev.shallow) { const lm = localModel(t, ss); return { ssWeights: lm.probs, local: lm.local, pred: lm.labels, restraints: null, frags: lm.frags }; }
      const key = real ? 'real' : 'ai';
      if (!t.lmCache[key]) t.lmCache[key] = F.buildLocalModel(t.seq, { ss: real ? t.ss : ev.labels });
      const lm = t.lmCache[key];
      return { ssWeights: lm.probs, local: lm.local, pred: real ? t.ss : ev.labels, restraints: ev.rest, frags: lm.frags };
    }
    if (method === 'kb') { const lm = localModel(t, ss); return { ssWeights: lm.probs, local: lm.local, pred: lm.labels, frags: lm.frags }; }
    if (method === 'seq') {
      const real = ss === 'real' && t.ss;
      return { ssWeights: real ? F.weightsFromLabels(t.ss) : t.cf.probs, local: null, pred: real ? t.ss : t.cf.labels };
    }
    return { ssWeights: null, local: null, pred: t.ss || t.cf.labels };
  }
  function buildSys(run) {
    return F.createSystem({ seq: run.target.seq, model: engineModel(run.method), native: run.target.X, ssWeights: run.ssWeights, local: run.local, params: run.params, restraints: run.restraints });
  }
  function relaxedEnergy(sys, X, T, steps) {
    const sim = new F.Simulation(sys, { seed: 5, x: X, gamma: 2 });
    sim.run(steps || 1500, T);
    let e = 0;
    for (let k = 0; k < 10; k++) { sim.run(50, T); e += sim.energy; }
    return e / 10;
  }
  function newRun() {
    stopRunner();
    const stopAtFold = state.stopAtFold;
    const t = state.target, m = state.method, steps = totalSteps();
    const inp = methodInputs(t, m, state.ssSource);
    const run = {
      id: Date.now(), target: t, method: m, ssSource: m === 'go' ? null : state.ssSource,
      params: Object.assign({}, state.params[m]), steps, every: sampleEvery(steps), tHi: state.tHi, tLo: state.tLo,
      seed: state.seed, R: state.replicas, ssWeights: inp.ssWeights, local: inp.local, pred: inp.pred, restraints: inp.restraints || null, needsEvo: !!inp.needsEvo, frags: inp.frags || null, stopAtFold, stopHit: null, hopSteps: m === 'go' || !state.hopOn ? 0 : HOP_PATIENCE, hopSeverity: state.hopSeverity, get hop() { return this.hopSteps > 0 || this.hopUsed; }, hopUsed: false,
      status: 'ready', late: [], final: null, runner: null, startedAt: 0, elapsed: 0,
    };
    run.reps = [];
    for (let r = 0; r < run.R; r++) {
      const seed = run.seed + 7919 * r;
      const x = F.randomCoil(t.seq.length, F.mulberry32(seed));
      const rep = { r, seed, trace: [], frames: [], x, step: 0, best: null, done: false, hops: [] };
      run.reps.push(rep);
      pushSample(run, rep, { step: 0, T: run.tHi, E: NaN, x: Float32Array.from(x) }, true);
    }
    run.sys = buildSys(run);
    nativeEnergies(run);
    state.run = run; state.focus = 0; state.scrub = null; state.frame = 'live'; state.replaying = false;
    prevDisplay = null;
    dirty.all(); renderControls(); renderReplicas();
    return run;
  }
  function nativeEnergies(run) {
    const X = run.target.X;
    run.nativeE = X ? relaxedEnergy(run.sys, X, 0.05) : null;
    run.nativeEThermal = X ? relaxedEnergy(run.sys, X, run.tLo, 1500) : null;
  }
  function payload(run, r) {
    const rep = run.reps[r];
    const R = run.restraints;
    return { seq: run.target.seq, model: engineModel(run.method), params: run.params, native: run.method === 'go' ? Array.from(run.target.X) : null,
      restraints: R ? { I: R.I, J: R.J, V: R.V, grid0: R.grid0, gridN: R.gridN } : null, hopSteps: run.hopSteps, hopSeverity: run.hopSeverity, frags: run.frags,
      ssWeights: run.ssWeights, local: run.local ? { theta: Array.from(run.local.theta), phi: Array.from(run.local.phi) } : null,
      seed: rep.seed, steps: run.steps, every: run.every, tHi: run.tHi, tLo: run.tLo, replica: r, watch: state.watch };
  }

  // ---------- runners ----------
  function WorkerRunner(run, onMsg) {
    this.workers = run.reps.map(rep => {
      const w = new Worker('worker.js?v8');
      w.onmessage = e => onMsg(rep.r, e.data);
      w.onerror = e => { e.preventDefault(); onMsg(rep.r, { type: 'error', message: e.message || 'worker failed' }); };
      return w;
    });
    this.run = run;
  }
  WorkerRunner.prototype.start = function () { this.workers.forEach((w, r) => w.postMessage({ type: 'start', job: payload(this.run, r) })); };
  WorkerRunner.prototype.pause = function () { this.workers.forEach(w => w.postMessage({ type: 'pause' })); };
  WorkerRunner.prototype.resume = function () { this.workers.forEach(w => w.postMessage({ type: 'resume' })); };
  WorkerRunner.prototype.update = function (job, rebuild) { this.workers.forEach(w => w.postMessage({ type: 'update', job, rebuild })); };
  WorkerRunner.prototype.stop = function () { this.workers.forEach(w => w.terminate()); };

  // Main-thread fallback when workers are unavailable: replicas take turns inside the animation loop.
  function LocalRunner(run, onMsg) {
    this.run = run; this.onMsg = onMsg; this.active = false; this.next = 0;
    this.sims = run.reps.map(rep => new F.Simulation(buildSys(run), { seed: rep.seed, dt: 0.01, gamma: 0.5 }));
    this.hoppers = this.sims.map((sim, r) => (run.hopSteps ? this.makeHopper(sim, r) : null));
  }
  LocalRunner.prototype.makeHopper = function (sim, r) {
    const run = this.run;
    return new F.Hopper(sim, { steps: run.steps, every: run.every, frags: run.frags, seed: run.reps[r].seed, patienceSteps: run.hopSteps, severity: run.hopSeverity });
  };
  LocalRunner.prototype.start = function () { this.active = true; };
  LocalRunner.prototype.pause = function () { this.active = false; };
  LocalRunner.prototype.resume = function () { this.active = true; };
  LocalRunner.prototype.stop = function () { this.active = false; };
  LocalRunner.prototype.update = function (job, rebuild) {
    const run = this.run;
    if (job.prevSteps && job.steps !== job.prevSteps) {
      this.anchors = this.anchors || [];
      this.sims.forEach((sim, r) => { this.anchors[r] = { step: sim.step, T: F.scheduleT(sim.step, job.prevSteps, run.tHi, run.tLo, this.anchors[r]) }; });
    }
    if (job.every) this.hoppers.forEach(h => { if (h) h.setSampling(job.every); });
    if (job.steps) this.hoppers.forEach(h => { if (h) h.steps = job.steps; });
    if (job.hopSeverity) this.hoppers.forEach(h => { if (h) h.severity = job.hopSeverity; });
    if ('hopSteps' in job) this.hoppers = this.sims.map((sim, r) => (!job.hopSteps ? null : this.hoppers[r] ? (this.hoppers[r].setPatience(job.hopSteps), this.hoppers[r]) : this.makeHopper(sim, r)));
    if (rebuild) { const sys = buildSys(this.run); this.sims.forEach(s => s.setSystem(sys)); }
  };
  LocalRunner.prototype.tick = function (budget) {
    if (!this.active) return;
    const run = this.run, t0 = performance.now();
    const anchors = this.anchors || (this.anchors = []);
    let guard = 0;
    while (performance.now() - t0 < budget && guard++ < 1000) {
      const live = this.sims.map((s, r) => r).filter(r => !run.reps[r].done);
      if (!live.length) { this.active = false; return; }
      const r = live[this.next++ % live.length], sim = this.sims[r], T = step => F.scheduleT(step, run.steps, run.tHi, run.tLo, anchors[r]);
      const toSample = run.every - (sim.step % run.every);
      const chunk = Math.min(state.watch ? 25 : 250, toSample, run.steps - sim.step);
      if (sim.sys.rest) F.restraintStage(sim.sys, sim.step / run.steps);
      if (chunk > 0) sim.run(chunk, T(sim.step));
      if (sim.step % run.every === 0 || sim.step >= run.steps) this.onMsg(r, { type: 'sample', step: sim.step, T: T(sim.step), E: sim.energy, x: Float32Array.from(sim.x) });
      else if (r === state.focus && performance.now() - (this.lastLive || 0) > 100) { this.lastLive = performance.now(); this.onMsg(r, { type: 'live', step: sim.step, T: T(sim.step), E: sim.energy, x: Float32Array.from(sim.x) }); }
      const ev = sim.step % run.every === 0 && this.hoppers[r] && this.hoppers[r].observe(T(sim.step));
      if (ev) { this.onMsg(r, { type: 'hop', ev }); this.onMsg(r, { type: 'sample', step: sim.step, T: T(sim.step), E: sim.energy, x: Float32Array.from(sim.x) }); }
      if (sim.step >= run.steps) this.onMsg(r, { type: 'done', step: sim.step });
      if (state.watch) break;
    }
  };

  let workersOk = typeof Worker !== 'undefined';
  function makeRunner(run) {
    const onMsg = (r, m) => handleMsg(run, r, m);
    if (workersOk) {
      try { return new WorkerRunner(run, onMsg); } catch (e) { workersOk = false; }
    }
    return new LocalRunner(run, onMsg);
  }
  function stopRunner() {
    const run = state.run;
    if (run && run.runner) { run.runner.stop(); run.runner = null; }
  }
  let evoWait = null;
  function startRun() {
    let run = state.run;
    if (state.method === 'evo' && !evoReady(state.target)) {
      const t = state.target, token = evoWait = {};
      ensureEvo(t).then(() => { if (evoWait === token && state.target === t && state.method === 'evo') { evoWait = null; newRun(); startRun(); } })
        .catch(() => { if (evoWait === token) { evoWait = null; renderControls(); } });
      renderControls();
      return;
    }
    evoWait = null;
    if (!run || run.status === 'done' || run.target !== state.target || run.method !== state.method || run.needsEvo) run = newRun();
    if (run.status === 'paused') { run.runner.resume(); run.status = 'running'; run.resumedAt = performance.now(); renderControls(); return; }
    run.runner = makeRunner(run);
    run.runner.start();
    run.status = 'running'; run.startedAt = run.resumedAt = performance.now();
    speedMark = { t: performance.now(), steps: 0 };
    renderControls();
  }
  function pauseRun() {
    const run = state.run;
    if (!run || run.status !== 'running') return;
    run.runner.pause();
    run.status = 'paused';
    run.elapsed += performance.now() - run.resumedAt;
    renderControls();
  }

  function pushSample(run, rep, m, initial) {
    const x = Float64Array.from(m.x), t = run.target;
    const s = { step: m.step, T: m.T, E: m.E, rg: F.radiusOfGyration(x) };
    if (t.X) { s.rmsd = F.superpose(x, t.X).rmsd; s.q = F.fractionNative(x, t.contacts); }
    rep.trace.push(s);
    rep.frames.push(m.x instanceof Float32Array ? m.x : Float32Array.from(m.x));
    rep.x = x; rep.step = m.step; rep.live = null;
    // closest approach to the PDB structure over the whole run (any replica, any saved frame)
    if (!initial && s.rmsd != null) {
      if (rep.minRmsd == null || s.rmsd < rep.minRmsd) rep.minRmsd = s.rmsd;
      if (!run.closest || s.rmsd < run.closest.rmsd) run.closest = { rmsd: s.rmsd, r: rep.r, k: rep.frames.length - 1, step: m.step };
    }
    if (!initial && isFinite(m.E) && (!rep.best || m.E < rep.best.E)) rep.best = { E: m.E, x, step: m.step, rmsd: s.rmsd };
    if (!initial && m.step >= 0.85 * run.steps) run.late.push({ r: rep.r, x, E: m.E, step: m.step });
    if (!initial && run.stopAtFold && !run.stopHit && t.X && run.status === 'running') checkSameFold(run, rep, x, s, m);
  }
  // "Stop at same fold": as soon as any replica's frame earns the "Same overall fold" verdict (or better) against the
  // PDB structure, stop all replicas and report that frame. It peeks at the answer, so such a run is not blind.
  function checkSameFold(run, rep, x, s, m) {
    const n = run.target.seq.length, t = run.target;
    if (s.rmsd > (n >= 25 ? 8 : 1.5 + 0.1 * n)) return;          // cheap pre-filter before the TM-score search
    const sc = F.compare(x, t.X, t.contacts, { fast: true }), v = verdict(n, sc);
    if (v.tone !== 'good' && v.tone !== 'warn') return;
    run.stopHit = { r: rep.r, x, E: m.E, step: m.step, stopped: true };
    if (run.runner) { run.runner.stop(); run.runner = null; }
    run.reps.forEach(q => { q.done = true; });
    setTimeout(() => finishRun(run), 0);
  }
  function handleMsg(run, r, m) {
    if (run !== state.run) return;
    const rep = run.reps[r];
    if (m.type === 'sample') {
      if (m.step === 0 && rep.trace.length) { rep.trace[0].E = m.E; return; }
      pushSample(run, rep, m, false);
      if (r === state.focus && state.scrub == null) dirty.viewer = true;
      dirty.charts = true;
    } else if (m.type === 'live') {
      rep.x = Float64Array.from(m.x); rep.step = Math.max(rep.step, m.step); rep.live = { step: m.step, T: m.T, E: m.E };
      if (r === state.focus && state.scrub == null) dirty.viewer = true;
    } else if (m.type === 'hop') {
      rep.hops.push(m.ev); run.hopUsed = true;
      dirty.charts = true;
    } else if (m.type === 'done') {
      rep.done = true;
      if (run.reps.every(q => q.done)) finishRun(run);
    } else if (m.type === 'error') {
      // workers are blocked here: restart this run on the main thread
      workersOk = false;
      stopRunner();
      const fresh = newRun();
      fresh.runner = makeRunner(fresh);
      fresh.runner.start();
      fresh.status = 'running'; fresh.startedAt = fresh.resumedAt = performance.now();
      renderControls();
    }
  }
  function finishRun(run) {
    run.status = 'done';
    run.elapsed += performance.now() - run.resumedAt;
    if (run.runner) { run.runner.stop(); run.runner = null; }
    const t = run.target;
    // late low-temperature frames (thinned to at most ~240 for the all-pairs clustering) plus each replica's best
    const late = run.late.filter(p => p.step >= 0.85 * run.steps);
    const pool = late.filter((_, k) => k % Math.max(1, Math.ceil(late.length / 240)) === 0);
    run.reps.forEach(rep => { if (rep.best) pool.push({ r: rep.r, x: rep.best.x, E: rep.best.E, step: rep.best.step, best: true }); });
    // with distance restraints a fold and its mirror image score alike on distances; pick between clusters by energy
    if (run.stopHit) pool.push(run.stopHit);
    // MSA + AI with network restraints: the cluster score adds 3× the backbone dihedral term, so a strained β-register shift that fits the
    // predicted distances slightly better does not beat the relaxed native-like fold (v7, chosen on the benchmark)
    const selE = run.method === 'evo' && run.restraints ? pool.map(p => { F.computeForces(run.sys, p.x, new Float64Array(p.x.length)); return p.E + DIHEDRAL_SEL * F.comp.dihedral; }) : pool.map(p => p.E);
    const c = run.stopHit ? { index: pool.length - 1, size: 1, total: pool.length, cutoff: 0 } : F.consensus(pool.map(p => p.x), selE, { byEnergy: run.method === 'evo' });
    const pick = pool[c.index];
    const lowest = pool.reduce((a, b) => (b.E < a.E ? b : a));
    const scoreOf = x => {
      if (!t.X) return null;
      const m = F.compare(x, t.X, t.contacts);
      m.ss = F.assignSS(x); m.q3 = F.q3(m.ss, t.ss || m.ss);
      return m;
    };
    run.final = {
      consensus: { x: pick.x, r: pick.r, step: pick.step, E: pick.E, cluster: c, metrics: scoreOf(pick.x), members: membersOf(pool, pick.x, c.cutoff) },
      lowest: { x: lowest.x, r: lowest.r, step: lowest.step, E: lowest.E, metrics: scoreOf(lowest.x) },
    };
    run.final.consensus.relaxedE = relaxedEnergy(run.sys, pick.x, 0.05);
    state.frame = 'consensus'; state.scrub = null;
    // if any replica passed through a near-native frame, end on it (picked with the answer, so labelled as such)
    const cl = run.closest;
    if (t.X && cl && run.reps[cl.r].frames[cl.k]) {
      const x = Float64Array.from(run.reps[cl.r].frames[cl.k]), mm = scoreOf(x);
      if (mm && verdict(t.seq.length, mm).tone === 'good') {
        run.final.nearNative = { x, r: cl.r, step: cl.step, E: run.reps[cl.r].trace[cl.k] ? run.reps[cl.r].trace[cl.k].E : NaN, metrics: mm };
        state.frame = 'nearnative';
      }
    }
    logRun(run);
    dirty.all(); renderControls(); renderReplicas();
    if (state.queue.length) setTimeout(nextQueued, 500);
  }
  function membersOf(pool, x, cut) {
    const per = {};
    pool.forEach(p => { if (F.rmsd(p.x, x) < cut) per[p.r] = (per[p.r] || 0) + 1; });
    return per;
  }
  function logRun(run) {
    if (run.example) return;
    const t = run.target, m = run.final.consensus.metrics, v = verdict(t.seq.length, m);
    state.history.unshift({
      id: t.id, name: t.name, n: t.seq.length, method: run.method,
      ss: run.method === 'go' ? '–' : run.ssSource === 'real' && t.ss ? 'PDB record' : run.method === 'evo' ? 'AI (MSA)' : 'Predicted',
      R: run.R, steps: run.steps, seed: run.seed, secs: Math.round(run.elapsed / 1000),
      rmsd: m ? +m.rmsd.toFixed(2) : null, best: run.closest ? +run.closest.rmsd.toFixed(2) : null, tm: m ? +m.tm.toFixed(2) : null, gdt: m ? +m.gdt.toFixed(2) : null, q: m ? +m.q.toFixed(2) : null,
      tone: v.tone, verdict: v.text, when: Date.now(),
    });
    saveHistory(); renderHistory(); renderProteins();
  }
  function nextQueued() {
    const t = state.queue.shift();
    if (!t) { renderControls(); return; }
    selectTarget(t);
    startRun();
  }

  // ---------- what is on screen ----------
  function shown() {
    const run = state.run;
    if (!run) return null;
    if (state.scrub != null) {
      const rep = run.reps[state.focus], k = clamp(state.scrub, 0, rep.frames.length - 1), s = rep.trace[k];
      return { x: Float64Array.from(rep.frames[k]), label: `Replica ${state.focus + 1}, step ${fmtInt(s.step)}`, step: s.step, T: s.T, E: s.E, kind: 'frame', key: `f${state.focus}:${k}` };
    }
    if (run.final && state.frame === 'consensus') {
      const c = run.final.consensus;
      return { x: c.x, label: `Consensus model`, step: c.step, E: c.E, kind: 'consensus', key: 'c' + run.id, metrics: c.metrics };
    }
    if (run.final && run.final.nearNative && state.frame === 'nearnative') {
      const c = run.final.nearNative;
      return { x: c.x, label: `Near-native frame`, step: c.step, E: c.E, kind: 'nearnative', key: 'n' + run.id, metrics: c.metrics };
    }
    if (run.final && state.frame === 'lowest') {
      const c = run.final.lowest;
      return { x: c.x, label: `Lowest-energy frame`, step: c.step, E: c.E, kind: 'lowest', key: 'l' + run.id, metrics: c.metrics };
    }
    const rep = run.reps[state.focus], s = rep.live || rep.trace[rep.trace.length - 1];
    return { x: rep.x, label: `Replica ${state.focus + 1}`, step: s.step, T: s.T, E: s.E, kind: 'live', key: `l${state.focus}:${rep.trace.length}:${s.step}` };
  }
  let scoreCache = { key: '', m: null };
  function shownScores() {
    const t = state.target, d = shown();
    if (!t.X || !d) return null;
    if (d.metrics) return d.metrics;
    if (scoreCache.key !== d.key + t.id) {
      const m = F.compare(d.x, t.X, t.contacts, { fast: true });
      m.ss = F.assignSS(d.x); m.q3 = F.q3(m.ss, t.ss || m.ss);
      scoreCache = { key: d.key + t.id, m };
    }
    return scoreCache.m;
  }

  // ---------- 3D viewer ----------
  let viewer = null, modelTopo = null, pdbDirty = true, prevDisplay = null, topoFor = null;
  function initViewer() {
    if (!window.THREE || !window.FoldViewer) return failViewer('The 3D viewer could not load (three.js is unavailable). Every chart below still works.');
    viewer = window.FoldViewer.createViewer($('stage'), {
      onHover: (h, e) => showHover(h, e),
      onView: () => { dirty.viewer = true; },
    });
    if (!viewer) return failViewer('This browser could not start WebGL, so the 3D view is off. Every chart below still works.');
    viewer.setBackground(unit(trgb.stage));
  }
  function failViewer(msg) {
    const d = document.createElement('div');
    d.className = 'webgl-fail';
    d.textContent = msg;
    $('stage').appendChild(d);
  }
  const cartoonSS = ss => ss.replace(/G/g, 'H');
  function colorsFor(which, X, ss) {
    const t = state.target, n = t.seq.length, out = new Array(n), mode = state.view.color;
    if (which === 'pdb' && state.view.layout === 'overlay' && state.view.showModel) { out.fill(unit(mix(trgb.muted, trgb.stage, 0.1))); return out; }
    if (mode === 'rainbow') for (let i = 0; i < n; i++) out[i] = rainbow(i, n);
    else if (mode === 'ss') for (let i = 0; i < n; i++) out[i] = unit(ss[i] === 'H' ? trgb.helix : ss[i] === 'E' ? trgb.strand : trgb.coil);
    else if (mode === 'dev') {
      const d = which === 'model' && t.X ? F.deviations(X, t.X) : null;
      for (let i = 0; i < n; i++) out[i] = d ? unit(mix(trgb['ramp-a'], trgb['ramp-b'], clamp(d[i] / 8, 0, 1))) : unit(trgb.muted);
    } else for (let i = 0; i < n; i++) out[i] = unit(mix(trgb['ramp-a'], trgb['ramp-b'], clamp(((F.HYDRO[t.seq[i]] || 0) + 1.01) / 3.26, 0, 1)));
    return out;
  }
  function atomsVisible() {
    const rep = state.view.rep;
    if (rep === 'sticks' || rep === 'ball' || rep === 'spacefill') return true;
    return rep === 'cartoon' && !!viewer && viewer.zoomFraction() < 1.45;
  }
  let pdbAtomsShown = null;
  function updateViewer() {
    if (!viewer || !state.run) return;
    const t = state.target, d = shown(), n = t.seq.length;
    if (topoFor !== t) {
      modelTopo = F.atomTopology(t.seq);
      viewer.setTopology('model', n, modelTopo);
      if (t.atomModel) viewer.setTopology('pdb', n, t.atomModel);
      topoFor = t; pdbDirty = true;
      const r = t.X ? Math.max(8, F.radiusOfGyration(t.X) * 1.9) : Math.max(8, 2.2 * Math.pow(n, 0.38) * 1.9);
      viewer.frame(t.X ? centroid(t.X) : [0, 0, 0], r, true);
    }
    let X = d.x;
    if (t.X) X = F.transform(X, F.superpose(X, t.X));
    else {
      X = F.center(Float64Array.from(X));
      if (prevDisplay && prevDisplay.length === X.length) X = F.transform(X, F.superpose(X, prevDisplay));
      prevDisplay = X;
    }
    const pos = F.atomPositions(X, modelTopo);
    const guides = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const b = modelTopo.start[i];
      for (let k = 0; k < 3; k++) guides[3 * i + k] = pos[3 * (b + 3) + k] - pos[3 * (b + 2) + k];
    }
    const ss = cartoonSS(F.assignSS(X));
    const rep = state.view.rep, atoms = atomsVisible();
    const style = rep === 'trace' ? 'trace' : 'cartoon', atomMode = rep === 'spacefill' ? 'spacefill' : 'stick';
    const cartoonOn = rep !== 'ball' && rep !== 'spacefill';
    viewer.update('model', { ca: X, guides, ss, colors: colorsFor('model', X, ss), style, atoms: atoms ? pos : null, atomMode });
    viewer.show('model', state.view.showModel, cartoonOn, atoms);
    if (t.X && t.atomModel) {
      if (pdbDirty || pdbAtomsShown !== atoms) {
        const pss = cartoonSS(t.ss || F.assignSS(t.X));
        viewer.update('pdb', { ca: t.X, guides: t.guides, ss: pss, colors: colorsFor('pdb', t.X, pss), style, atoms: atoms ? t.atomModel.pos : null, atomMode });
        pdbDirty = false; pdbAtomsShown = atoms;
      }
      viewer.show('pdb', state.view.showPdb, cartoonOn, atoms);
      viewer.ghost('pdb', state.view.layout === 'overlay' && state.view.showModel);
    } else viewer.show('pdb', false, false, false);
    viewer.setLayout(t.X ? state.view.layout : 'overlay');
    $('sidelabels').hidden = state.view.layout !== 'side' || !t.X;
    $('side-pdb').textContent = `PDB ${t.id}`;
    $('zoomhint').textContent = atoms ? 'Carbon in the cartoon colour · N blue · O red · S yellow' : 'Drag to rotate · scroll in to see atoms · right-drag to pan · double-click a residue to focus';
    renderLegend3d();
  }
  function centroid(X) { const n = X.length / 3, c = [0, 0, 0]; for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) c[d] += X[3 * i + d] / n; return c; }
  function renderLegend3d() {
    const el = $('legend3d'), mode = state.view.color;
    let html;
    if (mode === 'rainbow') html = `<div class="legendbar"><span>N-term</span><span class="grad" style="background:linear-gradient(90deg,${[0, 0.25, 0.5, 0.75, 1].map(f => css(hsl(240 * (1 - f), 0.72, 0.52).map(v => v * 255))).join(',')})"></span><span>C-term</span></div>`;
    else if (mode === 'ss') html = '<div class="legendbar"><span class="k"><span class="blk" style="background:var(--helix)"></span>Helix</span><span class="k"><span class="blk" style="background:var(--strand)"></span>Strand</span><span class="k"><span class="blk" style="background:var(--coil)"></span>Coil</span></div>';
    else if (mode === 'dev') html = '<div class="legendbar"><span>0 Å</span><span class="grad" style="background:linear-gradient(90deg,var(--ramp-a),var(--ramp-b))"></span><span>8 Å+ from PDB</span></div>';
    else html = '<div class="legendbar"><span>polar</span><span class="grad" style="background:linear-gradient(90deg,var(--ramp-a),var(--ramp-b))"></span><span>nonpolar</span></div>';
    if (el.innerHTML !== html) el.innerHTML = html;
  }
  function showHover(h) {
    const el = $('hover');
    if (!h) { el.hidden = true; return; }
    const t = state.target, i = h.res, m = shownScores();
    const ss = h.mol === 'pdb' ? (t.ss ? SS_NAME[t.ss[i]] : '') : SS_NAME[(m && m.ss ? m.ss : F.assignSS(shown().x))[i]];
    const dev = h.mol === 'model' && m ? ` · ${f1(m.perResidue[i])} Å from PDB` : '';
    el.innerHTML = `<b>${AA3[t.seq[i]] || t.seq[i]} ${(t.first || 1) + i}</b> · ${h.mol === 'pdb' ? 'PDB' : 'model'} · ${ss}${dev}`;
    el.style.left = h.x + 'px';
    el.style.top = h.y + 'px';
    el.hidden = false;
  }

  // ---------- chart helpers ----------
  function contentWidth(el) { const cs = getComputedStyle(el); return Math.floor(el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)); }
  function setup(canvas, height, width) {
    const dpr = window.devicePixelRatio || 1, w = width || contentWidth(canvas.parentElement);
    canvas.style.width = w + 'px'; canvas.style.height = height + 'px';
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, height);
    return { ctx, w, h: height };
  }
  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) hi = lo + 1;
    const span = hi - lo, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => span / st <= count) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return { ticks: out, step };
  }
  const tip = $('tip');
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + 'px'; tip.style.top = Math.max(8, top) + 'px';
  }
  const hideTip = () => { tip.hidden = true; };
  const tipRows = rows => '<div class="tl">' + rows.map(([k, v]) => `<span>${esc(k)}</span><span>${esc(v)}</span>`).join('') + '</div>';
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ---------- trajectory ----------
  const traj = { canvas: $('traj'), layout: null, hover: -1 };
  function trajStrips() {
    const t = state.target;
    const s = [{ key: 'T', label: 'Temperature (kT)', h: 44, fmt: f2 }, { key: 'E', label: 'Energy (ε)', h: 82, fmt: f1 }];
    if (t.X) { s.push({ key: 'rmsd', label: 'RMSD to PDB structure (Å)', h: 82, fmt: f1, lo: 0 }); s.push({ key: 'q', label: 'Native contacts formed, Q', h: 82, fmt: pct, lo: 0, hi: 1 }); }
    s.push({ key: 'rg', label: 'Radius of gyration (Å)', h: 82, fmt: f1, ref: t.rg, refLabel: t.rg ? `PDB ${t.rg.toFixed(1)}` : '' });
    return s;
  }
  function drawTraj() {
    const run = state.run;
    if (!run) return;
    const strips = trajStrips(), gapH = 26, axisH = 22, left = 46, right = 70;
    const H = strips.reduce((a, s) => a + s.h + gapH, 0) + axisH;
    const { ctx, w } = setup(traj.canvas, H);
    const N = run.steps, X = v => left + (v / N) * (w - left - right);
    const focus = run.reps[state.focus];
    let y0 = 0;
    traj.layout = { left, right, w, strips: [], X };
    for (const s of strips) {
      const top = y0 + gapH, bot = top + s.h;
      const vals = [];
      run.reps.forEach(rep => rep.trace.forEach(p => { const v = p[s.key]; if (v != null && isFinite(v)) vals.push(v); }));
      let lo = s.lo != null ? s.lo : vals.length ? Math.min(...vals) : 0, hi = s.hi != null ? s.hi : vals.length ? Math.max(...vals) : 1;
      if (s.key === 'E' && vals.length > 20) { const so = vals.slice().sort((a, b) => a - b); hi = Math.min(hi, so[Math.floor(so.length * 0.98)]); }
      if (s.ref != null) { lo = Math.min(lo, s.ref); hi = Math.max(hi, s.ref); }
      if (hi - lo < 1e-6) hi = lo + 1;
      const pad = s.hi != null ? 0 : (hi - lo) * 0.08;
      const { ticks } = niceTicks(lo - (s.lo != null ? 0 : pad), hi + pad, s.key === 'T' ? 2 : 4);
      const ylo = Math.min(ticks[0], lo), yhi = Math.max(ticks[ticks.length - 1], hi);
      const Y = v => bot - ((clamp(v, ylo, yhi) - ylo) / (yhi - ylo)) * s.h;
      traj.layout.strips.push({ s, top, bot, Y });
      ctx.fillStyle = theme.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.font = `500 12px ${theme.fontUi}`;
      ctx.fillText(s.label, left, top - 8);
      ctx.font = `11px ${theme.fontMono}`; ctx.strokeStyle = theme.grid; ctx.lineWidth = 1; ctx.fillStyle = theme.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const tv of ticks) {
        const yy = Math.round(Y(tv)) + 0.5;
        ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(w - right, yy); ctx.stroke();
        ctx.fillText(s.key === 'q' ? pct(tv) : (Math.abs(tv) >= 100 ? tv.toFixed(0) : String(+tv.toFixed(2))), left - 6, yy);
      }
      if (s.ref != null) {
        ctx.strokeStyle = theme.native; ctx.lineWidth = 1.5; const yy = Y(s.ref);
        ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(w - right, yy); ctx.stroke();
        ctx.fillStyle = theme['ink-2']; ctx.textAlign = 'left'; ctx.fillText(s.refLabel, w - right + 6, yy);
      }
      const line = (rep, color, width) => {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.beginPath(); let started = false;
        for (const p of rep.trace) { const v = p[s.key]; if (v == null || !isFinite(v)) continue; const xx = X(p.step), yy = Y(v); if (!started) { ctx.moveTo(xx, yy); started = true; } else ctx.lineTo(xx, yy); }
        ctx.stroke();
      };
      if (s.key !== 'T') run.reps.forEach(rep => { if (rep !== focus) line(rep, theme.coil, 1.2); });
      line(focus, s.key === 'T' ? theme.muted : theme.model, 2);
      const last = focus.trace[focus.trace.length - 1];
      if (last && last[s.key] != null && isFinite(last[s.key])) {
        ctx.fillStyle = s.key === 'T' ? theme.muted : theme.model;
        ctx.beginPath(); ctx.arc(X(last.step), Y(last[s.key]), 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = theme.panel; ctx.lineWidth = 2; ctx.stroke();
        if (s.ref == null) { ctx.fillStyle = theme['ink-2']; ctx.textAlign = 'left'; ctx.fillText(s.fmt(last[s.key]), Math.min(X(last.step) + 8, w - right + 6), Y(last[s.key])); }
      }
      y0 = bot;
    }
    ctx.fillStyle = theme.muted; ctx.font = `11px ${theme.fontMono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const tv of niceTicks(0, N, 6).ticks) ctx.fillText(fmtSteps(tv), X(tv), y0 + 6);
    ctx.textAlign = 'left'; ctx.fillText('steps', w - right + 20, y0 + 6);
    const d = shown();
    if (d && d.step != null && (d.kind !== 'live' || run.status === 'done')) {
      const xx = Math.round(X(d.step)) + 0.5;
      ctx.strokeStyle = theme['ink-2']; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xx, gapH - 4); ctx.lineTo(xx, y0); ctx.stroke();
      ctx.fillStyle = theme['ink-2']; ctx.font = `11px ${theme.fontUi}`; ctx.textBaseline = 'bottom';
      ctx.textAlign = xx > w - right - 90 ? 'right' : 'left';
      ctx.fillText(d.kind === 'consensus' ? 'consensus model' : d.kind === 'lowest' ? 'lowest energy' : d.kind === 'nearnative' ? 'near-native' : 'on screen', xx + (ctx.textAlign === 'left' ? 4 : -4), gapH - 6);
    }
    if (traj.hover >= 0 && traj.hover < focus.trace.length) {
      const p = focus.trace[traj.hover], xx = Math.round(X(p.step)) + 0.5;
      ctx.strokeStyle = theme.muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xx, gapH); ctx.lineTo(xx, y0); ctx.stroke();
      for (const L of traj.layout.strips) {
        const v = p[L.s.key];
        if (v == null || !isFinite(v)) continue;
        ctx.fillStyle = L.s.key === 'T' ? theme.muted : theme.model;
        ctx.beginPath(); ctx.arc(xx, L.Y(v), 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = theme.panel; ctx.lineWidth = 2; ctx.stroke();
      }
    }
    $('traj-focus').textContent = `Replica ${state.focus + 1} (shown)`;
  }
  traj.canvas.addEventListener('pointermove', e => {
    const L = traj.layout, run = state.run;
    if (!L || !run) return;
    const tr = run.reps[state.focus].trace;
    if (tr.length < 2) return;
    const r = traj.canvas.getBoundingClientRect(), mx = e.clientX - r.left;
    const step = ((mx - L.left) / (L.w - L.left - L.right)) * run.steps;
    let best = 0;
    for (let i = 0; i < tr.length; i++) if (Math.abs(tr[i].step - step) < Math.abs(tr[best].step - step)) best = i;
    traj.hover = best;
    const p = tr[best], rows = [['kT', f2(p.T)], ['Energy', signed(p.E) + ' ε']];
    if (p.rmsd != null) rows.push(['RMSD', f1(p.rmsd) + ' Å'], ['Q', pct(p.q)]);
    rows.push(['Rg', f1(p.rg) + ' Å']);
    showTip(`<b>Replica ${state.focus + 1} · step ${fmtInt(p.step)}</b>` + tipRows(rows) + '<div style="margin-top:4px;color:var(--muted)">Click to show this frame</div>', e.clientX, e.clientY);
    drawTraj();
  });
  traj.canvas.addEventListener('pointerleave', () => { traj.hover = -1; hideTip(); drawTraj(); });
  traj.canvas.addEventListener('click', e => {
    const L = traj.layout, run = state.run;
    if (!L || !run) return;
    const r = traj.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const strip = L.strips.find(S => my >= S.top - 6 && my <= S.bot + 6);
    const step = ((mx - L.left) / (L.w - L.left - L.right)) * run.steps;
    let pick = state.focus;
    if (strip && strip.s.key !== 'T') {
      let bd = 1e9;
      run.reps.forEach((rep, k) => {
        let bi = 0;
        rep.trace.forEach((p, i) => { if (Math.abs(p.step - step) < Math.abs(rep.trace[bi].step - step)) bi = i; });
        const v = rep.trace[bi][strip.s.key];
        if (v != null && isFinite(v)) { const dd = Math.abs(strip.Y(v) - my); if (dd < bd) { bd = dd; pick = k; } }
      });
    }
    state.focus = pick;
    const tr = run.reps[pick].trace;
    let bi = 0;
    tr.forEach((p, i) => { if (Math.abs(p.step - step) < Math.abs(tr[bi].step - step)) bi = i; });
    state.scrub = bi; state.replaying = false; setReplayIcon();
    dirty.all(); renderControls(); renderReplicas();
  });

  // ---------- residue tracks ----------
  const tracks = { canvas: $('tracks'), layout: null };
  function ssRuns(ss) { const runs = []; for (let i = 0; i < ss.length;) { let j = i; while (j < ss.length && ss[j] === ss[i]) j++; runs.push([ss[i], i, j - 1]); i = j; } return runs; }
  function drawSS(ctx, ss, x0, cw, y, h) {
    for (const [c, a, b] of ssRuns(ss)) {
      const xa = x0 + a * cw, xb = x0 + (b + 1) * cw;
      if (c === 'H' || c === 'G') {
        ctx.fillStyle = theme.helix; ctx.globalAlpha = c === 'G' ? 0.55 : 1;
        roundRect(ctx, xa + 1, y, xb - xa - 2, h, Math.min(4, h / 2)); ctx.fill(); ctx.globalAlpha = 1;
      } else if (c === 'E') {
        ctx.fillStyle = theme.strand;
        const tw = Math.min(cw * 1.2, (xb - xa) * 0.5);
        ctx.beginPath();
        ctx.moveTo(xa + 1, y + h * 0.2); ctx.lineTo(xb - tw, y + h * 0.2); ctx.lineTo(xb - tw, y - 1); ctx.lineTo(xb - 1, y + h / 2);
        ctx.lineTo(xb - tw, y + h + 1); ctx.lineTo(xb - tw, y + h * 0.8); ctx.lineTo(xa + 1, y + h * 0.8); ctx.closePath(); ctx.fill();
      } else { ctx.fillStyle = theme.coil; ctx.fillRect(xa, y + h / 2 - 1, xb - xa, 2); }
    }
  }
  function drawTracks() {
    const run = state.run;
    if (!run) return;
    const t = state.target, n = t.seq.length, gutter = 96;
    const avail = contentWidth(tracks.canvas.parentElement);
    const cw = Math.max(9, Math.min(30, (avail - gutter - 8) / n)), w = Math.max(avail, Math.ceil(gutter + n * cw + 8));
    const m = shownScores(), d = shown();
    const modelSS = m ? m.ss : F.assignSS(d.x);
    const rows = [{ label: 'Sequence', h: 16 }, { label: 'PDB record', h: 12, ss: t.ss },
      { label: run.method === 'evo' ? (run.ssSource === 'real' && run.target.ss ? 'PDB record' : 'AI (MSA)') : run.method === 'kb' ? 'Fragments' : run.method === 'seq' ? 'Chou–Fasman' : 'Predicted', h: 12, ss: run.method === 'go' ? null : run.pred, none: 'not used by this method' },
      { label: 'Model', h: 12, ss: modelSS }];
    const barH = 72, rowGap = 9;
    let H = 8;
    for (const r of rows) { r.y = H; H += r.h + rowGap; }
    const barTop = H + 6, barBot = barTop + barH;
    H = barBot + 24;
    const { ctx } = setup(tracks.canvas, H, w);
    tracks.layout = { gutter, cw, n };
    ctx.textBaseline = 'middle';
    for (const r of rows) {
      ctx.fillStyle = theme['ink-2']; ctx.font = `12px ${theme.fontUi}`; ctx.textAlign = 'left';
      ctx.fillText(r.label, 0, r.y + r.h / 2);
      if (r.label === 'Sequence') {
        ctx.font = `${cw >= 11 ? 12 : 10}px ${theme.fontMono}`; ctx.textAlign = 'center'; ctx.fillStyle = theme.ink;
        for (let i = 0; i < n; i++) ctx.fillText(t.seq[i], gutter + (i + 0.5) * cw, r.y + r.h / 2 + 1);
      } else if (r.ss) drawSS(ctx, r.ss, gutter, cw, r.y, r.h);
      else { ctx.fillStyle = theme.muted; ctx.textAlign = 'left'; ctx.fillText(r.none || 'no PDB record', gutter, r.y + r.h / 2); }
    }
    ctx.fillStyle = theme['ink-2']; ctx.font = `12px ${theme.fontUi}`; ctx.textAlign = 'left';
    ctx.fillText('Distance (Å)', 0, barTop + 8);
    if (m) {
      const dv = m.perResidue, top = niceTicks(0, Math.max(4, ...dv), 3).ticks.slice(-1)[0], ticks = niceTicks(0, top, 3).ticks;
      const Y = v => barBot - (Math.min(v, top) / top) * barH;
      ctx.strokeStyle = theme.grid; ctx.lineWidth = 1; ctx.font = `11px ${theme.fontMono}`; ctx.fillStyle = theme.muted; ctx.textAlign = 'right';
      for (const tv of ticks) { const yy = Math.round(Y(tv)) + 0.5; ctx.beginPath(); ctx.moveTo(gutter, yy); ctx.lineTo(gutter + n * cw, yy); ctx.stroke(); ctx.fillText(String(tv), gutter - 6, yy); }
      const bw = Math.min(24, Math.max(2, cw - 2));
      ctx.fillStyle = theme.model;
      for (let i = 0; i < n; i++) {
        const x = gutter + i * cw + (cw - bw) / 2, y = Y(dv[i]), hh = barBot - y;
        if (hh < 0.5) continue;
        const rr = Math.min(3, bw / 2, hh);
        ctx.beginPath(); ctx.moveTo(x, barBot); ctx.lineTo(x, y + rr); ctx.quadraticCurveTo(x, y, x + rr, y);
        ctx.lineTo(x + bw - rr, y); ctx.quadraticCurveTo(x + bw, y, x + bw, y + rr); ctx.lineTo(x + bw, barBot); ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = theme.coil; ctx.beginPath(); ctx.moveTo(gutter, barBot + 0.5); ctx.lineTo(gutter + n * cw, barBot + 0.5); ctx.stroke();
    } else { ctx.fillStyle = theme.muted; ctx.textAlign = 'left'; ctx.fillText('Load a structure with coordinates to compare distances.', gutter, barTop + barH / 2); }
    ctx.fillStyle = theme.muted; ctx.font = `11px ${theme.fontMono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const first = t.first || 1, every = cw >= 14 ? 5 : 10;
    for (let i = 0; i < n; i++) if ((first + i) % every === 0) ctx.fillText(String(first + i), gutter + (i + 0.5) * cw, barBot + 6);
  }
  tracks.canvas.addEventListener('pointermove', e => {
    const L = tracks.layout, run = state.run;
    if (!L || !run) return;
    const r = tracks.canvas.getBoundingClientRect(), i = Math.floor((e.clientX - r.left - L.gutter) / L.cw);
    if (i < 0 || i >= L.n) { hideTip(); return; }
    const t = state.target, m = shownScores(), rows = [];
    if (t.ss) rows.push(['PDB record', SS_NAME[t.ss[i]]]);
    if (run.pred && run.method !== 'go') rows.push(['Predicted', SS_NAME[run.pred[i]]]);
    rows.push(['Model', SS_NAME[(m ? m.ss : F.assignSS(shown().x))[i]]]);
    if (m) rows.push(['Distance from PDB', f1(m.perResidue[i]) + ' Å']);
    rows.push(['Hydrophobicity π', (F.HYDRO[t.seq[i]] || 0).toFixed(2)]);
    showTip(`<b>${AA3[t.seq[i]] || t.seq[i]} ${(t.first || 1) + i}</b>` + tipRows(rows), e.clientX, e.clientY);
  });
  tracks.canvas.addEventListener('pointerleave', hideTip);

  // ---------- contact map ----------
  const cmap = { canvas: $('cmap'), layout: null };
  function drawCmap() {
    if (!state.run) return;
    const t = state.target, n = t.seq.length;
    const W = Math.min(contentWidth(cmap.canvas.parentElement), 560), pad = 30, size = W - pad - 6;
    const { ctx } = setup(cmap.canvas, size + pad + 6, W);
    const cs = size / n, x0 = pad, y0 = 6;
    cmap.layout = { x0, y0, cs, n };
    const lo = trgb['heat-lo'], hi = trgb['heat-hi'], model = shown().x, N = t.X;
    const close = d => clamp((12 - d) / 8, 0, 1);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let d;
      if (j > i) d = F.dist(model, i, j); else if (j < i) { if (!N) continue; d = F.dist(N, i, j); } else continue;
      ctx.fillStyle = css(mix(lo, hi, close(d)));
      ctx.fillRect(x0 + j * cs, y0 + i * cs, Math.ceil(cs), Math.ceil(cs));
    }
    ctx.strokeStyle = theme.line; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, size, size);
    ctx.strokeStyle = theme.muted; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + size, y0 + size); ctx.stroke();
    ctx.fillStyle = theme.muted; ctx.font = `11px ${theme.fontMono}`;
    const first = t.first || 1, every = n > 40 ? 20 : 10;
    for (let i = 0; i < n; i++) if ((first + i) % every === 0) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(first + i), x0 - 5, y0 + (i + 0.5) * cs); }
    const label = (txt, x, y, align) => {
      ctx.font = `500 12px ${theme.fontUi}`;
      const tw = ctx.measureText(txt).width, bx = align === 'right' ? x - tw - 10 : x;
      ctx.fillStyle = theme.panel; roundRect(ctx, bx, y - 11, tw + 10, 22, 6); ctx.fill();
      ctx.fillStyle = theme.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(txt, bx + 5, y);
    };
    label('Model', x0 + size - 8, y0 + 18, 'right');
    if (N) label('PDB', x0 + 8, y0 + size - 16, 'left');
  }
  cmap.canvas.addEventListener('pointermove', e => {
    const L = cmap.layout;
    if (!L) return;
    const r = cmap.canvas.getBoundingClientRect(), j = Math.floor((e.clientX - r.left - L.x0) / L.cs), i = Math.floor((e.clientY - r.top - L.y0) / L.cs);
    if (i < 0 || j < 0 || i >= L.n || j >= L.n || i === j) { hideTip(); return; }
    const a = Math.min(i, j), b = Math.max(i, j), t = state.target, first = t.first || 1;
    const dm = F.dist(shown().x, a, b), dn = t.X ? F.dist(t.X, a, b) : null, rows = [['Model', f1(dm) + ' Å']];
    if (dn != null) { rows.push(['PDB', f1(dn) + ' Å']); if (b - a >= 4) rows.push(['Contact (< 8 Å)', dm < 8 && dn < 8 ? 'in both' : dn < 8 ? 'PDB only' : dm < 8 ? 'model only' : 'neither']); }
    showTip(`<b>${AA3[t.seq[a]]} ${first + a} – ${AA3[t.seq[b]]} ${first + b}</b>` + tipRows(rows), e.clientX, e.clientY);
  });
  cmap.canvas.addEventListener('pointerleave', hideTip);

  // ---------- energy landscape ----------
  const funnel = { canvas: $('funnel'), layout: null, hover: -1 };
  function drawFunnel() {
    const run = state.run;
    if (!run) return;
    const t = state.target, W = contentWidth(funnel.canvas.parentElement), H = Math.round(clamp(W * 0.5, 280, 440));
    const { ctx, w, h } = setup(funnel.canvas, H, W);
    const L = 52, Rr = 16, T = 12, B = 36;
    if (!t.X) { ctx.fillStyle = theme.muted; ctx.font = `13px ${theme.fontUi}`; ctx.textAlign = 'center'; ctx.fillText('Needs a PDB structure to measure distance from it.', w / 2, h / 2); funnel.layout = null; return; }
    const pts = [];
    run.reps.forEach(rep => rep.trace.forEach(p => { if (p.rmsd != null && isFinite(p.E)) pts.push({ r: rep.r, step: p.step, E: p.E, rmsd: p.rmsd, T: p.T, q: p.q }); }));
    const es = pts.map(p => p.E);
    if (run.nativeEThermal != null) es.push(run.nativeEThermal);
    if (!es.length) es.push(0);
    let elo = Math.min(...es), ehi = Math.max(...es);
    const so = pts.map(p => p.E).sort((a, b) => a - b);
    if (so.length > 20) ehi = Math.min(ehi, so[Math.floor(so.length * 0.95)]);
    if (run.nativeEThermal != null) ehi = Math.max(ehi, run.nativeEThermal);
    if (!(ehi > elo)) ehi = elo + 1;
    const rmax = Math.max(4, ...pts.map(p => p.rmsd));
    const xt = niceTicks(0, rmax, 6), yt = niceTicks(elo, ehi + (ehi - elo) * 0.05, 4);
    const xhi = xt.ticks[xt.ticks.length - 1], ylo = Math.min(yt.ticks[0], elo), yhi = Math.max(yt.ticks[yt.ticks.length - 1], ehi);
    const X = v => L + (v / xhi) * (w - L - Rr), Y = v => T + (1 - (v - ylo) / (yhi - ylo)) * (h - T - B);
    funnel.layout = { X, Y, pts, yhi };
    ctx.strokeStyle = theme.grid; ctx.lineWidth = 1; ctx.font = `11px ${theme.fontMono}`; ctx.fillStyle = theme.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of yt.ticks) { const yy = Math.round(Y(v)) + 0.5; ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(w - Rr, yy); ctx.stroke(); ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : String(+v.toFixed(1)), L - 6, yy); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const v of xt.ticks) ctx.fillText(String(v), X(v), h - B + 6);
    ctx.fillStyle = theme['ink-2']; ctx.font = `12px ${theme.fontUi}`;
    ctx.fillText('RMSD to PDB structure (Å)', (L + w - Rr) / 2, h - 17);
    ctx.save(); ctx.translate(13, (T + h - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText('Energy (ε)', 0, 0); ctx.restore();
    const a = trgb['ramp-a'], b = trgb['ramp-b'];
    for (const p of pts) {
      if (p.E > yhi) continue;
      ctx.fillStyle = css(mix(a, b, p.step / run.steps));
      ctx.beginPath(); ctx.arc(X(p.rmsd), Y(p.E), 3.2, 0, Math.PI * 2); ctx.fill();
    }
    if (run.nativeEThermal != null) {
      const xx = X(0), yy = Y(run.nativeEThermal);
      ctx.fillStyle = theme.native; ctx.strokeStyle = theme.panel; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(xx, yy - 7); ctx.lineTo(xx + 7, yy); ctx.lineTo(xx, yy + 7); ctx.lineTo(xx - 7, yy); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = theme['ink-2']; ctx.font = `11px ${theme.fontUi}`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(`PDB structure at kT ${run.tLo.toFixed(2)}`, xx + 11, yy);
    }
    if (run.final && run.final.consensus.metrics) {
      const c = run.final.consensus, xx = X(c.metrics.rmsd), yy = Y(Math.min(c.E, yhi));
      ctx.strokeStyle = theme.ink; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(xx, yy, 7, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = theme['ink-2']; ctx.font = `11px ${theme.fontUi}`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText('consensus model', xx + 11, yy);
    }
    if (funnel.hover >= 0 && funnel.hover < pts.length) {
      const p = pts[funnel.hover];
      ctx.strokeStyle = theme.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(p.rmsd), Y(p.E), 5.5, 0, Math.PI * 2); ctx.stroke();
    }
  }
  funnel.canvas.addEventListener('pointermove', e => {
    const L = funnel.layout;
    if (!L || !L.pts.length) return;
    const r = funnel.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 18 * 18;
    L.pts.forEach((p, k) => { if (p.E > L.yhi) return; const d = (L.X(p.rmsd) - mx) ** 2 + (L.Y(p.E) - my) ** 2; if (d < bd) { bd = d; best = k; } });
    funnel.hover = best;
    if (best < 0) hideTip();
    else { const p = L.pts[best]; showTip(`<b>Replica ${p.r + 1} · step ${fmtInt(p.step)}</b>` + tipRows([['Energy', signed(p.E) + ' ε'], ['RMSD', f1(p.rmsd) + ' Å'], ['kT', f2(p.T)], ['Q', pct(p.q)]]) + '<div style="margin-top:4px;color:var(--muted)">Click to show this frame</div>', e.clientX, e.clientY); }
    drawFunnel();
  });
  funnel.canvas.addEventListener('pointerleave', () => { funnel.hover = -1; hideTip(); drawFunnel(); });
  funnel.canvas.addEventListener('click', () => {
    const L = funnel.layout;
    if (!L || funnel.hover < 0) return;
    const p = L.pts[funnel.hover], rep = state.run.reps[p.r];
    state.focus = p.r; state.scrub = rep.trace.findIndex(q => q.step === p.step); state.replaying = false; setReplayIcon();
    dirty.all(); renderControls(); renderReplicas();
  });

  // ---------- scorecard ----------
  function meter(v, mark) { return `<div class="meter"><i style="width:${clamp(v, 0, 1) * 100}%"></i>${mark != null ? `<u style="left:calc(${mark * 100}% - 1px)"></u>` : ''}</div>`; }
  function renderScore() {
    const el = $('score'), run = state.run;
    if (!run) return;
    const t = state.target, n = t.seq.length, d = shown();
    const choices = [];
    if (run.final && run.final.nearNative) choices.push(['nearnative', 'Near-native frame (found)']);
    if (run.final) choices.push(['consensus', run.stopHit ? 'Same-fold frame (stopped)' : 'Consensus model'], ['lowest', 'Lowest-energy frame']);
    choices.push(['live', run.final ? `Replica ${state.focus + 1}, final frame` : `Replica ${state.focus + 1}, live`]);
    const current = state.scrub != null ? 'frame' : state.frame;
    const sel = `<select id="frame-pick" aria-label="Structure to show" style="border:1px solid var(--line-strong);border-radius:8px;padding:4px 6px;background:var(--panel);font-size:12px">
      ${choices.map(([v, l]) => `<option value="${v}"${v === current ? ' selected' : ''}>${l}</option>`).join('')}${current === 'frame' ? `<option value="frame" selected>${esc(d.label)}</option>` : ''}</select>`;
    if (!t.X) {
      el.innerHTML = `<div class="score-head"><span class="eyebrow">Model</span>${sel}</div>
        <p class="fine">This sequence has no experimental structure here, so there is nothing to score against. Upload a PDB file to compare.</p>
        <dl class="metrics"><div class="metric"><dt>Radius of gyration</dt><dd>${f1(F.radiusOfGyration(d.x))} Å</dd></div>
        <div class="metric"><dt>Secondary structure</dt><dd>${esc(F.assignSS(d.x))}</dd></div></dl>`;
      bindFramePick(); return;
    }
    const m = shownScores(), v = verdict(n, m);
    const c = run.final && run.final.consensus;
    let energyRow = '';
    if (run.final && d.kind === 'consensus' && run.nativeE != null) {
      const worse = run.nativeE > c.relaxedE;
      energyRow = `<div class="metric"><dt>Energy check</dt><dd>model ${signed(c.relaxedE)} · PDB ${signed(run.nativeE)} ε</dd><span class="why">${run.method === 'go'
        ? 'The Gō energy is lowest at the PDB structure by construction.'
        : worse ? 'The PDB structure scores worse than the model, so the force field itself prefers this fold. More sampling would not fix it.'
          : 'The real fold scores lower, so more steps or replicas could still find it.'}</span></div>`;
    }
    const clusterRow = c && d.kind === 'consensus' && run.stopHit ? `<div class="metric"><dt>Stopped early</dt><dd>replica ${run.stopHit.r + 1}, ${fmtSteps(run.stopHit.step)}</dd><span class="why">“Stop at same fold” ended the run when this frame reached the same overall fold as the PDB structure. That check uses the answer, so this result is not blind.</span></div>` : c && d.kind === 'consensus' ? `<div class="metric"><dt>Consensus cluster</dt><dd>${c.cluster.size} of ${c.cluster.total} frames</dd><span class="why">Late low-temperature frames from all ${run.R} replicas within ${c.cluster.cutoff.toFixed(1)} Å of this one.</span></div>` : '';
    const note = run.method === 'go' ? '<p class="fine">The Gō force field is built from the PDB contacts, so a close match is expected. Switch to Knowledge-based for a blind prediction.</p>'
      : run.ssSource === 'real' && t.ss ? '<p class="fine">The PDB secondary-structure record guided this run. Switch the source to Predicted for a fully blind prediction.</p>' : '';
    el.innerHTML = `<div class="score-head"><span class="eyebrow">Prediction vs PDB ${esc(t.id)}</span>${sel}</div>
      <div class="bigrow"><div class="big"><b>${m.rmsd.toFixed(1)}<small>Å</small></b><span>Cα RMSD after superposition</span></div>
        <div class="verdict" data-tone="${v.tone}">${ICON[v.tone]}<span>${v.text}</span></div></div>
      <div class="tiles">
        <div class="tile"><span>TM-score</span><b>${f2(m.tm)}</b>${meter(m.tm, 0.5)}</div>
        <div class="tile"><span>GDT-TS</span><b>${f2(m.gdt)}</b>${meter(m.gdt)}</div>
        <div class="tile"><span>Native contacts, Q</span><b>${pct(m.q)}</b>${meter(m.q)}</div>
        <div class="tile"><span>Sec. structure match</span><b>${pct(m.q3)}</b>${meter(m.q3)}</div>
      </div>
      <dl class="metrics">
        <div class="metric"><dt>Contact precision</dt><dd>${pct(m.contacts.precision)}</dd><span class="why">${m.contacts.shared} of ${m.contacts.predicted} model contacts exist in the PDB structure.</span></div>
        <div class="metric"><dt>Radius of gyration</dt><dd>${f1(m.rg)} / ${f1(t.rg)} Å</dd><span class="why">Model / PDB structure.</span></div>
        <div class="metric"><dt>Mirror-image RMSD</dt><dd>${f1(m.mirrorRmsd)} Å</dd><span class="why">${m.mirrorRmsd < m.rmsd ? 'Lower than the RMSD: the model is closer to the mirror image of the real fold.' : 'Higher than the RMSD, so the handedness is right.'}</span></div>
        ${d.kind === 'nearnative' ? `<div class="metric"><dt>Near-native frame</dt><dd>replica ${run.final.nearNative.r + 1}, ${fmtSteps(run.final.nearNative.step)}</dd><span class="why">The run passed through this near-native structure, so it is shown at the end. It was picked by comparing with the PDB structure; ${run.method === 'go' ? 'the' : 'the blind prediction is the'} consensus model (${f1(run.final.consensus.metrics.rmsd)} Å), in the menu above.</span></div>` : ''}
        ${run.closest ? `<div class="metric"><dt>Lowest RMSD reached</dt><dd>${f1(run.closest.rmsd)} Å <button class="btn small" id="closest-view" type="button">View</button></dd><span class="why">The closest any replica came to the PDB structure: replica ${run.closest.r + 1} at step ${fmtSteps(run.closest.step)}. Picking it needs the answer, so it is a best case, not the prediction.</span></div>` : ''}
        ${clusterRow}${energyRow}
      </dl>${note}<p class="fine">TM-score above 0.5 (the tick) usually means the same fold${n < 22 ? '; it is harsh below about 20 residues' : ''}.</p>`;
    bindFramePick();
    const cv = $('closest-view');
    if (cv) cv.addEventListener('click', () => {
      const cl = state.run && state.run.closest;
      if (!cl) return;
      state.focus = cl.r; state.scrub = cl.k; state.frame = 'live'; state.replaying = false; setReplayIcon();
      dirty.all(); renderControls(); renderReplicas();
    });
  }
  function bindFramePick() {
    const s = $('frame-pick');
    if (!s) return;
    s.addEventListener('change', () => {
      if (s.value === 'frame') return;
      state.frame = s.value; state.scrub = null; state.replaying = false; setReplayIcon();
      dirty.all(); renderControls();
    });
  }

  // ---------- replicas table ----------
  function renderReplicas() {
    const run = state.run, el = $('reptable');
    if (!run) return;
    const t = state.target, c = run.final && run.final.consensus;
    const head = `<thead><tr><th>Replica</th><th class="num">Seed</th><th class="num">Step</th><th class="num">Lowest E</th>${t.X ? '<th class="num">RMSD of lowest-E frame</th><th class="num">Lowest RMSD</th><th class="num">Current RMSD</th>' : ''}<th class="num">In consensus cluster</th>${run.hop ? '<th class="num">MC restarts kept</th>' : ''}<th></th></tr></thead>`;
    const body = run.reps.map(rep => {
      const last = rep.trace[rep.trace.length - 1];
      return `<tr class="${rep.r === state.focus ? 'sel' : ''}"><td><b>${rep.r + 1}</b>${rep.done ? '' : run.status === 'running' ? ' · running' : ''}</td><td class="num">${rep.seed}</td><td class="num">${fmtSteps(rep.step)}</td>
        <td class="num">${rep.best ? signed(rep.best.E) : '–'}</td>${t.X ? `<td class="num">${rep.best ? f1(rep.best.rmsd) : '–'}</td><td class="num">${f1(rep.minRmsd)}</td><td class="num">${f1(last.rmsd)}</td>` : ''}
        <td class="num">${c ? (c.members[rep.r] || 0) + ' frames' : '–'}</td>${run.hop ? `<td class="num" title="${esc(rep.hops.map(h => `${h.kind} at ${fmtSteps(h.from)}: ${h.kept ? 'kept' : 'undone'} (ΔE ${(h.E1 - h.E0).toFixed(1)})`).join('\n'))}">${rep.hops.filter(h => h.kept).length} of ${rep.hops.length}</td>` : ''}<td><button class="btn" data-rep="${rep.r}" type="button">Show</button></td></tr>`;
    }).join('');
    el.innerHTML = head + '<tbody>' + body + '</tbody>';
    el.querySelectorAll('[data-rep]').forEach(b => b.addEventListener('click', () => {
      state.focus = +b.dataset.rep; state.frame = 'live'; state.scrub = null;
      dirty.all(); renderControls(); renderReplicas();
    }));
  }

  // ---------- history ----------
  function renderHistory() {
    const rows = state.history, MN = { evo: 'MSA + AI', kb: 'Knowledge-based', seq: 'Simple physics', go: 'Gō' };
    const head = '<thead><tr><th>Protein</th><th>Method</th><th>Sec. structure</th><th class="num">Replicas</th><th class="num">Steps</th><th class="num">Time</th><th class="num">RMSD Å</th><th class="num" title="Closest approach to the PDB structure during the run">Lowest Å</th><th class="num">TM</th><th class="num">GDT</th><th class="num">Q</th><th>Result</th></tr></thead>';
    const body = rows.length ? rows.map(r => `<tr><td><b>${esc(r.id)}</b> ${esc(r.name)}</td><td>${MN[r.method] || r.method}</td><td>${esc(r.ss)}</td><td class="num">${r.R || 1}</td>
      <td class="num">${fmtSteps(r.steps)}</td><td class="num">${r.secs != null ? r.secs + ' s' : '–'}</td><td class="num">${r.rmsd == null ? '–' : r.rmsd.toFixed(1)}</td><td class="num">${r.best == null ? '–' : r.best.toFixed(1)}</td><td class="num">${r.tm == null ? '–' : r.tm.toFixed(2)}</td>
      <td class="num">${r.gdt == null ? '–' : r.gdt.toFixed(2)}</td><td class="num">${r.q == null ? '–' : pct(r.q)}</td><td><span class="tag" data-tone="${r.tone}">${ICON[r.tone] || ''}${esc(r.verdict)}</span></td></tr>`).join('')
      : '<tr class="empty"><td colspan="12">Finished runs appear here, newest first. Compare methods, settings and proteins; “Run all six” fills it in one go.</td></tr>';
    $('history').innerHTML = head + '<tbody>' + body + '</tbody>';
  }

  // ---------- benchmark (measured offline on this build; see the Method tab) ----------
  const BENCH = window.FOLD_BENCH || null;
  function benchTable() {
    if (!BENCH) return '';
    const cols = BENCH.columns, cell = v => (v == null ? '–' : v.toFixed(1));
    const head = '<thead><tr><th>Protein</th>' + cols.map(c => `<th class="num">${esc(c.label)}</th>`).join('') + '</tr></thead>';
    const body = BUILTIN.map(p => '<tr><td><b>' + esc(p.id) + '</b> ' + esc(p.name) + '</td>' + cols.map(c => {
      const v = BENCH.rmsd[c.key] && BENCH.rmsd[c.key][p.id];
      return `<td class="num">${cell(v)}</td>`;
    }).join('') + '</tr>').join('');
    return `<div class="table-wrap"><table>${head}<tbody>${body}</tbody></table></div><p class="fine">${esc(BENCH.note || '')}</p>`;
  }

  // ---------- evolution: MSA → coevolution → neural network → restraints ----------
  // trRosetta-form restraints and the network's own secondary structure scored best on the benchmark (Method tab)
  const EVO_MINP = 0.3, EVO_REF = 'dfire', EVO_SS_MIX = null;
  // below this many effective sequences the network has too little to go on: fold with the v2 inputs instead
  const EVO_NEFF_MIN = 5;
  let evoWorker = null, evoSeq = 0;
  const evoJobs = new Map();
  const evoReady = t => !!(t.evo && t.evo.status === 'ready');
  function builtinRows(t) {
    const m = MSA[t.id], n = t.seq.length;
    if (!m || m.rows.substr(0, n) !== t.seq) return null;
    const rows = [];
    for (let k = 1; k < m.n; k++) rows.push(m.rows.substr(k * n, n));
    return { rows, hits: m.hits };
  }
  function evoInput(t) {
    if (t.msaUpload) return { text: t.msaUpload.text, source: 'uploaded', label: t.msaUpload.name };
    const b = builtinRows(t);
    if (b) return { rows: b.rows, source: 'built-in', hits: b.hits, label: 'ColabFold MMseqs2 (UniRef30 + environmental)' };
    return { rows: [], source: 'none', label: 'none: sequence only' };
  }
  function runEvoHere(m) {
    const a = m.text != null ? Evo.parseAlignment(m.text, m.query) : { rows: m.rows, format: m.format, dropped: 0 };
    const msa = Evo.process(m.query, a.rows), t0 = performance.now(), feat = Evo.features(msa), t1 = performance.now();
    if (!runEvoHere.model) runEvoHere.model = Evo.loadModel(EVO_MODEL);
    const pred = Evo.predict(runEvoHere.model, feat), t2 = performance.now();
    return { format: a.format, dropped: a.dropped, input: a.rows.length, N: msa.N, L: msa.L, X: msa.X, neff: feat.neff, dca: Float32Array.from(feat.dca), mi: Float32Array.from(feat.mi),
      conservation: feat.conservation, ss: pred.ss, dist: pred.dist, ms: { filter: 0, dca: t1 - t0, net: t2 - t1 } };
  }
  function getEvoWorker() {
    if (evoWorker === false || typeof Worker === 'undefined') return null;
    if (evoWorker) return evoWorker;
    try {
      evoWorker = new Worker('evo-worker.js?v8d');
      evoWorker.onmessage = e => {
        const m = e.data, job = evoJobs.get(m.id);
        if (!job) return;
        if (m.type === 'progress') { job.ev.stage = m.stage; job.ev.frac = m.frac; dirty.evo = true; }
        else if (m.type === 'done') { evoJobs.delete(m.id); job.done(m.result); }
        else if (m.type === 'error') { evoJobs.delete(m.id); job.fail(m.message); }
      };
      evoWorker.onerror = e => {
        // workers blocked or failed to load: finish the pending jobs on the main thread
        e.preventDefault();
        try { evoWorker.terminate(); } catch (err) { /* already gone */ }
        evoWorker = false;
        const jobs = [...evoJobs.values()];
        evoJobs.clear();
        jobs.forEach(job => setTimeout(() => { try { job.done(runEvoHere(job.msg)); } catch (err) { job.fail(err.message); } }, 30));
      };
      return evoWorker;
    } catch (e) { evoWorker = false; return null; }
  }
  function ensureEvo(t) {
    if (evoReady(t)) return Promise.resolve(t.evo);
    if (t.evo && t.evo.status === 'running') return t.evo.promise;
    if (!EVO_MODEL || !window.Evo) return Promise.reject(new Error('The MSA network did not load.'));
    const inp = evoInput(t), id = ++evoSeq;
    const ev = t.evo = { id, status: 'running', stage: 'filter', frac: 0, source: inp.source, label: inp.label, hits: inp.hits };
    ev.promise = new Promise((resolve, reject) => {
      const done = res => { if (t.evo !== ev) return; finishEvo(t, ev, res); resolve(ev); };
      const fail = msg => { if (t.evo !== ev) return; ev.status = 'error'; ev.error = msg; dirty.evo = true; renderControls(); reject(new Error(msg)); };
      const msg = { type: 'run', id, query: t.seq, rows: inp.rows, text: inp.text, format: inp.source === 'built-in' ? 'A3M' : 'none' };
      const w = getEvoWorker();
      if (w) { evoJobs.set(id, { ev, done, fail, msg }); w.postMessage(msg); }
      else setTimeout(() => { try { done(runEvoHere(msg)); } catch (err) { fail(err.message); } }, 30);
    });
    ev.promise.catch(() => {});
    dirty.evo = true;
    return ev.promise;
  }
  function finishEvo(t, ev, res) {
    Object.assign(ev, res);
    const L = res.L, pred = { L, ss: res.ss, dist: res.dist };
    ev.pred = pred;
    ev.labelsNet = Evo.ssLabels(pred);
    ev.labels = EVO_SS_MIX != null ? Evo.ssLabels(pred, localModel(t, 'pred').probs, EVO_SS_MIX) : ev.labelsNet;
    ev.cmap = Evo.contactMap(pred, 8);
    ev.dexp = Evo.expectedDistance(pred);
    ev.rest = Evo.restraints(pred, { minP: EVO_MINP, ref: EVO_REF });
    const id = new Float32Array(res.N);
    for (let s = 0; s < res.N; s++) {
      let same = 0, cov = 0;
      for (let i = 0; i < L; i++) { const c = res.X[s * L + i]; if (c !== 20) { cov++; if (c === res.X[i]) same++; } }
      id[s] = cov ? same / cov : 0;
    }
    ev.identity = id;
    if (t.X) {
      const k = Math.max(1, L >> 1);
      ev.prec = Evo.precision(ev.cmap, L, t.X, 12, 1e9, k);
      ev.precDCA = Evo.precision(ev.dca, L, t.X, 12, 1e9, k);
    }
    if (t.ss) { ev.q3 = F.q3(ev.labels, t.ss); ev.q3net = F.q3(ev.labelsNet, t.ss); ev.q3frag = F.q3(localModel(t, 'pred').labels, t.ss); }
    delete t.lmCache.ai;
    ev.shallow = ev.neff < EVO_NEFF_MIN;
    ev.status = 'ready';
    dirty.evo = true;
    renderControls();
  }
  const EVO_STAGE = { align: 'Aligning homologs', filter: 'Filtering alignment', dca: 'Coevolution (DCA)', net: 'Neural network' };
  function evoStatusText(t) {
    const ev = t.evo;
    if (!ev || ev.status !== 'running') return 'Preparing MSA';
    return `${EVO_STAGE[ev.stage] || 'Working'}${ev.stage === 'dca' || ev.stage === 'net' ? ' · ' + Math.round(100 * (ev.frac || 0)) + '%' : ''}`;
  }

  // residue colours for the alignment (Clustal-like classes)
  const AA_CLASS = {};
  [['AILMFWV', [64, 120, 220]], ['KR', [224, 76, 70]], ['DE', [196, 84, 205]], ['NQST', [46, 170, 98]], ['C', [236, 128, 158]], ['G', [236, 150, 48]], ['P', [204, 182, 40]], ['HY', [36, 170, 186]]]
    .forEach(([aa, rgb]) => { for (const c of aa) AA_CLASS[c] = rgb; });
  const evoView = { map: 'ai', msaLayout: null, mapLayout: null };
  function renderEvo() {
    const t = state.target, pipe = $('evo-pipe');
    if (!pipe) return;
    if (!t.evo && state.tab === 'evo') ensureEvo(t).catch(() => {});
    const ev = t.evo;
    const L = t.seq.length, run = ev && ev.status === 'running', ready = evoReady(t);
    const stageIdx = !ev ? -1 : ready ? 4 : ({ align: 0, filter: 0, dca: 1, net: 2 }[ev.stage] || 0);
    const st = k => (stageIdx > k ? 'done' : stageIdx === k && run ? 'run' : 'wait');
    const bar = k => (stageIdx === k && run && (ev.stage === 'dca' || ev.stage === 'net') ? `<i class="bar" style="width:${(100 * (ev.frac || 0)).toFixed(0)}%"></i>` : '');
    const inp = ev || evoInput(t);
    const msaB = ready ? `${fmtInt(ev.N - 1)} <small>homolog${ev.N === 2 ? '' : 's'}</small>` : inp.source === 'none' ? 'none' : '…';
    const msaD = ready ? `${inp.source === 'built-in' ? fmtInt(ev.hits) + ' found, ' : fmtInt(ev.input) + ' read, '}filtered to ${fmtInt(ev.N - 1)}. Neff ${f1(ev.neff)} (${f2(ev.neff / L)} per residue)` : esc(inp.label || '');
    const steps = [
      ['1 · Alignment', msaB, msaD],
      ['2 · Coevolution', ready ? `${fmtInt(20 * L)}² <small>matrix</small>` : 'DCA', ready ? `Mean-field DCA inverts the pair covariance; APC removes background. ${ev.ms.stored ? 'Precomputed for this built-in protein.' : Math.round(ev.ms.dca) + ' ms'}` : 'Residues that mutate together are likely in contact'],
      ['3 · Neural network', ready ? `${fmtInt(EVO_MODEL.params / 1000)}k <small>parameters</small>` : 'ResNet', ready ? `1D + 2D dilated residual CNN → 18-bin Cα distance histogram for ${fmtInt(L * (L - 1) / 2)} pairs. ${ev.ms.stored ? `Precomputed for this built-in protein; loaded in ${Math.round(ev.ms.net)} ms.` : Math.round(ev.ms.net) + ' ms'}` : 'Distance histogram and secondary structure for every residue'],
      ['4 · Restraints', ready ? (ev.shallow ? 'off <small>too few homologs</small>' : `${fmtInt(ev.rest.I.length)} <small>pair potentials</small>`) : '→ fold', ready ? (ev.shallow ? `Neff ${f1(ev.neff)} is below ${EVO_NEFF_MIN}, too little for the network to be trusted: folding uses fragment secondary structure and no AI restraints (the v2 inputs).` : 'Smooth −ln p(d)/p_ref(d) curves added to the force field of “MSA + AI”') : 'The predictions steer the folding run'],
    ];
    pipe.innerHTML = steps.map(([k, b, d], i) => `<div class="step" data-s="${st(i)}"><span class="k"><i></i>${k}</span><b>${b}</b><span class="d">${d}</span>${bar(i)}</div>`).join('');
    const msg = $('msa-msg');
    if (ev && ev.status === 'error') { msg.textContent = 'The MSA pipeline failed: ' + ev.error; msg.classList.add('err'); }
    else if (!msg.dataset.keep) { msg.classList.remove('err'); msg.textContent = ''; }
    $('msa-reset').hidden = !t.msaUpload;
    if (!ready) {
      $('evo-table').innerHTML = ''; $('msa-cap').textContent = ''; $('evo-cap').textContent = '';
      for (const id of ['msa-canvas', 'evo-map']) { const c = $(id); c.style.height = '0px'; c.height = 0; }
      if (ev && ev.status === 'error') $('evo-table').innerHTML = '<p class="fine"><button class="btn" id="evo-go" type="button">Try again</button></p>';
      const go = $('evo-go');
      if (go) go.addEventListener('click', () => { t.evo = null; dirty.evo = true; ensureEvo(t).catch(() => {}); });
      return;
    }
    drawMsa(); drawEvoMap();
    const rows = [];
    const pctOr = v => (v == null ? '–' : pct(v));
    if (t.X) rows.push(['Contact precision, top L/2 pairs ≥ 12 apart', pctOr(ev.prec), pctOr(ev.precDCA), 'Share of predicted contacts (Cα < 8 Å) that are real in the PDB structure']);
    if (t.ss) rows.push(['Secondary structure, Q3 vs PDB', pctOr(ev.q3), pctOr(ev.q3frag), EVO_SS_MIX != null ? `Network averaged ${Math.round(EVO_SS_MIX * 100)}:${Math.round(100 - EVO_SS_MIX * 100)} with the v2 fragment vote (network alone: ${pctOr(ev.q3net)}), vs the fragment vote alone` : 'Network prediction vs the v2 fragment vote (single sequence)']);
    rows.push(['Effective sequences (Neff)', f1(ev.neff), '', 'Sequences reweighted by 1 / neighbours at ≥ 80% identity. Below ~1 per residue, coevolution is mostly noise.']);
    const tr = EVO_MODEL.training && EVO_MODEL.training.val;
    $('evo-table').innerHTML = `<table><thead><tr><th>Measure</th><th class="num">MSA + AI</th><th class="num">Baseline</th><th>Note</th></tr></thead><tbody>${rows.map(r => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td><td class="num">${r[2] || ''}</td><td class="wrap">${esc(r[3])}</td></tr>`).join('')}</tbody></table>`
      + `<p class="fine">Baselines: raw DCA scores for contacts; fragment-vote prediction for secondary structure. ${tr ? `On ${EVO_MODEL.training.nval || 'held-out'} validation chains the network scored ${pct(tr.lr_p)} contact precision and ${pct(tr.q3)} Q3.` : ''} ${ev.source === 'none' ? 'This sequence has no alignment, so the network sees the sequence alone.' : ''}</p>`;
  }

  function drawMsa() {
    const t = state.target, ev = t.evo, c = $('msa-canvas'), L = ev.L;
    const W = contentWidth(c.parentElement), gut = 34, cw = (W - gut - 4) / L;
    const order = Array.from({ length: ev.N - 1 }, (_, k) => k + 1).sort((a, b) => ev.identity[b] - ev.identity[a]);
    const maxRows = Math.min(order.length, 150), rh = order.length ? clamp(260 / Math.max(1, maxRows), 1.6, 7) : 0;
    const shownRows = [];
    for (let k = 0; k < maxRows; k++) shownRows.push(order[Math.floor(k * order.length / maxRows)]);
    const topH = 82, H = topH + Math.ceil(maxRows * rh) + 6;
    const { ctx } = setup(c, H, W);
    evoView.msaLayout = { gut, cw, L, topH, rh, rows: shownRows };
    const y = { cons: 4, consH: 26, ai: 36, pdb: 50, q: 64 };
    // conservation
    for (let i = 0; i < L; i++) {
      const v = ev.conservation[i], h = Math.max(1, v * y.consH);
      ctx.fillStyle = css(mix(trgb.grid, trgb.accent, 0.35 + 0.65 * v));
      ctx.fillRect(gut + i * cw + 0.5, y.cons + y.consH - h, Math.max(1, cw - 1), h);
    }
    ctx.fillStyle = theme.muted; ctx.font = `10.5px ${theme.fontUi}`; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    ctx.fillText('cons.', gut - 5, y.cons + y.consH / 2);
    ctx.fillText('AI', gut - 5, y.ai + 4);
    drawSS(ctx, ev.labels, gut, cw, y.ai, 8);
    if (t.ss) { ctx.fillText('PDB', gut - 5, y.pdb + 4); drawSS(ctx, t.ss, gut, cw, y.pdb, 8); }
    // query row
    ctx.fillText('query', gut - 5, y.q + 6);
    for (let i = 0; i < L; i++) {
      const rgb = AA_CLASS[t.seq[i]];
      ctx.fillStyle = rgb ? css(rgb) : theme.coil;
      ctx.fillRect(gut + i * cw, y.q, Math.max(1, cw - (cw > 4 ? 0.6 : 0)), 12);
      if (cw >= 8) { ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.font = `500 ${Math.min(10, cw - 1)}px ${theme.fontMono}`; ctx.fillText(t.seq[i], gut + (i + 0.5) * cw, y.q + 6.5); }
    }
    // homolog rows
    for (let k = 0; k < shownRows.length; k++) {
      const s = shownRows[k], yy = topH + k * rh;
      for (let i = 0; i < L; i++) {
        const code = ev.X[s * L + i];
        if (code === 20) continue;
        const rgb = AA_CLASS[Evo.ALPH[code]];
        ctx.fillStyle = css(mix(trgb.panel, rgb, code === ev.X[i] ? 0.95 : 0.5));
        ctx.fillRect(gut + i * cw, yy, Math.max(1, cw - (cw > 4 ? 0.5 : 0)), Math.max(1, rh - (rh > 3 ? 0.5 : 0)));
      }
    }
    if (shownRows.length) {
      ctx.fillStyle = theme.muted; ctx.textAlign = 'right'; ctx.font = `10.5px ${theme.fontMono}`;
      ctx.fillText(pct(ev.identity[shownRows[0]]), gut - 5, topH + 5);
      ctx.fillText(pct(ev.identity[shownRows[shownRows.length - 1]]), gut - 5, topH + shownRows.length * rh - 4);
    } else {
      ctx.fillStyle = theme.muted; ctx.textAlign = 'left'; ctx.font = `12.5px ${theme.fontUi}`;
      ctx.fillText(t.evo.source === 'none' ? 'No homologs: upload an alignment to add evolutionary information.' : 'No homologs survived filtering.', gut, topH + 14);
    }
    $('msa-cap').textContent = shownRows.length
      ? `${shownRows.length < ev.N - 1 ? `${shownRows.length} of ${fmtInt(ev.N - 1)} filtered homologs, evenly sampled` : `All ${fmtInt(ev.N - 1)} filtered homologs`}, most similar first (left numbers: identity to the query). Faded cells differ from the query; blank cells are gaps. Bars on top: conservation.`
      : '';
  }
  function drawEvoMap() {
    const t = state.target, ev = t.evo, c = $('evo-map'), L = ev.L;
    const W = Math.min(contentWidth(c.parentElement), 460), pad = 30, size = W - pad - 6;
    const { ctx } = setup(c, size + pad + 6, W);
    const cs = size / L, x0 = pad, y0 = 6;
    evoView.mapLayout = { x0, y0, cs, L };
    const lo = trgb['heat-lo'], hi = trgb['heat-hi'], useDca = evoView.map === 'dca';
    // DCA scores to 0–1 via their z-score (top pairs ≈ 4σ)
    let dm = 0, dsd = 1;
    if (useDca) {
      let s = 0, q = 0, n = 0;
      for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) { const v = ev.dca[i * L + j]; s += v; q += v * v; n++; }
      dm = s / n; dsd = Math.sqrt(Math.max(1e-12, q / n - dm * dm));
    }
    const score = (i, j) => (useDca ? clamp((ev.dca[i * L + j] - dm) / dsd / 4, 0, 1) : ev.cmap[i * L + j]);
    for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) {
      ctx.fillStyle = css(mix(lo, hi, score(i, j)));
      ctx.fillRect(x0 + j * cs, y0 + i * cs, Math.ceil(cs), Math.ceil(cs));
      if (!t.X) { ctx.fillRect(x0 + i * cs, y0 + j * cs, Math.ceil(cs), Math.ceil(cs)); }
    }
    // lower triangle: PDB contacts, then the top-L predictions marked right / wrong
    const top = [];
    for (let i = 0; i < L; i++) for (let j = i + 6; j < L; j++) top.push([score(i, j), i, j]);
    top.sort((a, b) => b[0] - a[0]);
    const k = Math.min(top.length, L);
    if (t.X) {
      ctx.fillStyle = css(mix(lo, trgb.coil, 0.9));
      for (let i = 0; i < L; i++) for (let j = i + 3; j < L; j++) if (F.dist(t.X, i, j) < 8) ctx.fillRect(x0 + i * cs, y0 + j * cs, Math.ceil(cs), Math.ceil(cs));
      for (let q = 0; q < k; q++) {
        const [, i, j] = top[q], ok = F.dist(t.X, i, j) < 8, r = Math.max(1.6, cs * 0.42);
        ctx.beginPath(); ctx.arc(x0 + (i + 0.5) * cs, y0 + (j + 0.5) * cs, r, 0, 2 * Math.PI);
        if (ok) { ctx.fillStyle = theme.good; ctx.fill(); } else { ctx.strokeStyle = theme.bad; ctx.lineWidth = 1.3; ctx.stroke(); }
      }
    }
    ctx.strokeStyle = theme.line; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, size, size);
    ctx.strokeStyle = theme.muted; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + size, y0 + size); ctx.stroke();
    ctx.fillStyle = theme.muted; ctx.font = `11px ${theme.fontMono}`;
    const first = t.first || 1, every = L > 40 ? 20 : 10;
    for (let i = 0; i < L; i++) if ((first + i) % every === 0) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(first + i), x0 - 5, y0 + (i + 0.5) * cs); }
    const label = (txt, x, y, align) => {
      ctx.font = `500 12px ${theme.fontUi}`;
      const tw = ctx.measureText(txt).width, bx = align === 'right' ? x - tw - 10 : x;
      ctx.fillStyle = theme.panel; roundRect(ctx, bx, y - 11, tw + 10, 22, 6); ctx.fill();
      ctx.fillStyle = theme.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(txt, bx + 5, y);
    };
    label(useDca ? 'DCA (raw)' : 'AI · P(d < 8 Å)', x0 + size - 8, y0 + 18, 'right');
    if (t.X) label('PDB', x0 + 8, y0 + size - 16, 'left');
    let hits = 0;
    if (t.X) for (let q = 0; q < k; q++) if (F.dist(t.X, top[q][1], top[q][2]) < 8) hits++;
    $('evo-cap').textContent = t.X
      ? `Upper right: ${useDca ? 'raw coevolution strength' : 'contact probability from the network'}. Lower left: PDB contacts (grey) and the top ${k} predictions at least 6 apart, ${hits} correct (filled) and ${k - hits} wrong (rings).`
      : `${useDca ? 'Raw coevolution strength' : 'Contact probability from the network'} for every residue pair. No PDB structure to check against.`;
  }
  function evoTipMap(e) {
    const Lr = evoView.mapLayout, t = state.target, ev = t.evo;
    if (!Lr || !evoReady(t)) return;
    const r = $('evo-map').getBoundingClientRect(), j = Math.floor((e.clientX - r.left - Lr.x0) / Lr.cs), i = Math.floor((e.clientY - r.top - Lr.y0) / Lr.cs);
    if (i < 0 || j < 0 || i >= Lr.L || j >= Lr.L || i === j) { hideTip(); return; }
    const a = Math.min(i, j), b = Math.max(i, j), L = Lr.L, first = t.first || 1;
    const rows = [['P(contact)', pct(ev.cmap[a * L + b])], ['Predicted distance', f1(ev.dexp[a * L + b]) + ' Å'], ['DCA score', f2(ev.dca[a * L + b])]];
    if (t.X) rows.push(['PDB distance', f1(F.dist(t.X, a, b)) + ' Å']);
    showTip(`<b>${AA3[t.seq[a]] || t.seq[a]} ${first + a} – ${AA3[t.seq[b]] || t.seq[b]} ${first + b}</b>` + tipRows(rows), e.clientX, e.clientY);
  }
  function evoTipMsa(e) {
    const Lr = evoView.msaLayout, t = state.target, ev = t.evo;
    if (!Lr || !evoReady(t)) return;
    const r = $('msa-canvas').getBoundingClientRect(), i = Math.floor((e.clientX - r.left - Lr.gut) / Lr.cw);
    if (i < 0 || i >= Lr.L) { hideTip(); return; }
    const L = Lr.L, cnt = {};
    let n = 0;
    for (let s = 1; s < ev.N; s++) { const c = ev.X[s * L + i]; if (c !== 20) { cnt[c] = (cnt[c] || 0) + 1; n++; } }
    const common = Object.entries(cnt).sort((x, y) => y[1] - x[1]).slice(0, 4).map(([c, v]) => Evo.ALPH[c] + ' ' + pct(v / n)).join(', ');
    let best = -1, bp = 0;
    for (let j = 0; j < L; j++) if (Math.abs(j - i) >= 6 && ev.cmap[i * L + j] > bp) { bp = ev.cmap[i * L + j]; best = j; }
    const first = t.first || 1, ss = ev.pred.ss;
    const rows = [['Conservation', pct(ev.conservation[i])], ['Most common', common || '–'], ['AI: helix / strand / coil', `${pct(ss[3 * i])} / ${pct(ss[3 * i + 1])} / ${pct(ss[3 * i + 2])}`]];
    if (best >= 0) rows.push(['Likeliest partner', `${t.seq[best]}${first + best} (${pct(bp)})`]);
    showTip(`<b>${AA3[t.seq[i]] || t.seq[i]} ${first + i}</b>` + tipRows(rows), e.clientX, e.clientY);
  }
  function useAlignment(text, name) {
    const t = state.target, msg = $('msa-msg');
    t.msaUpload = { text, name };
    t.evo = null; delete t.lmCache.ai;
    msg.dataset.keep = '1'; msg.classList.remove('err'); msg.textContent = `Reading ${name}…`;
    ensureEvo(t).then(ev => {
      msg.textContent = ev.input ? `${name}: ${fmtInt(ev.input)} sequences (${ev.format}${ev.dropped ? `, ${ev.dropped} too dissimilar to align` : ''}), ${fmtInt(ev.N - 1)} kept after filtering, Neff ${f1(ev.neff)}.`
        : `${name}: no usable homolog sequences found, so the network sees the query alone.`;
      if (state.method === 'evo') { state.queue = []; newRun(); startRun(); }
    }).catch(err => { msg.classList.add('err'); msg.textContent = err.message; });
  }

  // ---------- saved example runs ----------
  const EXAMPLES = window.FOLD_EXAMPLES || [];
  function renderExamples() {
    const el = $('examples');
    if (!el) return;
    el.closest('section').hidden = !EXAMPLES.length;
    el.innerHTML = EXAMPLES.map((ex, k) => {
      const t = BUILTIN.find(p => p.id === ex.id);
      return `<button type="button" class="prot" data-k="${k}"><span class="pid">${esc(ex.id)}</span><span class="pname">${esc(t ? t.name : ex.id)}</span><span class="pmeta">${ex.bench.rmsd.toFixed(1)} Å · TM ${ex.bench.tm.toFixed(2)} · ${fmtSteps(ex.steps)} steps</span></button>`;
    }).join('');
    el.querySelectorAll('button').forEach(b => b.addEventListener('click', () => loadExample(EXAMPLES[+b.dataset.k])));
  }
  function decodeFrames(b64, n) {
    const bin = atob(b64), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const v = new Int16Array(u8.buffer), frames = [];
    for (let o = 0; o < v.length; o += 3 * n) frames.push(Float32Array.from(v.subarray(o, o + 3 * n), q => q / 100));
    return frames;
  }
  // Replays a saved run through the normal analysis path: same pool, clustering and scores as a live run.
  async function loadExample(ex) {
    const t = BUILTIN.find(p => p.id === ex.id), msg = $('load-msg');
    if (!t) return;
    state.queue = [];
    stopRunner();
    if (state.method !== 'evo') setMethod('evo');
    state.seed = 11; $('seed').value = 11; state.replicas = ex.R;
    state.params.evo = Object.assign({}, DEF_PARAMS.evo, ex.params);
    selectTarget(t);
    state.stepsIdx = Math.max(0, STEP_CHOICES.indexOf(ex.steps)); state.tHi = ex.tHi; state.tLo = ex.tLo;
    buildSliders();
    msg.classList.remove('err'); msg.textContent = `Loading a saved run of ${t.name}…`;
    try { await ensureEvo(t); } catch (e) { msg.classList.add('err'); msg.textContent = e.message; return; }
    const run = newRun(), n = t.seq.length;
    run.example = true; run.hopSteps = 0;   // the saved runs were made without restarts
    ex.reps.forEach((saved, r) => {
      const rep = run.reps[r], frames = decodeFrames(saved.x, n);
      frames.forEach((x, k) => pushSample(run, rep, { step: saved.step[k], T: saved.T[k], E: saved.E[k], x }, false));
      rep.done = true;
    });
    run.resumedAt = performance.now();
    finishRun(run);
    msg.textContent = `Saved run of ${t.name}: MSA + AI, seed 11, ${ex.R} replicas × ${fmtSteps(ex.steps)} steps. Replay it with the timeline, or press Fold again to run it yourself.`;
  }

  // ---------- method panel ----------
  function renderMethodPanel() {
    $('method-panel').innerHTML = `
      <div class="panel-head"><h3>How it works</h3></div>
      <p>Each residue is a bead at its Cα atom on a chain of 3.8 Å virtual bonds. Every replica starts from its own random coil and is cooled from a high to a low temperature with Langevin dynamics (BAOAB integrator), so it can cross barriers early and settle late. Replicas run in parallel Web Workers.</p>
      <h4>MSA + AI (version 6)</h4>
      <p><b>Alignment.</b> Homologs of each built-in protein were found with the ColabFold MMseqs2 server (UniRef30 plus environmental sequence databases). Insertions relative to the query are dropped and the alignment is filtered to at most 1,500 sequences that cover at least half the query and are under 90% identical to each other. Each sequence is weighted by 1 / (number of neighbours at ≥ 80% identity); the weights add up to the effective count Neff. You can upload or paste your own alignment in the MSA + AI tab; plain homolog sequences are aligned to the query with Smith–Waterman.</p>
      <p><b>Coevolution.</b> Mean-field direct coupling analysis (Morcos et al., 2011): the regularised covariance of every residue pair (pseudocount 0.5) is inverted to separate direct couplings from chains of indirect ones; each pair is scored by the Frobenius norm of its couplings, minus the average-product correction (Dunn et al., 2008). Mutual information is computed alongside.</p>
      <p><b>Neural network.</b> A ${EVO_MODEL ? fmtInt(EVO_MODEL.params / 1000) + 'k' : 'small'}-parameter residual convolutional network in the style of RaptorX and trRosetta. Its 1D part reads the sequence profile and predicts helix, strand or coil; its 2D part, with dilated 3×3 convolutions, reads DCA, mutual information and both residues' 1D features and predicts an 18-bin histogram of the Cα–Cα distance (under 4 Å, 1 Å bins to 20 Å, beyond). It was trained offline with PyTorch on the alignments of 2,651 PDB chains: the ${KB.chains} knowledge-base chains plus 2,276 more from the same non-redundant set (X-ray, ≤ 1.8 Å, under 30% identity to each other), with every chain related to the six test proteins removed (113 in all). Alignment-depth subsampling and random cropping teach it to cope with shallow alignments and short chains. It runs here in plain JavaScript.</p>
      <p><b>Version 3 to 4.</b> Version 3 was the same network trained on the 375 knowledge-base chains alone. On 72 new chains that neither version saw, the larger training set raised long-range contact precision from 52% to 56% and secondary-structure accuracy from 75% to 79%. Folding the six test proteins stayed level (benchmark below): better predictions for villin and the homeodomain, worse for Trp-cage.</p>
      <p><b>Version 4 to 6.</b> Version 6 is the same network trained on 2,651 chains. On 162 new chains neither version saw, long-range contact precision rose from 59% to 62% and secondary-structure accuracy from 81% to 83%. This time the folds followed: over two random seeds, villin went from 7.2 to 2.4 Å, protein G from 5.3 to 3.8 Å and ubiquitin from 3.5 to 3.2 Å, with the homeodomain level at 2.7 Å (mean of the four 4.7 → 3.0 Å, TM-score 0.50 → 0.58). Chignolin and Trp-cage have no real alignment, so the fallback skips the network for them and they are unchanged.</p>
      <p><b>Versions 7 and 8.</b> Version 7 widened the network to 64 and 48 channels (583k parameters) and trained it on about 5,500 chains; on 287 chains neither saw, long-range contact precision rose from 61% to 63%. Version 8 fine-tuned v7 for 8 more epochs after adding 781 chains of 12–49 and 251–396 residues, and after removing every training chain related to four new test proteins (WW domain, CheY, T4 lysozyme, adenylate kinase). On unseen small chains its distance loss fell from 2.31 to 2.06, and on unseen large chains long-range precision rose from 69% to 71%. Over three seeds × 12 replicas the four new proteins fold to 3.8, 2.8, 4.2 and 4.1 Å (TM-score 0.56–0.77), and on 8 of the 10 proteins the three seeds land within 0.4 Å of each other.</p>
      <p><b>Choosing the final model.</b> On protein G, v7 sometimes found the native fold in half the replicas but picked a β-register shift (strands paired 2–4 residues off) that fits the predicted distances very slightly better. The cluster score now adds three times the backbone dihedral energy when restraints are on, and the lab runs 12 replicas by default. On two fresh seeds the protein G result went from 6.1 and 5.4 Å to 3.2 and 3.1 Å, with no protein made worse.</p>
      <p><b>Restraints.</b> Each pair likely to lie within 20 Å gets a smooth potential from its histogram, −ln[p(d) / p(19.5 Å)] + 1.57 ln(d / 19.5 Å), the form trRosetta uses. They are switched on by sequence separation (pairs up to 12 apart, then up to 24, then all) so that helices and turns form, with the handedness the fragments prefer, before the long-range pulls arrive; distances alone cannot tell a fold from its mirror image. Fragments are picked to match the network's secondary structure. Everything else is the v2 force field below.</p>
      <p><b>Choices made on the benchmark.</b> Two settings were picked by comparing them on these six proteins, so v3 is not entirely blind: the trRosetta reference against a background reference built from training-set distances (mean 4.9 vs 5.1 Å), and, after a control run with near-perfect restraints showed replicas folding into the mirror image, choosing the lowest-energy large cluster rather than the largest. Averaging the network's secondary structure with the fragment vote was also tried and dropped.</p>
      <p><b>Limits.</b> The two designed peptides (chignolin, Trp-cage) have no natural homologs, so the network sees one sequence and adds little. Coevolution resolves contacts well once Neff is roughly the chain length or more.</p>
      <h4>Small proteins (version 5)</h4>
      <p><b>Fallback for shallow alignments.</b> When an alignment has fewer than  effective sequences, as for the two designed peptides, the network is working from a single sequence and its predictions mostly add noise (20–25% secondary-structure accuracy on chignolin and Trp-cage). MSA + AI then folds with the v2 inputs instead: fragments picked by their own vote and no AI restraints. The MSA + AI tab says when this happens.</p>
      <p><b>Backbone hydrogen bonds.</b> One bead per residue has no hydrogen bonds of its own, and the strand-pairing term only acted where strands were predicted. The new term rewards any pair of residues at about 4.9 Å whose Cα-zigzag normals both point along the pair vector, the geometry of a backbone hydrogen bond in a hairpin or a helix turn, whatever the prediction says. Its strength is the “Backbone hydrogen bonds” slider (default 1.0 for MSA + AI, off for v2).</p>
      <p><b>Effect.</b> Over four random seeds, Trp-cage went from 3.4 Å with the fallback alone to 2.6 Å with hydrogen bonds as well (4.8 Å before either), while chignolin stayed around 3.6–3.7 Å (between 1.7 and 4.8 Å depending on the seed) and villin around 7.2 Å. The three larger proteins were unchanged within noise. The strength 1.0 was chosen from 0, 0.5 and 1.0 on this benchmark.</p>
      <h4>Monte Carlo restarts</h4>
      <p>Each replica watches its own energy. When the smoothed energy has not reached a new low for 50k steps, it tries a basin hop: one Monte Carlo move (insert library fragments at random windows, swing the chain about a pivot residue, or reheat), a short re-anneal, then the Metropolis rule decides whether to keep the new state or go back. Moves that worked are tried more often, and a move that just failed is not repeated. Hops happen only between 35% and 80% of the run, so the final stretch is ordinary sampling. Switch them off with the “Monte Carlo restarts” checkbox in the Method card, and use “Restart severity” to make each move gentler or more drastic (more fragment swaps, wider pivot swings, hotter reheats; the middle setting is the one benchmarked); the Replicas tab shows how many each replica kept.</p>
      <p>On the benchmark they made no real difference: mean 4.60 Å / TM 0.40 with restarts against 4.67 Å / 0.41 without, well within noise. Most kept restarts did reach lower energies, but those states were no closer to the real structure, which points at the energy function (its lowest-energy structure is not exactly the real one) more than at the search.</p>
      <h4>Knowledge-based force field (version 2)</h4>
      <p><b>Local structure.</b> For every 9-, 5- and 3-residue window of the sequence, the best-matching windows (BLOSUM62) are picked from a library of ${fmtInt(KB.residues)} residues in ${KB.chains} PDB chains, the way Rosetta picks fragments. Their Cα bond angles and dihedrals become position-specific potentials, and their secondary-structure labels vote for the prediction shown in the Residues tab. With “From PDB record” the picker also rewards fragments whose labels match the record.</p>
      <p><b>Packing.</b> Each residue carries a side-chain bead placed by a per-residue fit to real side-chain centroids, which captures the handedness of L-amino acids. Beads feel a burial potential, −ln P(neighbours | residue type) / P(neighbours), contact-reference pair preferences such as salt bridges, and excluded volume. Extended residues pair through a directional β-sheet term. All statistics come from the same library.</p>
      <p><b>No peeking.</b> The library is a 30%-identity non-redundant set of X-ray structures at 1.8 Å or better. Any chain that aligns to one of the six test proteins above a shuffled-sequence significance threshold was removed (19 chains, including NEDD8 and other ubiquitin-like proteins, protein L and homeodomain relatives).</p>
      <h4>Choosing the final model</h4>
      <p>Late low-temperature frames from all replicas are clustered by RMSD; the frame with the most neighbours (the medoid of the densest cluster) is the consensus model, as in SPICKER and Rosetta. It is usually more reliable than the single lowest-energy frame. With MSA restraints, a fold and its mirror image satisfy the predicted distances equally well, so among the large clusters the one with the lowest energies wins (energy plus three times the backbone dihedral term); the fragment potentials know which way helices twist.</p>
      <h4>Atoms on screen</h4>
      <p>The PDB structure shows its real heavy atoms. The model is coarse-grained, so its atoms are rebuilt: backbone N, C and O from peptide-plane orientations learned from the library (0.2–0.5 Å error for N and C on the test proteins), side chains from the library's most common rotamer for each residue type.</p>
      <h4>Scores</h4>
      <p>RMSD after optimal superposition; TM-score and GDT-TS maximised over superpositions; Q, the share of PDB contacts within 1.2× their native distance; secondary structure from Cα geometry against the PDB record; and an energy check that scores the PDB structure with the same force field.</p>
      <h4>Benchmark</h4>
      <p>Consensus-model Cα RMSD to the PDB structure (Å, lower is better). Columns up to v6 were measured with 6 replicas, seed 11 and the default schedule; the v8 column is the mean of three seeds (11, 101, 202) with 12 replicas each and the current model selection. Reproduce any row with “Run all six”.</p>
      ${benchTable()}
      <h4>Methods compared</h4>
      <p><b>Simple physics</b> is version 1 of this lab (Chou–Fasman propensities and a hydrophobic attraction). <b>Native contacts (Gō)</b> follows Clementi, Nymeyer &amp; Onuchic (2000); it is built from the answer and shows a folding route, not a prediction.</p>
      <p>Structures: ${BUILTIN.map(p => `<a href="https://www.rcsb.org/structure/${p.id}" target="_blank" rel="noopener">${p.id}</a>`).join(', ')}, RCSB Protein Data Bank.</p>`;
  }

  // ---------- controls ----------
  const PDB_ID = /^[1-9][A-Z0-9]{3}$/;   // a four-character PDB entry code
  function renderProteins() {
    const list = $('plist'), items = BUILTIN.slice();
    if (!BUILTIN.includes(state.target)) items.push(state.target);
    list.innerHTML = items.map((p, k) => {
      const last = state.history.find(h => h.id === p.id && h.n === p.seq.length && h.method === state.method);
      const meta = [p.seq.length + ' aa', p.fold].filter(Boolean).join(' · ') + (last && last.rmsd != null ? ` · last ${last.rmsd.toFixed(1)} Å` : '');
      const link = PDB_ID.test(p.id) ? `<a class="plink" href="https://www.rcsb.org/structure/${p.id}" target="_blank" rel="noopener" title="Open ${esc(p.id)} on the RCSB Protein Data Bank" aria-label="Open ${esc(p.id)} on the RCSB Protein Data Bank"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5H3.5v9h9V10M9 3.5h3.5V7M12.5 3.5 7 9" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></a>` : '';
      return `<div class="pitem"><button type="button" class="prot${link ? ' haslink' : ''}" role="radio" aria-checked="${p === state.target}" data-k="${k}"><span class="pid">${esc(p.id)}</span><span class="pname">${esc(p.name)}</span><span class="pmeta">${esc(meta)}</span></button>${link}</div>`;
    }).join('');
    list.querySelectorAll('.prot').forEach(b => b.addEventListener('click', () => { state.queue = []; selectTarget(items[+b.dataset.k]); startRun(); }));
  }
  function renderMethods() {
    const el = $('methods'), t = state.target;
    el.innerHTML = Object.entries(METHODS).map(([k, m]) => `<button type="button" class="method" role="radio" aria-checked="${k === state.method}" data-m="${k}" ${k === 'go' && !t.X ? 'disabled' : ''}>
      <span class="radio"></span><b>${m.name}${m.badge ? `<span class="badge">${m.badge}</span>` : ''}</b><span class="d">${m.desc}</span></button>`).join('');
    el.querySelectorAll('.method').forEach(b => b.addEventListener('click', () => {
      if (b.disabled || b.dataset.m === state.method) return;
      state.queue = [];
      setMethod(b.dataset.m);
      newRun(); startRun();
    }));
  }
  function selectTarget(t) {
    state.target = t;
    delete $('msa-msg').dataset.keep;
    dirty.evo = true;
    state.stepsIdx = autoStepsIdx(t.seq.length);
    if (!t.X && state.method === 'go') { state.method = 'kb'; state.tHi = METHODS.kb.tHi; state.tLo = METHODS.kb.tLo; }
    state.ssSource = state.ssPref === 'real' && t.ss ? 'real' : 'pred';
    topoFor = null;
    renderProteins(); renderMethods(); buildSliders();
    newRun();
  }
  function setMethod(m) {
    state.method = m;
    state.tHi = METHODS[m].tHi; state.tLo = METHODS[m].tLo;
    renderMethods(); buildSliders(); renderProteins();
  }
  function sliderHTML(id, label, value, min, max, step, hint, out) {
    return `<div class="slider"><label for="${id}">${label}</label><output id="${id}-out" for="${id}">${out}</output>
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}">${hint ? `<span class="hint">${hint}</span>` : ''}</div>`;
  }
  function estSeconds() {
    const per = 3.6e6 / state.target.seq.length * (state.method === 'kb' ? 0.8 : state.method === 'evo' ? 0.7 : 1);
    return totalSteps() / (state.watch ? 3000 : per);
  }
  function scheduleHint() {
    const s = estSeconds();
    return `About ${s < 90 ? Math.max(1, Math.round(s)) + ' s' : Math.round(s / 60) + ' min'} at ${state.watch ? 'slow-motion' : 'full'} speed, replicas in parallel. Auto-set to 15k × length.`;
  }
  function buildSliders() {
    const m = METHODS[state.method], P = state.params[state.method];
    $('sliders').innerHTML = m.sliders.map(s => sliderHTML('p-' + s.key, s.label, P[s.key], 0, 3, 0.1, s.hint, '×' + P[s.key].toFixed(1))).join('')
      + (m.sliders.length ? '<div><button class="btn" id="p-defaults" type="button">Restore defaults</button></div>' : '');
    m.sliders.forEach(s => {
      const inp = $('p-' + s.key);
      inp.addEventListener('input', () => {
        P[s.key] = +inp.value;
        $('p-' + s.key + '-out').textContent = '×' + (+inp.value).toFixed(1);
        liveParams();
      });
    });
    if ($('p-defaults')) $('p-defaults').addEventListener('click', () => { state.params[state.method] = Object.assign({}, DEF_PARAMS[state.method]); buildSliders(); liveParams(); });
    $('steps-field').innerHTML = sliderHTML('s-steps', 'Total steps per replica', state.stepsIdx, 0, STEP_CHOICES.length - 1, 1, scheduleHint(), fmtSteps(totalSteps()));
    $('schedule').innerHTML =
      sliderHTML('s-thi', 'Start temperature, kT', state.tHi, 0.3, 3, 0.05, 'Hot enough to unfold and cross barriers', state.tHi.toFixed(2))
      + sliderHTML('s-tlo', 'End temperature, kT', state.tLo, 0.05, 1.5, 0.05, 'Cooled geometrically over 85% of the run, then held', state.tLo.toFixed(2));
    $('s-steps').addEventListener('input', e => onStepsInput(+e.target.value));
    $('s-steps').addEventListener('change', () => commitSteps());
    $('s-thi').addEventListener('input', e => { state.tHi = +e.target.value; $('s-thi-out').textContent = state.tHi.toFixed(2); liveSchedule(); });
    $('s-tlo').addEventListener('input', e => { state.tLo = +e.target.value; $('s-tlo-out').textContent = state.tLo.toFixed(2); liveSchedule(); });
    renderControls();
  }
  function liveParams() {
    const run = state.run;
    if (!run || run.status === 'done' || run.method !== state.method) return;
    run.params = Object.assign({}, state.params[state.method]);
    run.sys = buildSys(run);
    nativeEnergies(run);
    if (run.runner) run.runner.update({ params: run.params }, true);
    dirty.charts = true;
  }
  // "Add steps": a running or paused run gets a longer schedule on the fly; a finished one folds again, longer.
  // Both step sliders (sidebar and under the 3D view) show the same value.
  function syncSteps() {
    const sl = $('s-steps'), tl = $('tl-steps');
    if (draggingSteps && document.activeElement && (document.activeElement === sl || document.activeElement === tl)) {
      $('s-steps-out') && ($('s-steps-out').textContent = fmtSteps(totalSteps())); $('tl-steps-out').textContent = fmtSteps(totalSteps());
      return;
    }
    if (sl) { sl.value = state.stepsIdx; $('s-steps-out').textContent = fmtSteps(totalSteps()); sl.parentElement.querySelector('.hint').textContent = scheduleHint(); }
    if (tl) { tl.max = STEP_CHOICES.length - 1; tl.value = state.stepsIdx; $('tl-steps-out').textContent = fmtSteps(totalSteps()); tl.title = scheduleHint(); }
  }
  // A run in progress can't be shortened below where it already is: snap the step choice up past the furthest replica.
  function clampSteps() {
    const run = state.run;
    if (!run || (run.status !== 'running' && run.status !== 'paused') || run.example) return;
    const at = Math.max(...run.reps.map(r => r.step)) + run.every;
    while (state.stepsIdx < STEP_CHOICES.length - 1 && STEP_CHOICES[state.stepsIdx] < at) state.stepsIdx++;
  }
  let stepsTimer = 0, draggingSteps = false;
  function onStepsInput(idx) {
    state.stepsIdx = idx; draggingSteps = true;
    syncSteps();
    clearTimeout(stepsTimer); stepsTimer = setTimeout(commitSteps, 250);
  }
  function commitSteps() {
    clearTimeout(stepsTimer); draggingSteps = false;
    liveSchedule(); renderControls();
  }
  function liveSchedule() {
    const run = state.run;
    if (!run || run.status === 'done' || run.example) return;
    clampSteps(); syncSteps();
    const prevSteps = run.steps;
    run.steps = totalSteps(); run.tHi = state.tHi; run.tLo = state.tLo;
    // keep roughly 300 samples per replica: widen the sampling interval in whole multiples so samples stay aligned
    const k = Math.max(1, Math.round(sampleEvery(run.steps) / run.every));
    if (k > 1) run.every *= k;
    if (run.runner) run.runner.update({ steps: run.steps, tHi: run.tHi, tLo: run.tLo, every: run.every, prevSteps }, false);
    dirty.charts = true;
  }
  function renderControls() {
    const run = state.run, t = state.target;
    document.querySelectorAll('#ss-seg button').forEach(b => {
      b.setAttribute('aria-pressed', String(b.dataset.v === state.ssSource));
      b.disabled = b.dataset.v === 'real' && !t.ss;
    });
    $('ss-field').hidden = state.method === 'go';
    $('hop-field').hidden = state.method === 'go';   // Gō runs never restart
    renderHop();
    document.querySelectorAll('#layout-seg button').forEach(b => { b.setAttribute('aria-pressed', String(b.dataset.v === state.view.layout)); b.disabled = !t.X && b.dataset.v === 'side'; });
    $('show-pdb').disabled = !t.X;
    const mname = METHODS[state.method].name;
    $('crumb').innerHTML = `<b>${esc(t.name)}</b> · ${esc(mname)}${state.method !== 'go' ? ` · ${state.ssSource === 'real' ? 'SS from PDB record' : 'blind'}` : ''} · ${run ? run.R : state.replicas} replicas`;
    const btn = $('run');
    btn.textContent = !run ? 'Fold' : run.status === 'running' ? 'Pause' : run.status === 'paused' ? 'Resume' : run.status === 'done' ? 'Fold again' : 'Fold';
    $('run-all').textContent = state.queue.length ? `Stop after this run (${state.queue.length} left)` : 'Run all six';
    const st = $('status'), steps = run ? Math.min(...run.reps.map(r => r.step)) : 0, frac = run ? steps / run.steps : 0;
    if (evoWait) { st.dataset.state = 'running'; $('status-text').textContent = evoStatusText(t); }
    else if (!run || run.status === 'ready') { st.dataset.state = 'idle'; $('status-text').textContent = 'Ready'; }
    else if (run.status === 'running') { st.dataset.state = 'running'; $('status-text').textContent = `Folding ${t.id} · ${frac < 0.1 ? (frac * 100).toFixed(1) : Math.round(frac * 100)}%`; }
    else if (run.status === 'paused') { st.dataset.state = 'paused'; $('status-text').textContent = `Paused at ${Math.round(frac * 100)}%`; }
    else { st.dataset.state = 'done'; const m = run.final.consensus.metrics; const nn = run.final.nearNative; $('status-text').textContent = nn ? `Done: near-native frame found, ${nn.metrics.rmsd.toFixed(1)} Å` : m ? `${run.stopHit ? 'Stopped at same fold' : 'Done'}: ${m.rmsd.toFixed(1)} Å from PDB ${t.id}` : 'Done'; }
    $('stopfold').checked = state.stopAtFold; $('stopfold').disabled = !t.X;
    $('stopfold-field').classList.toggle('off', !t.X);
    $('replicas').value = state.replicas; $('replicas-out').textContent = state.replicas;
    syncSteps();
    const go = $('tl-start');
    go.textContent = !run ? 'Start' : run.status === 'running' ? 'Pause' : run.status === 'paused' ? 'Resume' : run.status === 'done' ? 'Start again' : 'Start';
    go.title = `Fold with ${fmtSteps(totalSteps())} steps per replica`;
  }
  let speedMark = { t: 0, steps: 0 };
  function updateReadout() {
    const run = state.run;
    if (!run) return;
    const d = shown();
    $('ro-step').textContent = fmtInt(d.step || 0);
    $('ro-t').textContent = d.T != null ? f2(d.T) : '–';
    $('ro-e').textContent = isFinite(d.E) ? signed(d.E) + ' ε' : '–';
    const total = run.reps.reduce((a, r) => a + r.step, 0), now = performance.now();
    if (run.status === 'running' && now - speedMark.t > 800) { run.speed = (total - speedMark.steps) / ((now - speedMark.t) / 1000); speedMark = { t: now, steps: total }; }
    $('ro-speed').textContent = run.status === 'running' && run.speed > 0 ? `${fmtSteps(Math.round(run.speed / 1000) * 1000)} steps/s total` : '';
    $('progress').style.width = (100 * Math.min(...run.reps.map(r => r.step)) / run.steps).toFixed(1) + '%';
    const rep = run.reps[state.focus], sc = $('scrub');
    sc.max = String(Math.max(0, rep.frames.length - 1));
    sc.value = state.scrub == null ? sc.max : String(state.scrub);
    if (run.status === 'running') renderControls();
  }

  function bindControls() {
    $('run').addEventListener('click', () => { const r = state.run; if (r && r.status === 'running') pauseRun(); else startRun(); });
    document.querySelectorAll('#ss-seg button').forEach(b => b.addEventListener('click', () => {
      if (b.disabled || b.dataset.v === state.ssSource) return;
      state.ssSource = state.ssPref = b.dataset.v;
      state.queue = [];
      newRun(); startRun();
    }));
    $('replicas').addEventListener('input', e => { state.replicas = +e.target.value; $('replicas-out').textContent = state.replicas; });
    $('replicas').addEventListener('change', () => { state.queue = []; newRun(); startRun(); });
    $('seed').addEventListener('change', e => { state.seed = clamp(Math.round(+e.target.value) || 1, 1, 999999); e.target.value = state.seed; newRun(); startRun(); });
    $('reroll').addEventListener('click', () => { state.seed = 1 + Math.floor(Math.random() * 999998); $('seed').value = state.seed; newRun(); startRun(); });
    $('stopfold').addEventListener('change', e => {
      state.stopAtFold = e.target.checked;
      const run = state.run;
      if (run && !run.example && run.status !== 'done') run.stopAtFold = state.stopAtFold;
    });
    $('hop-sev').addEventListener('input', e => {
      state.hopSeverity = +e.target.value;
      renderHop();
      const run = state.run;
      if (!run || run.example || run.status === 'done') return;
      run.hopSeverity = state.hopSeverity;
      if (run.runner) run.runner.update({ hopSeverity: run.hopSeverity }, false);
    });
    $('hop').addEventListener('change', e => {
      state.hopOn = e.target.checked;
      renderHop();
      const run = state.run;
      if (!run || run.method === 'go' || run.example || run.status === 'done') return;
      run.hopSteps = state.hopOn ? HOP_PATIENCE : 0;
      if (run.runner) run.runner.update({ hopSteps: run.hopSteps }, false);
    });
    $('watch').addEventListener('change', e => {
      state.watch = e.target.checked;
      const run = state.run;
      if (run && run.runner) run.runner.update({ watch: state.watch }, false);
      const h = $('s-steps') && $('s-steps').parentElement.querySelector('.hint');
      if (h) h.textContent = scheduleHint();
    });
    $('run-all').addEventListener('click', () => {
      if (state.queue.length) { state.queue = []; renderControls(); return; }
      state.queue = BUILTIN.slice();
      nextQueued();
    });
    $('clear-log').addEventListener('click', () => { state.history = []; saveHistory(); renderHistory(); renderProteins(); });
    $('rep').addEventListener('change', e => { state.view.rep = e.target.value; pdbDirty = true; dirty.viewer = true; });
    $('colorby').addEventListener('change', e => { state.view.color = e.target.value; pdbDirty = true; dirty.viewer = true; });
    document.querySelectorAll('#layout-seg button').forEach(b => b.addEventListener('click', () => { if (b.disabled) return; state.view.layout = b.dataset.v; pdbDirty = true; dirty.viewer = true; renderControls(); }));
    $('show-model').addEventListener('click', e => { state.view.showModel = !state.view.showModel; e.currentTarget.setAttribute('aria-pressed', String(state.view.showModel)); pdbDirty = true; dirty.viewer = true; });
    $('show-pdb').addEventListener('click', e => { state.view.showPdb = !state.view.showPdb; e.currentTarget.setAttribute('aria-pressed', String(state.view.showPdb)); dirty.viewer = true; });
    $('spin').addEventListener('click', e => { state.view.spin = !state.view.spin; e.currentTarget.setAttribute('aria-pressed', String(state.view.spin)); if (viewer) viewer.setSpin(state.view.spin); });
    $('recenter').addEventListener('click', () => { if (viewer) viewer.resetView(); });
    $('scrub').addEventListener('input', e => {
      const run = state.run;
      if (!run) return;
      const rep = run.reps[state.focus], k = +e.target.value;
      state.scrub = k >= rep.frames.length - 1 && run.status !== 'done' ? null : k;
      if (state.scrub == null) state.frame = 'live';
      state.replaying = false; setReplayIcon();
      dirty.all(); renderControls();
    });
    $('tl-start').addEventListener('click', () => $('run').click());   // same as the main Fold button
    $('tl-steps').addEventListener('input', e => onStepsInput(+e.target.value));
    $('tl-steps').addEventListener('change', () => commitSteps());
    $('replay').addEventListener('click', () => {
      if (!state.run) return;
      if (state.replaying) { state.replaying = false; setReplayIcon(); return; }
      state.replaying = true; state.scrub = 0; replayAt = performance.now();
      setReplayIcon(); dirty.all();
    });
    document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {
      state.tab = b.dataset.tab;
      document.querySelectorAll('#tabs button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
      document.querySelectorAll('[data-panel]').forEach(p => { p.hidden = p.dataset.panel !== state.tab; });
      dirty.all();
    }));
    $('msa-paste-toggle').addEventListener('click', e => { const box = $('msa-paste'); box.hidden = !box.hidden; e.currentTarget.setAttribute('aria-expanded', String(!box.hidden)); });
    $('msa-use').addEventListener('click', () => {
      const text = $('msa-text').value;
      if (!text.trim()) { const m = $('msa-msg'); m.dataset.keep = '1'; m.classList.add('err'); m.textContent = 'Paste an alignment or a list of homolog sequences first.'; return; }
      useAlignment(text, 'pasted alignment');
    });
    $('msa-file').addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      if (file.size > 40e6) { const m = $('msa-msg'); m.dataset.keep = '1'; m.classList.add('err'); m.textContent = 'That file is over 40 MB; filter it first (e.g. hhfilter -diff 2000).'; e.target.value = ''; return; }
      const reader = new FileReader();
      reader.onload = () => { useAlignment(String(reader.result), file.name); e.target.value = ''; };
      reader.readAsText(file);
    });
    $('msa-reset').addEventListener('click', () => {
      const t = state.target;
      delete t.msaUpload; t.evo = null; delete t.lmCache.ai;
      const m = $('msa-msg'); delete m.dataset.keep; m.textContent = '';
      ensureEvo(t).then(() => { if (state.method === 'evo') { state.queue = []; newRun(); startRun(); } }).catch(() => {});
    });
    document.querySelectorAll('#evo-map-seg button').forEach(b => b.addEventListener('click', () => {
      evoView.map = b.dataset.v;
      document.querySelectorAll('#evo-map-seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      if (evoReady(state.target)) drawEvoMap();
    }));
    $('evo-map').addEventListener('pointermove', evoTipMap);
    $('evo-map').addEventListener('pointerleave', hideTip);
    $('msa-canvas').addEventListener('pointermove', evoTipMsa);
    $('msa-canvas').addEventListener('pointerleave', hideTip);
    $('seq-toggle').addEventListener('click', e => { const box = $('custom'); box.hidden = !box.hidden; e.currentTarget.setAttribute('aria-expanded', String(!box.hidden)); });
    $('seq-use').addEventListener('click', () => {
      const seq = $('seq-input').value.toUpperCase().replace(/[^A-Z]/g, ''), bad = seq.replace(/[ACDEFGHIKLMNPQRSTVWY]/g, ''), msg = $('load-msg');
      msg.classList.toggle('err', true);
      if (bad) { msg.textContent = `Unknown residue letters: ${[...new Set(bad)].join(', ')}. Use the 20 standard one-letter codes.`; return; }
      if (seq.length < 8 || seq.length > MAX_RES) { msg.textContent = `The sequence has ${seq.length} residues; use 8 to ${MAX_RES}.`; return; }
      msg.classList.toggle('err', false);
      msg.textContent = `Folding a ${seq.length}-residue custom sequence.`;
      state.queue = [];
      selectTarget(prepTarget({ id: 'SEQ', name: 'Custom sequence', fold: '', method: 'no structure', seq, ss: null, ca: null, first: 1 }));
      startRun();
    });
    $('pdb-file').addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const msg = $('load-msg');
      readStructureFile(file).then(text => {
        const p = parseStructure(text, file.name);
        msg.classList.toggle('err', false);
        msg.textContent = `Loaded ${p.id}: ${p.seq.length} residues from chain ${p.chain}.` + (p.warning ? ' ' + p.warning : '');
        state.queue = [];
        selectTarget(prepTarget(p));
        startRun();
      }).catch(err => { msg.classList.toggle('err', true); msg.textContent = err.message; })
        .finally(() => { e.target.value = ''; });
    });
  }
  let replayAt = 0;
  function setReplayIcon() {
    $('replay-icon').innerHTML = state.replaying ? '<rect x="4" y="3.5" width="3" height="9" rx="1" fill="currentColor"/><rect x="9" y="3.5" width="3" height="9" rx="1" fill="currentColor"/>' : '<path d="M5 3.5v9l7-4.5z" fill="currentColor"/>';
  }

  // ---------- PDB upload ----------
  const THREE3 = { ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F',
    PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', MSE: 'M', HSD: 'H', HSE: 'H', HIE: 'H', HID: 'H' };
  // Residues are gathered per chain; the first protein chain with 8–MAX_RES residues is used.
  function chainCollector() {
    const chains = new Map();
    return {
      chains,
      add(ch, key, num, resn, name, el, x, y, z) {
        el = (el || name[0]).toUpperCase();
        if (el === 'H' || el === 'D') return;
        if (resn === 'MSE' && name === 'SE') { name = 'SD'; el = 'S'; }
        let c = chains.get(ch);
        if (!c) { c = { res: [], byKey: new Map() }; chains.set(ch, c); }
        let r = c.byKey.get(key);
        if (!r) { r = { num, aa: THREE3[resn], atoms: [] }; c.byKey.set(key, r); c.res.push(r); }
        if (name !== 'OXT' && !r.atoms.some(q => q[0] === name)) r.atoms.push([name, el, x, y, z]);
      },
    };
  }
  function buildUploaded({ chains, id, title, ranges, filename, format }) {
    const all = [...chains].map(([ch, c]) => ({ ch, ca: c.res.filter(r => r.atoms.some(q => q[0] === 'CA')) })).filter(c => c.ca.length);
    if (!all.length) throw new Error(`No protein Cα atoms found in this ${format} file.`);
    const fit = all.find(c => c.ca.length >= 8 && c.ca.length <= MAX_RES);
    if (!fit) {
      const c = all[0];
      if (c.ca.length < 8) throw new Error(`Chain ${c.ch} has only ${c.ca.length} residues; the lab needs at least 8.`);
      throw new Error(`Chain ${c.ch} has ${c.ca.length} residues${all.length > 1 ? `, and no other chain has 8–${MAX_RES}` : ''}. This browser model handles up to ${MAX_RES}; try a single domain.`);
    }
    const { ch: chain, ca } = fit;
    const coords = [];
    for (const r of ca) { const q = r.atoms.find(t => t[0] === 'CA'); coords.push(q[2], q[3], q[4]); }
    const ss = ca.map(r => { let v = 'C'; for (const [ty, c, lo, hi] of ranges) if (c === chain && r.num >= lo && r.num <= hi) v = ty; return v; }).join('');
    let breaks = 0;
    for (let i = 0; i < ca.length - 1; i++) if (Math.hypot(coords[3 * i + 3] - coords[3 * i], coords[3 * i + 4] - coords[3 * i + 1], coords[3 * i + 5] - coords[3 * i + 2]) > 4.3) breaks++;
    const hasSS = ranges.some(r => r[1] === chain);
    const name = title.trim() ? title.trim().toLowerCase().replace(/^\w/, c => c.toUpperCase()).slice(0, 48) : filename;
    let warning = '';
    if (fit !== all[0]) warning += `Chain ${all[0].ch} has ${all[0].ca.length} residues, so chain ${chain} (${ca.length}) is used. `;
    if (breaks) warning += `Chain ${chain} has ${breaks} gap${breaks > 1 ? 's' : ''} (missing residues); the model joins across them with ordinary bonds. `;
    if (!hasSS) warning += `The file has no ${format === 'mmCIF' ? 'helix/sheet annotation' : 'HELIX/SHEET records'}, so “From PDB record” is off. `;
    return { id: (id || filename.replace(/\.(cif|pdb|ent|txt)?(\.gz)?$/i, '').slice(0, 8)).toUpperCase(), name, fold: '', method: 'uploaded file', seq: ca.map(r => r.aa).join(''),
      ss: hasSS ? ss : null, ca: coords, atoms: ca.map(r => r.atoms), first: ca[0].num, chain, warning: warning.trim() };
  }
  function parsePDB(text, filename) {
    let id = '', title = '';
    const col = chainCollector(), ranges = [];
    for (const line of text.split(/\r?\n/)) {
      const rec = line.slice(0, 6).trim();
      if (rec === 'HEADER') id = line.slice(62, 66).trim();
      else if (rec === 'TITLE') title += ' ' + line.slice(10).trim();
      else if (rec === 'HELIX') ranges.push([line.slice(38, 40).trim() === '5' ? 'G' : 'H', line[19], parseInt(line.slice(21, 25)), parseInt(line.slice(33, 37))]);
      else if (rec === 'SHEET') ranges.push(['E', line[21], parseInt(line.slice(22, 26)), parseInt(line.slice(33, 37))]);
      else if (rec === 'ENDMDL' && col.chains.size) break;
      else if (rec === 'ATOM' || rec === 'HETATM') {
        const resn = line.slice(17, 20).trim(), alt = line[16];
        if (!THREE3[resn] || (alt !== ' ' && alt !== 'A')) continue;
        col.add(line[21], line.slice(22, 27), parseInt(line.slice(22, 26)), resn, line.slice(12, 16).trim(), line.slice(76, 78).trim(),
          parseFloat(line.slice(30, 38)), parseFloat(line.slice(38, 46)), parseFloat(line.slice(46, 54)));
      }
    }
    return buildUploaded({ chains: col.chains, id, title, ranges, filename, format: 'PDB' });
  }
  // mmCIF / PDBx: tokens are bare words, 'quoted' or "quoted" strings, or ;-delimited text blocks; loop_ starts a table.
  function cifTables(text) {
    const tok = [], lines = text.split(/\r?\n/);
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      if (line[0] === ';') {
        const buf = [line.slice(1)];
        while (++li < lines.length && lines[li][0] !== ';') buf.push(lines[li]);
        tok.push(buf.join('\n')); continue;
      }
      let i = 0;
      while (i < line.length) {
        const c = line[i];
        if (c === ' ' || c === '\t') { i++; continue; }
        if (c === '#') break;
        if (c === "'" || c === '"') {
          let j = i + 1;
          while (j < line.length && !(line[j] === c && (j + 1 === line.length || line[j + 1] === ' ' || line[j + 1] === '\t'))) j++;
          tok.push(line.slice(i + 1, j)); i = j + 1; continue;
        }
        let j = i; while (j < line.length && line[j] !== ' ' && line[j] !== '\t') j++;
        tok.push(line.slice(i, j)); i = j;
      }
    }
    const T = {};
    const table = cat => T[cat] || (T[cat] = { cols: [], rows: [] });
    for (let k = 0; k < tok.length;) {
      const t = tok[k];
      if (t === 'loop_') {
        k++;
        const tags = [];
        while (k < tok.length && tok[k][0] === '_') tags.push(tok[k++]);
        if (!tags.length) continue;
        const cat = tags[0].slice(1, tags[0].indexOf('.')), tb = table(cat);
        tb.cols = tags.map(g => g.slice(g.indexOf('.') + 1));
        while (k < tok.length && tok[k] !== 'loop_' && !(tok[k][0] === '_' && tok[k].includes('.')) && !tok[k].startsWith('data_')) {
          const row = tok.slice(k, k + tags.length); k += tags.length;
          if (row.length === tags.length) tb.rows.push(row);
        }
      } else if (t[0] === '_' && t.includes('.')) {
        const cat = t.slice(1, t.indexOf('.')), tb = table(cat), col = t.slice(t.indexOf('.') + 1);
        if (!tb.rows.length) tb.rows.push([]);
        tb.cols.push(col); tb.rows[0].push(tok[k + 1]); k += 2;
      } else k++;
    }
    return T;
  }
  function cifCol(tb, ...names) { if (!tb) return -1; for (const n of names) { const i = tb.cols.indexOf(n); if (i >= 0) return i; } return -1; }
  function parseCIF(text, filename) {
    const T = cifTables(text), as = T.atom_site;
    if (!as || !as.rows.length) {
      const cc = T.chem_comp;
      if (cc && cc.rows.length) {
        const v = n => { const i = cifCol(cc, n); return i >= 0 ? cc.rows[0][i] : ''; };
        throw new Error(`${v('id') || filename} is a chemical-component file for ${v('name') ? v('name').toLowerCase() : 'a small molecule'} (${(v('type') || 'ligand').toLowerCase()}), not a protein structure, so there is no chain to fold. On rcsb.org, open a protein entry and use Download Files → PDBx/mmCIF or PDB format.`);
      }
      throw new Error('This mmCIF file has no _atom_site coordinates.');
    }
    const ix = n => cifCol(as, ...n);
    const cAtom = ix(['auth_atom_id', 'label_atom_id']), cRes = ix(['auth_comp_id', 'label_comp_id']), cCh = ix(['auth_asym_id', 'label_asym_id']),
      cSeq = ix(['auth_seq_id', 'label_seq_id']), cIns = ix(['pdbx_PDB_ins_code']), cAlt = ix(['label_alt_id']), cEl = ix(['type_symbol']),
      cX = ix(['Cartn_x']), cY = ix(['Cartn_y']), cZ = ix(['Cartn_z']), cModel = ix(['pdbx_PDB_model_num']), cGroup = ix(['group_PDB']);
    if (cAtom < 0 || cRes < 0 || cX < 0) throw new Error('This mmCIF file lacks atom names or Cartesian coordinates.');
    const col = chainCollector();
    let model = null;
    for (const r of as.rows) {
      if (cGroup >= 0 && r[cGroup] !== 'ATOM' && r[cGroup] !== 'HETATM') continue;
      if (cModel >= 0) { if (model === null) model = r[cModel]; else if (r[cModel] !== model) continue; }
      const resn = r[cRes], alt = cAlt >= 0 ? r[cAlt] : '.';
      if (!THREE3[resn] || (alt !== '.' && alt !== '?' && alt !== 'A')) continue;
      const ins = cIns >= 0 && r[cIns] !== '?' && r[cIns] !== '.' ? r[cIns] : '';
      col.add(cCh >= 0 ? r[cCh] : 'A', r[cSeq] + ins, parseInt(r[cSeq]), resn, r[cAtom].replace(/^"|"$/g, ''), cEl >= 0 ? r[cEl] : '', +r[cX], +r[cY], +r[cZ]);
    }
    const ranges = [];
    const sc = T.struct_conf;
    if (sc) {
      const t = cifCol(sc, 'conf_type_id'), ch = cifCol(sc, 'beg_auth_asym_id', 'beg_label_asym_id'), a = cifCol(sc, 'beg_auth_seq_id', 'beg_label_seq_id'),
        b = cifCol(sc, 'end_auth_seq_id', 'end_label_seq_id'), cls = cifCol(sc, 'pdbx_PDB_helix_class');
      for (const r of sc.rows) if (t < 0 || /^HELX/.test(r[t])) ranges.push([cls >= 0 && r[cls] === '5' ? 'G' : 'H', r[ch], parseInt(r[a]), parseInt(r[b])]);
    }
    const sh = T.struct_sheet_range;
    if (sh) {
      const ch = cifCol(sh, 'beg_auth_asym_id', 'beg_label_asym_id'), a = cifCol(sh, 'beg_auth_seq_id', 'beg_label_seq_id'), b = cifCol(sh, 'end_auth_seq_id', 'end_label_seq_id');
      for (const r of sh.rows) ranges.push(['E', r[ch], parseInt(r[a]), parseInt(r[b])]);
    }
    const one = (cat, n) => { const tb = T[cat], i = cifCol(tb, n); return i >= 0 && tb.rows.length ? tb.rows[0][i] : ''; };
    const title = one('struct', 'title');
    return buildUploaded({ chains: col.chains, id: one('entry', 'id'), title: title === '?' ? '' : title, ranges, filename, format: 'mmCIF' });
  }
  function parseStructure(text, filename) {
    const isCif = /\.cif$/i.test(filename.replace(/\.gz$/i, '')) || /^\s*data_/m.test(text.slice(0, 2000)) && !/^(ATOM|HETATM|HEADER)/m.test(text.slice(0, 20000));
    return isCif ? parseCIF(text, filename) : parsePDB(text, filename);
  }
  async function readStructureFile(file) {
    if (/\.gz$/i.test(file.name)) {
      if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot open .gz files; unzip it first.');
      return new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text();
    }
    return file.text();
  }

  // ---------- redraw scheduling ----------
  const dirty = {
    viewer: true, charts: true, score: true, evo: true,
    all() { this.viewer = this.charts = this.score = this.evo = true; },
  };
  let lastEvo = 0, lastUi = 0, lastCharts = 0, lastScore = 0, lastFrameAt = 0, lastTick = performance.now(), lastAtoms = null;
  function tick(now) {
    const dt = now - lastTick; lastTick = now;
    const run = state.run;
    if (run && run.runner instanceof LocalRunner) run.runner.tick(state.watch ? 2 : 12);
    if (state.replaying && run) {
      const rep = run.reps[state.focus];
      if (now - replayAt > 40) {
        replayAt = now;
        state.scrub = (state.scrub || 0) + 1;
        if (state.scrub >= rep.frames.length - 1) { state.scrub = rep.frames.length - 1; state.replaying = false; setReplayIcon(); }
        dirty.all();
      }
    }
    const av = viewer ? atomsVisible() : false;
    if (av !== lastAtoms) { lastAtoms = av; dirty.viewer = true; }
    if (dirty.viewer) { updateViewer(); dirty.viewer = false; }
    if (viewer) viewer.tick(dt);
    if (now - lastUi > 120) { updateReadout(); if (evoWait) renderControls(); lastUi = now; }
    if (dirty.evo && state.tab === 'evo' && now - lastEvo > 150) { renderEvo(); dirty.evo = false; lastEvo = now; }
    const busy = run && run.status === 'running';
    if (dirty.charts && now - lastCharts > (busy ? 250 : 0)) {
      if (state.tab === 'traj') drawTraj();
      else if (state.tab === 'land') drawFunnel();
      dirty.charts = false; lastCharts = now; dirty.score = true;
    }
    if (dirty.score && now - lastScore > (busy ? 500 : 0)) {
      renderScore();
      if (state.tab === 'res') drawTracks();
      else if (state.tab === 'cmap') drawCmap();
      else if (state.tab === 'reps') renderReplicas();
      else if (state.tab === 'traj') drawTraj();
      else if (state.tab === 'land') drawFunnel();
      dirty.score = false; lastScore = now;
    }
  }
  function frame(now) { lastFrameAt = performance.now(); tick(now); requestAnimationFrame(frame); }
  // animation frames stop when the page is hidden; keep a long run moving on timers
  function fallbackLoop() {
    const stalled = performance.now() - lastFrameAt > 300;
    if (stalled) tick(performance.now());
    setTimeout(fallbackLoop, stalled ? 16 : 300);
  }

  // ---------- boot ----------
  readTheme();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readTheme);
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => dirty.all());
  new ResizeObserver(() => dirty.all()).observe(document.querySelector('main'));
  initViewer();
  bindControls();
  renderProteins(); renderMethods(); buildSliders(); renderHistory(); renderMethodPanel(); renderExamples();
  newRun();
  startRun();
  requestAnimationFrame(frame);
  setTimeout(fallbackLoop, 300);
})();
