// One v8 folding trajectory for the website, with live measurements per frame. Frames are dense where the chain changes
// fastest: every 600 steps up to 60k, every 4,200 up to 400k, then every 10k (about 260 frames).
// usage: jsc prelude natives kb engine evo evo-msa evo-model traj.js -- <proteinIndex> <seed> > trajectory.json
var A = arguments, pi = +A[0], seed = +A[1], F = Fold, p = FOLD_NATIVES[pi], n = p.seq.length, X = Float64Array.from(p.ca);
var steps = 1.2e6, tHi = 2.0, tLo = 0.4, chunk = 200;
function due(st) { return st <= 60000 ? st % 600 === 0 : st <= 400000 ? (st - 60000) % 4200 === 0 : (st - 400000) % 10000 === 0; }
var m = FOLD_MSA[p.id], rows = []; for (var k = 1; k < m.n; k++) rows.push(m.rows.substr(k * n, n));
var pred = Evo.predict(Evo.loadModel(FOLD_EVO_MODEL), Evo.features(Evo.process(p.seq, rows)));
var ai = Evo.ssLabels(pred), lm = F.buildLocalModel(p.seq, { ss: ai }), rest = Evo.restraints(pred, { minP: 0.3, minSep: 3, ref: 'dfire' });
var sys = F.createSystem({ seq: p.seq, model: 'kb', native: X, ssWeights: lm.probs, local: lm.local, params: { evo: 1, hbond: 1 }, restraints: rest });
var sim = new F.Simulation(sys, { seed: seed, dt: 0.01, gamma: 0.5 }), nc = F.contacts(X), frames = [];
function snap(T) {
  var x = Float64Array.from(sim.x), c = F.compare(x, X, nc);
  frames.push({ step: sim.step, T: +T.toFixed(3), E: +sim.energy.toFixed(1), rmsd: +c.rmsd.toFixed(2), tm: +c.tm.toFixed(3), q: +c.q.toFixed(3), rg: +F.radiusOfGyration(x).toFixed(2), ss: F.assignSS(x).join ? F.assignSS(x).join('') : F.assignSS(x), x: Array.from(x) });
}
snap(tHi);
while (sim.step < steps) {
  var fr = Math.min(1, sim.step / (0.85 * steps)), T = tHi * Math.pow(tLo / tHi, fr);
  F.restraintStage(sys, sim.step / steps);
  sim.run(chunk, T);
  if (due(sim.step)) snap(T);
}
var last = frames[frames.length - 1];
print(JSON.stringify({ id: p.id, seed: seed, n: n, seq: p.seq, ssNative: p.ss, native: Array.from(X), aiSS: ai, restraints: rest.I.length, final: { rmsd: last.rmsd, tm: last.tm, q: last.q }, frames: frames }));
