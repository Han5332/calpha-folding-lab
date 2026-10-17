/* Runs the evolution pipeline off the main thread: filter the MSA, coevolution statistics, then the network. */
const v = self.location.search;
importScripts('kb.js' + v, 'evo.js' + v, 'evo-model.js' + v);
let model = null;

onmessage = e => {
  const m = e.data;
  if (m.type !== 'run') return;
  const say = (stage, frac) => postMessage({ type: 'progress', id: m.id, stage, frac });
  try {
    const t0 = performance.now();
    let rows = m.rows, format = m.format || 'built-in', dropped = 0;
    if (m.text != null) {
      say('align', 0);
      const a = Evo.parseAlignment(m.text, m.query);
      rows = a.rows; format = a.format; dropped = a.dropped;
    }
    say('filter', 0);
    const msa = Evo.process(m.query, rows);
    const t1 = performance.now();
    say('dca', 0);
    const feat = Evo.features(msa, f => say('dca', f));
    const t2 = performance.now();
    say('net', 0);
    if (!model) model = Evo.loadModel(self.FOLD_EVO_MODEL);
    const pred = Evo.predict(model, feat, f => say('net', f));
    const t3 = performance.now();
    postMessage({ type: 'done', id: m.id, result: {
      format, dropped, input: rows.length, N: msa.N, L: msa.L, X: msa.X, neff: feat.neff,
      dca: Float32Array.from(feat.dca), mi: Float32Array.from(feat.mi), conservation: feat.conservation,
      ss: pred.ss, dist: pred.dist, ms: { filter: t1 - t0, dca: t2 - t1, net: t3 - t2 },
    } });
  } catch (err) {
    postMessage({ type: 'error', id: m.id, message: err.message || String(err) });
  }
};
