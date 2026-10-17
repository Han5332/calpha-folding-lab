/* Runs the evolution pipeline off the main thread: filter the MSA, coevolution statistics, then the network. */
const v = self.location.search;
importScripts('kb.js' + v, 'evo.js' + v, 'evo-model.js' + v);
let model = null, featTried = false;

// Precomputed results for the large built-in proteins (evo-feat.js, loaded on the first built-in request). Used only
// when the query, the alignment rows and the weights match the stored fingerprints; anything else is computed live.
async function inflate(b64) {
  const b = atob(b64), u8 = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u8[i] = b.charCodeAt(i);
  const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
const STORED_MIN_LEN = 100;   // only large proteins are stored; smaller ones take seconds live, so skip the 1.3 MB file
async function stored(m, rows) {
  if (m.text != null || m.query.length < STORED_MIN_LEN || typeof DecompressionStream === 'undefined') return null;
  if (!featTried) { featTried = true; try { importScripts('evo-feat.js' + v); } catch (e) { /* not shipped: compute live */ } }
  const table = self.FOLD_EVO_FEAT, entry = table && Object.values(table).find(e => e.seq === m.query);
  if (!entry) return null;
  try {
    const raw = {};
    for (const k of Object.keys(entry.data)) raw[k] = await inflate(entry.data[k]);
    return Evo.fromStored(entry, raw, m.query, rows, self.FOLD_EVO_MODEL);
  } catch (e) { return null; }
}

onmessage = async e => {
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
    const pre = await stored(m, rows);
    if (pre) {
      postMessage({ type: 'done', id: m.id, result: {
        format, dropped, input: rows.length, N: msa.N, L: msa.L, X: msa.X, neff: pre.neff,
        dca: pre.dca, mi: pre.mi, conservation: pre.conservation, ss: pre.ss, dist: pre.dist,
        ms: { filter: t1 - t0, dca: 0, net: performance.now() - t1, stored: true },
      } });
      return;
    }
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
