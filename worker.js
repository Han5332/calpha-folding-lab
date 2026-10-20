/* One folding replica per worker: Langevin dynamics with an annealing schedule, reporting sampled frames. */
const v = self.location.search;   // lets a page bust caches with worker.js?v=…
importScripts('kb.js' + v, 'engine.js' + v);

let job = null, sim = null, running = false, hopper = null, lastLive = 0;
// Yield between chunks with a message to ourselves: timers get throttled in background tabs, messages do not.
const yieldChannel = new MessageChannel();
yieldChannel.port1.onmessage = () => loop();

function makeSystem(j) {
  return Fold.createSystem({
    seq: j.seq, model: j.model, params: j.params,
    native: j.native ? Float64Array.from(j.native) : null,
    ssWeights: j.ssWeights,
    local: j.local ? { theta: Float32Array.from(j.local.theta), phi: Float32Array.from(j.local.phi) } : null,
    restraints: j.restraints || null,
  });
}
function temperature(step) {
  return Fold.scheduleT(step, job.steps, job.tHi, job.tLo, job.anchor);
}
function report() {
  const x = Float32Array.from(sim.x);
  postMessage({ type: 'sample', replica: job.replica, step: sim.step, T: temperature(sim.step), E: sim.energy, x }, [x.buffer]);
}
function loop() {
  if (!running) return;
  const t0 = performance.now(), budget = job.watch ? 3 : 40;
  let done = 0;
  while (sim.step < job.steps) {
    const toSample = job.every - (sim.step % job.every);
    const chunk = Math.min(job.watch ? 25 : 500, toSample, job.steps - sim.step);
    if (sim.sys.rest) Fold.restraintStage(sim.sys, sim.step / job.steps);
    sim.run(chunk, temperature(sim.step));
    done += chunk;
    if (sim.step % job.every === 0) {
      report();
      // stalled? try a Monte Carlo restart (it runs a short re-anneal itself, so the step count jumps)
      const ev = hopper && hopper.observe(temperature(sim.step));
      if (ev) { postMessage({ type: 'hop', replica: job.replica, ev }); report(); }
    }
    if (job.watch && done >= 50) break;
    if (performance.now() - t0 > budget) break;
  }
  // a light display-only frame ~10× a second, however far apart the saved samples are
  const now = performance.now();
  if (now - lastLive > 100 && sim.step % job.every !== 0) {
    lastLive = now;
    const x = Float32Array.from(sim.x);
    postMessage({ type: 'live', replica: job.replica, step: sim.step, T: temperature(sim.step), E: sim.energy, x }, [x.buffer]);
  }
  if (sim.step >= job.steps) {
    running = false;
    postMessage({ type: 'done', replica: job.replica, step: sim.step });
    return;
  }
  if (job.watch) setTimeout(loop, 16);
  else yieldChannel.port2.postMessage(0);
}

onmessage = e => {
  const m = e.data;
  if (m.type === 'local') {   // the fragment search for a long chain, off the page's main thread
    postMessage({ type: 'local', lm: Fold.buildLocalModel(m.seq, { ss: m.ss }) });
  } else if (m.type === 'start') {
    job = m.job;
    sim = new Fold.Simulation(makeSystem(job), { seed: job.seed, dt: 0.01, gamma: 0.5 });
    hopper = job.hopSteps ? new Fold.Hopper(sim, { steps: job.steps, every: job.every, frags: job.frags, seed: job.seed, patienceSteps: job.hopSteps, severity: job.hopSeverity }) : null;
    report();
    running = true;
    loop();
  } else if (m.type === 'pause') {
    running = false;
  } else if (m.type === 'resume') {
    if (!running && sim && sim.step < job.steps) { running = true; loop(); }
  } else if (m.type === 'reseed') {
    // "Restart stray replicas": carry on from another replica's structure with a fresh random stream, same step count
    if (!sim) return;
    const step = sim.step;
    sim = new Fold.Simulation(sim.sys, { seed: m.seed, x: Float64Array.from(m.x), dt: 0.01, gamma: 0.5 });
    sim.step = step;
    if (hopper) hopper = new Fold.Hopper(sim, { steps: job.steps, every: job.every, frags: job.frags, seed: m.seed, patienceSteps: job.hopSteps, severity: job.hopSeverity });
  } else if (m.type === 'update') {
    // lengthening the run: keep the temperature continuous from here (see Fold.scheduleT)
    if (m.job.steps && sim && m.job.steps !== job.steps) job.anchor = { step: sim.step, T: temperature(sim.step) };
    Object.assign(job, m.job);
    if (hopper) { hopper.steps = job.steps; if (m.job.every) hopper.setSampling(job.every); }
    if ('hopSteps' in m.job) {
      if (!job.hopSteps) hopper = null;
      else if (hopper) hopper.setPatience(job.hopSteps);
      else if (sim) hopper = new Fold.Hopper(sim, { steps: job.steps, every: job.every, frags: job.frags, seed: job.seed, patienceSteps: job.hopSteps, severity: job.hopSeverity });
    }
    if (hopper && m.job.hopSeverity) hopper.severity = m.job.hopSeverity;
    if (m.rebuild) sim.setSystem(makeSystem(job));
    if (sim.step >= job.steps && running) { running = false; postMessage({ type: 'done', replica: job.replica, step: sim.step }); }
  }
};
