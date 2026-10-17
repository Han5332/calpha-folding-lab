/*
 * Molecular viewer for Cα Folding Lab (three.js r128).
 * Representations: cartoon (helix ribbons, strand arrows, coil tubes), ball-and-stick, spacefill, Cα trace.
 * In cartoon mode the atoms fade in as you zoom close, the way PyMOL or ChimeraX figures show side chains.
 * Two molecules: the model and the PDB structure, superposed in one view or side by side.
 */
(function (root) {
  'use strict';
  const THREE = root.THREE;

  const ELEMENT = { C: [0.56, 0.56, 0.56], N: [0.19, 0.31, 0.97], O: [1.0, 0.13, 0.08], S: [1.0, 0.78, 0.2] };
  const VDW = { C: 1.7, N: 1.55, O: 1.52, S: 1.8 };
  const SUB = 8, RING = 12;

  function catmull(p0, p1, p2, p3, t, out, dout) {
    const t2 = t * t, t3 = t2 * t;
    for (let d = 0; d < 3; d++) {
      const a = -p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d], b = 2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d], c = -p0[d] + p2[d];
      out[d] = 0.5 * (2 * p1[d] + c * t + b * t2 + a * t3);
      dout[d] = 0.5 * (c + 2 * b * t + 3 * a * t2);
    }
  }
  const norm3 = v => { const l = Math.hypot(v[0], v[1], v[2]) || 1; v[0] /= l; v[1] /= l; v[2] /= l; return v; };

  // One molecule: cartoon mesh, atoms and bonds as instanced meshes.
  function Molecule(scene, opts) {
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.opts = opts;
    this.cartoonMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.02, side: THREE.DoubleSide });
    this.atomMat = new THREE.MeshStandardMaterial({ roughness: 0.38, metalness: 0.02 });
    this.bondMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.02 });
    this.sphereGeo = new THREE.SphereGeometry(1, 18, 12);
    this.cylGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
    this.n = 0;
  }
  Molecule.prototype.setTopology = function (n, atoms) {
    // atoms: { count, el: [...], res: Int32Array, bonds: Uint32Array }
    this.n = n;
    this.atoms = atoms;
    const M = (n - 1) * SUB + 1;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(M * RING * 3), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(M * RING * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(M * RING * 3), 3));
    const idx = new Uint32Array((M - 1) * RING * 6);
    let q = 0;
    for (let j = 0; j < M - 1; j++) for (let a = 0; a < RING; a++) {
      const v0 = j * RING + a, v1 = j * RING + (a + 1) % RING, v2 = v0 + RING, v3 = v1 + RING;
      idx[q++] = v0; idx[q++] = v2; idx[q++] = v1; idx[q++] = v1; idx[q++] = v2; idx[q++] = v3;
    }
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    if (this.cartoon) { this.group.remove(this.cartoon); this.cartoon.geometry.dispose(); }
    this.cartoon = new THREE.Mesh(geo, this.cartoonMat);
    this.group.add(this.cartoon);
    if (this.spheres) { this.group.remove(this.spheres); this.group.remove(this.bonds); this.spheres.dispose(); this.bonds.dispose(); }
    this.spheres = new THREE.InstancedMesh(this.sphereGeo, this.atomMat, atoms.count);
    this.bonds = new THREE.InstancedMesh(this.cylGeo, this.bondMat, atoms.bonds.length);
    this.spheres.frustumCulled = false; this.bonds.frustumCulled = false;
    this.group.add(this.spheres); this.group.add(this.bonds);
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._v = new THREE.Vector3();
    this._s = new THREE.Vector3(); this._up = new THREE.Vector3(0, 1, 0); this._dir = new THREE.Vector3(); this._c = new THREE.Color();
  };
  // P: Cα (3n), G: carbonyl directions (3n), ss: 'H'/'E'/'C' per residue, cols: [r,g,b] per residue (0–1)
  Molecule.prototype.updateCartoon = function (P, G, ss, cols, style) {
    const n = this.n, M = (n - 1) * SUB + 1, geo = this.cartoon.geometry;
    const pos = geo.attributes.position.array, nor = geo.attributes.normal.array, col = geo.attributes.color.array;
    const trace = style === 'trace';
    // consistent guide directions (carbonyls alternate in strands)
    const g = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      let x = G[3 * i], y = G[3 * i + 1], z = G[3 * i + 2];
      if (i > 0 && x * g[3 * i - 3] + y * g[3 * i - 2] + z * g[3 * i - 1] < 0) { x = -x; y = -y; z = -z; }
      g[3 * i] = x; g[3 * i + 1] = y; g[3 * i + 2] = z;
    }
    const type = ss.split(''), strandEnd = type.map((t, i) => t === 'E' && type[i + 1] !== 'E');
    const pt = i => { const k = Math.max(0, Math.min(n - 1, i)); return [P[3 * k], P[3 * k + 1], P[3 * k + 2]]; };
    const c = [0, 0, 0], dc = [0, 0, 0];
    let j = 0;
    let prevN = null;
    for (let i = 0; i < n; i++) {
      const steps = i < n - 1 ? SUB : 1;
      for (let k = 0; k < steps; k++) {
        const t = i < n - 1 ? k / SUB : 1;
        const s0 = i < n - 1 ? i : n - 2;
        catmull(pt(s0 - 1), pt(s0), pt(s0 + 1), pt(s0 + 2), i < n - 1 ? t : 1, c, dc);
        const T = norm3(dc.slice());
        const a = i < n - 1 ? i : n - 1, b = Math.min(n - 1, a + 1), tt = i < n - 1 ? t : 0;
        let N = [g[3 * a] * (1 - tt) + g[3 * b] * tt, g[3 * a + 1] * (1 - tt) + g[3 * b + 1] * tt, g[3 * a + 2] * (1 - tt) + g[3 * b + 2] * tt];
        const dt = N[0] * T[0] + N[1] * T[1] + N[2] * T[2];
        N = [N[0] - dt * T[0], N[1] - dt * T[1], N[2] - dt * T[2]];
        if (Math.hypot(N[0], N[1], N[2]) < 1e-4) N = prevN ? prevN.slice() : [T[1], -T[0], 0];
        norm3(N);
        if (prevN && N[0] * prevN[0] + N[1] * prevN[1] + N[2] * prevN[2] < 0) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }
        prevN = N;
        const B = [T[1] * N[2] - T[2] * N[1], T[2] * N[0] - T[0] * N[2], T[0] * N[1] - T[1] * N[0]];
        // cross-section by secondary structure
        const r = tt < 0.5 ? a : b;
        let w = 0.3, h = 0.3;
        if (!trace) {
          if (strandEnd[a] && i < n - 1) { w = 1.75 * (1 - tt) + 0.3 * tt; h = 0.34; }
          else if (type[r] === 'H') { w = 1.15; h = 0.26; }
          else if (type[r] === 'E') { w = 1.0; h = 0.34; }
        } else { w = h = 0.42; }
        const cr = cols[r];
        for (let q = 0; q < RING; q++) {
          const ang = (q / RING) * Math.PI * 2, cs = Math.cos(ang), sn = Math.sin(ang), o = 3 * (j * RING + q);
          pos[o] = c[0] + cs * w * N[0] + sn * h * B[0];
          pos[o + 1] = c[1] + cs * w * N[1] + sn * h * B[1];
          pos[o + 2] = c[2] + cs * w * N[2] + sn * h * B[2];
          let nx = cs * h * N[0] + sn * w * B[0], ny = cs * h * N[1] + sn * w * B[1], nz = cs * h * N[2] + sn * w * B[2];
          const l = Math.hypot(nx, ny, nz) || 1;
          nor[o] = nx / l; nor[o + 1] = ny / l; nor[o + 2] = nz / l;
          col[o] = cr[0]; col[o + 1] = cr[1]; col[o + 2] = cr[2];
        }
        j++;
      }
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.normal.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.computeBoundingSphere();
  };
  // pos: atom coordinates (3 × count); resCols: carbon tint per residue; mode: 'stick' | 'spacefill'
  Molecule.prototype.updateAtoms = function (pos, resCols, mode) {
    const A = this.atoms, m = this._m, q = this._q, v = this._v, s = this._s, col = this._c;
    const fill = mode === 'spacefill';
    for (let a = 0; a < A.count; a++) {
      const e = A.el[a], r = fill ? (VDW[e] || 1.7) * 0.92 : (e === 'S' ? 0.42 : e === 'C' ? 0.3 : 0.32);
      v.set(pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]);
      s.set(r, r, r);
      m.compose(v, q.identity(), s);
      this.spheres.setMatrixAt(a, m);
      const base = e === 'C' ? resCols[A.res[a]] : ELEMENT[e] || ELEMENT.C;
      const cc = e === 'C' ? [base[0] * 0.75 + 0.14, base[1] * 0.75 + 0.14, base[2] * 0.75 + 0.14] : base;
      col.setRGB(cc[0], cc[1], cc[2]);
      this.spheres.setColorAt(a, col);
    }
    this.spheres.count = A.count;
    this.spheres.instanceMatrix.needsUpdate = true;
    if (this.spheres.instanceColor) this.spheres.instanceColor.needsUpdate = true;
    // each bond as two half-cylinders, each in its atom's colour
    let k = 0;
    if (!fill) {
      for (let b = 0; b < A.bonds.length; b += 2) {
        const i = A.bonds[b], jj = A.bonds[b + 1];
        const mx = (pos[3 * i] + pos[3 * jj]) / 2, my = (pos[3 * i + 1] + pos[3 * jj + 1]) / 2, mz = (pos[3 * i + 2] + pos[3 * jj + 2]) / 2;
        for (const [a, ex, ey, ez] of [[i, mx, my, mz], [jj, mx, my, mz]]) {
          const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
          this._dir.set(ex - ax, ey - ay, ez - az);
          const len = this._dir.length();
          if (len < 1e-4) continue;
          this._dir.multiplyScalar(1 / len);
          q.setFromUnitVectors(this._up, this._dir);
          v.set((ax + ex) / 2, (ay + ey) / 2, (az + ez) / 2);
          s.set(0.14, len, 0.14);
          m.compose(v, q, s);
          this.bonds.setMatrixAt(k, m);
          const e = A.el[a], base = e === 'C' ? resCols[A.res[a]] : ELEMENT[e] || ELEMENT.C;
          const cc = e === 'C' ? [base[0] * 0.75 + 0.14, base[1] * 0.75 + 0.14, base[2] * 0.75 + 0.14] : base;
          col.setRGB(cc[0], cc[1], cc[2]);
          this.bonds.setColorAt(k, col);
          k++;
        }
      }
    }
    this.bonds.count = k;
    this.bonds.instanceMatrix.needsUpdate = true;
    if (this.bonds.instanceColor) this.bonds.instanceColor.needsUpdate = true;
  };
  Molecule.prototype.setVisible = function (cartoon, atoms) {
    if (!this.cartoon) return;
    this.cartoon.visible = cartoon;
    this.spheres.visible = atoms;
    this.bonds.visible = atoms;
  };
  Molecule.prototype.setGhost = function (on) {
    if (this.ghosted === on) return;
    this.ghosted = on;
    for (const mat of [this.cartoonMat, this.atomMat, this.bondMat]) {
      mat.transparent = on; mat.opacity = on ? 0.42 : 1; mat.depthWrite = !on; mat.needsUpdate = true;
    }
  };

  function createViewer(container, hooks) {
    hooks = hooks || {};
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch (e) {
      return null;
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.touchAction = 'none';
    const camera = new THREE.PerspectiveCamera(30, 1, 0.5, 4000);
    const scenes = [new THREE.Scene(), new THREE.Scene()];
    const pivots = scenes.map(sc => { const p = new THREE.Group(); sc.add(p); return p; });
    for (const sc of scenes) {
      sc.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 0.55));
      const key = new THREE.DirectionalLight(0xffffff, 0.85);
      key.position.set(-0.6, 0.8, 1);
      sc.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.25);
      fill.position.set(0.8, -0.3, 0.6);
      sc.add(fill);
    }
    // molecules: model in scene 0; PDB in scene 0 (overlay) or scene 1 (side by side)
    const model = new Molecule(pivots[0]), pdb = new Molecule(pivots[0]);
    let layout = 'overlay', radius = 20, dist = 60, target = new THREE.Vector3(), spin = false, dirty = true;
    let pdbCenter = new THREE.Vector3();
    const rot = new THREE.Quaternion();
    const homeRot = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.35, 0.5, 0));
    rot.copy(homeRot);
    let bgColor = new THREE.Color(1, 1, 1);

    function setLayout(l) {
      if (l === layout) return;
      layout = l;
      pivots[0].remove(pdb.group); pivots[1].remove(pdb.group);
      (l === 'side' ? pivots[1] : pivots[0]).add(pdb.group);
      dirty = true;
    }
    function size() {
      const w = container.clientWidth, h = container.clientHeight;
      if (!w || !h) return null;
      renderer.setSize(w, h, false);
      return [w, h];
    }
    function render() {
      const wh = size();
      if (!wh) return;
      const [w, h] = wh;
      for (const p of pivots) { p.quaternion.copy(rot); p.position.set(0, 0, 0); }
      for (const p of pivots) p.children.forEach(c => c.position.copy(target).multiplyScalar(-1));
      const halves = layout === 'side' ? 2 : 1;
      camera.aspect = (w / halves) / h;
      // `dist` is tuned for a 1.3:1 view; on narrower (portrait, side-by-side) views back off so the molecule still fits
      const d = dist * Math.max(1, 1.3 / camera.aspect);
      camera.position.set(0, 0, d);
      camera.near = Math.max(0.5, d - radius * 3);
      camera.far = d + radius * 4;
      camera.updateProjectionMatrix();
      for (const sc of scenes) {
        if (!sc.fog) sc.fog = new THREE.Fog(bgColor, 1, 2);
        sc.fog.color.copy(bgColor); sc.fog.near = d - radius * 0.2; sc.fog.far = d + radius * 1.8;
      }
      renderer.setScissorTest(halves === 2);
      for (let k = 0; k < halves; k++) {
        renderer.setViewport(k * w / halves, 0, w / halves, h);
        renderer.setScissor(k * w / halves, 0, w / halves, h);
        renderer.setClearColor(bgColor, 1);
        renderer.render(scenes[k], camera);
      }
      dirty = false;
    }

    // interaction
    const pointers = new Map();
    let pinch0 = 0, dist0 = 0, moved = false, lastHover = -2;
    const el = renderer.domElement;
    el.addEventListener('pointerdown', e => {
      el.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
      moved = false;
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch0 = Math.hypot(a.x - b.x, a.y - b.y) || 1; dist0 = dist; }
    });
    el.addEventListener('pointermove', e => {
      const p = pointers.get(e.pointerId);
      if (!p) { hover(e); return; }
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      if (pointers.size === 1) {
        if (p.button === 2 || p.shift) {
          const s = dist * 0.0012;
          const right = new THREE.Vector3(1, 0, 0).applyQuaternion(rot.clone().invert()), up = new THREE.Vector3(0, 1, 0).applyQuaternion(rot.clone().invert());
          target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
        } else {
          const axis = new THREE.Vector3(dy, dx, 0);
          const ang = axis.length() * 0.006;
          if (ang > 0) rot.premultiply(new THREE.Quaternion().setFromAxisAngle(axis.normalize(), ang));
        }
      }
      p.x = e.clientX; p.y = e.clientY;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        dist = clampDist(dist0 * pinch0 / (Math.hypot(a.x - b.x, a.y - b.y) || 1));
      }
      dirty = true;
      if (hooks.onView) hooks.onView();
    });
    const up = e => { pointers.delete(e.pointerId); };
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('pointerleave', () => { if (hooks.onHover && lastHover !== -1) { lastHover = -1; hooks.onHover(null); } });
    el.addEventListener('contextmenu', e => e.preventDefault());
    el.addEventListener('wheel', e => {
      e.preventDefault();
      dist = clampDist(dist * Math.exp(e.deltaY * 0.0012));
      dirty = true;
      if (hooks.onView) hooks.onView();
    }, { passive: false });
    el.addEventListener('dblclick', e => {
      const h = pick(e);
      if (h) focusResidue(h.mol, h.res);
      else resetView();
    });
    function clampDist(d) { return Math.max(radius * 0.35, Math.min(radius * 9, d)); }

    // screen-space pick of the nearest Cα
    let lastCA = { model: null, pdb: null };
    function project(x, y, z, half) {
      const v = new THREE.Vector3(x - target.x, y - target.y, z - target.z).applyQuaternion(rot);
      v.z -= dist * Math.max(1, 1.3 / camera.aspect);   // same back-off as render()
      const f = 1 / Math.tan((camera.fov * Math.PI / 180) / 2);
      const w = container.clientWidth / (layout === 'side' ? 2 : 1), h = container.clientHeight;
      if (v.z >= -0.1) return null;
      return [half * w + w / 2 + (v.x * f / -v.z) * h / 2, h / 2 - (v.y * f / -v.z) * h / 2, -v.z];
    }
    function pick(e) {
      const r = el.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = null, bd = 16 * 16;
      for (const mol of ['model', 'pdb']) {
        const P = lastCA[mol];
        const m = mol === 'model' ? model : pdb;
        if (!P || !m.group.visible || !m.cartoon || (!m.cartoon.visible && !m.spheres.visible)) continue;
        const half = layout === 'side' && mol === 'pdb' ? 1 : 0;
        for (let i = 0; i < P.length / 3; i++) {
          const s = project(P[3 * i], P[3 * i + 1], P[3 * i + 2], half);
          if (!s) continue;
          const d = (s[0] - mx) ** 2 + (s[1] - my) ** 2;
          if (d < bd) { bd = d; best = { mol, res: i, x: s[0], y: s[1] }; }
        }
      }
      return best;
    }
    function hover(e) {
      if (!hooks.onHover) return;
      const h = pick(e);
      const key = h ? (h.mol === 'pdb' ? 10000 : 0) + h.res : -1;
      if (key !== lastHover) { lastHover = key; hooks.onHover(h, e); }
      else if (h) hooks.onHover(h, e);
    }
    function focusResidue(mol, i) {
      const P = lastCA[mol];
      if (!P) return;
      target.set(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
      dist = clampDist(radius * 0.6);
      dirty = true;
      if (hooks.onView) hooks.onView();
    }
    function resetView() {
      rot.copy(homeRot);
      target.copy(pdbCenter);
      dist = radius * 3.1;
      dirty = true;
      if (hooks.onView) hooks.onView();
    }

    let fitted = false;
    const api = {
      model, pdb,
      setTopology(which, n, atoms) { (which === 'model' ? model : pdb).setTopology(n, atoms); dirty = true; },
      update(which, data) {
        const mol = which === 'model' ? model : pdb;
        lastCA[which] = data.ca;
        mol.updateCartoon(data.ca, data.guides, data.ss, data.colors, data.style);
        if (data.atoms) mol.updateAtoms(data.atoms, data.colors, data.atomMode);
        dirty = true;
      },
      show(which, visible, cartoon, atoms) {
        const mol = which === 'model' ? model : pdb;
        if (!mol.cartoon) { mol.group.visible = false; return; }
        mol.group.visible = visible;
        mol.setVisible(cartoon, atoms);
        dirty = true;
      },
      ghost(which, on) { (which === 'model' ? model : pdb).setGhost(on); dirty = true; },
      setLayout,
      frame(center, r, force) {
        radius = r;
        pdbCenter.set(center[0], center[1], center[2]);
        if (!fitted || force) { target.copy(pdbCenter); dist = r * 3.1; fitted = true; }
        dirty = true;
      },
      setBackground(rgb) { bgColor = new THREE.Color(rgb[0], rgb[1], rgb[2]); dirty = true; },
      zoomFraction() { return dist / radius; },
      setSpin(on) { spin = on; },
      resetView, focusResidue,
      zoomBy(f) { dist = clampDist(dist * f); dirty = true; if (hooks.onView) hooks.onView(); },
      markDirty() { dirty = true; },
      tick(dt) {
        if (spin && !pointers.size) { rot.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dt * 0.0004)); dirty = true; }
        if (dirty) render();
      },
      canvas: el,
    };
    new ResizeObserver(() => { dirty = true; }).observe(container);
    return api;
  }

  root.FoldViewer = { createViewer };
})(window);
