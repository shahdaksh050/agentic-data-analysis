/**
 * The plate — a live technical drawing of the seven-stage RLM pipeline.
 *
 * The console is a draughtsman's sheet (DESIGN.md), so the pipeline is drawn
 * rather than lit: hairline cages around solids, flat paper fills, no gloss and
 * no glow. Two pens ink the drawing — blue for what the run measured, red for
 * where it failed. Stages the run never reached stay in pencil.
 *
 * The drawing also encodes real control flow: stage 5 arcs back to stage 3
 * (iterative refinement re-enters tool execution) and stage 6 carries two
 * satellites (the recursive RLM sub-calls).
 *
 * State arrives from Python on `window.__PIPELINE_STATE__`; see ui/pipeline_3d.py.
 */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const STATE = window.__PIPELINE_STATE__;
const C = STATE.palette;
const STAGES = STATE.stages;

const host = document.getElementById("scene");
const elTitle = document.getElementById("readout-title");
const elBody = document.getElementById("readout-body");

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------------------------------------------ *
 * Drawing geometry, in world units
 * ------------------------------------------------------------------ */
const GAP = 2.0; // spacing between stage modules along the rail
const CAGE = 0.92; // wireframe module edge length
const REFINE_FROM = 4; // stage 5 (0-indexed) …
const REFINE_TO = 2; // … loops back into stage 3
const RLM_INDEX = 5; // stage 6 spawns the recursive satellites
const STROKE_SAMPLES = 24; // points per drawn line, so a pen can travel it

/** World X for a stage, centring the rail on the origin. */
const xFor = (i) => (i - (STAGES.length - 1) / 2) * GAP;

const isReached = (s) => s === "done" || s === "active" || s === "error";

/** The ink a stage is drawn in once the run has reached it. */
function inkFor(status) {
  if (status === "error") return C.risk;
  if (status === "done" || status === "active") return C.pen;
  return C.graphite;
}

/** Line weight is unavailable in WebGL, so pressure is carried by opacity. */
const CAGE_OPACITY = {
  pending: 0.4,
  skipped: 0.22,
  done: 0.75,
  active: 1,
  error: 0.95,
};

/** Reached stages get inked in on paper; the rest stay as a pencil outline. */
const FILL_OPACITY = {
  pending: 0,
  skipped: 0,
  done: 0.92,
  active: 1,
  error: 0.55,
};

/** WebGL missing: the drawing is decoration, the stage list is not. */
function fallback(message) {
  host.innerHTML = "";
  const p = document.createElement("p");
  p.className = "fallback";
  p.textContent = message;
  host.appendChild(p);
}

/* ------------------------------------------------------------------ *
 * Renderer, scene, camera
 * ------------------------------------------------------------------ */
let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
} catch (err) {
  fallback("This browser can't render WebGL. Open the stage ledger for the same information as text.");
  throw err;
}

renderer.setClearColor(0x000000, 0); // blend into the sheet
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);

// No lights: every surface is a flat fill with a drawn edge, which is both the
// look the brief asks for and the cheapest thing the GPU can do.

/** Everything rotatable lives under the drawing, so the entrance can swing it. */
const drawing = new THREE.Group();
scene.add(drawing);

/* ------------------------------------------------------------------ *
 * Shared resources — created once, reused across all seven modules
 * ------------------------------------------------------------------ */
const disposables = [];
const keep = (resource) => {
  disposables.push(resource);
  return resource;
};

const cageGeo = keep(new THREE.EdgesGeometry(new THREE.BoxGeometry(CAGE, CAGE, CAGE)));
const pickGeo = keep(new THREE.BoxGeometry(CAGE * 1.35, CAGE * 1.35, CAGE * 1.35));
const pickMat = keep(
  new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0,
    depthWrite: false,
    colorWrite: false,
  })
);

/**
 * The solid at the heart of each module. The shape is the label: a flat sheet
 * arrives, a crystal reasons over it, a box does the work, and so on.
 */
function coreGeometries(index) {
  switch (index) {
    case 0: // Ingestion — the raw file lands as a flat sheet
      return [[new THREE.BoxGeometry(0.64, 0.07, 0.64), 0, 0, 0]];
    case 1: // Reasoning — a crystal of thought
      return [[new THREE.OctahedronGeometry(0.37), 0, 0, 0]];
    case 2: // Tool execution — solid machinery
      return [[new THREE.BoxGeometry(0.46, 0.46, 0.46), 0, 0, 0]];
    case 3: // Interpretation — the picture comes together
      return [[new THREE.IcosahedronGeometry(0.35, 1), 0, 0, 0]];
    case 4: // Refinement — the loop, echoing the arc overhead
      return [[new THREE.TorusGeometry(0.29, 0.08, 8, 22), 0, 0, 0]];
    case 5: // RLM decomposition — one problem splits into four
      return [-0.16, 0.16].flatMap((x) =>
        [-0.16, 0.16].map((z) => [new THREE.BoxGeometry(0.22, 0.22, 0.22), x, 0, z])
      );
    default: // Report — the finished document, two sheets stacked
      return [
        [new THREE.BoxGeometry(0.6, 0.05, 0.5), 0, 0.08, 0],
        [new THREE.BoxGeometry(0.6, 0.05, 0.5), 0, -0.06, 0],
      ];
  }
}

/** A drawn solid: a flat paper fill with its edges inked over the top. */
function buildCore(index, fillMat, edgeMat) {
  const group = new THREE.Group();
  for (const [geometry, x, y, z] of coreGeometries(index)) {
    keep(geometry);
    const fill = new THREE.Mesh(geometry, fillMat);
    fill.position.set(x, y, z);
    group.add(fill);

    const edges = new THREE.LineSegments(keep(new THREE.EdgesGeometry(geometry)), edgeMat);
    edges.position.set(x, y, z);
    group.add(edges);
  }
  return group;
}

/* ------------------------------------------------------------------ *
 * Modules
 * ------------------------------------------------------------------ */
const nodes = STAGES.map((stage, i) => {
  const group = new THREE.Group();
  group.position.x = xFor(i);

  const cageMat = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(C.graphite),
      transparent: true,
      opacity: CAGE_OPACITY.pending,
    })
  );
  group.add(new THREE.LineSegments(cageGeo, cageMat));

  // Fill sits fractionally behind its own edges, so the hairlines always win.
  const fillMat = keep(
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(C.sheet),
      transparent: true,
      opacity: 0,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
  );
  const edgeMat = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(C.graphite),
      transparent: true,
      opacity: 0.5,
    })
  );
  const core = buildCore(i, fillMat, edgeMat);
  group.add(core);

  // Invisible but raycastable, so hovering forgives a near miss.
  const pick = new THREE.Mesh(pickGeo, pickMat);
  pick.userData.index = i;
  group.add(pick);

  drawing.add(group);
  return { stage, group, cageMat, core, fillMat, edgeMat, pick };
});

/* ------------------------------------------------------------------ *
 * Rail, refinement arc, RLM satellites — structure that carries meaning.
 *
 * Every line is sampled into many points so `setDrawRange` can walk a pen
 * along it during the entrance.
 * ------------------------------------------------------------------ */
function makeStroke(points, color, opacity) {
  const material = keep(
    new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity })
  );
  const geometry = keep(new THREE.BufferGeometry().setFromPoints(points));
  const line = new THREE.Line(geometry, material);
  drawing.add(line);
  return { material, geometry, count: points.length };
}

/** Straight run between two points, sampled so it can be drawn on. */
function straight(from, to) {
  return new THREE.LineCurve3(from, to).getPoints(STROKE_SAMPLES);
}

/** One segment per hop, so each inks only once the data has crossed it. */
const links = nodes.slice(0, -1).map((_, i) =>
  makeStroke(
    straight(
      new THREE.Vector3(xFor(i) + CAGE / 2, 0, 0),
      new THREE.Vector3(xFor(i + 1) - CAGE / 2, 0, 0)
    ),
    C.graphite,
    0.42
  )
);

/** Stage 5 feeds back into stage 3 — the controller's iteration loop. */
const refineArc = makeStroke(
  new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(xFor(REFINE_FROM), CAGE / 2, 0),
    new THREE.Vector3((xFor(REFINE_FROM) + xFor(REFINE_TO)) / 2, 2.5, 0),
    new THREE.Vector3(xFor(REFINE_TO), CAGE / 2, 0)
  ).getPoints(56),
  C.graphite,
  0.34
);

/** Stage 6 spawns recursive sub-calls, shown as two satellites off the rail. */
const satellites = [-1, 1].map((side) => {
  const position = new THREE.Vector3(xFor(RLM_INDEX), 0.15, side * 1.25);
  const geometry = keep(new THREE.BoxGeometry(0.2, 0.2, 0.2));

  const fillMat = keep(
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(C.sheet),
      transparent: true,
      opacity: 0,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
  );
  const edgeMat = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(C.graphite),
      transparent: true,
      opacity: 0.5,
    })
  );

  const fill = new THREE.Mesh(geometry, fillMat);
  fill.position.copy(position);
  drawing.add(fill);

  const edges = new THREE.LineSegments(keep(new THREE.EdgesGeometry(geometry)), edgeMat);
  edges.position.copy(position);
  drawing.add(edges);

  const tether = makeStroke(
    straight(new THREE.Vector3(xFor(RLM_INDEX), 0, side * (CAGE / 2)), position.clone()),
    C.graphite,
    0.3
  );
  return { fillMat, edgeMat, tether };
});

/** The sheet's own quadrille, carried into the drawing as a bench plane. */
const grid = new THREE.GridHelper(20, 20, new THREE.Color(C.ink), new THREE.Color(C.ink));
grid.position.y = -1.35;
grid.material.transparent = true;
grid.material.opacity = 0.065;
keep(grid.geometry);
keep(grid.material);
drawing.add(grid);

/* ------------------------------------------------------------------ *
 * Payload — the dataset itself, as a lattice of cells riding the rail
 * ------------------------------------------------------------------ */
const CELLS = { x: 2, y: 3, z: 4 };
const CELL_COUNT = CELLS.x * CELLS.y * CELLS.z;
const CELL_STEP = 0.13;

const payload = new THREE.InstancedMesh(
  keep(new THREE.BoxGeometry(0.075, 0.075, 0.075)),
  keep(new THREE.MeshBasicMaterial({ color: new THREE.Color(C.pen) })),
  CELL_COUNT
);
payload.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
payload.visible = false;
drawing.add(payload);

/** Home offset of every cell in the lattice, reused each frame. */
const cellHomes = [];
for (let ix = 0; ix < CELLS.x; ix++) {
  for (let iy = 0; iy < CELLS.y; iy++) {
    for (let iz = 0; iz < CELLS.z; iz++) {
      cellHomes.push(
        new THREE.Vector3(
          (ix - (CELLS.x - 1) / 2) * CELL_STEP,
          (iy - (CELLS.y - 1) / 2) * CELL_STEP,
          (iz - (CELLS.z - 1) / 2) * CELL_STEP
        )
      );
    }
  }
}

const cellMatrix = new THREE.Matrix4();

/** `spread` fans the lattice apart in transit and packs it back on arrival. */
function layoutPayload(spread, phase) {
  for (let i = 0; i < CELL_COUNT; i++) {
    const home = cellHomes[i];
    const scale = 1 + spread * 0.9 + Math.sin(phase + i * 0.7) * 0.06;
    cellMatrix.makeTranslation(home.x * scale, home.y * scale, home.z * scale);
    payload.setMatrixAt(i, cellMatrix);
  }
  payload.instanceMatrix.needsUpdate = true;
}
layoutPayload(0, 0);

/* ------------------------------------------------------------------ *
 * Camera framing
 * ------------------------------------------------------------------ */
/**
 * The points the framing has to keep on screen: the eight corners of every
 * stage cage, the apex of the refinement arc, the RLM satellites, and where
 * the payload comes to rest. A crude bounding box would reserve height at the
 * ends of the rail, where nothing is actually drawn.
 */
const fitPoints = [];
STAGES.forEach((_, i) => {
  for (const dx of [-CAGE / 2, CAGE / 2]) {
    for (const dy of [-CAGE / 2, CAGE / 2]) {
      for (const dz of [-CAGE / 2, CAGE / 2]) {
        fitPoints.push(new THREE.Vector3(xFor(i) + dx, dy, dz));
      }
    }
  }
});
fitPoints.push(new THREE.Vector3((xFor(REFINE_FROM) + xFor(REFINE_TO)) / 2, 1.5, 0));
for (const side of [-1, 1]) {
  fitPoints.push(new THREE.Vector3(xFor(RLM_INDEX), 0.28, side * 1.35));
}
fitPoints.push(new THREE.Vector3(xFor(STAGES.length - 1), 0.75, 0));

const FIT_MARGIN = 0.9; // fraction of the frame the drawing may fill

const orbit = { theta: 0.42, phi: 1.16, radius: 12 };
const drag = { theta: 0, phi: 0 };
const parallax = { x: 0, y: 0 };
const smooth = { theta: 0.42, phi: 1.16 };
const probe = new THREE.PerspectiveCamera();
const ndc = new THREE.Vector3();
const camRight = new THREE.Vector3();
const camUp = new THREE.Vector3();
let framed = false;

function placeCamera(cam, radius, theta, phi) {
  cam.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta)
  );
  cam.lookAt(0, 0, 0);
}

/** Projected bounds of the drawing as the probe currently sees it. */
function projectedBounds() {
  probe.updateMatrixWorld(true);
  const b = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  for (const point of fitPoints) {
    ndc.copy(point).project(probe);
    b.minX = Math.min(b.minX, ndc.x);
    b.maxX = Math.max(b.maxX, ndc.x);
    b.minY = Math.min(b.minY, ndc.y);
    b.maxY = Math.max(b.maxY, ndc.y);
  }
  return b;
}

/**
 * Frames the drawing for the current viewport.
 *
 * The rail is long and thin, so viewed broadside it wastes a square frame; as
 * the viewport narrows it is swung into depth instead. The distance is then
 * solved by projecting the drawing and shrinking to fit, and the drawing is
 * slid so that what is actually on screen sits in the middle of it — with a
 * rail seen at an angle, perspective puts the near end much further from the
 * centre than the far one. Only ever runs on resize.
 */
function fitCamera(width, height) {
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();

  orbit.theta = THREE.MathUtils.clamp(
    THREE.MathUtils.mapLinear(camera.aspect, 1.1, 2.6, 0.92, 0.34),
    0.34,
    0.92
  );

  probe.copy(camera);
  drawing.position.set(0, 0, 0);

  let radius = 14;
  let bounds = null;
  for (let pass = 0; pass < 6; pass++) {
    placeCamera(probe, radius, orbit.theta, orbit.phi);
    bounds = projectedBounds();
    const spanX = (bounds.maxX - bounds.minX) / 2;
    const spanY = (bounds.maxY - bounds.minY) / 2;
    radius *= Math.max(spanX, spanY) / FIT_MARGIN;
  }
  orbit.radius = THREE.MathUtils.clamp(radius, 7, 40);

  placeCamera(probe, orbit.radius, orbit.theta, orbit.phi);
  bounds = projectedBounds();
  const halfHeight = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * orbit.radius;
  const halfWidth = halfHeight * camera.aspect;
  camRight.setFromMatrixColumn(probe.matrixWorld, 0);
  camUp.setFromMatrixColumn(probe.matrixWorld, 1);
  drawing.position
    .copy(camRight)
    .multiplyScalar((-(bounds.minX + bounds.maxX) / 2) * halfWidth)
    .addScaledVector(camUp, (-(bounds.minY + bounds.maxY) / 2) * halfHeight);

  // Don't ease into the very first framing — the entrance already has a swing.
  if (!framed) {
    smooth.theta = orbit.theta;
    smooth.phi = orbit.phi;
    framed = true;
  }
}

/** Eases the camera toward its target; reports whether it is still moving. */
function updateCamera() {
  const targetTheta = orbit.theta + drag.theta + parallax.x * 0.12;
  const targetPhi = THREE.MathUtils.clamp(orbit.phi + drag.phi + parallax.y * 0.08, 0.55, 1.7);
  const moving =
    Math.abs(targetTheta - smooth.theta) > 1e-4 || Math.abs(targetPhi - smooth.phi) > 1e-4;

  smooth.theta += (targetTheta - smooth.theta) * 0.12;
  smooth.phi += (targetPhi - smooth.phi) * 0.12;

  placeCamera(camera, orbit.radius, smooth.theta, smooth.phi);
  return moving;
}

/* ------------------------------------------------------------------ *
 * Readout — plain text describing the run, or whatever is hovered
 * ------------------------------------------------------------------ */
const STATUS_WORD = {
  done: "Finished",
  active: "Running",
  error: "Failed",
  skipped: "Skipped",
  pending: "Not run",
};

function defaultReadout() {
  const errored = STAGES.find((s) => s.status === "error");
  const running = STAGES.find((s) => s.status === "active");
  const finished = STAGES.filter((s) => s.status === "done").length;

  if (errored) return ["Run stopped", `Stage ${errored.num} failed: ${errored.name}.`];
  if (running) return ["Running", `Stage ${running.num} of ${STAGES.length} — ${running.name}.`];
  if (finished === STAGES.length) return ["Run complete", `All ${STAGES.length} stages finished.`];
  if (finished > 0) return ["Run finished", `${finished} of ${STAGES.length} stages ran.`];
  return ["Waiting for a dataset", "Upload a file in the sidebar, then start the run."];
}

function setReadout(title, body) {
  // textContent, never innerHTML — stage details come from tool output.
  elTitle.textContent = title;
  elBody.textContent = body;
}
setReadout(...defaultReadout());

/* ------------------------------------------------------------------ *
 * Inking state onto the drawing
 * ------------------------------------------------------------------ */
function ink(i, animated) {
  const { stage, group, cageMat, fillMat, edgeMat } = nodes[i];
  const color = new THREE.Color(inkFor(stage.status));
  const cageOpacity = CAGE_OPACITY[stage.status] ?? CAGE_OPACITY.pending;
  const fillOpacity = FILL_OPACITY[stage.status] ?? 0;
  const edgeOpacity = isReached(stage.status) ? 1 : 0.5;

  if (!animated) {
    cageMat.color.copy(color);
    cageMat.opacity = cageOpacity;
    edgeMat.color.copy(color);
    edgeMat.opacity = edgeOpacity;
    fillMat.opacity = fillOpacity;
    return;
  }

  const tl = gsap.timeline({ onUpdate: invalidate });
  tl.to(
    [cageMat.color, edgeMat.color],
    { r: color.r, g: color.g, b: color.b, duration: 0.4, ease: "power2.out" },
    0
  )
    .to(cageMat, { opacity: cageOpacity, duration: 0.4 }, 0)
    .to(edgeMat, { opacity: edgeOpacity, duration: 0.4 }, 0)
    .to(fillMat, { opacity: fillOpacity, duration: 0.45, ease: "power2.out" }, 0)
    .to(group.scale, { x: 1.1, y: 1.1, z: 1.1, duration: 0.16, ease: "power2.out" }, 0)
    .to(group.scale, { x: 1, y: 1, z: 1, duration: 0.34, ease: "power2.out" }, 0.16);
}

/** Ink the connectors, refinement arc and satellites the run actually used. */
function inkStructure(animated) {
  const reached = STAGES.map((s) => isReached(s.status));

  const paint = (material, color, opacity) => {
    const target = new THREE.Color(color);
    if (!animated) {
      material.color.copy(target);
      material.opacity = opacity;
      return;
    }
    gsap.to(material.color, {
      r: target.r,
      g: target.g,
      b: target.b,
      duration: 0.45,
      onUpdate: invalidate,
    });
    gsap.to(material, { opacity, duration: 0.45, onUpdate: invalidate });
  };

  links.forEach((link, i) => {
    const crossed = reached[i] && reached[i + 1];
    paint(link.material, crossed ? C.pen : C.graphite, crossed ? 0.85 : 0.42);
  });

  const refined = reached[REFINE_FROM];
  paint(refineArc.material, refined ? C.pen : C.graphite, refined ? 0.7 : 0.34);

  const recursed = reached[RLM_INDEX];
  for (const sat of satellites) {
    paint(sat.edgeMat, recursed ? C.pen : C.graphite, recursed ? 1 : 0.5);
    paint(sat.fillMat, C.sheet, recursed ? 0.92 : 0);
    paint(sat.tether.material, recursed ? C.pen : C.graphite, recursed ? 0.65 : 0.3);
  }
}

/* ------------------------------------------------------------------ *
 * Entrance — one orchestrated moment. The sheet is ruled, the rail is
 * struck left to right, the modules are set down, then the dataset runs
 * the rail and inks in the stages the run actually reached.
 * ------------------------------------------------------------------ */
const strokes = [...links, refineArc, ...satellites.map((s) => s.tether)];

/** Reveals a sampled line one vertex at a time, like a pen travelling it. */
function drawStroke(timeline, stroke, at, duration) {
  const walk = { n: 0 };
  stroke.geometry.setDrawRange(0, 0);
  timeline.to(
    walk,
    {
      n: stroke.count,
      duration,
      ease: "none",
      onUpdate: () => {
        stroke.geometry.setDrawRange(0, Math.ceil(walk.n));
        invalidate();
      },
    },
    at
  );
}

const traversal = STAGES.reduce((acc, s, i) => (isReached(s.status) ? [...acc, i] : acc), []);
const activeIndex = STAGES.findIndex((s) => s.status === "active");

const PAYLOAD_REST_Y = 0.62; // clear of the core it parks over

let payloadSpread = 0;
let payloadParked = true;

function finishDrawing() {
  for (const stroke of strokes) stroke.geometry.setDrawRange(0, stroke.count);
}

function play() {
  if (REDUCED_MOTION) {
    finishDrawing();
    nodes.forEach((_, i) => ink(i, false));
    inkStructure(false);
    if (traversal.length) {
      payload.visible = true;
      payload.position.x = xFor(traversal[traversal.length - 1]);
      payload.position.y = PAYLOAD_REST_Y;
    }
    return;
  }

  inkStructure(true);

  const tl = gsap.timeline({ onUpdate: invalidate, onComplete: settle });

  // 1. The sheet is ruled and the drawing swings into its axonometric view.
  tl.from(grid.material, { opacity: 0, duration: 0.7, ease: "power2.out" }, 0).from(
    drawing.rotation,
    { y: -0.45, duration: 1.3, ease: "power3.out" },
    0
  );

  // 2. The rail is struck left to right, one hop at a time.
  links.forEach((link, i) => drawStroke(tl, link, 0.15 + i * 0.07, 0.24));

  // 3. The modules are set down along it.
  tl.from(
    nodes.map((n) => n.group.scale),
    { x: 0.01, y: 0.01, z: 0.01, duration: 0.45, stagger: 0.055, ease: "back.out(2.2)" },
    0.22
  );

  // 4. The control flow that isn't a straight line gets drawn last.
  drawStroke(tl, refineArc, 0.62, 0.45);
  satellites.forEach((sat, i) => drawStroke(tl, sat.tether, 0.72 + i * 0.06, 0.2));

  if (!traversal.length) {
    // Nothing has run: stage 1 breathes, so the sheet reads as live not stalled.
    gsap.to(nodes[0].cageMat, {
      opacity: 0.85,
      duration: 1.4,
      repeat: -1,
      yoyo: true,
      ease: "sine.inOut",
      onUpdate: invalidate,
    });
    return;
  }

  // 5. The dataset enters ahead of stage 1 and hops forward, inking each stage.
  payload.visible = true;
  payload.position.x = xFor(traversal[0]) - GAP;
  payloadParked = false;

  const HOP = 0.34;
  let at = 0.85;

  for (const i of traversal) {
    const transit = { p: 0 };
    tl.to(payload.position, { x: xFor(i), duration: HOP, ease: "power1.inOut" }, at)
      .to(
        transit,
        {
          p: 1,
          duration: HOP,
          onUpdate: () => {
            payloadSpread = Math.sin(transit.p * Math.PI) * 0.55;
          },
        },
        at
      )
      .call(() => ink(i, true), undefined, at + HOP);
    at += HOP + 0.08;
  }

  tl.to(payload.position, { y: PAYLOAD_REST_Y, duration: 0.35, ease: "power2.out" }, at);
}

/** Once the entrance is done, only a live stage justifies burning frames. */
function settle() {
  payloadSpread = 0;
  payloadParked = true;
  setContinuous(activeIndex >= 0);
}

/* ------------------------------------------------------------------ *
 * Interaction — drag or arrow keys to orbit, hover to read a stage
 * ------------------------------------------------------------------ */
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const pickTargets = nodes.map((n) => n.pick);
const canvas = renderer.domElement;

let hovered = -1;
let dragging = false;
let lastPointer = { x: 0, y: 0 };

function setHover(i) {
  if (i === hovered) return;
  hovered = i;

  if (i < 0) {
    setReadout(...defaultReadout());
    canvas.style.cursor = "grab";
    return;
  }
  const stage = STAGES[i];
  const detail = stage.detail ? ` ${stage.detail}` : "";
  setReadout(`${stage.num}. ${stage.name}`, `${STATUS_WORD[stage.status] ?? "Not run"}.${detail}`);
  canvas.style.cursor = "pointer";
}

canvas.addEventListener("pointermove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const nx = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  const ny = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  if (dragging) {
    drag.theta = THREE.MathUtils.clamp(drag.theta - (event.clientX - lastPointer.x) * 0.005, -0.7, 0.7);
    drag.phi = THREE.MathUtils.clamp(drag.phi - (event.clientY - lastPointer.y) * 0.004, -0.3, 0.3);
    lastPointer = { x: event.clientX, y: event.clientY };
    invalidate();
    return;
  }

  parallax.x = nx;
  parallax.y = ny;
  pointer.set(nx, ny);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(pickTargets, false)[0];
  setHover(hit ? hit.object.userData.index : -1);
  invalidate();
});

canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  lastPointer = { x: event.clientX, y: event.clientY };
  canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = "grabbing";
});

function endDrag(event) {
  if (!dragging) return;
  dragging = false;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  canvas.style.cursor = "grab";
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

canvas.addEventListener("pointerleave", () => {
  parallax.x = 0;
  parallax.y = 0;
  setHover(-1);
  invalidate();
});

canvas.style.cursor = "grab";

/* Keyboard: the same orbit as the drag, plus stepping through the stages, so
   the drawing is not mouse-only. The stage ledger in the page carries the same
   text for anyone who skips it. */
const summary = defaultReadout();
canvas.tabIndex = 0;
canvas.setAttribute("role", "img");
canvas.setAttribute(
  "aria-label",
  `Technical drawing of the ${STAGES.length}-stage analysis pipeline. ${summary[0]}. ${summary[1]}`
);

const KEY_STEP = 0.12;
canvas.addEventListener("keydown", (event) => {
  const step = {
    ArrowLeft: () => (drag.theta = THREE.MathUtils.clamp(drag.theta + KEY_STEP, -0.7, 0.7)),
    ArrowRight: () => (drag.theta = THREE.MathUtils.clamp(drag.theta - KEY_STEP, -0.7, 0.7)),
    ArrowUp: () => (drag.phi = THREE.MathUtils.clamp(drag.phi + KEY_STEP * 0.6, -0.3, 0.3)),
    ArrowDown: () => (drag.phi = THREE.MathUtils.clamp(drag.phi - KEY_STEP * 0.6, -0.3, 0.3)),
  };
  if (event.key === "Home") {
    drag.theta = 0;
    drag.phi = 0;
    setHover(-1);
  } else if (step[event.key]) {
    step[event.key]();
  } else if (event.key === "]" || event.key === "[") {
    const next = event.key === "]" ? hovered + 1 : hovered - 1;
    setHover(next < 0 || next >= STAGES.length ? -1 : next);
  } else {
    return;
  }
  event.preventDefault();
  invalidate();
});

canvas.addEventListener("blur", () => setHover(-1));

/* ------------------------------------------------------------------ *
 * Render loop — gated on visibility, and on there being motion at all
 * ------------------------------------------------------------------ */
let onScreen = true;
let pageVisible = !document.hidden;
let continuous = !REDUCED_MOTION;
let dirty = true;

function invalidate() {
  dirty = true;
}

function setContinuous(value) {
  continuous = value && !REDUCED_MOTION;
  invalidate();
}

function resize() {
  const width = host.clientWidth;
  const height = host.clientHeight;
  if (!width || !height) return;
  // Cap the pixel ratio: hairlines gain nothing above 2x, and low-end GPUs pay
  // for every extra pixel.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height, false);
  fitCamera(width, height);
  invalidate();
}

new ResizeObserver(resize).observe(host);
resize();

new IntersectionObserver(
  ([entry]) => {
    onScreen = entry.isIntersecting;
    invalidate();
  },
  { threshold: 0 }
).observe(host);

document.addEventListener("visibilitychange", () => {
  pageVisible = !document.hidden;
  invalidate();
});

const clock = new THREE.Clock();

renderer.setAnimationLoop(() => {
  if (!onScreen || !pageVisible) return;

  const cameraMoving = updateCamera();
  if (!continuous && !dirty && !cameraMoving) return;
  dirty = false;

  const t = clock.getElapsedTime();

  if (activeIndex >= 0) {
    nodes[activeIndex].core.rotation.y = t * 0.6;
    nodes[activeIndex].core.rotation.x = Math.sin(t * 0.4) * 0.25;
  }
  if (payload.visible) {
    layoutPayload(payloadSpread, payloadParked ? t * 1.6 : t * 4);
  }

  renderer.render(scene, camera);
});

play();

/* ------------------------------------------------------------------ *
 * Teardown — Streamlit reruns replace this iframe, so release the GPU
 * ------------------------------------------------------------------ */
window.addEventListener("pagehide", () => {
  renderer.setAnimationLoop(null);
  gsap.globalTimeline.clear();
  for (const resource of disposables) resource.dispose?.();
  payload.dispose();
  renderer.dispose();
});
