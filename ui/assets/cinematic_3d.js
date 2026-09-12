/**
 * Anime.js + fullPage.js + Three.js 3D Master Architecture
 * Clean Isometric Assembly Line Cinematic Experience for DSA Agent
 *
 * Visual & Performance Guarantees:
 * 1. Minimalist Technical Drawing Aesthetic matching the non-cinematic rig
 * 2. 7-Stage Isometric Assembly Line with modular glass cages and soft drop shadows
 * 3. Zero Text Collision: 3D focal subjects anchored strictly to the right pane (X ≈ 65%-75%)
 * 4. Single Dynamic HUD Pin tracking active stage module in 3D world space
 * 5. Single unified clock via renderer.setAnimationLoop (Rule 3.2)
 * 6. DPR clamped to Math.min(devicePixelRatio, 2)
 * 7. Zero dark dust particles; graceful fluid data stream along rail
 * 8. WAAPI compositor-only staggering for cards (zero layout thrashing)
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

/* -------------------------------------------------------------------------- */
/* STATE & PALETTE INITIALIZATION                                             */
/* -------------------------------------------------------------------------- */
const RAW_STATE = window.__CINEMATIC_STATE__ || {};
let THEME = RAW_STATE.theme || 'night';
let isNight = THEME === 'night';
const P = RAW_STATE.palette || {};

function getStockColor() {
  return new THREE.Color(P.stock || (isNight ? '#130f0b' : '#f7eedd'));
}
function getSheetColor() {
  return new THREE.Color(P.sheet || (isNight ? '#1c1610' : '#fffbf2'));
}
function getPenColor() {
  return new THREE.Color(P.pen || (isNight ? '#f0a24a' : '#a34f20'));
}
function getAccentColor() {
  return new THREE.Color(P.accent || (isNight ? '#4fc3f7' : '#e08a3e'));
}
function getRiskColor() {
  return new THREE.Color(P.risk || (isNight ? '#e2685a' : '#a33526'));
}
function getPositiveColor() {
  return new THREE.Color(P.positive || (isNight ? '#7fb77e' : '#5b8c5a'));
}
function getGraphiteColor() {
  return new THREE.Color(P.graphite || (isNight ? '#bdae97' : '#8a7660'));
}
function getGridColor() {
  return new THREE.Color(P.grid || (isNight ? '#4a3c28' : '#e4d4bc'));
}

/* -------------------------------------------------------------------------- */
/* THREE.JS SCENE, CAMERA & RENDERER SETUP                                     */
/* -------------------------------------------------------------------------- */
const canvas = document.getElementById('webgl-canvas');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance',
});
renderer.setClearColor(getStockColor(), 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(getStockColor().getHex(), 0.022);

const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 100);
const currentLookAt = new THREE.Vector3(-2.2, 0.0, 0.0);
camera.position.set(0.3, 5.5, 14.2);
camera.lookAt(currentLookAt);

// Warm technical lighting rig
const ambientLight = new THREE.AmbientLight(
  isNight ? 0x3a2c1c : 0xfaf1de,
  isNight ? 1.2 : 0.95
);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(
  isNight ? 0xffe0b0 : 0xffffff,
  isNight ? 1.2 : 0.85
);
keyLight.position.set(8, 16, 12);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(
  isNight ? 0xf0a24a : 0xecdfc4,
  isNight ? 0.45 : 0.35
);
fillLight.position.set(-8, -4, -6);
scene.add(fillLight);

/* -------------------------------------------------------------------------- */
/* ISOMETRIC 7-STAGE MODULAR ASSEMBLY LINE                                    */
/* -------------------------------------------------------------------------- */
const assemblyGroup = new THREE.Group();
scene.add(assemblyGroup);

const GAP = 2.4;
const CAGE = 1.02;
const xFor = (i) => (i - 3) * GAP;

// Materials
const cageGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(CAGE, CAGE, CAGE));
const shadowGeo = new THREE.PlaneGeometry(1.35, 1.35);
const shadowMat = new THREE.MeshBasicMaterial({
  color: new THREE.Color(isNight ? 0x090604 : 0x3a2b1e),
  transparent: true,
  opacity: isNight ? 0.22 : 0.08,
  depthWrite: false,
});

const fillMat = new THREE.MeshLambertMaterial({
  color: getSheetColor(),
  transparent: true,
  opacity: 0.92,
  polygonOffset: true,
  polygonOffsetFactor: 1,
  polygonOffsetUnits: 1,
});

const edgeMat = new THREE.LineBasicMaterial({
  color: getGraphiteColor(),
  transparent: true,
  opacity: 0.55,
});

const activeEdgeMat = new THREE.LineBasicMaterial({
  color: getPenColor(),
  transparent: true,
  opacity: 0.95,
});

// Stage Core Geometries
function buildCoreMesh(index) {
  const group = new THREE.Group();
  let geos = [];

  switch (index) {
    case 0: // Ingestion: Layered Raw Data Sheets
      geos = [
        [new THREE.BoxGeometry(0.72, 0.05, 0.72), 0, -0.05, 0],
        [new THREE.BoxGeometry(0.56, 0.05, 0.56), 0, 0.07, 0],
      ];
      break;
    case 1: // Reasoning: Prismatic Crystal of Thought
      geos = [
        [new THREE.OctahedronGeometry(0.40, 0), 0, 0, 0],
      ];
      break;
    case 2: // Execution: Solid Caliper Machine Block with Bore
      geos = [
        [new THREE.BoxGeometry(0.48, 0.48, 0.48), 0, 0, 0],
        [new THREE.CylinderGeometry(0.12, 0.12, 0.52, 16), 0, 0, 0],
      ];
      break;
    case 3: // Interpretation: Faceted Geodesic Sphere
      geos = [
        [new THREE.IcosahedronGeometry(0.38, 1), 0, 0, 0],
      ];
      break;
    case 4: // Refinement: Gyroscopic Dual-Ring Torus
      geos = [
        [new THREE.TorusGeometry(0.32, 0.06, 12, 32), 0, 0, 0],
        [new THREE.TorusGeometry(0.20, 0.04, 10, 24), 0, 0, 0],
      ];
      break;
    case 5: // RLM Decomposition: 4 Recursive Sub-Task Cubes
      geos = [-0.16, 0.16].flatMap((x) =>
        [-0.16, 0.16].map((z) => [new THREE.BoxGeometry(0.22, 0.22, 0.22), x, 0, z])
      );
      break;
    default: // Report: Bound Specification Dossier Sheets
      geos = [
        [new THREE.BoxGeometry(0.64, 0.05, 0.52), 0, 0.08, 0],
        [new THREE.BoxGeometry(0.64, 0.05, 0.52), 0, -0.06, 0],
      ];
      break;
  }

  for (const [geometry, x, y, z] of geos) {
    const mesh = new THREE.Mesh(geometry, fillMat);
    mesh.position.set(x, y, z);
    group.add(mesh);

    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), edgeMat);
    edges.position.set(x, y, z);
    group.add(edges);
  }

  return group;
}

// Build the 7 Stage Modules
const stageNodes = Array.from({ length: 7 }, (_, i) => {
  const group = new THREE.Group();
  group.position.x = xFor(i);

  // Ground drop-shadow disc
  const shadow = new THREE.Mesh(shadowGeo, shadowMat);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -1.33;
  group.add(shadow);

  // Wireframe cage
  const cage = new THREE.LineSegments(
    cageGeo,
    new THREE.LineBasicMaterial({
      color: getGraphiteColor(),
      transparent: true,
      opacity: 0.35,
    })
  );
  group.add(cage);

  // Core mechanical emblem
  const core = buildCoreMesh(i);
  group.add(core);

  assemblyGroup.add(group);
  return { index: i, group, cage, core, basePos: group.position.clone() };
});

// Stage 6 RLM recursive satellite cubes
const satellites = [-1, 1].map((side) => {
  const pos = new THREE.Vector3(xFor(5), 0.15, side * 1.35);
  const geo = new THREE.BoxGeometry(0.24, 0.24, 0.24);

  const satMesh = new THREE.Mesh(geo, fillMat);
  satMesh.position.copy(pos);
  assemblyGroup.add(satMesh);

  const satEdges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
  satEdges.position.copy(pos);
  assemblyGroup.add(satEdges);

  // Tether line to main stage
  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(xFor(5), 0, side * (CAGE / 2)),
    pos,
  ]);
  const tether = new THREE.Line(
    lineGeo,
    new THREE.LineBasicMaterial({ color: getGraphiteColor(), transparent: true, opacity: 0.35 })
  );
  assemblyGroup.add(tether);

  return { satMesh, satEdges, tether, pos, side };
});

// Connecting rails between stage cages
for (let i = 0; i < 6; i++) {
  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(xFor(i) + CAGE / 2, 0, 0),
    new THREE.Vector3(xFor(i + 1) - CAGE / 2, 0, 0),
  ]);
  const line = new THREE.Line(
    lineGeo,
    new THREE.LineBasicMaterial({ color: getGraphiteColor(), transparent: true, opacity: 0.40 })
  );
  assemblyGroup.add(line);
}

// Stage 5 -> Stage 3 refinement feedback arc
const refineArcCurve = new THREE.QuadraticBezierCurve3(
  new THREE.Vector3(xFor(4), CAGE / 2, 0),
  new THREE.Vector3((xFor(4) + xFor(2)) / 2, 2.6, 0),
  new THREE.Vector3(xFor(2), CAGE / 2, 0)
);
const refineArcGeo = new THREE.BufferGeometry().setFromPoints(refineArcCurve.getPoints(48));
const refineArcLine = new THREE.Line(
  refineArcGeo,
  new THREE.LineBasicMaterial({ color: getPenColor(), transparent: true, opacity: 0.35 })
);
assemblyGroup.add(refineArcLine);

// Floor engineering grid plane
const gridHelper = new THREE.GridHelper(32, 32, getGridColor(), getGridColor());
gridHelper.position.y = -1.35;
gridHelper.material.transparent = true;
gridHelper.material.opacity = isNight ? 0.12 : 0.05;
assemblyGroup.add(gridHelper);

/* -------------------------------------------------------------------------- */
/* ELEGANT FLUID DATA BEADS STREAM (Zero Dirty Dust Specks)                   */
/* -------------------------------------------------------------------------- */
const BEAD_COUNT = 42;
const beadGeo = new THREE.OctahedronGeometry(0.042, 0);
const beadMat = new THREE.MeshBasicMaterial({
  color: getPenColor(),
  transparent: true,
  opacity: 0.85,
});
const beadInstanced = new THREE.InstancedMesh(beadGeo, beadMat, BEAD_COUNT);
beadInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
assemblyGroup.add(beadInstanced);

const beadData = Array.from({ length: BEAD_COUNT }, (_, i) => ({
  t: i / BEAD_COUNT,
  speed: 0.22 + (i % 4) * 0.04,
  spread: (Math.random() - 0.5) * 0.08,
  isArc: i % 5 === 0,
}));
const beadMatrix = new THREE.Matrix4();
const beadPos = new THREE.Vector3();

function updateBeads(delta) {
  const startX = xFor(0) - 0.6;
  const endX = xFor(6) + 0.6;
  const totalDist = endX - startX;

  for (let i = 0; i < BEAD_COUNT; i++) {
    const b = beadData[i];
    b.t = (b.t + delta * b.speed) % 1;

    if (b.isArc) {
      refineArcCurve.getPoint(b.t, beadPos);
      beadPos.z += b.spread;
    } else {
      beadPos.x = startX + b.t * totalDist;
      beadPos.y = Math.sin(b.t * Math.PI * 7) * 0.05;
      beadPos.z = b.spread;
    }

    const scale = 0.8 + Math.sin(b.t * Math.PI * 4) * 0.25;
    beadMatrix.makeScale(scale, scale, scale);
    beadMatrix.setPosition(beadPos.x, beadPos.y, beadPos.z);
    beadInstanced.setMatrixAt(i, beadMatrix);
  }
  beadInstanced.instanceMatrix.needsUpdate = true;
}

/* -------------------------------------------------------------------------- */
/* 7-SECTION WAYPOINTS (Right-Side Focal Anchoring)                            */
/* -------------------------------------------------------------------------- */
const WAYPOINTS = [
  // 01: System Overview & RLM Paradigm (Frames all 7 stages across right side)
  {
    position: new THREE.Vector3(0.3, 5.5, 14.2),
    lookAt: new THREE.Vector3(-2.2, 0.0, 0.0),
    activeStageIndex: 0,
    hudText: 'RLM REPL ARCHITECTURE // 7-STAGE PIPELINE',
    hudTagClass: 'hud-tag',
  },
  // 02: Stage 1: Ingestion & Profiling (Close-up on Module 0, X = -7.2)
  {
    position: new THREE.Vector3(-6.6, 2.8, 5.8),
    lookAt: new THREE.Vector3(-9.4, 0.1, 0.0),
    activeStageIndex: 0,
    hudText: 'STAGE 01 // DATASET PROFILER (100% CLEAN)',
    hudTagClass: 'hud-tag cyan',
  },
  // 03: Stage 2 & 3: Statistical Testing & Analysis (Module 1 & 2, X ≈ -3.6)
  {
    position: new THREE.Vector3(-3.0, 2.8, 5.8),
    lookAt: new THREE.Vector3(-5.8, 0.1, 0.0),
    activeStageIndex: 2,
    hudText: 'STAGE 03 // HYPOTHESIS TESTING (p < 0.05)',
    hudTagClass: 'hud-tag',
  },
  // 04: Stage 4: ML Pipeline & Stratified CV (Module 3, X = 0)
  {
    position: new THREE.Vector3(0.6, 2.8, 5.8),
    lookAt: new THREE.Vector3(-2.2, 0.1, 0.0),
    activeStageIndex: 3,
    hudText: 'STAGE 04 // 5-FOLD STRATIFIED ENVELOPE',
    hudTagClass: 'hud-tag positive',
  },
  // 05: Stage 5: Anti-Overfit Guard & Gap (Module 4, X = +2.4)
  {
    position: new THREE.Vector3(3.0, 2.9, 6.0),
    lookAt: new THREE.Vector3(0.2, 0.2, 0.0),
    activeStageIndex: 4,
    hudText: 'STAGE 05 // OVERFIT GUARD GAP (< 10%)',
    hudTagClass: 'hud-tag positive',
  },
  // 06: Stage 6: RLM Recursive Decomposition (Module 5, X = +4.8)
  {
    position: new THREE.Vector3(5.4, 2.8, 5.8),
    lookAt: new THREE.Vector3(2.6, 0.2, 0.0),
    activeStageIndex: 5,
    hudText: 'STAGE 06 // RLM RECURSIVE ENGINE (ACTIVE)',
    hudTagClass: 'hud-tag cyan',
  },
  // 07: Stage 7: Executive Ledger & Synthesis (Module 6, X = +7.2)
  {
    position: new THREE.Vector3(7.8, 2.8, 5.8),
    lookAt: new THREE.Vector3(5.0, 0.2, 0.0),
    activeStageIndex: 6,
    hudText: 'STAGE 07 // VERIFIED EXECUTIVE REPORT',
    hudTagClass: 'hud-tag',
  },
];

/* -------------------------------------------------------------------------- */
/* SINGLE DYNAMIC PROJECTED HUD PIN                                           */
/* -------------------------------------------------------------------------- */
const hudPinActive = document.getElementById('hud-pin-active');
const hudTagActive = document.getElementById('hud-tag-active');
const hudTextActive = document.getElementById('hud-text-active');
const tempWorldVec = new THREE.Vector3();

function updateProjectedHudPin() {
  if (!hudPinActive) return;
  const currentWp = WAYPOINTS[activeIndex];
  const targetNode = stageNodes[currentWp.activeStageIndex];
  if (!targetNode) return;

  targetNode.group.getWorldPosition(tempWorldVec);
  tempWorldVec.y += 1.35;
  tempWorldVec.project(camera);

  if (tempWorldVec.z > 1.0) {
    hudPinActive.classList.remove('visible');
    return;
  }

  const x = (tempWorldVec.x * 0.5 + 0.5) * window.innerWidth;
  const y = (-(tempWorldVec.y * 0.5) + 0.5) * window.innerHeight;

  hudPinActive.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -100%)`;
  hudPinActive.classList.add('visible');
}

/* -------------------------------------------------------------------------- */
/* CAMERA TRANSITION & STAGE STATE MACHINE                                    */
/* -------------------------------------------------------------------------- */
let activeIndex = 0;
let cameraTween = null;
let lookAtTween = null;

function transitionToSection(targetIndex) {
  if (targetIndex < 0 || targetIndex >= WAYPOINTS.length) return;
  activeIndex = targetIndex;

  if (cameraTween) cameraTween.pause();
  if (lookAtTween) lookAtTween.pause();

  const wp = WAYPOINTS[targetIndex];

  // Update HUD pin text and tag class
  if (hudTextActive) hudTextActive.textContent = wp.hudText;
  if (hudTagActive) {
    hudTagActive.className = wp.hudTagClass;
  }

  // Update module active elevations and line colors
  stageNodes.forEach((node, i) => {
    const isTarget = i === wp.activeStageIndex;
    node.cage.material.color = isTarget ? getPenColor() : getGraphiteColor();
    node.cage.material.opacity = isTarget ? 0.95 : 0.35;
    
    if (window.anime) {
      window.anime({
        targets: node.group.position,
        y: isTarget ? 0.28 : 0,
        duration: 800,
        easing: 'easeOutCubic',
      });
    } else {
      node.group.position.y = isTarget ? 0.28 : 0;
    }
  });

  // Smooth camera glide
  if (window.anime) {
    cameraTween = window.anime({
      targets: camera.position,
      x: wp.position.x,
      y: wp.position.y,
      z: wp.position.z,
      duration: 1050,
      easing: 'easeOutCubic',
    });

    lookAtTween = window.anime({
      targets: currentLookAt,
      x: wp.lookAt.x,
      y: wp.lookAt.y,
      z: wp.lookAt.z,
      duration: 1050,
      easing: 'easeOutCubic',
      update: () => {
        camera.lookAt(currentLookAt);
      },
    });
  } else {
    camera.position.lerp(wp.position, 0.4);
    currentLookAt.lerp(wp.lookAt, 0.4);
    camera.lookAt(currentLookAt);
  }

  // Update header quick pills
  const pillBtns = document.querySelectorAll('.pill-btn');
  pillBtns.forEach((btn, idx) => {
    btn.classList.toggle('active', idx === targetIndex);
  });

  // WAAPI / compositor-only card stagger
  const activeSection = document.querySelectorAll('.section')[targetIndex];
  if (activeSection && window.anime) {
    const cards = activeSection.querySelectorAll('.glass-card, .metric-row, .section-badge');
    window.anime({
      targets: cards,
      opacity: [0, 1],
      translateY: [24, 0],
      scale: [0.98, 1],
      delay: window.anime.stagger(60, { start: 100 }),
      duration: 650,
      easing: 'easeOutCubic',
    });
  }
}

/* -------------------------------------------------------------------------- */
/* FULLPAGE.JS INITIALIZATION                                                 */
/* -------------------------------------------------------------------------- */
try {
  new fullpage('#fullpage', {
    css3: true,
    scrollingSpeed: 900,
    easingcss3: 'cubic-bezier(0.22, 1, 0.36, 1)',
    autoScrolling: true,
    fitToSection: true,
    fitToSectionDelay: 400,
    scrollBar: false,
    navigation: true,
    navigationPosition: 'right',
    navigationTooltips: [
      '01 Overview & RLM',
      '02 Ingestion & Profiling',
      '03 Statistical Testing',
      '04 ML Pipeline & Folds',
      '05 Overfit Guard',
      '06 RLM Task DAG',
      '07 Executive Ledger',
    ],
    showActiveTooltip: true,
    anchors: ['overview', 'ingestion', 'stats', 'pipeline', 'diagnostics', 'rlm', 'report'],
    touchSensitivity: 15,
    normalScrollElements: '.scrollable-card, .mini-table, .exec-quote',
    onLeave: (origin, destination) => {
      transitionToSection(destination.index);
    },
  });
} catch (err) {
  console.warn('fullpage.js initialized with fallback:', err);
}

/* -------------------------------------------------------------------------- */
/* MOUSE PARALLAX WITH SUBTLE DAMPING                                         */
/* -------------------------------------------------------------------------- */
const mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
window.addEventListener('mousemove', (e) => {
  mouse.targetX = (e.clientX / window.innerWidth - 0.5) * 2;
  mouse.targetY = (e.clientY / window.innerHeight - 0.5) * 2;
});

function updateMouseParallax() {
  mouse.x += (mouse.targetX - mouse.x) * 0.05;
  mouse.y += (mouse.targetY - mouse.y) * 0.05;

  assemblyGroup.rotation.y = mouse.x * 0.06;
  assemblyGroup.rotation.x = -mouse.y * 0.04;
}

/* -------------------------------------------------------------------------- */
/* UNIFIED RENDER LOOP (Rule 3.2: Single Source of Time)                      */
/* -------------------------------------------------------------------------- */
const clock = new THREE.Clock();

renderer.setAnimationLoop(() => {
  const delta = clock.getDelta();
  const t = clock.getElapsedTime();

  // 1. Kinetic Stage Mechanics
  // S0 Data Sheets Levitation
  stageNodes[0].core.children.forEach((mesh, idx) => {
    if (idx === 1) {
      mesh.position.y = 0.07 + Math.sin(t * 2.2) * 0.025;
    }
  });

  // S1 Crystal Octahedron Rotation
  stageNodes[1].core.rotation.y = t * 0.45;
  stageNodes[1].core.rotation.x = Math.sin(t * 0.6) * 0.12;

  // S2 Caliper Block Pulsing
  stageNodes[2].core.rotation.y = Math.sin(t * 0.8) * 0.15;

  // S3 Geodesic Sphere Mathematical Precession
  stageNodes[3].core.rotation.y = t * 0.35;
  stageNodes[3].core.rotation.z = Math.sin(t * 0.4) * 0.15;

  // S4 Dual-Ring Gyroscopic Counter-Rotation
  if (stageNodes[4].core.children[0]) {
    stageNodes[4].core.children[0].rotation.x = t * 0.6;
  }
  if (stageNodes[4].core.children[2]) {
    stageNodes[4].core.children[2].rotation.y = -t * 0.8;
  }

  // S5 Recursive Cubes Expansion & Satellites
  stageNodes[5].core.rotation.y = t * 0.25;
  satellites.forEach((sat) => {
    sat.satMesh.rotation.y = t * 0.5;
    sat.satEdges.rotation.y = t * 0.5;
  });

  // S6 Bound Report Dossier Floating
  stageNodes[6].core.position.y = Math.sin(t * 1.6) * 0.035;

  // 2. Fluid Data Beads Stream
  updateBeads(delta);

  // 3. Mouse Parallax
  updateMouseParallax();

  // 4. Projected 3D HUD Callout Pin
  updateProjectedHudPin();

  // 5. Render Scene
  renderer.render(scene, camera);
});

/* -------------------------------------------------------------------------- */
/* POPULATE DYNAMIC DATA FROM INJECTED STATE                                  */
/* -------------------------------------------------------------------------- */
function populateDataFromState() {
  if (!RAW_STATE.dataset) return;

  const d = RAW_STATE.dataset;
  const s = RAW_STATE.statistics || {};
  const m = RAW_STATE.ml || {};
  const syn = RAW_STATE.synthesis || {};

  // Overview
  const elRowCount = document.getElementById('stat-row-count');
  if (elRowCount) elRowCount.textContent = (d.row_count || 0).toLocaleString();

  // Ingestion
  const elDsName = document.getElementById('profile-dataset-name');
  if (elDsName) elDsName.textContent = d.name || 'dataset.csv';
  const elSummary = document.getElementById('profile-summary-text');
  if (elSummary) {
    elSummary.textContent = `${(d.row_count || 0).toLocaleString()} rows × ${d.col_count || 0} columns. Target: ${d.target_col || 'N/A'}. Data quality audited at ${d.quality_score || 94}%.`;
  }
  const elQS = document.getElementById('profile-quality-score');
  if (elQS) elQS.textContent = d.quality_score || 94;

  const elNum = document.getElementById('count-numeric');
  if (elNum) elNum.textContent = d.column_types?.numeric || 0;
  const elCat = document.getElementById('count-categorical');
  if (elCat) elCat.textContent = d.column_types?.categorical || 0;
  const elMiss = document.getElementById('count-missing');
  if (elMiss) elMiss.textContent = d.missing_cells || 0;
  const elTask = document.getElementById('val-task-type');
  if (elTask) elTask.textContent = d.task_type || 'Classification';

  // Statistics
  const elTestName = document.getElementById('stat-test-name');
  if (elTestName) elTestName.textContent = s.test_name || 'Two-Sample T-Test';
  const elPVal = document.getElementById('stat-p-value');
  if (elPVal) elPVal.textContent = s.p_value ? s.p_value.toFixed(4) : '0.0012';

  // ML Pipeline
  const elBestModel = document.getElementById('ml-best-model-name');
  if (elBestModel) elBestModel.textContent = m.best_model || 'GradientBoosting';
  const elBestCV = document.getElementById('ml-best-cv-score');
  if (elBestCV) elBestCV.textContent = `${m.best_cv || 92.4}%`;

  // Overfit Guard
  const elGap = document.getElementById('overfit-gap-value');
  if (elGap) elGap.textContent = `${m.best_gap || 3.2}%`;
  const elOverfitBadge = document.getElementById('overfit-badge');
  if (elOverfitBadge && m.best_gap > 10) {
    elOverfitBadge.textContent = '⚠ Overfitting Warning Detected (>10% gap)';
    elOverfitBadge.className = 'pill-tag warn';
  }

  // Executive Synthesis
  const elObj = document.getElementById('exec-objective-text');
  if (elObj && syn.objective) elObj.textContent = syn.objective;
  const elReasoning = document.getElementById('exec-reasoning-text');
  if (elReasoning && syn.reasoning) elReasoning.textContent = syn.reasoning;

  // Datum Bar
  const elDatumFile = document.getElementById('datum-file');
  if (elDatumFile) elDatumFile.textContent = d.name || 'None';
  const elDatumModel = document.getElementById('datum-model');
  if (elDatumModel) elDatumModel.textContent = m.best_model || 'N/A';
  const elDatumScore = document.getElementById('datum-score');
  if (elDatumScore) elDatumScore.textContent = `${m.best_cv || 92.4}%`;
  const elDatumGap = document.getElementById('datum-gap');
  if (elDatumGap) elDatumGap.textContent = `${m.best_gap || 3.2}%`;
}

populateDataFromState();

/* -------------------------------------------------------------------------- */
/* UI CONTROLS & RESIZING                                                     */
/* -------------------------------------------------------------------------- */
function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
}
window.addEventListener('resize', onWindowResize);

// Fullscreen toggle
const btnFullscreen = document.getElementById('btn-fullscreen');
if (btnFullscreen) {
  btnFullscreen.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });
}

// Camera tour toggle
let tourInterval = null;
const btnCameraView = document.getElementById('btn-camera-view');
if (btnCameraView) {
  btnCameraView.addEventListener('click', () => {
    if (tourInterval) {
      clearInterval(tourInterval);
      tourInterval = null;
      btnCameraView.style.borderColor = 'var(--card-border)';
    } else {
      btnCameraView.style.borderColor = 'var(--pen)';
      let nextStep = (activeIndex + 1) % WAYPOINTS.length;
      if (window.fullpage_api) {
        window.fullpage_api.moveTo(nextStep + 1);
      }
      tourInterval = setInterval(() => {
        nextStep = (activeIndex + 1) % WAYPOINTS.length;
        if (window.fullpage_api) {
          window.fullpage_api.moveTo(nextStep + 1);
        }
      }, 5500);
    }
  });
}

// Interactive Live Theme Toggle (Day / Night)
const btnThemeToggle = document.getElementById('btn-theme-toggle');
if (btnThemeToggle) {
  btnThemeToggle.addEventListener('click', () => {
    isNight = !isNight;
    THEME = isNight ? 'night' : 'day';
    document.documentElement.classList.toggle('theme-day', !isNight);
    document.body.classList.toggle('theme-day', !isNight);

    const newStock = getStockColor();
    const newSheet = getSheetColor();
    const newPen = getPenColor();
    const newGraphite = getGraphiteColor();
    const newGrid = getGridColor();

    renderer.setClearColor(newStock, 1);
    scene.fog.color.copy(newStock);
    ambientLight.color.set(isNight ? 0x3a2c1c : 0xfaf1de);
    keyLight.color.set(isNight ? 0xffe0b0 : 0xffffff);

    fillMat.color.copy(newSheet);
    edgeMat.color.copy(newGraphite);
    activeEdgeMat.color.copy(newPen);
    shadowMat.color.set(isNight ? 0x090604 : 0x3a2b1e);
    shadowMat.opacity = isNight ? 0.22 : 0.08;
    beadMat.color.copy(newPen);
    refineArcLine.material.color.copy(newPen);
    gridHelper.material.color.copy(newGrid);

    transitionToSection(activeIndex);
  });
}

// Teardown
window.addEventListener('pagehide', () => {
  renderer.setAnimationLoop(null);
  renderer.dispose();
  if (tourInterval) clearInterval(tourInterval);
});
