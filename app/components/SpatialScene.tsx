"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import {
  type MutableRefObject,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import type {
  HandLandmarker as HandLandmarkerInstance,
  HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

const CAROUSEL_PARTS = [
  {
    name: "Signature Burger Combo",
    description: "Double wagyu patty, truffle mayo, aged cheddar, brioche bun.",
  },
  {
    name: "Sushi Roll Combo",
    description: "California roll with crab, avocado, cucumber, toasted sesame.",
  },
  {
    name: "Classic Fries Set",
    description: "Hand-cut golden fries with house seasoning and dipping sauce.",
  },
  {
    name: "Artisan Coffee Set",
    description: "Single-origin espresso, oat milk foam, seasonal flavor notes.",
  },
  {
    name: "Chef's Dessert Combo",
    description: "Warm chocolate fondant with vanilla bean ice cream.",
  },
] as const;

const BURGER_INGREDIENTS = [
  { name: "Bottom Bun", cal: "160 kcal", allergen: "Gluten, Sesame", flavor: "Toasted brioche base, golden butter crust" },
  { name: "Truffle Sauce", cal: "90 kcal", allergen: "Eggs, Mustard", flavor: "House truffle mayo with Dijon, earthy and tangy" },
  { name: "Wagyu Patty", cal: "340 kcal", allergen: "None", flavor: "Flame-grilled premium wagyu, rich umami char" },
  { name: "Aged Cheddar", cal: "110 kcal", allergen: "Dairy", flavor: "Two-year reserve, sharp and creamy" },
  { name: "Tomato", cal: "10 kcal", allergen: "None", flavor: "Vine-ripened beefsteak, sweet and bright" },
  { name: "Crisp Lettuce", cal: "5 kcal", allergen: "None", flavor: "Iceberg leaf, cool and fresh" },
  { name: "Top Bun", cal: "180 kcal", allergen: "Gluten, Sesame", flavor: "Toasted brioche dome with sesame seeds" },
] as const;

const ITEM_PRICES = [18.90, 15.50, 7.90, 6.50, 12.80];

const FOOD_INSPECT_DATA = [
  {
    special: "TODAY'S SPECIAL",
    calories: "780 kcal",
    protein: "38g",
    allergens: "Wheat, dairy, egg",
    flavorProfile: "Savory, creamy, smoky, lightly sweet",
    ingredients: "Brioche bun · wagyu beef patty · aged cheddar · crisp lettuce · tomato · truffle sauce",
    chefNote: "Signature layered burger with warm brioche and rich truffle sauce.",
  },
  {
    special: null,
    calories: "420 kcal",
    protein: "18g",
    allergens: "Shellfish, soy, sesame",
    flavorProfile: "Clean, oceanic, nutty, lightly vinegared",
    ingredients: "Sushi rice · nori · crab · avocado · cucumber · sesame seeds",
    chefNote: "House-rolled California with ripe avocado and toasted sesame.",
  },
  {
    special: null,
    calories: "380 kcal",
    protein: "5g",
    allergens: "None",
    flavorProfile: "Crispy, salty, lightly herbed",
    ingredients: "Russet potatoes · sunflower oil · sea salt · house seasoning",
    chefNote: "Hand-cut daily and seasoned with our house spice blend.",
  },
  {
    special: null,
    calories: "120 kcal",
    protein: "4g",
    allergens: "Dairy (oat milk)",
    flavorProfile: "Rich, floral, creamy, subtly sweet",
    ingredients: "Single-origin espresso · oat milk · seasonal syrup",
    chefNote: "Pulled from single-origin beans with precision-steamed oat milk.",
  },
  {
    special: null,
    calories: "640 kcal",
    protein: "8g",
    allergens: "Wheat, dairy, egg",
    flavorProfile: "Warm, chocolatey, rich, cooling",
    ingredients: "Dark chocolate · butter · egg · flour · vanilla bean ice cream",
    chefNote: "Warm fondant paired with housemade vanilla bean ice cream.",
  },
] as const;

const INSPECT_ROTATION_STEP = 0.32;
const MEDIAPIPE_WASM_PATH = "/mediapipe/wasm";
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Swipe detection tuning — all distances are normalized palm-X (0.0–1.0).
const SWIPE_WINDOW_MS = 350;              // rolling sample buffer
const SWIPE_COOLDOWN_MS = 600;            // hard block after any swipe fires
const OPPOSITE_LOCK_MS = 900;             // opposite direction requires elevated thresholds within this window
const MIN_SWIPE_DISTANCE = 0.17;          // minimum palm-X displacement for a same/unlocked direction
const MIN_SWIPE_VELOCITY = 0.00042;       // palm-X per ms — rejects slow drift
const OPPOSITE_DISTANCE_MULT = 1.55;      // extra distance required for opposite swipe within OPPOSITE_LOCK_MS
const OPPOSITE_VELOCITY_MULT = 1.45;      // extra velocity required for opposite swipe within OPPOSITE_LOCK_MS

// Inspect gesture — palm width growing in frame (open hand approaching camera).
// Palm width uses landmark 5 (index base) → 17 (pinky base): stable across finger poses.
const INSPECT_COOLDOWN_MS = 2500;
const INSPECT_GROW_THRESHOLD = 0.048;   // palm-width growth to confirm approaching hand
const INSPECT_GUIDANCE_THRESHOLD = 0.018; // lower threshold for "OPEN HAND INSPECT" hint
const INSPECT_SAMPLE_WINDOW_MS = 600;   // wider window for a more relaxed trigger

// Double-fist pulse gesture — two distinct fist "pulses" (open→fist transitions)
// within DOUBLE_FIST_WINDOW_MS confirm add-to-order. Thumb excluded from fist score
// because its resting position varies across fist styles.
// Pulse = rising edge only: holding a fist counts as one pulse, never more.
// Hysteresis prevents threshold-boundary flickering: enter requires tighter close,
// exit requires a clearly open hand. Separate enter/exit thresholds are tracked via
// isFistLatchRef so the rising-edge detector sees clean, stable transitions.
const FIST_ENTER_THRESHOLD = 1.1;        // normalised avg tip distance — must dip below to enter fist
const FIST_EXIT_THRESHOLD = 1.32;        // must rise above to leave fist (hysteresis band)
const DOUBLE_FIST_WINDOW_MS = 1800;      // second fist must arrive within this window
const FIST_ADD_COOLDOWN_MS = 3000;       // post-fire lockout before gesture can re-arm
const FIST_AFTER_SWIPE_IGNORE_MS = 1200; // suppress if a swipe fired recently
const MIN_OPEN_FRAMES = 3;               // min consecutive non-fist frames between pulses

// Exit-inspect gesture — open hand shrinking = hand retreating from camera ("push away").
// Uses same palm-width metric as enter so they are symmetrically detectable.
// Mutual exclusion: enter only fires when NOT in inspect; exit only fires when IN inspect.
// Cross-bumping cooldowns prevent immediate re-entry after exit and vice versa.
const EXIT_INSPECT_SHRINK_THRESHOLD = 0.045;   // palm-width shrink required to confirm exit
const EXIT_INSPECT_GUIDANCE_THRESHOLD = 0.016; // lower threshold for "MOVE HAND BACK" hint
const EXIT_INSPECT_SAMPLE_WINDOW_MS = 650;      // rolling window (slightly wider than enter)
const EXIT_INSPECT_COOLDOWN_MS = 2500;          // post-exit lockout prevents double-fire

const LOGO_TEXT = "AURA ONE";
const LOGO_LETTERS: Record<string, string[]> = {
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
  N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
  O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
};

type LogoParticle = {
  home: THREE.Vector3;
  scatter: THREE.Vector3;
  seed: number;
};

type GestureAction =
  | "EXPLODE"
  | "ASSEMBLE"
  | "TOGGLE_EXPLODE"
  | "NEXT_PART"
  | "PREV_PART"
  | "ENTER_INSPECT"
  | "EXIT_INSPECT"
  | "RESET"
  | "ROTATE_INSPECT_LEFT"
  | "ROTATE_INSPECT_RIGHT"
  | "TOGGLE_BURGER_EXPLODE"
  | "ADD_TO_ORDER"
  | "REMOVE_LAST"
  | "CLEAR_ORDER";

type CameraGestureStatus =
  | "CAMERA OFF"
  | "LOADING CAMERA"
  | "TRYING GPU"
  | "GPU FAILED"
  | "FALLBACK CPU"
  | "HAND TRACKING READY"
  | "CAMERA READY"
  | "READY"
  | "SWIPE LEFT"
  | "SWIPE RIGHT"
  | "COOLDOWN"
  | "OPPOSITE LOCK"
  | "OPEN HAND INSPECT"
  | "INSPECT GESTURE"
  | "FIST AGAIN TO ADD"
  | "ORDER CANCELLED"
  | "ADDED TO ORDER"
  | "MOVE HAND BACK"
  | "EXIT INSPECT"
  | "CAMERA ERROR";

type CameraDebugStep =
  | "Idle"
  | "Requesting camera"
  | "Camera stream received"
  | "Video element assigned"
  | "Video playback started"
  | "Loading MediaPipe"
  | "MediaPipe loaded"
  | "Initializing HandLandmarker (GPU)"
  | "GPU init failed — retrying CPU"
  | "Initializing HandLandmarker (CPU)"
  | "HandLandmarker ready"
  | "Detection loop started";


function smoothStep(value: number) {
  return value * value * (3 - 2 * value);
}

function seededUnit(index: number) {
  const value = Math.sin(index * 12.9898) * 43758.5453;

  return value - Math.floor(value);
}

function formatCameraError(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function phasedPathProgress(progress: number, phase = 0) {
  const clampedPhase = THREE.MathUtils.clamp(phase, 0, 0.42);
  const delayed = THREE.MathUtils.clamp(
    (progress - clampedPhase) / (1 - clampedPhase),
    0,
    1
  );
  const blended = THREE.MathUtils.lerp(progress, delayed, 0.42);

  return THREE.MathUtils.clamp(blended, 0, 1);
}

function writeQuadraticBezier(
  target: THREE.Vector3,
  start: [number, number, number],
  control: [number, number, number],
  end: [number, number, number],
  progress: number
) {
  const inverse = 1 - progress;
  const startWeight = inverse * inverse;
  const controlWeight = 2 * inverse * progress;
  const endWeight = progress * progress;

  target.set(
    start[0] * startWeight + control[0] * controlWeight + end[0] * endWeight,
    start[1] * startWeight + control[1] * controlWeight + end[1] * endWeight,
    start[2] * startWeight + control[2] * controlWeight + end[2] * endWeight
  );
}

function resolvePathPosition(
  target: THREE.Vector3,
  basePosition: [number, number, number],
  explodedPosition: [number, number, number],
  midPosition: [number, number, number] | undefined,
  progress: number
) {
  if (!midPosition) {
    writeQuadraticBezier(
      target,
      basePosition,
      [
        (basePosition[0] + explodedPosition[0]) * 0.5,
        (basePosition[1] + explodedPosition[1]) * 0.5,
        (basePosition[2] + explodedPosition[2]) * 0.5,
      ],
      explodedPosition,
      progress
    );

    return target;
  }

  writeQuadraticBezier(
    target,
    basePosition,
    midPosition,
    explodedPosition,
    progress
  );

  return target;
}

function wrappedSlot(partIndex: number, activePartIndex: number, totalParts: number) {
  const rawSlot = partIndex - activePartIndex;
  const wrapped = ((rawSlot % totalParts) + totalParts) % totalParts;

  return wrapped > totalParts / 2 ? wrapped - totalParts : wrapped;
}

function createLogoParticles() {
  const particles: LogoParticle[] = [];
  const cellSize = 0.066;
  const letterGap = 1.5;
  const logoY = -1.12;
  const logoZ = 1.02;
  let cursor = 0;

  LOGO_TEXT.split("").forEach((letter) => {
    if (letter === " ") {
      cursor += 2.1;
      return;
    }

    const grid = LOGO_LETTERS[letter];

    grid.forEach((row, rowIndex) => {
      row.split("").forEach((filled, columnIndex) => {
        if (filled !== "1") return;

        const x = (cursor + columnIndex) * cellSize;
        const y = (3 - rowIndex) * cellSize;
        const home = new THREE.Vector3(x, y, logoZ);
        const seed = particles.length + 1;
        const scatter = new THREE.Vector3();

        particles.push({ home, scatter, seed });
      });
    });

    cursor += 5 + letterGap;
  });

  const centerX =
    particles.reduce((total, particle) => total + particle.home.x, 0) /
    particles.length;

  particles.forEach((particle) => {
    particle.home.x -= centerX;
    particle.home.y += logoY;
    const horizontal = particle.home.x >= 0 ? 1 : -1;
    const vertical = particle.home.y >= logoY ? 1 : -1;

    particle.scatter.set(
      particle.home.x * 1.22 +
        horizontal * (0.74 + seededUnit(particle.seed * 4) * 0.58),
      particle.home.y * 1.04 +
        vertical * (0.34 + seededUnit(particle.seed * 5) * 0.34),
      -1.68 - seededUnit(particle.seed * 6) * 1.05
    );
  });

  return particles;
}

function Part({
  partIndex,
  basePosition,
  midPosition,
  explodedPosition,
  focusScale = 1.45,
  secondaryScale = 0.74,
  inspectZFocus = 0.9,
  explodeDelay = 0,
  assembleDelay = 0,
  baseRotation = [0, 0, 0],
  explodedRotation = [0, 0, 0],
  selfRotationAmount = 1,
  motionSeed = 0,
  meshOpacity = 1,
  children,
  progressRef,
  activePartIndex,
  totalParts,
  carouselEnabled,
  inspectMode,
  inspectRotationRef,
}: {
  partIndex: number;
  basePosition: [number, number, number];
  midPosition?: [number, number, number];
  explodedPosition: [number, number, number];
  focusScale?: number;
  secondaryScale?: number;
  inspectZFocus?: number;
  explodeDelay?: number;
  assembleDelay?: number;
  baseRotation?: [number, number, number];
  explodedRotation?: [number, number, number];
  selfRotationAmount?: number;
  motionSeed?: number;
  meshOpacity?: number;
  children: ReactNode;
  progressRef: MutableRefObject<number>;
  activePartIndex: number;
  totalParts: number;
  carouselEnabled: boolean;
  inspectMode: boolean;
  inspectRotationRef: MutableRefObject<{ x: number; y: number }>;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const activeLightRef = useRef<THREE.PointLight>(null);
  const selfRotationRef = useRef(new THREE.Euler(0, 0, 0));
  const manualRotationRef = useRef(new THREE.Vector2());
  const manualRotationTargetRef = useRef(new THREE.Vector2());
  const partScaleRef = useRef(1);
  const highlightRef = useRef(0);
  const previousProgressRef = useRef(0);
  const previousRawProgressRef = useRef(0);
  const dockedRef = useRef(true);
  const dockPhaseRef = useRef<number | null>(null);
  const pathPositionRef = useRef(new THREE.Vector3());
  const renderPositionRef = useRef(new THREE.Vector3());
  const targetPositionRef = useRef(new THREE.Vector3());
  const inspectPositionRef = useRef(new THREE.Vector3());
  const positionInitializedRef = useRef(false);
  const inspectBlendRef = useRef(0);
  const dimRef = useRef(0);
  const inspectIdleYRef = useRef(0);
  const motion = useMemo(() => {
    const direction = motionSeed % 2 === 0 ? 1 : -1;
    const dockStart = midPosition ?? explodedPosition;
    const travel = new THREE.Vector3(
      basePosition[0] - dockStart[0],
      basePosition[1] - dockStart[1],
      basePosition[2] - dockStart[2]
    );
    const travelDistance = travel.length();

    return {
      floatSpeed: 0.55 + (motionSeed % 5) * 0.08,
      floatAmount: (0.024 + (motionSeed % 4) * 0.005) * selfRotationAmount,
      driftAmount: (0.012 + (motionSeed % 3) * 0.004) * selfRotationAmount,
      dockAmplitude:
        travelDistance > 0.001
          ? Math.min(travelDistance * 0.045, 0.07) *
            (0.86 + (motionSeed % 3) * 0.08)
          : 0,
      dockDirection:
        travelDistance > 0.001 ? travel.normalize() : new THREE.Vector3(),
      dockRotation: new THREE.Vector3(
        0,
        -direction * (0.018 + (motionSeed % 4) * 0.004),
        0
      ),
      rotationSpeed: new THREE.Vector3(
        0,
        -direction * (0.06 + (motionSeed % 4) * 0.012) * selfRotationAmount,
        0
      ),
    };
  }, [basePosition, explodedPosition, midPosition, motionSeed, selfRotationAmount]);

  useFrame(({ clock }, delta) => {
    if (!groupRef.current) return;

    const t = clock.getElapsedTime();
    const rawProgress = progressRef.current;
    const previousRawProgress = previousRawProgressRef.current;
    const assembling = rawProgress < previousRawProgress;
    const localProgress = assembling
      ? 1 - phasedPathProgress(1 - rawProgress, assembleDelay)
      : phasedPathProgress(rawProgress, explodeDelay);
    const previousProgress = previousProgressRef.current;
    const p = smoothStep(localProgress);
    const separatedProgress = smoothStep(
      THREE.MathUtils.clamp((localProgress - 0.22) / 0.78, 0, 1)
    );
    const floatPhase = t * motion.floatSpeed + motionSeed;
    const pathPosition = resolvePathPosition(
      pathPositionRef.current,
      basePosition,
      explodedPosition,
      midPosition,
      p
    );
    const dockThreshold = 0.045;
    const slot = wrappedSlot(partIndex, activePartIndex, totalParts);
    const isInspectActive = inspectMode && slot === 0;
    const angle = (slot / totalParts) * Math.PI * 2;
    const activePresence = carouselEnabled && slot === 0 ? separatedProgress : 0;
    const carouselPresence = carouselEnabled ? separatedProgress : 0;
    const inspectPresence = inspectMode && carouselEnabled ? separatedProgress : 0;
    const depth = Math.cos(angle);
    const side = Math.abs(Math.sin(angle));
    const slotDistance = Math.abs(slot);
    const carouselScale =
      slot === 0
        ? focusScale
        : THREE.MathUtils.lerp(0.5, secondaryScale, Math.max(depth, 0) * 0.55);
    const inspectScale = slot === 0 ? focusScale * 1.12 : 0.22;
    const scaleTarget = THREE.MathUtils.lerp(
      THREE.MathUtils.lerp(0, carouselScale, carouselPresence),
      inspectScale,
      inspectPresence
    );
    const highlightTarget = activePresence * (inspectMode ? 1.0 : 0.86);

    if (!assembling && localProgress > dockThreshold * 2) {
      dockedRef.current = false;
      dockPhaseRef.current = null;
    }

    if (
      assembling &&
      !dockedRef.current &&
      previousProgress > dockThreshold &&
      localProgress <= dockThreshold
    ) {
      dockedRef.current = true;
      dockPhaseRef.current = 0;
    }

    let dockPulse = 0;

    if (dockPhaseRef.current !== null) {
      dockPhaseRef.current += delta;
      dockPulse =
        Math.sin(dockPhaseRef.current * 15) *
        Math.exp(-dockPhaseRef.current * 5.2);

      if (dockPhaseRef.current > 1.1) {
        dockPhaseRef.current = null;
      }
    }

    const activeRotationBoost = activePresence && !isInspectActive ? 2.1 : 1;

    if (!isInspectActive) {
      // Decay any legacy X/Z to zero — only Y (horizontal) rotation accumulates.
      selfRotationRef.current.x = THREE.MathUtils.lerp(
        selfRotationRef.current.x, 0, 1 - Math.exp(-delta * 2.5)
      );
      selfRotationRef.current.y +=
        delta * motion.rotationSpeed.y * separatedProgress * activeRotationBoost;
      selfRotationRef.current.z = THREE.MathUtils.lerp(
        selfRotationRef.current.z, 0, 1 - Math.exp(-delta * 2.5)
      );
    }

    if (isInspectActive && separatedProgress > 0.1) {
      inspectIdleYRef.current += delta * 0.26;
    }
    partScaleRef.current = THREE.MathUtils.lerp(
      partScaleRef.current,
      scaleTarget,
      1 - Math.exp(-delta * 7)
    );
    highlightRef.current = THREE.MathUtils.lerp(
      highlightRef.current,
      highlightTarget,
      1 - Math.exp(-delta * 6)
    );
    if (activeLightRef.current) {
      // Warm overhead spotlight — selection presence without color distortion.
      activeLightRef.current.intensity =
        highlightRef.current * (isInspectActive ? 1.1 : 0.22);
    }
    inspectBlendRef.current = THREE.MathUtils.lerp(
      inspectBlendRef.current,
      inspectPresence,
      1 - Math.exp(-delta * 4.8)
    );

    const normalX =
      pathPosition.x +
      Math.cos(floatPhase * 0.7) * motion.driftAmount * separatedProgress +
      motion.dockDirection.x * motion.dockAmplitude * dockPulse;
    const normalY =
      pathPosition.y +
      Math.sin(floatPhase) * motion.floatAmount * separatedProgress +
      motion.dockDirection.y * motion.dockAmplitude * dockPulse;
    const normalZ =
      pathPosition.z +
      Math.sin(floatPhase * 0.8) * motion.driftAmount * separatedProgress +
      motion.dockDirection.z * motion.dockAmplitude * dockPulse;
    const neighborSpread = slotDistance === 1 ? 0.85 : slotDistance === 2 ? 0.35 : 0;
    const neighborDepthOffset = slotDistance === 1 ? 0.55 : 0;
    const carouselX = Math.sin(angle) * (3.8 + neighborSpread);
    const carouselY = 0.1 - depth * 0.52;
    const carouselZ = depth * 2.1 - side * 0.65 - neighborDepthOffset;
    const rearDirection = slot === 0 ? 0 : Math.sign(slot) || 1;
    const inspectX = slot === 0 ? 0 : rearDirection * (4.2 + slotDistance * 0.85);
    const inspectY = slot === 0 ? 0.0 : -0.15 - slotDistance * 0.12;
    const inspectZ = slot === 0 ? inspectZFocus : -4.5 - slotDistance * 1.8;
    inspectPositionRef.current.set(inspectX, inspectY, inspectZ);

    const dimTarget = inspectMode && slot !== 0 ? 1 : 0;
    dimRef.current = THREE.MathUtils.lerp(
      dimRef.current,
      dimTarget,
      1 - Math.exp(-delta * 3.2)
    );

    targetPositionRef.current.set(
      THREE.MathUtils.lerp(normalX, carouselX, carouselPresence),
      THREE.MathUtils.lerp(normalY, carouselY, carouselPresence),
      THREE.MathUtils.lerp(normalZ, carouselZ, carouselPresence)
    );
    targetPositionRef.current.lerp(
      inspectPositionRef.current,
      inspectBlendRef.current
    );
    if (!positionInitializedRef.current) {
      renderPositionRef.current.copy(targetPositionRef.current);
      positionInitializedRef.current = true;
    }

    renderPositionRef.current.lerp(
      targetPositionRef.current,
      1 - Math.exp(-delta * 5.5)
    );
    groupRef.current.position.copy(renderPositionRef.current);
    manualRotationTargetRef.current.set(
      0,
      inspectMode && activePresence ? inspectRotationRef.current.y : 0
    );
    manualRotationRef.current.lerp(
      manualRotationTargetRef.current,
      1 - Math.exp(-delta * 6)
    );

    groupRef.current.rotation.set(
      THREE.MathUtils.lerp(baseRotation[0], explodedRotation[0], separatedProgress) +
        motion.dockRotation.x * dockPulse,
      THREE.MathUtils.lerp(baseRotation[1], explodedRotation[1], separatedProgress) +
        (isInspectActive
          ? inspectIdleYRef.current
          : selfRotationRef.current.y * separatedProgress) +
        manualRotationRef.current.y +
        motion.dockRotation.y * dockPulse,
      THREE.MathUtils.lerp(baseRotation[2], explodedRotation[2], separatedProgress) +
        motion.dockRotation.z * dockPulse
    );
    groupRef.current.scale.setScalar(partScaleRef.current);

    groupRef.current.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const material = mesh.material as THREE.MeshStandardMaterial | undefined;

      if (!material || !("emissiveIntensity" in material)) return;

      // Only manage dim/opacity — never override each material's own emissive color.
      // Food surfaces keep their natural colors; only brand accent rings carry emissive.
      const emergeFade = smoothStep(THREE.MathUtils.clamp(carouselPresence * 2.5, 0, 1));
      material.transparent = meshOpacity < 1 || dimRef.current > 0.005 || emergeFade < 1;
      material.opacity = THREE.MathUtils.lerp(meshOpacity, 0.07, dimRef.current) * emergeFade;
    });

    previousProgressRef.current = localProgress;
    previousRawProgressRef.current = rawProgress;
  });

  return (
    <group ref={groupRef}>
      {children}
      <pointLight
        ref={activeLightRef}
        color="#fff4e0"
        distance={3.2}
        intensity={0}
        position={[0, 1.1, 0.6]}
      />
    </group>
  );
}

function BurgerModel({ explodeActive }: { explodeActive: boolean }) {
  const { scene } = useGLTF("/models/burger.glb");
  const groupRef = useRef<THREE.Group>(null);
  const explodeFadeRef = useRef(1);

  const { center, normalizedScale } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const c = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    return { center: c, normalizedScale: maxDim > 0 ? 0.92 / maxDim : 1 };
  }, [scene]);

  useEffect(() => {
    scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mats = Array.isArray(obj.material)
        ? (obj.material as THREE.MeshStandardMaterial[])
        : [obj.material as THREE.MeshStandardMaterial];
      mats.forEach((m) => {
        if (!m.isMeshStandardMaterial) return;
        m.transparent = true;
        m.roughness = Math.max(m.roughness, 0.55);
      });
    });
  }, [scene]);

  useFrame((_, delta) => {
    explodeFadeRef.current = THREE.MathUtils.lerp(
      explodeFadeRef.current,
      explodeActive ? 0 : 1,
      1 - Math.exp(-delta * 3.5)
    );
    if (!groupRef.current) return;
    groupRef.current.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mats = Array.isArray(obj.material)
        ? (obj.material as THREE.MeshStandardMaterial[])
        : [obj.material as THREE.MeshStandardMaterial];
      mats.forEach((m) => {
        if (!m.isMeshStandardMaterial) return;
        m.opacity = explodeFadeRef.current;
      });
    });
  });

  return (
    <group ref={groupRef} scale={normalizedScale}>
      <group position={[-center.x, -center.y, -center.z]}>
        <primitive object={scene} />
      </group>
    </group>
  );
}
useGLTF.preload("/models/burger.glb");

function FoodModel({
  path,
  targetSize = 0.92,
  positionOffset = [0, 0, 0] as [number, number, number],
  rotationOffset = [0, 0, 0] as [number, number, number],
}: {
  path: string;
  targetSize?: number;
  positionOffset?: [number, number, number];
  rotationOffset?: [number, number, number];
}) {
  const { scene } = useGLTF(path);

  const { center, normalizedScale } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const c = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    return { center: c, normalizedScale: maxDim > 0 ? targetSize / maxDim : 1 };
  }, [scene, targetSize]);

  useEffect(() => {
    scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mats = Array.isArray(obj.material)
        ? (obj.material as THREE.MeshStandardMaterial[])
        : [obj.material as THREE.MeshStandardMaterial];
      mats.forEach((m) => {
        if (!m.isMeshStandardMaterial) return;
        m.transparent = true;
        m.roughness = Math.max(m.roughness, 0.55);
      });
    });
  }, [scene]);

  return (
    <group scale={normalizedScale} position={positionOffset} rotation={rotationOffset}>
      <group position={[-center.x, -center.y, -center.z]}>
        <primitive object={scene} />
      </group>
    </group>
  );
}
useGLTF.preload("/models/california-roll.glb");
useGLTF.preload("/models/fries.glb");
useGLTF.preload("/models/coffee.glb");
useGLTF.preload("/models/ice-cream.glb");

// Assembled Y positions match the stacked burger look; exploded positions give 0.65 spacing.
const BURGER_ASSEMBLED_Y = [-0.40, -0.23, -0.13, -0.03, 0.04, 0.09, 0.22] as const;
const BURGER_EXPLODED_Y  = [-1.20, -0.68, -0.32,  0.02, 0.36, 0.70, 1.20] as const;

function BurgerExplodedView({ active }: { active: boolean }) {
  const progressRef = useRef(0);
  const wrapperRef = useRef<THREE.Group>(null);
  const g0 = useRef<THREE.Group>(null);
  const g1 = useRef<THREE.Group>(null);
  const g2 = useRef<THREE.Group>(null);
  const g3 = useRef<THREE.Group>(null);
  const g4 = useRef<THREE.Group>(null);
  const g5 = useRef<THREE.Group>(null);
  const g6 = useRef<THREE.Group>(null);

  useFrame(({ clock }, delta) => {
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 2.0)
    );
    if (wrapperRef.current) {
      wrapperRef.current.visible = active || progressRef.current > 0.02;
    }
    const p = smoothStep(progressRef.current);
    const t = clock.getElapsedTime();

    const yFor = (assembled: number, exploded: number, i: number) =>
      THREE.MathUtils.lerp(assembled, exploded, p) +
      Math.sin(t * 0.38 + i * 0.72) * 0.010 * p;

    if (g0.current) g0.current.position.y = yFor(BURGER_ASSEMBLED_Y[0], BURGER_EXPLODED_Y[0], 0);
    if (g1.current) g1.current.position.y = yFor(BURGER_ASSEMBLED_Y[1], BURGER_EXPLODED_Y[1], 1);
    if (g2.current) g2.current.position.y = yFor(BURGER_ASSEMBLED_Y[2], BURGER_EXPLODED_Y[2], 2);
    if (g3.current) g3.current.position.y = yFor(BURGER_ASSEMBLED_Y[3], BURGER_EXPLODED_Y[3], 3);
    if (g4.current) g4.current.position.y = yFor(BURGER_ASSEMBLED_Y[4], BURGER_EXPLODED_Y[4], 4);
    if (g5.current) g5.current.position.y = yFor(BURGER_ASSEMBLED_Y[5], BURGER_EXPLODED_Y[5], 5);
    if (g6.current) g6.current.position.y = yFor(BURGER_ASSEMBLED_Y[6], BURGER_EXPLODED_Y[6], 6);
  });

  return (
    <group ref={wrapperRef}>
      {/* Layer 0 — Bottom Bun */}
      <group ref={g0}>
        <mesh>
          <cylinderGeometry args={[0.42, 0.44, 0.18, 28]} />
          <meshStandardMaterial color="#c87941" metalness={0.06} roughness={0.80} />
        </mesh>
        <mesh position={[0, -0.10, 0]}>
          <cylinderGeometry args={[0.44, 0.44, 0.04, 28]} />
          <meshStandardMaterial color="#a86030" metalness={0.05} roughness={0.86} />
        </mesh>
        <mesh position={[0, -0.10, 0]}>
          <torusGeometry args={[0.46, 0.008, 8, 40]} />
          <meshStandardMaterial color="#00d4ff" emissive="#00d4ff" emissiveIntensity={0.5} metalness={0.5} roughness={0.1} />
        </mesh>
      </group>

      {/* Layer 1 — Truffle Sauce */}
      <group ref={g1}>
        <mesh>
          <cylinderGeometry args={[0.38, 0.39, 0.07, 24]} />
          <meshStandardMaterial color="#d4b050" metalness={0.05} roughness={0.74} />
        </mesh>
        <mesh position={[0, -0.03, 0]}>
          <cylinderGeometry args={[0.41, 0.41, 0.02, 24]} />
          <meshStandardMaterial color="#e0c060" metalness={0.04} roughness={0.78} />
        </mesh>
      </group>

      {/* Layer 2 — Wagyu Patty */}
      <group ref={g2}>
        <mesh>
          <cylinderGeometry args={[0.40, 0.42, 0.20, 24]} />
          <meshStandardMaterial color="#2e1208" metalness={0.10} roughness={0.88} />
        </mesh>
        <mesh position={[0, 0.09, 0]}>
          <cylinderGeometry args={[0.40, 0.40, 0.025, 24]} />
          <meshStandardMaterial color="#180804" metalness={0.08} roughness={0.92} />
        </mesh>
        {([0, 1, 2] as const).map((i) => (
          <mesh key={i} position={[(i - 1) * 0.14, 0.10, 0]} rotation={[0, 0, 0.08]}>
            <boxGeometry args={[0.038, 0.012, 0.72]} />
            <meshStandardMaterial color="#0c0604" metalness={0.1} roughness={0.94} />
          </mesh>
        ))}
      </group>

      {/* Layer 3 — Aged Cheddar */}
      <group ref={g3}>
        <mesh>
          <boxGeometry args={[0.76, 0.055, 0.76]} />
          <meshStandardMaterial color="#f4b428" metalness={0.06} roughness={0.64} />
        </mesh>
        <mesh position={[0.37, -0.024, 0]}>
          <boxGeometry args={[0.04, 0.055, 0.72]} />
          <meshStandardMaterial color="#e8a820" metalness={0.05} roughness={0.70} />
        </mesh>
        <mesh position={[-0.37, -0.024, 0]}>
          <boxGeometry args={[0.04, 0.055, 0.72]} />
          <meshStandardMaterial color="#e8a820" metalness={0.05} roughness={0.70} />
        </mesh>
      </group>

      {/* Layer 4 — Tomato */}
      <group ref={g4}>
        <mesh>
          <cylinderGeometry args={[0.37, 0.38, 0.08, 24]} />
          <meshStandardMaterial color="#c83428" metalness={0.08} roughness={0.70} />
        </mesh>
        {([0, 1, 2, 3, 4, 5] as const).map((i) => (
          <mesh
            key={i}
            position={[
              Math.cos((i / 6) * Math.PI * 2) * 0.18,
              0.038,
              Math.sin((i / 6) * Math.PI * 2) * 0.18,
            ]}
          >
            <sphereGeometry args={[0.030, 6, 5]} />
            <meshStandardMaterial color="#f8d0c0" metalness={0.04} roughness={0.82} />
          </mesh>
        ))}
      </group>

      {/* Layer 5 — Crisp Lettuce */}
      <group ref={g5}>
        <mesh>
          <cylinderGeometry args={[0.46, 0.46, 0.046, 24]} />
          <meshStandardMaterial color="#4a8c3a" metalness={0.04} roughness={0.88} />
        </mesh>
        <mesh position={[0, 0.018, 0]}>
          <cylinderGeometry args={[0.36, 0.36, 0.02, 24]} />
          <meshStandardMaterial color="#6ab050" metalness={0.04} roughness={0.86} />
        </mesh>
        <mesh>
          <torusGeometry args={[0.44, 0.022, 8, 32]} />
          <meshStandardMaterial color="#3a7230" metalness={0.04} roughness={0.90} />
        </mesh>
      </group>

      {/* Layer 6 — Top Bun */}
      <group ref={g6}>
        <mesh position={[0, -0.07, 0]}>
          <cylinderGeometry args={[0.40, 0.43, 0.20, 28]} />
          <meshStandardMaterial color="#d4874a" metalness={0.06} roughness={0.78} />
        </mesh>
        <mesh position={[0, 0.07, 0]}>
          <sphereGeometry args={[0.38, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.52]} />
          <meshStandardMaterial color="#c87941" metalness={0.05} roughness={0.80} />
        </mesh>
        {([0, 1, 2, 3, 4, 5, 6, 7] as const).map((i) => (
          <mesh
            key={i}
            position={[
              Math.cos((i / 8) * Math.PI * 2) * (0.16 + seededUnit(i * 7) * 0.1),
              0.15,
              Math.sin((i / 8) * Math.PI * 2) * (0.16 + seededUnit(i * 7) * 0.1),
            ]}
          >
            <sphereGeometry args={[0.022, 6, 5]} />
            <meshStandardMaterial color="#f0e8c8" metalness={0.06} roughness={0.70} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function MenuBook({ open }: { open: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const coverRef = useRef<THREE.Group>(null);
  const bookLightRef = useRef<THREE.PointLight>(null);
  const progressRef = useRef(0);

  useEffect(() => {
    groupRef.current?.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      const mat = mesh.material as THREE.MeshStandardMaterial | undefined;
      if (mat && "opacity" in mat) mesh.userData.baseOp = mat.opacity;
    });
  }, []);

  useFrame((_, delta) => {
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      open ? 1 : 0,
      1 - Math.exp(-delta * 1.55)
    );
    const p = smoothStep(progressRef.current);
    const bookVisible = 1 - smoothStep(Math.min(p * 2.2, 1));

    if (groupRef.current) {
      groupRef.current.scale.setScalar(THREE.MathUtils.lerp(1.0, 0.42, p));
      groupRef.current.position.y = THREE.MathUtils.lerp(0, -0.55, p);
      groupRef.current.rotation.x = THREE.MathUtils.lerp(0, 0.14, p);

      groupRef.current.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        const mat = mesh.material as THREE.MeshStandardMaterial | undefined;
        if (!mat || !("opacity" in mat)) return;
        mat.transparent = true;
        mat.opacity = ((mesh.userData.baseOp as number) ?? 1) * bookVisible;
      });
    }

    if (coverRef.current) {
      coverRef.current.rotation.y = THREE.MathUtils.lerp(0, Math.PI * 0.52, p);
    }

    if (bookLightRef.current) {
      bookLightRef.current.intensity = bookVisible * 0.55;
    }
  });

  return (
    <group ref={groupRef}>
      <pointLight
        ref={bookLightRef}
        color="#fff6e0"
        distance={6}
        intensity={0}
        position={[0, 2.8, 2.2]}
      />

      {/* Soft shadow beneath book */}
      <mesh position={[0, -0.94, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.88, 36]} />
        <meshStandardMaterial
          color="#c0a882"
          transparent
          opacity={0.20}
          roughness={1}
          depthWrite={false}
        />
      </mesh>

      {/* Back cover */}
      <mesh position={[0, 0, -0.078]}>
        <boxGeometry args={[1.38, 1.86, 0.058]} />
        <meshStandardMaterial color="#ede8df" metalness={0.08} roughness={0.72} transparent opacity={1} />
      </mesh>

      {/* Pages block */}
      <mesh position={[0.01, 0, 0]}>
        <boxGeometry args={[1.30, 1.78, 0.12]} />
        <meshStandardMaterial color="#f8f5f0" metalness={0.04} roughness={0.84} transparent opacity={1} />
      </mesh>

      {/* Spine */}
      <mesh position={[-0.695, 0, 0]}>
        <boxGeometry args={[0.042, 1.86, 0.26]} />
        <meshStandardMaterial color="#c4a070" metalness={0.22} roughness={0.54} transparent opacity={1} />
      </mesh>

      {/* Front cover — group pivoted at spine edge for page-opening motion */}
      <group ref={coverRef} position={[-0.695, 0, 0.078]}>
        <mesh position={[0.695, 0, 0]}>
          <boxGeometry args={[1.38, 1.86, 0.058]} />
          <meshStandardMaterial color="#e6ddd0" metalness={0.10} roughness={0.66} transparent opacity={1} />
        </mesh>
        {/* Top gold rule */}
        <mesh position={[0.695, 0.70, 0.030]}>
          <boxGeometry args={[0.94, 0.007, 0.006]} />
          <meshStandardMaterial
            color="#c8a050"
            emissive="#9a7420"
            emissiveIntensity={0.45}
            metalness={0.65}
            roughness={0.28}
            transparent
            opacity={1}
          />
        </mesh>
        {/* Bottom gold rule */}
        <mesh position={[0.695, -0.70, 0.030]}>
          <boxGeometry args={[0.94, 0.007, 0.006]} />
          <meshStandardMaterial
            color="#c8a050"
            emissive="#9a7420"
            emissiveIntensity={0.45}
            metalness={0.65}
            roughness={0.28}
            transparent
            opacity={1}
          />
        </mesh>
        {/* Menu title plate */}
        <mesh position={[0.695, 0.12, 0.030]}>
          <boxGeometry args={[0.76, 0.038, 0.006]} />
          <meshStandardMaterial
            color="#d4b068"
            emissive="#a07828"
            emissiveIntensity={0.28}
            metalness={0.55}
            roughness={0.32}
            transparent
            opacity={1}
          />
        </mesh>
        {/* Subtitle plate */}
        <mesh position={[0.695, -0.04, 0.030]}>
          <boxGeometry args={[0.52, 0.018, 0.006]} />
          <meshStandardMaterial
            color="#c4a058"
            emissive="#906820"
            emissiveIntensity={0.22}
            metalness={0.50}
            roughness={0.36}
            transparent
            opacity={1}
          />
        </mesh>
      </group>
    </group>
  );
}

function SpatialMenuCarousel({
  exploded,
  activePartIndex,
  inspectMode,
  inspectRotationRef,
  burgerExploded,
}: {
  exploded: boolean;
  activePartIndex: number;
  inspectMode: boolean;
  inspectRotationRef: MutableRefObject<{ x: number; y: number }>;
  burgerExploded: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const progressRef = useRef(0);
  const n = CAROUSEL_PARTS.length;

  useFrame(({ clock }, delta) => {
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      exploded ? 1 : 0,
      1 - Math.exp(-delta * 1.4)
    );

    if (!groupRef.current) return;

    const p = smoothStep(progressRef.current);
    const t = clock.getElapsedTime();

    groupRef.current.rotation.y += delta * 0.08 * (1 - p);
    groupRef.current.position.y = Math.sin(t * 0.38) * 0.025 * (1 - p);
  });

  return (
    <group ref={groupRef} scale={0.88}>

      {/* ── Item 0: Signature Burger Combo ── */}
      <Part
        partIndex={0}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        basePosition={[0, 0, 0]}
        midPosition={[0, 0.92, 1.05]}
        explodedPosition={[0, 0.18, 0]}
        focusScale={1.22}
        secondaryScale={0.66}
        inspectZFocus={0.9}
        selfRotationAmount={0.12}
        motionSeed={1}
      >
        <BurgerModel explodeActive={burgerExploded && inspectMode && activePartIndex === 0} />
        <BurgerExplodedView active={burgerExploded && inspectMode && activePartIndex === 0} />
      </Part>

      {/* ── Item 1: Crispy Chicken Combo ── */}
      <Part
        partIndex={1}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        basePosition={[0, 0, 0]}
        midPosition={[-0.12, 0.86, 0.98]}
        explodedPosition={[0, 0.22, 0]}
        focusScale={1.32}
        secondaryScale={0.64}
        inspectZFocus={1.5}
        explodeDelay={0.022}
        assembleDelay={0.042}
        selfRotationAmount={0.14}
        motionSeed={2}
      >
        <FoodModel
          path="/models/california-roll.glb"
          targetSize={0.88}
          rotationOffset={[0.18, 0, 0]}
        />
      </Part>

      {/* ── Item 2: Classic Fries Set ── */}
      <Part
        partIndex={2}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        basePosition={[0, 0, 0]}
        midPosition={[0, 0.80, 1.08]}
        explodedPosition={[0, 0.15, 0]}
        focusScale={1.2}
        secondaryScale={0.64}
        inspectZFocus={1.1}
        explodeDelay={0.04}
        assembleDelay={0.06}
        selfRotationAmount={0.16}
        motionSeed={3}
      >
        <FoodModel
          path="/models/fries.glb"
          targetSize={0.92}
        />
      </Part>

      {/* ── Item 3: Artisan Coffee Set ── */}
      <Part
        partIndex={3}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        basePosition={[0, 0, 0]}
        midPosition={[0.14, 0.88, 0.92]}
        explodedPosition={[0, 0.2, 0]}
        focusScale={1.38}
        secondaryScale={0.66}
        inspectZFocus={1.6}
        explodeDelay={0.055}
        assembleDelay={0.04}
        selfRotationAmount={0.18}
        motionSeed={4}
      >
        <FoodModel
          path="/models/coffee.glb"
          targetSize={0.88}
        />
      </Part>

      {/* ── Item 4: Chef's Dessert Combo ── */}
      <Part
        partIndex={4}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        basePosition={[0, 0, 0]}
        midPosition={[0, 0.84, 1.02]}
        explodedPosition={[0, 0.16, 0]}
        focusScale={1.28}
        secondaryScale={0.66}
        inspectZFocus={1.25}
        explodeDelay={0.07}
        assembleDelay={0.025}
        selfRotationAmount={0.2}
        motionSeed={5}
      >
        <FoodModel
          path="/models/ice-cream.glb"
          targetSize={0.92}
        />
      </Part>

    </group>
  );
}

function AmbientParticles() {
  const pointsRef = useRef<THREE.Points>(null);

  const positions = useMemo(() => {
    const count = 700;
    const array = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      array[i * 3] = (seededUnit(i * 3 + 1) - 0.5) * 14;
      array[i * 3 + 1] = (seededUnit(i * 3 + 2) - 0.5) * 8;
      array[i * 3 + 2] = (seededUnit(i * 3 + 3) - 0.5) * 10;
    }

    return array;
  }, []);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;
    pointsRef.current.rotation.y += delta * 0.025;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#a08868"
        size={0.013}
        transparent
        opacity={0.16}
        depthWrite={false}
      />
    </points>
  );
}

function AuraLogoParticles({ exploded }: { exploded: boolean }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const progressRef = useRef(0);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const particles = useMemo(() => createLogoParticles(), []);

  useFrame(({ clock }, delta) => {
    if (!meshRef.current) return;

    const mesh = meshRef.current;

    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      exploded ? 1 : 0,
      1 - Math.exp(-delta * 2.35)
    );

    const t = clock.getElapsedTime();
    const p = smoothStep(progressRef.current);
    const scale = THREE.MathUtils.lerp(0.042, 0.024, p);

    particles.forEach((particle, index) => {
      const idle = Math.sin(t * 0.42 + particle.seed) * 0.008 * (1 - p);
      const shockDrift = Math.sin(t * 0.32 + particle.seed * 0.7) * 0.024 * p;

      dummy.position.set(
        THREE.MathUtils.lerp(particle.home.x, particle.scatter.x, p) +
          shockDrift,
        THREE.MathUtils.lerp(particle.home.y, particle.scatter.y, p) +
          idle,
        THREE.MathUtils.lerp(particle.home.z, particle.scatter.z, p)
      );
      dummy.rotation.set(
        p * (0.35 + seededUnit(particle.seed * 2) * 0.65),
        p * (0.45 + seededUnit(particle.seed * 3) * 0.8),
        p * 0.18
      );
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
    const material = mesh.material as THREE.MeshStandardMaterial;
    material.opacity = THREE.MathUtils.lerp(0.38, 0.10, p);
    material.emissiveIntensity = THREE.MathUtils.lerp(0.06, 0.02, p);
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, particles.length]}
      frustumCulled={false}
    >
      <boxGeometry args={[1, 1, 0.18]} />
      <meshStandardMaterial
        color="#b07840"
        emissive="#6a4818"
        emissiveIntensity={0.06}
        metalness={0.28}
        opacity={0.38}
        roughness={0.58}
        transparent
      />
    </instancedMesh>
  );
}

function InspectSceneLighting({ inspectMode }: { inspectMode: boolean }) {
  const keyLightRef = useRef<THREE.DirectionalLight>(null);
  const rimLightRef = useRef<THREE.PointLight>(null);
  const blendRef = useRef(0);

  useFrame((_, delta) => {
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      inspectMode ? 1 : 0,
      1 - Math.exp(-delta * 4.5)
    );

    if (keyLightRef.current) {
      keyLightRef.current.intensity = blendRef.current * 1.2;
    }

    if (rimLightRef.current) {
      rimLightRef.current.intensity = blendRef.current * 0.65;
    }
  });

  return (
    <>
      <directionalLight
        ref={keyLightRef}
        position={[-1.8, 5.5, 3.5]}
        color="#fff8e8"
        intensity={0}
      />
      <pointLight
        ref={rimLightRef}
        position={[3.5, 1.8, -1.2]}
        color="#ffe0a0"
        distance={14}
        intensity={0}
      />
    </>
  );
}

function CameraGestureLayer({
  onGesture,
  inspectMode,
}: {
  onGesture: (action: GestureAction) => void;
  inspectMode: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const handLandmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isInitializingRef = useRef(false);
  const frameRef = useRef<number | null>(null);
  const runDetectionLoopRef = useRef<() => void>(() => undefined);
  const samplesRef = useRef<Array<{ time: number; x: number }>>([]);
  const palmSizeSamplesRef = useRef<Array<{ time: number; size: number }>>([]);
  const statusRef = useRef<CameraGestureStatus>("CAMERA OFF");
  const statusHoldUntilRef = useRef(0);
  const lastDetectionAtRef = useRef(0);
  const lastSwipeDirectionRef = useRef<"left" | "right" | null>(null);
  const lastSwipeTimeRef = useRef(0);
  const lastInspectTimeRef = useRef(0);
  const lastExitInspectTimeRef = useRef(0);
  const lastFistAddTimeRef = useRef(0);
  const fistPulseCountRef = useRef(0);       // 0=idle, 1=armed (first pulse seen)
  const firstFistTimeRef = useRef(0);        // timestamp of first pulse
  const prevIsFistRef = useRef(false);       // previous frame's fist state (for rising-edge)
  const isFistLatchRef = useRef(false);      // hysteresis latch: true while hand is in fist state
  const openFramesSinceLastFistRef = useRef(255); // non-fist frames since last fist frame
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraDebugStep, setCameraDebugStep] =
    useState<CameraDebugStep>("Idle");
  const [cameraErrorMessage, setCameraErrorMessage] = useState("");
  const [cameraStatus, setCameraStatus] =
    useState<CameraGestureStatus>("CAMERA OFF");

  const updateStatus = useCallback((status: CameraGestureStatus) => {
    if (statusRef.current === status) return;

    statusRef.current = status;
    setCameraStatus(status);
  }, []);

  const updateDebugStep = useCallback((step: CameraDebugStep) => {
    console.info(`[AURA CAMERA] ${step}`);
    setCameraDebugStep(step);
  }, []);

  const reportCameraError = useCallback(
    (prefix: string, error: unknown) => {
      const message = `${prefix}: ${formatCameraError(error)}`;

      console.error(`[AURA CAMERA] ${prefix}`, error);
      setCameraErrorMessage(message);
      updateStatus("CAMERA ERROR");
    },
    [updateStatus]
  );

  const releaseCameraResources = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    samplesRef.current = [];
    palmSizeSamplesRef.current = [];
    lastSwipeDirectionRef.current = null;
    lastSwipeTimeRef.current = 0;
    lastInspectTimeRef.current = 0;
    lastExitInspectTimeRef.current = 0;
    lastFistAddTimeRef.current = 0;
    fistPulseCountRef.current = 0;
    firstFistTimeRef.current = 0;
    prevIsFistRef.current = false;
    openFramesSinceLastFistRef.current = 255;
    handLandmarkerRef.current?.close?.();
    handLandmarkerRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    releaseCameraResources();

    setCameraEnabled(false);
    setCameraDebugStep("Idle");
    updateStatus("CAMERA OFF");
  }, [releaseCameraResources, updateStatus]);

  const processHandResult = useCallback(
    (result: HandLandmarkerResult, now: number) => {
      const landmarks = result.landmarks[0];

      if (!landmarks?.length) {
        samplesRef.current = [];

        if (now > statusHoldUntilRef.current) {
          updateStatus("CAMERA READY");
        }

        return;
      }

      const palmX =
        ((landmarks[0]?.x ?? 0) +
          (landmarks[5]?.x ?? 0) +
          (landmarks[17]?.x ?? 0)) /
        3;

      samplesRef.current = [
        ...samplesRef.current.filter((s) => now - s.time <= SWIPE_WINDOW_MS),
        { time: now, x: palmX },
      ];

      // ── Hand shape detection ─────────────────────────────────────────────
      // Two independent classifiers:
      //   isHandOpen — thumb-index spread; gates inspect enter/exit gestures.
      //   isFist     — normalised fingertip-to-palm distances; gates add gesture.
      // isFist and prevIsFistRef are updated before the isHandOpen branch so that
      // openFramesSinceLastFistRef tracks correctly even during open-hand frames.
      const wrist = landmarks[0];
      const middleTip = landmarks[12];
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];

      if (wrist && middleTip && thumbTip && indexTip) {
        const pinchDist = Math.hypot(
          thumbTip.x - indexTip.x,
          thumbTip.y - indexTip.y
        );
        const isHandOpen = pinchDist > 0.13;

        const lm5  = landmarks[5];
        const lm9  = landmarks[9];
        const lm13 = landmarks[13];
        const lm16 = landmarks[16];
        const lm17 = landmarks[17];
        const lm20 = landmarks[20];

        // ── Fist score (computed every frame) ────────────────────────────────
        // Average normalised distance of 4 fingertips to palm center.
        // Thumb excluded: its resting position varies too much across fist styles.
        // Hysteresis: enter fist when score < FIST_ENTER_THRESHOLD;
        //             exit fist when score > FIST_EXIT_THRESHOLD.
        // This prevents rapid on/off flickering at the threshold boundary.
        let isFist = false;
        if (lm5 && lm9 && lm13 && lm17 && lm16 && lm20) {
          const pcx = (wrist.x + lm5.x + lm9.x + lm13.x + lm17.x) / 5;
          const pcy = (wrist.y + lm5.y + lm9.y + lm13.y + lm17.y) / 5;
          const palmWidth = Math.hypot(lm5.x - lm17.x, lm5.y - lm17.y);
          if (palmWidth > 0.001) {
            const d8  = Math.hypot(indexTip.x  - pcx, indexTip.y  - pcy) / palmWidth;
            const d12 = Math.hypot(middleTip.x - pcx, middleTip.y - pcy) / palmWidth;
            const d16 = Math.hypot(lm16.x      - pcx, lm16.y      - pcy) / palmWidth;
            const d20 = Math.hypot(lm20.x      - pcx, lm20.y      - pcy) / palmWidth;
            const fistScore = (d8 + d12 + d16 + d20) / 4;
            if (isFistLatchRef.current) {
              isFist = fistScore < FIST_EXIT_THRESHOLD;
            } else {
              isFist = fistScore < FIST_ENTER_THRESHOLD;
            }
            isFistLatchRef.current = isFist;
          }
        }

        // Pulse tracking — updated every frame so transitions are never missed.
        if (isFist) {
          openFramesSinceLastFistRef.current = 0;
        } else {
          openFramesSinceLastFistRef.current = Math.min(
            openFramesSinceLastFistRef.current + 1, 255
          );
        }
        const isFistPulse = isFist && !prevIsFistRef.current; // rising edge only
        prevIsFistRef.current = isFist;

        // ── Inspect gesture: open palm approaching / retreating ───────────────
        // Accumulate palm-size samples only when hand is genuinely open and not a fist.
        if (isHandOpen && !isFist) {
          const palmWidth = lm5 && lm17
            ? Math.hypot(lm5.x - lm17.x, lm5.y - lm17.y)
            : Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y);

          const sampleWindow = inspectMode
            ? EXIT_INSPECT_SAMPLE_WINDOW_MS
            : INSPECT_SAMPLE_WINDOW_MS;
          palmSizeSamplesRef.current = [
            ...palmSizeSamplesRef.current.filter(
              (s) => now - s.time <= Math.max(sampleWindow, INSPECT_SAMPLE_WINDOW_MS)
            ),
            { time: now, size: palmWidth },
          ];

          if (palmSizeSamplesRef.current.length >= 3) {
            const oldest = palmSizeSamplesRef.current[0]!;
            const delta = palmWidth - oldest.size;
            const elapsed = now - oldest.time;

            if (!inspectMode) {
              if (now - lastInspectTimeRef.current > INSPECT_COOLDOWN_MS) {
                if (delta > INSPECT_GUIDANCE_THRESHOLD && elapsed > 100 && now > statusHoldUntilRef.current) {
                  updateStatus("OPEN HAND INSPECT");
                }
                if (delta > INSPECT_GROW_THRESHOLD && elapsed > 150) {
                  lastInspectTimeRef.current = now;
                  lastExitInspectTimeRef.current = now;
                  statusHoldUntilRef.current = now + 1200;
                  palmSizeSamplesRef.current = [];
                  updateStatus("INSPECT GESTURE");
                  onGesture("ENTER_INSPECT");
                }
              }
            } else {
              if (now - lastExitInspectTimeRef.current > EXIT_INSPECT_COOLDOWN_MS) {
                const shrink = -delta;
                if (shrink > EXIT_INSPECT_GUIDANCE_THRESHOLD && elapsed > 100 && now > statusHoldUntilRef.current) {
                  updateStatus("MOVE HAND BACK");
                }
                if (shrink > EXIT_INSPECT_SHRINK_THRESHOLD && elapsed > 150) {
                  lastExitInspectTimeRef.current = now;
                  lastInspectTimeRef.current = now;
                  statusHoldUntilRef.current = now + 1200;
                  palmSizeSamplesRef.current = [];
                  updateStatus("EXIT INSPECT");
                  onGesture("EXIT_INSPECT");
                }
              }
            }
          }
        } else {
          palmSizeSamplesRef.current = [];
        }

        // ── Double-fist pulse: add to order ──────────────────────────────────
        // Two distinct open→fist rising edges within DOUBLE_FIST_WINDOW_MS.
        // MIN_OPEN_FRAMES of non-fist required between pulses to prevent a single
        // sustained fist from counting twice.
        const fistPostSwipeSuppressed =
          now - lastSwipeTimeRef.current < FIST_AFTER_SWIPE_IGNORE_MS;
        const fistPostFireLocked =
          now - lastFistAddTimeRef.current < FIST_ADD_COOLDOWN_MS;

        // Expire armed state when window closes without a second fist.
        if (
          fistPulseCountRef.current === 1 &&
          now - firstFistTimeRef.current > DOUBLE_FIST_WINDOW_MS
        ) {
          fistPulseCountRef.current = 0;
          statusHoldUntilRef.current = now + 600;
          updateStatus("ORDER CANCELLED");
        }

        // Re-display armed hint if status was overwritten inside the window.
        if (fistPulseCountRef.current === 1 && now > statusHoldUntilRef.current) {
          updateStatus("FIST AGAIN TO ADD");
        }

        if (isFistPulse && !fistPostSwipeSuppressed && !fistPostFireLocked) {
          if (fistPulseCountRef.current === 0) {
            fistPulseCountRef.current = 1;
            firstFistTimeRef.current = now;
            statusHoldUntilRef.current = now + DOUBLE_FIST_WINDOW_MS;
            palmSizeSamplesRef.current = []; // prevent inspect gesture cross-firing
            updateStatus("FIST AGAIN TO ADD");
          } else if (
            openFramesSinceLastFistRef.current >= MIN_OPEN_FRAMES &&
            now - firstFistTimeRef.current <= DOUBLE_FIST_WINDOW_MS
          ) {
            fistPulseCountRef.current = 0;
            lastFistAddTimeRef.current = now;
            statusHoldUntilRef.current = now + 1500;
            updateStatus("ADDED TO ORDER");
            onGesture("ADD_TO_ORDER");
          }
        }
      }

      const lastDir = lastSwipeDirectionRef.current;
      const sinceLastSwipe = now - lastSwipeTimeRef.current;
      const inCooldown = lastDir !== null && sinceLastSwipe < SWIPE_COOLDOWN_MS;
      const inOppositeLock =
        lastDir !== null && sinceLastSwipe < OPPOSITE_LOCK_MS;

      // HUD status reflects the current gate phase.
      if (now > statusHoldUntilRef.current) {
        if (inCooldown) {
          updateStatus("COOLDOWN");
        } else if (inOppositeLock) {
          updateStatus("OPPOSITE LOCK");
        } else {
          updateStatus("READY");
        }
      }

      // Gate 1: hard cooldown — no input at all.
      if (inCooldown) return;

      // Need at least two samples spanning a minimum window.
      if (samplesRef.current.length < 2) return;

      const firstSample = samplesRef.current[0];
      const movement = palmX - firstSample.x;
      const duration = now - firstSample.time;

      if (duration < 80) return;

      const swipeDir: "left" | "right" = movement < 0 ? "left" : "right";
      const isOpposite = lastDir !== null && swipeDir !== lastDir;

      // Apply elevated thresholds for opposite-direction swipes within OPPOSITE_LOCK_MS.
      const distThreshold = isOpposite && inOppositeLock
        ? MIN_SWIPE_DISTANCE * OPPOSITE_DISTANCE_MULT
        : MIN_SWIPE_DISTANCE;
      const velThreshold = isOpposite && inOppositeLock
        ? MIN_SWIPE_VELOCITY * OPPOSITE_VELOCITY_MULT
        : MIN_SWIPE_VELOCITY;

      if (Math.abs(movement) < distThreshold) return;

      const swipeVelocity = Math.abs(movement) / Math.max(duration, 1);

      // Gate 2: velocity check (rejects slow drift and weak opposite returns).
      if (swipeVelocity < velThreshold) return;

      // Commit the swipe.
      lastSwipeDirectionRef.current = swipeDir;
      lastSwipeTimeRef.current = now;
      samplesRef.current = [];
      statusHoldUntilRef.current = now + 600;

      if (swipeDir === "left") {
        updateStatus("SWIPE LEFT");
        onGesture("PREV_PART");
      } else {
        updateStatus("SWIPE RIGHT");
        onGesture("NEXT_PART");
      }
    },
    [inspectMode, onGesture, updateStatus]
  );

  const runDetectionLoop = useCallback(() => {
    const video = videoRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const now = performance.now();

    if (video && handLandmarker && video.readyState >= 2) {
      if (now - lastDetectionAtRef.current > 72) {
        lastDetectionAtRef.current = now;

        try {
          processHandResult(handLandmarker.detectForVideo(video, now), now);
        } catch (error) {
          console.error("[AURA CAMERA] detect loop failed", error);
          setCameraErrorMessage(
            `detectForVideo failed: ${formatCameraError(error)}`
          );
          updateStatus("CAMERA ERROR");
          handLandmarkerRef.current = null;
          return;
        }
      }
    }

    frameRef.current = requestAnimationFrame(() => runDetectionLoopRef.current());
  }, [processHandResult, updateStatus]);

  useEffect(() => {
    runDetectionLoopRef.current = runDetectionLoop;
  }, [runDetectionLoop]);

  const enableCamera = useCallback(async () => {
    if (cameraEnabled) {
      stopCamera();
      return;
    }

    if (isInitializingRef.current) return;
    isInitializingRef.current = true;

    if (!navigator.mediaDevices?.getUserMedia) {
      updateStatus("CAMERA ERROR");
      setCameraErrorMessage("getUserMedia failed: API unavailable");
      console.error("[AURA CAMERA] getUserMedia failed", "API unavailable");
      isInitializingRef.current = false;
      return;
    }

    setCameraErrorMessage("");
    updateStatus("LOADING CAMERA");
    updateDebugStep("Requesting camera");

    let stream: MediaStream;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          height: { ideal: 360 },
          width: { ideal: 480 },
        },
      });
      updateDebugStep("Camera stream received");
    } catch (error) {
      reportCameraError("getUserMedia failed", error);
      isInitializingRef.current = false;
      return;
    }

    const video = videoRef.current;

    if (!video) {
      stream.getTracks().forEach((track) => track.stop());
      reportCameraError("video element failed", "Video element unavailable");
      isInitializingRef.current = false;
      return;
    }

    streamRef.current = stream;
    video.srcObject = stream;
    updateDebugStep("Video element assigned");

    try {
      await video.play();
      updateDebugStep("Video playback started");
    } catch (error) {
      releaseCameraResources();
      console.error("[AURA CAMERA] video play failed", error);
      setCameraErrorMessage(`video.play failed: ${formatCameraError(error)}`);
      updateStatus("CAMERA ERROR");
      isInitializingRef.current = false;
      return;
    }

    let visionModule: typeof import("@mediapipe/tasks-vision");

    try {
      updateDebugStep("Loading MediaPipe");
      visionModule = await import("@mediapipe/tasks-vision");
      updateDebugStep("MediaPipe loaded");
    } catch (error) {
      releaseCameraResources();
      reportCameraError("MediaPipe load failed", error);
      isInitializingRef.current = false;
      return;
    }

    const { FilesetResolver, HandLandmarker } = visionModule;

    let vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

    try {
      vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);
    } catch (error) {
      releaseCameraResources();
      reportCameraError("FilesetResolver failed", error);
      isInitializingRef.current = false;
      return;
    }

    const handLandmarkerOptions = {
      minHandDetectionConfidence: 0.58,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      numHands: 1,
      runningMode: "VIDEO" as const,
    };

    updateStatus("TRYING GPU");
    updateDebugStep("Initializing HandLandmarker (GPU)");

    try {
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          delegate: "GPU",
          modelAssetPath: HAND_LANDMARKER_MODEL_URL,
        },
        ...handLandmarkerOptions,
      });
    } catch (gpuError) {
      console.warn("[AURA CAMERA] GPU init failed", gpuError);
      console.info("[AURA CAMERA] Retrying CPU delegate");
      updateStatus("GPU FAILED");
      updateDebugStep("GPU init failed — retrying CPU");

      try {
        updateStatus("FALLBACK CPU");
        updateDebugStep("Initializing HandLandmarker (CPU)");
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            delegate: "CPU",
            modelAssetPath: HAND_LANDMARKER_MODEL_URL,
          },
          ...handLandmarkerOptions,
        });
        console.info("[AURA CAMERA] CPU fallback success");
      } catch (cpuError) {
        releaseCameraResources();
        reportCameraError("HandLandmarker init failed (GPU + CPU)", cpuError);
        isInitializingRef.current = false;
        return;
      }
    }

    setCameraEnabled(true);
    updateStatus("HAND TRACKING READY");
    updateDebugStep("HandLandmarker ready");
    runDetectionLoop();
    updateDebugStep("Detection loop started");
    isInitializingRef.current = false;
  }, [
    cameraEnabled,
    releaseCameraResources,
    reportCameraError,
    runDetectionLoop,
    stopCamera,
    updateDebugStep,
    updateStatus,
  ]);

  useEffect(() => stopCamera, [stopCamera]);

  return (
    <div className="absolute right-4 top-4 w-40 border border-stone-400/20 bg-white/45 p-2 text-stone-700 shadow-lg shadow-stone-300/20 backdrop-blur-md md:right-6 md:top-6">
      <div className="relative aspect-video overflow-hidden bg-stone-200/50">
        <video
          ref={videoRef}
          autoPlay
          className={`h-full w-full scale-x-[-1] object-cover transition duration-500 ${
            cameraEnabled ? "opacity-80" : "opacity-30"
          }`}
          muted
          playsInline
        />
        {!cameraEnabled ? (
          <div className="absolute inset-0 flex items-center justify-center text-[0.56rem] tracking-[0.2em] text-stone-500/60">
            CAMERA
          </div>
        ) : null}
      </div>

      <div className="mt-2 flex items-center justify-between gap-2">
        <p className="text-[0.56rem] tracking-[0.18em] text-stone-500/70">
          {cameraStatus}
        </p>
        <button
          onClick={enableCamera}
          className="border border-stone-400/30 bg-stone-800/8 px-2 py-1 text-[0.52rem] tracking-[0.18em] text-stone-700 transition hover:bg-stone-800/14"
        >
          {cameraEnabled ? "OFF" : "ENABLE"}
        </button>
      </div>
      <p className="mt-2 text-[0.52rem] leading-4 text-stone-400/60">
        {cameraDebugStep}
      </p>
      {cameraErrorMessage ? (
        <p className="mt-1 text-[0.5rem] leading-4 text-rose-600/80">
          {cameraErrorMessage}
        </p>
      ) : null}
    </div>
  );
}

export default function SpatialScene() {
  const [exploded, setExploded] = useState(false);
  const [activePartIndex, setActivePartIndex] = useState(0);
  const [inspectMode, setInspectMode] = useState(false);
  const [burgerExploded, setBurgerExploded] = useState(false);
  const [introVisible, setIntroVisible] = useState(true);
  const [introFading, setIntroFading] = useState(false);
  const [orderItems, setOrderItems] = useState<{ partIndex: number; qty: number }[]>([]);
  const [flyParticle, setFlyParticle] = useState<{
    x: number; y: number; opacity: number; scale: number;
  } | null>(null);
  const [trayGlow, setTrayGlow] = useState(false);
  const [reviewMode, setReviewMode] = useState(false);
  const inspectRotationRef = useRef({ x: 0, y: 0 });
  const trayRef = useRef<HTMLDivElement>(null);

  const orderSubtotal = orderItems.reduce(
    (s, { partIndex, qty }) => s + ITEM_PRICES[partIndex] * qty,
    0
  );
  const orderTax = orderSubtotal * 0.10;
  const orderTotal = orderSubtotal + orderTax;
  const totalItemCount = orderItems.reduce((s, { qty }) => s + qty, 0);

  const activePart = CAROUSEL_PARTS[activePartIndex];
  const hudOnRight = activePartIndex % 2 === 0;
  const hudPositionClass = hudOnRight
    ? "right-4 bottom-36 md:right-10 md:bottom-32"
    : "left-4 bottom-36 md:left-10 md:bottom-32";
  const burgerInspectActive = inspectMode && activePartIndex === 0;
  const gestureHint = !exploded
    ? "SPACE — BROWSE MENU"
    : inspectMode
      ? burgerInspectActive
        ? "WASD ROTATE • ESC BACK • E LAYERS"
        : "WASD ROTATE • ESC BACK • R RESET"
      : "← → BROWSE • ENTER VIEW ITEM • ESC BACK";

  useEffect(() => {
    const t1 = setTimeout(() => setIntroFading(true), 1800);
    const t2 = setTimeout(() => setExploded(true), 2200);
    const t3 = setTimeout(() => setIntroVisible(false), 3200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  const resetInspectRotation = useCallback(() => {
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
  }, []);

  const showPreviousPart = useCallback(() => {
    resetInspectRotation();
    setActivePartIndex(
      (value) => (value - 1 + CAROUSEL_PARTS.length) % CAROUSEL_PARTS.length
    );
  }, [resetInspectRotation]);

  const showNextPart = useCallback(() => {
    resetInspectRotation();
    setActivePartIndex((value) => (value + 1) % CAROUSEL_PARTS.length);
  }, [resetInspectRotation]);

  const addToOrder = useCallback(() => {
    // Start position: active item lives roughly in the center of the viewport
    const startX = window.innerWidth * 0.5;
    const startY = window.innerHeight * 0.42;

    // End position: centre of the tray panel
    let endX = window.innerWidth - 112;
    let endY = window.innerHeight * 0.5;
    if (trayRef.current) {
      const r = trayRef.current.getBoundingClientRect();
      endX = r.left + r.width * 0.5;
      endY = r.top + r.height * 0.5;
    }

    // Phase 1 — spawn particle at food item position
    setFlyParticle({ x: startX, y: startY, opacity: 1, scale: 1 });

    // Phase 2 — fly to tray (browser paints phase 1 first, then transitions)
    setTimeout(() => {
      setFlyParticle({ x: endX, y: endY, opacity: 0, scale: 0.32 });
    }, 16);

    // Phase 3 — update order state and trigger glow after fly lands
    setTimeout(() => {
      setFlyParticle(null);
      setOrderItems((prev) => {
        const idx = prev.findIndex((e) => e.partIndex === activePartIndex);
        if (idx >= 0) {
          return prev.map((e, i) => (i === idx ? { ...e, qty: e.qty + 1 } : e));
        }
        return [...prev, { partIndex: activePartIndex, qty: 1 }];
      });
      setTrayGlow(true);
      setTimeout(() => setTrayGlow(false), 900);
    }, 730);
  }, [activePartIndex]);

  const removeFromOrder = useCallback((partIndex: number) => {
    setOrderItems((prev) => {
      const idx = prev.findIndex((e) => e.partIndex === partIndex);
      if (idx < 0) return prev;
      if (prev[idx].qty <= 1) return prev.filter((_, i) => i !== idx);
      return prev.map((e, i) => (i === idx ? { ...e, qty: e.qty - 1 } : e));
    });
  }, []);

  const clearOrder = useCallback(() => {
    setOrderItems([]);
    setReviewMode(false);
  }, []);

  // Unified Gesture Action Layer — all input sources (keyboard, camera, UI buttons,
  // future MediaPipe gestures) route through this single function.
  const applyGestureAction = useCallback((action: GestureAction) => {
    if (action === "EXPLODE") {
      if (!exploded) setExploded(true);
      return;
    }

    if (action === "ASSEMBLE") {
      resetInspectRotation();
      setBurgerExploded(false);
      setInspectMode(false);
      setExploded(false);
      return;
    }

    if (action === "TOGGLE_EXPLODE") {
      setExploded((value) => {
        if (value) {
          resetInspectRotation();
          setBurgerExploded(false);
          setInspectMode(false);
        }

        return !value;
      });
      return;
    }

    if (action === "RESET") {
      resetInspectRotation();
      setBurgerExploded(false);

      if (!inspectMode) {
        setExploded(false);
      }

      return;
    }

    if (action === "EXIT_INSPECT") {
      resetInspectRotation();

      if (inspectMode) {
        setBurgerExploded(false);
        setInspectMode(false);
      } else if (exploded) {
        setExploded(false);
      }

      return;
    }

    if (action === "ENTER_INSPECT") {
      if (exploded) {
        setInspectMode(true);
      }

      return;
    }

    if (action === "ADD_TO_ORDER") {
      if (exploded) addToOrder();
      return;
    }

    if (action === "REMOVE_LAST") {
      setOrderItems((prev) => {
        if (prev.length === 0) return prev;
        const last = prev[prev.length - 1];
        if (last.qty <= 1) return prev.slice(0, -1);
        return prev.map((e, i) => i === prev.length - 1 ? { ...e, qty: e.qty - 1 } : e);
      });
      return;
    }

    if (action === "CLEAR_ORDER") {
      clearOrder();
      return;
    }

    if (action === "TOGGLE_BURGER_EXPLODE") {
      if (inspectMode && activePartIndex === 0) {
        setBurgerExploded((v) => !v);
      }

      return;
    }

    if (action === "PREV_PART") {
      if (exploded) {
        setBurgerExploded(false);
        showPreviousPart();
      }

      return;
    }

    if (action === "NEXT_PART") {
      if (exploded) {
        setBurgerExploded(false);
        showNextPart();
      }

      return;
    }

    if (!inspectMode) return;

    if (action === "ROTATE_INSPECT_LEFT") {
      inspectRotationRef.current.y += INSPECT_ROTATION_STEP;
    }

    if (action === "ROTATE_INSPECT_RIGHT") {
      inspectRotationRef.current.y -= INSPECT_ROTATION_STEP;
    }

  }, [activePartIndex, addToOrder, clearOrder, exploded, inspectMode, resetInspectRotation, showNextPart, showPreviousPart]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // In review mode ESC exits review; all other keys are suppressed.
      if (reviewMode) {
        if (event.key === "Escape") {
          event.preventDefault();
          setReviewMode(false);
        }
        return;
      }

      const actionMap: Record<string, GestureAction | undefined> = {
        ArrowLeft: inspectMode ? "ROTATE_INSPECT_LEFT" : "PREV_PART",
        ArrowRight: inspectMode ? "ROTATE_INSPECT_RIGHT" : "NEXT_PART",
        Enter: "ENTER_INSPECT",
        Escape: "EXIT_INSPECT",
        Backspace: event.shiftKey ? "CLEAR_ORDER" : "REMOVE_LAST",
        a: "ROTATE_INSPECT_LEFT",
        A: "ROTATE_INSPECT_LEFT",
        d: "ROTATE_INSPECT_RIGHT",
        D: "ROTATE_INSPECT_RIGHT",
        r: "RESET",
        R: "RESET",
        e: "TOGGLE_BURGER_EXPLODE",
        E: "TOGGLE_BURGER_EXPLODE",
        " ": "TOGGLE_EXPLODE",
      };
      const action = actionMap[event.key];

      if (!action) return;

      event.preventDefault();
      applyGestureAction(action);
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [applyGestureAction, inspectMode, reviewMode]);

  return (
    <div className="absolute inset-0">
      {introVisible && (
        <div
          className={`pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center bg-stone-50 transition-opacity duration-1000 ${
            introFading ? "opacity-0" : "opacity-100"
          }`}
        >
          <p className="mb-3 text-[0.6rem] tracking-[0.55em] text-amber-700/55">
            WELCOME TO
          </p>
          <h1 className="text-4xl font-light tracking-[0.28em] text-stone-700 md:text-5xl">
            AURA ONE
          </h1>
          <p className="mt-4 text-[0.62rem] tracking-[0.4em] text-amber-700/42">
            SPATIAL DINING
          </p>
        </div>
      )}

      <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
        <color attach="background" args={["#f6f2ea"]} />

        <ambientLight intensity={0.92} color="#fff8f0" />
        <directionalLight position={[4, 5, 3]} intensity={0.88} color="#fff8f0" />
        <pointLight position={[-4, -2, 3]} intensity={0.22} color="#ffecd0" />
        <pointLight position={[0, 3, -2]} intensity={0.14} color="#ffe8c8" />
        <InspectSceneLighting inspectMode={inspectMode} />

        <AmbientParticles />
        <AuraLogoParticles exploded={exploded} />
        <MenuBook open={exploded} />
        <SpatialMenuCarousel
          exploded={exploded}
          activePartIndex={activePartIndex}
          inspectMode={inspectMode}
          inspectRotationRef={inspectRotationRef}
          burgerExploded={burgerExploded}
        />

        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>

      {/* ── Inspect mode vignette ── */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-700 ${inspectMode ? "opacity-100" : "opacity-0"}`}
        style={{ background: "radial-gradient(ellipse at 50% 52%, transparent 34%, rgba(28,20,10,0.30) 100%)" }}
      />

      <CameraGestureLayer onGesture={applyGestureAction} inspectMode={inspectMode} />

      <div
        className={`pointer-events-none absolute ${hudPositionClass} w-[min(22rem,calc(100vw-2rem))] border border-stone-300/25 bg-white/40 p-4 text-left text-stone-800 shadow-lg shadow-stone-400/12 backdrop-blur-md transition-all duration-500 md:p-5 ${
          exploded
            ? "translate-y-0 opacity-100"
            : "translate-y-3 opacity-0"
        }`}
      >
        <p className="mb-2 text-[0.58rem] tracking-[0.36em] text-amber-700/62">
          {inspectMode ? "INSPECT MODE" : "FEATURED COMBO"}
        </p>

        {/* Special tag — carousel and inspect mode */}
        {FOOD_INSPECT_DATA[activePartIndex].special && (
          <div className="mb-2 inline-flex items-center gap-1.5 border border-amber-400/48 bg-amber-50/62 px-2 py-0.5">
            <span className="h-1 w-1 rounded-full bg-amber-500/60" />
            <span className="text-[0.50rem] tracking-[0.32em] text-amber-800/78">
              {FOOD_INSPECT_DATA[activePartIndex].special}
            </span>
          </div>
        )}

        <div className="flex items-start justify-between gap-3">
          <h2 className="text-base font-light leading-snug tracking-[0.12em] text-stone-800">
            {activePart.name}
          </h2>
          <span className="mt-0.5 shrink-0 rounded border border-amber-400/38 bg-amber-50/55 px-2 py-0.5 text-[0.66rem] font-light tracking-[0.06em] text-amber-900">
            ${ITEM_PRICES[activePartIndex].toFixed(2)}
          </span>
        </div>

        {inspectMode ? (
          <>
            {/* Rich inspect-mode data */}
            <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 border-t border-stone-200/38 pt-3">
              <div>
                <p className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">CALORIES</p>
                <p className="text-[0.68rem] font-light text-stone-700">{FOOD_INSPECT_DATA[activePartIndex].calories}</p>
              </div>
              <div>
                <p className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">PROTEIN</p>
                <p className="text-[0.68rem] font-light text-stone-700">{FOOD_INSPECT_DATA[activePartIndex].protein}</p>
              </div>
            </div>
            <div className="mt-2">
              <p className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">ALLERGENS</p>
              <p className="text-[0.62rem] font-light text-stone-600">{FOOD_INSPECT_DATA[activePartIndex].allergens}</p>
            </div>
            <div className="mt-2 border-t border-stone-200/30 pt-2">
              <p className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">INGREDIENTS</p>
              <p className="mt-0.5 text-[0.60rem] leading-4 text-stone-600/80">{FOOD_INSPECT_DATA[activePartIndex].ingredients}</p>
            </div>
            <div className="mt-2 border-t border-stone-200/30 pt-2">
              <p className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">FLAVOR</p>
              <p className="mt-0.5 text-[0.60rem] leading-4 text-stone-500/75">{FOOD_INSPECT_DATA[activePartIndex].flavorProfile}</p>
            </div>
            <p className="mt-2 border-t border-stone-200/30 pt-2 text-[0.58rem] italic leading-4 text-amber-800/50">
              {FOOD_INSPECT_DATA[activePartIndex].chefNote}
            </p>
            <p className="mt-3 text-[0.52rem] tracking-[0.22em] text-amber-700/40">
              WASD ROTATE • ESC BACK
            </p>
          </>
        ) : (
          <p className="mt-2.5 text-[0.78rem] leading-5 text-stone-500/72">
            {activePart.description}
          </p>
        )}
      </div>

      {/* Ingredient HUD — appears only when burger layers are exploded in inspect mode */}
      <div
        className={`pointer-events-none absolute left-4 top-1/2 w-[min(18rem,calc(100vw-2rem))] -translate-y-1/2 border border-amber-400/28 bg-white/42 p-4 text-stone-800 shadow-lg shadow-stone-400/12 backdrop-blur-md transition-all duration-500 md:left-6 md:p-5 ${
          burgerExploded && inspectMode && activePartIndex === 0
            ? "opacity-100"
            : "pointer-events-none opacity-0"
        }`}
      >
        <p className="mb-3 text-[0.6rem] tracking-[0.38em] text-amber-700/65">
          INGREDIENTS
        </p>
        <ul className="space-y-2.5">
          {BURGER_INGREDIENTS.map((ingredient) => (
            <li key={ingredient.name} className="border-b border-stone-300/20 pb-2 last:border-0 last:pb-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[0.75rem] font-light tracking-[0.1em] text-stone-800">
                  {ingredient.name}
                </span>
                <span className="shrink-0 text-[0.56rem] tracking-[0.12em] text-amber-700/55">
                  {ingredient.cal}
                </span>
              </div>
              <p className="mt-0.5 text-[0.6rem] leading-4 text-stone-500/75">
                {ingredient.flavor}
              </p>
              <p className="mt-0.5 text-[0.55rem] tracking-[0.08em] text-amber-700/45">
                {ingredient.allergen !== "None" ? `Allergens: ${ingredient.allergen}` : ""}
              </p>
            </li>
          ))}
        </ul>
      </div>

      <div className="pointer-events-none absolute bottom-4 right-4 max-w-[20rem] border border-stone-300/20 bg-white/30 px-3 py-2.5 text-right backdrop-blur-md md:right-6">
        <p className="text-[0.58rem] tracking-[0.18em] text-stone-500/65">{gestureHint}</p>
        <p className="mt-1.5 text-[0.54rem] tracking-[0.14em] text-amber-700/48">
          GESTURE: SWIPE ← → CHANGE • OPEN PALM INSPECT • FIST TWICE TO ADD
        </p>
      </div>

      <div
        className={`absolute bottom-8 left-1/2 flex -translate-x-1/2 flex-wrap items-center justify-center gap-2 transition duration-500 ${
          exploded ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        <button
          onClick={() => applyGestureAction(inspectMode ? "EXIT_INSPECT" : "ENTER_INSPECT")}
          className="rounded-full border border-stone-400/30 bg-white/40 px-4 py-2 text-[0.65rem] tracking-[0.24em] text-stone-700 backdrop-blur-md transition hover:bg-white/60"
        >
          {inspectMode ? "BACK" : "VIEW ITEM"}
        </button>
        {burgerInspectActive && (
          <button
            onClick={() => applyGestureAction("TOGGLE_BURGER_EXPLODE")}
            className="rounded-full border border-amber-500/35 bg-amber-100/45 px-4 py-2 text-[0.65rem] tracking-[0.24em] text-amber-900 backdrop-blur-md transition hover:bg-amber-100/65"
          >
            {burgerExploded ? "ASSEMBLE" : "EXPLODE LAYERS"}
          </button>
        )}
        <button
          onClick={addToOrder}
          className="rounded-full border border-amber-600/40 bg-amber-900/88 px-5 py-2 text-[0.65rem] tracking-[0.24em] text-amber-50 backdrop-blur-md transition hover:bg-amber-900/95"
        >
          ADD TO ORDER
        </button>
      </div>

      <button
        onClick={() => applyGestureAction("TOGGLE_EXPLODE")}
        className={`absolute left-1/2 -translate-x-1/2 rounded-full border border-amber-500/38 bg-amber-100/42 px-6 py-3 text-sm tracking-[0.25em] text-amber-900 backdrop-blur-md transition hover:bg-amber-100/62 ${
          exploded ? "bottom-20" : "bottom-8"
        }`}
      >
        {exploded ? "CLOSE MENU" : "BROWSE MENU"}
      </button>

      {/* ── Spatial Order Tray ── */}
      <div
        ref={trayRef}
        className={`pointer-events-auto absolute right-4 top-1/2 z-10 w-52 -translate-y-1/2 border p-4 backdrop-blur-md transition-all duration-700 md:right-6 md:w-56 ${
          orderItems.length > 0
            ? "translate-x-0 opacity-100"
            : "pointer-events-none translate-x-4 opacity-0"
        } ${
          trayGlow
            ? "border-amber-400/48 bg-white/55 shadow-lg shadow-amber-300/32"
            : "border-stone-300/28 bg-white/38 shadow-md shadow-stone-200/18"
        }`}
      >
        <div className="mb-3 flex items-center justify-between">
          <p className="text-[0.56rem] tracking-[0.42em] text-amber-700/60">
            ORDER TRAY
          </p>
          <button
            onClick={clearOrder}
            className="text-[0.48rem] tracking-[0.22em] text-stone-400/55 transition hover:text-rose-500/70"
          >
            CLEAR ALL
          </button>
        </div>

        <ul className="max-h-[38vh] space-y-2 overflow-y-auto">
          {orderItems.map(({ partIndex, qty }) => (
            <li
              key={partIndex}
              className="flex items-center gap-2 border-b border-stone-200/45 pb-2 last:border-0 last:pb-0"
            >
              <button
                onClick={() => removeFromOrder(partIndex)}
                className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-stone-300/45 bg-stone-100/55 text-[0.60rem] leading-none text-stone-400/70 transition hover:border-rose-300/55 hover:bg-rose-50/55 hover:text-rose-500/80"
                aria-label={`Remove one ${CAROUSEL_PARTS[partIndex].name}`}
              >
                −
              </button>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[0.70rem] font-light leading-snug tracking-[0.06em] text-stone-800">
                  {CAROUSEL_PARTS[partIndex].name}
                </p>
                <p className="mt-0.5 text-[0.56rem] tracking-[0.12em] text-stone-400/80">
                  ×{qty}
                </p>
              </div>
              <p className="shrink-0 text-[0.66rem] tracking-[0.06em] text-stone-700">
                ${(ITEM_PRICES[partIndex] * qty).toFixed(2)}
              </p>
            </li>
          ))}
        </ul>

        <div className="my-3 h-px bg-stone-300/32" />

        <div className="flex items-baseline justify-between">
          <p className="text-[0.56rem] tracking-[0.30em] text-stone-500/60">TOTAL</p>
          <p className="text-[0.88rem] font-light tracking-[0.06em] text-stone-800">
            ${orderTotal.toFixed(2)}
          </p>
        </div>

        <button
          onClick={() => setReviewMode(true)}
          className="mt-4 w-full border border-amber-500/35 bg-amber-50/40 py-2 text-[0.60rem] tracking-[0.30em] text-amber-900 transition hover:bg-amber-100/58"
        >
          REVIEW ORDER
        </button>
      </div>

      {/* ── Review dim backdrop — clicking it dismisses review ── */}
      <div
        onClick={() => setReviewMode(false)}
        className={`absolute inset-0 z-[19] transition-all duration-700 ${
          reviewMode
            ? "cursor-pointer bg-stone-100/62 opacity-100 backdrop-blur-[2px]"
            : "pointer-events-none opacity-0"
        }`}
      />

      {/* ── Spatial Review Panel ── */}
      <div
        className={`absolute right-0 top-0 z-[20] flex h-full w-[min(92vw,28rem)] flex-col border-l border-stone-200/38 bg-stone-50/94 shadow-2xl shadow-stone-400/22 backdrop-blur-xl transition-all duration-700 ${
          reviewMode
            ? "translate-x-0 opacity-100"
            : "pointer-events-none translate-x-full opacity-0"
        }`}
        style={{ transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)" }}
      >
        <div className="flex flex-1 flex-col overflow-hidden p-6 md:p-8">

          {/* Header */}
          <div className="mb-7 flex items-start justify-between">
            <div>
              <p className="text-[0.54rem] tracking-[0.52em] text-amber-700/58">YOUR ORDER</p>
              <p className="mt-1.5 text-[0.68rem] font-light tracking-[0.12em] text-stone-500/65">
                {totalItemCount} item{totalItemCount !== 1 ? "s" : ""}
              </p>
            </div>
            <button
              onClick={() => setReviewMode(false)}
              aria-label="Close review"
              className="mt-0.5 text-[0.82rem] text-stone-400/50 transition hover:text-stone-600/80"
            >
              ✕
            </button>
          </div>

          {/* Items list */}
          <ul className="flex-1 space-y-5 overflow-y-auto pr-1">
            {orderItems.map(({ partIndex, qty }) => (
              <li
                key={partIndex}
                className="border-b border-stone-200/42 pb-5 last:border-0 last:pb-0"
              >
                <div className="flex items-start justify-between gap-4">
                  <p className="text-[0.84rem] font-light leading-snug tracking-[0.06em] text-stone-800">
                    {CAROUSEL_PARTS[partIndex].name}
                  </p>
                  <p className="shrink-0 text-[0.80rem] font-light tracking-[0.04em] text-stone-800">
                    ${(ITEM_PRICES[partIndex] * qty).toFixed(2)}
                  </p>
                </div>
                <p className="mt-1.5 text-[0.60rem] tracking-[0.12em] text-stone-400/68">
                  ×{qty} · ${ITEM_PRICES[partIndex].toFixed(2)} each
                </p>
              </li>
            ))}
          </ul>

          {/* Totals */}
          <div className="mt-6 border-t border-stone-200/40 pt-5">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <p className="text-[0.60rem] tracking-[0.24em] text-stone-500/60">SUBTOTAL</p>
                <p className="text-[0.78rem] font-light text-stone-600">
                  ${orderSubtotal.toFixed(2)}
                </p>
              </div>
              <div className="flex items-baseline justify-between">
                <p className="text-[0.60rem] tracking-[0.24em] text-stone-500/60">EST. TAX 10%</p>
                <p className="text-[0.78rem] font-light text-stone-600">
                  ${orderTax.toFixed(2)}
                </p>
              </div>
            </div>

            <div className="mt-4 flex items-baseline justify-between border-t border-stone-300/28 pt-4">
              <p className="text-[0.62rem] tracking-[0.30em] text-stone-600/65">TOTAL</p>
              <p className="text-xl font-light tracking-[0.06em] text-stone-800">
                ${orderTotal.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Return CTA */}
          <button
            onClick={() => setReviewMode(false)}
            className="mt-6 w-full border border-stone-400/25 bg-white/45 py-3 text-[0.65rem] tracking-[0.28em] text-stone-700 transition hover:bg-white/65"
          >
            CONTINUE BROWSING
          </button>

        </div>
      </div>

      {/* ── Fly-to-tray particle ── */}
      {flyParticle && (
        <div
          className="pointer-events-none z-30"
          style={{
            position: "absolute",
            left: flyParticle.x,
            top: flyParticle.y,
            width: 11,
            height: 11,
            borderRadius: "50%",
            background: "rgba(180, 110, 30, 0.78)",
            boxShadow: "0 0 16px rgba(200, 130, 40, 0.60)",
            transform: `translate(-50%, -50%) scale(${flyParticle.scale})`,
            opacity: flyParticle.opacity,
            transition:
              "left 680ms cubic-bezier(0.16, 1, 0.3, 1), top 680ms cubic-bezier(0.16, 1, 0.3, 1), opacity 620ms ease-out, transform 680ms ease-out",
          }}
        />
      )}
    </div>
  );
}
