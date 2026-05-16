"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
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

// Burger layer reveal — stable open-palm hold while in inspect mode.
// Stable = palm-width delta < PALM_HOLD_STABLE_THRESHOLD over the sample window.
// This zone is safely below the exit-inspect shrink threshold so there is no conflict.
const PALM_HOLD_DURATION_MS = 900;
const PALM_HOLD_STABLE_THRESHOLD = 0.012;
const BURGER_EXPLODE_COOLDOWN_MS = 2000;

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

type LandingPhase = "intro" | "menu";

type GestureAction =
  | "EXPLODE"
  | "ASSEMBLE"
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
  | "REVEALING LAYERS"
  | "ASSEMBLING BURGER"
  | "CAMERA ERROR";



function smoothStep(value: number) {
  return value * value * (3 - 2 * value);
}

// Smooth 0→1 ramp inside a [a, b] band. Returns 0 below `a`, 1 above `b`,
// and a smoothStep curve between. Used to slice a single progress value
// (0..1) into disjoint timeline phases for staged transitions.
function smoothBand(value: number, a: number, b: number) {
  if (b <= a) return value >= b ? 1 : 0;
  const t = (value - a) / (b - a);
  return smoothStep(Math.max(0, Math.min(1, t)));
}

function seededUnit(index: number) {
  const value = Math.sin(index * 12.9898) * 43758.5453;

  return value - Math.floor(value);
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
  inspectScaleMultiplier = 1.38,
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
  inspectScaleMultiplier?: number;
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
  const prevIsInspectActiveRef = useRef(false);
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
    const inspectScale = slot === 0 ? focusScale * inspectScaleMultiplier : 0.22;
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

    // Detect inspect entry/exit for this slot and keep rotation continuous.
    const enteredInspect = isInspectActive && !prevIsInspectActiveRef.current;
    const exitedInspect  = !isInspectActive && prevIsInspectActiveRef.current && slot === 0;
    if (enteredInspect) {
      // Seed inspect idle from the live self-rotation so there is no jump on entry.
      inspectIdleYRef.current = selfRotationRef.current.y * Math.max(separatedProgress, 0.01);
    }
    if (exitedInspect) {
      // Re-seed self-rotation from inspect idle so exit is equally seamless.
      selfRotationRef.current.y = separatedProgress > 0.01
        ? inspectIdleYRef.current / separatedProgress
        : inspectIdleYRef.current;
    }
    prevIsInspectActiveRef.current = isInspectActive;

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
      // Slow deliberate yaw — cinematic idle rotation.
      inspectIdleYRef.current += delta * 0.15;
    }
    // Cinematic easing: all inspect transitions use slower damping so movement
    // feels heavy and intentional rather than snappy.
    partScaleRef.current = THREE.MathUtils.lerp(
      partScaleRef.current,
      scaleTarget,
      1 - Math.exp(-delta * 4.2)
    );
    highlightRef.current = THREE.MathUtils.lerp(
      highlightRef.current,
      highlightTarget,
      1 - Math.exp(-delta * 3.8)
    );
    if (activeLightRef.current) {
      // Warm premium overhead spotlight — significantly stronger in inspect.
      activeLightRef.current.intensity =
        highlightRef.current * (isInspectActive ? 2.6 : 0.28);
    }
    inspectBlendRef.current = THREE.MathUtils.lerp(
      inspectBlendRef.current,
      inspectPresence,
      1 - Math.exp(-delta * 2.4)
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
    // Inactive items pushed wider and deeper for strong spatial separation.
    const inspectX = slot === 0 ? 0 : rearDirection * (5.0 + slotDistance * 1.0);
    // Active item gets a very slow gentle float — alive but restrained.
    const inspectY = slot === 0
      ? Math.sin(t * 0.28) * 0.036 * inspectBlendRef.current
      : -0.18 - slotDistance * 0.14;
    const inspectZ = slot === 0 ? inspectZFocus : -5.8 - slotDistance * 2.2;
    inspectPositionRef.current.set(inspectX, inspectY, inspectZ);

    const dimTarget = inspectMode && slot !== 0 ? 1 : 0;
    // Slower dim — background recedes gracefully, not abruptly.
    dimRef.current = THREE.MathUtils.lerp(
      dimRef.current,
      dimTarget,
      1 - Math.exp(-delta * 1.9)
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

    // Heavier, controlled glide — gives the product weight during inspect entry/exit.
    renderPositionRef.current.lerp(
      targetPositionRef.current,
      1 - Math.exp(-delta * 2.9)
    );
    groupRef.current.position.copy(renderPositionRef.current);
    manualRotationTargetRef.current.set(
      0,
      inspectMode && activePresence ? inspectRotationRef.current.y : 0
    );
    manualRotationRef.current.lerp(
      manualRotationTargetRef.current,
      1 - Math.exp(-delta * 4.5)
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
        color="#fff8ec"
        distance={5.2}
        intensity={0}
        position={[0, 1.9, 0.9]}
      />
    </group>
  );
}

// Single declarative config for the 9 ingredient layers. Indexed top → bottom
// (idx 0 = top-bun, idx 8 = bottom-bun) so the index doubles as the
// stack-position-from-top stagger key (top lifts first, bottom settles first).
type BurgerLayerConfig = {
  path: string;
  name: string;
  assembledY: number;
  revealedY: number;
  // Reveal-only X/Z drift (gated by spreadP).
  revealedOffset: readonly [number, number];
  // Always-applied corrective rotation (compensates for GLB orientation).
  baseRotation: readonly [number, number, number];
  // Reveal-only extra rotation tilt (gated by spreadP).
  revealedRotation: readonly [number, number, number];
  // Idle Y-axis spin speed (rad/sec), active only in revealed state.
  idleYRotSpeed: number;
  scale: number;
  doubleSide: boolean;
};

const BURGER_LAYERS: ReadonlyArray<BurgerLayerConfig> = [
  // 0 — TOP BUN
  {
    path: "/models/burger-layers/top-bun.glb",
    name: "Top Bun",
    assembledY:  0.42,
    revealedY:   1.40,
    revealedOffset:  [ 0.00, -0.03],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.16,         0, 0],
    idleYRotSpeed: -0.018,
    scale: 0.82,
    doubleSide: false,
  },
  // 1 — SAUCE
  {
    path: "/models/burger-layers/sauce1.glb",
    name: "Sauce",
    assembledY:  0.30,
    revealedY:   1.04,
    revealedOffset:  [-0.02,  0.03],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.06,  0.20,    0],
    idleYRotSpeed:  0.020,
    scale: 0.88,
    doubleSide: true,
  },
  // 2 — LETTUCE  (ships standing vertical — +π/2 X lays the leaf flat)
  {
    path: "/models/burger-layers/lettuce%20(1).glb",
    name: "Lettuce",
    assembledY:  0.20,
    revealedY:   0.68,
    revealedOffset:  [-0.03,  0.00],
    baseRotation:    [ Math.PI / 2,  0, 0],
    revealedRotation:[ 0.01, -0.12,  0.01],
    idleYRotSpeed:  0.016,
    scale: 0.82,
    doubleSide: true,
  },
  // 3 — TOMATO
  {
    path: "/models/burger-layers/tomato1.glb",
    name: "Tomato",
    assembledY:  0.10,
    revealedY:   0.34,
    revealedOffset:  [ 0.03,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.10,  0.25,    0],
    idleYRotSpeed: -0.018,
    scale: 0.88,
    doubleSide: true,
  },
  // 4 — ONION RING (anchored at center)
  {
    path: "/models/burger-layers/onion%20ring.glb",
    name: "Onion Ring",
    assembledY:  0.00,
    revealedY:   0.00,
    revealedOffset:  [ 0.00,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.04,  0.15,    0],
    idleYRotSpeed:  0.022,
    scale: 0.85,
    doubleSide: true,
  },
  // 5 — CHEESE
  {
    path: "/models/burger-layers/Cheese%20(1).glb",
    name: "Cheese",
    assembledY: -0.10,
    revealedY:  -0.34,
    revealedOffset:  [ 0.00,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.10,  0.18,    0],
    idleYRotSpeed: -0.014,
    scale: 0.90,
    doubleSide: true,
  },
  // 6 — BACON
  {
    path: "/models/burger-layers/bacon%20(1).glb",
    name: "Bacon",
    assembledY: -0.20,
    revealedY:  -0.68,
    revealedOffset:  [ 0.00,  0.05],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.14,  0.40, -0.05],
    idleYRotSpeed:  0.020,
    scale: 0.80,
    doubleSide: true,
  },
  // 7 — PATTY  (near-flat tilt; preserve grilled top via subtle X tilt + camera angle)
  {
    path: "/models/burger-layers/Patty%20(1).glb",
    name: "Patty",
    assembledY: -0.30,
    revealedY:  -1.04,
    revealedOffset:  [ 0.00,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.02,  0.08,    0],
    idleYRotSpeed: -0.026,
    scale: 0.95,
    doubleSide: false,
  },
  // 8 — BOTTOM BUN
  {
    path: "/models/burger-layers/bottom.glb",
    name: "Bottom Bun",
    assembledY: -0.42,
    revealedY:  -1.40,
    revealedOffset:  [ 0.00, -0.03],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.08,         0, 0],
    idleYRotSpeed:  0.018,
    scale: 0.82,
    doubleSide: false,
  },
];

BURGER_LAYERS.forEach((layer) => useGLTF.preload(layer.path));

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

// Uniform scale applied to the whole burger group so it fits the viewport.
const EXPLODED_STACK_SCALE = 0.72;

// Loads a single burger layer GLB, normalizes it, and applies safe material defaults.
// Transparency is OFF by default — the parent BurgerExplodedView toggles it only while
// a layer is fading. This prevents see-through artifacts on fully-revealed ingredients.
function BurgerLayerGLB({ path, doubleSide }: { path: string; doubleSide: boolean }) {
  const { scene } = useGLTF(path);

  const { center, normalizedScale } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const c = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    return { center: c, normalizedScale: maxDim > 0 ? 0.85 / maxDim : 1 };
  }, [scene]);

  useEffect(() => {
    scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mats = Array.isArray(obj.material)
        ? (obj.material as THREE.MeshStandardMaterial[])
        : [obj.material as THREE.MeshStandardMaterial];
      mats.forEach((m) => {
        if (!m.isMeshStandardMaterial) return;
        // Default to fully opaque, solid rendering. Parent toggles transparency
        // only during fade transitions to avoid z-sort / see-through artifacts.
        m.transparent = false;
        m.depthWrite = true;
        m.opacity = 1;
        m.envMapIntensity = 0.72;
        // Tripo GLBs omit metallicFactor/roughnessFactor — Three.js GLTF defaults to
        // metalness=1, roughness=1, which renders black without an env map.
        // Force metalness to 0 to restore warm food color from texture.
        m.metalness = 0;
        // Only cap the roughness ceiling; preserve per-texture relative differences.
        m.roughness = Math.min(m.roughness, 0.88);
        // DoubleSide only for thin ingredients where the back face would otherwise be culled.
        m.side = doubleSide ? THREE.DoubleSide : THREE.FrontSide;
        m.needsUpdate = true;
      });
    });
  }, [scene, doubleSide]);

  return (
    <group scale={normalizedScale}>
      <group position={[-center.x, -center.y, -center.z]}>
        <primitive object={scene} />
      </group>
    </group>
  );
}

function BurgerExplodedView({ active }: { active: boolean }) {
  // Single progress: 0 = assembled (looks like one burger), 1 = fully spread.
  const progressRef = useRef(0);
  const shadowRef = useRef<THREE.Mesh>(null);

  // Callback-ref array — one slot per BURGER_LAYERS entry. Scales to any
  // number of layers without per-layer useRef declarations.
  const groupRefs = useRef<(THREE.Group | null)[]>(
    new Array(BURGER_LAYERS.length).fill(null)
  );
  const layerRot = useRef<number[]>(new Array(BURGER_LAYERS.length).fill(0));

  useFrame(({ clock }, delta) => {
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 2.5)
    );
    const progress = progressRef.current;
    const t = clock.getElapsedTime();

    BURGER_LAYERS.forEach((layer, i) => {
      const group = groupRefs.current[i];
      if (!group) return;

      // Per-layer staggered spread band keyed by stack position from top.
      // Index already orders top → bottom, so idx 0 (top-bun) lifts first.
      const spreadP = smoothBand(progress, i * 0.025, 0.92);

      // Y position — lerp assembled → revealed by spread progress.
      group.position.y =
        THREE.MathUtils.lerp(layer.assembledY, layer.revealedY, spreadP) +
        Math.sin(t * 0.26 + i * 0.80) * 0.005 * spreadP;

      // X/Z drift — zero when assembled; only applies during spread.
      group.position.x = THREE.MathUtils.lerp(0, layer.revealedOffset[0], spreadP);
      group.position.z = THREE.MathUtils.lerp(0, layer.revealedOffset[1], spreadP);

      // Per-layer scale.
      group.scale.setScalar(layer.scale);

      // Base rotation always applied (corrects GLB orientation — e.g. lettuce
      // ships standing vertically and needs +π/2 X to lay flat).
      // Idle Y spin and reveal tilt are both gated by spreadP so assembled
      // layers stack flat with no rotation drift.
      layerRot.current[i] += delta * layer.idleYRotSpeed * spreadP;
      const [bx, byBase, bz] = layer.baseRotation;
      const [rx, ryBase, rz] = layer.revealedRotation;
      group.rotation.set(
        bx + rx * spreadP,
        byBase + ryBase * spreadP + layerRot.current[i],
        bz + rz * spreadP
      );
    });

    if (shadowRef.current) {
      // Shadow deepens slightly as layers spread.
      const shadowP = smoothStep(Math.max(0, Math.min(1, progress)));
      (shadowRef.current.material as THREE.MeshStandardMaterial).opacity =
        0.18 + shadowP * 0.10;
    }
  });

  return (
    <group>
      {/* Uniform framing scale — keeps stack inside viewport without camera change */}
      <group scale={EXPLODED_STACK_SCALE}>
        {/* Contact shadow — sits just below the bottom bun local Y */}
        <mesh ref={shadowRef} position={[0, -1.30, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[0.42, 36]} />
          <meshStandardMaterial
            color="#180804"
            transparent
            opacity={0}
            roughness={1}
            depthWrite={false}
          />
        </mesh>

        {BURGER_LAYERS.map((layer, i) => (
          <group
            key={layer.path}
            ref={(el) => {
              groupRefs.current[i] = el;
            }}
          >
            <BurgerLayerGLB path={layer.path} doubleSide={layer.doubleSide} />
          </group>
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
    // Sit the carousel below camera eye-line so items feel like they're on a stage.
    groupRef.current.position.y = 0.0 + Math.sin(t * 0.38) * 0.025 * (1 - p);
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
        inspectZFocus={1.7}
        inspectScaleMultiplier={1.54}
        selfRotationAmount={0.12}
        motionSeed={1}
      >
        <BurgerExplodedView active={burgerExploded && inspectMode && activePartIndex === 0} />
      </Part>

      {/* ── Item 1: Sushi Roll Combo ── */}
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
        inspectScaleMultiplier={1.52}
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
        inspectZFocus={1.3}
        inspectScaleMultiplier={1.44}
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
        inspectScaleMultiplier={1.55}
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
        inspectZFocus={1.4}
        inspectScaleMultiplier={1.50}
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
  const floorPoolRef = useRef<THREE.PointLight>(null);
  const blendRef = useRef(0);

  useFrame((_, delta) => {
    // Slow cinematic fade — lighting builds and withdraws gradually.
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      inspectMode ? 1 : 0,
      1 - Math.exp(-delta * 2.4)
    );

    if (keyLightRef.current) {
      keyLightRef.current.intensity = blendRef.current * 1.5;
    }
    if (rimLightRef.current) {
      rimLightRef.current.intensity = blendRef.current * 0.78;
    }
    if (floorPoolRef.current) {
      floorPoolRef.current.intensity = blendRef.current * 0.50;
    }
  });

  return (
    <>
      <directionalLight
        ref={keyLightRef}
        position={[-1.4, 5.2, 3.4]}
        color="#fff6e4"
        intensity={0}
      />
      <pointLight
        ref={rimLightRef}
        position={[3.2, 2.0, -1.4]}
        color="#ffd070"
        distance={16}
        intensity={0}
      />
      <pointLight
        ref={floorPoolRef}
        color="#ffcc88"
        position={[0, -1.4, 2.0]}
        distance={7}
        intensity={0}
      />
    </>
  );
}

// Warm cinematic hero lighting for the assembled burger in inspect mode.
// Distinct from InspectSceneLighting (which serves all items generically).
function BurgerInspectLighting({ active }: { active: boolean }) {
  const heroFillRef  = useRef<THREE.PointLight>(null);
  const backRimRef   = useRef<THREE.DirectionalLight>(null);
  const blendRef     = useRef(0);

  useFrame((_, delta) => {
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 2.2)
    );
    // Low front-warm fill catches the bun underside and sesame seeds.
    if (heroFillRef.current)  heroFillRef.current.intensity  = blendRef.current * 0.80;
    // Back rim separates the burger silhouette from the darkened background.
    if (backRimRef.current)   backRimRef.current.intensity   = blendRef.current * 0.52;
  });

  return (
    <>
      <pointLight
        ref={heroFillRef}
        position={[-1.0, 0.6, 4.0]}
        color="#ffcc80"
        distance={9}
        intensity={0}
      />
      <directionalLight
        ref={backRimRef}
        position={[1.4, 2.8, -3.2]}
        color="#ffe4c0"
        intensity={0}
      />
    </>
  );
}

// Dedicated warm lighting for the burger exploded ingredient showcase.
// Fades in independently of InspectSceneLighting to add extra depth when layers open.
function BurgerExplodedLighting({ active }: { active: boolean }) {
  const topKeyRef    = useRef<THREE.DirectionalLight>(null);
  const warmFillRef  = useRef<THREE.PointLight>(null);
  const underlightRef = useRef<THREE.PointLight>(null);
  const backRimRef   = useRef<THREE.DirectionalLight>(null);
  const ambientFillRef = useRef<THREE.HemisphereLight>(null);
  const blendRef     = useRef(0);

  useFrame((_, delta) => {
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 1.8)
    );
    if (topKeyRef.current)    topKeyRef.current.intensity    = blendRef.current * 1.2;
    if (warmFillRef.current)  warmFillRef.current.intensity  = blendRef.current * 0.80;
    if (underlightRef.current) underlightRef.current.intensity = blendRef.current * 1.10;
    if (backRimRef.current)   backRimRef.current.intensity   = blendRef.current * 0.40;
    if (ambientFillRef.current) ambientFillRef.current.intensity = blendRef.current * 0.28;
  });

  return (
    <>
      {/* Warm overhead key — rakes across the stack from upper-left */}
      <directionalLight
        ref={topKeyRef}
        position={[-0.8, 5.2, 2.8]}
        color="#ffd89a"
        intensity={0}
      />
      {/* Soft warm fill from the right */}
      <pointLight
        ref={warmFillRef}
        position={[2.2, 1.6, 2.0]}
        color="#ffb860"
        distance={11}
        intensity={0}
      />
      {/* Warm underlight — rakes up from below to light the bottom bun underside */}
      <pointLight
        ref={underlightRef}
        position={[0, -1.8, 1.8]}
        color="#ffb870"
        distance={10}
        intensity={0}
      />
      {/* Back separation rim — gives depth between layers */}
      <directionalLight
        ref={backRimRef}
        position={[1.2, 2.5, -3.0]}
        color="#ffe8c8"
        intensity={0}
      />
      {/* Warm hemisphere fill — lifts shadowed underside without washing out warm look */}
      <hemisphereLight
        ref={ambientFillRef}
        args={["#fff1d8", "#3a2010", 0]}
      />
    </>
  );
}

// Retry up to 10 animation frames waiting for the video element to mount.
// Required because getUserMedia is async and may resolve before React has
// committed the <video> element to the DOM.
async function waitForVideoElement(
  ref: { current: HTMLVideoElement | null }
): Promise<HTMLVideoElement | null> {
  for (let i = 0; i < 10; i++) {
    if (ref.current) return ref.current;
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
  return null;
}

function CameraGestureLayer({
  onGesture,
  inspectMode,
  burgerExploded,
}: {
  onGesture: (action: GestureAction) => void;
  inspectMode: boolean;
  burgerExploded: boolean;
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
  const palmHoldStartRef = useRef<number | null>(null);
  const lastBurgerExplodeTimeRef = useRef(0);
  const burgerExplodedRef = useRef(burgerExploded);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraStatus, setCameraStatus] =
    useState<CameraGestureStatus>("CAMERA OFF");

  const updateStatus = useCallback((status: CameraGestureStatus) => {
    if (statusRef.current === status) return;

    statusRef.current = status;
    setCameraStatus(status);
  }, []);

  const updateDebugStep = useCallback((step: string) => {
    console.info(`[AURA CAMERA] ${step}`);
  }, []);

  const reportCameraError = useCallback(
    (prefix: string, error: unknown) => {
      console.error(`[AURA CAMERA] ${prefix}`, error);
      updateStatus("CAMERA ERROR");
    },
    [updateStatus]
  );

  useEffect(() => { burgerExplodedRef.current = burgerExploded; }, [burgerExploded]);

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
    palmHoldStartRef.current = null;
    lastBurgerExplodeTimeRef.current = 0;
    handLandmarkerRef.current?.close?.();
    handLandmarkerRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    releaseCameraResources();

    setCameraEnabled(false);
    updateStatus("CAMERA OFF");
  }, [releaseCameraResources, updateStatus]);

  const processHandResult = useCallback(
    (result: HandLandmarkerResult, now: number) => {
      const landmarks = result.landmarks[0];

      if (!landmarks?.length) {
        samplesRef.current = [];
        palmHoldStartRef.current = null;

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
                  palmHoldStartRef.current = null;
                  updateStatus("EXIT INSPECT");
                  onGesture("EXIT_INSPECT");
                }
              }

              // ── Burger layer reveal: stable palm hold ──────────────────────────
              // |delta| < PALM_HOLD_STABLE_THRESHOLD → not approaching or retreating.
              // Zone is safely below EXIT_INSPECT_SHRINK_THRESHOLD — no conflict.
              if (now - lastBurgerExplodeTimeRef.current > BURGER_EXPLODE_COOLDOWN_MS) {
                const absDelta = Math.abs(delta);
                const isStable = absDelta < PALM_HOLD_STABLE_THRESHOLD && elapsed > 400;
                if (isStable) {
                  if (palmHoldStartRef.current === null) {
                    palmHoldStartRef.current = now;
                  } else if (now - palmHoldStartRef.current >= PALM_HOLD_DURATION_MS) {
                    palmHoldStartRef.current = null;
                    lastBurgerExplodeTimeRef.current = now;
                    statusHoldUntilRef.current = now + 1400;
                    updateStatus(burgerExplodedRef.current ? "ASSEMBLING BURGER" : "REVEALING LAYERS");
                    onGesture("TOGGLE_BURGER_EXPLODE");
                  }
                } else {
                  palmHoldStartRef.current = null;
                }
              }
            }
          }
        } else {
          palmSizeSamplesRef.current = [];
          palmHoldStartRef.current = null;
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
      console.error("[AURA CAMERA] getUserMedia failed", "API unavailable");
      isInitializingRef.current = false;
      return;
    }
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

    // Wait for the <video> element to mount (it may not exist yet if React
    // hasn't re-rendered since the user clicked the button).
    const video = await waitForVideoElement(videoRef);

    if (!video) {
      stream.getTracks().forEach((track) => track.stop());
      reportCameraError("video element failed", "Video element unavailable after retries");
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

  // Derive a customer-facing label from the raw gesture status
  const isReady =
    cameraStatus === "HAND TRACKING READY" ||
    cameraStatus === "CAMERA READY" ||
    cameraStatus === "READY" ||
    cameraStatus === "SWIPE LEFT" ||
    cameraStatus === "SWIPE RIGHT" ||
    cameraStatus === "OPEN HAND INSPECT" ||
    cameraStatus === "INSPECT GESTURE" ||
    cameraStatus === "FIST AGAIN TO ADD" ||
    cameraStatus === "ADDED TO ORDER" ||
    cameraStatus === "ORDER CANCELLED" ||
    cameraStatus === "MOVE HAND BACK" ||
    cameraStatus === "EXIT INSPECT" ||
    cameraStatus === "REVEALING LAYERS" ||
    cameraStatus === "ASSEMBLING BURGER" ||
    cameraStatus === "COOLDOWN" ||
    cameraStatus === "OPPOSITE LOCK";
  const isError = cameraStatus === "CAMERA ERROR";
  const isLoading = !isReady && !isError && cameraEnabled;

  return (
    <div className="absolute right-4 top-4 md:right-6 md:top-6">
      {/* Card + video always mounted so videoRef.current is never null when
          enableCamera's async getUserMedia resolves. Hidden via sr-only when off. */}
      <div className={cameraEnabled
        ? "w-36 overflow-hidden border border-stone-200/25 bg-white/38 shadow-md shadow-stone-200/14 backdrop-blur-md"
        : "sr-only"
      }>
        <div className="relative aspect-video overflow-hidden bg-stone-100/60">
          <video
            ref={videoRef}
            autoPlay
            className="h-full w-full scale-x-[-1] object-cover"
            muted
            playsInline
          />
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-stone-50/40">
              <span className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">LOADING</span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between px-2.5 py-1.5">
          <div className="flex items-center gap-1.5">
            <span className={`h-1.5 w-1.5 rounded-full transition-colors duration-700 ${
              isReady ? "bg-emerald-400/75" : isError ? "bg-rose-400/75" : "bg-amber-400/65"
            }`} />
            <p className="text-[0.46rem] tracking-[0.16em] text-stone-500/60">
              {isReady ? "GESTURE READY" : isError ? "UNAVAILABLE" : "CONNECTING"}
            </p>
          </div>
          <button
            onClick={stopCamera}
            className="text-[0.44rem] tracking-[0.12em] text-stone-400/45 transition hover:text-stone-600/65"
          >
            OFF
          </button>
        </div>
      </div>

      {/* Enable button shown only when camera is off */}
      {!cameraEnabled && (
        <button
          onClick={enableCamera}
          className="flex items-center gap-2 border border-stone-300/32 bg-white/48 px-3.5 py-2 text-[0.50rem] tracking-[0.26em] text-stone-500/70 shadow-sm shadow-stone-200/18 backdrop-blur-md transition hover:bg-white/68 hover:text-stone-700/85"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-stone-400/45" />
          ENABLE CAMERA
        </button>
      )}
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
  const [landingPhase, setLandingPhase] = useState<LandingPhase>("intro");
  const landingPhaseRef = useRef<LandingPhase>("intro");
  const inspectRotationRef = useRef({ x: 0, y: 0 });
  const trayRef = useRef<HTMLDivElement>(null);

  // Auto-demo mode — all stable refs, no extra re-renders
  const lastInteractionRef = useRef(0);
  const demoActiveRef = useRef(false);
  const demoPhaseRef = useRef<"wait_inspect" | "inspecting" | "wait_next" | null>(null);
  const demoPhaseDueRef = useRef(0);
  // mirrors so the interval closure always sees current React state
  const explodedRef = useRef(false);
  const inspectModeRef = useRef(false);

  const orderSubtotal = orderItems.reduce(
    (s, { partIndex, qty }) => s + ITEM_PRICES[partIndex] * qty,
    0
  );
  const orderTax = orderSubtotal * 0.10;
  const orderTotal = orderSubtotal + orderTax;
  const totalItemCount = orderItems.reduce((s, { qty }) => s + qty, 0);

  const activePart = CAROUSEL_PARTS[activePartIndex];

  useEffect(() => {
    lastInteractionRef.current = Date.now();
    // Intro title fades at 1.8s
    const t1 = setTimeout(() => setIntroFading(true), 1800);
    // Title unmounts at 2.8s; menu auto-opens at 3s
    const t2 = setTimeout(() => setIntroVisible(false), 2800);
    const t3 = setTimeout(() => {
      setLandingPhase("menu");
      setExploded(true);
    }, 3000);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, []);

  const resetInspectRotation = useCallback(() => {
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
  }, []);

  // Keep mirrors in sync
  useEffect(() => { explodedRef.current = exploded; }, [exploded]);
  useEffect(() => { inspectModeRef.current = inspectMode; }, [inspectMode]);
  useEffect(() => { landingPhaseRef.current = landingPhase; }, [landingPhase]);

  // Auto-demo loop — ticks every 120ms, driven entirely by refs
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const isCarousel = explodedRef.current && !inspectModeRef.current;

      if (!demoActiveRef.current) {
        if (isCarousel && now - lastInteractionRef.current > 7000) {
          demoActiveRef.current = true;
          demoPhaseRef.current = "wait_inspect";
          setActivePartIndex((v) => (v + 1) % CAROUSEL_PARTS.length);
          demoPhaseDueRef.current = now + 2200;
        }
        return;
      }

      if (demoPhaseRef.current === "wait_inspect" && now >= demoPhaseDueRef.current && isCarousel) {
        demoPhaseRef.current = "inspecting";
        setInspectMode(true);
        demoPhaseDueRef.current = now + 3800;
      } else if (demoPhaseRef.current === "inspecting" && now >= demoPhaseDueRef.current && inspectModeRef.current) {
        demoPhaseRef.current = "wait_next";
        setInspectMode(false);
        demoPhaseDueRef.current = now + 2000;
      } else if (demoPhaseRef.current === "wait_next" && now >= demoPhaseDueRef.current && isCarousel) {
        demoPhaseRef.current = "wait_inspect";
        setActivePartIndex((v) => (v + 1) % CAROUSEL_PARTS.length);
        demoPhaseDueRef.current = now + 2200;
      }
    }, 120);

    return () => clearInterval(id);
  }, []); // stable — only refs and stable setters

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
  // All state reads use refs so the closure is never stale.
  const applyGestureAction = useCallback((action: GestureAction) => {
    lastInteractionRef.current = Date.now();
    if (demoActiveRef.current) {
      demoActiveRef.current = false;
      demoPhaseRef.current = null;
    }

    // Landing gate — any gesture during intro immediately opens the menu.
    // Refs are updated synchronously so the action also processes in this call.
    if (landingPhaseRef.current !== "menu") {
      landingPhaseRef.current = "menu";
      explodedRef.current = true;
      setLandingPhase("menu");
      setExploded(true);
      // fall through — process the action with freshly updated refs
    }

    if (action === "EXPLODE") {
      if (!explodedRef.current) setExploded(true);
      return;
    }

    if (action === "ASSEMBLE") {
      resetInspectRotation();
      setBurgerExploded(false);
      setInspectMode(false);
      setExploded(false);
      return;
    }

    if (action === "RESET") {
      resetInspectRotation();
      setBurgerExploded(false);
      if (!inspectModeRef.current) setExploded(false);
      return;
    }

    if (action === "EXIT_INSPECT") {
      resetInspectRotation();
      if (inspectModeRef.current) {
        setBurgerExploded(false);
        setInspectMode(false);
      } else if (explodedRef.current) {
        setExploded(false);
      }
      return;
    }

    if (action === "ENTER_INSPECT") {
      if (explodedRef.current) setInspectMode(true);
      return;
    }

    if (action === "ADD_TO_ORDER") {
      if (explodedRef.current) addToOrder();
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
      if (inspectModeRef.current && activePartIndex === 0) setBurgerExploded((v) => !v);
      return;
    }

    if (action === "PREV_PART") {
      if (explodedRef.current) { setBurgerExploded(false); showPreviousPart(); }
      return;
    }

    if (action === "NEXT_PART") {
      if (explodedRef.current) { setBurgerExploded(false); showNextPart(); }
      return;
    }

    if (!inspectModeRef.current) return;

    if (action === "ROTATE_INSPECT_LEFT")  inspectRotationRef.current.y += INSPECT_ROTATION_STEP;
    if (action === "ROTATE_INSPECT_RIGHT") inspectRotationRef.current.y -= INSPECT_ROTATION_STEP;

  }, [activePartIndex, addToOrder, clearOrder, resetInspectRotation, showNextPart, showPreviousPart]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Any key press interrupts demo and resets idle timer
      lastInteractionRef.current = Date.now();
      demoActiveRef.current = false;
      demoPhaseRef.current = null;

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
          <div className="mb-7 h-px w-10 bg-amber-500/28" />
          <p className="mb-3 text-[0.48rem] tracking-[0.62em] text-amber-700/44">
            EXPERIENCE
          </p>
          <h1 className="text-5xl font-extralight tracking-[0.30em] text-stone-700 md:text-6xl">
            AURA ONE
          </h1>
          <p className="mt-4 text-[0.60rem] font-light tracking-[0.22em] text-stone-500/52">
            Spatial Dining Experience
          </p>
          <div className="mt-7 h-px w-10 bg-amber-500/28" />
        </div>
      )}

      <Canvas camera={{ position: [0, 2.4, 7.2], fov: 40 }}>
        <color attach="background" args={["#f6f2ea"]} />

        <ambientLight intensity={0.92} color="#fff8f0" />
        <directionalLight position={[4, 5, 3]} intensity={0.88} color="#fff8f0" />
        <pointLight position={[-4, -2, 3]} intensity={0.22} color="#ffecd0" />
        <pointLight position={[0, 3, -2]} intensity={0.14} color="#ffe8c8" />
        <InspectSceneLighting inspectMode={inspectMode} />
        <BurgerInspectLighting active={inspectMode && activePartIndex === 0 && !burgerExploded} />
        <BurgerExplodedLighting active={burgerExploded && inspectMode && activePartIndex === 0} />

        <AmbientParticles />
        <AuraLogoParticles exploded={exploded} />
        {/* MenuBook collapses when menu opens, stays prominent during intro */}
        <MenuBook open={landingPhase === "menu"} />
        <SpatialMenuCarousel
          exploded={exploded}
          activePartIndex={activePartIndex}
          inspectMode={inspectMode}
          inspectRotationRef={inspectRotationRef}
          burgerExploded={burgerExploded}
        />
      </Canvas>

      {/* ── Inspect mode vignette — cinematic depth haze ── */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-[1400ms] ${inspectMode ? "opacity-100" : "opacity-0"}`}
        style={{ background: "radial-gradient(ellipse 110% 90% at 50% 52%, transparent 28%, rgba(18,11,4,0.44) 100%)" }}
      />
      {/* Subtle bottom depth gradient — spatial stage floor */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-[1400ms] ${inspectMode ? "opacity-100" : "opacity-0"}`}
        style={{ background: "linear-gradient(to top, rgba(14,9,3,0.26) 0%, transparent 42%)" }}
      />
      <CameraGestureLayer onGesture={applyGestureAction} inspectMode={inspectMode} burgerExploded={burgerExploded} />

      {/* ── Product info panel — Apple-style premium glass card ── */}
      <div
        className={`pointer-events-none absolute bottom-[6.5rem] right-4 md:right-8 w-[min(20rem,calc(50vw-1rem))] max-h-[min(46vh,24rem)] overflow-y-auto border border-white/18 bg-white/26 p-5 text-left text-stone-800 shadow-2xl shadow-stone-900/6 backdrop-blur-2xl transition-all duration-700 md:p-6 ${
          exploded
            ? "translate-y-0 opacity-100"
            : "translate-y-4 opacity-0"
        }`}
      >
        {/* Context label — only shown in inspect */}
        {inspectMode && (
          <p className="mb-3 text-[0.46rem] tracking-[0.46em] text-stone-400/42">
            INSPECT
          </p>
        )}

        {/* Special tag — refined, minimal */}
        {FOOD_INSPECT_DATA[activePartIndex].special && (
          <div className="mb-3 inline-flex items-center gap-2">
            <span className="h-[2px] w-2.5 bg-amber-500/50" />
            <span className="text-[0.44rem] tracking-[0.34em] text-amber-700/60">
              {FOOD_INSPECT_DATA[activePartIndex].special}
            </span>
          </div>
        )}

        {/* Name + price row */}
        <div className="flex items-start justify-between gap-4">
          <h2
            className={`font-light leading-snug tracking-[0.10em] text-stone-800 transition-all duration-500 ${
              inspectMode ? "text-lg" : "text-base"
            }`}
          >
            {activePart.name}
          </h2>
          <span className="mt-0.5 shrink-0 text-[0.72rem] font-light tracking-[0.04em] text-amber-800/70">
            ${ITEM_PRICES[activePartIndex].toFixed(2)}
          </span>
        </div>

        {inspectMode ? (
          <>
            {/* Nutritional grid — clean two-column */}
            <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-stone-100/60 pt-4">
              <div>
                <p className="text-[0.44rem] tracking-[0.24em] text-stone-400/55">CALORIES</p>
                <p className="mt-0.5 text-[0.70rem] font-light text-stone-700">
                  {FOOD_INSPECT_DATA[activePartIndex].calories}
                </p>
              </div>
              <div>
                <p className="text-[0.44rem] tracking-[0.24em] text-stone-400/55">PROTEIN</p>
                <p className="mt-0.5 text-[0.70rem] font-light text-stone-700">
                  {FOOD_INSPECT_DATA[activePartIndex].protein}
                </p>
              </div>
            </div>

            {/* Allergens */}
            <div className="mt-3">
              <p className="text-[0.44rem] tracking-[0.24em] text-stone-400/55">ALLERGENS</p>
              <p className="mt-0.5 text-[0.60rem] font-light text-stone-500/80">
                {FOOD_INSPECT_DATA[activePartIndex].allergens}
              </p>
            </div>

            {/* Ingredients */}
            <div className="mt-4 border-t border-stone-100/55 pt-4">
              <p className="text-[0.44rem] tracking-[0.24em] text-stone-400/55">INGREDIENTS</p>
              <p className="mt-1 text-[0.58rem] leading-[1.65] text-stone-600/78">
                {FOOD_INSPECT_DATA[activePartIndex].ingredients}
              </p>
            </div>

            {/* Flavor */}
            <div className="mt-3">
              <p className="text-[0.44rem] tracking-[0.24em] text-stone-400/55">FLAVOR</p>
              <p className="mt-0.5 text-[0.58rem] leading-[1.65] text-stone-500/70">
                {FOOD_INSPECT_DATA[activePartIndex].flavorProfile}
              </p>
            </div>

            {/* Chef note — italic, quiet */}
            <p className="mt-4 border-t border-stone-100/50 pt-4 text-[0.56rem] italic leading-[1.7] text-amber-800/45">
              {FOOD_INSPECT_DATA[activePartIndex].chefNote}
            </p>

          </>
        ) : (
          <p className="mt-3 text-[0.76rem] leading-[1.65] text-stone-500/68">
            {activePart.description}
          </p>
        )}
      </div>

      {/* ── Ingredient HUD — premium tasting note card, burger exploded only ── */}
      <div
        className={`pointer-events-none absolute left-4 top-1/2 w-[min(17rem,calc(100vw-2rem))] -translate-y-1/2 overflow-hidden border border-amber-200/22 bg-white/28 shadow-2xl shadow-stone-900/8 backdrop-blur-2xl transition-all duration-700 md:left-6 ${
          burgerExploded && inspectMode && activePartIndex === 0
            ? "opacity-100"
            : "opacity-0"
        }`}
      >
        {/* Warm amber top accent bar */}
        <div className="h-[2px] w-full bg-gradient-to-r from-amber-500/0 via-amber-500/40 to-amber-500/0" />
        <div className="p-5 md:p-6">
          <p className="mb-1 text-[0.40rem] tracking-[0.50em] text-amber-700/50">
            TASTING NOTES
          </p>
          <p className="mb-4 text-[0.64rem] font-light tracking-[0.12em] text-stone-700/80">
            Signature Burger
          </p>
          <ul className="space-y-3">
            {BURGER_INGREDIENTS.map((ingredient) => (
              <li key={ingredient.name} className="border-b border-stone-200/35 pb-3 last:border-0 last:pb-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[0.68rem] font-light tracking-[0.06em] text-stone-800/90">
                    {ingredient.name}
                  </span>
                  <span className="shrink-0 text-[0.48rem] tracking-[0.10em] text-amber-700/44">
                    {ingredient.cal}
                  </span>
                </div>
                <p className="mt-0.5 text-[0.54rem] leading-[1.55] text-stone-500/68">
                  {ingredient.flavor}
                </p>
                {ingredient.allergen !== "None" && (
                  <p className="mt-0.5 text-[0.46rem] tracking-[0.06em] text-amber-800/32">
                    {ingredient.allergen}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </div>
        {/* Warm bottom fade */}
        <div className="h-[1px] w-full bg-gradient-to-r from-stone-300/0 via-stone-300/28 to-stone-300/0" />
      </div>

      {/* ── Bottom branding + contextual hint ── */}
      <div className="pointer-events-none absolute bottom-5 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-center">
        {/* Gesture hint — visible in menu phase */}
        <p className={`whitespace-nowrap text-[0.44rem] tracking-[0.28em] text-stone-400/28 transition-opacity duration-700 ${landingPhase === "menu" && exploded ? "opacity-100" : "opacity-0"}`}>
          {inspectMode && activePartIndex === 0
            ? burgerExploded
              ? "Hold open hand · E to assemble"
              : "Hold open hand · E to reveal layers"
            : inspectMode
              ? "Open hand to add  ·  Swipe to return"
              : "Swipe to explore  ·  Open hand to inspect"}
        </p>
        <p className="text-[0.54rem] tracking-[0.52em] text-stone-500/38">AURA ONE</p>
      </div>

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
