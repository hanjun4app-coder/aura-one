"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
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
    name: "Central AI Core",
    description: "Real-time surgical intelligence and coordination module.",
  },
  {
    name: "Robotic Arm",
    description: "Precision robotic arm for stable surgical assistance.",
  },
  {
    name: "Vision Camera Module",
    description: "High-resolution surgical vision and depth sensing system.",
  },
  {
    name: "Surgical Tool Tip",
    description: "Interchangeable precision tool interface.",
  },
  {
    name: "Stabilization Base",
    description: "Anti-vibration platform for stable operation.",
  },
  {
    name: "Sensor Ring",
    description: "Spatial awareness and patient-side safety sensor array.",
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
  | "ROTATE_INSPECT_UP"
  | "ROTATE_INSPECT_DOWN";

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
        direction * (0.012 + (motionSeed % 3) * 0.004),
        -direction * (0.018 + (motionSeed % 4) * 0.004),
        direction * (0.01 + (motionSeed % 5) * 0.003)
      ),
      rotationSpeed: new THREE.Vector3(
        direction * (0.045 + (motionSeed % 3) * 0.012) * selfRotationAmount,
        -direction * (0.06 + (motionSeed % 4) * 0.012) * selfRotationAmount,
        direction * (0.035 + (motionSeed % 5) * 0.008) * selfRotationAmount
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
      THREE.MathUtils.lerp(1, carouselScale, carouselPresence),
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
      selfRotationRef.current.x +=
        delta * motion.rotationSpeed.x * separatedProgress * activeRotationBoost;
      selfRotationRef.current.y +=
        delta * motion.rotationSpeed.y * separatedProgress * activeRotationBoost;
      selfRotationRef.current.z +=
        delta * motion.rotationSpeed.z * separatedProgress * activeRotationBoost;
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
      activeLightRef.current.intensity =
        highlightRef.current * (isInspectActive ? 3.2 : 0.66);
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
    const neighborSpread = slotDistance === 1 ? 0.95 : slotDistance === 2 ? 0.4 : 0;
    const neighborDepthOffset = slotDistance === 1 ? 0.62 : 0;
    const carouselX = Math.sin(angle) * (4.25 + neighborSpread);
    const carouselY = 0.24 + Math.sin(angle * 2) * 0.14 - side * 0.1;
    const carouselZ = depth * 1.85 - side * 0.78 - neighborDepthOffset;
    const rearDirection = slot === 0 ? 0 : Math.sign(slot) || 1;
    const inspectX = slot === 0 ? 0 : rearDirection * (4.2 + slotDistance * 0.85);
    const inspectY = slot === 0 ? 0.0 : -0.15 - slotDistance * 0.12;
    const inspectZ = slot === 0 ? 0.9 : -4.5 - slotDistance * 1.8;
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
      inspectMode && activePresence ? inspectRotationRef.current.x : 0,
      inspectMode && activePresence ? inspectRotationRef.current.y : 0
    );
    manualRotationRef.current.lerp(
      manualRotationTargetRef.current,
      1 - Math.exp(-delta * 6)
    );

    groupRef.current.rotation.set(
      THREE.MathUtils.lerp(baseRotation[0], explodedRotation[0], separatedProgress) +
        (isInspectActive ? 0 : selfRotationRef.current.x * separatedProgress) +
        manualRotationRef.current.x +
        motion.dockRotation.x * dockPulse,
      THREE.MathUtils.lerp(baseRotation[1], explodedRotation[1], separatedProgress) +
        (isInspectActive
          ? inspectIdleYRef.current
          : selfRotationRef.current.y * separatedProgress) +
        manualRotationRef.current.y +
        motion.dockRotation.y * dockPulse,
      THREE.MathUtils.lerp(baseRotation[2], explodedRotation[2], separatedProgress) +
        (isInspectActive ? 0 : selfRotationRef.current.z * separatedProgress) +
        motion.dockRotation.z * dockPulse
    );
    groupRef.current.scale.setScalar(partScaleRef.current);

    groupRef.current.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const material = mesh.material as THREE.MeshStandardMaterial | undefined;

      if (!material || !("emissiveIntensity" in material)) return;

      material.emissive.set("#67e8f9");
      const emissiveBoost = isInspectActive ? 1.8 : 1;
      material.emissiveIntensity =
        highlightRef.current * emissiveBoost * (1 - dimRef.current * 0.85);
      material.transparent = meshOpacity < 1 || dimRef.current > 0.005;
      material.opacity = THREE.MathUtils.lerp(meshOpacity, 0.22, dimRef.current);
    });

    previousProgressRef.current = localProgress;
    previousRawProgressRef.current = rawProgress;
  });

  return (
    <group ref={groupRef}>
      {children}
      <pointLight
        ref={activeLightRef}
        color="#67e8f9"
        distance={2.4}
        intensity={0}
        position={[0, 0.25, 0]}
      />
    </group>
  );
}

function SurgicalRobotProduct({
  exploded,
  activePartIndex,
  inspectMode,
  inspectRotationRef,
}: {
  exploded: boolean;
  activePartIndex: number;
  inspectMode: boolean;
  inspectRotationRef: MutableRefObject<{ x: number; y: number }>;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const progressRef = useRef(0);
  const n = CAROUSEL_PARTS.length;

  useFrame(({ clock }, delta) => {
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      exploded ? 1 : 0,
      delta * 2.2
    );

    if (!groupRef.current) return;

    const p = smoothStep(progressRef.current);
    const t = clock.getElapsedTime();

    groupRef.current.rotation.y += delta * 0.1 * (1 - p);
    groupRef.current.position.y = Math.sin(t * 0.46) * 0.03 * (1 - p);
  });

  return (
    <group ref={groupRef} scale={0.9}>

      {/* ── Part 0: Central AI Core — tall precision column ── */}
      <Part
        partIndex={0}
        basePosition={[0, 0.06, 0]}
        explodedPosition={[0, 0.32, 0]}
        focusScale={1.18}
        secondaryScale={0.64}
        selfRotationAmount={0.1}
        motionSeed={1}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
      >
        {/* Main lower column body — matte white ceramic */}
        <mesh position={[0, -0.28, 0]}>
          <cylinderGeometry args={[0.182, 0.212, 1.6, 32]} />
          <meshStandardMaterial color="#F0F4F8" metalness={0.38} roughness={0.42} />
        </mesh>
        {/* Shoulder mount flare — slightly wider section where arms attach */}
        <mesh position={[0, 0.24, 0]}>
          <cylinderGeometry args={[0.268, 0.262, 0.09, 32]} />
          <meshStandardMaterial color="#E2E8F0" metalness={0.52} roughness={0.3} />
        </mesh>
        {/* Upper column — tapers toward vision head */}
        <mesh position={[0, 0.66, 0]}>
          <cylinderGeometry args={[0.158, 0.182, 0.72, 28]} />
          <meshStandardMaterial color="#F0F4F8" metalness={0.38} roughness={0.42} />
        </mesh>
        {/* Vision module neck transition */}
        <mesh position={[0, 1.07, 0]}>
          <cylinderGeometry args={[0.128, 0.158, 0.24, 24]} />
          <meshStandardMaterial color="#E8EDF2" metalness={0.46} roughness={0.34} />
        </mesh>
        {/* Base pedestal stub — connects to stabilization base */}
        <mesh position={[0, -1.04, 0]}>
          <cylinderGeometry args={[0.218, 0.258, 0.14, 24]} />
          <meshStandardMaterial color="#D0D8E4" metalness={0.62} roughness={0.24} />
        </mesh>
        {/* Shoulder status ring — single thin cyan line only */}
        <mesh position={[0, 0.25, 0]}>
          <torusGeometry args={[0.292, 0.01, 10, 48]} />
          <meshStandardMaterial color="#00d4ff" emissive="#00d4ff" emissiveIntensity={0.7} metalness={0.5} roughness={0.1} />
        </mesh>
        {/* Panel seam rings — subtle surface language */}
        <mesh position={[0, -0.5, 0]}>
          <torusGeometry args={[0.202, 0.006, 8, 48]} />
          <meshStandardMaterial color="#C8D4E0" metalness={0.7} roughness={0.22} />
        </mesh>
        <mesh position={[0, 0.48, 0]}>
          <torusGeometry args={[0.178, 0.006, 8, 48]} />
          <meshStandardMaterial color="#C8D4E0" metalness={0.7} roughness={0.22} />
        </mesh>
      </Part>

      {/* ── Part 1: Robotic Arms — 4-arm precision cross array ── */}
      <Part
        partIndex={1}
        basePosition={[0, 0.24, 0]}
        explodedPosition={[0, 0.65, 0]}
        focusScale={1.14}
        secondaryScale={0.6}
        explodeDelay={0.022}
        assembleDelay={0.042}
        selfRotationAmount={0.15}
        motionSeed={2}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
      >
        {/* 4 arms at 90° intervals — each a rotated group along +X */}
        {([0, 1, 2, 3] as const).map((i) => (
          <group key={i} rotation={[0, (i / 4) * Math.PI * 2, 0]}>
            {/* Shoulder mount sphere — ball joint at column face */}
            <mesh position={[0.27, 0, 0]}>
              <sphereGeometry args={[0.07, 14, 10]} />
              <meshStandardMaterial color="#E8EDF2" metalness={0.84} roughness={0.14} />
            </mesh>
            {/* Upper arm segment — brushed aluminum */}
            <mesh position={[0.52, -0.09, 0]} rotation={[0, 0, -0.26]}>
              <cylinderGeometry args={[0.038, 0.052, 0.52, 14]} />
              <meshStandardMaterial color="#D0D8E4" metalness={0.8} roughness={0.2} />
            </mesh>
            {/* Elbow joint sphere */}
            <mesh position={[0.8, -0.23, 0]}>
              <sphereGeometry args={[0.058, 12, 10]} />
              <meshStandardMaterial color="#E2E8F0" metalness={0.86} roughness={0.13} />
            </mesh>
            {/* Forearm segment — slimmer, more precise */}
            <mesh position={[1.04, -0.42, 0]} rotation={[0, 0, -0.42]}>
              <cylinderGeometry args={[0.026, 0.038, 0.5, 12]} />
              <meshStandardMaterial color="#CBD5E1" metalness={0.82} roughness={0.18} />
            </mesh>
            {/* Wrist joint sphere */}
            <mesh position={[1.23, -0.63, 0]}>
              <sphereGeometry args={[0.044, 10, 8]} />
              <meshStandardMaterial color="#E2E8F0" metalness={0.86} roughness={0.13} />
            </mesh>
            {/* Instrument shaft — precision thin */}
            <mesh position={[1.4, -0.82, 0]} rotation={[0, 0, -0.56]}>
              <cylinderGeometry args={[0.012, 0.018, 0.42, 10]} />
              <meshStandardMaterial color="#F0F4F8" metalness={0.92} roughness={0.07} />
            </mesh>
          </group>
        ))}
      </Part>

      {/* ── Part 2: Vision Camera Module — Vision Pro black-glass face ── */}
      <Part
        partIndex={2}
        basePosition={[0, 1.16, 0.04]}
        midPosition={[0.06, 2.05, 0.1]}
        explodedPosition={[0.12, 2.82, 0.18]}
        focusScale={1.28}
        secondaryScale={0.64}
        explodeDelay={0.04}
        assembleDelay={0.065}
        explodedRotation={[0.08, 0.18, 0]}
        selfRotationAmount={0.44}
        motionSeed={3}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
      >
        {/* White ceramic housing body */}
        <mesh>
          <cylinderGeometry args={[0.222, 0.222, 0.26, 32]} />
          <meshStandardMaterial color="#F0F4F8" metalness={0.42} roughness={0.36} />
        </mesh>
        {/* Black glass front face — Vision Pro language */}
        <mesh position={[0, 0, 0.19]}>
          <cylinderGeometry args={[0.19, 0.19, 0.055, 32]} />
          <meshStandardMaterial color="#06090f" metalness={0.94} roughness={0.04} />
        </mesh>
        {/* Left stereo camera aperture */}
        <mesh position={[-0.076, 0, 0.22]}>
          <cylinderGeometry args={[0.04, 0.04, 0.036, 18]} />
          <meshStandardMaterial color="#020408" metalness={0.96} roughness={0.02} />
        </mesh>
        {/* Right stereo camera aperture */}
        <mesh position={[0.076, 0, 0.22]}>
          <cylinderGeometry args={[0.04, 0.04, 0.036, 18]} />
          <meshStandardMaterial color="#020408" metalness={0.96} roughness={0.02} />
        </mesh>
        {/* Depth sensing ring — single restrained cyan line */}
        <mesh position={[0, -0.11, 0]}>
          <torusGeometry args={[0.228, 0.01, 8, 48]} />
          <meshStandardMaterial color="#00d4ff" emissive="#00d4ff" emissiveIntensity={0.55} metalness={0.5} roughness={0.1} />
        </mesh>
        {/* Mount stub — connects to column neck */}
        <mesh position={[0, -0.2, 0]}>
          <cylinderGeometry args={[0.098, 0.138, 0.11, 18]} />
          <meshStandardMaterial color="#D0D8E4" metalness={0.66} roughness={0.24} />
        </mesh>
      </Part>

      {/* ── Part 3: Surgical Tool Tip — precision instrument ── */}
      <Part
        partIndex={3}
        basePosition={[1.14, -0.74, 0.38]}
        midPosition={[1.95, -1.02, 0.72]}
        explodedPosition={[2.88, -1.32, 1.08]}
        focusScale={1.26}
        secondaryScale={0.62}
        explodeDelay={0.055}
        assembleDelay={0.038}
        explodedRotation={[0.18, 0.28, 0.1]}
        selfRotationAmount={0.5}
        motionSeed={4}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
      >
        {/* Handle grip — brushed aluminum */}
        <mesh position={[0, 0.38, 0]}>
          <cylinderGeometry args={[0.062, 0.078, 0.54, 18]} />
          <meshStandardMaterial color="#CBD5E1" metalness={0.84} roughness={0.18} />
        </mesh>
        {/* Interface collar — single cyan accent band */}
        <mesh position={[0, 0.07, 0]}>
          <cylinderGeometry args={[0.088, 0.088, 0.066, 18]} />
          <meshStandardMaterial color="#00d4ff" emissive="#00d4ff" emissiveIntensity={0.5} metalness={0.65} roughness={0.1} />
        </mesh>
        {/* Tapered precision shaft */}
        <mesh position={[0, -0.3, 0]}>
          <cylinderGeometry args={[0.02, 0.052, 0.64, 16]} />
          <meshStandardMaterial color="#F0F4F8" metalness={0.92} roughness={0.08} />
        </mesh>
        {/* Precision tip — polished white */}
        <mesh position={[0, -0.69, 0]}>
          <coneGeometry args={[0.02, 0.14, 14]} />
          <meshStandardMaterial color="#F8FAFC" metalness={0.96} roughness={0.04} />
        </mesh>
      </Part>

      {/* ── Part 4: Stabilization Base — wide premium medical platform ── */}
      <Part
        partIndex={4}
        basePosition={[0, -1.3, 0]}
        midPosition={[0, -1.82, 0]}
        explodedPosition={[0, -2.52, 0]}
        focusScale={1.12}
        secondaryScale={0.64}
        explodeDelay={0.085}
        assembleDelay={0.014}
        explodedRotation={[0.04, 0, 0]}
        selfRotationAmount={0.18}
        motionSeed={5}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
      >
        {/* Main outer platform disc — dark gray composite */}
        <mesh>
          <cylinderGeometry args={[1.65, 1.72, 0.1, 48]} />
          <meshStandardMaterial color="#1a2230" metalness={0.64} roughness={0.32} />
        </mesh>
        {/* Tray rim bevel */}
        <mesh position={[0, 0.07, 0]}>
          <cylinderGeometry args={[1.54, 1.65, 0.04, 48]} />
          <meshStandardMaterial color="#212d40" metalness={0.6} roughness={0.35} />
        </mesh>
        {/* Mid-tier riser disc */}
        <mesh position={[0, 0.14, 0]}>
          <cylinderGeometry args={[0.92, 1.02, 0.1, 36]} />
          <meshStandardMaterial color="#1e293b" metalness={0.66} roughness={0.28} />
        </mesh>
        {/* Column mount center hub */}
        <mesh position={[0, 0.24, 0]}>
          <cylinderGeometry args={[0.3, 0.36, 0.14, 24]} />
          <meshStandardMaterial color="#2d3a4e" metalness={0.72} roughness={0.22} />
        </mesh>
        {/* Anti-vibration ring — one restrained cyan line */}
        <mesh position={[0, 0.07, 0]}>
          <torusGeometry args={[1.3, 0.024, 10, 64]} />
          <meshStandardMaterial color="#00d4ff" emissive="#00d4ff" emissiveIntensity={0.24} metalness={0.58} roughness={0.2} />
        </mesh>
        {/* 6 stabilizer foot pads */}
        {([0, 1, 2, 3, 4, 5] as const).map((i) => (
          <mesh
            key={i}
            position={[
              Math.cos((i / 6) * Math.PI * 2) * 1.52,
              -0.08,
              Math.sin((i / 6) * Math.PI * 2) * 1.52,
            ]}
          >
            <cylinderGeometry args={[0.05, 0.065, 0.06, 10]} />
            <meshStandardMaterial color="#0d1520" metalness={0.6} roughness={0.38} />
          </mesh>
        ))}
      </Part>

      {/* ── Part 5: Sensor Ring — safety proximity array ── */}
      <Part
        partIndex={5}
        basePosition={[0, -0.88, 0]}
        midPosition={[0, -0.55, 0]}
        explodedPosition={[0, -0.35, 0]}
        focusScale={1.2}
        secondaryScale={0.62}
        explodeDelay={0.068}
        assembleDelay={0.05}
        explodedRotation={[0.1, 0.3, 0]}
        selfRotationAmount={0.46}
        motionSeed={6}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
      >
        {/* Outer structural band — brushed aluminum */}
        <mesh>
          <torusGeometry args={[0.72, 0.05, 14, 72]} />
          <meshStandardMaterial color="#D8E0EA" metalness={0.78} roughness={0.2} />
        </mesh>
        {/* Inner emissive fill — restrained cyan */}
        <mesh>
          <torusGeometry args={[0.72, 0.016, 10, 72]} />
          <meshStandardMaterial color="#00d4ff" emissive="#00d4ff" emissiveIntensity={0.42} metalness={0.5} roughness={0.1} />
        </mesh>
        {/* 12 sensor node blocks — evenly spaced */}
        {([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] as const).map((i) => (
          <mesh
            key={i}
            position={[
              Math.cos((i / 12) * Math.PI * 2) * 0.72,
              0,
              Math.sin((i / 12) * Math.PI * 2) * 0.72,
            ]}
          >
            <boxGeometry args={[0.024, 0.024, 0.038]} />
            <meshStandardMaterial color="#0ea5e9" emissive="#0ea5e9" emissiveIntensity={0.65} metalness={0.7} roughness={0.08} />
          </mesh>
        ))}
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
        color="#7dd3fc"
        size={0.018}
        transparent
        opacity={0.55}
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
    material.opacity = THREE.MathUtils.lerp(0.78, 0.28, p);
    material.emissiveIntensity = THREE.MathUtils.lerp(0.18, 0.06, p);
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, particles.length]}
      frustumCulled={false}
    >
      <boxGeometry args={[1, 1, 0.18]} />
      <meshStandardMaterial
        color="#e0faff"
        emissive="#67e8f9"
        emissiveIntensity={0.18}
        metalness={0.28}
        opacity={0.78}
        roughness={0.38}
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
      1 - Math.exp(-delta * 2.8)
    );

    if (keyLightRef.current) {
      keyLightRef.current.intensity = blendRef.current * 2.4;
    }

    if (rimLightRef.current) {
      rimLightRef.current.intensity = blendRef.current * 1.6;
    }
  });

  return (
    <>
      <directionalLight
        ref={keyLightRef}
        position={[-2.5, 4.5, 3.0]}
        color="#ddeeff"
        intensity={0}
      />
      <pointLight
        ref={rimLightRef}
        position={[4.0, 0.8, -0.8]}
        color="#67e8f9"
        distance={12}
        intensity={0}
      />
    </>
  );
}

function CameraGestureLayer({
  onGesture,
}: {
  onGesture: (action: GestureAction) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const handLandmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isInitializingRef = useRef(false);
  const frameRef = useRef<number | null>(null);
  const runDetectionLoopRef = useRef<() => void>(() => undefined);
  const samplesRef = useRef<Array<{ time: number; x: number }>>([]);
  const statusRef = useRef<CameraGestureStatus>("CAMERA OFF");
  const statusHoldUntilRef = useRef(0);
  const lastDetectionAtRef = useRef(0);
  const lastSwipeDirectionRef = useRef<"left" | "right" | null>(null);
  const lastSwipeTimeRef = useRef(0);
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
    lastSwipeDirectionRef.current = null;
    lastSwipeTimeRef.current = 0;
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
    [onGesture, updateStatus]
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
    <div className="absolute right-4 top-4 w-40 border border-cyan-200/15 bg-slate-950/45 p-2 text-cyan-50 shadow-xl shadow-cyan-950/20 backdrop-blur-md md:right-6 md:top-6">
      <div className="relative aspect-video overflow-hidden bg-slate-950/80">
        <video
          ref={videoRef}
          autoPlay
          className={`h-full w-full scale-x-[-1] object-cover transition duration-500 ${
            cameraEnabled ? "opacity-70" : "opacity-20"
          }`}
          muted
          playsInline
        />
        {!cameraEnabled ? (
          <div className="absolute inset-0 flex items-center justify-center text-[0.56rem] tracking-[0.2em] text-cyan-100/42">
            CAMERA
          </div>
        ) : null}
      </div>

      <div className="mt-2 flex items-center justify-between gap-2">
        <p className="text-[0.56rem] tracking-[0.18em] text-cyan-100/60">
          {cameraStatus}
        </p>
        <button
          onClick={enableCamera}
          className="border border-cyan-200/20 bg-cyan-200/10 px-2 py-1 text-[0.52rem] tracking-[0.18em] text-cyan-100 transition hover:bg-cyan-200/18"
        >
          {cameraEnabled ? "OFF" : "ENABLE"}
        </button>
      </div>
      <p className="mt-2 text-[0.52rem] leading-4 text-cyan-100/42">
        {cameraDebugStep}
      </p>
      {cameraErrorMessage ? (
        <p className="mt-1 text-[0.5rem] leading-4 text-rose-200/75">
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
  const inspectRotationRef = useRef({ x: 0, y: 0 });
  const activePart = CAROUSEL_PARTS[activePartIndex];
  const hudOnRight = activePartIndex % 2 === 0;
  const hudPositionClass = hudOnRight
    ? "right-4 bottom-36 md:right-10 md:bottom-32"
    : "left-4 bottom-36 md:left-10 md:bottom-32";
  const gestureHint = !exploded
    ? "SPACE EXPLODE"
    : inspectMode
      ? "WASD ROTATE • ESC EXIT • R RESET"
      : "← → CHANGE PART • ENTER INSPECT • ESC BACK";

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

  // Unified Gesture Action Layer — all input sources (keyboard, camera, UI buttons,
  // future MediaPipe gestures) route through this single function.
  const applyGestureAction = useCallback((action: GestureAction) => {
    if (action === "EXPLODE") {
      if (!exploded) setExploded(true);
      return;
    }

    if (action === "ASSEMBLE") {
      resetInspectRotation();
      setInspectMode(false);
      setExploded(false);
      return;
    }

    if (action === "TOGGLE_EXPLODE") {
      setExploded((value) => {
        if (value) {
          resetInspectRotation();
          setInspectMode(false);
        }

        return !value;
      });
      return;
    }

    if (action === "RESET") {
      resetInspectRotation();

      if (!inspectMode) {
        setExploded(false);
      }

      return;
    }

    if (action === "EXIT_INSPECT") {
      resetInspectRotation();

      if (inspectMode) {
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

    if (action === "PREV_PART") {
      if (exploded) showPreviousPart();
      return;
    }

    if (action === "NEXT_PART") {
      if (exploded) showNextPart();
      return;
    }

    if (!inspectMode) return;

    if (action === "ROTATE_INSPECT_LEFT") {
      inspectRotationRef.current.y += INSPECT_ROTATION_STEP;
    }

    if (action === "ROTATE_INSPECT_RIGHT") {
      inspectRotationRef.current.y -= INSPECT_ROTATION_STEP;
    }

    if (action === "ROTATE_INSPECT_UP") {
      inspectRotationRef.current.x += INSPECT_ROTATION_STEP;
    }

    if (action === "ROTATE_INSPECT_DOWN") {
      inspectRotationRef.current.x -= INSPECT_ROTATION_STEP;
    }
  }, [exploded, inspectMode, resetInspectRotation, showNextPart, showPreviousPart]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const actionMap: Record<string, GestureAction | undefined> = {
        ArrowLeft: inspectMode ? "ROTATE_INSPECT_LEFT" : "PREV_PART",
        ArrowRight: inspectMode ? "ROTATE_INSPECT_RIGHT" : "NEXT_PART",
        ArrowUp: inspectMode ? "ROTATE_INSPECT_UP" : undefined,
        ArrowDown: inspectMode ? "ROTATE_INSPECT_DOWN" : undefined,
        Enter: "ENTER_INSPECT",
        Escape: "EXIT_INSPECT",
        a: "ROTATE_INSPECT_LEFT",
        A: "ROTATE_INSPECT_LEFT",
        d: "ROTATE_INSPECT_RIGHT",
        D: "ROTATE_INSPECT_RIGHT",
        w: "ROTATE_INSPECT_UP",
        W: "ROTATE_INSPECT_UP",
        s: "ROTATE_INSPECT_DOWN",
        S: "ROTATE_INSPECT_DOWN",
        r: "RESET",
        R: "RESET",
        " ": "TOGGLE_EXPLODE",
      };
      const action = actionMap[event.key];

      if (!action) return;

      event.preventDefault();
      applyGestureAction(action);
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [applyGestureAction, inspectMode]);

  return (
    <div className="absolute inset-0">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
        <color attach="background" args={["#020617"]} />

        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 4, 4]} intensity={1.8} />
        <pointLight position={[-4, -2, 3]} intensity={2} color="#38bdf8" />
        <InspectSceneLighting inspectMode={inspectMode} />

        <AmbientParticles />
        <AuraLogoParticles exploded={exploded} />
        <SurgicalRobotProduct
          exploded={exploded}
          activePartIndex={activePartIndex}
          inspectMode={inspectMode}
          inspectRotationRef={inspectRotationRef}
        />

        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>

      <CameraGestureLayer onGesture={applyGestureAction} />

      <div
        className={`pointer-events-none absolute ${hudPositionClass} w-[min(22rem,calc(100vw-2rem))] border border-cyan-200/20 bg-slate-950/48 p-4 text-left text-cyan-50 shadow-2xl shadow-cyan-950/20 backdrop-blur-md transition-all duration-500 md:p-5 ${
          exploded
            ? "translate-y-0 opacity-100"
            : "translate-y-3 opacity-0"
        }`}
      >
        <p className="mb-2 text-[0.65rem] tracking-[0.32em] text-cyan-200/60">
          {inspectMode ? "INSPECT MODE" : "ACTIVE COMPONENT"}
        </p>
        <h2 className="text-lg font-light tracking-[0.16em]">
          {activePart.name}
        </h2>
        <p className="mt-3 text-sm leading-6 text-cyan-50/65">
          {activePart.description}
        </p>
        {inspectMode ? (
          <p className="mt-4 text-[0.62rem] tracking-[0.22em] text-cyan-200/50">
            WASD ROTATE • ESC EXIT
          </p>
        ) : null}
      </div>

      <div className="pointer-events-none absolute bottom-4 right-4 max-w-[18rem] border border-cyan-200/15 bg-slate-950/35 px-3 py-2 text-right text-[0.6rem] tracking-[0.2em] text-cyan-100/52 backdrop-blur-md md:right-6">
        {gestureHint}
      </div>

      <div
        className={`absolute bottom-8 left-1/2 flex -translate-x-1/2 items-center gap-3 transition duration-500 ${
          exploded ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        <button
          onClick={() => applyGestureAction("PREV_PART")}
          className="rounded-full border border-cyan-200/25 bg-cyan-200/10 px-4 py-2 text-[0.65rem] tracking-[0.24em] text-cyan-100 backdrop-blur-md transition hover:bg-cyan-200/18"
        >
          PREV PART
        </button>
        <button
          onClick={() => applyGestureAction("NEXT_PART")}
          className="rounded-full border border-cyan-200/25 bg-cyan-200/10 px-4 py-2 text-[0.65rem] tracking-[0.24em] text-cyan-100 backdrop-blur-md transition hover:bg-cyan-200/18"
        >
          NEXT PART
        </button>
        <button
          onClick={() => applyGestureAction(inspectMode ? "EXIT_INSPECT" : "ENTER_INSPECT")}
          className="rounded-full border border-cyan-200/35 bg-cyan-200/14 px-4 py-2 text-[0.65rem] tracking-[0.24em] text-cyan-50 backdrop-blur-md transition hover:bg-cyan-200/24"
        >
          {inspectMode ? "EXIT INSPECT" : "INSPECT"}
        </button>
      </div>

      <button
        onClick={() => applyGestureAction("TOGGLE_EXPLODE")}
        className={`absolute left-1/2 -translate-x-1/2 rounded-full border border-cyan-300/40 bg-cyan-300/10 px-6 py-3 text-sm tracking-[0.25em] text-cyan-100 backdrop-blur-md transition hover:bg-cyan-300/20 ${
          exploded ? "bottom-20" : "bottom-8"
        }`}
      >
        {exploded ? "ASSEMBLE" : "EXPLODE"}
      </button>
    </div>
  );
}
