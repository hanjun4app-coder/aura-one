"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import {
  type MutableRefObject,
  type ReactNode,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";

type PartId =
  | "vehicle-body"
  | "cabin-module"
  | "front-module"
  | "rear-module"
  | "battery-plate"
  | "wheel-module";

const PART_INFO: Record<PartId, { name: string; description: string }> = {
  "vehicle-body": {
    name: "Vehicle Body",
    description: "Main structural platform that anchors the vehicle assembly.",
  },
  "cabin-module": {
    name: "Cabin Module",
    description: "Passenger and interface area with a quiet panoramic shell.",
  },
  "front-module": {
    name: "Front Module",
    description: "Forward sensing and lighting section for spatial awareness.",
  },
  "rear-module": {
    name: "Rear Module",
    description: "Rear structure and signaling area for vehicle presence.",
  },
  "battery-plate": {
    name: "Battery Plate",
    description: "Power storage foundation positioned below the main body.",
  },
  "wheel-module": {
    name: "Wheel Module",
    description: "Motion and support system separated for inspection.",
  },
};

function smoothStep(value: number) {
  return value * value * (3 - 2 * value);
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

function Part({
  partId,
  basePosition,
  midPosition,
  explodedPosition,
  explodeDelay = 0,
  assembleDelay = 0,
  baseRotation = [0, 0, 0],
  explodedRotation = [0, 0, 0],
  selfRotationAmount = 1,
  motionSeed = 0,
  children,
  progressRef,
  selectedPartId,
  selectionEnabled,
  onSelect,
}: {
  partId: PartId;
  basePosition: [number, number, number];
  midPosition?: [number, number, number];
  explodedPosition: [number, number, number];
  explodeDelay?: number;
  assembleDelay?: number;
  baseRotation?: [number, number, number];
  explodedRotation?: [number, number, number];
  selfRotationAmount?: number;
  motionSeed?: number;
  children: ReactNode;
  progressRef: MutableRefObject<number>;
  selectedPartId: PartId | null;
  selectionEnabled: boolean;
  onSelect: (partId: PartId) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const selfRotationRef = useRef(new THREE.Euler(0, 0, 0));
  const selectionScaleRef = useRef(1);
  const highlightRef = useRef(0);
  const previousProgressRef = useRef(0);
  const previousRawProgressRef = useRef(0);
  const dockedRef = useRef(true);
  const dockPhaseRef = useRef<number | null>(null);
  const pathPositionRef = useRef(new THREE.Vector3());
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
    const selected = selectionEnabled && selectedPartId === partId;
    const selectionPresence = selected ? separatedProgress : 0;
    const scaleTarget = 1 + selectionPresence * 0.07;
    const highlightTarget = selectionPresence * 0.55;

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

    selfRotationRef.current.x +=
      delta * motion.rotationSpeed.x * separatedProgress * (selected ? 1.42 : 1);
    selfRotationRef.current.y +=
      delta * motion.rotationSpeed.y * separatedProgress * (selected ? 1.42 : 1);
    selfRotationRef.current.z +=
      delta * motion.rotationSpeed.z * separatedProgress * (selected ? 1.42 : 1);
    selectionScaleRef.current = THREE.MathUtils.lerp(
      selectionScaleRef.current,
      scaleTarget,
      1 - Math.exp(-delta * 7)
    );
    highlightRef.current = THREE.MathUtils.lerp(
      highlightRef.current,
      highlightTarget,
      1 - Math.exp(-delta * 6)
    );

    groupRef.current.position.set(
      pathPosition.x +
        Math.cos(floatPhase * 0.7) * motion.driftAmount * separatedProgress +
        motion.dockDirection.x * motion.dockAmplitude * dockPulse,
      pathPosition.y +
        Math.sin(floatPhase) * motion.floatAmount * separatedProgress +
        motion.dockDirection.y * motion.dockAmplitude * dockPulse,
      pathPosition.z +
        Math.sin(floatPhase * 0.8) * motion.driftAmount * separatedProgress +
        motion.dockDirection.z * motion.dockAmplitude * dockPulse
    );

    groupRef.current.rotation.set(
      THREE.MathUtils.lerp(baseRotation[0], explodedRotation[0], separatedProgress) +
        selfRotationRef.current.x * separatedProgress +
        motion.dockRotation.x * dockPulse,
      THREE.MathUtils.lerp(baseRotation[1], explodedRotation[1], separatedProgress) +
        selfRotationRef.current.y * separatedProgress +
        motion.dockRotation.y * dockPulse,
      THREE.MathUtils.lerp(baseRotation[2], explodedRotation[2], separatedProgress) +
        selfRotationRef.current.z * separatedProgress +
        motion.dockRotation.z * dockPulse
    );
    groupRef.current.scale.setScalar(selectionScaleRef.current);

    groupRef.current.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const material = mesh.material as THREE.MeshStandardMaterial | undefined;

      if (!material || !("emissiveIntensity" in material)) return;

      material.emissive.set("#67e8f9");
      material.emissiveIntensity = highlightRef.current;
    });

    previousProgressRef.current = localProgress;
    previousRawProgressRef.current = rawProgress;
  });

  return (
    <group
      ref={groupRef}
      onPointerDown={(event) => {
        if (!selectionEnabled) return;

        event.stopPropagation();
        onSelect(partId);
      }}
    >
      {children}
      {selectionEnabled && selectedPartId === partId ? (
        <pointLight
          color="#67e8f9"
          distance={1.8}
          intensity={0.28}
          position={[0, 0.25, 0]}
        />
      ) : null}
    </group>
  );
}

function SimplifiedCarProduct({
  exploded,
  selectedPartId,
  onSelectPart,
}: {
  exploded: boolean;
  selectedPartId: PartId | null;
  onSelectPart: (partId: PartId) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const progressRef = useRef(0);

  useFrame(({ clock }, delta) => {
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      exploded ? 1 : 0,
      delta * 2.2
    );

    if (!groupRef.current) return;

    const p = smoothStep(progressRef.current);
    const assembledSpin = 1 - p;
    const t = clock.getElapsedTime();

    groupRef.current.rotation.y += delta * 0.18 * assembledSpin;
    groupRef.current.position.y = Math.sin(t * 0.65) * 0.045 * assembledSpin;
  });

  return (
    <group ref={groupRef} scale={0.85}>
      <Part
        partId="vehicle-body"
        basePosition={[0, 0, 0]}
        explodedPosition={[0, 0, 0]}
        selfRotationAmount={0}
        motionSeed={1}
        progressRef={progressRef}
        selectedPartId={selectedPartId}
        selectionEnabled={exploded}
        onSelect={onSelectPart}
      >
        <mesh>
          <boxGeometry args={[3.2, 0.55, 1.35]} />
          <meshStandardMaterial
            color="#38bdf8"
            metalness={0.55}
            roughness={0.22}
          />
        </mesh>
      </Part>

      <Part
        partId="cabin-module"
        basePosition={[0.15, 0.55, 0]}
        midPosition={[0.15, 1.32, 0.08]}
        explodedPosition={[0.15, 1.86, 0.28]}
        explodeDelay={0}
        assembleDelay={0.07}
        explodedRotation={[0, 0.08, 0]}
        selfRotationAmount={0.55}
        motionSeed={2}
        progressRef={progressRef}
        selectedPartId={selectedPartId}
        selectionEnabled={exploded}
        onSelect={onSelectPart}
      >
        <mesh>
          <boxGeometry args={[1.35, 0.65, 1.05]} />
          <meshStandardMaterial
            color="#67e8f9"
            metalness={0.4}
            roughness={0.18}
            transparent
            opacity={0.82}
          />
        </mesh>
      </Part>

      <Part
        partId="front-module"
        basePosition={[1.75, 0.05, 0]}
        midPosition={[2.58, 0.42, 0.12]}
        explodedPosition={[3.42, 0.3, 0.16]}
        explodeDelay={0.035}
        assembleDelay={0.04}
        explodedRotation={[0, 0.14, 0]}
        selfRotationAmount={0.62}
        motionSeed={3}
        progressRef={progressRef}
        selectedPartId={selectedPartId}
        selectionEnabled={exploded}
        onSelect={onSelectPart}
      >
        <mesh>
          <boxGeometry args={[0.35, 0.35, 1.2]} />
          <meshStandardMaterial
            color="#0ea5e9"
            metalness={0.5}
            roughness={0.25}
          />
        </mesh>
      </Part>

      <Part
        partId="rear-module"
        basePosition={[-1.75, 0.05, 0]}
        midPosition={[-2.58, 0.42, -0.12]}
        explodedPosition={[-3.42, 0.3, -0.16]}
        explodeDelay={0.035}
        assembleDelay={0.04}
        explodedRotation={[0, -0.14, 0]}
        selfRotationAmount={0.62}
        motionSeed={4}
        progressRef={progressRef}
        selectedPartId={selectedPartId}
        selectionEnabled={exploded}
        onSelect={onSelectPart}
      >
        <mesh>
          <boxGeometry args={[0.35, 0.35, 1.2]} />
          <meshStandardMaterial
            color="#0ea5e9"
            metalness={0.5}
            roughness={0.25}
          />
        </mesh>
      </Part>

      <Part
        partId="battery-plate"
        basePosition={[0, -0.38, 0]}
        midPosition={[0, -0.86, 0]}
        explodedPosition={[0, -1.56, 0]}
        explodeDelay={0.075}
        assembleDelay={0.02}
        explodedRotation={[0.08, 0, 0]}
        selfRotationAmount={0.45}
        motionSeed={5}
        progressRef={progressRef}
        selectedPartId={selectedPartId}
        selectionEnabled={exploded}
        onSelect={onSelectPart}
      >
        <mesh>
          <boxGeometry args={[2.2, 0.16, 1.05]} />
          <meshStandardMaterial
            color="#22c55e"
            metalness={0.35}
            roughness={0.3}
          />
        </mesh>
      </Part>

      {[
        {
          base: [1.05, -0.35, 0.78],
          mid: [1.56, -0.58, 1.18],
          exploded: [2.32, -0.92, 1.72],
          rot: [Math.PI / 2, 0.18, 0],
        },
        {
          base: [-1.05, -0.35, 0.78],
          mid: [-1.56, -0.58, 1.18],
          exploded: [-2.32, -0.92, 1.72],
          rot: [Math.PI / 2, -0.18, 0],
        },
        {
          base: [1.05, -0.35, -0.78],
          mid: [1.56, -0.58, -1.18],
          exploded: [2.32, -0.92, -1.72],
          rot: [Math.PI / 2, -0.18, 0],
        },
        {
          base: [-1.05, -0.35, -0.78],
          mid: [-1.56, -0.58, -1.18],
          exploded: [-2.32, -0.92, -1.72],
          rot: [Math.PI / 2, 0.18, 0],
        },
      ].map((wheel, index) => (
        <Part
          key={index}
          partId="wheel-module"
          basePosition={wheel.base as [number, number, number]}
          midPosition={wheel.mid as [number, number, number]}
          explodedPosition={wheel.exploded as [number, number, number]}
          explodeDelay={0.095 + index * 0.012}
          assembleDelay={index * 0.008}
          baseRotation={[Math.PI / 2, 0, 0]}
          explodedRotation={wheel.rot as [number, number, number]}
          selfRotationAmount={0.48}
          motionSeed={index + 6}
          progressRef={progressRef}
          selectedPartId={selectedPartId}
          selectionEnabled={exploded}
          onSelect={onSelectPart}
        >
          <mesh>
            <cylinderGeometry args={[0.34, 0.34, 0.22, 32]} />
            <meshStandardMaterial
              color="#020617"
              metalness={0.45}
              roughness={0.3}
            />
          </mesh>
        </Part>
      ))}
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

export default function SpatialScene() {
  const [exploded, setExploded] = useState(false);
  const [selectedPartId, setSelectedPartId] = useState<PartId | null>(null);
  const selectedPart = selectedPartId ? PART_INFO[selectedPartId] : null;

  return (
    <div className="absolute inset-0">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
        <color attach="background" args={["#020617"]} />

        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 4, 4]} intensity={1.8} />
        <pointLight position={[-4, -2, 3]} intensity={2} color="#38bdf8" />

        <AmbientParticles />
        <SimplifiedCarProduct
          exploded={exploded}
          selectedPartId={selectedPartId}
          onSelectPart={setSelectedPartId}
        />

        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>

      <div
        className={`pointer-events-none absolute left-6 top-6 w-[min(22rem,calc(100vw-3rem))] border border-cyan-200/20 bg-slate-950/45 p-5 text-cyan-50 shadow-2xl shadow-cyan-950/20 backdrop-blur-md transition duration-500 ${
          exploded && selectedPart
            ? "translate-y-0 opacity-100"
            : "-translate-y-2 opacity-0"
        }`}
      >
        <p className="mb-2 text-[0.65rem] tracking-[0.32em] text-cyan-200/60">
          COMPONENT
        </p>
        <h2 className="text-lg font-light tracking-[0.14em]">
          {selectedPart?.name}
        </h2>
        <p className="mt-3 text-sm leading-6 text-cyan-50/65">
          {selectedPart?.description}
        </p>
      </div>

      <button
        onClick={() =>
          setExploded((value) => {
            if (value) {
              setSelectedPartId(null);
            }

            return !value;
          })
        }
        className="absolute bottom-8 left-1/2 -translate-x-1/2 rounded-full border border-cyan-300/40 bg-cyan-300/10 px-6 py-3 text-sm tracking-[0.25em] text-cyan-100 backdrop-blur-md transition hover:bg-cyan-300/20"
      >
        {exploded ? "ASSEMBLE" : "EXPLODE"}
      </button>
    </div>
  );
}
