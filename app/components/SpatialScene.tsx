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

function smoothStep(value: number) {
  return value * value * (3 - 2 * value);
}

function seededUnit(index: number) {
  const value = Math.sin(index * 12.9898) * 43758.5453;

  return value - Math.floor(value);
}

function Part({
  basePosition,
  explodedPosition,
  baseRotation = [0, 0, 0],
  explodedRotation = [0, 0, 0],
  motionSeed = 0,
  children,
  progressRef,
}: {
  basePosition: [number, number, number];
  explodedPosition: [number, number, number];
  baseRotation?: [number, number, number];
  explodedRotation?: [number, number, number];
  motionSeed?: number;
  children: ReactNode;
  progressRef: MutableRefObject<number>;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const selfRotationRef = useRef(new THREE.Euler(0, 0, 0));
  const motion = useMemo(() => {
    const direction = motionSeed % 2 === 0 ? 1 : -1;

    return {
      floatSpeed: 0.55 + (motionSeed % 5) * 0.08,
      floatAmount: 0.035 + (motionSeed % 4) * 0.007,
      driftAmount: 0.018 + (motionSeed % 3) * 0.006,
      rotationSpeed: new THREE.Vector3(
        direction * (0.08 + (motionSeed % 3) * 0.018),
        -direction * (0.11 + (motionSeed % 4) * 0.016),
        direction * (0.06 + (motionSeed % 5) * 0.012)
      ),
    };
  }, [motionSeed]);

  useFrame(({ clock }, delta) => {
    if (!groupRef.current) return;

    const t = clock.getElapsedTime();
    const p = smoothStep(progressRef.current);
    const floatPhase = t * motion.floatSpeed + motionSeed;

    selfRotationRef.current.x += delta * motion.rotationSpeed.x * p;
    selfRotationRef.current.y += delta * motion.rotationSpeed.y * p;
    selfRotationRef.current.z += delta * motion.rotationSpeed.z * p;

    groupRef.current.position.set(
      THREE.MathUtils.lerp(basePosition[0], explodedPosition[0], p) +
        Math.cos(floatPhase * 0.7) * motion.driftAmount * p,
      THREE.MathUtils.lerp(basePosition[1], explodedPosition[1], p) +
        Math.sin(floatPhase) * motion.floatAmount * p,
      THREE.MathUtils.lerp(basePosition[2], explodedPosition[2], p) +
        Math.sin(floatPhase * 0.8) * motion.driftAmount * p
    );

    groupRef.current.rotation.set(
      THREE.MathUtils.lerp(baseRotation[0], explodedRotation[0], p) +
        selfRotationRef.current.x * p,
      THREE.MathUtils.lerp(baseRotation[1], explodedRotation[1], p) +
        selfRotationRef.current.y * p,
      THREE.MathUtils.lerp(baseRotation[2], explodedRotation[2], p) +
        selfRotationRef.current.z * p
    );
  });

  return <group ref={groupRef}>{children}</group>;
}

function SimplifiedCarProduct({ exploded }: { exploded: boolean }) {
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
        basePosition={[0, 0, 0]}
        explodedPosition={[0, 0, 0]}
        motionSeed={1}
        progressRef={progressRef}
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
        basePosition={[0.15, 0.55, 0]}
        explodedPosition={[0.15, 1.55, 0]}
        explodedRotation={[0, 0.15, 0]}
        motionSeed={2}
        progressRef={progressRef}
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
        basePosition={[1.75, 0.05, 0]}
        explodedPosition={[3.05, 0.18, 0]}
        explodedRotation={[0, 0.25, 0]}
        motionSeed={3}
        progressRef={progressRef}
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
        basePosition={[-1.75, 0.05, 0]}
        explodedPosition={[-3.05, 0.18, 0]}
        explodedRotation={[0, -0.25, 0]}
        motionSeed={4}
        progressRef={progressRef}
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
        basePosition={[0, -0.38, 0]}
        explodedPosition={[0, -1.25, 0]}
        explodedRotation={[0.12, 0, 0]}
        motionSeed={5}
        progressRef={progressRef}
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
          exploded: [1.95, -0.72, 1.35],
          rot: [Math.PI / 2, 0.35, 0],
        },
        {
          base: [-1.05, -0.35, 0.78],
          exploded: [-1.95, -0.72, 1.35],
          rot: [Math.PI / 2, -0.35, 0],
        },
        {
          base: [1.05, -0.35, -0.78],
          exploded: [1.95, -0.72, -1.35],
          rot: [Math.PI / 2, -0.35, 0],
        },
        {
          base: [-1.05, -0.35, -0.78],
          exploded: [-1.95, -0.72, -1.35],
          rot: [Math.PI / 2, 0.35, 0],
        },
      ].map((wheel, index) => (
        <Part
          key={index}
          basePosition={wheel.base as [number, number, number]}
          explodedPosition={wheel.exploded as [number, number, number]}
          baseRotation={[Math.PI / 2, 0, 0]}
          explodedRotation={wheel.rot as [number, number, number]}
          motionSeed={index + 6}
          progressRef={progressRef}
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

  return (
    <div className="absolute inset-0">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
        <color attach="background" args={["#020617"]} />

        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 4, 4]} intensity={1.8} />
        <pointLight position={[-4, -2, 3]} intensity={2} color="#38bdf8" />

        <AmbientParticles />
        <SimplifiedCarProduct exploded={exploded} />

        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>

      <button
        onClick={() => setExploded((value) => !value)}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 rounded-full border border-cyan-300/40 bg-cyan-300/10 px-6 py-3 text-sm tracking-[0.25em] text-cyan-100 backdrop-blur-md transition hover:bg-cyan-300/20"
      >
        {exploded ? "ASSEMBLE" : "EXPLODE"}
      </button>
    </div>
  );
}
