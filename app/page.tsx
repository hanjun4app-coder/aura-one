"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const WORDMARK_PARTICLE_COUNT = 3200;
const BACKGROUND_PARTICLE_COUNT = 180;

type WordParticle = {
  target: THREE.Vector3;
  start: THREE.Vector3;
  exit: THREE.Vector3;
  phase: number;
};

type FieldParticle = {
  position: THREE.Vector3;
  phase: number;
};

const LETTERS: Record<string, string[]> = {
  A: [
    "01110",
    "10001",
    "10001",
    "11111",
    "10001",
    "10001",
    "10001",
  ],
  U: [
    "10001",
    "10001",
    "10001",
    "10001",
    "10001",
    "10001",
    "01110",
  ],
  R: [
    "11110",
    "10001",
    "10001",
    "11110",
    "10100",
    "10010",
    "10001",
  ],
  O: [
    "01110",
    "10001",
    "10001",
    "10001",
    "10001",
    "10001",
    "01110",
  ],
  N: [
    "10001",
    "11001",
    "10101",
    "10011",
    "10001",
    "10001",
    "10001",
  ],
  E: [
    "11111",
    "10000",
    "10000",
    "11110",
    "10000",
    "10000",
    "11111",
  ],
};

function seededNoise(seed: number) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function smoothStep01(value: number) {
  const x = THREE.MathUtils.clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}

function createWordCells() {
  const text = "AURA ONE";
  const cells: { x: number; y: number }[] = [];
  let cursor = 0;

  for (const char of text) {
    if (char === " ") {
      cursor += 3;
      continue;
    }

    const pattern = LETTERS[char];
    pattern.forEach((row, rowIndex) => {
      [...row].forEach((value, columnIndex) => {
        if (value === "1") {
          cells.push({ x: cursor + columnIndex, y: -rowIndex });
        }
      });
    });
    cursor += 6;
  }

  const minX = Math.min(...cells.map((cell) => cell.x));
  const maxX = Math.max(...cells.map((cell) => cell.x));
  const minY = Math.min(...cells.map((cell) => cell.y));
  const maxY = Math.max(...cells.map((cell) => cell.y));
  const centerX = (minX + maxX) * 0.5;
  const centerY = (minY + maxY) * 0.5;
  const scale = 0.112;
  const samples: { x: number; y: number }[] = [];

  cells.forEach((cell) => {
    for (let y = 0; y < 3; y += 1) {
      for (let x = 0; x < 3; x += 1) {
        samples.push({
          x: (cell.x + (x - 1) * 0.24 - centerX) * scale,
          y: (cell.y + (y - 1) * 0.24 - centerY) * scale,
        });
      }
    }
  });

  return samples;
}

function createWordParticles() {
  const cells = createWordCells();
  const particles: WordParticle[] = [];

  for (let i = 0; i < WORDMARK_PARTICLE_COUNT; i += 1) {
    const cell = cells[Math.floor(seededNoise(i + 17) * cells.length)];
    const target = new THREE.Vector3(
      cell.x + (seededNoise(i + 31) - 0.5) * 0.034,
      cell.y + (seededNoise(i + 43) - 0.5) * 0.034 + 0.30,
      (seededNoise(i + 59) - 0.5) * 0.085
    );
    const startAngle = seededNoise(i + 71) * Math.PI * 2;
    const startRadius = 2.1 + seededNoise(i + 83) * 1.35;
    const start = new THREE.Vector3(
      Math.cos(startAngle) * startRadius,
      (seededNoise(i + 97) - 0.5) * 2.0,
      -1.15 - seededNoise(i + 109) * 1.2
    );
    const exitAngle = startAngle + (seededNoise(i + 127) - 0.5) * 0.9;
    const exitRadius = 2.7 + seededNoise(i + 139) * 1.6;
    const exit = new THREE.Vector3(
      Math.cos(exitAngle) * exitRadius,
      Math.sin(exitAngle) * exitRadius * 0.52 + (seededNoise(i + 149) - 0.5) * 0.8,
      -0.55 - seededNoise(i + 157) * 1.35
    );

    particles.push({
      target,
      start,
      exit,
      phase: seededNoise(i + 173) * Math.PI * 2,
    });
  }

  return particles;
}

function createFieldParticles() {
  const particles: FieldParticle[] = [];

  for (let i = 0; i < BACKGROUND_PARTICLE_COUNT; i += 1) {
    const angle = seededNoise(i + 211) * Math.PI * 2;
    const radius = 1.8 + seededNoise(i + 223) * 1.4;
    particles.push({
      position: new THREE.Vector3(
        Math.cos(angle) * radius,
        (seededNoise(i + 239) - 0.5) * 1.45,
        -0.65 - seededNoise(i + 251) * 0.9
      ),
      phase: seededNoise(i + 263) * Math.PI * 2,
    });
  }

  return particles;
}

function LogoPortalParticles({ entering }: { entering: boolean }) {
  const wordPositionsRef = useRef<THREE.BufferAttribute>(null);
  const fieldPositionsRef = useRef<THREE.BufferAttribute>(null);
  const wordMaterialRef = useRef<THREE.PointsMaterial>(null);
  const fieldMaterialRef = useRef<THREE.PointsMaterial>(null);
  const groupRef = useRef<THREE.Group>(null);
  const wordParticles = useMemo(() => createWordParticles(), []);
  const fieldParticles = useMemo(() => createFieldParticles(), []);
  const wordPositions = useMemo(() => {
    const positions = new Float32Array(WORDMARK_PARTICLE_COUNT * 3);
    wordParticles.forEach((particle, index) => {
      const offset = index * 3;
      positions[offset] = particle.start.x;
      positions[offset + 1] = particle.start.y;
      positions[offset + 2] = particle.start.z;
    });
    return positions;
  }, [wordParticles]);
  const fieldPositions = useMemo(() => {
    const positions = new Float32Array(BACKGROUND_PARTICLE_COUNT * 3);
    fieldParticles.forEach((particle, index) => {
      const offset = index * 3;
      positions[offset] = particle.position.x;
      positions[offset + 1] = particle.position.y;
      positions[offset + 2] = particle.position.z;
    });
    return positions;
  }, [fieldParticles]);
  const introProgressRef = useRef(0);
  const exitProgressRef = useRef(0);

  useFrame(({ clock }, delta) => {
    const wordAttr = wordPositionsRef.current;
    const fieldAttr = fieldPositionsRef.current;
    if (!wordAttr || !fieldAttr) return;

    introProgressRef.current = THREE.MathUtils.lerp(
      introProgressRef.current,
      1,
      1 - Math.exp(-delta * 1.15)
    );
    exitProgressRef.current = THREE.MathUtils.lerp(
      exitProgressRef.current,
      entering ? 1 : 0,
      1 - Math.exp(-delta * 4.4)
    );

    const time = clock.getElapsedTime();
    const introP = smoothStep01(introProgressRef.current);
    const exitP = smoothStep01(exitProgressRef.current);
    const breathe = 1 + Math.sin(time * 0.36) * 0.012;

    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(time * 0.14) * 0.045;
      groupRef.current.rotation.z = Math.sin(time * 0.16) * 0.010;
      groupRef.current.scale.setScalar(breathe + exitP * 0.10);
    }

    if (wordMaterialRef.current) {
      const formedOpacity = THREE.MathUtils.lerp(0.18, 0.88, introP);
      wordMaterialRef.current.opacity = THREE.MathUtils.lerp(formedOpacity, 0.04, exitP);
    }

    if (fieldMaterialRef.current) {
      fieldMaterialRef.current.opacity = THREE.MathUtils.lerp(0.16, 0.025, exitP);
    }

    const wordArray = wordAttr.array as Float32Array;
    wordParticles.forEach((particle, index) => {
      const shimmer = Math.sin(time * 0.72 + particle.phase) * 0.006 * introP * (1 - exitP);
      const targetX = particle.target.x + shimmer;
      const targetY = particle.target.y + Math.cos(time * 0.54 + particle.phase) * 0.004 * introP * (1 - exitP);
      const targetZ = particle.target.z + Math.sin(time * 0.42 + particle.phase) * 0.020 * introP;
      const formedX = THREE.MathUtils.lerp(particle.start.x, targetX, introP);
      const formedY = THREE.MathUtils.lerp(particle.start.y, targetY, introP);
      const formedZ = THREE.MathUtils.lerp(particle.start.z, targetZ, introP);
      const offset = index * 3;
      wordArray[offset] = THREE.MathUtils.lerp(formedX, particle.exit.x, exitP);
      wordArray[offset + 1] = THREE.MathUtils.lerp(formedY, particle.exit.y, exitP);
      wordArray[offset + 2] = THREE.MathUtils.lerp(formedZ, particle.exit.z, exitP);
    });

    const fieldArray = fieldAttr.array as Float32Array;
    fieldParticles.forEach((particle, index) => {
      const drift = Math.sin(time * 0.18 + particle.phase) * 0.035;
      const offset = index * 3;
      fieldArray[offset] = particle.position.x + drift;
      fieldArray[offset + 1] = particle.position.y + Math.cos(time * 0.16 + particle.phase) * 0.026;
      fieldArray[offset + 2] = particle.position.z;
    });

    wordAttr.needsUpdate = true;
    fieldAttr.needsUpdate = true;
  });

  return (
    <group ref={groupRef} position={[0, 0.42, 0]}>
      <points>
        <bufferGeometry>
          <bufferAttribute
            ref={wordPositionsRef}
            attach="attributes-position"
            args={[wordPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          ref={wordMaterialRef}
          color="#f8f6ef"
          size={0.017}
          sizeAttenuation
          transparent
          opacity={0.18}
          depthWrite={false}
        />
      </points>
      <points>
        <bufferGeometry>
          <bufferAttribute
            ref={fieldPositionsRef}
            attach="attributes-position"
            args={[fieldPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          ref={fieldMaterialRef}
          color="#dff8ff"
          size={0.014}
          sizeAttenuation
          transparent
          opacity={0.16}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

function PortalScene({ entering }: { entering: boolean }) {
  return (
    <Canvas
      dpr={[1, 1.5]}
      camera={{ position: [0, 0.28, 5.2], fov: 38 }}
      gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
    >
      <color attach="background" args={["#020202"]} />
      <ambientLight intensity={0.30} />
      <directionalLight position={[2.5, 2.8, 4]} intensity={0.58} color="#fff9ef" />
      <LogoPortalParticles entering={entering} />
    </Canvas>
  );
}

export default function Home() {
  const router = useRouter();
  const [entering, setEntering] = useState(false);

  useEffect(() => {
    if (!entering) return;

    const id = window.setTimeout(() => {
      router.push("/experience");
    }, 860);

    return () => window.clearTimeout(id);
  }, [entering, router]);

  return (
    <main
      className="aura-app-shell relative isolate bg-black text-white"
      style={{
        overscrollBehavior: "none",
        touchAction: "manipulation",
        userSelect: "none",
        WebkitUserSelect: "none",
        WebkitTouchCallout: "none",
      }}
    >
      <div
        className={`absolute inset-0 transition-opacity duration-700 ${
          entering ? "opacity-60" : "opacity-100"
        }`}
      >
        <PortalScene entering={entering} />
      </div>

      <section
        className={`pointer-events-none absolute inset-x-0 bottom-[max(3.2rem,env(safe-area-inset-bottom))] z-10 flex flex-col items-center px-6 text-center transition-all duration-700 ${
          entering ? "translate-y-3 opacity-0" : "translate-y-0 opacity-100"
        }`}
      >
        <h1 className="sr-only">AURA ONE</h1>
        <p className="mb-8 text-[0.62rem] uppercase tracking-[0.48em] text-white/46">
          Spatial Hospitality Experience
        </p>
        <button
          type="button"
          onClick={() => setEntering(true)}
          className="pointer-events-auto border border-white/18 bg-white/9 px-6 py-3 text-[0.68rem] uppercase tracking-[0.34em] text-white/78 shadow-[0_20px_80px_rgba(0,0,0,0.32)] backdrop-blur-xl transition duration-300 hover:border-white/30 hover:bg-white/14 hover:text-white focus:outline-none focus:ring-1 focus:ring-white/35"
          style={{ borderRadius: 999 }}
          aria-label="Enter AURA ONE experience"
        >
          Enter Experience
        </button>
      </section>

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.05),transparent_40%),linear-gradient(to_bottom,rgba(0,0,0,0.08),rgba(0,0,0,0.60))]" />
    </main>
  );
}
