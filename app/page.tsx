"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const PORTAL_PARTICLE_COUNT = 1800;

type PortalParticle = {
  base: THREE.Vector3;
  scatter: THREE.Vector3;
  scale: number;
  phase: number;
};

function seededNoise(seed: number) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function createPortalParticles() {
  const particles: PortalParticle[] = [];

  for (let i = 0; i < PORTAL_PARTICLE_COUNT; i += 1) {
    const t = i / PORTAL_PARTICLE_COUNT;
    const ringBand = seededNoise(i + 3);
    const angle = t * Math.PI * 2 + (seededNoise(i + 11) - 0.5) * 0.045;
    const radius = 1.58 + (ringBand - 0.5) * 0.34;
    const halo = seededNoise(i + 37) > 0.78;
    const haloLift = halo ? (seededNoise(i + 43) - 0.5) * 0.62 : 0;
    const x = Math.cos(angle) * radius * (halo ? 1.18 : 1);
    const y = Math.sin(angle) * radius * 0.66 + haloLift;
    const z = (seededNoise(i + 19) - 0.5) * (halo ? 0.82 : 0.30);
    const base = new THREE.Vector3(x, y, z);
    const scatterPower = 1.45 + seededNoise(i + 71) * 0.85;
    const scatter = base
      .clone()
      .multiplyScalar(scatterPower)
      .add(
        new THREE.Vector3(
          (seededNoise(i + 83) - 0.5) * 0.9,
          (seededNoise(i + 97) - 0.5) * 0.54,
          -0.35 - seededNoise(i + 109) * 0.95
        )
      );

    particles.push({
      base,
      scatter,
      scale: halo ? 0.014 : 0.019 + seededNoise(i + 131) * 0.010,
      phase: seededNoise(i + 151) * Math.PI * 2,
    });
  }

  return particles;
}

function LogoPortalParticles({ entering }: { entering: boolean }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);
  const groupRef = useRef<THREE.Group>(null);
  const particles = useMemo(() => createPortalParticles(), []);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const enterProgressRef = useRef(0);

  useFrame(({ clock }, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const target = entering ? 1 : 0;
    enterProgressRef.current = THREE.MathUtils.lerp(
      enterProgressRef.current,
      target,
      1 - Math.exp(-delta * 4.2)
    );

    const time = clock.getElapsedTime();
    const enterP = enterProgressRef.current;
    const breathe = 1 + Math.sin(time * 0.34) * 0.012;

    if (groupRef.current) {
      groupRef.current.rotation.y = time * 0.085;
      groupRef.current.rotation.z = Math.sin(time * 0.18) * 0.025;
      groupRef.current.scale.setScalar(breathe + enterP * 0.12);
    }

    if (materialRef.current) {
      materialRef.current.opacity = THREE.MathUtils.lerp(0.82, 0.10, enterP);
    }

    particles.forEach((particle, index) => {
      const float = Math.sin(time * 0.62 + particle.phase) * 0.010;
      dummy.position
        .copy(particle.base)
        .lerp(particle.scatter, enterP)
        .addScalar(0);
      dummy.position.y += float;
      dummy.rotation.set(time * 0.05, time * 0.06 + particle.phase, 0);
      dummy.scale.setScalar(particle.scale * (1 + enterP * 0.35));
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <group ref={groupRef} position={[0, 0.42, 0]}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, PORTAL_PARTICLE_COUNT]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          ref={materialRef}
          color="#f7f5ef"
          emissive="#dff9ff"
          emissiveIntensity={0.06}
          metalness={0.18}
          roughness={0.72}
          transparent
          opacity={0.82}
          depthWrite={false}
        />
      </instancedMesh>
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
      <ambientLight intensity={0.34} />
      <directionalLight position={[2.5, 2.8, 4]} intensity={1.18} color="#fff9ef" />
      <pointLight position={[0, 0.8, 2.2]} intensity={0.48} color="#d8fbff" />
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
    }, 760);

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
          entering ? "opacity-55" : "opacity-100"
        }`}
      >
        <PortalScene entering={entering} />
      </div>

      <section
        className={`pointer-events-none absolute inset-x-0 bottom-[max(3.2rem,env(safe-area-inset-bottom))] z-10 flex flex-col items-center px-6 text-center transition-all duration-700 ${
          entering ? "translate-y-3 opacity-0" : "translate-y-0 opacity-100"
        }`}
      >
        <p className="mb-4 text-[0.62rem] uppercase tracking-[0.48em] text-white/42">
          Spatial Hospitality Experience
        </p>
        <h1 className="text-4xl font-light tracking-[0.38em] text-white/92 sm:text-5xl">
          AURA ONE
        </h1>
        <button
          type="button"
          onClick={() => setEntering(true)}
          className="pointer-events-auto mt-9 border border-white/18 bg-white/9 px-6 py-3 text-[0.68rem] uppercase tracking-[0.34em] text-white/78 shadow-[0_20px_80px_rgba(0,0,0,0.32)] backdrop-blur-xl transition duration-300 hover:border-white/30 hover:bg-white/14 hover:text-white focus:outline-none focus:ring-1 focus:ring-white/35"
          style={{ borderRadius: 999 }}
          aria-label="Enter AURA ONE experience"
        >
          Enter Experience
        </button>
      </section>

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.055),transparent_38%),linear-gradient(to_bottom,rgba(0,0,0,0.1),rgba(0,0,0,0.56))]" />
    </main>
  );
}
