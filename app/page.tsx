import SpatialScene from "./components/SpatialScene";

export default function Home() {
  return (
    <main className="relative w-screen h-screen overflow-hidden bg-black text-white">
      <SpatialScene />

      <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center">
        <p className="text-xs tracking-[0.5em] text-cyan-200/70 mb-4">
          SPATIAL AI SYSTEM
        </p>

        <h1 className="text-5xl md:text-7xl font-light tracking-[0.35em]">
          AURA ONE
        </h1>

        <p className="mt-6 text-sm md:text-base text-white/60 tracking-wide">
          Gesture-Controlled Spatial Product Experience
        </p>
      </div>
    </main>
  );
}