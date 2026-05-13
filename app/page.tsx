import SpatialScene from "./components/SpatialScene";

export default function Home() {
  return (
    <main className="relative w-screen h-screen overflow-hidden bg-black text-white">
      <SpatialScene />

      <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-start pt-8">
        <p className="mb-3 text-xs tracking-[0.5em] text-cyan-200/70">
          SPATIAL AI SYSTEM
        </p>

        <p className="text-xs md:text-sm text-white/50 tracking-wide">
          Gesture-Controlled Spatial Product Experience
        </p>
      </div>
    </main>
  );
}
