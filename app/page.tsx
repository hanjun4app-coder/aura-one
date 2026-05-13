import SpatialScene from "./components/SpatialScene";

export default function Home() {
  return (
    <main className="relative w-screen h-screen overflow-hidden bg-black text-white">
      <SpatialScene />

      <div className="pointer-events-none absolute bottom-4 left-4 flex max-w-[18rem] flex-col items-start text-left md:left-6">
        <p className="mb-2 text-[0.62rem] tracking-[0.42em] text-cyan-200/55">
          SPATIAL AI SYSTEM
        </p>

        <p className="text-[0.68rem] tracking-[0.16em] text-white/38 md:text-xs">
          Gesture-Controlled Spatial Product Experience
        </p>
      </div>
    </main>
  );
}
