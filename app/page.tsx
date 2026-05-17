import SpatialScene from "./components/SpatialScene";

// Demo root. iPad/Safari-aware sizing:
//   - `h-[100dvh] w-[100dvw]` uses the dynamic viewport so the bottom HUD
//     isn't hidden under Safari's collapsing address bar.
//   - `touchAction: "none"` prevents pinch-zoom and pan inside the demo
//     surface — gestures stay scoped to hand tracking + keyboard.
//   - `userSelect: "none"` keeps long-press from selecting overlay text.
// No legacy developer overlay is rendered here — the in-scene UI provides
// the AURA ONE wordmark and customer-facing copy.
export default function Home() {
  return (
    <main
      className="relative h-[100dvh] w-[100dvw] overflow-hidden bg-black text-white"
      style={{
        touchAction: "none",
        userSelect: "none",
        WebkitUserSelect: "none",
        WebkitTouchCallout: "none",
      }}
    >
      <SpatialScene />
    </main>
  );
}
