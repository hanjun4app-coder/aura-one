import SpatialScene from "../components/SpatialScene";

// Demo experience route. iPad/Safari-aware sizing:
//   - `h-[100dvh] w-[100dvw]` uses the dynamic viewport so the bottom HUD
//     isn't hidden under Safari's collapsing address bar.
//   - `touchAction: "none"` prevents pinch-zoom and pan inside the demo
//     surface — gestures stay scoped to hand tracking + keyboard.
//   - `userSelect: "none"` keeps long-press from selecting overlay text.
// No legacy developer overlay is rendered here — the in-scene UI provides
// the AURA ONE wordmark and customer-facing copy.
export default function ExperiencePage() {
  return (
    <main
      className="aura-app-shell relative bg-black text-white"
      style={{
        overscrollBehavior: "none",
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
