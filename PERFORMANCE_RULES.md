# AURA ONE - PERFORMANCE RULES

## Targets

Desktop:
- 60 FPS target

Mobile:
- 30 FPS minimum

## Rules

- Avoid unnecessary React re-renders.
- Do not store high-frequency animation values in React state.
- Use refs, animation loops, or Three.js systems for high-frequency motion.
- Reuse geometry and materials.
- Avoid excessive particle count during MVP.
- Avoid heavy CPU particle simulation.
- Avoid uncontrolled requestAnimationFrame loops.
- Avoid memory leaks from camera, animation, or Three.js resources.

## Priority

Performance > visual complexity.
Stability > effects.
