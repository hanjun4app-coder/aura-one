# AURA ONE Spatial Behavior Rules

## Purpose

AURA ONE is a premium cinematic spatial hospitality experience. Its behavior should feel composed, intentional, and quietly alive. Every interaction should preserve calmness, premium pacing, spatial presence, and Apple/Vision-Pro-inspired restraint.

This document guides future development and AI-assisted changes. It defines behavioral direction, not implementation architecture.

## 1. Spatial Philosophy

AURA ONE should feel like a product existing in a calm spatial environment, not a webpage, game, or dashboard.

- Ambient presence should be subtle and continuous, never attention-seeking.
- Interaction should feel calm, controlled, and deliberate.
- Spatial breathing is encouraged: small changes in depth, opacity, light, or position may make the system feel alive.
- Motion should remain restrained, with smooth starts and soft landings.
- Premium pacing is more important than speed. The experience should give the eye time to understand what changed.
- The system should guide attention with presence and composition, not effects.

## 2. Idle Behavior Rules

Idle behavior should make the scene feel alive without becoming a performance.

- Use subtle motion only.
- Avoid aggressive automatic movement.
- Avoid arcade-style attract modes.
- Avoid constant product cycling unless explicitly requested for a demo loop.
- Idle movement should be slow enough that the user can ignore it.
- Floating, breathing, and slight rotation are acceptable when they preserve calmness.
- If the scene is already visually rich, silence is preferable to adding more movement.

## 3. Camera Behavior Rules

The camera should feel stable, observant, and cinematic.

- Use minimal breathing motion only when needed.
- Avoid dramatic camera travel.
- Avoid rapid zooms, whip pans, shake, bounce, or orbiting camera effects.
- Camera movement should never feel like a game camera.
- Depth should be created through object placement, lighting, and scale before camera motion.
- If a product enters Inspect mode, prefer smooth object movement over moving the camera aggressively.

## 4. Lighting Behavior Rules

Lighting should support warmth, material quality, and focus.

- Use warm cinematic lighting only.
- Subtle intensity drift is allowed when it feels natural and quiet.
- Transitions should be restrained and eased.
- Avoid neon, cyberpunk, saturated glow spam, strobing, or high-contrast spectacle.
- Highlighting should feel like a premium showroom cue, not a game selection effect.
- Lighting should clarify the product and guide attention without overwhelming the scene.

## 5. Motion Language Rules

Motion is part of the product language. It should communicate confidence.

- Use luxury pacing: smooth, measured, and unhurried.
- Prefer eased motion over linear motion.
- Intentional pauses are acceptable when they help the user understand hierarchy.
- Movement should guide attention, not demand attention.
- Avoid chaotic, springy, rubbery, or toy-like behavior.
- Avoid excessive simultaneous motion.
- The best motion often feels almost inevitable: calm acceleration, clear arrival, soft settling.

## 6. Silence Rules

Visual silence is a feature.

- The system should remain visually calm when the user is reading, inspecting, listening, or deciding.
- Do not fill every mode with continuous movement.
- Preserve negative space around the active product.
- Avoid competing animations near product information, ingredient details, or Ask AURA.
- If a new effect does not improve understanding, it should not be added.
- A stable scene can feel more premium than an animated one.

## 7. Performance Philosophy

Performance is part of the spatial experience.

- iPad Safari is a priority runtime.
- Treat GPU, CPU, memory, camera, and video decode budgets as limited.
- Prefer subtle CSS or lightweight animation over heavy runtime systems.
- Avoid adding new render loops unless they are essential.
- Avoid physics engines, heavy postprocessing, and high-frequency state updates unless explicitly approved.
- Hidden or secondary elements should not keep doing expensive work.
- Smoothness and thermal stability matter more than decorative complexity.

## 8. AI Collaboration Rules

Future AI assistants must preserve spatial restraint.

- Do not add unnecessary motion, glow, particles, camera movement, or tutorial UI.
- Do not make AURA ONE feel like a game, kiosk toy, SaaS dashboard, or consumer onboarding flow.
- Prefer minimal safe changes over broad refactors.
- Tune existing values before introducing new systems.
- Keep interaction behavior aligned with premium hospitality: calm, intentional, cinematic, and respectful of attention.
- When uncertain, choose less motion, fewer effects, and more breathing room.
- Preserve the established product identity unless the user explicitly asks for a new direction.
