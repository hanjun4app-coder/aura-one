# AURA ONE - ARCHITECTURE

## Core Flow

Camera Input
→ Hand Tracking
→ Gesture Engine
→ Interaction State Machine
→ Scene Controller
→ 3D Product System
→ Particle System
→ HUD/UI

## Core Modules

- Gesture System: camera, hand tracking, gesture detection
- State System: interaction state, scene state, product state
- Scene System: Three.js scene, camera, lights, environment
- Product System: 3D product carousel, focus, open/close
- Particle System: ambient particles, attraction, collapse, burst
- HUD System: visual feedback for current state

## Architecture Rules

- Gesture system must not directly mutate Three.js objects.
- Scene system must respond to approved state changes.
- Particle system must remain isolated from UI state.
- UI/HUD must read state, not own core interaction logic.
- All major interaction transitions must pass through the state machine.
