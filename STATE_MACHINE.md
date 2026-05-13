# AURA ONE - STATE MACHINE

## Official States

IDLE
→ HAND_DETECTED
→ TRACKING
→ PRODUCT_HOVER
→ PRODUCT_OPEN
→ COLLAPSE
→ HOLD
→ RELEASE
→ RECONSTRUCT
→ RETURN_IDLE

## Rules

- No hidden state mutation.
- No random global state for interaction logic.
- No state transition may bypass the approved flow without approval.
- Gesture recognition produces intent.
- State machine decides system state.
- Scene and particles respond to state.
