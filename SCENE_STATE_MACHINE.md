# AURA ONE Scene State Machine

## Purpose

This document defines how AURA ONE spatial states should coexist safely as the product enters the Spatial Experience Direction phase. It is intended for Claude, Codex, GPT, and future AI-assisted handoff work.

AURA ONE should remain calm, premium, and predictable. Any stateful behavior must have clear priority, allowed transitions, blocked transitions, and cleanup rules.

## 1. Core States

### Menu Closed / Intro

The opening system state before the spatial menu is visible.

- The intro may show brand presence and system startup motion.
- The menu should become accessible without requiring camera permission.
- User input may skip or complete the intro.
- No product-specific interaction should run in this state.

### Carousel Idle

The menu is open and products are arranged in the spatial carousel, but the user is not actively navigating.

- Ambient breathing may run at low intensity.
- Products may maintain minimal calm presence.
- No automated behavior should compete with the active product.

### Carousel Browsing

The user is actively moving between products through keyboard, touch, gesture, or future input.

- Navigation should cancel or reduce idle automation.
- Product transitions should remain smooth and readable.
- Active index must remain within visible menu bounds.

### Inspect Mode

The active product is brought forward for focused inspection.

- Inspect motion has priority over ambient behavior.
- Manual rotation may be active.
- Non-active products should remain secondary.
- Ask AURA is only eligible if the inspected product is Burger.

### Burger Reveal Mode

Burger Inspect Mode with ingredient layers revealed.

- Only Burger may enter reveal mode.
- Reveal should not trigger for Po' Boy, Dessert, Fried Chicken, Crawfish, or inactive products.
- Reveal state should clear when leaving Inspect unless an explicit future product decision says otherwise.

### Touch Drag Active

The user is manually rotating the inspected product by touch or pointer drag.

- Drag has top priority.
- Ambient camera and light breathing should pause or reduce.
- Demo automation must not override active drag.
- Vertical touch movement should not introduce unintended rotation axes.

### Gesture Active

The camera gesture layer has recognized or is processing a meaningful user gesture.

- Gesture input should route through the same semantic action layer as keyboard/touch controls.
- Gesture actions should cancel or reduce automation.
- Gesture recognition should not bypass carousel, inspect, or reveal state rules.

### Ask AURA Listening

The Burger Inspect voice system is actively listening.

- Available only in Burger Inspect Mode.
- It must not appear or remain active for non-Burger products.
- Leaving Inspect should stop recognition.
- Listening should not modify carousel, inspect, or reveal state.

### Ask AURA Speaking

The system is speaking a predefined response.

- Available only after Ask AURA has been invoked.
- Speech should be cancellable.
- Leaving Inspect should cancel speech if needed.
- Speech should not trigger product navigation or reveal behavior.

### Ambient Breathing Active

Subtle camera and lighting breathing may run when the scene is idle or low-interaction.

- Breathing is the lowest-priority visual state.
- It must not fight inspect transitions, drag, gestures, or speech focus.
- It should remain almost imperceptible.

### Future Demo Mode

Future automatic demo sequencing, if added, should be treated as automation.

- Demo mode must be interruptible by any user input.
- Demo mode should never override touch drag, active gesture, Ask AURA, or inspect/reveal interaction.
- Demo timers must be cancellable.

## 2. State Priority

Priority from highest to lowest:

1. User touch / drag
2. Explicit user gesture
3. Inspect / reveal interaction
4. Ask AURA speech / listening
5. Ambient breathing
6. Future demo automation

When two states conflict, the higher-priority state wins. Lower-priority systems should pause, reduce intensity, or cancel.

## 3. Allowed Transitions

Common allowed transitions:

- Menu closed / intro -> carousel idle
- Carousel idle -> carousel browsing
- Carousel browsing -> carousel idle
- Carousel browsing -> inspect mode
- Inspect mode -> carousel idle
- Inspect Burger -> burger reveal mode
- Burger reveal mode -> Burger Inspect assembled state
- Inspect Burger -> Ask AURA listening
- Ask AURA listening -> Ask AURA speaking
- Ask AURA speaking -> Burger Inspect
- Carousel idle -> future demo mode
- Future demo mode -> carousel browsing on user input
- Any user input -> cancel or reduce automation

Transitions should be smooth, cancellable where practical, and should not leave stale timers or hidden state behind.

## 4. Blocked Transitions

Blocked or unsafe transitions:

- Future demo mode must not override active drag.
- Future demo mode must not force navigation while Ask AURA is listening or speaking.
- Ambient breathing must not fight inspect transitions.
- Ambient breathing must not create visible camera sickness or product drift.
- Ask AURA must not appear outside Burger Inspect.
- Ask AURA must not remain active after leaving Inspect.
- Burger reveal must not trigger on non-Burger products.
- Burger reveal must not persist unexpectedly after exiting Inspect.
- Gesture actions must not bypass the shared semantic action layer.
- Hidden or inactive products must not keep running expensive stateful behavior.

## 5. Cancellation Rules

User input always wins.

- Touch drag pauses or reduces ambient breathing.
- Explicit gesture input cancels or reduces future automation.
- Keyboard input cancels or reduces future automation.
- Inspect transition suppresses idle automation.
- Carousel navigation cancels pending inspect entry if needed.
- Leaving Inspect cancels Burger reveal if needed.
- Leaving Burger Inspect cancels Ask AURA recognition and speech if needed.
- Reset / assemble actions should clear inspect, reveal, pending timers, and manual rotation state as appropriate.

Cancellation should be quiet and should not create visual snapping.

## 6. Timer / Cleanup Rules

All timers and long-running loops must be owned and cancellable.

- Every timeout must have a clear cancellation path.
- Every interval must be cleared on unmount.
- requestAnimationFrame loops must stop when their owning feature stops.
- Camera streams must stop tracks when disabled or on failure.
- Speech recognition must stop when leaving its eligible state.
- Speech synthesis should cancel when leaving Inspect or unmounting.
- No animation state should survive mode exit unless explicitly documented.
- No hidden global state should be introduced.

Cleanup is part of the interaction design. A calm system should not accumulate invisible work.

## 7. AI Collaboration Rules

Future AI assistants must check this document before adding stateful behavior.

- Do not add new state without documenting its priority.
- Do not add new state without documenting cancellation rules.
- Do not introduce hidden timers, intervals, RAF loops, camera loops, or speech loops without cleanup.
- Route new input types through existing semantic action patterns when possible.
- Prefer reducing lower-priority systems instead of layering more behavior on top.
- Avoid game-like state machines, aggressive automation, and surprise transitions.
- Preserve Burger-only reveal and Burger-only Ask AURA boundaries unless the user explicitly changes product direction.
- When uncertain, choose fewer states, calmer transitions, and more explicit cleanup.
