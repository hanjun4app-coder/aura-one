# AURA ONE — Spatial Restaurant Ordering MVP

## Product Direction

AURA ONE becomes a gesture-driven spatial ordering experience for restaurants.

The goal is NOT a traditional kiosk.
The goal is:

* cinematic spatial food browsing
* touchless gesture interaction
* premium Apple/Vision Pro-like food presentation
* self-service ordering experience
* modular food inspection
* memorable visual experience for customers

---

# EXPERIENCE FLOW

## Phase 1 — Logo Intro Sequence

### Initial State

Screen is dark.

Restaurant logo appears in center:

* glowing softly
* premium cinematic lighting
* floating particles around logo
* calm ambient motion

Example:

* McDonald's
* Local burger brand
* Coffee shop logo
* AURA ONE demo brand

---

## Phase 2 — Molecular Particle Transition

### Logo Breakdown

Logo dissolves into:

* floating particles
* cubes
* food-color fragments
* molecular-style motion

Particles:

* spiral outward
* rotate in 3D space
* scatter softly
* then reorganize

Transition target:

The particles reform into:

# Spatial Menu Carousel

---

# SPATIAL MENU SYSTEM

## Menu Arrangement

Food combos arranged in a circular spatial carousel.

Example:

* Burger Combo
* Chicken Combo
* Fries Set
* Coffee Set
* Dessert Combo

---

## Carousel Composition

### Front Item

The front item:

* largest
* lowest vertical position
* closest to user
* brightest lighting
* strongest detail
* active selectable item

### Rear Items

Items behind:

* gradually higher vertically
* smaller scale
* darker lighting
* softer focus
* positioned in circular depth

This creates:

* depth layering
* premium spatial feeling
* Vision Pro style presentation

---

# MOTION LANGUAGE

## Idle Motion

Entire carousel:

* rotates slowly
* calm cinematic movement
* smooth easing
* no harsh motion

Food items:

* subtle floating
* slow self rotation
* gentle lighting reflections

---

# GESTURE INTERACTION

## Swipe Left / Right

Gesture:

* hand swipe left
* hand swipe right

Behavior:

* rotate menu carousel
* next food combo moves forward
* active combo becomes focused

---

## Select Gesture

Possible future gestures:

* pinch
* hold
* push forward

Current MVP:

* Enter key
* UI button
* later replace with hand gesture

Behavior:

# Enter Inspect Mode

---

# INSPECT MODE

## Food Focus Stage

Selected combo moves:

* to center screen
* larger scale
* spotlight lighting
* background dims
* other menu items retreat backward

The food slowly rotates in place.

---

# FOOD EXPLODED VIEW

## Burger Example

Burger separates into layers:

* top bun
* lettuce
* tomato
* cheese
* beef patty
* sauce
* bottom bun

Each ingredient:

* floats separately
* evenly spaced vertically
* slowly rotates
* receives soft lighting

---

# INGREDIENT HUD

When ingredient selected:

HUD shows:

* ingredient name
* description
* calories
* allergen info
* flavor notes

Example:

## Wagyu Beef Patty

* flame grilled
* premium wagyu blend
* rich savory flavor

---

# VISUAL DESIGN LANGUAGE

## Style Direction

Inspired by:

* Apple Vision Pro
* premium restaurant branding
* luxury food commercials
* spatial computing UI

---

## Materials

Food:

* realistic materials
* glossy sauces
* crispy textures
* soft bread shading
* believable lighting

UI:

* translucent glass panels
* subtle blur
* restrained highlights
* minimal cyan/white accents

---

# MVP TECHNICAL ARCHITECTURE

## Preserve Existing Systems

Keep:

* Gesture Action Layer
* Camera gesture detection
* Carousel system
* Inspect mode
* Focus lighting
* Voxel particle system
* State machine
* Runtime architecture

---

# Replace Current Surgical Product

Replace:

* surgical robot product system

With:

# Food Product System

---

# FOOD MODULE SYSTEM

Example structure:

```ts
const foodItems = {
  burgerCombo: {
    name: "Signature Burger Combo",
    parts: [
      "Top Bun",
      "Lettuce",
      "Cheese",
      "Patty",
      "Sauce",
      "Bottom Bun"
    ]
  }
}
```

---

# CAMERA / GESTURE MVP

## Current Gestures

### Swipe Left

* previous menu item

### Swipe Right

* next menu item

---

# Future Gestures

## Pinch

* select item

## Open Palm

* return to carousel

## Rotate Hand

* rotate food manually

---

# SPATIAL DEPTH RULES

## Front Menu Item

* lower Y
* larger scale
* stronger brightness
* closer camera depth

## Rear Menu Items

* progressively higher Y
* progressively smaller scale
* darker lighting
* deeper Z placement

This creates:

* spatial restaurant stage feeling
* cinematic menu presentation

---

# AUDIO DIRECTION (Future)

## Ambient Audio

* soft restaurant ambience
* subtle UI sound design
* food interaction sounds
* premium transitions

---

# COMMERCIAL MVP GOAL

The system should feel like:

"The future of restaurant ordering."

NOT:

* a kiosk UI
* a website menu
* a touch screen

Instead:

# Spatial Food Interaction Experience

---

# MVP PRIORITIES

## Priority 1

* logo molecular transition
* spatial carousel
* gesture rotation
* inspect mode

## Priority 2

* exploded burger ingredients
* ingredient HUD
* improved food realism

## Priority 3

* ordering cart
* payment flow
* restaurant backend

---

# CURRENT RECOMMENDATION

Build:

1. Logo particle transition
2. Spatial food carousel
3. Burger inspect mode
4. Exploded ingredient system
5. Ingredient HUD
6. Gesture rotation refinement

Before:

* checkout system
* payment system
* restaurant backend
* ordering APIs
