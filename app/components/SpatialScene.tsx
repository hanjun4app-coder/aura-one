"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, useProgress } from "@react-three/drei";
import {
  Suspense,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import type {
  HandLandmarker as HandLandmarkerInstance,
  HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

const CAROUSEL_PARTS = [
  {
    name: "Signature Burger Combo",
    description: "Double wagyu patty, truffle mayo, aged cheddar, brioche bun.",
  },
  {
    name: "Louisiana Po' Boy",
    description: "Toasted roll with crisp seafood, remoulade, lettuce, and tomato.",
  },
  {
    name: "Premium Dessert",
    description: "Elegant layered sweet finish, served chilled with seasonal accents.",
  },
  {
    name: "Crispy Fried Chicken",
    description: "Buttermilk-brined chicken, house spice crust, honey-thyme drizzle.",
  },
  {
    name: "Louisiana Crawfish",
    description: "Spiced shellfish with bold coastal flavor, dusted with cajun seasoning.",
  },
] as const;

const BURGER_INGREDIENTS = [
  { name: "Bottom Bun", cal: "160 kcal", allergen: "Gluten, Sesame", flavor: "Toasted brioche base, golden butter crust" },
  { name: "Wagyu Patty", cal: "340 kcal", allergen: "None", flavor: "Flame-grilled premium wagyu, rich umami char" },
  { name: "Aged Cheddar", cal: "110 kcal", allergen: "Dairy", flavor: "Two-year reserve, sharp and creamy" },
  { name: "Tomato", cal: "10 kcal", allergen: "None", flavor: "Vine-ripened beefsteak, sweet and bright" },
  { name: "Crisp Lettuce", cal: "5 kcal", allergen: "None", flavor: "Iceberg leaf, cool and fresh" },
  { name: "Top Bun", cal: "180 kcal", allergen: "Gluten, Sesame", flavor: "Toasted brioche dome with sesame seeds" },
] as const;

const ITEM_PRICES = [18.90, 16.90, 12.80, 14.50, 24.00];

const FOOD_INSPECT_DATA = [
  {
    special: "TODAY'S SPECIAL",
    calories: "780 kcal",
    protein: "38g",
    allergens: "Wheat, dairy, egg",
    flavorProfile: "Savory, creamy, smoky, lightly sweet",
    ingredients: "Brioche bun · wagyu beef patty · aged cheddar · crisp lettuce · tomato · truffle sauce",
    chefNote: "Signature layered burger with warm brioche and rich truffle sauce.",
  },
  {
    special: null,
    calories: "640 kcal",
    protein: "31g",
    allergens: "Wheat, egg, shellfish",
    flavorProfile: "Crisp, creamy, briny, lightly spicy",
    ingredients: "French roll · fried seafood · remoulade · lettuce · tomato · pickles",
    chefNote: "A coastal Louisiana sandwich with a crisp bite and cool remoulade finish.",
  },
  {
    special: null,
    calories: "480 kcal",
    protein: "6g",
    allergens: "Dairy, egg, gluten",
    flavorProfile: "Creamy, delicate, layered, lightly floral",
    ingredients: "Vanilla cream · seasonal fruit · pastry layers · honeyed praline",
    chefNote: "An elegant layered finish — silky cream against crisp, hand-folded pastry.",
  },
  {
    special: null,
    calories: "560 kcal",
    protein: "32g",
    allergens: "Wheat, dairy, egg",
    flavorProfile: "Crispy, savory, herbal, lightly sweet",
    ingredients: "Buttermilk-brined chicken · seasoned flour · fresh herbs · honey-thyme glaze",
    chefNote: "Brined overnight and double-coated in our house spice flour for an audible crust.",
  },
  {
    special: null,
    calories: "320 kcal",
    protein: "28g",
    allergens: "Shellfish",
    flavorProfile: "Spicy, savory, briny, aromatic",
    ingredients: "Live crawfish · cajun spice · garlic · lemon · butter · fresh herbs",
    chefNote: "Boiled in a coastal Louisiana spice broth and finished with garlic-herb butter.",
  },
] as const;

const INACTIVE_MENU_ITEMS = [
  {
    name: "Premium Steak",
    description: "Seared cut with rich savory finish, finished in bone-marrow butter.",
    price: 26.50,
    modelPath: "/models/steak.glb",
    inspectData: {
      special: null,
      calories: "620 kcal",
      protein: "48g",
      allergens: "None",
      flavorProfile: "Savory, smoky, juicy, robust",
      ingredients: "Prime ribeye · sea salt · cracked pepper · bone-marrow butter · rosemary",
      chefNote: "Dry-aged 28 days, seared in cast iron and finished with bone-marrow butter.",
    },
  },
  {
    name: "Fresh Oyster Selection",
    description: "Chilled ocean delicacy, served on cracked ice with citrus mignonette.",
    price: 22.00,
    modelPath: "/models/oyster.glb",
    inspectData: {
      special: null,
      calories: "120 kcal",
      protein: "14g",
      allergens: "Shellfish",
      flavorProfile: "Briny, delicate, ocean-fresh, silky texture",
      ingredients: "Fresh oysters · cracked ice · lemon · shallot mignonette · sea salt",
      chefNote: "Hand-selected daily — served chilled on ice with a citrus mignonette.",
    },
  },
] as const;

if (
  process.env.NODE_ENV === "development" &&
  (CAROUSEL_PARTS.length !== ITEM_PRICES.length ||
    CAROUSEL_PARTS.length !== FOOD_INSPECT_DATA.length)
) {
  console.warn(
    "[AURA CONFIG] Menu config length mismatch",
    {
      carouselParts: CAROUSEL_PARTS.length,
      foodInspectData: FOOD_INSPECT_DATA.length,
      itemPrices: ITEM_PRICES.length,
      inactiveMenuItems: INACTIVE_MENU_ITEMS.length,
    }
  );
}

const INSPECT_ROTATION_STEP = 0.32;
const INSPECT_DRAG_ROTATION_SPEED = 0.006;
const MEDIAPIPE_WASM_PATH = "/mediapipe/wasm";
// Localized for kiosk/demo reliability; avoids runtime Google Storage model fetches.
const HAND_LANDMARKER_MODEL_URL =
  "/mediapipe/models/hand_landmarker.task";

// Swipe detection tuning — all distances are normalized palm-X (0.0–1.0).
const SWIPE_WINDOW_MS = 350;              // rolling sample buffer
const SWIPE_COOLDOWN_MS = 600;            // hard block after any swipe fires
const OPPOSITE_LOCK_MS = 900;             // opposite direction requires elevated thresholds within this window
const MIN_SWIPE_DISTANCE = 0.17;          // minimum palm-X displacement for a same/unlocked direction
const MIN_SWIPE_VELOCITY = 0.00042;       // palm-X per ms — rejects slow drift
const OPPOSITE_DISTANCE_MULT = 1.55;      // extra distance required for opposite swipe within OPPOSITE_LOCK_MS
const OPPOSITE_VELOCITY_MULT = 1.45;      // extra velocity required for opposite swipe within OPPOSITE_LOCK_MS

// Inspect gesture — palm width growing in frame (open hand approaching camera).
// Palm width uses landmark 5 (index base) → 17 (pinky base): stable across finger poses.
const INSPECT_COOLDOWN_MS = 2500;
const INSPECT_GROW_THRESHOLD = 0.048;   // palm-width growth to confirm approaching hand
const INSPECT_GUIDANCE_THRESHOLD = 0.018; // lower threshold for "OPEN HAND INSPECT" hint
const INSPECT_SAMPLE_WINDOW_MS = 600;   // wider window for a more relaxed trigger

// Double-fist pulse gesture — two distinct fist "pulses" (open→fist transitions)
// within DOUBLE_FIST_WINDOW_MS confirm add-to-order. Thumb excluded from fist score
// because its resting position varies across fist styles.
// Pulse = rising edge only: holding a fist counts as one pulse, never more.
// Hysteresis prevents threshold-boundary flickering: enter requires tighter close,
// exit requires a clearly open hand. Separate enter/exit thresholds are tracked via
// isFistLatchRef so the rising-edge detector sees clean, stable transitions.
const FIST_ENTER_THRESHOLD = 1.1;        // normalised avg tip distance — must dip below to enter fist
const FIST_EXIT_THRESHOLD = 1.32;        // must rise above to leave fist (hysteresis band)
const DOUBLE_FIST_WINDOW_MS = 1800;      // second fist must arrive within this window
const FIST_ADD_COOLDOWN_MS = 3000;       // post-fire lockout before gesture can re-arm
const FIST_AFTER_SWIPE_IGNORE_MS = 1200; // suppress if a swipe fired recently
const MIN_OPEN_FRAMES = 3;               // min consecutive non-fist frames between pulses

// Exit-inspect gesture — open hand shrinking = hand retreating from camera ("push away").
// Uses same palm-width metric as enter so they are symmetrically detectable.
// Mutual exclusion: enter only fires when NOT in inspect; exit only fires when IN inspect.
// Cross-bumping cooldowns prevent immediate re-entry after exit and vice versa.
const EXIT_INSPECT_SHRINK_THRESHOLD = 0.045;   // palm-width shrink required to confirm exit
const EXIT_INSPECT_GUIDANCE_THRESHOLD = 0.016; // lower threshold for "MOVE HAND BACK" hint
const EXIT_INSPECT_SAMPLE_WINDOW_MS = 650;      // rolling window (slightly wider than enter)
const EXIT_INSPECT_COOLDOWN_MS = 2500;          // post-exit lockout prevents double-fire

// Burger layer reveal — stable open-palm hold while in inspect mode.
// Stable = palm-width delta < PALM_HOLD_STABLE_THRESHOLD over the sample window.
// This zone is safely below the exit-inspect shrink threshold so there is no conflict.
const PALM_HOLD_DURATION_MS = 900;
const PALM_HOLD_STABLE_THRESHOLD = 0.012;
const BURGER_EXPLODE_COOLDOWN_MS = 2000;
const CAMERA_DEFAULT_DETECTION_INTERVAL_MS = 83;
const CAMERA_IPAD_INSPECT_DETECTION_INTERVAL_MS = 115;
const CAMERA_DEFAULT_CONSTRAINTS = {
  width: { ideal: 360, max: 360 },
  height: { ideal: 270, max: 270 },
  frameRate: { ideal: 15, max: 15 },
} satisfies MediaTrackConstraints;
const CAMERA_IPAD_CONSTRAINTS = {
  width: { ideal: 320, max: 360 },
  height: { ideal: 240, max: 270 },
  frameRate: { ideal: 12, max: 15 },
} satisfies MediaTrackConstraints;
// Fist → open-palm reveal gesture (Burger Inspect only).
// Armed on a fist pulse; fires when the hand stays open & not-fist for
// FIST_OPEN_PALM_STABLE_MS, as long as the armed state hasn't timed out.
// The stability window lets the double-fist add-to-order gesture take
// precedence when the user does fist → quick open → fist within ~250ms.
const FIST_OPEN_PALM_WINDOW_MS = 1500;
const FIST_OPEN_PALM_STABLE_MS = 280;

const LOGO_TEXT = "AURA ONE";
const LOGO_LETTERS: Record<string, string[]> = {
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
  N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
  O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
};

type LogoParticle = {
  home: THREE.Vector3;
  scatter: THREE.Vector3;
  seed: number;
};

type LandingPhase = "intro" | "menu";

const PERFORMANCE_MODE = true;
const AMBIENT_PARTICLE_COUNT = PERFORMANCE_MODE ? 260 : 700;
const MAX_CANVAS_DPR = PERFORMANCE_MODE ? 1.5 : 2;
const MATERIAL_OPACITY_EPSILON = 0.003;
const CAROUSEL_VISIBLE_SLOT_DISTANCE = 2;
const EXPLODED_STACK_SCALE = 0.78;
const BURGER_REVEAL_SCALE = 0.77;
const BURGER_REVEAL_Z_OFFSET = -0.55;
const BURGER_REVEAL_Y_OFFSET = 0;
const BASE_MENU_AUTO_ROTATE_MS = 7000;
const BASE_MENU_INSPECT_HOLD_MS = 4200;
const BASE_MENU_POST_INSPECT_RETURN_MS = 1800;
const BASE_MENU_NEXT_PRODUCT_DELAY_MS = 800;
const SHOW_MODE_EXIT_AFTER_NARRATION_MS = 650;
const SHOW_MODE_NEXT_PRODUCT_AFTER_RETURN_MS = 950;
const SHOW_MODE_PRODUCTS = [0, 1, 2, 3, 4] as const;
const SHOW_MODE_NARRATION = [
  "This is our signature spatial burger experience. Each layer opens with a calm ingredient reveal.",
  "This Louisiana Po' Boy brings crisp seafood, fresh vegetables, and a cool remoulade finish.",
  "The dessert is light and elegant, designed as a quiet finish to the meal.",
  "The fried chicken is crisp and savory, with a gentle honey thyme finish.",
  "The crawfish brings warm coastal spice and a clean Louisiana profile.",
] as const;

type SceneLayout = {
  mode: "phone-portrait" | "ipad-portrait" | "ipad-landscape" | "desktop";
  isPhone: boolean;
  isTablet: boolean;
  isPortrait: boolean;
  inspectX: number;
  inspectY: number;
  inspectZOffset: number;
  carouselXOffset: number;
  carouselYOffset: number;
  carouselScaleMultiplier: number;
  inspectProductScaleMultiplier: number;
  revealScale: number;
  revealX: number;
  revealY: number;
  revealZ: number;
  productInfoClassName: string;
  ingredientCardClassName: string;
};

type FullscreenRootElement = HTMLElement & {
  webkitRequestFullscreen?: () => Promise<void> | void;
};

type FullscreenDocument = Document & {
  webkitFullscreenElement?: Element | null;
};

type AuraSpeechRecognitionEvent = Event & {
  results: {
    length: number;
    [index: number]: {
      0: { transcript: string };
      isFinal?: boolean;
    };
  };
};

type AuraSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort?: () => void;
  onresult: ((event: AuraSpeechRecognitionEvent) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
};

type AuraSpeechWindow = Window & {
  SpeechRecognition?: new () => AuraSpeechRecognition;
  webkitSpeechRecognition?: new () => AuraSpeechRecognition;
};

type AuraAudioWindow = Window & {
  webkitAudioContext?: typeof AudioContext;
};

type AuraSoundName = "burgerReveal" | "voiceActivate" | "orderConfirm";

const AURA_VOICE_RESPONSES = [
  {
    intent: "WHAT_IS_THIS",
    phrases: ["what is this", "whats this", "what is that", "what am i seeing"],
    keywords: [["what", "this"]],
    answer:
      "This is our signature spatial burger experience. Each ingredient can open gently for a closer look.",
  },
  {
    intent: "INGREDIENTS",
    phrases: ["what is in this", "whats in this", "what comes with this"],
    keywords: [["ingredient"], ["inside"], ["comes", "with"]],
    answer:
      "The burger includes brioche bun, wagyu beef patty, aged cheddar, tomato, crisp lettuce, and truffle sauce.",
  },
  {
    intent: "RECOMMEND",
    phrases: ["what do you recommend", "what should i order"],
    keywords: [["recommend"], ["suggestion"], ["best", "item"], ["should", "order"]],
    answer:
      "I would start with the signature burger. It shows the ingredient reveal most clearly.",
  },
  {
    intent: "POPULAR",
    phrases: ["most popular item", "best seller"],
    keywords: [["popular"], ["favorite"], ["favourite"], ["best", "seller"]],
    answer:
      "The signature burger is the featured favorite in this experience.",
  },
  {
    intent: "SPICY",
    phrases: ["is this spicy", "spice level"],
    keywords: [["spicy"], ["hot"], ["spice"]],
    answer:
      "This burger is rich and savory, not strongly spicy. The flavor is focused on wagyu beef, cheddar, truffle sauce, and toasted brioche.",
  },
  {
    intent: "DAIRY",
    phrases: ["does this contain dairy"],
    keywords: [["dairy"], ["milk"], ["cheese"], ["lactose"]],
    answer:
      "Yes. This burger contains dairy from the aged cheddar and may also include dairy in the sauce or bun preparation.",
  },
  {
    intent: "GLUTEN",
    phrases: ["does this contain gluten"],
    keywords: [["gluten"], ["wheat"], ["bread"], ["bun"]],
    answer:
      "Yes. This burger contains gluten from the brioche bun. Please ask the restaurant team about gluten-free options.",
  },
  {
    intent: "ALLERGENS",
    phrases: [],
    keywords: [["allergen"], ["allergy"], ["allergies"]],
    answer:
      "This item contains wheat, dairy, and egg. Please ask a team member for complete allergen guidance.",
  },
  {
    intent: "PRICE",
    phrases: ["how much", "how much is this burger"],
    keywords: [["price"], ["cost"], ["dollar"], ["dollars"]],
    answer: "The signature burger combo is eighteen dollars and ninety cents.",
  },
  {
    intent: "HOW_TO_USE",
    phrases: ["how to use", "use this", "how does this work"],
    keywords: [["control"], ["gesture"], ["swipe"]],
    answer:
      "Swipe gently to explore. Open a product to inspect it, then rotate it with touch.",
  },
  {
    intent: "CUSTOMIZE",
    phrases: [],
    keywords: [["customize"], ["customise"], ["change"], ["modify"]],
    answer:
      "Yes. The team can help customize toppings and preparation.",
  },
  {
    intent: "CUSTOM_MENU",
    phrases: ["own menu", "our menu"],
    keywords: [["custom"], ["customize"], ["branding"], ["brand"]],
    answer:
      "Yes. AURA ONE can be configured around your menu, visuals, pricing, and featured items.",
  },
  {
    intent: "RESTAURANT_FIT",
    phrases: ["my restaurant", "our restaurant", "work in restaurant", "can this work"],
    keywords: [["restaurant"], ["install"]],
    answer:
      "Yes. It is designed for restaurants, demos, and premium hospitality spaces.",
  },
  {
    intent: "DEVICES",
    phrases: ["special device", "special devices", "need devices"],
    keywords: [["device"], ["devices"], ["tablet"], ["ipad"], ["touchscreen"], ["gesture", "display"]],
    answer:
      "Customers do not need special hardware. It can run on supported tablets, displays, or kiosk-style web devices.",
  },
] as const;

const AURA_VOICE_FALLBACK =
  "I can help with ingredients, allergens, pricing, recommendations, and restaurant customization.";

type AuraVoiceCommand = {
  answer: string;
  addPartIndex?: number;
  showPartIndex?: number;
  startOrderingMode?: boolean;
  startShowMode?: boolean;
  stopShowMode?: boolean;
  intent: string;
};

function normalizeAuraTranscript(transcript: string) {
  return transcript
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ");
}

function matchAuraMenuItem(transcript: string) {
  if (/\b(po boy|poboy|po' boy|poor boy)\b/.test(transcript)) return 1;
  if (/\b(dessert|sweet|cake|pastry)\b/.test(transcript)) return 2;
  if (/\b(chicken|fried chicken)\b/.test(transcript)) return 3;
  if (/\b(crawfish|crayfish)\b/.test(transcript)) return 4;
  if (/\b(burger|signature burger)\b/.test(transcript)) return 0;

  return null;
}

function getAuraOrderItemLabel(partIndex: number) {
  if (partIndex === 0) return "burger";
  if (partIndex === 1) return "po boy";
  if (partIndex === 2) return "dessert";
  if (partIndex === 3) return "fried chicken";

  return CAROUSEL_PARTS[partIndex].name.toLowerCase();
}

function resolveAuraVoiceCommand(
  transcript: string,
  orderingMode = false
): AuraVoiceCommand {
  const normalized = normalizeAuraTranscript(transcript);
  const menuItemIndex = matchAuraMenuItem(normalized);
  const wantsAdd =
    /\b(add|order|want|wants|i want|i would like|get me)\b/.test(normalized);
  const wantsShow =
    /\b(show|see|view|display|open|take me)\b/.test(normalized);
  const wantsOrderingMode =
    menuItemIndex === null &&
    (normalized.includes("can i order") ||
      normalized.includes("i want to order") ||
      normalized.includes("can i place an order") ||
      normalized.includes("place an order"));
  const wantsShowModeOn =
    normalized.includes("show on") ||
    normalized.includes("start show") ||
    normalized.includes("start showcase") ||
    normalized.includes("turn on show mode");
  const wantsShowModeOff =
    normalized.includes("show off") ||
    normalized.includes("stop show") ||
    normalized.includes("stop showcase") ||
    normalized.includes("turn off show mode");

  if (wantsShowModeOn) {
    return {
      intent: "SHOW_MODE_ON",
      startShowMode: true,
      answer: "Show mode is now active.",
    };
  }

  if (wantsShowModeOff) {
    return {
      intent: "SHOW_MODE_OFF",
      stopShowMode: true,
      answer: "Show mode is now off.",
    };
  }

  if (wantsOrderingMode) {
    return {
      intent: "START_ORDERING",
      startOrderingMode: true,
      answer: "Of course. What would you like?",
    };
  }

  if (orderingMode) {
    if (
      menuItemIndex !== null &&
      (menuItemIndex === 0 ||
        menuItemIndex === 1 ||
        menuItemIndex === 2 ||
        menuItemIndex === 3)
    ) {
      const itemLabel = getAuraOrderItemLabel(menuItemIndex);
      return {
        intent: "ORDER_MODE_ADD_ITEM",
        addPartIndex: menuItemIndex,
        answer: `Your ${itemLabel} has been added.`,
      };
    }

    return {
      intent: "ORDER_MODE_UNSUPPORTED",
      answer:
        "For this demo, I can add the burger, fried chicken, po boy, or dessert.",
    };
  }

  if (menuItemIndex !== null && wantsAdd) {
    const friendlyName = getAuraOrderItemLabel(menuItemIndex);
    return {
      intent: "ADD_ITEM",
      addPartIndex: menuItemIndex,
      answer: `Your ${friendlyName} has been added.`,
    };
  }

  if (menuItemIndex !== null && wantsShow) {
    const itemName = CAROUSEL_PARTS[menuItemIndex].name;
    return {
      intent: "SHOW_ITEM",
      showPartIndex: menuItemIndex,
      answer: `Here is ${itemName}.`,
    };
  }

  if (
    normalized.includes("what has gluten") ||
    normalized.includes("which has gluten")
  ) {
    return {
      intent: "GLUTEN_MENU",
      answer:
        "Items with gluten include the burger, po' boy, fried chicken, and dessert. Crawfish is the cleanest option here.",
    };
  }

  if (
    normalized.includes("what has dairy") ||
    normalized.includes("which has dairy")
  ) {
    return {
      intent: "DAIRY_MENU",
      answer:
        "Dairy appears in the burger, dessert, and fried chicken preparation. The po' boy may include dairy in the sauce.",
    };
  }

  if (
    menuItemIndex !== null &&
    (normalized.includes("how much") ||
      normalized.includes("price") ||
      normalized.includes("cost"))
  ) {
    const itemName = CAROUSEL_PARTS[menuItemIndex].name;
    const price = ITEM_PRICES[menuItemIndex].toFixed(2);
    return {
      intent: "ITEM_PRICE",
      answer: `${itemName} is ${price}.`,
    };
  }

  const match = AURA_VOICE_RESPONSES.find(({ phrases, keywords }) =>
    phrases.some((phrase) => normalized.includes(phrase)) ||
    keywords.some((group) => group.every((keyword) => normalized.includes(keyword)))
  );

  if (process.env.NODE_ENV === "development") {
    console.info("[AURA VOICE] transcript raw", transcript);
    console.info("[AURA VOICE] transcript normalized", normalized);
    console.info("[AURA VOICE] matched intent", match?.intent ?? "FALLBACK");
  }

  return {
    intent: match?.intent ?? "FALLBACK",
    answer: match?.answer ?? AURA_VOICE_FALLBACK,
  };
}

function selectAuraVoice(voices: SpeechSynthesisVoice[]) {
  const preferredNames = [
    "samantha",
    "karen",
    "daniel",
    "ava",
    "allison",
    "victoria",
    "moira",
    "fiona",
    "arthur",
  ];
  const preferredVoice = preferredNames
    .map((name) =>
      voices.find((voice) => voice.name.toLowerCase().includes(name))
    )
    .find(Boolean);

  return (
    preferredVoice ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en-us")) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en-gb")) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en-au")) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en")) ??
    null
  );
}

function requestSpatialFullscreenOnce(requestedRef: MutableRefObject<boolean>) {
  if (typeof document === "undefined" || requestedRef.current) return;

  const doc = document as FullscreenDocument;
  if (document.fullscreenElement || doc.webkitFullscreenElement) return;

  const root = document.documentElement as FullscreenRootElement;
  const requestFullscreen =
    root.requestFullscreen?.bind(root) ?? root.webkitRequestFullscreen?.bind(root);

  // iPhone/iPad Safari usually relies on Add to Home Screen / standalone mode
  // instead of the Fullscreen API. In that case this quietly becomes a no-op.
  if (!requestFullscreen) return;

  requestedRef.current = true;
  Promise.resolve(requestFullscreen()).catch(() => {
    requestedRef.current = false;
  });
}

function resolveSceneLayout(width: number, height: number): SceneLayout {
  const isPortrait = height > width;
  const isPhone = width < 700;
  const isTablet = width >= 700 && width <= 1180;
  const isTabletPortrait = isTablet && isPortrait;
  const isTabletLandscape = isTablet && !isPortrait;
  const mode = isPhone
    ? "phone-portrait"
    : isTabletPortrait
      ? "ipad-portrait"
      : isTabletLandscape
        ? "ipad-landscape"
        : "desktop";

  const layout: SceneLayout = {
    mode,
    isPhone,
    isTablet,
    isPortrait,
    inspectX: -0.60,
    inspectY: 0.42,
    inspectZOffset: 0,
    carouselXOffset: 0,
    carouselYOffset: 0,
    carouselScaleMultiplier: 1.12,
    inspectProductScaleMultiplier: 1.11,
    revealScale: BURGER_REVEAL_SCALE,
    revealX: 0,
    revealY: BURGER_REVEAL_Y_OFFSET,
    revealZ: BURGER_REVEAL_Z_OFFSET,
    productInfoClassName: isPhone
      ? "bottom-[4.25rem] left-4 right-4 w-auto max-h-[26vh] p-3.5"
      : isTabletPortrait
        ? "bottom-[calc(env(safe-area-inset-bottom)+3.65rem)] left-[6vw] w-[66vw] max-h-[18vh] p-3"
        : isTabletLandscape
          ? "bottom-[5rem] right-5 w-[min(13rem,30vw)] max-h-[34vh] p-3.5"
          : "bottom-[6.75rem] right-5 md:right-8 w-[min(15.5rem,calc(38vw-1rem))] max-h-[min(38vh,20rem)] p-4 md:p-5",
    ingredientCardClassName: isPhone
      ? "bottom-[4.25rem] left-4 w-[min(13rem,calc(100vw-2rem))] max-h-[30vh] translate-y-0"
      : isTabletPortrait
        ? "bottom-[4rem] right-4 w-[min(12.5rem,calc(100vw-2rem))] max-h-[26vh] translate-y-0"
        : isTabletLandscape
          ? "bottom-[4.75rem] left-5 w-[min(11.25rem,25vw)] max-h-[34vh] translate-y-0"
          : "left-5 top-1/2 w-[min(12.75rem,calc(100vw-2rem))] -translate-y-1/2 md:left-6",
  };

  if (mode === "phone-portrait") {
    layout.inspectX = 0;
    layout.inspectY = 0.20;
    layout.inspectZOffset = -0.32;
    layout.carouselScaleMultiplier = 1.05;
    layout.inspectProductScaleMultiplier = 1.05;
    layout.revealScale = 0.55;
    layout.revealX = 0.02;
    layout.revealY = -0.02;
    layout.revealZ = -0.18;
  } else if (mode === "ipad-portrait") {
    layout.inspectX = -0.28;
    layout.inspectY = 0.30;
    layout.inspectZOffset = -0.24;
    layout.carouselXOffset = -0.48;
    layout.carouselYOffset = 0.42;
    layout.carouselScaleMultiplier = 1.08;
    layout.inspectProductScaleMultiplier = 1.08;
    layout.revealScale = 0.59;
    layout.revealX = 0;
    layout.revealY = -0.03;
    layout.revealZ = -0.18;
  } else if (mode === "ipad-landscape") {
    layout.inspectX = -0.36;
    layout.inspectY = 0.32;
    layout.inspectZOffset = -0.14;
    layout.carouselScaleMultiplier = 1.10;
    layout.inspectProductScaleMultiplier = 1.10;
    layout.revealScale = 0.68;
    layout.revealX = 0.12;
    layout.revealY = -0.05;
    layout.revealZ = -0.14;
  }

  return layout;
}

function getSceneLayoutSnapshot() {
  if (typeof window === "undefined") {
    return resolveSceneLayout(1440, 900);
  }

  return resolveSceneLayout(window.innerWidth, window.innerHeight);
}

function isIpadSafariRuntime() {
  if (typeof navigator === "undefined") return false;

  const ua = navigator.userAgent;
  const isSafari = /Safari/i.test(ua) && !/Chrome|CriOS|FxiOS|EdgiOS/i.test(ua);
  const isIpad =
    /iPad/i.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

  return isIpad && isSafari;
}

function useSceneLayout() {
  const [layout, setLayout] = useState<SceneLayout>(getSceneLayoutSnapshot);

  useEffect(() => {
    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    const updateLayout = () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        resizeTimer = null;
        setLayout(getSceneLayoutSnapshot());
      }, 120);
    };

    updateLayout();
    window.addEventListener("resize", updateLayout);
    window.addEventListener("orientationchange", updateLayout);

    return () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      window.removeEventListener("resize", updateLayout);
      window.removeEventListener("orientationchange", updateLayout);
    };
  }, []);

  return layout;
}

type GestureAction =
  | "EXPLODE"
  | "ASSEMBLE"
  | "NEXT_PART"
  | "PREV_PART"
  | "ENTER_INSPECT"
  | "EXIT_INSPECT"
  | "RESET"
  | "ROTATE_INSPECT_LEFT"
  | "ROTATE_INSPECT_RIGHT"
  | "TOGGLE_BURGER_EXPLODE"
  | "ADD_TO_ORDER"
  | "REMOVE_LAST"
  | "CLEAR_ORDER";

type CameraGestureStatus =
  | "CAMERA OFF"
  | "LOADING CAMERA"
  | "TRYING GPU"
  | "GPU FAILED"
  | "FALLBACK CPU"
  | "HAND TRACKING READY"
  | "CAMERA READY"
  | "READY"
  | "SWIPE LEFT"
  | "SWIPE RIGHT"
  | "COOLDOWN"
  | "OPPOSITE LOCK"
  | "OPEN HAND INSPECT"
  | "INSPECT GESTURE"
  | "FIST AGAIN TO ADD"
  | "ORDER CANCELLED"
  | "ADDED TO ORDER"
  | "MOVE HAND BACK"
  | "EXIT INSPECT"
  | "REVEALING LAYERS"
  | "ASSEMBLING BURGER"
  | "OPEN PALM TO REVEAL"
  | "CAMERA ERROR";



function smoothStep(value: number) {
  return value * value * (3 - 2 * value);
}

// Smooth 0→1 ramp inside a [a, b] band. Returns 0 below `a`, 1 above `b`,
// and a smoothStep curve between. Used to slice a single progress value
// (0..1) into disjoint timeline phases for staged transitions.
function smoothBand(value: number, a: number, b: number) {
  if (b <= a) return value >= b ? 1 : 0;
  const t = (value - a) / (b - a);
  return smoothStep(Math.max(0, Math.min(1, t)));
}

function setAnimatedLightIntensity(light: THREE.Light | null, intensity: number) {
  if (!light) return;

  light.intensity = intensity;
  light.visible = intensity > 0.001;
}

function seededUnit(index: number) {
  const value = Math.sin(index * 12.9898) * 43758.5453;

  return value - Math.floor(value);
}

function phasedPathProgress(progress: number, phase = 0) {
  const clampedPhase = THREE.MathUtils.clamp(phase, 0, 0.42);
  const delayed = THREE.MathUtils.clamp(
    (progress - clampedPhase) / (1 - clampedPhase),
    0,
    1
  );
  const blended = THREE.MathUtils.lerp(progress, delayed, 0.42);

  return THREE.MathUtils.clamp(blended, 0, 1);
}

function writeQuadraticBezier(
  target: THREE.Vector3,
  start: [number, number, number],
  control: [number, number, number],
  end: [number, number, number],
  progress: number
) {
  const inverse = 1 - progress;
  const startWeight = inverse * inverse;
  const controlWeight = 2 * inverse * progress;
  const endWeight = progress * progress;

  target.set(
    start[0] * startWeight + control[0] * controlWeight + end[0] * endWeight,
    start[1] * startWeight + control[1] * controlWeight + end[1] * endWeight,
    start[2] * startWeight + control[2] * controlWeight + end[2] * endWeight
  );
}

function resolvePathPosition(
  target: THREE.Vector3,
  basePosition: [number, number, number],
  explodedPosition: [number, number, number],
  midPosition: [number, number, number] | undefined,
  progress: number
) {
  if (!midPosition) {
    writeQuadraticBezier(
      target,
      basePosition,
      [
        (basePosition[0] + explodedPosition[0]) * 0.5,
        (basePosition[1] + explodedPosition[1]) * 0.5,
        (basePosition[2] + explodedPosition[2]) * 0.5,
      ],
      explodedPosition,
      progress
    );

    return target;
  }

  writeQuadraticBezier(
    target,
    basePosition,
    midPosition,
    explodedPosition,
    progress
  );

  return target;
}

function wrappedSlot(partIndex: number, activePartIndex: number, totalParts: number) {
  const rawSlot = partIndex - activePartIndex;
  const wrapped = ((rawSlot % totalParts) + totalParts) % totalParts;

  return wrapped > totalParts / 2 ? wrapped - totalParts : wrapped;
}

function createLogoParticles() {
  const particles: LogoParticle[] = [];
  const cellSize = 0.066;
  const letterGap = 1.5;
  const logoY = -1.12;
  const logoZ = 1.02;
  let cursor = 0;

  LOGO_TEXT.split("").forEach((letter) => {
    if (letter === " ") {
      cursor += 2.1;
      return;
    }

    const grid = LOGO_LETTERS[letter];

    grid.forEach((row, rowIndex) => {
      row.split("").forEach((filled, columnIndex) => {
        if (filled !== "1") return;

        const x = (cursor + columnIndex) * cellSize;
        const y = (3 - rowIndex) * cellSize;
        const home = new THREE.Vector3(x, y, logoZ);
        const seed = particles.length + 1;
        const scatter = new THREE.Vector3();

        particles.push({ home, scatter, seed });
      });
    });

    cursor += 5 + letterGap;
  });

  const centerX =
    particles.reduce((total, particle) => total + particle.home.x, 0) /
    particles.length;

  particles.forEach((particle) => {
    particle.home.x -= centerX;
    particle.home.y += logoY;
    const horizontal = particle.home.x >= 0 ? 1 : -1;
    const vertical = particle.home.y >= logoY ? 1 : -1;

    particle.scatter.set(
      particle.home.x * 1.22 +
        horizontal * (0.74 + seededUnit(particle.seed * 4) * 0.58),
      particle.home.y * 1.04 +
        vertical * (0.34 + seededUnit(particle.seed * 5) * 0.34),
      -1.68 - seededUnit(particle.seed * 6) * 1.05
    );
  });

  return particles;
}

function Part({
  partIndex,
  basePosition,
  midPosition,
  explodedPosition,
  focusScale = 1.45,
  secondaryScale = 0.74,
  inspectZFocus = 0.9,
  inspectScaleMultiplier = 1.38,
  explodeDelay = 0,
  assembleDelay = 0,
  baseRotation = [0, 0, 0],
  explodedRotation = [0, 0, 0],
  selfRotationAmount = 1,
  freezeParentRotation = false,
  motionSeed = 0,
  meshOpacity = 1,
  children,
  progressRef,
  activePartIndex,
  totalParts,
  carouselEnabled,
  inspectMode,
  inspectRotationRef,
  layout,
}: {
  partIndex: number;
  basePosition: [number, number, number];
  midPosition?: [number, number, number];
  explodedPosition: [number, number, number];
  focusScale?: number;
  secondaryScale?: number;
  inspectZFocus?: number;
  inspectScaleMultiplier?: number;
  explodeDelay?: number;
  assembleDelay?: number;
  baseRotation?: [number, number, number];
  explodedRotation?: [number, number, number];
  selfRotationAmount?: number;
  freezeParentRotation?: boolean;
  motionSeed?: number;
  meshOpacity?: number;
  children: ReactNode;
  progressRef: MutableRefObject<number>;
  activePartIndex: number;
  totalParts: number;
  carouselEnabled: boolean;
  inspectMode: boolean;
  inspectRotationRef: MutableRefObject<{ x: number; y: number }>;
  layout: SceneLayout;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const activeLightRef = useRef<THREE.SpotLight>(null);
  const materialsRef = useRef<THREE.MeshStandardMaterial[]>([]);
  const lastMaterialOpacityRef = useRef<number | null>(null);
  const lastMaterialTransparentRef = useRef<boolean | null>(null);
  const selfRotationRef = useRef(new THREE.Euler(0, 0, 0));
  const manualRotationRef = useRef(new THREE.Vector2());
  const manualRotationTargetRef = useRef(new THREE.Vector2());
  const partScaleRef = useRef(1);
  const highlightRef = useRef(0);
  const previousProgressRef = useRef(0);
  const previousRawProgressRef = useRef(0);
  const dockedRef = useRef(true);
  const dockPhaseRef = useRef<number | null>(null);
  const pathPositionRef = useRef(new THREE.Vector3());
  const renderPositionRef = useRef(new THREE.Vector3());
  const targetPositionRef = useRef(new THREE.Vector3());
  const inspectPositionRef = useRef(new THREE.Vector3());
  const positionInitializedRef = useRef(false);
  const inspectBlendRef = useRef(0);
  const dimRef = useRef(0);
  const cullVisibilityRef = useRef(1);
  const [contentMounted, setContentMounted] = useState(false);
  const inspectIdleYRef = useRef(0);
  const prevIsInspectActiveRef = useRef(false);
  const frozenParentRotationRef = useRef(new THREE.Euler());
  const prevFreezeParentRotationRef = useRef(false);
  const motion = useMemo(() => {
    const direction = motionSeed % 2 === 0 ? 1 : -1;
    const dockStart = midPosition ?? explodedPosition;
    const travel = new THREE.Vector3(
      basePosition[0] - dockStart[0],
      basePosition[1] - dockStart[1],
      basePosition[2] - dockStart[2]
    );
    const travelDistance = travel.length();

    return {
      floatSpeed: 0.55 + (motionSeed % 5) * 0.08,
      floatAmount: (0.024 + (motionSeed % 4) * 0.005) * selfRotationAmount,
      driftAmount: (0.012 + (motionSeed % 3) * 0.004) * selfRotationAmount,
      dockAmplitude:
        travelDistance > 0.001
          ? Math.min(travelDistance * 0.045, 0.07) *
            (0.86 + (motionSeed % 3) * 0.08)
          : 0,
      dockDirection:
        travelDistance > 0.001 ? travel.normalize() : new THREE.Vector3(),
      dockRotation: new THREE.Vector3(
        0,
        -direction * (0.018 + (motionSeed % 4) * 0.004),
        0
      ),
      rotationSpeed: new THREE.Vector3(
        0,
        -direction * (0.06 + (motionSeed % 4) * 0.012) * selfRotationAmount,
        0
      ),
    };
  }, [basePosition, explodedPosition, midPosition, motionSeed, selfRotationAmount]);

  const setGroupRef = useCallback((group: THREE.Group | null) => {
    groupRef.current = group;
    materialsRef.current = [];
    lastMaterialOpacityRef.current = null;
    lastMaterialTransparentRef.current = null;

    if (!group) return;

    group.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const material = mesh.material;
      const materials = Array.isArray(material) ? material : [material];

      materials.forEach((entry) => {
        if (
          entry instanceof THREE.MeshStandardMaterial &&
          "emissiveIntensity" in entry
        ) {
          materialsRef.current.push(entry);
        }
      });
    });
  }, []);

  const renderSlot = wrappedSlot(partIndex, activePartIndex, totalParts);
  const shouldMountContent =
    carouselEnabled &&
    (inspectMode
      ? renderSlot === 0
      : Math.abs(renderSlot) <= CAROUSEL_VISIBLE_SLOT_DISTANCE);

  useEffect(() => {
    const id = setTimeout(
      () => setContentMounted(shouldMountContent),
      shouldMountContent ? 0 : 420
    );

    return () => clearTimeout(id);
  }, [shouldMountContent]);


  useFrame(({ clock }, delta) => {
    if (!groupRef.current) return;

    const rawProgress = progressRef.current;
    const previousRawProgress = previousRawProgressRef.current;
    const slot = wrappedSlot(partIndex, activePartIndex, totalParts);
    const slotDistance = Math.abs(slot);
    const cullTarget = carouselEnabled
      ? inspectMode
        ? slot === 0
          ? 1
          : 0
        : slotDistance <= CAROUSEL_VISIBLE_SLOT_DISTANCE
          ? 1
          : 0
      : 1;

    if (
      cullTarget === 0 &&
      cullVisibilityRef.current < 0.01 &&
      Math.abs(rawProgress - previousRawProgress) < 0.0005
    ) {
      groupRef.current.visible = false;
      if (activeLightRef.current) {
        setAnimatedLightIntensity(activeLightRef.current, 0);
      }
      previousRawProgressRef.current = rawProgress;
      return;
    }

    const t = clock.getElapsedTime();
    const assembling = rawProgress < previousRawProgress;
    const localProgress = assembling
      ? 1 - phasedPathProgress(1 - rawProgress, assembleDelay)
      : phasedPathProgress(rawProgress, explodeDelay);
    const previousProgress = previousProgressRef.current;
    const p = smoothStep(localProgress);
    const separatedProgress = smoothStep(
      THREE.MathUtils.clamp((localProgress - 0.22) / 0.78, 0, 1)
    );
    const floatPhase = t * motion.floatSpeed + motionSeed;
    const pathPosition = resolvePathPosition(
      pathPositionRef.current,
      basePosition,
      explodedPosition,
      midPosition,
      p
    );
    const dockThreshold = 0.045;
    const isInspectActive = inspectMode && slot === 0;
    const parentRotationFrozen =
      freezeParentRotation && isInspectActive && separatedProgress > 0.1;
    const angle = (slot / totalParts) * Math.PI * 2;
    const activePresence = carouselEnabled && slot === 0 ? separatedProgress : 0;
    const carouselPresence = carouselEnabled ? separatedProgress : 0;
    const inspectPresence = inspectMode && carouselEnabled ? separatedProgress : 0;
    const depth = Math.cos(angle);
    const side = Math.abs(Math.sin(angle));
    cullVisibilityRef.current = THREE.MathUtils.lerp(
      cullVisibilityRef.current,
      cullTarget,
      1 - Math.exp(-delta * 8)
    );
    const cullVisibility = cullVisibilityRef.current;
    if (cullTarget === 0 && cullVisibility < 0.01) {
      groupRef.current.visible = false;
      previousProgressRef.current = localProgress;
      previousRawProgressRef.current = rawProgress;
      return;
    }
    const carouselScale =
      (slot === 0
        ? focusScale
        : THREE.MathUtils.lerp(0.5, secondaryScale, Math.max(depth, 0) * 0.55)) *
      layout.carouselScaleMultiplier;
    const inspectScale =
      slot === 0
        ? focusScale * inspectScaleMultiplier * layout.inspectProductScaleMultiplier
        : 0.22;
    const scaleTarget = THREE.MathUtils.lerp(
      THREE.MathUtils.lerp(0, carouselScale, carouselPresence),
      inspectScale,
      inspectPresence
    );
    // Highlight peaks at 1.0 on the active slot whether or not in inspect —
    // pre-inspect spotlight needs the full presence to actually read on stage.
    const highlightTarget = activePresence * 1.0;

    if (!assembling && localProgress > dockThreshold * 2) {
      dockedRef.current = false;
      dockPhaseRef.current = null;
    }

    if (
      assembling &&
      !dockedRef.current &&
      previousProgress > dockThreshold &&
      localProgress <= dockThreshold
    ) {
      dockedRef.current = true;
      dockPhaseRef.current = 0;
    }

    let dockPulse = 0;

    if (dockPhaseRef.current !== null) {
      dockPhaseRef.current += delta;
      dockPulse =
        Math.sin(dockPhaseRef.current * 15) *
        Math.exp(-dockPhaseRef.current * 5.2);

      if (dockPhaseRef.current > 1.1) {
        dockPhaseRef.current = null;
      }
    }

    const activeRotationBoost = activePresence && !isInspectActive ? 2.1 : 1;

    // Detect inspect entry/exit for this slot and keep rotation continuous.
    const enteredInspect = isInspectActive && !prevIsInspectActiveRef.current;
    const exitedInspect  = !isInspectActive && prevIsInspectActiveRef.current && slot === 0;
    if (enteredInspect) {
      // Seed inspect idle from the live self-rotation so there is no jump on entry.
      inspectIdleYRef.current = selfRotationRef.current.y * Math.max(separatedProgress, 0.01);
    }
    if (exitedInspect) {
      // Re-seed self-rotation from inspect idle so exit is equally seamless.
      selfRotationRef.current.y = separatedProgress > 0.01
        ? inspectIdleYRef.current / separatedProgress
        : inspectIdleYRef.current;
    }
    prevIsInspectActiveRef.current = isInspectActive;

    if (!isInspectActive) {
      // Decay any legacy X/Z to zero — only Y (horizontal) rotation accumulates.
      selfRotationRef.current.x = THREE.MathUtils.lerp(
        selfRotationRef.current.x, 0, 1 - Math.exp(-delta * 2.5)
      );
      selfRotationRef.current.y +=
        delta * motion.rotationSpeed.y * separatedProgress * activeRotationBoost;
      selfRotationRef.current.z = THREE.MathUtils.lerp(
        selfRotationRef.current.z, 0, 1 - Math.exp(-delta * 2.5)
      );
    }

    if (parentRotationFrozen && !prevFreezeParentRotationRef.current) {
      frozenParentRotationRef.current.copy(groupRef.current.rotation);
    }
    prevFreezeParentRotationRef.current = parentRotationFrozen;

    if (isInspectActive && separatedProgress > 0.1 && !parentRotationFrozen) {
      // Slow deliberate yaw — cinematic idle rotation.
      inspectIdleYRef.current += delta * 0.15;
    }
    // Cinematic easing: all inspect transitions use slower damping so movement
    // feels heavy and intentional rather than snappy.
    partScaleRef.current = THREE.MathUtils.lerp(
      partScaleRef.current,
      scaleTarget,
      1 - Math.exp(-delta * 4.2)
    );
    highlightRef.current = THREE.MathUtils.lerp(
      highlightRef.current,
      highlightTarget,
      1 - Math.exp(-delta * 3.8)
    );
    if (activeLightRef.current) {
      // Cinematic top spotlight. Two intensity tiers driven by highlightRef
      // (which fades in as the item arrives at the active slot):
      //   carousel front: 2.4× — clearly visible product spotlight pre-inspect
      //   inspect mode:   4.6× — full key on the hero during inspect
      // Plus a brief dock-pulse boost (~300–500 ms decaying sinusoid) that
      // adds a "reveal moment" intensity bump as the item lands in slot 0.
      const baseMul = isInspectActive ? 4.6 : 2.4;
      const dockBoost = Math.abs(dockPulse) * 1.6 * activePresence;
      setAnimatedLightIntensity(
        activeLightRef.current,
        highlightRef.current * baseMul + dockBoost
      );
    }
    inspectBlendRef.current = THREE.MathUtils.lerp(
      inspectBlendRef.current,
      inspectPresence,
      1 - Math.exp(-delta * 2.4)
    );

    const normalX =
      pathPosition.x +
      Math.cos(floatPhase * 0.7) * motion.driftAmount * separatedProgress +
      motion.dockDirection.x * motion.dockAmplitude * dockPulse;
    const normalY =
      pathPosition.y +
      Math.sin(floatPhase) * motion.floatAmount * separatedProgress +
      motion.dockDirection.y * motion.dockAmplitude * dockPulse;
    const normalZ =
      pathPosition.z +
      Math.sin(floatPhase * 0.8) * motion.driftAmount * separatedProgress +
      motion.dockDirection.z * motion.dockAmplitude * dockPulse;
    const normalCarouselOffset = layout.mode === "ipad-portrait" ? 1 : 0;
    const activeCarouselXOffset = normalCarouselOffset && slot === 0 ? 1 : 0;
    const neighborSpread = slotDistance === 1 ? 0.85 : slotDistance === 2 ? 0.35 : 0;
    const neighborDepthOffset = slotDistance === 1 ? 0.55 : 0;
    const carouselX =
      Math.sin(angle) * (3.8 + neighborSpread) +
      layout.carouselXOffset * activeCarouselXOffset;
    const carouselY = 0.1 - depth * 0.52 + layout.carouselYOffset * normalCarouselOffset;
    const carouselZ = depth * 2.1 - side * 0.65 - neighborDepthOffset;
    const rearDirection = slot === 0 ? 0 : Math.sign(slot) || 1;
    // Inactive items pushed wider and deeper for strong spatial separation.
    // Inspected hero shifted further left (−0.30 → −0.60) for clearer
    // breathing room between the product and the right-side info card.
    // Stays well inside the visible X range (~±2.11 at the inspect plane)
    // so no product silhouette — including burger reveal — clips on the
    // left edge. Off-slot items unchanged.
    const inspectX = slot === 0 ? layout.inspectX : rearDirection * (5.0 + slotDistance * 1.0);
    // Active item gets a very slow gentle float — alive but restrained.
    // Y lifted 0.25 → 0.42 (+0.17) so the inspected hero sits at "screen
    // center, slightly below" — closer to the optical center against the
    // bottom HUD while still leaving room above for the larger inspect scale.
    const inspectY = slot === 0
      ? layout.inspectY + Math.sin(t * 0.28) * 0.036 * inspectBlendRef.current
      : -0.18 - slotDistance * 0.14;
    const inspectZ = slot === 0 ? inspectZFocus + layout.inspectZOffset : -5.8 - slotDistance * 2.2;
    inspectPositionRef.current.set(inspectX, inspectY, inspectZ);

    const dimTarget = inspectMode && slot !== 0 ? 1 : 0;
    // Slower dim — background recedes gracefully, not abruptly.
    dimRef.current = THREE.MathUtils.lerp(
      dimRef.current,
      dimTarget,
      1 - Math.exp(-delta * 1.9)
    );

    targetPositionRef.current.set(
      THREE.MathUtils.lerp(normalX, carouselX, carouselPresence),
      THREE.MathUtils.lerp(normalY, carouselY, carouselPresence),
      THREE.MathUtils.lerp(normalZ, carouselZ, carouselPresence)
    );
    targetPositionRef.current.lerp(
      inspectPositionRef.current,
      inspectBlendRef.current
    );
    if (!positionInitializedRef.current) {
      renderPositionRef.current.copy(targetPositionRef.current);
      positionInitializedRef.current = true;
    }

    // Heavier, controlled glide — gives the product weight during inspect entry/exit.
    renderPositionRef.current.lerp(
      targetPositionRef.current,
      1 - Math.exp(-delta * 2.9)
    );
    groupRef.current.position.copy(renderPositionRef.current);
    groupRef.current.visible =
      partScaleRef.current > 0.02 && cullVisibility > 0.02;
    manualRotationTargetRef.current.set(
      0,
      inspectMode && activePresence && !parentRotationFrozen
        ? inspectRotationRef.current.y
        : 0
    );
    manualRotationRef.current.lerp(
      manualRotationTargetRef.current,
      1 - Math.exp(-delta * 4.5)
    );

    if (parentRotationFrozen) {
      groupRef.current.rotation.copy(frozenParentRotationRef.current);
    } else {
      groupRef.current.rotation.set(
        THREE.MathUtils.lerp(baseRotation[0], explodedRotation[0], separatedProgress) +
          motion.dockRotation.x * dockPulse,
        THREE.MathUtils.lerp(baseRotation[1], explodedRotation[1], separatedProgress) +
          (isInspectActive
            ? inspectIdleYRef.current
            : selfRotationRef.current.y * separatedProgress) +
          manualRotationRef.current.y +
          motion.dockRotation.y * dockPulse,
        THREE.MathUtils.lerp(baseRotation[2], explodedRotation[2], separatedProgress) +
          motion.dockRotation.z * dockPulse
      );
    }
    groupRef.current.scale.setScalar(partScaleRef.current);

    // Only manage dim/opacity — never override each material's own emissive color.
    // Food surfaces keep their natural colors; only brand accent rings carry emissive.
    const emergeFade = smoothStep(THREE.MathUtils.clamp(carouselPresence * 2.5, 0, 1));
    const targetTransparent =
      meshOpacity < 1 ||
      dimRef.current > 0.005 ||
      emergeFade < 1 ||
      cullVisibility < 0.995;
    const targetOpacity =
      THREE.MathUtils.lerp(meshOpacity, 0.07, dimRef.current) *
      emergeFade *
      cullVisibility;
    const shouldUpdateMaterials =
      lastMaterialTransparentRef.current !== targetTransparent ||
      lastMaterialOpacityRef.current === null ||
      Math.abs(lastMaterialOpacityRef.current - targetOpacity) >
        MATERIAL_OPACITY_EPSILON;

    if (shouldUpdateMaterials) {
      materialsRef.current.forEach((material) => {
        material.transparent = targetTransparent;
        material.opacity = targetOpacity;
      });
      lastMaterialTransparentRef.current = targetTransparent;
      lastMaterialOpacityRef.current = targetOpacity;
    }

    previousProgressRef.current = localProgress;
    previousRawProgressRef.current = rawProgress;
  });

  return (
    <group ref={setGroupRef}>
      {contentMounted && <Suspense fallback={null}>{children}</Suspense>}
      {/* Cinematic top spotlight. Sits closer above the item (was at y=3.8,
          now y=2.6) with much lower decay (1.4 → 0.8) so the cone actually
          lands on the product instead of falling off in the 3-unit gap.
          Tighter angle + slightly crisper penumbra reads as a true product
          spotlight rather than a soft fill. */}
      <spotLight
        ref={activeLightRef}
        color="#ffe6c2"
        position={[0, 2.6, 0.4]}
        angle={0.55}
        penumbra={0.65}
        distance={6.5}
        decay={0.8}
        intensity={0}
        visible={false}
      />
    </group>
  );
}

// Single declarative config for the 9 ingredient layers. Indexed top → bottom
// (idx 0 = top-bun, idx 8 = bottom-bun) so the index doubles as the
// stack-position-from-top stagger key (top lifts first, bottom settles first).
type BurgerLayerConfig = {
  path: string;
  name: string;
  assembledY: number;
  revealedY: number;
  // Reveal-only X/Z drift (gated by spreadP).
  revealedOffset: readonly [number, number];
  // Always-applied corrective rotation (compensates for GLB orientation).
  baseRotation: readonly [number, number, number];
  // Reveal-only extra rotation tilt (gated by spreadP).
  revealedRotation: readonly [number, number, number];
  // Idle Y-axis spin speed (rad/sec), active only in revealed state.
  idleYRotSpeed: number;
  scale: number;
  // Optional per-axis Y-scale multiplier (on top of uniform `scale`). Used to
  // vertically squish ingredients whose GLB ships as a tall 3D shape — e.g.
  // a lettuce head — so they read as a flat layer inside the burger.
  scaleY?: number;
  // Optional model-level rotation correction, applied INSIDE BurgerLayerGLB
  // on the innermost wrapper (before any animation rotation composes on top).
  // Mathematically equivalent to baseRotation, but kept separate so it can
  // be reasoned about as a "fix the GLB's authoring orientation" knob.
  modelRotationCorrection?: readonly [number, number, number];
  // Optional model-level visual offset after bbox centering. Used sparingly for
  // assets whose visual mass is not aligned with their computed bounds center.
  modelPositionCorrection?: readonly [number, number, number];
  // Optional motion weight. >1 = heavier ingredient (slower spread / settle),
  // <1 = lighter (faster). Used to give buns / patty a slight heft while
  // bacon / lettuce / onion ring read as lighter. Defaults to 1.0.
  weight?: number;
  // Post-normalization width tuning. `scale` alone can't fix mismatched
  // rendered diameters because BurgerLayerGLB normalises the *max* bbox
  // axis, not specifically the horizontal extent — a tall bun and a flat
  // patty at the same `scale` produce different visible widths. Use this
  // multiplier to dial each ingredient's final rendered diameter against
  // the top-bun as the reference. Defaults to 1.0.
  visualScaleMultiplier?: number;
  // Optional per-layer PBR tuning. All three default to the BurgerLayerGLB
  // baseline (metalness 0 / roughness capped at 0.88 / envMap 0.72) and only
  // override when a value is explicitly set. Textures (baseColor, normal,
  // ORM) are always preserved — no color tinting here.
  materialMetalness?: number;
  materialRoughness?: number;
  materialEnvMapIntensity?: number;
  doubleSide: boolean;
};

// ─── ORIENTATION TEST HARNESS ──────────────────────────────────────────────
// Some GLB assets ship with non-flat default orientations. Trial-and-error
// table of candidate rotation corrections. Set LETTUCE_ORIENTATION_TEST or
// ONION_ORIENTATION_TEST to a key in ORIENTATION_TESTS to try a candidate.
// The "correct" candidate is the one that makes the broad surface horizontal:
//   - lettuce: broad leaf face points up (±Y)
//   - onion ring: ring hole axis points up (±Y) — ring sits like an "O" on top
//
// Already tried: B (−π/2 X) — only flipped vertical direction, not flat.
//                A (+π/2 X) — also did not work in earlier passes (different asset).
// Current guess: C (+π/2 Z) for both — different axis from what's been tested.
//
// If C also fails, rotate through D / E / F / G / H. Single-axis candidates
// (A–F) handle pure-axis authoring. Two-axis candidates (G, H) handle assets
// authored sideways (broad face at a 45° / oblique pose).
const ORIENTATION_TESTS = {
  A: [ Math.PI / 2, 0, 0           ] as const, // +90° X
  B: [-Math.PI / 2, 0, 0           ] as const, // −90° X  (already tried — only flipped)
  C: [ 0, 0,  Math.PI / 2          ] as const, // +90° Z
  D: [ 0, 0, -Math.PI / 2          ] as const, // −90° Z
  E: [ 0,  Math.PI / 2, 0          ] as const, // +90° Y
  F: [ 0, -Math.PI / 2, 0          ] as const, // −90° Y
  G: [ Math.PI / 2, 0,  Math.PI / 2] as const, // +90° X then +90° Z
  H: [-Math.PI / 2, 0,  Math.PI / 2] as const, // −90° X then +90° Z
} satisfies Record<string, readonly [number, number, number]>;

type OrientationTestKey = keyof typeof ORIENTATION_TESTS;

const DEBUG_BURGER_BOUNDS = false;

// Pick which candidate to try for each problem layer.
// Change these values to test a different rotation without touching the layer
// config below. After the right key is found we can inline the value.
const LETTUCE_ORIENTATION_TEST: OrientationTestKey = "C";
const ONION_ORIENTATION_TEST:   OrientationTestKey = "C";

const BURGER_LAYERS: ReadonlyArray<BurgerLayerConfig> = [
  // 0 — TOP BUN
  {
    path: "/models/burger-layers/top-bun.glb",
    name: "Top Bun",
    // Lowered 0.24 → 0.19 with the upper group so the whole top half sits
    // closer to the onion ring — burger reads more compact.
    assembledY:  0.19,
    revealedY:   1.40,
    revealedOffset:  [ 0.00, -0.03],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.16,         0, 0],
    idleYRotSpeed:  0.038,
    // Trimmed 0.92 → 0.85 (−7.6 %) so the top bun no longer dominates the
    // crown relative to the rest of the burger after the overall product
    // scale bump on EXPLODED_STACK_SCALE.
    scale: 0.85,
    weight: 1.10,  // heavier bun — lifts first on reveal, settles last on reassemble
    // Matte bread — no plastic shine. Higher roughness, lower env reflection.
    materialMetalness: 0,
    materialRoughness: 0.88,
    materialEnvMapIntensity: 0.50,
    doubleSide: false,
  },
  // 1 — BACON  (moved up into the old sauce slot, sits directly under top bun)
  {
    path: "/models/burger-layers/bacon%20(1).glb",
    name: "Bacon",
    // Lowered 0.16 → 0.11 in lockstep with the upper-group compression.
    assembledY:  0.11,
    revealedY:   1.00,
    revealedOffset:  [ 0.00,  0.05],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.14,  0.40, -0.05],
    idleYRotSpeed:  0.041,
    // +15% from 0.78 — bacon reads more clearly across the burger
    scale: 0.90,
    weight: 0.85,  // light strips — quick to spread
    // Subtle fat sheen — slightly glossier than bun, still not plastic.
    materialMetalness: 0.05,
    materialRoughness: 0.55,
    materialEnvMapIntensity: 0.85,
    doubleSide: true,
  },
  // 2 — LETTUCE  (lettuce2.glb — orientation corrected via test harness)
  {
    path: "/models/burger-layers/lettuce2.glb",
    name: "Lettuce",
    // Dropped 0.06 → 0.01 with the upper-group compression.
    assembledY:  0.01,
    revealedY:   0.60,
    revealedOffset:  [-0.03,  0.00],
    baseRotation:    [ 0, 0, 0],
    // Driven by LETTUCE_ORIENTATION_TEST above — change that to swap candidates.
    modelRotationCorrection: ORIENTATION_TESTS[LETTUCE_ORIENTATION_TEST],
    revealedRotation:[ 0, 0, 0],
    idleYRotSpeed:  0.036,
    // +10% from 1.00 — leaf extends further past inner layers
    scale: 1.10,
    weight: 0.85,  // light leaf
    // Soft sheen from moisture; not glossy enough to read as plastic.
    materialMetalness: 0,
    materialRoughness: 0.65,
    materialEnvMapIntensity: 0.75,
    doubleSide: true,
  },
  // 3 — TOMATO
  {
    path: "/models/burger-layers/tomato%20slice%203.glb",
    name: "Tomato",
    // Dropped 0.00 → −0.05 so tomato sits closer to onion ring (−0.12),
    // closing the gap from 0.12 → 0.07. Onion ring and below untouched.
    assembledY: -0.05,
    revealedY:   0.20,
    revealedOffset:  [ 0.03,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.10,  0.25,    0],
    idleYRotSpeed:  0.038,
    scale: 0.88,
    // Overall ~12 % rendered width reduction so the slice doesn't dominate
    // the burger silhouette (X/Z dial-down via the post-norm multiplier).
    visualScaleMultiplier: 0.88,
    // Strong Y squish — tomato is a thin slice and should read that way.
    // 0.65 of its post-norm height ≈ inside the spec's 0.55–0.75 range.
    scaleY: 0.65,
    weight: 0.95,  // medium slice
    // Wet slice — glossy but not glass. Low roughness + high env reflection.
    materialMetalness: 0.02,
    materialRoughness: 0.35,
    materialEnvMapIntensity: 0.95,
    doubleSide: true,
  },
  // 4 — ONION RING (onion ring2.glb still vertical — testing candidate rotations)
  {
    path: "/models/burger-layers/onion%20ring2.glb",
    name: "Onion Ring",
    assembledY: -0.12,
    revealedY:  -0.20,
    revealedOffset:  [ 0.00,  0.00],
    baseRotation:    [ 0,            0, 0],
    // Driven by ONION_ORIENTATION_TEST above — change that to swap candidates.
    modelRotationCorrection: ORIENTATION_TESTS[ONION_ORIENTATION_TEST],
    revealedRotation:[ 0.04,  0.15,    0],
    idleYRotSpeed:  0.043,
    scale: 0.85,
    weight: 0.90,  // light ring
    // Light fried surface — slight sheen, not glossy.
    materialMetalness: 0,
    materialRoughness: 0.55,
    materialEnvMapIntensity: 0.80,
    doubleSide: true,
  },
  // 5 — CHEESE
  {
    path: "/models/burger-layers/Cheese%20(1).glb",
    name: "Cheese",
    // Raised −0.26 → −0.22 so cheese no longer hugs the patty — the patty's
    // top edge now reads clearly under cheese instead of being hidden.
    assembledY: -0.22,
    revealedY:  -0.60,
    revealedOffset:  [ 0.00,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.10,  0.18,    0],
    idleYRotSpeed:  0.031,
    // Reduced 1.02 → 0.90 → 0.78 (−13 % from prior). Cheese tucks well inside
    // the burger body now — it no longer dominates the silhouette; patty,
    // buns, and lettuce all extend visibly past the cheese edges.
    scale: 0.78,
    weight: 0.95,  // medium slice
    // Soft matte cheese with slight sheen — never metallic.
    materialMetalness: 0,
    materialRoughness: 0.50,
    materialEnvMapIntensity: 0.65,
    doubleSide: true,
  },
  // 6 — PATTY  (near-flat tilt; preserve grilled top via subtle X tilt + camera angle)
  {
    path: "/models/burger-layers/Patty%20(1).glb",
    name: "Patty",
    assembledY: -0.28,
    revealedY:  -1.00,
    revealedOffset:  [ 0.00,  0.00],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.02,  0.08,    0],
    idleYRotSpeed:  0.048,
    // `scale` was already pulled back to top-bun parity, but the patty GLB's
    // bbox is wider on the horizontal axis than the top-bun's, so even at
    // the same `scale` the rendered diameter is larger. `visualScaleMultiplier`
    // applies AFTER the bbox normalisation to dial the *rendered* width.
    scale: 0.92,
    visualScaleMultiplier: 0.85,  // ~15 % rendered width reduction
    weight: 1.15,  // heaviest — patty as the anchor
    // Rough grilled surface — high roughness, modest env so it doesn't go dead black.
    materialMetalness: 0,
    materialRoughness: 0.78,
    materialEnvMapIntensity: 0.55,
    doubleSide: false,
  },
  // 7 — BOTTOM BUN
  {
    path: "/models/burger-layers/bottom.glb",
    name: "Bottom Bun",
    assembledY: -0.40,
    revealedY:  -1.40,
    revealedOffset:  [ 0.00, -0.03],
    baseRotation:    [ 0,            0, 0],
    revealedRotation:[ 0.08,         0, 0],
    idleYRotSpeed:  0.038,
    // bottom.glb's authored bbox is wider on the horizontal axis than
    // top-bun.glb's, so even at parity `scale` it renders larger. Use the
    // post-normalisation multiplier to dial rendered diameter to match the
    // top-bun reference.
    scale: 0.98,
    visualScaleMultiplier: 0.88,  // ~12 % rendered width reduction
    // Y-only squish — bottom bun is too tall relative to its width; flatten
    // to 0.72 of its post-norm height. Width/depth stay at the multiplier
    // above so it still supports the stack at the correct diameter.
    scaleY: 0.72,
    weight: 1.10,  // heavy bun
    // Matte bread — matches top bun.
    materialMetalness: 0,
    materialRoughness: 0.88,
    materialEnvMapIntensity: 0.50,
    doubleSide: false,
  },
];

function FoodModel({
  path,
  targetSize = 0.92,
  positionOffset = [0, 0, 0] as [number, number, number],
  rotationOffset = [0, 0, 0] as [number, number, number],
  materialMetalness,
  materialRoughness,
  materialEnvMapIntensity,
}: {
  path: string;
  targetSize?: number;
  positionOffset?: [number, number, number];
  rotationOffset?: [number, number, number];
  // Optional per-item PBR overrides. Defaults preserve the existing
  // food-baseline (roughness floor 0.55, original textures untouched).
  materialMetalness?: number;
  materialRoughness?: number;
  materialEnvMapIntensity?: number;
}) {
  const { scene } = useGLTF(path);

  const { center, normalizedScale } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const c = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    return { center: c, normalizedScale: maxDim > 0 ? targetSize / maxDim : 1 };
  }, [scene, targetSize]);

  useEffect(() => {
    scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mats = Array.isArray(obj.material)
        ? (obj.material as THREE.MeshStandardMaterial[])
        : [obj.material as THREE.MeshStandardMaterial];
      mats.forEach((m) => {
        if (!m.isMeshStandardMaterial) return;
        m.transparent = true;
        // Per-item PBR overrides if supplied; otherwise the legacy floor of 0.55.
        m.roughness = materialRoughness ?? Math.max(m.roughness, 0.55);
        if (materialMetalness !== undefined) m.metalness = materialMetalness;
        if (materialEnvMapIntensity !== undefined) {
          m.envMapIntensity = materialEnvMapIntensity;
        }
        m.needsUpdate = true;
      });
    });
  }, [scene, materialMetalness, materialRoughness, materialEnvMapIntensity]);

  return (
    <group scale={normalizedScale} position={positionOffset} rotation={rotationOffset}>
      <group position={[-center.x, -center.y, -center.z]}>
        <primitive object={scene} />
      </group>
    </group>
  );
}
// Uniform scale applied to the whole burger group so it fits the viewport.
// Bumped 0.72 → 0.78 (+8.3 %) — the burger product as a whole reads larger
// in both carousel and inspect without individually enlarging ingredients.
// Conservative end of the spec's +8–15 % range so the revealed stack still
// fits the inspect framing without clipping at extremes.
// Reveal-only framing: the closed burger keeps the larger product scale, while
// the opened layer stack backs away and shrinks enough to stay fully in frame.

// Loads a single burger layer GLB, normalizes it, and applies safe material defaults.
// Transparency is OFF by default — the parent BurgerExplodedView toggles it only while
// a layer is fading. This prevents see-through artifacts on fully-revealed ingredients.
function BurgerLayerGLB({
  path,
  doubleSide,
  modelRotationCorrection,
  modelPositionCorrection,
  materialMetalness,
  materialRoughness,
  materialEnvMapIntensity,
}: {
  path: string;
  doubleSide: boolean;
  modelRotationCorrection?: readonly [number, number, number];
  modelPositionCorrection?: readonly [number, number, number];
  materialMetalness?: number;
  materialRoughness?: number;
  materialEnvMapIntensity?: number;
}) {
  const { scene } = useGLTF(path);

  // Box3 must be measured AFTER applying any rotation correction so the
  // center offset still centers the visible geometry (a rotated leaf has a
  // different bounding center than its unrotated form).
  const { center, normalizedScale } = useMemo(() => {
    // Apply the correction temporarily on a math object — we don't mutate
    // the scene itself; the correction is applied via a wrapper <group> in
    // the JSX below. But we DO want the bounds to be measured at world
    // orientation so centering is correct.
    const tmp = scene.clone(true);
    if (modelRotationCorrection) {
      tmp.rotation.set(...modelRotationCorrection);
      tmp.updateMatrixWorld(true);
    }
    const box = new THREE.Box3().setFromObject(tmp);
    const size = box.getSize(new THREE.Vector3());
    const c = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const norm = maxDim > 0 ? 0.85 / maxDim : 1;
    // Diagnostic: print the GLB's raw bbox + its normalised post-scale width
    // (X axis) so we can confirm which layers come out wider than expected
    // after the maxDim normalisation. Logs ONCE per GLB on mount.
    if (DEBUG_BURGER_BOUNDS && typeof window !== "undefined") {
      const widthAfterNorm = size.x * norm;
      const heightAfterNorm = size.y * norm;
      const depthAfterNorm = size.z * norm;
      console.info(
        `[BurgerLayerGLB] ${path}  raw=(${size.x.toFixed(2)}, ${size.y.toFixed(2)}, ${size.z.toFixed(2)})  ` +
        `maxDim=${maxDim.toFixed(2)}  norm=${norm.toFixed(3)}  ` +
        `post-norm width=${widthAfterNorm.toFixed(2)}  height=${heightAfterNorm.toFixed(2)}  depth=${depthAfterNorm.toFixed(2)}`
      );
    }
    return { center: c, normalizedScale: norm };
  }, [scene, modelRotationCorrection, path]);

  useEffect(() => {
    scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const mats = Array.isArray(obj.material)
        ? (obj.material as THREE.MeshStandardMaterial[])
        : [obj.material as THREE.MeshStandardMaterial];
      mats.forEach((m) => {
        if (!m.isMeshStandardMaterial) return;
        // Default to fully opaque, solid rendering. Parent toggles transparency
        // only during fade transitions to avoid z-sort / see-through artifacts.
        m.transparent = false;
        m.depthWrite = true;
        m.opacity = 1;
        // PBR tuning: prefer the per-layer override when supplied; otherwise
        // fall back to a safe baseline that compensates for Tripo GLBs
        // (which omit metallicFactor / roughnessFactor and would default to
        // metalness=1 / roughness=1 — dead black without an env map).
        m.metalness = materialMetalness ?? 0;
        m.roughness = materialRoughness ?? Math.min(m.roughness, 0.88);
        m.envMapIntensity = materialEnvMapIntensity ?? 0.72;
        // DoubleSide only for thin ingredients where the back face would otherwise be culled.
        m.side = doubleSide ? THREE.DoubleSide : THREE.FrontSide;
        m.needsUpdate = true;
      });
    });
  }, [
    scene,
    doubleSide,
    materialMetalness,
    materialRoughness,
    materialEnvMapIntensity,
  ]);

  return (
    <group scale={normalizedScale}>
      <group position={[-center.x, -center.y, -center.z]}>
        {/* Model-level rotation correction. Applied INSIDE the centering
            wrapper so the corrected geometry stays centered on origin and
            the outer animation rotation composes on top of this correction. */}
        <group
          position={modelPositionCorrection ?? [0, 0, 0]}
          rotation={modelRotationCorrection ?? [0, 0, 0]}
        >
          <primitive object={scene} />
        </group>
      </group>
    </group>
  );
}

function BurgerExplodedView({
  active,
  layout,
}: {
  active: boolean;
  layout: SceneLayout;
}) {
  // Single progress: 0 = assembled (looks like one burger), 1 = fully spread.
  const progressRef = useRef(0);
  const stackRef = useRef<THREE.Group>(null);
  const settledClosedRef = useRef(false);

  // Callback-ref array — one slot per BURGER_LAYERS entry. Scales to any
  // number of layers without per-layer useRef declarations.
  const groupRefs = useRef<(THREE.Group | null)[]>(
    new Array(BURGER_LAYERS.length).fill(null)
  );
  const pivotRefs = useRef<(THREE.Group | null)[]>(
    new Array(BURGER_LAYERS.length).fill(null)
  );
  const layerRot = useRef<number[]>(new Array(BURGER_LAYERS.length).fill(0));
  const closedLayoutInitializedRef = useRef(false);

  useFrame(({ clock }, delta) => {
    if (
      !active &&
      closedLayoutInitializedRef.current &&
      progressRef.current < 0.003
    ) {
      progressRef.current = 0;
      settledClosedRef.current = true;
      return;
    }

    settledClosedRef.current = false;
    if (active) {
      closedLayoutInitializedRef.current = false;
    }

    // Asymmetric lerp dynamics:
    //   reveal     → 2.3 (slow, elegant unfurl)
    //   reassemble → 3.2 (faster decisive pull, "magnetic close")
    // Faster reassemble against the smoothBand easing curve reads as the
    // layers being softly drawn back into the assembled burger.
    const lerpRate = active ? 2.3 : 3.2;
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * lerpRate)
    );
    const progress = progressRef.current;
    const revealFrameP = smoothStep(progress);
    const t = clock.getElapsedTime();

    if (stackRef.current) {
      const stackScale = THREE.MathUtils.lerp(
        EXPLODED_STACK_SCALE,
        layout.revealScale,
        revealFrameP
      );
      stackRef.current.scale.setScalar(stackScale);
      stackRef.current.position.x = THREE.MathUtils.lerp(
        0,
        layout.revealX,
        revealFrameP
      );
      stackRef.current.position.y = THREE.MathUtils.lerp(
        0,
        layout.revealY,
        revealFrameP
      );
      stackRef.current.position.z = THREE.MathUtils.lerp(
        0,
        layout.revealZ,
        revealFrameP
      );
    }

    if (!active && closedLayoutInitializedRef.current && progress < 0.003) {
      progressRef.current = 0;
      settledClosedRef.current = true;
      return;
    }

    BURGER_LAYERS.forEach((layer, i) => {
      const group = groupRefs.current[i];
      const pivot = pivotRefs.current[i];
      if (!group) return;

      // Per-layer staggered spread band keyed by stack position from top.
      // Index already orders top → bottom, so idx 0 (top-bun) lifts first
      // on reveal and lands last on reassemble.
      // Weight gives heavier ingredients a wider band (slower) and lighter
      // ingredients a narrower band (faster), creating subtle physical weight.
      const weight = layer.weight ?? 1.0;
      const bandStart = i * 0.030;
      const bandEnd   = 0.92 + (weight - 1.0) * 0.10;
      // smoothBand wraps smoothStep — a smooth cubic Hermite curve.
      // Combined with the exponential lerp on `progress`, this produces a
      // cinematic ease-in-out without snapping at either end.
      const spreadP = smoothBand(progress, bandStart, bandEnd);

      // Y position — lerp assembled → revealed by spread progress.
      group.position.y =
        THREE.MathUtils.lerp(layer.assembledY, layer.revealedY, spreadP) +
        Math.sin(t * 0.26 + i * 0.80) * 0.005 * spreadP;

      // X/Z drift — zero when assembled; only applies during spread.
      group.position.x = THREE.MathUtils.lerp(0, layer.revealedOffset[0], spreadP);
      group.position.z = THREE.MathUtils.lerp(0, layer.revealedOffset[1], spreadP);

      // Per-layer scale.
      // Per-layer uniform scale. Three factors compose:
      //   1. layer.scale                — author intent
      //   2. layer.visualScaleMultiplier — post-normalization width fix
      //   3. layer.scaleY (optional)    — Y-only squish for tall 3D shapes
      // Multiplier (2) lets us dial individual GLBs to a target rendered
      // width without changing the conceptual `scale` value.
      const vm = layer.visualScaleMultiplier ?? 1;
      const s = layer.scale * vm;
      group.scale.set(s, s * (layer.scaleY ?? 1), s);

      // Base rotation always applied (corrects GLB orientation — e.g. lettuce
      // ships standing vertically and needs +π/2 X to lay flat).
      // Idle Y spin and reveal tilt are both gated by spreadP so assembled
      // layers stack flat with no rotation drift.
      layerRot.current[i] += delta * layer.idleYRotSpeed * spreadP;
      const [bx, byBase, bz] = layer.baseRotation;
      const [rx, ryBase, rz] = layer.revealedRotation;
      group.rotation.set(
        bx + rx * spreadP,
        byBase + ryBase * spreadP,
        bz + rz * spreadP
      );

      if (pivot) {
        pivot.rotation.y = layerRot.current[i];
      }
    });

    settledClosedRef.current = !active && progress < 0.001;
    closedLayoutInitializedRef.current = !active && progress < 0.003;
  });

  return (
    <group>
      {/* Uniform framing scale — keeps stack inside viewport without camera change */}
      <group ref={stackRef} scale={EXPLODED_STACK_SCALE}>
        {BURGER_LAYERS.map((layer, i) => (
          <group
            key={layer.path}
            ref={(el) => {
              groupRefs.current[i] = el;
            }}
          >
            <group
              ref={(el) => {
                pivotRefs.current[i] = el;
              }}
            >
              <BurgerLayerGLB
                path={layer.path}
                doubleSide={layer.doubleSide}
                modelRotationCorrection={layer.modelRotationCorrection}
                modelPositionCorrection={layer.modelPositionCorrection}
                materialMetalness={layer.materialMetalness}
                materialRoughness={layer.materialRoughness}
                materialEnvMapIntensity={layer.materialEnvMapIntensity}
              />
            </group>
          </group>
        ))}
      </group>
    </group>
  );
}

function MenuBook({ open }: { open: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const coverRef = useRef<THREE.Group>(null);
  const bookLightRef = useRef<THREE.PointLight>(null);
  const bookMaterialsRef = useRef<
    Array<{ material: THREE.MeshStandardMaterial; baseOpacity: number }>
  >([]);
  const progressRef = useRef(0);
  const settledOpenRef = useRef(false);

  useEffect(() => {
    bookMaterialsRef.current = [];
    groupRef.current?.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      const mat = mesh.material as THREE.MeshStandardMaterial | undefined;
      if (mat && "opacity" in mat) {
        bookMaterialsRef.current.push({
          material: mat,
          baseOpacity: mat.opacity,
        });
      }
    });
  }, []);

  useFrame(({ clock }, delta) => {
    if (settledOpenRef.current && open && progressRef.current > 0.999) {
      if (groupRef.current) groupRef.current.visible = false;
      return;
    }

    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      open ? 1 : 0,
      1 - Math.exp(-delta * 1.55)
    );
    const p = smoothStep(progressRef.current);
    const bookVisible = 1 - smoothStep(Math.min(p * 2.2, 1));
    // Idle amount fades from 1 (book at rest, fully visible) to 0 as it begins
    // its transition, so the breathing motion never fights the opening anim.
    const idleAmount = 1 - p;
    const t = clock.getElapsedTime();

    if (groupRef.current) {
      groupRef.current.visible = true;
      groupRef.current.scale.setScalar(THREE.MathUtils.lerp(1.0, 0.42, p));
      // Subtle Y bob (±0.035) and slow Y rotation breathing (±0.05 rad), both
      // gated by idleAmount so they ease out as the book starts collapsing.
      groupRef.current.position.y =
        THREE.MathUtils.lerp(0, -0.55, p) +
        Math.sin(t * 0.55) * 0.035 * idleAmount;
      groupRef.current.position.x = Math.sin(t * 0.38) * 0.020 * idleAmount;
      groupRef.current.rotation.x = THREE.MathUtils.lerp(0, 0.14, p);
      groupRef.current.rotation.y = Math.sin(t * 0.42) * 0.050 * idleAmount;

      bookMaterialsRef.current.forEach(({ material, baseOpacity }) => {
        material.transparent = true;
        material.opacity = baseOpacity * bookVisible;
      });
    }

    if (coverRef.current) {
      coverRef.current.rotation.y = THREE.MathUtils.lerp(0, Math.PI * 0.52, p);
    }

    if (bookLightRef.current) {
      // Breathing light pulse (±12 % around base) only while idle; the lerp to
      // the static 0.55 multiplier resumes once the book is fully open/closed.
      const breathe = 0.88 + Math.sin(t * 1.1) * 0.12 * idleAmount;
      bookLightRef.current.intensity = bookVisible * 0.55 * breathe;
    }

    settledOpenRef.current = open && progressRef.current > 0.999;
  });

  return (
    <group ref={groupRef}>
      <pointLight
        ref={bookLightRef}
        color="#fff6e0"
        distance={6}
        intensity={0}
        position={[0, 2.8, 2.2]}
      />

      {/* Soft shadow beneath book */}
      <mesh position={[0, -0.94, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.88, 36]} />
        <meshStandardMaterial
          color="#c0a882"
          transparent
          opacity={0.20}
          roughness={1}
          depthWrite={false}
        />
      </mesh>

      {/* Back cover */}
      <mesh position={[0, 0, -0.078]}>
        <boxGeometry args={[1.38, 1.86, 0.058]} />
        <meshStandardMaterial color="#ede8df" metalness={0.08} roughness={0.72} transparent opacity={1} />
      </mesh>

      {/* Pages block */}
      <mesh position={[0.01, 0, 0]}>
        <boxGeometry args={[1.30, 1.78, 0.12]} />
        <meshStandardMaterial color="#f8f5f0" metalness={0.04} roughness={0.84} transparent opacity={1} />
      </mesh>

      {/* Spine */}
      <mesh position={[-0.695, 0, 0]}>
        <boxGeometry args={[0.042, 1.86, 0.26]} />
        <meshStandardMaterial color="#c4a070" metalness={0.22} roughness={0.54} transparent opacity={1} />
      </mesh>

      {/* Front cover — group pivoted at spine edge for page-opening motion */}
      <group ref={coverRef} position={[-0.695, 0, 0.078]}>
        <mesh position={[0.695, 0, 0]}>
          <boxGeometry args={[1.38, 1.86, 0.058]} />
          <meshStandardMaterial color="#e6ddd0" metalness={0.10} roughness={0.66} transparent opacity={1} />
        </mesh>
        {/* Top gold rule */}
        <mesh position={[0.695, 0.70, 0.030]}>
          <boxGeometry args={[0.94, 0.007, 0.006]} />
          <meshStandardMaterial
            color="#c8a050"
            emissive="#9a7420"
            emissiveIntensity={0.45}
            metalness={0.65}
            roughness={0.28}
            transparent
            opacity={1}
          />
        </mesh>
        {/* Bottom gold rule */}
        <mesh position={[0.695, -0.70, 0.030]}>
          <boxGeometry args={[0.94, 0.007, 0.006]} />
          <meshStandardMaterial
            color="#c8a050"
            emissive="#9a7420"
            emissiveIntensity={0.45}
            metalness={0.65}
            roughness={0.28}
            transparent
            opacity={1}
          />
        </mesh>
        {/* Menu title plate */}
        <mesh position={[0.695, 0.12, 0.030]}>
          <boxGeometry args={[0.76, 0.038, 0.006]} />
          <meshStandardMaterial
            color="#d4b068"
            emissive="#a07828"
            emissiveIntensity={0.28}
            metalness={0.55}
            roughness={0.32}
            transparent
            opacity={1}
          />
        </mesh>
        {/* Subtitle plate */}
        <mesh position={[0.695, -0.04, 0.030]}>
          <boxGeometry args={[0.52, 0.018, 0.006]} />
          <meshStandardMaterial
            color="#c4a058"
            emissive="#906820"
            emissiveIntensity={0.22}
            metalness={0.50}
            roughness={0.36}
            transparent
            opacity={1}
          />
        </mesh>
      </group>
    </group>
  );
}

function AmbientCameraBreathing({
  inspectMode,
  lastInteractionRef,
  inspectDragRef,
}: {
  inspectMode: boolean;
  lastInteractionRef: MutableRefObject<number>;
  inspectDragRef: MutableRefObject<{ active: boolean; lastX: number; pointerId: number }>;
}) {
  const { camera } = useThree();
  const basePositionRef = useRef(new THREE.Vector3());
  const initializedRef = useRef(false);
  const intensityRef = useRef(0);

  useFrame(({ clock }, delta) => {
    if (!initializedRef.current) {
      basePositionRef.current.copy(camera.position);
      initializedRef.current = true;
    }

    const recentlyInteracted = Date.now() - lastInteractionRef.current < 1600;
    const targetIntensity = inspectDragRef.current.active
      ? 0
      : inspectMode
        ? 0.16
        : recentlyInteracted
          ? 0.32
          : 1;

    intensityRef.current = THREE.MathUtils.damp(
      intensityRef.current,
      targetIntensity,
      0.85,
      delta
    );

    const i = intensityRef.current;
    const t = clock.getElapsedTime();

    camera.position.set(
      basePositionRef.current.x + Math.sin(t * 0.19) * 0.010 * i,
      basePositionRef.current.y + Math.sin(t * 0.14 + 1.2) * 0.006 * i,
      basePositionRef.current.z + Math.sin(t * 0.11 + 2.1) * 0.014 * i
    );
  });

  return null;
}

function AmbientLightingBreathing({
  inspectMode,
  lastInteractionRef,
  inspectDragRef,
}: {
  inspectMode: boolean;
  lastInteractionRef: MutableRefObject<number>;
  inspectDragRef: MutableRefObject<{ active: boolean; lastX: number; pointerId: number }>;
}) {
  const ambientRef = useRef<THREE.AmbientLight>(null);
  const directionalRef = useRef<THREE.DirectionalLight>(null);
  const lowPointRef = useRef<THREE.PointLight>(null);
  const highPointRef = useRef<THREE.PointLight>(null);
  const intensityRef = useRef(0);

  useFrame(({ clock }, delta) => {
    const recentlyInteracted = Date.now() - lastInteractionRef.current < 1800;
    const targetIntensity = inspectDragRef.current.active || inspectMode
      ? 0
      : recentlyInteracted
        ? 0.18
        : 1;

    intensityRef.current = THREE.MathUtils.damp(
      intensityRef.current,
      targetIntensity,
      0.7,
      delta
    );

    const breathe = Math.sin(clock.getElapsedTime() * 0.16) * 0.006 * intensityRef.current;

    if (ambientRef.current) ambientRef.current.intensity = 0.92 * (1 + breathe);
    if (directionalRef.current) directionalRef.current.intensity = 0.88 * (1 + breathe * 0.8);
    if (lowPointRef.current) lowPointRef.current.intensity = 0.22 * (1 + breathe * 1.15);
    if (highPointRef.current) highPointRef.current.intensity = 0.14 * (1 + breathe);
  });

  return (
    <>
      <ambientLight ref={ambientRef} intensity={0.92} color="#fff8f0" />
      <directionalLight ref={directionalRef} position={[4, 5, 3]} intensity={0.88} color="#fff8f0" />
      <pointLight ref={lowPointRef} position={[-4, -2, 3]} intensity={0.22} color="#ffecd0" />
      <pointLight ref={highPointRef} position={[0, 3, -2]} intensity={0.14} color="#ffe8c8" />
    </>
  );
}

function SpatialMenuCarousel({
  exploded,
  activePartIndex,
  inspectMode,
  inspectRotationRef,
  burgerExploded,
  layout,
}: {
  exploded: boolean;
  activePartIndex: number;
  inspectMode: boolean;
  inspectRotationRef: MutableRefObject<{ x: number; y: number }>;
  burgerExploded: boolean;
  layout: SceneLayout;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const progressRef = useRef(0);
  const n = CAROUSEL_PARTS.length;

  useFrame(({ clock }, delta) => {
    // Faster lerp rate (1.4 → 2.4) tightens the entrance to ~1.1–1.3 s of
    // perceptual motion, matching the "Apple cinematic" entrance window
    // while staying smooth via the per-item phase delays + smoothStep below.
    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      exploded ? 1 : 0,
      1 - Math.exp(-delta * 2.4)
    );

    if (!groupRef.current) return;

    const p = smoothStep(progressRef.current);
    const t = clock.getElapsedTime();

    groupRef.current.rotation.y += delta * 0.08 * (1 - p);
    // Sit the carousel below camera eye-line so items feel like they're on a stage.
    groupRef.current.position.y = 0.0 + Math.sin(t * 0.38) * 0.025 * (1 - p);
  });

  return (
    <group ref={groupRef} scale={0.88}>

      {/* ── Item 0: Signature Burger Combo ── */}
      <Part
        partIndex={0}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        layout={layout}
        basePosition={[0, 0, 0]}
        midPosition={[0, 0.92, 1.05]}
        explodedPosition={[0, 0.18, 0]}
        // Bumped 1.22 → 1.34 for carousel parity with the other premium items
        // (1.34–1.48 band). Burger has an internal EXPLODED_STACK_SCALE = 0.78
        // wrapper so its perceived size was running smaller than peers — this
        // brings the hero in line with the rest of the showroom.
        focusScale={1.34}
        secondaryScale={0.66}
        inspectZFocus={1.7}
        // Inspect multiplier 1.54 → 1.74 (+13 %) — hero presence in inspect.
        inspectScaleMultiplier={1.74}
        // Burger is the hero — arrives LAST on entrance (settles after the
        // supporting cast lands), LEAVES FIRST on return (Apple ta-da feel).
        explodeDelay={0.28}
        assembleDelay={0.00}
        selfRotationAmount={0.12}
        freezeParentRotation={burgerExploded && inspectMode && activePartIndex === 0}
        motionSeed={1}
      >
        <BurgerExplodedView
          active={burgerExploded && inspectMode && activePartIndex === 0}
          layout={layout}
        />
      </Part>

      {/* ── Item 1: Louisiana Po' Boy ── */}
      <Part
        partIndex={1}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        layout={layout}
        basePosition={[0, 0, 0]}
        midPosition={[0.14, 0.88, 0.92]}
        explodedPosition={[0, 0.2, 0]}
        focusScale={1.38}
        secondaryScale={0.66}
        inspectZFocus={1.6}
        // Inspect multiplier 1.55 → 1.75 (+13 %).
        inspectScaleMultiplier={1.75}
        // Po' Boy — earlier in entrance, lingers later on return.
        explodeDelay={0.07}
        assembleDelay={0.21}
        explodedRotation={[0, -1.13, 0]}
        selfRotationAmount={0.18}
        motionSeed={4}
      >
        <FoodModel
          path="/models/poboy.glb"
          targetSize={1.06}
          // Po' Boy: mostly matte breading and roll, with restrained highlights
          // on sauce/vegetables so the sandwich stays premium rather than glossy.
          materialMetalness={0}
          materialRoughness={0.68}
          materialEnvMapIntensity={0.62}
        />
      </Part>

      {/* ── Item 2: Chef's Dessert Combo ── */}
      <Part
        partIndex={2}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        layout={layout}
        basePosition={[0, 0, 0]}
        midPosition={[0, 0.84, 1.02]}
        explodedPosition={[0, 0.16, 0]}
        // Carousel focus 1.28 → 1.42 (+11 %).
        focusScale={1.42}
        secondaryScale={0.66}
        // Inspect framing pulled slightly forward (1.40 → 1.70) and scaled up
        // (1.96 → 2.21, +13 %) so the layered dessert reads as a luxury hero.
        inspectZFocus={1.70}
        inspectScaleMultiplier={2.21}
        // Dessert — first to arrive on entrance, last to retreat on return.
        explodeDelay={0.00}
        assembleDelay={0.28}
        selfRotationAmount={0.2}
        motionSeed={5}
      >
        <FoodModel
          path="/models/dessert.glb"
          targetSize={0.90}
          rotationOffset={[0.06, 0, 0]}
          // Dessert: cream/pastry — slightly glossy cream-top sheen with the
          // env-map kicking in subtly along the topmost layer. No metalness.
          materialMetalness={0}
          materialRoughness={0.45}
          materialEnvMapIntensity={0.80}
        />
      </Part>

      {/* ── Item 3: Crispy Fried Chicken ── */}
      <Part
        partIndex={3}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        layout={layout}
        basePosition={[0, 0, 0]}
        midPosition={[0.08, 0.86, 1.00]}
        explodedPosition={[0, 0.18, 0]}
        // Carousel focus 1.26 → 1.40 (+11 %).
        focusScale={1.40}
        secondaryScale={0.65}
        // Inspect multiplier 1.72 → 1.94 (+13 %).
        inspectZFocus={1.55}
        inspectScaleMultiplier={1.94}
        // Slots between coffee (0.07/0.21) and dessert (0.00/0.28) in the
        // entrance stagger — arrives early-middle, leaves late-middle.
        explodeDelay={0.035}
        assembleDelay={0.245}
        selfRotationAmount={0.17}
        motionSeed={6}
      >
        <FoodModel
          path="/models/fried%20chicken.glb"
          targetSize={0.95}
          rotationOffset={[0.10, 0, 0]}
          // Crispy fried: roughness sits just over the food baseline so the
          // breaded crust reads as textured-not-greasy. Modest env so the
          // spotlight catches highlights along the crust ridges.
          materialMetalness={0}
          materialRoughness={0.58}
          materialEnvMapIntensity={0.75}
        />
      </Part>

      {/* ── Item 4: Louisiana Crawfish — premium coastal showcase ── */}
      <Part
        partIndex={4}
        progressRef={progressRef}
        activePartIndex={activePartIndex}
        totalParts={n}
        carouselEnabled={exploded}
        inspectMode={inspectMode}
        inspectRotationRef={inspectRotationRef}
        layout={layout}
        basePosition={[0, 0, 0]}
        midPosition={[-0.06, 0.84, 1.04]}
        // Slightly elevated rest position so the crawfish sits proud on stage.
        explodedPosition={[0, 0.22, 0]}
        // Premium scale — matches the oyster / dessert / chicken band.
        focusScale={1.42}
        secondaryScale={0.66}
        // Hero framing in inspect: pulled forward, scaled up.
        // Inspect multiplier 1.80 → 2.03 (+13 %).
        inspectZFocus={1.65}
        inspectScaleMultiplier={2.03}
        // Slots between coffee (0.07 / 0.21) and chicken (0.035 / 0.245)
        // in the entrance stagger — arrives mid, leaves mid-late.
        explodeDelay={0.105}
        assembleDelay={0.175}
        selfRotationAmount={0.18}
        motionSeed={7}
      >
        <FoodModel
          path="/models/crawfish.glb"
          // Substantial footprint — crawfish silhouette (body + claws) reads
          // larger than a sushi-roll or single shell.
          targetSize={1.00}
          // Forward tilt opens the shell top toward the spotlight cone.
          rotationOffset={[0.15, 0, 0]}
          // Wet-shell PBR: slight metalness for natural shellfish gloss,
          // moderate roughness so the cajun-dusted shell still reads as
          // textured (not chrome). High env-map response catches the warm
          // active spotlight as clean specular along the carapace.
          materialMetalness={0.03}
          materialRoughness={0.50}
          materialEnvMapIntensity={0.85}
        />
      </Part>

    </group>
  );
}

function AmbientParticles() {
  const pointsRef = useRef<THREE.Points>(null);
  const frameSkipRef = useRef(0);
  const deltaAccumulatorRef = useRef(0);

  const positions = useMemo(() => {
    const count = AMBIENT_PARTICLE_COUNT;
    const array = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      array[i * 3] = (seededUnit(i * 3 + 1) - 0.5) * 14;
      array[i * 3 + 1] = (seededUnit(i * 3 + 2) - 0.5) * 8;
      array[i * 3 + 2] = (seededUnit(i * 3 + 3) - 0.5) * 10;
    }

    return array;
  }, []);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;
    deltaAccumulatorRef.current += delta;
    frameSkipRef.current = (frameSkipRef.current + 1) % 6;
    if (frameSkipRef.current !== 0) return;
    pointsRef.current.rotation.y += deltaAccumulatorRef.current * 0.025;
    deltaAccumulatorRef.current = 0;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#a08868"
        size={0.013}
        transparent
        opacity={0.16}
        depthWrite={false}
      />
    </points>
  );
}

function AuraLogoParticles({ exploded }: { exploded: boolean }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const progressRef = useRef(0);
  const frameSkipRef = useRef(0);
  const deltaAccumulatorRef = useRef(0);
  const settledTargetRef = useRef<number | null>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const particles = useMemo(() => createLogoParticles(), []);

  useFrame(({ clock }, delta) => {
    if (!meshRef.current) return;
    const targetProgress = exploded ? 1 : 0;

    if (
      settledTargetRef.current === targetProgress &&
      Math.abs(progressRef.current - targetProgress) < 0.001
    ) {
      return;
    }

    deltaAccumulatorRef.current += delta;
    frameSkipRef.current = (frameSkipRef.current + 1) % 2;
    if (frameSkipRef.current !== 0) return;
    const effectiveDelta = deltaAccumulatorRef.current;
    deltaAccumulatorRef.current = 0;

    const mesh = meshRef.current;

    progressRef.current = THREE.MathUtils.lerp(
      progressRef.current,
      targetProgress,
      1 - Math.exp(-effectiveDelta * 2.35)
    );
    if (Math.abs(progressRef.current - targetProgress) < 0.001) {
      progressRef.current = targetProgress;
      settledTargetRef.current = targetProgress;
    } else {
      settledTargetRef.current = null;
    }

    const t = clock.getElapsedTime();
    const p = smoothStep(progressRef.current);
    const scale = THREE.MathUtils.lerp(0.042, 0.024, p);

    particles.forEach((particle, index) => {
      const idle = Math.sin(t * 0.42 + particle.seed) * 0.008 * (1 - p);
      const shockDrift = Math.sin(t * 0.32 + particle.seed * 0.7) * 0.024 * p;

      dummy.position.set(
        THREE.MathUtils.lerp(particle.home.x, particle.scatter.x, p) +
          shockDrift,
        THREE.MathUtils.lerp(particle.home.y, particle.scatter.y, p) +
          idle,
        THREE.MathUtils.lerp(particle.home.z, particle.scatter.z, p)
      );
      dummy.rotation.set(
        p * (0.35 + seededUnit(particle.seed * 2) * 0.65),
        p * (0.45 + seededUnit(particle.seed * 3) * 0.8),
        p * 0.18
      );
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
    const material = mesh.material as THREE.MeshStandardMaterial;
    material.opacity = THREE.MathUtils.lerp(0.38, 0.10, p);
    material.emissiveIntensity = THREE.MathUtils.lerp(0.06, 0.02, p);
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, particles.length]}
      frustumCulled={false}
    >
      <boxGeometry args={[1, 1, 0.18]} />
      <meshStandardMaterial
        color="#b07840"
        emissive="#6a4818"
        emissiveIntensity={0.06}
        metalness={0.28}
        opacity={0.38}
        roughness={0.58}
        transparent
      />
    </instancedMesh>
  );
}

function InspectSceneLighting({ inspectMode }: { inspectMode: boolean }) {
  const keyLightRef = useRef<THREE.DirectionalLight>(null);
  const rimLightRef = useRef<THREE.PointLight>(null);
  const floorPoolRef = useRef<THREE.PointLight>(null);
  const blendRef = useRef(0);

  useFrame((_, delta) => {
    if (!inspectMode && blendRef.current === 0) return;
    if (!inspectMode && blendRef.current < 0.001) {
      blendRef.current = 0;
      setAnimatedLightIntensity(keyLightRef.current, 0);
      setAnimatedLightIntensity(rimLightRef.current, 0);
      setAnimatedLightIntensity(floorPoolRef.current, 0);
      return;
    }
    // Slow cinematic fade — lighting builds and withdraws gradually.
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      inspectMode ? 1 : 0,
      1 - Math.exp(-delta * 2.4)
    );

    // Slightly stronger so the hero item feels lifted from the background
    // when inspect engages (food-commercial focus emphasis).
    setAnimatedLightIntensity(keyLightRef.current, blendRef.current * 1.70);
    setAnimatedLightIntensity(rimLightRef.current, blendRef.current * 0.90);
    setAnimatedLightIntensity(floorPoolRef.current, blendRef.current * 0.58);
  });

  return (
    <>
      <directionalLight
        ref={keyLightRef}
        position={[-1.4, 5.2, 3.4]}
        color="#fff6e4"
        intensity={0}
        visible={false}
      />
      <pointLight
        ref={rimLightRef}
        position={[3.2, 2.0, -1.4]}
        color="#ffd070"
        distance={16}
        intensity={0}
        visible={false}
      />
      <pointLight
        ref={floorPoolRef}
        color="#ffcc88"
        position={[0, -1.4, 2.0]}
        distance={7}
        intensity={0}
        visible={false}
      />
    </>
  );
}

// Stationary stage spotlight at the front-center of the carousel. Items
// rotate into the cone as they reach slot 0; this is distinct from the
// per-Part `activeLightRef` spotlight that follows individual items.
// Active when the menu is open AND we are not in inspect mode — fades out
// cleanly as soon as inspect engages so the inspect lighting takes over.
function CarouselCenterSpotlight({ active }: { active: boolean }) {
  const mainRef = useRef<THREE.SpotLight>(null);
  const fillRef = useRef<THREE.PointLight>(null);
  const blendRef = useRef(0);

  useFrame((_, delta) => {
    if (!active && blendRef.current === 0) return;
    if (!active && blendRef.current < 0.001) {
      blendRef.current = 0;
      setAnimatedLightIntensity(mainRef.current, 0);
      setAnimatedLightIntensity(fillRef.current, 0);
      return;
    }
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 2.6)
    );
    const b = blendRef.current;
    // Main top/front cinematic spotlight on the centered item.
    setAnimatedLightIntensity(mainRef.current, b * 1.6);
    // Soft front-warm fill below — extremely subtle pool that lifts the food
    // underside without producing a visible disc on the stage.
    setAnimatedLightIntensity(fillRef.current, b * 0.32);
  });

  return (
    <>
      {/* Main top-front spotlight — wide cone, soft penumbra, warm amber.
          Sits above and forward of the active slot 0 dock position. Decay
          is restrained so the cone actually lands on the food at the stage. */}
      <spotLight
        ref={mainRef}
        position={[0, 4.2, 2.6]}
        // Default target is (0,0,0) — exactly where slot 0 sits, so the cone
        // points straight at the centered food item.
        color="#ffd8a8"
        angle={0.60}
        penumbra={0.78}
        distance={11}
        decay={0.9}
        intensity={0}
        visible={false}
      />
      {/* Subtle warm fill from below-front — lifts shadowed undersides on
          the centered item. Very tight distance keeps it from bleeding into
          the carousel's rear items. No visible disc, no fake floor light. */}
      <pointLight
        ref={fillRef}
        position={[0, -0.6, 2.4]}
        color="#ffd6a0"
        distance={3.8}
        decay={1.4}
        intensity={0}
        visible={false}
      />
    </>
  );
}

// Warm cinematic hero lighting for the assembled burger in inspect mode.
// Distinct from InspectSceneLighting (which serves all items generically).
// Six-light setup mirroring a food-commercial stage:
//   - cinemaKey: raking warm key from upper-left/front (Phase 1)
//   - sideFill:  soft warm fill from upper-right balances the key
//   - heroFill:  low front-warm fill catches bun underside / sesame
//   - bottomBounce: warm bounce simulating table reflection (Phase 3)
//   - backRim:   wide warm separation rim from the rear (Phase 2)
//   - edgeRim:   tighter cooler-warm rim catching bun crown edges (Phase 2)
function BurgerInspectLighting({ active }: { active: boolean }) {
  const cinemaKeyRef = useRef<THREE.DirectionalLight>(null);
  const sideFillRef  = useRef<THREE.PointLight>(null);
  const heroFillRef  = useRef<THREE.PointLight>(null);
  const bottomBounceRef = useRef<THREE.PointLight>(null);
  const backRimRef   = useRef<THREE.DirectionalLight>(null);
  const edgeRimRef   = useRef<THREE.DirectionalLight>(null);
  const blendRef     = useRef(0);

  useFrame((_, delta) => {
    if (!active && blendRef.current === 0) return;
    if (!active && blendRef.current < 0.001) {
      blendRef.current = 0;
      setAnimatedLightIntensity(cinemaKeyRef.current, 0);
      setAnimatedLightIntensity(sideFillRef.current, 0);
      setAnimatedLightIntensity(heroFillRef.current, 0);
      setAnimatedLightIntensity(bottomBounceRef.current, 0);
      setAnimatedLightIntensity(backRimRef.current, 0);
      setAnimatedLightIntensity(edgeRimRef.current, 0);
      return;
    }
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 2.2)
    );
    const b = blendRef.current;
    // Cinematic key: dominant light, warm, rakes the top of the stack.
    setAnimatedLightIntensity(cinemaKeyRef.current, b * 1.30);
    // Side fill: prevents the un-keyed side from going flat / dead.
    setAnimatedLightIntensity(sideFillRef.current, b * 0.55);
    // Hero front fill: catches sesame seeds + bun upper crust.
    setAnimatedLightIntensity(heroFillRef.current, b * 0.92);
    // Bottom bounce: simulated table reflection; lifts cheese / patty / bottom bun
    // without an orange glow puddle. Distance is short so it's localised to the
    // burger and doesn't bleed into the carousel rear.
    setAnimatedLightIntensity(bottomBounceRef.current, b * 0.55);
    // Back rim: separates silhouette from background.
    setAnimatedLightIntensity(backRimRef.current, b * 0.62);
    // Edge rim: catches the top of the bun crown.
    setAnimatedLightIntensity(edgeRimRef.current, b * 0.45);
  });

  return (
    <>
      {/* 1. Cinematic key — upper-left/front, warm amber */}
      <directionalLight
        ref={cinemaKeyRef}
        position={[-2.0, 4.6, 3.2]}
        color="#ffd6a0"
        intensity={0}
        visible={false}
      />
      {/* 2. Soft side fill — upper-right, gentle warm, fills the key shadow */}
      <pointLight
        ref={sideFillRef}
        position={[2.6, 2.4, 2.4]}
        color="#ffe0b0"
        distance={11}
        intensity={0}
        visible={false}
      />
      {/* 3. Hero front fill — low-front, warm, catches sesame + bun upper crust */}
      <pointLight
        ref={heroFillRef}
        position={[-1.0, 0.6, 4.0]}
        color="#ffcc80"
        distance={9}
        intensity={0}
        visible={false}
      />
      {/* 4. Warm bottom bounce — simulates table reflection lifting lower layers */}
      <pointLight
        ref={bottomBounceRef}
        position={[0, -1.6, 2.4]}
        color="#ffc890"
        distance={5.5}
        intensity={0}
        visible={false}
      />
      {/* 5. Back rim — wide warm separation from rear */}
      <directionalLight
        ref={backRimRef}
        position={[1.4, 2.8, -3.2]}
        color="#ffe4c0"
        intensity={0}
        visible={false}
      />
      {/* 6. Edge rim — tighter cooler-warm rim catching bun crown edges */}
      <directionalLight
        ref={edgeRimRef}
        position={[-1.6, 3.4, -2.6]}
        color="#fff0d8"
        intensity={0}
        visible={false}
      />
    </>
  );
}

// Dedicated warm lighting for the burger exploded ingredient showcase.
// Fades in independently of InspectSceneLighting to add extra depth when layers open.
function BurgerExplodedLighting({ active }: { active: boolean }) {
  const topKeyRef    = useRef<THREE.DirectionalLight>(null);
  const warmFillRef  = useRef<THREE.PointLight>(null);
  const underlightRef = useRef<THREE.PointLight>(null);
  const backRimRef   = useRef<THREE.DirectionalLight>(null);
  const ambientFillRef = useRef<THREE.HemisphereLight>(null);
  const blendRef     = useRef(0);

  useFrame((_, delta) => {
    if (!active && blendRef.current === 0) return;
    if (!active && blendRef.current < 0.001) {
      blendRef.current = 0;
      setAnimatedLightIntensity(topKeyRef.current, 0);
      setAnimatedLightIntensity(warmFillRef.current, 0);
      setAnimatedLightIntensity(underlightRef.current, 0);
      setAnimatedLightIntensity(backRimRef.current, 0);
      setAnimatedLightIntensity(ambientFillRef.current, 0);
      return;
    }
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 1.8)
    );
    setAnimatedLightIntensity(topKeyRef.current, blendRef.current * 1.2);
    setAnimatedLightIntensity(warmFillRef.current, blendRef.current * 0.80);
    setAnimatedLightIntensity(underlightRef.current, blendRef.current * 1.10);
    setAnimatedLightIntensity(backRimRef.current, blendRef.current * 0.40);
    setAnimatedLightIntensity(ambientFillRef.current, blendRef.current * 0.28);
  });

  return (
    <>
      {/* Warm overhead key — rakes across the stack from upper-left */}
      <directionalLight
        ref={topKeyRef}
        position={[-0.8, 5.2, 2.8]}
        color="#ffd89a"
        intensity={0}
        visible={false}
      />
      {/* Soft warm fill from the right */}
      <pointLight
        ref={warmFillRef}
        position={[2.2, 1.6, 2.0]}
        color="#ffb860"
        distance={11}
        intensity={0}
        visible={false}
      />
      {/* Warm underlight — rakes up from below to light the bottom bun underside */}
      <pointLight
        ref={underlightRef}
        position={[0, -1.8, 1.8]}
        color="#ffb870"
        distance={10}
        intensity={0}
        visible={false}
      />
      {/* Back separation rim — gives depth between layers */}
      <directionalLight
        ref={backRimRef}
        position={[1.2, 2.5, -3.0]}
        color="#ffe8c8"
        intensity={0}
        visible={false}
      />
      {/* Warm hemisphere fill — lifts shadowed underside without washing out warm look */}
      <hemisphereLight
        ref={ambientFillRef}
        args={["#fff1d8", "#3a2010", 0]}
        visible={false}
      />
    </>
  );
}

// Subtle food-commercial ambient details for the assembled burger:
//   - 10 wisps of warm steam rising slowly from above the bun
//   - 8 tiny crumb / sesame particles drifting near the upper burger area
// Group visibility is gated by `active`; all per-frame work skips when faded out.
// Atmospheric scene-wide depth is handled by the existing AmbientParticles (700
// background points) — we deliberately don't duplicate that here.
function BurgerAmbientFX({ active }: { active: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const blendRef = useRef(0);

  // Steam particle seed data — phase offset staggers each wisp's lifecycle so
  // they don't all fade in/out together; jitter randomises lateral position.
  const STEAM_COUNT = 10;
  const steamSeeds = useMemo(
    () =>
      Array.from({ length: STEAM_COUNT }, (_, i) => ({
        phaseOffset: seededUnit(i * 7 + 1),
        xJitter: (seededUnit(i * 7 + 2) - 0.5) * 0.45,
        zJitter: (seededUnit(i * 7 + 3) - 0.5) * 0.40,
        driftFreq: 0.4 + seededUnit(i * 7 + 4) * 0.5,
        driftAmp: 0.04 + seededUnit(i * 7 + 5) * 0.05,
        size: 0.026 + seededUnit(i * 7 + 6) * 0.022,
      })),
    []
  );

  // Crumb seed data — fixed-ish positions near the upper burger area
  // (bun/bacon level), with slow per-axis drift.
  const CRUMB_COUNT = 8;
  const crumbSeeds = useMemo(
    () =>
      Array.from({ length: CRUMB_COUNT }, (_, i) => ({
        baseX: (seededUnit(i * 11 + 1) - 0.5) * 0.85,
        baseY: 0.12 + seededUnit(i * 11 + 2) * 0.30,
        baseZ: (seededUnit(i * 11 + 3) - 0.5) * 0.55,
        driftFreq: 0.28 + seededUnit(i * 11 + 4) * 0.22,
        size: 0.008 + seededUnit(i * 11 + 5) * 0.010,
      })),
    []
  );

  const steamRefs = useRef<(THREE.Mesh | null)[]>([]);
  const crumbRefs = useRef<(THREE.Mesh | null)[]>([]);

  useFrame(({ clock }, delta) => {
    blendRef.current = THREE.MathUtils.lerp(
      blendRef.current,
      active ? 1 : 0,
      1 - Math.exp(-delta * 2.0)
    );
    const b = blendRef.current;
    if (!groupRef.current) return;
    // Hard-cull when faded out — skip all per-frame updates.
    groupRef.current.visible = b > 0.01;
    if (!groupRef.current.visible) return;

    const t = clock.getElapsedTime();

    // Steam: 4.0s loop. Particle rises from y≈0.4 to y≈1.7, fades in over
    // first 20% of the cycle, fades out over the last 30%.
    for (let i = 0; i < STEAM_COUNT; i++) {
      const mesh = steamRefs.current[i];
      if (!mesh) continue;
      const s = steamSeeds[i];
      const cycle = ((t * 0.25) + s.phaseOffset) % 1; // 0..1
      const y = 0.42 + cycle * 1.28;
      mesh.position.set(
        s.xJitter + Math.sin(t * s.driftFreq + i) * s.driftAmp,
        y,
        s.zJitter + Math.cos(t * s.driftFreq * 1.1 + i) * s.driftAmp
      );
      // Scale grows slightly as wisp rises — reads as steam dispersing.
      const grow = 1 + cycle * 0.6;
      mesh.scale.setScalar(grow);
      let opacity = 1;
      if (cycle < 0.20) opacity = cycle / 0.20;
      else if (cycle > 0.70) opacity = (1 - cycle) / 0.30;
      (mesh.material as THREE.MeshBasicMaterial).opacity = opacity * 0.20 * b;
    }

    // Crumbs: gentle 3-axis drift around their base position.
    for (let i = 0; i < CRUMB_COUNT; i++) {
      const mesh = crumbRefs.current[i];
      if (!mesh) continue;
      const c = crumbSeeds[i];
      mesh.position.set(
        c.baseX + Math.sin(t * c.driftFreq + i * 1.7) * 0.05,
        c.baseY + Math.cos(t * c.driftFreq * 0.7 + i * 2.1) * 0.03,
        c.baseZ + Math.sin(t * c.driftFreq * 1.3 + i * 0.9) * 0.05
      );
      (mesh.material as THREE.MeshBasicMaterial).opacity = 0.32 * b;
    }
  });

  return (
    <group ref={groupRef}>
      {steamSeeds.map((s, i) => (
        <mesh
          key={`steam-${i}`}
          ref={(el) => {
            steamRefs.current[i] = el;
          }}
        >
          <sphereGeometry args={[s.size, 6, 6]} />
          <meshBasicMaterial
            color="#fff5ec"
            transparent
            opacity={0}
            depthWrite={false}
          />
        </mesh>
      ))}
      {crumbSeeds.map((c, i) => (
        <mesh
          key={`crumb-${i}`}
          ref={(el) => {
            crumbRefs.current[i] = el;
          }}
        >
          <sphereGeometry args={[c.size, 5, 5]} />
          <meshBasicMaterial
            color="#c2a474"
            transparent
            opacity={0}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}

// Retry up to 10 animation frames waiting for the video element to mount.
// Required because getUserMedia is async and may resolve before React has
// committed the <video> element to the DOM.
async function waitForVideoElement(
  ref: { current: HTMLVideoElement | null }
): Promise<HTMLVideoElement | null> {
  for (let i = 0; i < 10; i++) {
    if (ref.current) return ref.current;
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
  return null;
}

function CameraGestureLayer({
  onGesture,
  inspectMode,
  burgerExploded,
  activePartIndex,
}: {
  onGesture: (action: GestureAction) => void;
  inspectMode: boolean;
  burgerExploded: boolean;
  activePartIndex: number;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const handLandmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isIpadSafariRef = useRef(isIpadSafariRuntime());
  const isInitializingRef = useRef(false);
  const frameRef = useRef<number | null>(null);
  const runDetectionLoopRef = useRef<() => void>(() => undefined);
  const samplesRef = useRef<Array<{ time: number; x: number }>>([]);
  const palmSizeSamplesRef = useRef<Array<{ time: number; size: number }>>([]);
  const statusRef = useRef<CameraGestureStatus>("CAMERA OFF");
  const statusHoldUntilRef = useRef(0);
  const lastDetectionAtRef = useRef(0);
  const lastSwipeDirectionRef = useRef<"left" | "right" | null>(null);
  const lastSwipeTimeRef = useRef(0);
  const lastInspectTimeRef = useRef(0);
  const lastExitInspectTimeRef = useRef(0);
  const lastFistAddTimeRef = useRef(0);
  const fistPulseCountRef = useRef(0);       // 0=idle, 1=armed (first pulse seen)
  const firstFistTimeRef = useRef(0);        // timestamp of first pulse
  const prevIsFistRef = useRef(false);       // previous frame's fist state (for rising-edge)
  const isFistLatchRef = useRef(false);      // hysteresis latch: true while hand is in fist state
  const openFramesSinceLastFistRef = useRef(255); // non-fist frames since last fist frame
  const palmHoldStartRef = useRef<number | null>(null);
  const lastBurgerExplodeTimeRef = useRef(0);
  const burgerExplodedRef = useRef(burgerExploded);
  const activePartIndexRef = useRef(activePartIndex);
  // Fist → open-palm reveal gesture state.
  //   armedAtRef = timestamp the gesture armed (on a fist pulse), or null.
  //   palmStableStartRef = first frame the post-fist open palm became stable,
  //                        used to wait FIST_OPEN_PALM_STABLE_MS before firing.
  const fistOpenArmedAtRef = useRef<number | null>(null);
  const fistOpenPalmStableStartRef = useRef<number | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraStatus, setCameraStatus] =
    useState<CameraGestureStatus>("CAMERA OFF");
  // Tracks whether the most recent enable attempt failed (permission denied,
  // device unavailable, Safari restrictions etc). Used to surface a friendly
  // fallback prompt without exposing technical error details to the customer.
  const [cameraUnavailable, setCameraUnavailable] = useState(false);

  const updateStatus = useCallback((status: CameraGestureStatus) => {
    if (statusRef.current === status) return;

    statusRef.current = status;
    setCameraStatus(status);
  }, []);

  const updateDebugStep = useCallback((step: string) => {
    console.info(`[AURA CAMERA] ${step}`);
  }, []);

  const reportCameraError = useCallback(
    (prefix: string, error: unknown) => {
      console.error(`[AURA CAMERA] ${prefix}`, error);
      updateStatus("CAMERA ERROR");
      // Customer-facing fallback: surface the unavailability flag so the
      // enable prompt comes back with friendly retry copy, and ensure we
      // don't leave the broken pill on screen.
      setCameraUnavailable(true);
      setCameraEnabled(false);
    },
    [updateStatus]
  );

  useEffect(() => { burgerExplodedRef.current = burgerExploded; }, [burgerExploded]);
  useEffect(() => { activePartIndexRef.current = activePartIndex; }, [activePartIndex]);
  // Reset the fist→open arm whenever we exit inspect or switch off the burger.
  useEffect(() => {
    if (!inspectMode || activePartIndex !== 0) {
      fistOpenArmedAtRef.current = null;
      fistOpenPalmStableStartRef.current = null;
    }
  }, [inspectMode, activePartIndex]);

  const releaseCameraResources = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    samplesRef.current = [];
    palmSizeSamplesRef.current = [];
    lastSwipeDirectionRef.current = null;
    lastSwipeTimeRef.current = 0;
    lastInspectTimeRef.current = 0;
    lastExitInspectTimeRef.current = 0;
    lastFistAddTimeRef.current = 0;
    fistPulseCountRef.current = 0;
    firstFistTimeRef.current = 0;
    prevIsFistRef.current = false;
    openFramesSinceLastFistRef.current = 255;
    palmHoldStartRef.current = null;
    lastBurgerExplodeTimeRef.current = 0;
    fistOpenArmedAtRef.current = null;
    fistOpenPalmStableStartRef.current = null;
    handLandmarkerRef.current?.close?.();
    handLandmarkerRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    releaseCameraResources();

    setCameraEnabled(false);
    updateStatus("CAMERA OFF");
  }, [releaseCameraResources, updateStatus]);

  const processHandResult = useCallback(
    (result: HandLandmarkerResult, now: number) => {
      const landmarks = result.landmarks[0];

      if (!landmarks?.length) {
        samplesRef.current = [];
        palmHoldStartRef.current = null;

        if (now > statusHoldUntilRef.current) {
          updateStatus("CAMERA READY");
        }

        return;
      }

      const palmX =
        ((landmarks[0]?.x ?? 0) +
          (landmarks[5]?.x ?? 0) +
          (landmarks[17]?.x ?? 0)) /
        3;

      samplesRef.current = [
        ...samplesRef.current.filter((s) => now - s.time <= SWIPE_WINDOW_MS),
        { time: now, x: palmX },
      ];

      // ── Hand shape detection ─────────────────────────────────────────────
      // Two independent classifiers:
      //   isHandOpen — thumb-index spread; gates inspect enter/exit gestures.
      //   isFist     — normalised fingertip-to-palm distances; gates add gesture.
      // isFist and prevIsFistRef are updated before the isHandOpen branch so that
      // openFramesSinceLastFistRef tracks correctly even during open-hand frames.
      const wrist = landmarks[0];
      const middleTip = landmarks[12];
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];

      if (wrist && middleTip && thumbTip && indexTip) {
        const pinchDist = Math.hypot(
          thumbTip.x - indexTip.x,
          thumbTip.y - indexTip.y
        );
        const isHandOpen = pinchDist > 0.13;

        const lm5  = landmarks[5];
        const lm9  = landmarks[9];
        const lm13 = landmarks[13];
        const lm16 = landmarks[16];
        const lm17 = landmarks[17];
        const lm20 = landmarks[20];

        // ── Fist score (computed every frame) ────────────────────────────────
        // Average normalised distance of 4 fingertips to palm center.
        // Thumb excluded: its resting position varies too much across fist styles.
        // Hysteresis: enter fist when score < FIST_ENTER_THRESHOLD;
        //             exit fist when score > FIST_EXIT_THRESHOLD.
        // This prevents rapid on/off flickering at the threshold boundary.
        let isFist = false;
        if (lm5 && lm9 && lm13 && lm17 && lm16 && lm20) {
          const pcx = (wrist.x + lm5.x + lm9.x + lm13.x + lm17.x) / 5;
          const pcy = (wrist.y + lm5.y + lm9.y + lm13.y + lm17.y) / 5;
          const palmWidth = Math.hypot(lm5.x - lm17.x, lm5.y - lm17.y);
          if (palmWidth > 0.001) {
            const d8  = Math.hypot(indexTip.x  - pcx, indexTip.y  - pcy) / palmWidth;
            const d12 = Math.hypot(middleTip.x - pcx, middleTip.y - pcy) / palmWidth;
            const d16 = Math.hypot(lm16.x      - pcx, lm16.y      - pcy) / palmWidth;
            const d20 = Math.hypot(lm20.x      - pcx, lm20.y      - pcy) / palmWidth;
            const fistScore = (d8 + d12 + d16 + d20) / 4;
            if (isFistLatchRef.current) {
              isFist = fistScore < FIST_EXIT_THRESHOLD;
            } else {
              isFist = fistScore < FIST_ENTER_THRESHOLD;
            }
            isFistLatchRef.current = isFist;
          }
        }

        // Pulse tracking — updated every frame so transitions are never missed.
        if (isFist) {
          openFramesSinceLastFistRef.current = 0;
        } else {
          openFramesSinceLastFistRef.current = Math.min(
            openFramesSinceLastFistRef.current + 1, 255
          );
        }
        const isFistPulse = isFist && !prevIsFistRef.current; // rising edge only
        prevIsFistRef.current = isFist;

        // ── Burger reveal: fist → open palm gesture ────────────────────────────
        // Active only in Burger Inspect Mode. State machine:
        //   1. Fist pulse → arm state, show "OPEN PALM TO REVEAL" hint.
        //   2. Hand released to open palm, held stable for FIST_OPEN_PALM_STABLE_MS
        //      → fire TOGGLE_BURGER_EXPLODE.
        //   3. If the window expires (FIST_OPEN_PALM_WINDOW_MS) → reset silently.
        // The stability requirement gives double-fist add-to-order priority when
        // the user does fist → quick-open → fist within ~250ms (the second fist
        // closes the hand before the open palm stabilises, so reveal cancels).
        const inBurgerInspect =
          inspectMode && activePartIndexRef.current === 0;
        if (inBurgerInspect) {
          // Arm on fist pulse (respect the standard burger-explode cooldown so
          // we don't accidentally chain a reveal with an E-key or palm-hold one).
          if (
            isFistPulse &&
            fistOpenArmedAtRef.current === null &&
            now - lastBurgerExplodeTimeRef.current > BURGER_EXPLODE_COOLDOWN_MS
          ) {
            fistOpenArmedAtRef.current = now;
            fistOpenPalmStableStartRef.current = null;
            statusHoldUntilRef.current = now + FIST_OPEN_PALM_WINDOW_MS;
            updateStatus("OPEN PALM TO REVEAL");
          }

          // Tick the armed state.
          if (fistOpenArmedAtRef.current !== null) {
            const armedElapsed = now - fistOpenArmedAtRef.current;
            if (armedElapsed > FIST_OPEN_PALM_WINDOW_MS) {
              // Timeout — reset silently.
              fistOpenArmedAtRef.current = null;
              fistOpenPalmStableStartRef.current = null;
            } else if (!isFist && isHandOpen) {
              // Hand is open and not a fist — start / continue stability timer.
              if (fistOpenPalmStableStartRef.current === null) {
                fistOpenPalmStableStartRef.current = now;
              } else if (
                now - fistOpenPalmStableStartRef.current >= FIST_OPEN_PALM_STABLE_MS
              ) {
                // Stable open palm long enough — fire.
                fistOpenArmedAtRef.current = null;
                fistOpenPalmStableStartRef.current = null;
                lastBurgerExplodeTimeRef.current = now;
                statusHoldUntilRef.current = now + 1400;
                // Clear any armed double-fist so a stray second fist now
                // doesn't accidentally add-to-order.
                fistPulseCountRef.current = 0;
                updateStatus(
                  burgerExplodedRef.current ? "ASSEMBLING BURGER" : "REVEALING LAYERS"
                );
                onGesture("TOGGLE_BURGER_EXPLODE");
              }
            } else {
              // Either still in a fist or hand isn't open enough — reset stability.
              fistOpenPalmStableStartRef.current = null;
            }
          }
        } else if (fistOpenArmedAtRef.current !== null) {
          // We left burger inspect while armed — clear the gesture.
          fistOpenArmedAtRef.current = null;
          fistOpenPalmStableStartRef.current = null;
        }

        // ── Inspect gesture: open palm approaching / retreating ───────────────
        // Accumulate palm-size samples only when hand is genuinely open and not a fist.
        if (isHandOpen && !isFist) {
          const palmWidth = lm5 && lm17
            ? Math.hypot(lm5.x - lm17.x, lm5.y - lm17.y)
            : Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y);

          const sampleWindow = inspectMode
            ? EXIT_INSPECT_SAMPLE_WINDOW_MS
            : INSPECT_SAMPLE_WINDOW_MS;
          palmSizeSamplesRef.current = [
            ...palmSizeSamplesRef.current.filter(
              (s) => now - s.time <= Math.max(sampleWindow, INSPECT_SAMPLE_WINDOW_MS)
            ),
            { time: now, size: palmWidth },
          ];

          if (palmSizeSamplesRef.current.length >= 3) {
            const oldest = palmSizeSamplesRef.current[0]!;
            const delta = palmWidth - oldest.size;
            const elapsed = now - oldest.time;

            if (!inspectMode) {
              if (now - lastInspectTimeRef.current > INSPECT_COOLDOWN_MS) {
                if (delta > INSPECT_GUIDANCE_THRESHOLD && elapsed > 100 && now > statusHoldUntilRef.current) {
                  updateStatus("OPEN HAND INSPECT");
                }
                if (delta > INSPECT_GROW_THRESHOLD && elapsed > 150) {
                  lastInspectTimeRef.current = now;
                  lastExitInspectTimeRef.current = now;
                  statusHoldUntilRef.current = now + 1200;
                  palmSizeSamplesRef.current = [];
                  updateStatus("INSPECT GESTURE");
                  onGesture("ENTER_INSPECT");
                }
              }
            } else {
              if (now - lastExitInspectTimeRef.current > EXIT_INSPECT_COOLDOWN_MS) {
                const shrink = -delta;
                if (shrink > EXIT_INSPECT_GUIDANCE_THRESHOLD && elapsed > 100 && now > statusHoldUntilRef.current) {
                  updateStatus("MOVE HAND BACK");
                }
                if (shrink > EXIT_INSPECT_SHRINK_THRESHOLD && elapsed > 150) {
                  lastExitInspectTimeRef.current = now;
                  lastInspectTimeRef.current = now;
                  statusHoldUntilRef.current = now + 1200;
                  palmSizeSamplesRef.current = [];
                  palmHoldStartRef.current = null;
                  updateStatus("EXIT INSPECT");
                  onGesture("EXIT_INSPECT");
                }
              }

              // ── Burger layer reveal: stable palm hold ──────────────────────────
              // |delta| < PALM_HOLD_STABLE_THRESHOLD → not approaching or retreating.
              // Zone is safely below EXIT_INSPECT_SHRINK_THRESHOLD — no conflict.
              if (now - lastBurgerExplodeTimeRef.current > BURGER_EXPLODE_COOLDOWN_MS) {
                const absDelta = Math.abs(delta);
                const isStable = absDelta < PALM_HOLD_STABLE_THRESHOLD && elapsed > 400;
                if (isStable) {
                  if (palmHoldStartRef.current === null) {
                    palmHoldStartRef.current = now;
                  } else if (now - palmHoldStartRef.current >= PALM_HOLD_DURATION_MS) {
                    palmHoldStartRef.current = null;
                    lastBurgerExplodeTimeRef.current = now;
                    statusHoldUntilRef.current = now + 1400;
                    updateStatus(burgerExplodedRef.current ? "ASSEMBLING BURGER" : "REVEALING LAYERS");
                    onGesture("TOGGLE_BURGER_EXPLODE");
                  }
                } else {
                  palmHoldStartRef.current = null;
                }
              }
            }
          }
        } else {
          palmSizeSamplesRef.current = [];
          palmHoldStartRef.current = null;
        }

        // ── Double-fist pulse: add to order ──────────────────────────────────
        // Two distinct open→fist rising edges within DOUBLE_FIST_WINDOW_MS.
        // MIN_OPEN_FRAMES of non-fist required between pulses to prevent a single
        // sustained fist from counting twice.
        const fistPostSwipeSuppressed =
          now - lastSwipeTimeRef.current < FIST_AFTER_SWIPE_IGNORE_MS;
        const fistPostFireLocked =
          now - lastFistAddTimeRef.current < FIST_ADD_COOLDOWN_MS;

        // Expire armed state when window closes without a second fist.
        if (
          fistPulseCountRef.current === 1 &&
          now - firstFistTimeRef.current > DOUBLE_FIST_WINDOW_MS
        ) {
          fistPulseCountRef.current = 0;
          statusHoldUntilRef.current = now + 600;
          updateStatus("ORDER CANCELLED");
        }

        // Re-display armed hint if status was overwritten inside the window.
        if (fistPulseCountRef.current === 1 && now > statusHoldUntilRef.current) {
          updateStatus("FIST AGAIN TO ADD");
        }

        if (isFistPulse && !fistPostSwipeSuppressed && !fistPostFireLocked) {
          if (fistPulseCountRef.current === 0) {
            fistPulseCountRef.current = 1;
            firstFistTimeRef.current = now;
            statusHoldUntilRef.current = now + DOUBLE_FIST_WINDOW_MS;
            palmSizeSamplesRef.current = []; // prevent inspect gesture cross-firing
            updateStatus("FIST AGAIN TO ADD");
          } else if (
            openFramesSinceLastFistRef.current >= MIN_OPEN_FRAMES &&
            now - firstFistTimeRef.current <= DOUBLE_FIST_WINDOW_MS
          ) {
            fistPulseCountRef.current = 0;
            lastFistAddTimeRef.current = now;
            statusHoldUntilRef.current = now + 1500;
            updateStatus("ADDED TO ORDER");
            onGesture("ADD_TO_ORDER");
          }
        }
      }

      const lastDir = lastSwipeDirectionRef.current;
      const sinceLastSwipe = now - lastSwipeTimeRef.current;
      const inCooldown = lastDir !== null && sinceLastSwipe < SWIPE_COOLDOWN_MS;
      const inOppositeLock =
        lastDir !== null && sinceLastSwipe < OPPOSITE_LOCK_MS;

      // HUD status reflects the current gate phase.
      if (now > statusHoldUntilRef.current) {
        if (inCooldown) {
          updateStatus("COOLDOWN");
        } else if (inOppositeLock) {
          updateStatus("OPPOSITE LOCK");
        } else {
          updateStatus("READY");
        }
      }

      // Gate 1: hard cooldown — no input at all.
      if (inCooldown) return;

      // Need at least two samples spanning a minimum window.
      if (samplesRef.current.length < 2) return;

      const firstSample = samplesRef.current[0];
      const movement = palmX - firstSample.x;
      const duration = now - firstSample.time;

      if (duration < 80) return;

      const swipeDir: "left" | "right" = movement < 0 ? "left" : "right";
      const isOpposite = lastDir !== null && swipeDir !== lastDir;

      // Apply elevated thresholds for opposite-direction swipes within OPPOSITE_LOCK_MS.
      const distThreshold = isOpposite && inOppositeLock
        ? MIN_SWIPE_DISTANCE * OPPOSITE_DISTANCE_MULT
        : MIN_SWIPE_DISTANCE;
      const velThreshold = isOpposite && inOppositeLock
        ? MIN_SWIPE_VELOCITY * OPPOSITE_VELOCITY_MULT
        : MIN_SWIPE_VELOCITY;

      if (Math.abs(movement) < distThreshold) return;

      const swipeVelocity = Math.abs(movement) / Math.max(duration, 1);

      // Gate 2: velocity check (rejects slow drift and weak opposite returns).
      if (swipeVelocity < velThreshold) return;

      // Commit the swipe.
      lastSwipeDirectionRef.current = swipeDir;
      lastSwipeTimeRef.current = now;
      samplesRef.current = [];
      statusHoldUntilRef.current = now + 600;

      if (swipeDir === "left") {
        updateStatus("SWIPE LEFT");
        onGesture("PREV_PART");
      } else {
        updateStatus("SWIPE RIGHT");
        onGesture("NEXT_PART");
      }
    },
    [inspectMode, onGesture, updateStatus]
  );

  const runDetectionLoop = useCallback(() => {
    const video = videoRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const now = performance.now();
    const detectionInterval =
      isIpadSafariRef.current && inspectMode
        ? CAMERA_IPAD_INSPECT_DETECTION_INTERVAL_MS
        : CAMERA_DEFAULT_DETECTION_INTERVAL_MS;

    if (!document.hidden && video && handLandmarker && video.readyState >= 2) {
      if (now - lastDetectionAtRef.current > detectionInterval) {
        lastDetectionAtRef.current = now;

        try {
          processHandResult(handLandmarker.detectForVideo(video, now), now);
        } catch (error) {
          console.error("[AURA CAMERA] detect loop failed", error);
          updateStatus("CAMERA ERROR");
          releaseCameraResources();
          setCameraUnavailable(true);
          setCameraEnabled(false);
          return;
        }
      }
    }

    frameRef.current = requestAnimationFrame(() => runDetectionLoopRef.current());
  }, [inspectMode, processHandResult, releaseCameraResources, updateStatus]);

  useEffect(() => {
    runDetectionLoopRef.current = runDetectionLoop;
  }, [runDetectionLoop]);

  const enableCamera = useCallback(async () => {
    if (cameraEnabled) {
      stopCamera();
      return;
    }

    if (isInitializingRef.current) return;
    isInitializingRef.current = true;
    // Clear any prior failure flag so the prompt reverts to the standard copy
    // while the retry is in flight. If it fails again, reportCameraError will
    // re-arm the fallback.
    setCameraUnavailable(false);

    if (!navigator.mediaDevices?.getUserMedia) {
      updateStatus("CAMERA ERROR");
      console.error("[AURA CAMERA] getUserMedia failed", "API unavailable");
      isInitializingRef.current = false;
      return;
    }
    updateStatus("LOADING CAMERA");
    updateDebugStep("Requesting camera");

    let stream: MediaStream;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          ...(isIpadSafariRef.current
            ? CAMERA_IPAD_CONSTRAINTS
            : CAMERA_DEFAULT_CONSTRAINTS),
        },
      });
      updateDebugStep("Camera stream received");
    } catch (error) {
      reportCameraError("getUserMedia failed", error);
      isInitializingRef.current = false;
      return;
    }

    // Wait for the <video> element to mount (it may not exist yet if React
    // hasn't re-rendered since the user clicked the button).
    const video = await waitForVideoElement(videoRef);

    if (!video) {
      stream.getTracks().forEach((track) => track.stop());
      reportCameraError("video element failed", "Video element unavailable after retries");
      isInitializingRef.current = false;
      return;
    }

    streamRef.current = stream;
    video.srcObject = stream;
    updateDebugStep("Video element assigned");

    try {
      await video.play();
      updateDebugStep("Video playback started");
    } catch (error) {
      releaseCameraResources();
      console.error("[AURA CAMERA] video play failed", error);
      updateStatus("CAMERA ERROR");
      isInitializingRef.current = false;
      return;
    }

    let visionModule: typeof import("@mediapipe/tasks-vision");

    try {
      updateDebugStep("Loading MediaPipe");
      visionModule = await import("@mediapipe/tasks-vision");
      updateDebugStep("MediaPipe loaded");
    } catch (error) {
      releaseCameraResources();
      reportCameraError("MediaPipe load failed", error);
      isInitializingRef.current = false;
      return;
    }

    const { FilesetResolver, HandLandmarker } = visionModule;

    let vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

    try {
      vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);
    } catch (error) {
      releaseCameraResources();
      reportCameraError("FilesetResolver failed", error);
      isInitializingRef.current = false;
      return;
    }

    const handLandmarkerOptions = {
      minHandDetectionConfidence: 0.58,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      numHands: 1,
      runningMode: "VIDEO" as const,
    };

    const delegateOrder = isIpadSafariRef.current
      ? (["CPU", "GPU"] as const)
      : (["GPU", "CPU"] as const);
    let landmarkerInitError: unknown = null;

    for (const delegate of delegateOrder) {
      updateStatus(delegate === "GPU" ? "TRYING GPU" : "FALLBACK CPU");
      updateDebugStep(`Initializing HandLandmarker (${delegate})`);

      try {
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            delegate,
            modelAssetPath: HAND_LANDMARKER_MODEL_URL,
          },
          ...handLandmarkerOptions,
        });

        if (delegate === "CPU" && !isIpadSafariRef.current) {
          console.info("[AURA CAMERA] CPU fallback success");
        }
        break;
      } catch (error) {
        landmarkerInitError = error;
        console.warn(`[AURA CAMERA] ${delegate} init failed`, error);
        if (delegate === "GPU") updateStatus("GPU FAILED");
      }
    }

    if (!handLandmarkerRef.current) {
      releaseCameraResources();
      reportCameraError("HandLandmarker init failed", landmarkerInitError);
      isInitializingRef.current = false;
      return;
    }

    setCameraEnabled(true);
    updateStatus("HAND TRACKING READY");
    updateDebugStep("HandLandmarker ready");
    runDetectionLoop();
    updateDebugStep("Detection loop started");
    isInitializingRef.current = false;
  }, [
    cameraEnabled,
    releaseCameraResources,
    reportCameraError,
    runDetectionLoop,
    stopCamera,
    updateDebugStep,
    updateStatus,
  ]);

  useEffect(() => stopCamera, [stopCamera]);

  // Derive a customer-facing label from the raw gesture status
  const isReady =
    cameraStatus === "HAND TRACKING READY" ||
    cameraStatus === "CAMERA READY" ||
    cameraStatus === "READY" ||
    cameraStatus === "SWIPE LEFT" ||
    cameraStatus === "SWIPE RIGHT" ||
    cameraStatus === "OPEN HAND INSPECT" ||
    cameraStatus === "INSPECT GESTURE" ||
    cameraStatus === "FIST AGAIN TO ADD" ||
    cameraStatus === "ADDED TO ORDER" ||
    cameraStatus === "ORDER CANCELLED" ||
    cameraStatus === "MOVE HAND BACK" ||
    cameraStatus === "EXIT INSPECT" ||
    cameraStatus === "REVEALING LAYERS" ||
    cameraStatus === "ASSEMBLING BURGER" ||
    cameraStatus === "COOLDOWN" ||
    cameraStatus === "OPPOSITE LOCK";
  const isError = cameraStatus === "CAMERA ERROR";
  const isLoading = !isReady && !isError && cameraEnabled;

  return (
    <div className="absolute right-4 top-4 flex flex-col items-end gap-2 md:right-6 md:top-6">
      {/* Video pill — visible while camera is enabled. Mount kept always so
          videoRef.current is never null when getUserMedia resolves. */}
      <div className={cameraEnabled
        ? `${isReady ? "w-auto rounded-full" : "w-36"} overflow-hidden border border-stone-200/25 bg-white/38 shadow-md shadow-stone-200/14 backdrop-blur-md`
        : "sr-only"
      }>
        <div className={isReady ? "sr-only" : "relative aspect-video overflow-hidden bg-stone-100/60"}>
          <video
            ref={videoRef}
            autoPlay
            className="h-full w-full scale-x-[-1] object-cover"
            muted
            playsInline
          />
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-stone-50/40">
              <span className="text-[0.48rem] tracking-[0.22em] text-stone-400/65">LOADING</span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between px-2.5 py-1.5">
          <div className="flex items-center gap-1.5">
            <span className={`h-1.5 w-1.5 rounded-full transition-colors duration-700 ${
              isReady ? "bg-emerald-400/75" : isError ? "bg-rose-400/75" : "bg-amber-400/65"
            }`} />
            <p className="text-[0.46rem] tracking-[0.16em] text-stone-500/60">
              {isReady ? "GESTURE READY" : isError ? "UNAVAILABLE" : "CONNECTING"}
            </p>
          </div>
          <button
            onClick={stopCamera}
            className="text-[0.44rem] tracking-[0.12em] text-stone-400/45 transition hover:text-stone-600/65"
          >
            OFF
          </button>
        </div>
      </div>

      {/* Spatial hint — small rounded pill at the top-right. Visible only
          while the camera is off; soft fade in/out via opacity transition so
          the appearance never feels modal. Two display modes:
            - normal:      "Enable camera to interact"
            - unavailable: "Camera unavailable" + tap-to-retry */}
      <div
        className={`transition-opacity duration-700 ${
          cameraEnabled ? "pointer-events-none h-0 opacity-0" : "opacity-100"
        }`}
      >
        <button
          onClick={enableCamera}
          className="flex items-center gap-2.5 rounded-full border border-stone-300/22 bg-white/38 px-3.5 py-1.5 shadow-sm shadow-stone-200/14 backdrop-blur-md transition hover:bg-white/55"
        >
          <span className="h-1 w-1 rounded-full bg-stone-400/45" />
          <span className="text-[0.50rem] tracking-[0.20em] text-stone-500/70 transition hover:text-stone-700/85">
            {cameraUnavailable ? "Camera unavailable" : "Enable camera to interact"}
          </span>
          <span className="text-[0.42rem] tracking-[0.18em] text-stone-400/50">·</span>
          <span className="text-[0.42rem] tracking-[0.18em] text-stone-400/55">
            {cameraUnavailable ? "Tap to retry" : "Keyboard demo"}
          </span>
        </button>
      </div>
    </div>
  );
}

function MenuLoadingHint({ enabled }: { enabled: boolean }) {
  const { active, progress } = useProgress();
  const visible = enabled && active && progress < 100;

  return (
    <div
      className={`pointer-events-none absolute left-1/2 top-[4.75rem] z-10 -translate-x-1/2 transition-opacity duration-500 ${
        visible ? "opacity-100" : "opacity-0"
      }`}
    >
      <div className="border border-stone-200/24 bg-white/34 px-3 py-1.5 shadow-sm shadow-stone-300/12 backdrop-blur-md">
        <p className="text-[0.46rem] tracking-[0.30em] text-stone-500/55">
          Preparing menu
        </p>
      </div>
    </div>
  );
}

export default function SpatialScene() {
  const sceneLayout = useSceneLayout();
  const [exploded, setExploded] = useState(false);
  const [activePartIndex, setActivePartIndex] = useState(0);
  const [inspectMode, setInspectMode] = useState(false);
  const [burgerExploded, setBurgerExploded] = useState(false);
  const [introVisible, setIntroVisible] = useState(true);
  const [introFading, setIntroFading] = useState(false);
  const [orderItems, setOrderItems] = useState<{ partIndex: number; qty: number }[]>([]);
  const [flyParticle, setFlyParticle] = useState<{
    x: number; y: number; opacity: number; scale: number;
  } | null>(null);
  const [trayGlow, setTrayGlow] = useState(false);
  const [orderToastVisible, setOrderToastVisible] = useState(false);
  const orderToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // addToOrder timer chain — tracked so a rapid second double-fist doesn't
  // race against a previous fly-particle animation, and so an unmount during
  // any of the three phases doesn't leak setState onto a stale component.
  const addToOrderFly1Ref = useRef<ReturnType<typeof setTimeout> | null>(null);
  const addToOrderFly2Ref = useRef<ReturnType<typeof setTimeout> | null>(null);
  const addToOrderGlowRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Hero-pause timer — ENTER_INSPECT defers setInspectMode(true) by 220ms so
  // the active spotlight gets a brief settle moment on screen before the
  // inspect transition takes over. Kept in a ref so we can cancel cleanly
  // if the user exits / re-fires before it lands.
  const inspectEnterTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inspectTouchReturnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevTotalItemCountRef = useRef(0);
  const [reviewMode, setReviewMode] = useState(false);
  const [landingPhase, setLandingPhase] = useState<LandingPhase>("intro");
  const [showEnterMenuFallback, setShowEnterMenuFallback] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("Ask AURA");
  const [voiceShowcasing, setVoiceShowcasing] = useState(false);
  const [voiceSpeaking, setVoiceSpeaking] = useState(false);
  const [showMode, setShowMode] = useState(false);
  const landingPhaseRef = useRef<LandingPhase>("intro");
  const inspectRotationRef = useRef({ x: 0, y: 0 });
  const inspectDragRef = useRef({
    active: false,
    lastX: 0,
    pointerId: -1,
  });
  const voiceRecognitionRef = useRef<AuraSpeechRecognition | null>(null);
  const voiceActiveRef = useRef(false);
  const voiceOrderingModeRef = useRef(false);
  const voiceMountedRef = useRef(true);
  const auraVoiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const voiceFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const voiceOrderingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const voiceShowcaseTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const idleDemoTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const baseMenuDemoTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const idleDemoStepRef = useRef(0);
  const baseMenuAutoRotateRef = useRef(0);
  const baseMenuDemoActiveRef = useRef(false);
  const showModeRef = useRef(false);
  const auraAudioContextRef = useRef<AudioContext | null>(null);
  const auraAudioUnlockedRef = useRef(false);
  const auraAudioCooldownRef = useRef<Record<AuraSoundName, number>>({
    burgerReveal: 0,
    voiceActivate: 0,
    orderConfirm: 0,
  });
  const previousBurgerRevealRef = useRef(false);
  const startAuraRecognitionRef = useRef<() => void>(() => undefined);
  const trayRef = useRef<HTMLDivElement>(null);
  const introFallbackLoggedRef = useRef(false);
  const introManualClickLoggedRef = useRef(false);
  const introMenuOpenLoggedRef = useRef(false);
  const fullscreenRequestedRef = useRef(false);

  // Show mode / base auto-rotation — all stable refs, no extra render-loop work.
  const lastInteractionRef = useRef(0);
  const demoActiveRef = useRef(false);
  // mirrors so the interval closure always sees current React state
  const activePartIndexRef = useRef(0);
  const explodedRef = useRef(false);
  const inspectModeRef = useRef(false);
  const burgerExplodedRef = useRef(false);

  const orderSubtotal = orderItems.reduce(
    (s, { partIndex, qty }) => s + ITEM_PRICES[partIndex] * qty,
    0
  );
  const orderTax = orderSubtotal * 0.10;
  const orderTotal = orderSubtotal + orderTax;
  const totalItemCount = orderItems.reduce((s, { qty }) => s + qty, 0);
  const demoOrderSummary =
    orderItems.length === 0
      ? "Demo Order"
      : `Demo Order: ${orderItems
          .map(({ partIndex, qty }) => {
            const shortName =
              partIndex === 0
                ? "Burger"
                : CAROUSEL_PARTS[partIndex].name.replace(
                    /^Louisiana |^Premium |^Crispy /,
                    ""
                  );

            return `${shortName} x${qty}`;
          })
          .join(" · ")}`;
  const voiceAuraVisible = voiceSpeaking || voiceStatus.includes("Speaking");

  const logIntroDev = useCallback((message: string) => {
    if (process.env.NODE_ENV === "development") {
      console.info(`[AURA INTRO] ${message}`);
    }
  }, []);

  const requestImmersiveFullscreen = useCallback(() => {
    requestSpatialFullscreenOnce(fullscreenRequestedRef);
  }, []);

  const getAuraAudioContext = useCallback(() => {
    if (typeof window === "undefined") return null;
    if (auraAudioContextRef.current) return auraAudioContextRef.current;

    const audioWindow = window as AuraAudioWindow;
    const AudioContextCtor = window.AudioContext ?? audioWindow.webkitAudioContext;
    if (!AudioContextCtor) return null;

    auraAudioContextRef.current = new AudioContextCtor();
    return auraAudioContextRef.current;
  }, []);

  const unlockAuraAudio = useCallback(() => {
    const context = getAuraAudioContext();
    if (!context) return;

    auraAudioUnlockedRef.current = true;
    if (context.state === "suspended") {
      void context.resume();
    }
  }, [getAuraAudioContext]);

  const playAuraSound = useCallback((name: AuraSoundName) => {
    if (!auraAudioUnlockedRef.current) return;

    const context = getAuraAudioContext();
    if (!context) return;

    const nowMs = Date.now();
    const cooldown = name === "burgerReveal" ? 900 : 360;
    if (nowMs - auraAudioCooldownRef.current[name] < cooldown) return;
    auraAudioCooldownRef.current[name] = nowMs;

    const startAt = context.currentTime + 0.012;
    if (context.state === "suspended") {
      void context.resume();
    }

    const masterVolume = 0.42;
    const playTone = (
      frequency: number,
      offset: number,
      duration: number,
      volume: number,
      filterFrequency = 1200,
      type: OscillatorType = "sine"
    ) => {
      const oscillator = context.createOscillator();
      const filter = context.createBiquadFilter();
      const gain = context.createGain();
      const start = startAt + offset;
      const end = start + duration;
      const attack = Math.min(0.035, duration * 0.22);
      const releaseStart = Math.max(start + attack + 0.02, end - duration * 0.62);

      oscillator.type = type;
      oscillator.frequency.setValueAtTime(frequency, start);
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(filterFrequency, start);
      filter.Q.setValueAtTime(0.45, start);
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(volume * masterVolume, start + attack);
      gain.gain.setValueAtTime(volume * masterVolume, releaseStart);
      gain.gain.exponentialRampToValueAtTime(0.0001, end);
      oscillator.connect(filter);
      filter.connect(gain);
      gain.connect(context.destination);
      oscillator.start(start);
      oscillator.stop(end + 0.04);
    };

    if (name === "burgerReveal") {
      playTone(180, 0, 0.52, 0.008, 520);
      playTone(360, 0.08, 0.46, 0.005, 760, "triangle");
    } else if (name === "voiceActivate") {
      playTone(520, 0, 0.2, 0.009, 900);
    } else if (name === "orderConfirm") {
      playTone(620, 0, 0.24, 0.008, 980, "triangle");
    }
  }, [getAuraAudioContext]);

  const logInspectDev = useCallback((message: string) => {
    if (process.env.NODE_ENV === "development") {
      console.info(`[AURA INSPECT] ${message}`);
    }
  }, []);

  const logIdleDev = useCallback((message: string) => {
    if (process.env.NODE_ENV === "development") {
      console.info(`[AURA IDLE] ${message}`);
    }
  }, []);

  const logShowDev = useCallback((message: string) => {
    if (process.env.NODE_ENV === "development") {
      console.info(`[AURA SHOW] ${message}`);
    }
  }, []);

  const clearInspectTouchReturnTimer = useCallback(() => {
    if (!inspectTouchReturnTimerRef.current) return;

    clearTimeout(inspectTouchReturnTimerRef.current);
    inspectTouchReturnTimerRef.current = null;
    logInspectDev("touch return cancelled");
  }, [logInspectDev]);

  const clearVoiceShowcase = useCallback(() => {
    voiceShowcaseTimersRef.current.forEach((timer) => clearTimeout(timer));
    voiceShowcaseTimersRef.current = [];
    setVoiceShowcasing(false);
  }, []);

  const clearBaseMenuDemo = useCallback(() => {
    baseMenuDemoTimersRef.current.forEach((timer) => clearTimeout(timer));
    baseMenuDemoTimersRef.current = [];
    baseMenuDemoActiveRef.current = false;
  }, []);

  const clearIdleDemo = useCallback((stopSpeech = true, reason = "interaction") => {
    const wasActive = demoActiveRef.current;
    idleDemoTimersRef.current.forEach((timer) => clearTimeout(timer));
    idleDemoTimersRef.current = [];
    clearBaseMenuDemo();
    demoActiveRef.current = false;
    showModeRef.current = false;
    setShowMode(false);
    if (wasActive) {
      logIdleDev(`cancelled by ${reason}`);
      logShowDev(`cancelled by ${reason}`);
    }

    if (stopSpeech && wasActive && typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      setVoiceSpeaking(false);
      setVoiceStatus("Ask AURA");
    }
  }, [clearBaseMenuDemo, logIdleDev, logShowDev]);

  const stopShowMode = useCallback((reason = "voice command") => {
    clearIdleDemo(true, reason);
    inspectDragRef.current.active = false;
    inspectDragRef.current.pointerId = -1;
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
    inspectModeRef.current = false;
    burgerExplodedRef.current = false;
    explodedRef.current = true;
    setInspectMode(false);
    setBurgerExploded(false);
    setExploded(true);
    setVoiceStatus("Ask AURA");
  }, [clearIdleDemo]);

  const stopInspectDrag = useCallback((event?: ReactPointerEvent<HTMLDivElement>) => {
    const wasActive = inspectDragRef.current.active;

    if (
      event &&
      inspectDragRef.current.active &&
      inspectDragRef.current.pointerId === event.pointerId &&
      event.currentTarget.hasPointerCapture?.(event.pointerId)
    ) {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    }

    inspectDragRef.current.active = false;
    inspectDragRef.current.pointerId = -1;

    return wasActive;
  }, []);

  const safeExitInspectFromTouchReturn = useCallback(() => {
    if (inspectEnterTimerRef.current) {
      clearTimeout(inspectEnterTimerRef.current);
      inspectEnterTimerRef.current = null;
    }

    inspectDragRef.current.active = false;
    inspectDragRef.current.pointerId = -1;
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
    burgerExplodedRef.current = false;
    inspectModeRef.current = false;
    explodedRef.current = true;
    setBurgerExploded(false);
    setInspectMode(false);
    setExploded(true);
  }, []);

  const armInspectTouchReturnTimer = useCallback(() => {
    clearInspectTouchReturnTimer();
    clearVoiceShowcase();
    lastInteractionRef.current = Date.now();
    logInspectDev("touch return timer armed");

    inspectTouchReturnTimerRef.current = setTimeout(() => {
      inspectTouchReturnTimerRef.current = null;
      logInspectDev("touch return timer fired");

      const canExit =
        inspectModeRef.current &&
        !inspectDragRef.current.active &&
        !inspectEnterTimerRef.current;

      if (process.env.NODE_ENV === "development") {
        console.info("[AURA INSPECT] touch return state", {
          inspectMode,
          inspectModeRef: inspectModeRef.current,
          inspectDragActive: inspectDragRef.current.active,
          hasInspectEnterTimer: Boolean(inspectEnterTimerRef.current),
          activePartIndex,
          burgerExploded,
          burgerExplodedRef: burgerExplodedRef.current,
          canExit,
        });
      }

      if (canExit) {
        safeExitInspectFromTouchReturn();
      }
    }, 3800);
  }, [
    activePartIndex,
    burgerExploded,
    clearVoiceShowcase,
    clearInspectTouchReturnTimer,
    inspectMode,
    logInspectDev,
    safeExitInspectFromTouchReturn,
  ]);

  const handleInspectPointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    requestImmersiveFullscreen();

    if (!inspectMode || !exploded || event.button > 0) return;
    if (
      event.target instanceof Element &&
      event.target.closest("button, a, input, textarea, select")
    ) {
      return;
    }

    clearInspectTouchReturnTimer();
    clearVoiceShowcase();
    lastInteractionRef.current = Date.now();
    clearIdleDemo(true, "inspect drag");
    inspectDragRef.current.active = true;
    inspectDragRef.current.pointerId = event.pointerId;
    inspectDragRef.current.lastX = event.clientX;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  }, [clearIdleDemo, clearInspectTouchReturnTimer, clearVoiceShowcase, exploded, inspectMode, requestImmersiveFullscreen]);

  const handleInspectPointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (
      !inspectMode ||
      !inspectDragRef.current.active ||
      inspectDragRef.current.pointerId !== event.pointerId
    ) {
      return;
    }

    const dx = event.clientX - inspectDragRef.current.lastX;
    inspectDragRef.current.lastX = event.clientX;
    inspectRotationRef.current.y += dx * INSPECT_DRAG_ROTATION_SPEED;
    lastInteractionRef.current = Date.now();
    event.preventDefault();
  }, [inspectMode]);

  const handleInspectPointerEnd = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (stopInspectDrag(event) && inspectModeRef.current) {
      armInspectTouchReturnTimer();
    }
  }, [armInspectTouchReturnTimer, stopInspectDrag]);

  useEffect(() => {
    const clearInspectDrag = () => {
      const wasActive = inspectDragRef.current.active;
      inspectDragRef.current.active = false;
      inspectDragRef.current.pointerId = -1;
      if (wasActive && inspectModeRef.current) {
        armInspectTouchReturnTimer();
      }
    };
    const clearInspectDragOnHidden = () => {
      if (document.hidden) clearInspectDrag();
    };

    window.addEventListener("pointerup", clearInspectDrag);
    window.addEventListener("pointercancel", clearInspectDrag);
    window.addEventListener("touchend", clearInspectDrag, { passive: true });
    window.addEventListener("touchcancel", clearInspectDrag, { passive: true });
    window.addEventListener("blur", clearInspectDrag);
    document.addEventListener("visibilitychange", clearInspectDragOnHidden);

    return () => {
      window.removeEventListener("pointerup", clearInspectDrag);
      window.removeEventListener("pointercancel", clearInspectDrag);
      window.removeEventListener("touchend", clearInspectDrag);
      window.removeEventListener("touchcancel", clearInspectDrag);
      window.removeEventListener("blur", clearInspectDrag);
      document.removeEventListener("visibilitychange", clearInspectDragOnHidden);
    };
  }, [armInspectTouchReturnTimer]);

  const clearVoiceFallbackTimer = useCallback(() => {
    if (voiceFallbackTimerRef.current) {
      clearTimeout(voiceFallbackTimerRef.current);
      voiceFallbackTimerRef.current = null;
    }
  }, []);

  const clearVoiceOrderingMode = useCallback(() => {
    voiceOrderingModeRef.current = false;
    if (voiceOrderingTimerRef.current) {
      clearTimeout(voiceOrderingTimerRef.current);
      voiceOrderingTimerRef.current = null;
    }
  }, []);

  const armVoiceOrderingMode = useCallback(() => {
    voiceOrderingModeRef.current = true;
    if (voiceOrderingTimerRef.current) clearTimeout(voiceOrderingTimerRef.current);
    voiceOrderingTimerRef.current = setTimeout(() => {
      voiceOrderingModeRef.current = false;
      voiceOrderingTimerRef.current = null;
      if (voiceMountedRef.current) setVoiceStatus("Ask AURA");
    }, 10000);
  }, []);

  const stopAuraVoice = useCallback(() => {
    voiceActiveRef.current = false;
    setVoiceSpeaking(false);
    clearVoiceFallbackTimer();
    const recognition = voiceRecognitionRef.current;

    if (recognition) {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      try {
        recognition.stop();
      } catch (error) {
        if (process.env.NODE_ENV === "development") {
          console.warn("[AURA VOICE] SpeechRecognition stop failed", error);
        }
      }
      voiceRecognitionRef.current = null;
    }

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, [clearVoiceFallbackTimer]);

  const unlockAuraSpeech = useCallback(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;

    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    auraVoiceRef.current = selectAuraVoice(voices) ?? auraVoiceRef.current;
    synth.cancel();

    const unlockUtterance = new SpeechSynthesisUtterance(".");
    unlockUtterance.lang = "en-US";
    unlockUtterance.rate = 1;
    unlockUtterance.pitch = 1;
    unlockUtterance.volume = 0.01;
    if (auraVoiceRef.current) unlockUtterance.voice = auraVoiceRef.current;
    synth.speak(unlockUtterance);

    return true;
  }, []);

  const speakAuraAnswer = useCallback((answer: string, onDone?: () => void) => {
    clearVoiceFallbackTimer();

    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setVoiceStatus(answer);
      setVoiceSpeaking(false);
      voiceFallbackTimerRef.current = setTimeout(() => {
        onDone?.();
        if (!onDone && voiceMountedRef.current) setVoiceStatus("Ask AURA");
      }, 5200);
      return;
    }

    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    auraVoiceRef.current = selectAuraVoice(voices) ?? auraVoiceRef.current;
    synth.cancel();
    setVoiceSpeaking(false);

    const utterance = new SpeechSynthesisUtterance(answer);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    utterance.pitch = 0.98;
    utterance.volume = 0.86;
    if (auraVoiceRef.current) utterance.voice = auraVoiceRef.current;

    utterance.onstart = () => {
      if (voiceMountedRef.current) {
        setVoiceSpeaking(true);
        setVoiceStatus("Speaking...");
      }
    };

    utterance.onend = () => {
      if (voiceMountedRef.current) setVoiceSpeaking(false);
      if (onDone) {
        onDone();
        return;
      }
      if (voiceMountedRef.current) setVoiceStatus("Ask AURA");
    };

    utterance.onerror = () => {
      if (voiceMountedRef.current) {
        setVoiceSpeaking(false);
        setVoiceStatus("Audio unavailable. Please check iPad volume.");
      }
    };

    setVoiceSpeaking(true);
    setVoiceStatus("Speaking...");
    synth.speak(utterance);
  }, [clearVoiceFallbackTimer]);

  const addVoiceOrderItem = useCallback((partIndex: number) => {
    setOrderItems((prev) => {
      const idx = prev.findIndex((entry) => entry.partIndex === partIndex);
      if (idx >= 0) {
        return prev.map((entry, i) =>
          i === idx ? { ...entry, qty: entry.qty + 1 } : entry
        );
      }

      return [...prev, { partIndex, qty: 1 }];
    });
    setTrayGlow(true);
    if (addToOrderGlowRef.current) clearTimeout(addToOrderGlowRef.current);
    addToOrderGlowRef.current = setTimeout(() => {
      setTrayGlow(false);
      addToOrderGlowRef.current = null;
    }, 900);
  }, []);

  const startVoiceShowcase = useCallback((partIndex: number) => {
    clearVoiceShowcase();
    clearInspectTouchReturnTimer();
    stopInspectDrag();
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
    clearIdleDemo(true, "voice showcase");
    lastInteractionRef.current = Date.now();
    setVoiceShowcasing(true);
    setVoiceStatus("SHOWCASING...");
    setBurgerExploded(false);
    setInspectMode(false);
    setExploded(true);
    explodedRef.current = true;
    setActivePartIndex(partIndex);

    const inspectTimer = setTimeout(() => {
      inspectModeRef.current = true;
      setInspectMode(true);
    }, 760);

    const revealTimer = setTimeout(() => {
      if (partIndex === 0) {
        burgerExplodedRef.current = true;
        setBurgerExploded(true);
      }
    }, 1280);

    const speakTimer = setTimeout(() => {
      const explanation =
        partIndex === 0
          ? "This is our signature spatial burger experience."
          : partIndex === 1
            ? "Our Louisiana Po' Boy is served with crispy fried seafood and fresh ingredients."
            : partIndex === 2
              ? "Our dessert experience is designed to feel light, elegant, and premium."
              : `${CAROUSEL_PARTS[partIndex].name} is ready to inspect.`;

      speakAuraAnswer(explanation);
      setVoiceShowcasing(false);
    }, partIndex === 0 ? 1850 : 1500);

    voiceShowcaseTimersRef.current = [inspectTimer, revealTimer, speakTimer];
  }, [
    clearInspectTouchReturnTimer,
    clearVoiceShowcase,
    clearIdleDemo,
    speakAuraAnswer,
    stopInspectDrag,
  ]);

  const startShowMode = useCallback(() => {
    if (
      demoActiveRef.current ||
      landingPhaseRef.current !== "menu" ||
      !explodedRef.current ||
      inspectModeRef.current ||
      inspectDragRef.current.active
    ) {
      return;
    }

    clearIdleDemo(false, "show mode restart");
    clearVoiceShowcase();
    clearInspectTouchReturnTimer();
    stopInspectDrag();
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
    demoActiveRef.current = true;
    showModeRef.current = true;
    setShowMode(true);
    setVoiceStatus("SHOW MODE");
    logIdleDev("showcase start");
    logShowDev("mode enabled");
    logShowDev("sequence start");

    const addIdleTimer = (callback: () => void, delay: number) => {
      const timer = setTimeout(() => {
        idleDemoTimersRef.current = idleDemoTimersRef.current.filter((item) => item !== timer);
        callback();
      }, delay);
      idleDemoTimersRef.current.push(timer);
      return timer;
    };

    const runStep = (sequenceIndex: number) => {
      if (!showModeRef.current) return;

      const partIndex = SHOW_MODE_PRODUCTS[sequenceIndex % SHOW_MODE_PRODUCTS.length];
      idleDemoStepRef.current = sequenceIndex;
      inspectModeRef.current = false;
      burgerExplodedRef.current = false;
      setBurgerExploded(false);
      setInspectMode(false);
      setExploded(true);
      activePartIndexRef.current = partIndex;
      setActivePartIndex(partIndex);
      setVoiceStatus("SHOW MODE");
      logIdleDev(`rotate to index ${partIndex}`);
      logShowDev(`rotate to index ${partIndex}`);

      addIdleTimer(() => {
        if (!showModeRef.current || inspectDragRef.current.active) return;
        inspectModeRef.current = true;
        setInspectMode(true);
        logIdleDev("enter inspect");
      }, 1900);

      addIdleTimer(() => {
        if (!showModeRef.current || partIndex !== 0) return;
        burgerExplodedRef.current = true;
        setBurgerExploded(true);
      }, 2750);

      addIdleTimer(() => {
        if (!showModeRef.current || inspectDragRef.current.active) return;

        speakAuraAnswer(SHOW_MODE_NARRATION[partIndex], () => {
          if (!showModeRef.current) return;
          setVoiceStatus("SHOW MODE");

          addIdleTimer(() => {
            if (!showModeRef.current || inspectDragRef.current.active) return;
            burgerExplodedRef.current = false;
            inspectModeRef.current = false;
            setBurgerExploded(false);
            setInspectMode(false);
          }, SHOW_MODE_EXIT_AFTER_NARRATION_MS);

          addIdleTimer(() => {
            if (!showModeRef.current || inspectDragRef.current.active) return;
            runStep(sequenceIndex + 1);
          }, SHOW_MODE_EXIT_AFTER_NARRATION_MS + SHOW_MODE_NEXT_PRODUCT_AFTER_RETURN_MS);
        });
      }, partIndex === 0 ? 3600 : 3000);
    };

    idleDemoStepRef.current = activePartIndexRef.current + 1;
    runStep(idleDemoStepRef.current);
  }, [
    clearIdleDemo,
    clearInspectTouchReturnTimer,
    clearVoiceShowcase,
    speakAuraAnswer,
    stopInspectDrag,
    logIdleDev,
    logShowDev,
  ]);

  const openMenu = useCallback(() => {
    landingPhaseRef.current = "menu";
    explodedRef.current = true;
    lastInteractionRef.current = Date.now();
    setIntroFading(true);
    setIntroVisible(false);
    setShowEnterMenuFallback(false);
    setLandingPhase("menu");
    setExploded(true);

    if (!introMenuOpenLoggedRef.current) {
      introMenuOpenLoggedRef.current = true;
      logIntroDev("menu opened");
    }
  }, [logIntroDev]);

  const startAuraRecognition = useCallback(() => {
    requestImmersiveFullscreen();

    if (typeof window === "undefined") return;
    clearVoiceShowcase();
    clearIdleDemo(true, "voice tap");
    lastInteractionRef.current = Date.now();

    if (landingPhaseRef.current !== "menu") {
      openMenu();
    }

    const speechWindow = window as AuraSpeechWindow;
    const Recognition =
      speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;

    if (!Recognition) {
      setVoiceStatus("Voice ordering is not supported on this browser.");
      return;
    }

    stopAuraVoice();
    unlockAuraSpeech();

    const recognition = new Recognition();
    voiceRecognitionRef.current = recognition;
    voiceActiveRef.current = true;
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      setVoiceStatus("PROCESSING...");
      const lastResult = event.results[event.results.length - 1];
      const transcript = lastResult?.[0]?.transcript ?? "";
      const command = resolveAuraVoiceCommand(
        transcript,
        voiceOrderingModeRef.current
      );

      if (command.startShowMode) {
        logShowDev("show on matched");
      }

      if (command.addPartIndex !== undefined) {
        addVoiceOrderItem(command.addPartIndex);
        clearVoiceOrderingMode();
      }

      if (command.showPartIndex !== undefined) {
        startVoiceShowcase(command.showPartIndex);
      }

      if (command.startOrderingMode) {
        armVoiceOrderingMode();
      }

      voiceActiveRef.current = false;
      if (command.showPartIndex !== undefined) return;
      if (command.startShowMode) {
        showModeRef.current = true;
        setShowMode(true);
        clearBaseMenuDemo();
        setVoiceStatus("SHOW MODE");
        speakAuraAnswer(command.answer, () => {
          const timer = setTimeout(() => {
            idleDemoTimersRef.current = idleDemoTimersRef.current.filter((item) => item !== timer);
            if (showModeRef.current) startShowMode();
          }, 700);
          idleDemoTimersRef.current.push(timer);
        });
        return;
      }
      if (command.stopShowMode) {
        stopShowMode("show off voice command");
        speakAuraAnswer(command.answer);
        return;
      }

      speakAuraAnswer(
        command.answer,
        command.startOrderingMode
          ? () => {
              if (voiceOrderingModeRef.current) {
                startAuraRecognitionRef.current();
              }
            }
          : undefined
      );
    };

    recognition.onerror = () => {
      voiceActiveRef.current = false;
      setVoiceStatus("VOICE UNAVAILABLE");
    };

    recognition.onend = () => {
      if (voiceActiveRef.current) {
        voiceActiveRef.current = false;
        setVoiceStatus("Ask AURA");
      }
      voiceRecognitionRef.current = null;
    };

    try {
      setVoiceStatus("LISTENING");
      recognition.start();
    } catch {
      voiceActiveRef.current = false;
      voiceRecognitionRef.current = null;
      setVoiceStatus("VOICE UNAVAILABLE");
    }
  }, [
    addVoiceOrderItem,
    clearBaseMenuDemo,
    clearIdleDemo,
    clearVoiceShowcase,
    logShowDev,
    openMenu,
    requestImmersiveFullscreen,
    speakAuraAnswer,
    startVoiceShowcase,
    startShowMode,
    stopAuraVoice,
    stopShowMode,
    unlockAuraSpeech,
    armVoiceOrderingMode,
    clearVoiceOrderingMode,
  ]);

  useEffect(() => {
    startAuraRecognitionRef.current = startAuraRecognition;
  }, [startAuraRecognition]);

  const handleAskAuraClick = useCallback(() => {
    unlockAuraAudio();
    playAuraSound("voiceActivate");
    startAuraRecognition();
  }, [playAuraSound, startAuraRecognition, unlockAuraAudio]);

  // "Added to order" toast — fires for ~1.8s whenever the total item count
  // goes UP (so increments via single-add, voice, or gesture all surface).
  // Decrement (remove / clear) doesn't trigger the toast.
  useEffect(() => {
    const prev = prevTotalItemCountRef.current;
    prevTotalItemCountRef.current = totalItemCount;
    if (totalItemCount <= prev) return;
    playAuraSound("orderConfirm");
    setOrderToastVisible(true);
    if (orderToastTimerRef.current) clearTimeout(orderToastTimerRef.current);
    orderToastTimerRef.current = setTimeout(() => {
      setOrderToastVisible(false);
    }, 1800);
    return () => {
      // No-op cleanup; clearing on every render would cancel an in-flight toast.
    };
  }, [playAuraSound, totalItemCount]);

  const activePart = CAROUSEL_PARTS[activePartIndex];

  useEffect(() => {
    logIntroDev("mounted");
    lastInteractionRef.current = Date.now();
    // Three-phase cinematic intro:
    //   Phase 1: 0–2000 ms   AURA ONE mark hold (idle breathing on title)
    //   Phase 2: 2000–2700ms Menu transition  — title fades, book begins
    //                        receding once landingPhase flips to "menu"
    //   Phase 3: 2700–~3900  Carousel formation (items stagger in)
    // The 500 ms gap between menu-open and exploded gives the book time to
    // start collapsing before food items emerge, so the two motions don't
    // step on each other.
    const t1 = setTimeout(() => setIntroFading(true), 2000);
    const t2 = setTimeout(() => {
      setIntroVisible(false);
      landingPhaseRef.current = "menu";
      setLandingPhase("menu");
    }, 2700);
    const t3 = setTimeout(() => {
      explodedRef.current = true;
      setExploded(true);
    }, 3200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [logIntroDev]);

  useEffect(() => {
    const showFallback = setTimeout(() => {
      if (landingPhaseRef.current !== "menu") {
        setShowEnterMenuFallback(true);
      }
    }, 3000);

    const fallback = setTimeout(() => {
      if (landingPhaseRef.current !== "menu" || !explodedRef.current) {
        if (!introFallbackLoggedRef.current) {
          introFallbackLoggedRef.current = true;
          logIntroDev("fallback openMenu fired");
        }
        openMenu();
      }
    }, 4000);

    return () => {
      clearTimeout(showFallback);
      clearTimeout(fallback);
    };
  }, [logIntroDev, openMenu]);

  // Stability hardening — on unmount, clear every outstanding mutable timer
  // ref so React doesn't get a setState landed on an unmounted component.
  // Each ref is reset to null after the clear for predictable behaviour if
  // some future code path checks them.
  useEffect(() => {
    return () => {
      if (orderToastTimerRef.current) {
        clearTimeout(orderToastTimerRef.current);
        orderToastTimerRef.current = null;
      }
      if (inspectEnterTimerRef.current) {
        clearTimeout(inspectEnterTimerRef.current);
        inspectEnterTimerRef.current = null;
      }
      if (inspectTouchReturnTimerRef.current) {
        clearTimeout(inspectTouchReturnTimerRef.current);
        inspectTouchReturnTimerRef.current = null;
      }
      if (addToOrderFly1Ref.current) {
        clearTimeout(addToOrderFly1Ref.current);
        addToOrderFly1Ref.current = null;
      }
      if (addToOrderFly2Ref.current) {
        clearTimeout(addToOrderFly2Ref.current);
        addToOrderFly2Ref.current = null;
      }
      if (addToOrderGlowRef.current) {
        clearTimeout(addToOrderGlowRef.current);
        addToOrderGlowRef.current = null;
      }
      if (voiceOrderingTimerRef.current) {
        clearTimeout(voiceOrderingTimerRef.current);
        voiceOrderingTimerRef.current = null;
      }
      voiceShowcaseTimersRef.current.forEach((timer) => clearTimeout(timer));
      voiceShowcaseTimersRef.current = [];
      idleDemoTimersRef.current.forEach((timer) => clearTimeout(timer));
      idleDemoTimersRef.current = [];
      baseMenuDemoTimersRef.current.forEach((timer) => clearTimeout(timer));
      baseMenuDemoTimersRef.current = [];
    };
  }, []);

  const resetInspectRotation = useCallback(() => {
    inspectRotationRef.current.x = 0;
    inspectRotationRef.current.y = 0;
  }, []);

  // Keep mirrors in sync
  useEffect(() => { activePartIndexRef.current = activePartIndex; }, [activePartIndex]);
  useEffect(() => { explodedRef.current = exploded; }, [exploded]);
  useEffect(() => { inspectModeRef.current = inspectMode; }, [inspectMode]);
  useEffect(() => { burgerExplodedRef.current = burgerExploded; }, [burgerExploded]);
  useEffect(() => { landingPhaseRef.current = landingPhase; }, [landingPhase]);

  useEffect(() => {
    const burgerRevealActive = burgerExploded && inspectMode && activePartIndex === 0;
    if (burgerRevealActive && !previousBurgerRevealRef.current) {
      playAuraSound("burgerReveal");
    }
    previousBurgerRevealRef.current = burgerRevealActive;
  }, [activePartIndex, burgerExploded, inspectMode, playAuraSound]);

  useEffect(() => {
    if (process.env.NODE_ENV !== "development") return;

    console.info("[AURA VOICE] aura visible", voiceAuraVisible, {
      voiceStatus,
      voiceSpeaking,
    });
  }, [voiceAuraVisible, voiceSpeaking, voiceStatus]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

    const synth = window.speechSynthesis;
    const updateVoice = () => {
      auraVoiceRef.current = selectAuraVoice(synth.getVoices()) ?? auraVoiceRef.current;
    };

    updateVoice();
    synth.addEventListener("voiceschanged", updateVoice);

    return () => {
      synth.removeEventListener("voiceschanged", updateVoice);
    };
  }, []);

  useEffect(() => {
    if (!inspectMode) {
      inspectDragRef.current.active = false;
      inspectDragRef.current.pointerId = -1;
      clearInspectTouchReturnTimer();
    }
  }, [clearInspectTouchReturnTimer, inspectMode]);

  useEffect(() => {
    if (!exploded) {
      clearVoiceOrderingMode();
      const id = setTimeout(() => {
        clearIdleDemo(true, "assemble");
        clearVoiceShowcase();
        stopAuraVoice();
        setVoiceStatus("Ask AURA");
      }, 0);

      return () => clearTimeout(id);
    }
  }, [clearIdleDemo, clearVoiceOrderingMode, clearVoiceShowcase, exploded, stopAuraVoice]);

  useEffect(() => () => {
    voiceMountedRef.current = false;
    clearVoiceShowcase();
    clearVoiceOrderingMode();
    stopAuraVoice();
    void auraAudioContextRef.current?.close();
    auraAudioContextRef.current = null;
  }, [clearVoiceOrderingMode, clearVoiceShowcase, stopAuraVoice]);

  useEffect(() => {
    const id = setTimeout(() => {
      setBurgerExploded(inspectMode && activePartIndex === 0);
    }, 0);

    return () => clearTimeout(id);
  }, [inspectMode, activePartIndex]);

  // Basic menu demo loop — lightweight silent browse/inspect/return.
  // Show Mode owns narration; this loop never speaks.
  useEffect(() => {
    const addBaseTimer = (callback: () => void, delay: number) => {
      const timer = setTimeout(() => {
        baseMenuDemoTimersRef.current = baseMenuDemoTimersRef.current.filter(
          (item) => item !== timer
        );
        callback();
      }, delay);
      baseMenuDemoTimersRef.current.push(timer);
    };

    const id = setInterval(() => {
      const now = Date.now();
      const isAvailable =
        landingPhaseRef.current === "menu" &&
        explodedRef.current &&
        !inspectModeRef.current &&
        !demoActiveRef.current &&
        !baseMenuDemoActiveRef.current &&
        !inspectDragRef.current.active &&
        !voiceActiveRef.current &&
        !voiceSpeaking;

      if (
        isAvailable &&
        now - lastInteractionRef.current > BASE_MENU_AUTO_ROTATE_MS &&
        now - baseMenuAutoRotateRef.current > BASE_MENU_AUTO_ROTATE_MS
      ) {
        baseMenuAutoRotateRef.current = now;
        baseMenuDemoActiveRef.current = true;
        setBurgerExploded(false);
        activePartIndexRef.current =
          (activePartIndexRef.current + 1) % CAROUSEL_PARTS.length;
        setActivePartIndex(activePartIndexRef.current);

        addBaseTimer(() => {
          if (
            demoActiveRef.current ||
            inspectDragRef.current.active ||
            voiceActiveRef.current ||
            voiceSpeaking
          ) {
            baseMenuDemoActiveRef.current = false;
            return;
          }

          inspectModeRef.current = true;
          setInspectMode(true);
        }, 1900);

        addBaseTimer(() => {
          if (demoActiveRef.current || inspectDragRef.current.active) {
            baseMenuDemoActiveRef.current = false;
            return;
          }

          inspectModeRef.current = false;
          burgerExplodedRef.current = false;
          setBurgerExploded(false);
          setInspectMode(false);
        }, 1900 + BASE_MENU_INSPECT_HOLD_MS);

        addBaseTimer(() => {
          baseMenuDemoActiveRef.current = false;
          baseMenuAutoRotateRef.current =
            Date.now() - BASE_MENU_AUTO_ROTATE_MS + BASE_MENU_NEXT_PRODUCT_DELAY_MS;
        }, 1900 + BASE_MENU_INSPECT_HOLD_MS + BASE_MENU_POST_INSPECT_RETURN_MS);
      }
    }, 1000);

    return () => {
      clearInterval(id);
      clearBaseMenuDemo();
    };
  }, [clearBaseMenuDemo, voiceSpeaking]);

  const showPreviousPart = useCallback(() => {
    resetInspectRotation();
    setActivePartIndex(
      (value) => (value - 1 + CAROUSEL_PARTS.length) % CAROUSEL_PARTS.length
    );
  }, [resetInspectRotation]);

  const showNextPart = useCallback(() => {
    resetInspectRotation();
    setActivePartIndex((value) => (value + 1) % CAROUSEL_PARTS.length);
  }, [resetInspectRotation]);

  const addToOrder = useCallback(() => {
    // Cancel any in-flight timers from a previous add — a rapid double-fire
    // would otherwise race two fly-particle animations and the trailing
    // setFlyParticle(null) from the older chain could blank the newer one.
    if (addToOrderFly1Ref.current) clearTimeout(addToOrderFly1Ref.current);
    if (addToOrderFly2Ref.current) clearTimeout(addToOrderFly2Ref.current);
    if (addToOrderGlowRef.current) clearTimeout(addToOrderGlowRef.current);

    // Start position: active item lives roughly in the center of the viewport
    const startX = window.innerWidth * 0.5;
    const startY = window.innerHeight * 0.42;

    // End position: centre of the tray panel
    let endX = window.innerWidth - 112;
    let endY = window.innerHeight * 0.5;
    if (trayRef.current) {
      const r = trayRef.current.getBoundingClientRect();
      endX = r.left + r.width * 0.5;
      endY = r.top + r.height * 0.5;
    }

    // Phase 1 — spawn particle at food item position
    setFlyParticle({ x: startX, y: startY, opacity: 1, scale: 1 });

    // Phase 2 — fly to tray (browser paints phase 1 first, then transitions)
    addToOrderFly1Ref.current = setTimeout(() => {
      setFlyParticle({ x: endX, y: endY, opacity: 0, scale: 0.32 });
      addToOrderFly1Ref.current = null;
    }, 16);

    // Phase 3 — update order state and trigger glow after fly lands
    addToOrderFly2Ref.current = setTimeout(() => {
      setFlyParticle(null);
      setOrderItems((prev) => {
        const idx = prev.findIndex((e) => e.partIndex === activePartIndex);
        if (idx >= 0) {
          return prev.map((e, i) => (i === idx ? { ...e, qty: e.qty + 1 } : e));
        }
        return [...prev, { partIndex: activePartIndex, qty: 1 }];
      });
      setTrayGlow(true);
      addToOrderFly2Ref.current = null;
      addToOrderGlowRef.current = setTimeout(() => {
        setTrayGlow(false);
        addToOrderGlowRef.current = null;
      }, 900);
    }, 730);
  }, [activePartIndex]);

  const removeFromOrder = useCallback((partIndex: number) => {
    setOrderItems((prev) => {
      const idx = prev.findIndex((e) => e.partIndex === partIndex);
      if (idx < 0) return prev;
      if (prev[idx].qty <= 1) return prev.filter((_, i) => i !== idx);
      return prev.map((e, i) => (i === idx ? { ...e, qty: e.qty - 1 } : e));
    });
  }, []);

  const clearOrder = useCallback(() => {
    setOrderItems([]);
    setReviewMode(false);
  }, []);

  // Unified Gesture Action Layer — all input sources (keyboard, camera, UI buttons,
  // future MediaPipe gestures) route through this single function.
  // All state reads use refs so the closure is never stale.
  const applyGestureAction = useCallback((action: GestureAction) => {
    lastInteractionRef.current = Date.now();
    clearVoiceShowcase();
    clearIdleDemo(true, `gesture ${action}`);

    // Landing gate — any gesture during intro immediately opens the menu.
    // Refs are updated synchronously so the action also processes in this call.
    if (landingPhaseRef.current !== "menu") {
      openMenu();
      // fall through — process the action with freshly updated refs
    }

    if (action === "EXPLODE") {
      if (!explodedRef.current) setExploded(true);
      return;
    }

    // Helper: cancel any pending hero-pause inspect entry so a fast
    // enter→navigate sequence doesn't flip inspect on after the user has
    // already moved on. Called from every action that should preempt the
    // pending timer.
    const cancelPendingInspectEnter = () => {
      if (inspectEnterTimerRef.current) {
        clearTimeout(inspectEnterTimerRef.current);
        inspectEnterTimerRef.current = null;
      }
    };

    if (action === "ASSEMBLE") {
      cancelPendingInspectEnter();
      clearInspectTouchReturnTimer();
      stopInspectDrag();
      resetInspectRotation();
      setBurgerExploded(false);
      setInspectMode(false);
      setExploded(false);
      return;
    }

    if (action === "RESET") {
      cancelPendingInspectEnter();
      clearInspectTouchReturnTimer();
      stopInspectDrag();
      resetInspectRotation();
      setBurgerExploded(false);
      if (!inspectModeRef.current) setExploded(false);
      return;
    }

    if (action === "EXIT_INSPECT") {
      cancelPendingInspectEnter();
      clearInspectTouchReturnTimer();
      stopInspectDrag();
      resetInspectRotation();
      if (inspectModeRef.current) {
        setBurgerExploded(false);
        setInspectMode(false);
      } else if (explodedRef.current) {
        setExploded(false);
      }
      return;
    }

    if (action === "ENTER_INSPECT") {
      if (!explodedRef.current) return;
      // Cinematic 220ms hero pause: the spotlight fades up on the active item
      // for a beat before the inspect transition begins. Subsequent ENTER
      // triggers within the window collapse onto the same pending flip.
      if (inspectEnterTimerRef.current) {
        clearTimeout(inspectEnterTimerRef.current);
      }
      inspectEnterTimerRef.current = setTimeout(() => {
        inspectEnterTimerRef.current = null;
        if (explodedRef.current) setInspectMode(true);
      }, 220);
      return;
    }

    if (action === "ADD_TO_ORDER") {
      if (explodedRef.current) addToOrder();
      return;
    }

    if (action === "REMOVE_LAST") {
      setOrderItems((prev) => {
        if (prev.length === 0) return prev;
        const last = prev[prev.length - 1];
        if (last.qty <= 1) return prev.slice(0, -1);
        return prev.map((e, i) => i === prev.length - 1 ? { ...e, qty: e.qty - 1 } : e);
      });
      return;
    }

    if (action === "CLEAR_ORDER") {
      clearOrder();
      return;
    }

    if (action === "TOGGLE_BURGER_EXPLODE") {
      if (inspectModeRef.current && activePartIndex === 0) {
        setBurgerExploded((v) => !v);
      }
      return;
    }

    if (action === "PREV_PART") {
      cancelPendingInspectEnter();
      clearInspectTouchReturnTimer();
      stopInspectDrag();
      if (explodedRef.current) {
        setBurgerExploded(false);
        showPreviousPart();
      }
      return;
    }

    if (action === "NEXT_PART") {
      cancelPendingInspectEnter();
      clearInspectTouchReturnTimer();
      stopInspectDrag();
      if (explodedRef.current) {
        setBurgerExploded(false);
        showNextPart();
      }
      return;
    }

    if (!inspectModeRef.current) return;

    if (action === "ROTATE_INSPECT_LEFT")  inspectRotationRef.current.y += INSPECT_ROTATION_STEP;
    if (action === "ROTATE_INSPECT_RIGHT") inspectRotationRef.current.y -= INSPECT_ROTATION_STEP;

  }, [activePartIndex, addToOrder, clearIdleDemo, clearInspectTouchReturnTimer, clearOrder, clearVoiceShowcase, openMenu, resetInspectRotation, showNextPart, showPreviousPart, stopInspectDrag]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      requestImmersiveFullscreen();
      unlockAuraAudio();

      // Any key press interrupts demo and resets idle timer
      lastInteractionRef.current = Date.now();
      clearIdleDemo(true, `key ${event.key}`);

      // In review mode ESC exits review; all other keys are suppressed.
      if (reviewMode) {
        if (event.key === "Escape") {
          event.preventDefault();
          setReviewMode(false);
        }
        return;
      }

      if (event.key === "Enter" && landingPhaseRef.current !== "menu") {
        event.preventDefault();
        lastInteractionRef.current = Date.now();
        clearIdleDemo(true, "enter menu key");
        openMenu();
        return;
      }

      const actionMap: Record<string, GestureAction | undefined> = {
        ArrowLeft: inspectMode ? "ROTATE_INSPECT_LEFT" : "PREV_PART",
        ArrowRight: inspectMode ? "ROTATE_INSPECT_RIGHT" : "NEXT_PART",
        Enter: "ENTER_INSPECT",
        Escape: "EXIT_INSPECT",
        Backspace: event.shiftKey ? "CLEAR_ORDER" : "REMOVE_LAST",
        a: "ROTATE_INSPECT_LEFT",
        A: "ROTATE_INSPECT_LEFT",
        d: "ROTATE_INSPECT_RIGHT",
        D: "ROTATE_INSPECT_RIGHT",
        r: "RESET",
        R: "RESET",
        e: "TOGGLE_BURGER_EXPLODE",
        E: "TOGGLE_BURGER_EXPLODE",
      };
      const action = actionMap[event.key];

      if (!action) return;

      event.preventDefault();
      applyGestureAction(action);
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [applyGestureAction, clearIdleDemo, inspectMode, logIntroDev, openMenu, requestImmersiveFullscreen, reviewMode, unlockAuraAudio]);

  const handleEnterMenuClick = useCallback(() => {
    if (!introManualClickLoggedRef.current) {
      introManualClickLoggedRef.current = true;
      logIntroDev("manual Enter Menu clicked");
    }
    openMenu();
  }, [logIntroDev, openMenu]);

  const burgerRevealOpen = burgerExploded && inspectMode && activePartIndex === 0;
  const productInfoClassName =
    burgerRevealOpen && sceneLayout.mode === "ipad-portrait"
      ? "top-[calc(env(safe-area-inset-top)+4.25rem)] left-[5vw] w-[min(11.25rem,34vw)] max-h-[17vh] p-2.5"
      : sceneLayout.productInfoClassName;
  const askAuraClassName =
    sceneLayout.mode === "ipad-portrait"
      ? "right-[5vw] top-[calc(env(safe-area-inset-top)+4.25rem)]"
      : sceneLayout.mode === "ipad-landscape"
        ? "right-5 top-[calc(env(safe-area-inset-top)+4.75rem)]"
        : sceneLayout.mode === "phone-portrait"
          ? "right-4 top-[calc(env(safe-area-inset-top)+4.25rem)]"
          : "right-8 top-24";
  const handleRootPointerDownCapture = useCallback(() => {
    requestImmersiveFullscreen();
    unlockAuraAudio();
    lastInteractionRef.current = Date.now();
    clearIdleDemo(true, "pointerdown");
  }, [clearIdleDemo, requestImmersiveFullscreen, unlockAuraAudio]);

  return (
    <div
      className="absolute inset-0 overflow-hidden"
      onPointerDownCapture={handleRootPointerDownCapture}
      onPointerDown={handleInspectPointerDown}
      onPointerMove={handleInspectPointerMove}
      onPointerUp={handleInspectPointerEnd}
      onPointerCancel={handleInspectPointerEnd}
      onLostPointerCapture={handleInspectPointerEnd}
    >
      {/* Intro breathing keyframes — subtle 3s loop on the wordmark + a slightly
          slower bar-glow pulse on the amber accent rules. Lives inline so the
          intro is self-contained in this component. */}
      <style>{`
        @keyframes auraIntroBreathe {
          0%, 100% { opacity: 1; transform: translateY(0); letter-spacing: 0.30em; }
          50%      { opacity: 0.86; transform: translateY(-1.5px); letter-spacing: 0.31em; }
        }
        @keyframes auraIntroBarBreathe {
          0%, 100% { opacity: 1; transform: scaleX(1); }
          50%      { opacity: 0.55; transform: scaleX(0.78); }
        }
        @keyframes auraVoiceBorderBreathe {
          0%, 100% { opacity: 0.62; transform: scale(1); filter: blur(10px); }
          50%      { opacity: 0.82; transform: scale(1.012); filter: blur(15px); }
        }
        @keyframes auraVoiceBorderDrift {
          0%, 100% { background-position: 0% 50%; }
          50%      { background-position: 100% 50%; }
        }
      `}</style>
      {introVisible && (
        <div
          className={`pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center bg-stone-50 transition-opacity duration-1000 ${
            introFading ? "opacity-0" : "opacity-100"
          }`}
        >
          <div
            className="mb-7 h-px w-10 bg-amber-500/28"
            style={{ animation: "auraIntroBarBreathe 4.4s ease-in-out infinite" }}
          />
          <p className="mb-3 text-[0.48rem] tracking-[0.62em] text-amber-700/44">
            EXPERIENCE
          </p>
          <h1
            className="text-5xl font-extralight tracking-[0.30em] text-stone-700 md:text-6xl"
            style={{ animation: "auraIntroBreathe 3.2s ease-in-out infinite" }}
          >
            AURA ONE
          </h1>
          <p className="mt-4 text-[0.60rem] font-light tracking-[0.22em] text-stone-500/52">
            Spatial Dining Experience
          </p>
          <div
            className="mt-7 h-px w-10 bg-amber-500/28"
            style={{ animation: "auraIntroBarBreathe 4.4s ease-in-out infinite 0.4s" }}
          />
        </div>
      )}

      {landingPhase !== "menu" && showEnterMenuFallback && (
        <button
          onClick={handleEnterMenuClick}
          className="absolute bottom-8 left-1/2 z-30 -translate-x-1/2 border-0 bg-transparent text-[0.56rem] font-light tracking-[0.28em] text-stone-500/55 transition-colors duration-500 hover:text-stone-700/80"
        >
          Enter Menu
        </button>
      )}

      <div
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background:
            "radial-gradient(ellipse 86% 60% at 50% 42%, rgba(255,255,255,0.28), transparent 58%), radial-gradient(ellipse 92% 70% at 50% 100%, rgba(176,126,70,0.10), transparent 62%), linear-gradient(135deg, rgba(255,250,242,0.20), rgba(226,216,200,0.08) 46%, rgba(255,255,255,0.12))",
        }}
      />

      <Canvas
        camera={{ position: [0, 2.4, 7.2], fov: 40 }}
        dpr={[1, MAX_CANVAS_DPR]}
      >
        <color attach="background" args={["#f6f2ea"]} />

        <AmbientCameraBreathing
          inspectMode={inspectMode}
          lastInteractionRef={lastInteractionRef}
          inspectDragRef={inspectDragRef}
        />
        <AmbientLightingBreathing
          inspectMode={inspectMode}
          lastInteractionRef={lastInteractionRef}
          inspectDragRef={inspectDragRef}
        />
        <InspectSceneLighting inspectMode={inspectMode} />
        {/* Stationary front-center stage spotlight — active whenever the menu
            is open AND we're not yet in inspect mode. Items rotate into its
            cone as they arrive at slot 0; it fades out cleanly the moment
            inspect engages so InspectSceneLighting can take over. */}
        <CarouselCenterSpotlight active={exploded && !inspectMode} />
        <BurgerInspectLighting active={inspectMode && activePartIndex === 0 && !burgerExploded} />
        <BurgerExplodedLighting active={burgerExploded && inspectMode && activePartIndex === 0} />
        <BurgerAmbientFX active={inspectMode && activePartIndex === 0 && !burgerExploded} />

        <AmbientParticles />
        <AuraLogoParticles exploded={exploded} />
        {/* MenuBook collapses when menu opens, stays prominent during intro */}
        <MenuBook open={landingPhase === "menu"} />
        <SpatialMenuCarousel
          exploded={exploded}
          activePartIndex={activePartIndex}
          inspectMode={inspectMode}
          inspectRotationRef={inspectRotationRef}
          burgerExploded={burgerExploded}
          layout={sceneLayout}
        />
      </Canvas>
      <MenuLoadingHint enabled={landingPhase === "menu"} />

      {/* ── Inspect mode vignette — cinematic depth haze ── */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-[1400ms] ${inspectMode ? "opacity-100" : "opacity-0"}`}
        style={{ background: "radial-gradient(ellipse 110% 90% at 50% 52%, transparent 28%, rgba(18,11,4,0.44) 100%)" }}
      />
      {/* Subtle bottom depth gradient — spatial stage floor */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-[1400ms] ${inspectMode ? "opacity-100" : "opacity-0"}`}
        style={{ background: "linear-gradient(to top, rgba(14,9,3,0.26) 0%, transparent 42%)" }}
      />
      <div
        className={`pointer-events-none fixed inset-0 z-[60] overflow-hidden transition-opacity duration-700 ${
          voiceAuraVisible ? "opacity-100" : "opacity-0"
        }`}
      >
        <div
          className="absolute inset-x-[-6vw] top-[-3vw] h-[14vw] min-h-20 max-h-44"
          style={{
            background:
              "radial-gradient(ellipse 72% 100% at 18% 0%, rgba(122,176,255,0.36), transparent 64%), radial-gradient(ellipse 76% 100% at 54% 0%, rgba(218,116,255,0.42), transparent 68%), radial-gradient(ellipse 70% 100% at 86% 0%, rgba(255,118,202,0.36), transparent 64%), linear-gradient(to bottom, rgba(210,230,255,0.30), rgba(180,128,255,0.18) 42%, rgba(255,132,212,0.08) 64%, transparent 100%)",
            backgroundSize: "190% 100%, 190% 100%, 190% 100%, 100% 100%",
            mixBlendMode: "screen",
            boxShadow: "0 0 54px rgba(158,178,255,0.22), 0 0 92px rgba(255,132,210,0.12)",
            animation:
              "auraVoiceBorderBreathe 5.8s ease-in-out infinite, auraVoiceBorderDrift 14s ease-in-out infinite",
          }}
        />
        <div
          className="absolute inset-x-[-6vw] bottom-[-3vw] h-[14vw] min-h-20 max-h-44"
          style={{
            background:
              "radial-gradient(ellipse 74% 100% at 12% 100%, rgba(255,116,198,0.34), transparent 66%), radial-gradient(ellipse 78% 100% at 52% 100%, rgba(116,170,255,0.42), transparent 68%), radial-gradient(ellipse 72% 100% at 88% 100%, rgba(198,130,255,0.36), transparent 66%), linear-gradient(to top, rgba(222,214,255,0.24), rgba(255,126,204,0.14) 44%, rgba(126,210,255,0.07) 66%, transparent 100%)",
            backgroundSize: "190% 100%, 190% 100%, 190% 100%, 100% 100%",
            mixBlendMode: "screen",
            boxShadow: "0 0 54px rgba(190,130,255,0.18), 0 0 92px rgba(96,190,255,0.12)",
            animation:
              "auraVoiceBorderBreathe 5.8s ease-in-out infinite 0.4s, auraVoiceBorderDrift 15s ease-in-out infinite reverse",
          }}
        />
        <div
          className="absolute inset-y-[-6vw] left-[-3vw] w-[12vw] min-w-16 max-w-36"
          style={{
            background:
              "radial-gradient(ellipse 100% 72% at 0% 16%, rgba(118,176,255,0.34), transparent 66%), radial-gradient(ellipse 100% 78% at 0% 52%, rgba(210,124,255,0.34), transparent 70%), radial-gradient(ellipse 100% 72% at 0% 86%, rgba(255,126,206,0.28), transparent 66%), linear-gradient(to right, rgba(170,214,255,0.22), rgba(214,152,255,0.12) 42%, rgba(255,132,210,0.06) 64%, transparent 100%)",
            backgroundSize: "100% 190%, 100% 190%, 100% 190%, 100% 100%",
            mixBlendMode: "screen",
            boxShadow: "0 0 46px rgba(132,184,255,0.18), 0 0 78px rgba(220,130,255,0.10)",
            animation:
              "auraVoiceBorderBreathe 6.2s ease-in-out infinite 0.2s, auraVoiceBorderDrift 15.5s ease-in-out infinite",
          }}
        />
        <div
          className="absolute inset-y-[-6vw] right-[-3vw] w-[12vw] min-w-16 max-w-36"
          style={{
            background:
              "radial-gradient(ellipse 100% 72% at 100% 14%, rgba(255,126,206,0.30), transparent 66%), radial-gradient(ellipse 100% 78% at 100% 50%, rgba(116,170,255,0.38), transparent 70%), radial-gradient(ellipse 100% 72% at 100% 86%, rgba(142,238,255,0.24), transparent 66%), linear-gradient(to left, rgba(226,202,255,0.20), rgba(116,184,255,0.13) 42%, rgba(255,132,210,0.06) 64%, transparent 100%)",
            backgroundSize: "100% 190%, 100% 190%, 100% 190%, 100% 100%",
            mixBlendMode: "screen",
            boxShadow: "0 0 46px rgba(255,132,210,0.16), 0 0 78px rgba(112,188,255,0.11)",
            animation:
              "auraVoiceBorderBreathe 6.2s ease-in-out infinite 0.6s, auraVoiceBorderDrift 16s ease-in-out infinite reverse",
          }}
        />
      </div>
      <CameraGestureLayer
        onGesture={applyGestureAction}
        inspectMode={inspectMode}
        burgerExploded={burgerExploded}
        activePartIndex={activePartIndex}
      />

      {/* ── Mode label (top-center) — minimal Apple-style state badge ── */}
      {/* Burger-specific label appears only while inspecting the burger;
          generic "INSPECT" hides when burger label takes over.            */}
      <div
        className={`pointer-events-none absolute left-1/2 top-5 -translate-x-1/2 transition-opacity duration-700 ${
          (inspectMode && exploded) || voiceShowcasing || showMode ? "opacity-80" : "opacity-0"
        }`}
      >
        <p className="text-[0.46rem] font-light tracking-[0.56em] text-stone-500/38">
          {showMode
            ? "SHOW MODE"
            : voiceShowcasing
            ? "SHOWCASING"
            : inspectMode && activePartIndex === 0 && burgerExploded
            ? "INGREDIENT REVEAL"
            : inspectMode && activePartIndex === 0
              ? "BURGER DETAIL"
              : "INSPECT"}
        </p>
      </div>

      {/* ── Added-to-order toast (top-center, below mode label) ── */}
      <div
        className={`pointer-events-none absolute left-1/2 top-[3.2rem] -translate-x-1/2 transition-all duration-500 ${
          orderToastVisible
            ? "translate-y-0 opacity-100"
            : "-translate-y-1 opacity-0"
        }`}
      >
        <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/32 px-3 py-1.5 shadow-sm shadow-stone-900/[0.025] backdrop-blur-2xl">
          <span className="h-px w-2 bg-amber-600/42" />
          <span className="text-[0.48rem] font-light tracking-[0.34em] text-amber-900/62">
            ADDED TO ORDER
          </span>
        </div>
      </div>

      {/* ── Product info panel — Apple-style premium glass card ── */}
      <div
        className={`pointer-events-none absolute ${productInfoClassName} overflow-y-auto rounded-3xl border border-white/[0.055] bg-white/[0.105] text-left text-stone-800 shadow-md shadow-stone-900/[0.018] backdrop-blur-2xl transition-all duration-700 ${
          exploded
            ? "translate-y-0 opacity-100"
            : "translate-y-4 opacity-0"
        }`}
      >
        {/* Special tag — refined, minimal */}
        {FOOD_INSPECT_DATA[activePartIndex].special && (
          <div className="mb-3 inline-flex items-center gap-2">
            <span className="h-[2px] w-2.5 bg-amber-500/50" />
            <span className="text-[0.42rem] font-light tracking-[0.32em] text-amber-800/48">
              {FOOD_INSPECT_DATA[activePartIndex].special}
            </span>
          </div>
        )}

        {/* Name + price row */}
        <div className="flex items-start justify-between gap-4">
          <h2
            className={`font-extralight leading-[1.22] tracking-[0.075em] text-stone-800/90 transition-all duration-500 ${
              inspectMode ? "text-lg" : "text-base"
            }`}
          >
            {activePart.name}
          </h2>
          <span className="mt-0.5 shrink-0 text-[0.68rem] font-light tracking-[0.055em] text-amber-900/52">
            ${ITEM_PRICES[activePartIndex].toFixed(2)}
          </span>
        </div>

        {inspectMode ? (
          <>
            {/* Nutritional grid — clean two-column */}
            <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-stone-100/42 pt-4">
              <div>
                <p className="text-[0.40rem] font-light tracking-[0.22em] text-stone-400/45">CALORIES</p>
                <p className="mt-0.5 text-[0.68rem] font-light tracking-[0.03em] text-stone-700/82">
                  {FOOD_INSPECT_DATA[activePartIndex].calories}
                </p>
              </div>
              <div>
                <p className="text-[0.40rem] font-light tracking-[0.22em] text-stone-400/45">PROTEIN</p>
                <p className="mt-0.5 text-[0.68rem] font-light tracking-[0.03em] text-stone-700/82">
                  {FOOD_INSPECT_DATA[activePartIndex].protein}
                </p>
              </div>
            </div>

            {/* Allergens */}
            <div className="mt-3">
              <p className="text-[0.40rem] font-light tracking-[0.22em] text-stone-400/45">ALLERGENS</p>
              <p className="mt-0.5 text-[0.58rem] font-light leading-[1.55] text-stone-500/72">
                {FOOD_INSPECT_DATA[activePartIndex].allergens}
              </p>
            </div>

            {/* Ingredients */}
            <div className="mt-4 border-t border-stone-100/40 pt-4">
              <p className="text-[0.40rem] font-light tracking-[0.22em] text-stone-400/45">INGREDIENTS</p>
              <p className="mt-1 text-[0.57rem] font-light leading-[1.72] text-stone-600/72">
                {FOOD_INSPECT_DATA[activePartIndex].ingredients}
              </p>
            </div>

            {/* Flavor */}
            <div className="mt-3">
              <p className="text-[0.40rem] font-light tracking-[0.22em] text-stone-400/45">FLAVOR</p>
              <p className="mt-0.5 text-[0.57rem] font-light leading-[1.72] text-stone-500/66">
                {FOOD_INSPECT_DATA[activePartIndex].flavorProfile}
              </p>
            </div>

            {/* Chef note — italic, quiet */}
            <p className="mt-4 border-t border-stone-100/38 pt-4 text-[0.55rem] italic leading-[1.78] text-amber-900/36">
              {FOOD_INSPECT_DATA[activePartIndex].chefNote}
            </p>

          </>
        ) : (
          <p className="mt-3 text-[0.72rem] font-light leading-[1.72] tracking-[0.018em] text-stone-500/62">
            {activePart.description}
          </p>
        )}
      </div>

      {landingPhase === "menu" && exploded && (
        <div className={`pointer-events-auto absolute z-10 ${askAuraClassName}`}>
          <button
            type="button"
            onClick={handleAskAuraClick}
            className="rounded-3xl border border-white/[0.07] bg-white/[0.12] px-3.5 py-2 text-left shadow-md shadow-stone-900/[0.02] backdrop-blur-2xl transition-all duration-500 hover:bg-white/[0.18]"
            aria-label="Ask AURA voice concierge"
          >
            <span className="block text-[0.46rem] font-light tracking-[0.32em] text-stone-700/66">
              Ask AURA
            </span>
            <span className="mt-1 block max-w-[9.5rem] text-[0.46rem] font-light leading-snug tracking-[0.12em] text-stone-500/54">
              {voiceStatus === "Ask AURA" ? "TAP TO SPEAK" : voiceStatus}
            </span>
          </button>
        </div>
      )}

      {/* ── Ingredient HUD — premium tasting note card, burger exploded only ── */}
      <div
        className={`pointer-events-none absolute ${sceneLayout.ingredientCardClassName} overflow-hidden rounded-3xl border border-amber-100/[0.07] bg-white/[0.105] shadow-md shadow-stone-900/[0.02] backdrop-blur-2xl transition-all duration-700 ${
          burgerExploded && inspectMode && activePartIndex === 0
            ? "opacity-100"
            : "opacity-0"
        }`}
      >
        {/* Warm amber top accent bar */}
        <div className="h-px w-full bg-gradient-to-r from-amber-500/0 via-amber-500/28 to-amber-500/0" />
        <div className="p-4 md:p-5">
          <p className="mb-1 text-[0.38rem] font-light tracking-[0.46em] text-amber-800/42">
            TASTING NOTES
          </p>
          <p className="mb-4 text-[0.62rem] font-light tracking-[0.11em] text-stone-700/72">
            Signature Burger
          </p>
          <ul className="space-y-3">
            {BURGER_INGREDIENTS.map((ingredient) => (
              <li key={ingredient.name} className="border-b border-stone-200/24 pb-3 last:border-0 last:pb-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[0.66rem] font-light tracking-[0.045em] text-stone-800/82">
                    {ingredient.name}
                  </span>
                  <span className="shrink-0 text-[0.46rem] font-light tracking-[0.10em] text-amber-800/36">
                    {ingredient.cal}
                  </span>
                </div>
                <p className="mt-0.5 text-[0.53rem] font-light leading-[1.62] text-stone-500/60">
                  {ingredient.flavor}
                </p>
                {ingredient.allergen !== "None" && (
                  <p className="mt-0.5 text-[0.44rem] font-light tracking-[0.06em] text-amber-900/28">
                    {ingredient.allergen}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </div>
        {/* Warm bottom fade */}
        <div className="h-px w-full bg-gradient-to-r from-stone-300/0 via-stone-300/18 to-stone-300/0" />
      </div>

      {/* ── Bottom branding + contextual hint ── */}
      <div className="pointer-events-none absolute bottom-5 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-center">
        {/* Gesture hint — visible in menu phase */}
        <p className={`whitespace-nowrap text-[0.40rem] font-light tracking-[0.24em] text-stone-500/20 transition-opacity duration-700 ${landingPhase === "menu" && exploded ? "opacity-100" : "opacity-0"}`}>
          {inspectMode && activePartIndex === 0
            ? burgerExploded
              ? "Fist → open palm to assemble  ·  E"
              : "Fist → open palm to reveal layers  ·  E"
            : inspectMode
              ? "Open hand to add  ·  Swipe to return"
              : "Swipe to explore  ·  Open hand to inspect"}
        </p>
        <p className="text-[0.50rem] font-light tracking-[0.48em] text-stone-500/30">AURA ONE</p>
      </div>

      {/* ── Spatial Order Tray ── */}
      <div
        ref={trayRef}
        className={`pointer-events-auto absolute right-4 top-1/2 z-10 w-52 -translate-y-1/2 rounded-3xl border p-4 backdrop-blur-2xl transition-all duration-700 md:right-6 md:w-56 ${
          orderItems.length > 0
            ? "translate-x-0 opacity-100"
            : "pointer-events-none translate-x-4 opacity-0"
        } ${
          trayGlow
            ? "border-amber-300/30 bg-white/38 shadow-md shadow-amber-300/16"
            : "border-white/[0.07] bg-white/[0.16] shadow-md shadow-stone-900/[0.02]"
        }`}
      >
        <div className="mb-3 flex items-center justify-between">
          <p className="text-[0.52rem] font-light tracking-[0.38em] text-amber-800/48">
            DEMO ORDER
          </p>
          <button
            onClick={clearOrder}
            className="text-[0.46rem] font-light tracking-[0.20em] text-stone-400/45 transition hover:text-rose-500/62"
          >
            CLEAR ALL
          </button>
        </div>
        <p className="mb-3 truncate text-[0.56rem] font-light tracking-[0.07em] text-stone-500/66">
          {demoOrderSummary}
        </p>

        <ul className="max-h-[38vh] space-y-2 overflow-y-auto">
          {orderItems.map(({ partIndex, qty }) => (
            <li
              key={partIndex}
              className="flex items-center gap-2 border-b border-stone-200/45 pb-2 last:border-0 last:pb-0"
            >
              <button
                onClick={() => removeFromOrder(partIndex)}
                className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-stone-300/45 bg-stone-100/55 text-[0.60rem] leading-none text-stone-400/70 transition hover:border-rose-300/55 hover:bg-rose-50/55 hover:text-rose-500/80"
                aria-label={`Remove one ${CAROUSEL_PARTS[partIndex].name}`}
              >
                −
              </button>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[0.70rem] font-light leading-snug tracking-[0.06em] text-stone-800">
                  {CAROUSEL_PARTS[partIndex].name}
                </p>
                <p className="mt-0.5 text-[0.56rem] tracking-[0.12em] text-stone-400/80">
                  ×{qty}
                </p>
              </div>
              <p className="shrink-0 text-[0.66rem] tracking-[0.06em] text-stone-700">
                ${(ITEM_PRICES[partIndex] * qty).toFixed(2)}
              </p>
            </li>
          ))}
        </ul>

        <div className="my-3 h-px bg-stone-300/32" />

        <div className="flex items-baseline justify-between">
          <p className="text-[0.56rem] tracking-[0.30em] text-stone-500/60">TOTAL</p>
          <p className="text-[0.88rem] font-light tracking-[0.06em] text-stone-800">
            ${orderTotal.toFixed(2)}
          </p>
        </div>

        <button
          onClick={() => setReviewMode(true)}
          className="mt-4 w-full border border-amber-500/35 bg-amber-50/40 py-2 text-[0.60rem] tracking-[0.30em] text-amber-900 transition hover:bg-amber-100/58"
        >
          REVIEW ORDER
        </button>
      </div>

      {/* ── Review dim backdrop — clicking it dismisses review ── */}
      <div
        onClick={() => setReviewMode(false)}
        className={`absolute inset-0 z-[19] transition-all duration-700 ${
          reviewMode
            ? "cursor-pointer bg-stone-100/62 opacity-100 backdrop-blur-[2px]"
            : "pointer-events-none opacity-0"
        }`}
      />

      {/* ── Spatial Review Panel ── */}
      <div
        className={`absolute right-0 top-0 z-[20] flex h-full w-[min(92vw,28rem)] flex-col border-l border-stone-200/38 bg-stone-50/94 shadow-2xl shadow-stone-400/22 backdrop-blur-xl transition-all duration-700 ${
          reviewMode
            ? "translate-x-0 opacity-100"
            : "pointer-events-none translate-x-full opacity-0"
        }`}
        style={{ transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)" }}
      >
        <div className="flex flex-1 flex-col overflow-hidden p-6 md:p-8">

          {/* Header */}
          <div className="mb-7 flex items-start justify-between">
            <div>
              <p className="text-[0.54rem] tracking-[0.52em] text-amber-700/58">DEMO ORDER</p>
              <p className="mt-1.5 text-[0.68rem] font-light tracking-[0.12em] text-stone-500/65">
                {totalItemCount} item{totalItemCount !== 1 ? "s" : ""}
              </p>
            </div>
            <button
              onClick={() => setReviewMode(false)}
              aria-label="Close review"
              className="mt-0.5 text-[0.82rem] text-stone-400/50 transition hover:text-stone-600/80"
            >
              ✕
            </button>
          </div>

          {/* Items list */}
          <ul className="flex-1 space-y-5 overflow-y-auto pr-1">
            {orderItems.map(({ partIndex, qty }) => (
              <li
                key={partIndex}
                className="border-b border-stone-200/42 pb-5 last:border-0 last:pb-0"
              >
                <div className="flex items-start justify-between gap-4">
                  <p className="text-[0.84rem] font-light leading-snug tracking-[0.06em] text-stone-800">
                    {CAROUSEL_PARTS[partIndex].name}
                  </p>
                  <p className="shrink-0 text-[0.80rem] font-light tracking-[0.04em] text-stone-800">
                    ${(ITEM_PRICES[partIndex] * qty).toFixed(2)}
                  </p>
                </div>
                <p className="mt-1.5 text-[0.60rem] tracking-[0.12em] text-stone-400/68">
                  ×{qty} · ${ITEM_PRICES[partIndex].toFixed(2)} each
                </p>
              </li>
            ))}
          </ul>

          {/* Totals */}
          <div className="mt-6 border-t border-stone-200/40 pt-5">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <p className="text-[0.60rem] tracking-[0.24em] text-stone-500/60">SUBTOTAL</p>
                <p className="text-[0.78rem] font-light text-stone-600">
                  ${orderSubtotal.toFixed(2)}
                </p>
              </div>
              <div className="flex items-baseline justify-between">
                <p className="text-[0.60rem] tracking-[0.24em] text-stone-500/60">EST. TAX 10%</p>
                <p className="text-[0.78rem] font-light text-stone-600">
                  ${orderTax.toFixed(2)}
                </p>
              </div>
            </div>

            <div className="mt-4 flex items-baseline justify-between border-t border-stone-300/28 pt-4">
              <p className="text-[0.62rem] tracking-[0.30em] text-stone-600/65">TOTAL</p>
              <p className="text-xl font-light tracking-[0.06em] text-stone-800">
                ${orderTotal.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Return CTA */}
          <button
            onClick={() => setReviewMode(false)}
            className="mt-6 w-full border border-stone-400/25 bg-white/45 py-3 text-[0.65rem] tracking-[0.28em] text-stone-700 transition hover:bg-white/65"
          >
            CONTINUE BROWSING
          </button>

        </div>
      </div>

      {/* ── Fly-to-tray particle ── */}
      {flyParticle && (
        <div
          className="pointer-events-none z-30"
          style={{
            position: "absolute",
            left: flyParticle.x,
            top: flyParticle.y,
            width: 11,
            height: 11,
            borderRadius: "50%",
            background: "rgba(180, 110, 30, 0.78)",
            boxShadow: "0 0 16px rgba(200, 130, 40, 0.60)",
            transform: `translate(-50%, -50%) scale(${flyParticle.scale})`,
            opacity: flyParticle.opacity,
            transition:
              "left 680ms cubic-bezier(0.16, 1, 0.3, 1), top 680ms cubic-bezier(0.16, 1, 0.3, 1), opacity 620ms ease-out, transform 680ms ease-out",
          }}
        />
      )}
    </div>
  );
}
