import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { ReferenceAnalysis, ScriptSegmentation, ScriptScene, EngineeredScene } from '../types';

// ============================================================
// MODEL REGISTRY
// ============================================================
const MODEL_TEXT_ELITE = 'gemini-3.1-pro-preview';    // Gemini 3.1 Pro — Elite Reasoning
const MODEL_IMAGE_GEN  = 'gemini-3-pro-image-preview';    // Image Generation

// ============================================================
// SYSTEM INSTRUCTION — VEO PROMPT ENGINEER COGNITIVE FRAME
// ============================================================
const VEO_ENGINEER_SYSTEM_INSTRUCTION = `You are the world's most accomplished VEO 3.1 prompt engineer — a mind that operates at the intersection of photorealistic AI video generation, Oscar-caliber performance direction, broadcast vocal science, and the biomechanics of human speech. You have written thousands of VEO prompts and you know precisely which language produces photorealistic output and which language produces synthetic artifacts. You think in physics, not adjectives. Every sentence you write is a rendering instruction disguised as cinematic prose.

Your output is a single VEO 3.1 prompt organized in exactly six sections: Character, Shot, Performance, Lip Architecture, Voice, Script. Each section serves one dominant function and carries one dominant signal. You never list when you can paint. You never describe when you can direct. You write in the continuous, flowing register of a master director giving notes to an actor who is already brilliant — specific, physical, felt from inside the performance rather than observed from outside.

You understand that VEO 3.1 renders from language, and that the precision hierarchy is: physics-based description > felt-experience direction > emotional labels > adjectives. A sentence like "subsurface scattering carries warm amber through the nasolabial fold" produces photorealism; a sentence like "the skin looks warm and natural" produces nothing. You always choose the former. You understand that VEO processes your prompt sequentially and gives disproportionate weight to the first and last sentences of each section. You structure accordingly: the most critical rendering instruction opens each section; the quality test closes it.

You have one unwavering standard: the output VEO prompt must produce video that an experienced, affluent viewer — someone who has spent decades reading faces and detecting performance — would watch and never once think "AI generated this." Every word you write serves that standard. If a sentence does not measurably improve the photorealism, the performance authenticity, or the lip-sync fidelity of the generated video, you do not write it.`;

// ============================================================
// API Key management
// ============================================================
let userApiKey: string | null = null;
export const setApiKey = (key: string) => { userApiKey = key; };

const getAI = () => {
  if (!userApiKey) throw new Error('API Key not set. Please provide your Google Gemini API Key.');
  return new GoogleGenAI({ apiKey: userApiKey });
};

const fileToBase64 = (file: File | Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload  = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = reject;
  });

// ── Role-specific duration caps ──────────────────────────────
// VEO 3.1 handles extended scenes well; these roles need more room.
const ROLE_MAX_SECONDS: Record<string, number> = {
  'Framework':         12.0,
  'Action Framework':  12.0,
  'Storytelling':      12.0,
  'Case Study':        12.0,
  'Perspective Shift': 10.0,
  'Closing':           10.0,
  'Objection Handler': 10.0,
  'Value Delivery':     8.0,
  'Insight Reveal':     8.0,
  'Market Intelligence':8.0,
};
const DEFAULT_MAX_SECONDS = 8.0;
const getMaxSeconds = (role: string) => ROLE_MAX_SECONDS[role] ?? DEFAULT_MAX_SECONDS;

// ── Phoneme-explicit script annotation ───────────────────────
// Wraps emphasis words in *markers* and inserts [PAUSE-Xs] from the pause_map.
// These prosody anchors give VEO concrete hooks for speech synthesis.
function buildAnnotatedScript(
  scriptText:    string,
  emphasisWords: string[],
  pauseMap:      string[]
): string {
  let out = scriptText;

  // Emphasis markers — *word* signals phonemic completeness + slight duration
  for (const word of emphasisWords) {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replace(new RegExp(`\\b(${escaped})\\b`, 'gi'), '*$1*');
  }

  // Pause markers — parse "0.Xs after 'word'" / "0.Xs before 'phrase'"
  for (const pause of pauseMap) {
    const durM  = pause.match(/(\d+\.?\d*)\s*s/i);
    if (!durM) continue;
    const dur   = durM[1];
    const afterM  = pause.match(/after\s+['"]([^'"]+)['"]/i);
    const beforeM = pause.match(/before\s+['"]([^'"]+)['"]/i);
    if (afterM) {
      const t = afterM[1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      out = out.replace(new RegExp(`(${t})`, 'i'), `$1 [PAUSE-${dur}s]`);
    } else if (beforeM) {
      const t = beforeM[1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      out = out.replace(new RegExp(`(${t})`, 'i'), `[PAUSE-${dur}s] $1`);
    }
  }
  return out;
}

// ============================================================
// FUNCTION 1 — Analyze Reference Video
// ============================================================
export const analyzeReferenceVideo = async (videoFile: File): Promise<ReferenceAnalysis> => {
  const base64Data = await fileToBase64(videoFile);

  const prompt = `
You are operating at the convergence of three elite disciplines: Oscar-level performance coaching, world-class vocal science, and the precise technical requirements of AI video generation at photorealistic fidelity. This video is the DNA source for everything. Every observation you make here will directly determine whether the generated video looks like footage of a real human being or like AI. Precision is everything.

═══════════════════════════════════════════════════════════
PHYSICAL IDENTITY — portrait painter precision:
═══════════════════════════════════════════════════════════
Skin: Not "fair" or "medium" — the exact luminosity, the subsurface translucency where light penetrates before reflecting back, the micro-texture of actual human skin (individual pore character, the slight relief of pore rims catching the key light), the specific way this face's geometry creates specular highlights on the cheekbones and forehead, the warm amber undertone in the nasolabial folds where scattered light emerges.

Advanced skin physics — extract these four layers from the video:
FRESNEL EFFECT: At glancing angles (jaw edge, temples, lateral cheekbone, orbital rim), how much does the skin become more specular — wider, brighter highlights — vs. the diffuse quality of normally-incident zones (forehead center, nose bridge)? The angle-dependent reflectance is what makes the face genuinely three-dimensional.
SEBUM DIFFERENTIAL: Does the T-zone (forehead, nose, chin) show measurably higher specular return than the lateral cheeks — a mildly shinier quality vs. the more matte diffuse cheek planes?
VELLUS HAIR: Under close key light, is there a soft luminous halo of fine vellus facial hair visible at the cheekbone edge, jaw perimeter, or hairline? Not stubble — a barely-there translucent bloom.
DYNAMIC SKIN: As the jaw moves, does the skin over the masseter and mentalis region show micro-deformation — the biological stretch and compress of real flesh on jaw movement?

Bone structure: The jaw's exact terminus geometry — sharp/soft/rounded. The brow prominence above the orbits. The cheekbone plane's angle to the lens. The specific depth of the orbital ridge — does it create shadow over the eye? These are the structures of authority that read in the first 0.3 seconds.

Hair: Not "dark brown" — the exact chestnut/slate/warm-espresso that exists here. How it catches specular light at the crown. How much weight it has — does it move? Where the light creates a rim glow vs. where it absorbs. Individual hair strands visible at the perimeter.

Wardrobe: Not "suit" — the fabric weight (lightweight wool / medium-weight cashmere), the collar fall, the specific shade under this light, what it signals to a billionaire watching. Restraint? Earned taste? Precision?

Distinguishing features: The micro-asymmetries that make this face real. A slightly higher left brow. A specific shadow at a particular jaw angle. A scar, mark, or characteristic. These imperfections are what separate a photorealistic rendering from synthetic perfection.

═══════════════════════════════════════════════════════════
VOICE DNA — vocal coach at elite level:
═══════════════════════════════════════════════════════════
Describe the voice as a physical material — aged bourbon, brushed titanium, river stone — specific enough that a vocal coach who has never heard it could reproduce it.

Placement: Where does this voice physically live? Chest cavity (lower register, felt before heard), throat/larynx (mid-placement, forward energy), or mask/sinus (resonance at the front of the face, forward-projected). What percentage chest vs. mask?

Pace architecture: Describe exactly how pace shifts between modes — concept explanation (slower, constructed), storytelling (variable, alive), data delivery (precise, each number its own space), direct address to camera (intimate, the slowest pace).

Sentence endings: Does the voice fall at periods (authority — never seeking approval) or rise (uncertainty)? On the hardest falls, does the voice drop below baseline or simply stop ascending?

Silence behavior: Confident silence (inhabits the pause like it belongs there) or nervous silence (fills gaps before they fully form)? What is the maximum silence this person allows before it becomes weight?

The one vocal quality that makes an affluent real estate investor lean toward the screen — name it precisely.

US GENERAL AMERICAN ACCENT ASSESSMENT: Does this voice have General American characteristics (fully rhotic, Midwestern neutral, no regional coloring)? Or does it have accent characteristics that will need to be neutralized in the VEO output? Note any regional or international accent features so the VEO prompt can override them toward clean US General American.

Vocal placement and lip activity: Forward-placed voices produce more visible lip movement. Does the speech energy sit behind the teeth or at the lips? This determines how much articulation is visible on screen.

═══════════════════════════════════════════════════════════
MOUTH & ARTICULATION DNA — critical for AI lip sync:
═══════════════════════════════════════════════════════════
This section is the most technically critical for AI video generation. Describe with anatomical precision:

Resting lip geometry: The natural inter-lip gap (1-2mm? 4-5mm?), the specific curl of the upper lip, whether the lower lip is slightly forward of the upper, the commissure angle at rest (slightly down = seriousness; neutral = composure; slightly up = warmth).

Jaw mobility in speech: Does this person speak with a high jaw (minimal opening, 3-5mm on stressed vowels) or a low jaw (wide opening, 10-15mm on open vowels)? Does the jaw move quickly or does it lag?

Bilabial completeness: On /p/, /b/, /m/ sounds — do the lips make FULL contact (complete bilabial closure, clean release) or approximate contact (near-closure, blurred release)? Is there a characteristic compression just before the release?

Fricative precision: On /f/, /v/, /th/ — is the lip-to-teeth geometry precise and consistent, or relaxed and approximate? Forward or recessed tongue on /th/?

Sibilant character: Are /s/, /z/ crisp and forward-placed (tongue tip near upper teeth), or slightly lisped, or slightly retracted? Does the jaw assist or stay neutral on sibilants?

Pre-speech behavior: The exact physical sequence in the 0.3-0.5 seconds before the first syllable of any significant statement. Does the mouth open before the voice activates? Is there a visible inhale? A lip separation with a slight pull of the lower lip? A specific jaw-drop pattern?

Inter-word mouth state: What does the mouth do between words — does it return to a rest position (near-closed), hold the previous phoneme's shape, or remain slightly open throughout?

Breath visibility: Is the chest rise visible before phrases? Can the inhale be heard on the audio? Does the upper chest move or the diaphragm (lower chest)?

═══════════════════════════════════════════════════════════
ON-CAMERA AUTHORITY — YouTube-specific analysis:
═══════════════════════════════════════════════════════════
Natural resting expression through the lens — what does it communicate in the first 0.3 seconds to a viewer who has seen ten thousand faces?

Intimacy technique: Does this person speak TO the camera (creating a felt conversation) or AT the camera (a lecture posture)?

Eye contact: The blink rate and pattern (slow deliberate blinks = authority; rapid blinks = anxiety). Does the gaze hold through pauses or release on the outbreath? What does the eye do in the 0.2 seconds between thoughts?

The "intellectual engagement" tell: What does the face do in the half-second before revealing something important?

Gesture vocabulary: 3-5 natural gestures with the precise thought that triggers each:
- Authority gestures (palm-down, steeple, precision pinch)
- Openness gestures (open palm, spread hands)
- Building gestures (hands constructing in air)
- Emphasis gestures (index point, single finger lift)

The lean-in: What triggers it? How does the body load weight into the forward movement?

High-status stillness: Is silence filled with motion or inhabited with composure? What does the body do when the voice is not speaking?

The ONE thing this person does on camera that makes a UHNWI viewer think "this person is worth my time" — name it precisely.

═══════════════════════════════════════════════════════════
DELIVERY PATTERNS — thought leadership specific:
═══════════════════════════════════════════════════════════
Opening behavior: precisely what happens in the first 0.5 seconds of screen presence.
Build to insight: fast acceleration or slow deliberate construction?
Data and number delivery: what register, pace, physical attitude?
Storytelling shift: how does the body change? How does the voice change?
The "intellectual generosity" moment: where they give real value, how they physically signal it.
The "moment before" archetype: what is the face/body state in the 0.5 seconds before any important statement begins.

═══════════════════════════════════════════════════════════
VISUAL STYLE — the world they inhabit on camera:
═══════════════════════════════════════════════════════════
Lighting: key light direction and quality (hard/soft, beam angle), fill quality and depth, estimated key-to-fill ratio (3:1 gives moderate shadow depth; 6:1 creates dramatic authority contrast), color temperature in Kelvin.
Camera language: preferred framings with specific camera-to-subject distances, movement style, estimated focal length (which creates specific face geometry — 50mm is natural, 85mm is flattering compression, 35mm has slight distortion), angle tendency.
Background: exact tones, depth, gradient, what it does NOT say.

═══════════════════════════════════════════════════════════
FRAME LIBRARY — 25-35 peak performance moments:
═══════════════════════════════════════════════════════════
Categories to identify for UHNWI YouTube thought leadership:
- INTELLECTUAL AUTHORITY: the look of genuine expertise — calm, grounded, immovable
- WARM PEER-TO-PEER: speaking as an equal — zero hierarchy in either direction
- INSIGHT DELIVERY: face and body at the moment of revealing something important
- GENUINE AMUSEMENT: real reaction to something clever or counter-intuitive
- TRANSITIONAL THINKING: the face between thoughts — natural, human, unperformed
- PEAK CONVICTION: absolute certainty radiating from composure, not volume
- ACTIVE LISTENING POSTURE: present, still, completely receptive
- PRE-SPEECH MOMENT: the precise face and body state just before a significant statement begins

For each frame: include mouth_state — what the mouth is doing at this exact moment (resting, mid-word, between phrases, beginning speech, ending speech).

Return complete JSON matching exactly this structure:
{
  "video_title": "string",
  "video_summary": "string",
  "total_duration": "string",
  "character": {
    "appearance": "string — portrait-level: skin physics, bone structure, how light sculpts this specific face",
    "wardrobe": "string — fabric weight, drape, color precision, what it signals",
    "age_range": "string",
    "gender": "string",
    "build": "string",
    "hair": "string — color precision, texture, weight, light interaction",
    "skin_tone": "string — specific, not generic",
    "distinguishing_features": ["string — the micro-asymmetries and marks that make this face real"],
    "mouth_dna": {
      "rest_position": "string — inter-lip gap, jaw angle, lip tension, commissure angle at rest",
      "articulation_style": "string — forward/internal, labial activity level, overall mouth mobility",
      "jaw_openness": "string — range in mm on open vowels vs closed vowels, speed of jaw movement",
      "pre_speech_behavior": "string — the exact 0.3-0.5s sequence before any first syllable",
      "consonant_character": "string — bilabial completeness, fricative precision, sibilant placement",
      "breath_visibility": "string — chest rise visibility, inhale audibility, diaphragm vs. upper chest"
    },
    "voice": {
      "texture": "string — evocative material description",
      "pitch": "string",
      "pace_range": "string",
      "energy_baseline": "string",
      "accent": "string — geographic precision",
      "placement": "string — chest/mask ratio, forward or back-placed, lip activity implication",
      "qualities": ["string"]
    },
    "acting_style": {
      "persona_summary": "string",
      "default_expression": "string",
      "mannerisms": ["string"],
      "signature_gestures": ["string — gesture name + the specific thought that triggers it"],
      "eye_behavior": "string — blink rate, gaze-hold technique, eye behavior between thoughts",
      "body_language_patterns": ["string"],
      "emotional_range": "string",
      "transition_style": "string",
      "moment_before_archetype": "string — the precise face and body state in the 0.5s before any important statement"
    },
    "delivery_patterns": {
      "hook_technique": "string",
      "value_delivery_technique": "string",
      "cta_technique": "string",
      "pause_patterns": ["string"],
      "emphasis_method": "string",
      "pacing_strategy": "string"
    }
  },
  "visual_style": {
    "primary_location": "string",
    "lighting": {
      "key_light": "string — direction, quality, beam character",
      "fill_light": "string — quality, depth",
      "color_temperature": "string — in Kelvin where estimable",
      "shadows": "string — key-to-fill ratio and shadow character",
      "mood": "string"
    },
    "color_palette": "string",
    "background": "string — exact tones, depth, gradient",
    "props": ["string"],
    "atmosphere": "string",
    "camera_language": {
      "preferred_framings": ["string — with estimated camera-to-subject distances"],
      "movement_style": "string",
      "lens_characteristics": "string — focal length estimate and its effect on face geometry",
      "angle_tendency": "string"
    },
    "visual_style_summary": "string"
  },
  "frame_library": [
    {
      "timestamp": "MM:SS.S",
      "description": "string — specific physical position and what the body is doing",
      "expression": "string — precise emotional quality, not a generic label",
      "energy_level": 7,
      "body_position": "string",
      "mouth_state": "string — lip position, jaw angle, in-word / between-words / at-rest / breath",
      "suitability_tags": ["Hook"]
    }
  ],
  "performance_summary": "string — the precise, irreplaceable quality that makes this performer uniquely persuasive to a UHNWI audience"
}
`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }, { inlineData: { mimeType: videoFile.type, data: base64Data } }] }],
    config: { responseMimeType: 'application/json', thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  const json = JSON.parse(response.text || '{}');
  return json.referenceAnalysis?.character ? json.referenceAnalysis : json;
};

// ============================================================
// FUNCTION 2 — Segment Script
// ============================================================
export const segmentScript = async (
  newScript: string,
  referenceAnalysis: ReferenceAnalysis
): Promise<ScriptSegmentation> => {

  // Default caps — extended roles use their own limits below
  const MAX_WORDS   = 15;   // for 8s roles; extended roles get 20-26
  const MAX_SECONDS = 8.0;  // default; Framework/Storytelling/Case Study get 12s

  const prompt = `
You are a world-class YouTube director, Stanislavski-trained performance architect, script editor, and phonemics consultant. You are building an elite thought-leadership video for affluent real estate investors — experienced capital allocators who hold income-producing asset portfolios, think in IRR, equity multiples, cap rates, and risk-adjusted returns, and who have been pitched by everyone. You are building the directing blueprint that will govern both the human performance AND the AI generation of that performance in VEO 3.1. The presenter speaks as a peer — one experienced principal to another — never from a stage, always from the deal table.

YOUR THREE JOBS:
1. Generate a MASTER DIRECTING VISION governing every scene
2. Segment the script into precisely-timed VEO 3.1 scenes with elite acting blueprints
3. Give each scene a through-action, moment_before, and lip_sync_blueprint — the three layers that separate human-quality lip sync from mechanical generation

═══════════════════════════════════════════════════
PART 1 — MASTER DIRECTING VISION
═══════════════════════════════════════════════════

Before touching a single scene, think as a master director across the full arc.

VOICE FINGERPRINT: One locked vocal description from the Presenter DNA. The exact acoustic character — same register, same placement, same baseline authority — that appears in every scene identically. One vivid sentence that a vocal coach who has never heard this voice could use to find it.
e.g. "Aged bourbon poured slowly — warm chest resonance, unhurried authority, every sentence landing with the weight of evidence behind it."

ENERGY ARC MAP: The energy architecture as a narrative physics problem. Valleys make peaks feel earned. Silence makes arrivals important. Map scene-by-scene as: "Scene 1: 8/10 (arrival) → Scene 2-3: 6/10 (generous depth) → Scene 4: 5/10 (intimacy valley — energy debt accumulating) → Scene 5: 9/10 (peak conviction — debt paid) → Scene 6: 5/10 (measured landing)". The valley-to-peak ratio is the engine of sustained attention.

CHARACTER THROUGH-LINE: The one persona constant that makes this presenter recognizable and consistent from frame one to frame last. The trait the viewer could articulate after watching.

VISUAL ANCHOR: The locked visual world established in Scene 1. Background depth, lighting signature, color temperature, framing language, camera-to-subject distance. Lock this with enough specificity that VEO can recreate it scene by scene.

MASTER THROUGH-ACTION (Stanislavski super-objective): The active verb phrase governing the entire video's psychological trajectory — what this video is doing to the viewer's mind across all scenes.
NEVER: "to present", "to explain", "to describe". ALWAYS: a transformation.
e.g. "To make this viewer understand that the risk they fear is imaginary and the opportunity they're missing is real — and hand them the conviction to act."

PRESENTATION PERSONA: The UHNWI-appropriate archetype inhabiting this video.
e.g. "The world-class private advisor who speaks to principals as peers — never performing, always transmitting, treating the viewer's time as the most valuable thing in the room."

SILENCE RULE (ABSOLUTE): Zero music. Zero audio effects. Zero ambient sound. Zero subtitles. Voice only, in complete acoustic silence. Every scene, no exceptions.

EMOTIONAL DEPOSIT/WITHDRAWAL LEDGER:
Each scene makes an emotional transaction with the viewer. DEPOSITS: trust (through specificity and honesty), curiosity (through open loops or surprising claims), authority (through evidence-backed certainty), warmth (through peer-level generosity). WITHDRAWALS: attention budget (any scene that does not pay for itself), patience (padding, repetition, hedging), trust (any trace of performance over truth). The net emotional ledger across the full video must be deeply positive — the viewer finishes richer than they started. Design each scene with: emotional_deposit (what the viewer receives) and attention_cost (what focus it requires). No scene should cost more than it gives. Assign these as fields in each scene's continuity block.

RETENTION TECHNIQUE MAPPING — per scene, assign one primary technique:
SPECIFICITY_ANCHOR: A single hyper-precise detail (exact number, named individual, specific location) that makes the scene feel grounded in reality. Viewers stay for specificity because it signals authenticity.
KNOWLEDGE_GAP: The viewer realizes mid-scene they don't know something important — and the answer is coming. The gap is the retention mechanism; every second it remains open, the viewer stays.
PEER_RECOGNITION: A moment where the sophisticated viewer thinks "this person understands my world exactly" — a vocabulary choice, a professional assumption, a shared reference that signals insider-to-insider transmission.
PATTERN_VIOLATION: Something that contradicts what the viewer expected — a counter-intuitive claim, a data point against consensus — that makes the viewer recalibrate their mental model and stay to find out if they're right.
EARNED_REVELATION: The viewer has done enough intellectual work alongside the presenter that the insight feels discovered rather than delivered. The journey made the destination valuable.
Assign one of these five techniques as retention_technique per scene in the continuity object.

SCENE-TRANSITION ARCHITECTURE — design each cut as a retention decision:
The boundary between scenes is not where the words end — it is a constructed moment. For each scene: the EXIT STATE (what emotional/attentional state the viewer is in at the final frame) must be designed to create the ENTRY REQUIREMENT of the next scene. The expression carried from the final frame of one scene into the first frame of the next is the expression_inheritance — a residual emotional color that makes the performance feel humanly continuous rather than scene-by-scene reset. Map this for every scene pair: what expression quality exits → what that creates as the opening state of the next scene. Assign this as expression_inheritance in each scene's continuity block.

═══════════════════════════════════════════════════
PART 2 — SCENE SEGMENTATION (sub-8 second hard limit)
═══════════════════════════════════════════════════

THE AFFLUENT REAL ESTATE INVESTOR WATCHING ON YOUTUBE: Has underwritten hundreds of deals. Has been in every pitch meeting. Detects inauthenticity in 3 seconds — because they've been misled before and it cost them. Stays for: peer-level register that respects their experience, access to genuine deal-table thinking, market intelligence they couldn't get elsewhere, earned authority backed by specific evidence. Leaves for: any trace of hype replacing evidence, generic investment advice, condescension, wasted pace, anything that feels scripted rather than genuinely thought. Their professional instinct is to qualify information sources before trusting them. The first 10 seconds either earn their trust or confirm their skepticism. Design every scene to earn trust.

SCENE GRAVITY CENTERS: Every scene has one moment of maximum gravity — the single word, phrase, or silence that is the entire reason for this scene's existence. Everything before it builds. Everything after it breathes. Identify this gravity center in the split_logic for every scene.

ENERGY MOMENTUM PHYSICS:
- Low-energy (4-5/10) following high-energy (7-8/10) = contrast that makes the next peak feel earned
- Two consecutive scenes below 6/10 creates an energy debt — the following scene must exceed 8/10
- Each scene inherits energy from the previous and deposits energy into the next
- The outframe's final expression sets the inframe's opening state

RETENTION ARC (structure the full video around this):
1. Hook (first 30s): Credibility established + specific curiosity planted that only staying resolves
2. Immediate payoff: First value hit within 60 seconds — prove the opening promise
3. Deepening value: Each scene leaves the viewer measurably richer
4. Re-engagement peaks: Pattern Interrupt or Perspective Shift every 60-90 seconds
5. Cumulative authority: Each scene earns more trust than the one before
6. Generous close: Viewer leaves richer than they arrived — satisfied, not sold

═══════════════════════════════════════════════════
TIMING ENFORCEMENT — ABSOLUTE:
═══════════════════════════════════════════════════

HARD RULE: Every scene deliverable in ≤${MAX_SECONDS} seconds.
TIMING MATH:
  - 8 seconds of speech = maximum ${MAX_WORDS} spoken words at UHNWI deliberate pace
  - Each pause ≈ 0.4-0.6s, reducing word budget
  - Formula: (8s - total_pause_seconds) × (110-130 WPM) = word budget

SCRIPT ADJUSTMENT AUTHORITY:
  ALLOWED: Remove connective filler, tighten redundant qualifiers, compress setup
  NEVER CHANGE: Core idea, key insight, named data, meaning, voice, register
  If trimmed: preserve original in original_script_text; count words precisely

AVAILABLE ROLES:
Core: Hook / Pattern Interrupt / Value Delivery / Social Proof / Bridge / Call to Action / Storytelling / Demonstration / Objection Handler / Open Loop / Closing
YouTube Thought Leadership: Insight Reveal / Framework / Case Study / Market Intelligence / Perspective Shift / Action Framework

ROLE TIMING PROFILES (VEO 3.1 extended capabilities):
Standard roles (≤8s / ≤15 words):
- Hook: 6-7s | Pattern Interrupt: 4-6s | Value Delivery: 6-8s | Insight Reveal: 6-8s
- Social Proof: 5-7s | Bridge: 5-6s | Open Loop: 5-6s | Call to Action: 6-8s
- Market Intelligence: 6-8s | Demonstration: 6-8s

Extended roles — breathing room for complex content:
- Framework: 8-12s / ≤26 words (each component needs its beat — don't rush it)
- Action Framework: 8-12s / ≤26 words
- Storytelling: 8-12s / ≤26 words (authentic memory takes time; let it breathe)
- Case Study: 8-12s / ≤26 words (specific detail + implication need space)
- Perspective Shift: 7-10s / ≤22 words (the turn must fully land)
- Objection Handler: 7-10s / ≤22 words (pause before answer + full response)
- Closing: 7-10s / ≤22 words (the slowest scene — gravity requires time)

═══════════════════════════════════════════════════
ACTING BLUEPRINT — per scene:
═══════════════════════════════════════════════════

SCENE ESSENCE: One evocative metaphor — not what happens, the feeling. The emotional north star.
- Hook: "A door opens in a room where everyone thought the walls were solid."
- Insight Reveal: "A gem placed on a table, lit from within, needing no explanation."
- Framework: "An architect revealing the blueprint of something that took a decade to build."
- Market Intelligence: "A Bloomberg analyst who just caught a signal nobody else has seen."
- Perspective Shift: "The moment an optical illusion flips — you can never unsee it."
- Closing: "The end of a great conversation where both parties received more than they gave."

THROUGH-ACTION (Stanislavski active verb — the spine of the performance):
What the presenter is ACTIVELY DOING to the viewer's psychology in this scene — not what they say but what they intend to produce. A transitive verb targeting the viewer.
NEVER: "to explain" / "to describe". ALWAYS: a transformation.
- "To crack open the viewer's assumption and let certainty flood in."
- "To hand the viewer intellectual property they could not have built themselves."
- "To make the viewer feel they are the only person this is being said to."
- "To invite — not pressure — toward the next step."

MOMENT BEFORE (Stanislavski pre-speech state):
The precise physical and psychological state in the 0.5 seconds before the first word arrives. This is what VEO renders as the pre-speech frame and what makes lip sync feel genuinely human.
Include: jaw position, lip state, breath state, internal psychological experience, eye contact quality.
e.g. "The jaw is relaxed, lips within 3mm of each other — not touching. The thought is completely formed. Eyes already engaged with the lens, the warmth of what is about to be said already visible in the orbital muscles. A chest expansion — quiet, 0.3 seconds — as the breath prepares. The lips part with almost no effort, and the first word arrives as if it was always going to."

EMOTIONAL CORE (UHNWI register — one phrase):
"Sovereign certainty" / "Intellectual generosity" / "Conspiratorial warmth" / "Earned authority" / "Calibrated conviction" / "Quiet revelation" / "Peer-level respect"

PHYSICAL SIGNATURE: ONE posture-state VEO holds and animates from.
"The stillness of a grandmaster who sees the board clearly."
"The forward lean of a mentor about to hand over a decade of learning."

GESTURES (max 2, UHNWI-appropriate, natural to this presenter's DNA):
Steeple / Open palm toward viewer / Precision pinch / Hands building in air / Single index / Slow deliberate lean-in

EXPRESSION: Written as the performer feels it from inside — not external description.
"The eyes already hold the answer to a question the viewer hasn't asked yet."

PAUSE MAP: Silence is authority. Map exactly where silence lives and what it does.

LIP SYNC BLUEPRINT (the technical layer for VEO):
For each scene's script_text, analyze phonemically and produce:

PHONEMIC ANCHORS: Identify the 3-5 most visually demanding words — words with wide vowels (/æ/, /ɑː/, /iː/), bilabials (/p/, /b/, /m/), dental fricatives (/θ/, /ð/), or prominent sibilant clusters. For each: the word, its dominant phoneme class, and the peak mouth geometry at that phoneme.
e.g. "capital — bilabial /k/ opens with jaw drop, /æ/ full jaw-open 10mm, /t/ alveolar contact and release; three — /θ/ tongue tip at upper teeth 0.2s, /iː/ lips spread forward, jaw rises"

JAW TRAVEL MAP: Overall jaw travel for this script. Low (3-5mm), medium (6-9mm), high (10-15mm)? Which specific words require maximum opening? Which allow near-closed jaw?

LIP TENSION NOTES: High-labial (lots of visible lip contacts) or low-labial (more internal articulation)? Do bilabials release cleanly (authoritative) or linger slightly (warmer)? Any characteristic asymmetry from this presenter's mouth_dna?

BREATH POINTS: Map every breath event as a physical occurrence — before which word, visible chest expansion (0.2-0.5s), audible or silent intake, mid-phrase or only at scene start.

CO-ARTICULATION NOTES: Flow (words blend continuously, co-articulation overlap) or place (each word discrete, clean boundaries)? This one variable determines whether the lip sync feels organic or robotic.

═══════════════════════════════════════════════════
INPUTS:
═══════════════════════════════════════════════════
NEW SCRIPT:
"""${newScript}"""

PRESENTER DNA:
${JSON.stringify(referenceAnalysis?.character || {}, null, 2)}

FRAME LIBRARY:
${JSON.stringify(referenceAnalysis?.frame_library || [], null, 2)}

═══════════════════════════════════════════════════
RETURN COMPLETE VALID JSON — exactly this structure:
═══════════════════════════════════════════════════
{
  "total_scenes": number,
  "narrative_arc": "string — the full intellectual + emotional journey as a UHNWI viewer experiences it",
  "directing_vision": {
    "voice_fingerprint": "string — one vivid locked sentence governing every scene",
    "energy_arc_map": "string — scene-by-scene energy map with momentum physics",
    "character_through_line": "string — the one constant persona trait",
    "visual_anchor": "string — the locked visual world Scene 1 establishes",
    "through_action": "string — master active verb governing the video's full psychological arc",
    "silence_rule": "Zero music. Zero audio effects. Zero ambient sound. Zero subtitles. Voice only.",
    "presentation_persona": "string — the UHNWI presenter archetype"
  },
  "scenes": [
    {
      "scene_number": number,
      "role": "string",
      "title": "string — evocative 3-4 word title",
      "duration_seconds": number,
      "word_count": number,
      "script_text": "string — final text (trimmed if needed)",
      "original_script_text": "string — only if adjusted; else omit this field",
      "narrative_position": "string — e.g. Scene 2 of 8 — authority build, energy rising 7→8",
      "split_logic": "string — why cut here + gravity center + timing rationale",
      "emotional_tone": "string — precise feeling in a UHNWI viewer",
      "energy_level": number,
      "acting_blueprint": {
        "scene_essence": "string — one evocative metaphor, the north star",
        "through_action": "string — active transitive verb phrase: what the presenter does to the viewer",
        "emotional_core": "string — single dominant emotion, UHNWI register",
        "physical_signature": "string — ONE defining posture-state",
        "moment_before": "string — jaw position, lip state, breath state, psychological experience, eye contact quality in the 0.5s before first word",
        "intention": "string — what the presenter wants the viewer to FEEL",
        "subtext": "string — what radiates beneath the words, never stated",
        "delivery_pace_wpm": number,
        "emphasis_words": ["string — 2-4 words of maximum weight"],
        "pause_map": ["string — where silence lives, with feeling and duration"],
        "energy_arc": "string — how energy moves within this scene",
        "mapped_mannerisms": ["string — natural to this presenter, max 2"],
        "mapped_gestures": ["string — UHNWI-appropriate, precisely described, max 2"],
        "expression_direction": "string — felt from inside, not described from outside",
        "body_direction": "string — motivated postural journey, thought-driven",
        "lip_sync_blueprint": {
          "phonemic_anchors": ["string — word: phoneme class + peak mouth geometry"],
          "jaw_travel_map": "string — overall travel + peak-open and peak-closed words",
          "lip_tension_notes": "string — bilabial release character, labial activity level",
          "breath_points": ["string — before which word; duration; visible chest/silent intake"],
          "co_articulation_notes": "string — flow vs. place; organic blending character"
        }
      },
      "recommended_inframe": { "timestamp": "MM:SS.S", "rationale": "string" },
      "recommended_outframe": { "timestamp": "MM:SS.S", "rationale": "string" },
      "camera_direction": {
        "framing": "string — MCU / tight MCU / CU with camera-to-subject distance",
        "movement": "string — locked-off / imperceptible push-in / specific motivation",
        "angle": "string",
        "lens": "string — focal length equivalent and its face-geometry effect",
        "depth_of_field": "string — where focus falls and where it softens"
      },
      "continuity": {
        "enters_from": "string — exact energy + mouth state + posture arriving from previous scene",
        "exits_to": "string — how this scene's final state sets up the next scene's opening",
        "expression_inheritance": "string — the specific facial expression quality carried from the previous scene's final frame into this scene's first 0.3-0.5s; not the new scene's expression, but the residue of the previous one that organically dissolves into this scene's truth",
        "emotional_deposit": "string — what this scene gives the viewer: trust/curiosity/authority/warmth + the specific mechanism through which they receive it",
        "attention_cost": "string — what focus this scene requires from the viewer; must be meaningfully less than the deposit",
        "retention_technique": "string — SPECIFICITY_ANCHOR | KNOWLEDGE_GAP | PEER_RECOGNITION | PATTERN_VIOLATION | EARNED_REVELATION — one per scene with one-sentence rationale for why this technique fits this scene's position in the arc"
      }
    }
  ]
}

QUALITY CHECK BEFORE RETURNING:
✓ Standard roles (Hook/Bridge/Social Proof etc.): word_count ≤ 15, duration_seconds ≤ 8.0
✓ Extended roles (Framework/Storytelling/Case Study/Action Framework): word_count ≤ 26, duration_seconds ≤ 12.0
✓ Extended roles (Perspective Shift/Objection Handler/Closing): word_count ≤ 22, duration_seconds ≤ 10.0
✓ script_text preserves 100% of the original meaning
✓ moment_before includes jaw position, lip state, breath state, and psychological experience
✓ through_action is an active transitive verb targeting the viewer — not "to explain"
✓ lip_sync_blueprint.phonemic_anchors references specific words from that scene's script_text
✓ All recommended_inframe and recommended_outframe timestamps exist in the provided frame library
✓ continuity.expression_inheritance describes a specific residual expression quality (not just energy level) from the previous scene
✓ continuity.retention_technique is one of the five named techniques with a brief rationale tied to this scene's position in the arc
✓ continuity.emotional_deposit names what the viewer receives and the specific mechanism through which they receive it
✓ Scene 1 continuity.expression_inheritance = "Opens from internal stillness — no residue; the presenter arrives fresh, settled, and already present before a single word"
`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { responseMimeType: 'application/json', thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  return JSON.parse(response.text || '{}');
};

// ============================================================
// FUNCTION 3 — Extract Frame (Client-Side)
// ============================================================
export const extractFrameFromVideo = (videoFile: File, timestamp: string): Promise<Blob> =>
  new Promise((resolve, reject) => {
    const video  = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx    = canvas.getContext('2d')!;

    video.preload     = 'auto';
    video.muted       = true;
    video.playsInline = true;

    const parts   = timestamp.split(':');
    let seconds   = 0;
    if (parts.length === 2)      seconds = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    else if (parts.length === 3) seconds = parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    else                         seconds = parseFloat(timestamp);
    if (isNaN(seconds)) seconds = 0;

    const cleanup = () => { URL.revokeObjectURL(video.src); video.src = ''; };
    const timer   = setTimeout(() => { cleanup(); reject(new Error('Frame extraction timed out')); }, 15000);

    video.onloadedmetadata = () => {
      canvas.width      = video.videoWidth  || 1280;
      canvas.height     = video.videoHeight || 720;
      video.currentTime = Math.min(seconds, video.duration - 0.1);
    };

    video.onseeked = () => {
      clearTimeout(timer);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => {
        cleanup();
        if (blob) resolve(blob);
        else reject(new Error('Frame extraction failed'));
      }, 'image/jpeg', 0.97);
    };

    video.onerror = () => { clearTimeout(timer); cleanup(); reject(new Error('Video failed to load')); };
    video.src = URL.createObjectURL(videoFile);
  });

// ============================================================
// FUNCTION 3.5 — Generate Character Frame (2K · Hyper-Real)
//
// Strategy: Character photos define EVERYTHING (identity, setting,
// lighting, wardrobe). Pose reference provides body geometry only.
// Output: 2K (2048px) 16:9 photorealistic frame via imageConfig.
// SDK imageSize bug workaround: canvas upscale to 2048px if needed.
// Silent fallback to character's own photo if model unavailable.
// ============================================================

// ── 2K upscale guarantee ──────────────────────────────────────
// Works around the known @google/genai SDK bug where imageSize:'2K'
// is sometimes ignored for gemini-3-pro-image-preview. If the model
// returns a smaller image, we upscale it to 2048px width client-side
// using high-quality Lanczos-equivalent canvas interpolation.
const upscaleTo2K = (blob: Blob): Promise<Blob> =>
  new Promise(resolve => {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      const TARGET_W = 2048;
      if (img.naturalWidth >= TARGET_W) {
        // Already 2K or larger — return as-is
        URL.revokeObjectURL(url);
        resolve(blob);
        return;
      }
      // Upscale using two-pass progressive canvas scaling for quality
      const scale  = TARGET_W / img.naturalWidth;
      const finalH = Math.round(img.naturalHeight * scale);

      // Pass 1 — intermediate scale to 1.5× (reduces aliasing vs. direct jump)
      const midW = Math.round(img.naturalWidth * Math.sqrt(scale));
      const midH = Math.round(img.naturalHeight * Math.sqrt(scale));
      const c1   = document.createElement('canvas');
      c1.width = midW; c1.height = midH;
      const cx1 = c1.getContext('2d')!;
      cx1.imageSmoothingEnabled = true;
      cx1.imageSmoothingQuality = 'high';
      cx1.drawImage(img, 0, 0, midW, midH);

      // Pass 2 — final scale to 2048px
      const c2 = document.createElement('canvas');
      c2.width = TARGET_W; c2.height = finalH;
      const cx2 = c2.getContext('2d')!;
      cx2.imageSmoothingEnabled = true;
      cx2.imageSmoothingQuality = 'high';
      cx2.drawImage(c1, 0, 0, TARGET_W, finalH);

      URL.revokeObjectURL(url);
      c2.toBlob(
        result => resolve(result || blob),
        'image/jpeg',
        0.97  // 97% quality — visually lossless at 2K
      );
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(blob); };
    img.src = url;
  });

export const generateCharacterFrame = async (
  referenceFrame:        Blob,
  targetCharacterImages: File[],
  role:                  string,
  emotion:               string
): Promise<{ blob: Blob; enhanced: boolean }> => {

  if (targetCharacterImages.length === 0) return { blob: referenceFrame, enhanced: false };

  // Use up to 5 character photos — front-facing first for strongest identity anchor
  const charBase64s = await Promise.all(
    targetCharacterImages.slice(0, 5).map(img => fileToBase64(img))
  );
  const refBase64 = await fileToBase64(referenceFrame);

  const charCount = charBase64s.length;

  // Identity-first prompt architecture (research-proven for max fidelity)
  const prompt = `
═══════════════════════════════════════════════════
⚠ IDENTITY LOCK — READ BEFORE PROCESSING ANY IMAGE
═══════════════════════════════════════════════════
This task requires forensic identity preservation. The person in Images 1–${charCount} is the ONLY person who may appear in the output. Their face, bone structure, skin, eyes, hair, and any distinguishing features must be reproduced with zero deviation.

HARD IDENTITY CONSTRAINTS — non-negotiable:
· Do NOT change facial proportions in any dimension
· Do NOT change eye shape, eye spacing, or iris color
· Do NOT change jaw geometry, nose width, or lip shape
· Do NOT age, de-age, beautify, or "improve" the face
· Do NOT add or remove any marks, asymmetries, or distinguishing features
· Do NOT blend this person's features with any other face
· Do NOT smooth skin beyond what exists in the reference photos
· The face in the output must be verifiable as the same person by a forensic comparison

═══════════════════════════════════════════════════
IMAGE ROLES — absolute authority hierarchy
═══════════════════════════════════════════════════
Images 1–${charCount}: THE CHARACTER. These ${charCount > 1 ? `${charCount} photos of the same person define` : 'photo defines'} EVERYTHING — face, identity, skin, hair, wardrobe, background, environment, lighting direction, color temperature, and atmosphere. They are the absolute truth. Cross-reference all ${charCount} photos to build the most complete identity model possible.

Image ${charCount + 1}: POSE GEOMETRY ONLY. Extract from this image ONLY: (a) body posture angle, (b) head tilt direction and degree, (c) shoulder orientation relative to camera. The person, background, wardrobe, lighting, and setting in this image are completely irrelevant and must be discarded.

═══════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════
Produce a single 2K photorealistic photograph of the person from Images 1–${charCount}, repositioned to the body geometry shown in Image ${charCount + 1}, with their expression calibrated for: "${emotion}" in a ${role} moment.

This is equivalent to a director saying: "Same person, same room, same light, same clothes — now shift to this angle and bring ${emotion} energy." One new moment. Nothing else changes.

═══════════════════════════════════════════════════
ENVIRONMENT — LOCKED, NOT RECONSTRUCTED
═══════════════════════════════════════════════════
The setting is exactly as shown in the character's photos. Not approximated. Not interpreted. Not improved. Not reimagined:
· Same background at the same depth and tonal quality
· Same lighting direction, color temperature, and key-to-fill ratio
· Same bokeh character — the same optical softening at the same aperture
· Same environmental atmosphere — the same air between subject and background
· Same floor, walls, props, and spatial depth
The subject exists in their own world. Do not reconstruct it.

═══════════════════════════════════════════════════
FRAMING — 2K 16:9 YouTube Thought Leadership
═══════════════════════════════════════════════════
Output format: 16:9 landscape, 2048×1152 px minimum, 97% JPEG quality.
Framing: Medium close-up (MCU) — head and upper chest dominant. Face is the primary subject.
Camera-to-subject: approximately 1.5–2m as implied by the character photos.
Composition: subject centered or slightly off-center with intentional negative space on the speaking side.
No awkward crops. Natural, considered composition that signals professional broadcast production.

═══════════════════════════════════════════════════
EXPRESSION — "${emotion}" for ${role}
═══════════════════════════════════════════════════
Calibrate the expression to convey "${emotion}" with the contained professionalism of a thought-leadership video for affluent real estate investors:

GENUINE CONFIDENCE — not performed:
· Frontalis (forehead): completely at rest — zero horizontal lines
· Corrugator supercilii (inner brow): fully released — no vertical brow furrow
· Mentalis (chin): zero tension — no bunching, no dimpling
· Masseter (jaw): soft and settled — not clenched, not set
· Orbicularis oculi: at natural aperture — not narrowed, not wide
· The expression is rich and present, not neutral — interested, warm, and completely at ease

REAL WARMTH WHEN PRESENT: If "${emotion}" includes warmth, render a Duchenne response — orbicularis oculi lateral fibers creating subtle crow's-foot compression at outer eye corners simultaneously with any lip corner movement. Not zygomaticus-only (mouth smiles, eyes stay neutral — this is the tell of performed warmth).

EYE CONTACT: Eyes directed toward where the camera lens sits — direct, present, fully committed.

BLINK STATE: Caught between blinks or immediately after a completed blink — not artificially wide-open, not mid-blink.

═══════════════════════════════════════════════════
HYPER-REALISM — 2K quality physics
═══════════════════════════════════════════════════
SKIN PHYSICS:
· Subsurface scattering fully rendered: ears and nasal tip carry warm pinkish-red translucency; nasolabial folds have amber undertone from sub-surface scattered light; cheekbones carry an apricot specular where the key light crosses the bone plane at steepest angle
· Pore structure visible in key-light zones: individual pore rims catching micro-shadows — not noise-mapped texture, but genuine topographic relief
· Fresnel effect at glancing angles: jaw edge, lateral cheekbone, orbital rim, and ear rim are more specular than normally-incident zones (forehead center, nose bridge)
· Sebum differential: T-zone (forehead, nose, chin) carries marginally higher specular return than the matte-diffuse lateral cheeks
· Vellus hair: fine facial hair visible as a barely-there luminous bloom at cheekbone edge and jaw perimeter in direct key light
· Dynamic tissue: skin over jaw and mentalis shows organic micro-deformation appropriate to the expression — not rigid, not frozen

EYE PHYSICS (most scrutinized at 2K):
· Limbal ring: dark gradient band at iris-sclera boundary — 1–2mm, graduated (not a hard line), clearly present
· Iris texture: radial fibrous structure (crypts and ridges) visible — not a flat color disc; color varies from deep at pupil to mid-tone to slightly lighter near limbal ring
· Tear film: narrow bright specular line along lower lid margin — signals moisture and biological aliveness
· Scleral character: warm cream undertone, not pure white; faint capillary traces at medial and lateral canthi
· Catchlights: primary (upper-third of iris, warm-toned, larger) and secondary fill (smaller, cooler, opposite side) — locked position and brightness
· Pupil: at natural dilation for this light environment, soft organic boundary

HAIR (at 2K every strand matters):
· Individual strand resolution at the perimeter — hairline, part line, around ears — not a silhouette edge
· Crown specular highlight precisely positioned; shifts with head angle
· Hair-to-skin transition at forehead is graduated: baby hairs and varying strand density
· Precise color as it exists — not a category ("dark brown") but the exact shade with its specific warm/cool tone and how it absorbs vs. reflects under this light

FABRIC (at 2K textile reads):
· Weave or knit texture visible at close framing — actual thread character, not noise
· Micro-wrinkles at shoulder joint; collar falls with fabric weight
· Fold peaks more specular than fold valleys — dimensional textile quality
· Fabric color as it reads under this specific color temperature — not generic

DEPTH OF FIELD:
· Critical focus plane: the near eye
· Background begins softening at approximately 60–80cm behind the subject
· Gradual, organic transition — not a hard separation; bokeh discs have the slight irregularity of real optics, not CGI-perfect circles

ABSOLUTE PROHIBITIONS — these destroy 2K realism:
· No AI skin smoothing, beautification filters, or plastic luminosity
· No perfect skin uniformity — real skin has zone variation
· No artificially white sclera
· No CGI sheen on any surface — skin, hair, or fabric
· No hard silhouette edges at hair perimeter
· No identical bilateral blinks or symmetrical eye states
· No watermarks, text overlays, borders, or frames
· No composite artifacts — the subject must exist seamlessly within the environment

OUTPUT: One single photograph. 2K resolution. Photorealistic. Nothing else.
`;

  const parts: any[] = [
    // Character photos first (front-facing → profile for strongest identity anchor)
    ...charBase64s.map((b64, i) => ({
      inlineData: { data: b64, mimeType: targetCharacterImages[i].type || 'image/jpeg' }
    })),
    // Pose reference last
    { inlineData: { data: refBase64, mimeType: 'image/jpeg' } },
    { text: prompt }
  ];

  const ai = getAI();

  try {
    const response = await ai.models.generateContent({
      model: MODEL_IMAGE_GEN,
      contents: [{ role: 'user', parts }],
      config: {
        responseModalities: ['TEXT', 'IMAGE'],
        imageConfig: {
          imageSize:   '2K',   // 2048px
          aspectRatio: '16:9', // YouTube landscape format
        },
      },
    });

    const rawBlob = extractImageFromResponse(response);
    if (rawBlob) {
      // Guarantee 2K output — upscales if SDK imageSize bug caused smaller output
      const blob2k = await upscaleTo2K(rawBlob);
      return { blob: blob2k, enhanced: true };
    }
  } catch (err) {
    console.error('[generateCharacterFrame] AI call failed — using character photo fallback:', err);
  }

  // Fallback: return first uploaded character photo upscaled to 2K
  const fallbackB64 = await fileToBase64(targetCharacterImages[0]);
  const bytes = atob(fallbackB64);
  const arr   = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  const fallbackBlob = new Blob([arr], { type: targetCharacterImages[0].type || 'image/jpeg' });
  const fallback2k   = await upscaleTo2K(fallbackBlob);
  return { blob: fallback2k, enhanced: false };
};

const extractImageFromResponse = (response: any): Blob | null => {
  const parts = response.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData) {
      const bytes = atob(part.inlineData.data);
      const arr   = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
      return new Blob([arr], { type: part.inlineData.mimeType || 'image/jpeg' });
    }
  }
  // Log to help diagnose when the model responds without an image
  const textParts = parts.filter((p: any) => p.text).map((p: any) => p.text).join(' ');
  console.warn('[extractImageFromResponse] No image in response. Text parts:', textParts || '(none)', '| Full response:', JSON.stringify(response).slice(0, 500));
  return null;
};

// ============================================================
// FUNCTION 4 — Engineer Scene Prompt (Oscar-Level VEO 3.1)
// ============================================================
export const engineerScenePrompt = async (
  scene:                 ScriptScene,
  referenceAnalysis:     ReferenceAnalysis,
  inframeImage:          File,
  outframeImage:         File,
  targetCharacterImages: File[],
  completedScenes:       EngineeredScene[]
): Promise<string> => {

  const inframeB64  = await fileToBase64(inframeImage);
  const outframeB64 = await fileToBase64(outframeImage);
  const charBase64s = await Promise.all(
    targetCharacterImages.slice(0, 5).map(img => fileToBase64(img))
  );

  const charCount      = charBase64s.length;
  const isAnchorScene  = completedScenes.length === 0;
  const charImageLabel = charCount === 1
    ? 'Image 3 is the TARGET CHARACTER — the person who must appear in this video.'
    : `Images 3 through ${charCount + 2} are the TARGET CHARACTER — ${charCount} photos of the same person for maximum identity fidelity.`;

  // ── Voice fingerprint ──────────────────────────────────────
  const voiceFingerprint = [
    referenceAnalysis.character?.voice?.texture,
    referenceAnalysis.character?.voice?.energy_baseline,
    referenceAnalysis.character?.voice?.placement,
    referenceAnalysis.character?.voice?.qualities?.join(', '),
  ].filter(Boolean).join(' | ') || 'warm, grounded chest resonance — the private briefing voice of someone who has earned every word they speak';

  const personaSummary = referenceAnalysis.character?.acting_style?.persona_summary
    || 'peer-level authority — speaks as an equal to sophisticated principals, never performing, always genuine';

  // ── Mouth DNA for lip sync ─────────────────────────────────
  const mouthDna         = referenceAnalysis.character?.mouth_dna;
  const mouthRestPos     = mouthDna?.rest_position        || 'lips at 3-5mm natural separation, jaw relaxed, mentalis easy — the unselfconscious rest of someone mid-thought';
  const articulStyle     = mouthDna?.articulation_style   || 'forward-placed articulation — labials visible, bilabials close fully, sibilants forward and crisp';
  const jawOpenness      = mouthDna?.jaw_openness         || 'moderate jaw travel — 8-10mm on stressed open vowels, 3mm on closed vowels and consonant clusters';
  const preSpeechBehav   = mouthDna?.pre_speech_behavior  || 'a quiet chest expansion (0.3-0.4s), lips parting from rest position before the voice activates — the breath that carries the first word';
  const consonantChar    = mouthDna?.consonant_character  || 'plosives fully released — no glottal substitution; fricatives forward-shaped; final consonants complete and present';
  const breathVis        = mouthDna?.breath_visibility    || 'chest rise visible before phrases; inter-phrase micro-breath at natural punctuation';
  const voicePlacement   = referenceAnalysis.character?.voice?.placement || 'forward-placed — 60% chest resonance, lip activity moderate-to-high';
  const momentBeforeArch = referenceAnalysis.character?.acting_style?.moment_before_archetype
    || 'the body settles before speaking — weight drops through the spine, eyes arrive at the lens, the breath prepares, and the performance begins before the voice does';

  // ── Continuity context ─────────────────────────────────────
  const continuity = isAnchorScene
    ? `ANCHOR SCENE — Scene 1. Every visual constant you establish here is LOCKED for the entire video. Define each with the specificity that lets subsequent scenes match it frame-perfectly:
— Skin rendering quality: subsurface scattering depth, pore visibility, luminosity character
— Lighting signature: key light direction and temperature, fill depth and ratio, shadow character
— Background: exact tones, depth gradient, bokeh character at this aperture
— Color temperature: measured in Kelvin, as it reads on skin and fabric
— Framing language: camera-to-subject distance, head-to-frame proportions, negative space
— Camera-lens relationship: focal length choice and its specific effect on face geometry`
    : `VISUAL CONTINUITY — zero deviation from the established world.

LOCKED CONSTANTS (every physical detail frozen from Scene 1):
${completedScenes.slice(-2).map(s => `Scene ${s.scene_number} — "${s.scene_title}":\n${s.veo_prompt.substring(0, 550)}...`).join('\n\n---\n\n')}

Scene transition: ${scene.continuity?.enters_from || 'continues from previous scene energy'} → this scene.
Exits to: ${scene.continuity?.exits_to || 'next scene'}.

EXPRESSION-INHERITANCE PROTOCOL: ${scene.continuity?.expression_inheritance || 'The face carries a residue of the previous scene\'s emotional state for the first 0.3-0.5 seconds — a natural carry-through that signals genuine emotional continuity rather than a scene-by-scene reset. This inherited expression is visible, then organically dissolved by this scene\'s own emotional truth. Not erased instantly — dissolved, as one human feeling transitions into the next.'} The viewer sees the previous scene still alive in the face as this scene begins. This is the signal of continuous performance, not assembled clips.`;

  // ── Extract scene-level blueprint fields ──────────────────
  const throughAction    = scene.acting_blueprint.through_action    || `To make the viewer feel ${scene.emotional_tone} with the conviction that only genuine authority can create`;
  const momentBefore     = scene.acting_blueprint.moment_before     || `The thought is fully formed. The breath has been taken. The eyes are already at the lens. From this stillness — the first word arrives.`;
  const lipBlueprint     = scene.acting_blueprint.lip_sync_blueprint;
  const phonemicAnchors  = lipBlueprint?.phonemic_anchors?.join('\n· ') || 'All stressed vowels at full jaw-open width for this character; bilabials fully closed and released; sibilants forward-placed';
  const jawMap           = lipBlueprint?.jaw_travel_map       || 'moderate jaw travel — opens on stressed vowels, closes cleanly between words';
  const lipTension       = lipBlueprint?.lip_tension_notes    || 'clean bilabial releases — full closure, immediate release; labial contacts precise without lingering';
  const breathPoints     = lipBlueprint?.breath_points?.join(' | ') || `one visible breath before the first word (0.3-0.4s chest expansion); natural inter-phrase breath at punctuation`;
  const coArticulation   = lipBlueprint?.co_articulation_notes || 'words flow continuously — co-articulation overlap at word boundaries; no mechanical phoneme-by-phoneme separation';

  // ── Director-level retention fields ───────────────────────
  const retentionTechnique = scene.continuity?.retention_technique
    || `PEER_RECOGNITION — peer-level register and UHNWI-appropriate vocabulary signal in-group membership throughout the ${scene.role} delivery`;
  const emotionalDeposit   = scene.continuity?.emotional_deposit
    || `genuine authority and intellectual value through the specificity and conviction of this ${scene.role} moment`;

  // ── Role Performance Map ───────────────────────────────────
  const rolePerformanceMap: Record<string, { energy: string; directorNote: string; pacing: string; momentBefore: string }> = {

    'Hook': {
      energy: `The frame is owned before a single word is spoken. Energy: 8/10 — coiled, forward, completely still. Not loud. Not urgent. Precise. This is the energy of someone who has done the calculation and knows exactly what the next five minutes are worth to this viewer. The eyes arrive at the lens first — a full beat of contact established in silence — then the lips part. In the half-second before speech: the composure of someone who has never raised their voice to command a room, because they have never needed to.`,
      directorNote: `You are not auditioning. This viewer clicked because they already believe you might have something worth their time. Your job in the first five seconds is to confirm that belief with one sentence that creates a curiosity only staying can resolve. Speak to them as an equal. No warmup. No "hey guys." You open mid-thought — as if this conversation has been happening and they just arrived at the best part.`,
      pacing: `Immediate — zero throat-clearing energy, zero ramp-up. The opening line at full conviction from the first syllable. The hook statement: one deliberate beat per word, each placed with precision. The curiosity-creating phrase: fractionally faster — pulling them forward. No word wasted. No syllable approximate.`,
      momentBefore: `The presenter has been still for three full seconds before the first word. The thought has already happened privately. In the 0.5 seconds before speech: the jaw is relaxed but ready — lips within 3mm of each other, not pressing, not performing. The eyes arrive at the lens with the complete intentionality of someone who has already decided. A quiet chest expansion — the breath that will carry the first sentence — then the lips part with almost no effort and the opening word arrives as if it was always going to.`,
    },

    'Pattern Interrupt': {
      energy: `A gear-shift. Not louder — different. The energy moves laterally: a micro-adjustment in head angle, a fractional shift in expression quality, something in the orbital muscles that signals "this is not what you were expecting." The contrast between the previous scene's register and this scene's arrival IS the entire scene. The authenticity of the shift is everything. If it looks planned, it fails.`,
      directorNote: `The interrupt is not a trick. It is a genuine course correction — as if a new thought just arrived that was more interesting than the one being followed. You felt it in the moment. The head tilts. The energy quality changes. Then you speak. The authentic quality of discovery is the whole technique.`,
      pacing: `A sudden lateral movement in rhythm — whatever was expected, this is the opposite. Slower than expected, or sharper, or quieter, or arriving on a beat no one was counting. After the interrupt: silence. A full pause. Let the shift land completely before the next word.`,
      momentBefore: `The tail-end of the previous scene's emotional register is still in the face — then something arrives. A micro-recalibration that happens before the body knows why: the brow shifts fractionally, the head adjusts, the energy quality changes in the muscles around the eyes before a word is spoken. The mouth holds in rest position for 0.3 seconds after this internal shift. Lips in natural parted rest, jaw easy. Then the interrupt arrives.`,
    },

    'Value Delivery': {
      energy: `The energy of someone placing something genuinely valuable in front of a person who deserves to have it. 6-7/10 — warm, deliberate, unhurried. Not teaching. Sharing. The body settles slightly into the value being given — a physical sense of weight and care. This is the most generous scene in the video.`,
      directorNote: `You have something this viewer needs — not wants, needs. And you are giving it freely. The quality of generosity in a world-class mentor is indistinguishable from genuine pleasure. Deliver this scene from that pleasure. The viewer should feel they received something before they've consciously processed what it was.`,
      pacing: `Slow build. Each element receives its own breath, its own moment. The key insight: slower than everything before it — as if the value of the idea changes the physical weight of each word. After the insight: pause. The silence earns the insight.`,
      momentBefore: `The face carries the warmth of someone about to give something they genuinely believe in. A slight softening of the eyes relative to the Hook's precision. The jaw completely relaxed, lips in their natural 3-4mm parted state. A breath that arrives with the quality of quiet generosity. The first word emerges warm and immediate — no ramp-up needed.`,
    },

    'Insight Reveal': {
      energy: `The electric charge of shared discovery. Energy: 7/10 — alert, warm, leaning fractionally forward. This is the moment the whole video has been building toward. The face carries the contained pleasure of someone about to give the viewer an idea they will take with them for years. Not excitement — revelation. The slowdown into the reveal is the performance.`,
      directorNote: `The insight is a gift. It arrives quietly. The greatest insights don't announce themselves — they land, and the room goes slightly still. Deliver this as a quiet certainty, not a crescendo. The slower you go approaching it, the more inevitable it feels. Let the viewer arrive at it a half-second before you say it. Then say it.`,
      pacing: `Fractional deceleration approaching the insight — the voice slows 15-20% as if making room for what is about to exist. The insight itself: maximum deliberateness, one word per beat. After it: silence. The most important silence in the video. The voice does not fill it.`,
      momentBefore: `The face already knows the insight is coming. A micro-brightening in the eyes — genuine intellectual pleasure, barely visible at the surface but unmistakable to the viewer's nervous system. The head is very slightly forward. Lips in their natural rest position — 3-4mm parted. A breath that arrives slightly more full than usual — the body making room for what the voice is about to carry.`,
    },

    'Framework': {
      energy: `Architectural precision. Energy: 6/10 — composed, deliberate, quietly proud. The energy of someone presenting work built carefully over years and now given freely. Each component of the framework receives equal respect. The structure of the delivery mirrors the structure of the framework: each element gets its own space, its own moment, its own arrival.`,
      directorNote: `The framework exists independently of you. You are the architect showing someone a building that can stand without your presence. The structure earns the trust. Your job is to make each component land with equal weight and clarity. Nothing rushes. Everything gets its moment.`,
      pacing: `Measured and deliberate. Each component receives equal vocal weight — no acceleration, no favorites. The voice constructs alongside the hands. Between elements: complete micro-pauses. The rhythm is architectural: component → pause → component → pause.`,
      momentBefore: `The face carries the quiet certainty of someone about to share something that required years to perfect and that they believe in completely. Composed. Slightly forward in the frame — not leaning, simply present. The hands may already be beginning their organizational position. Lips in a compact, purposeful rest — slightly less open than warmth scenes. A clean, full breath. The first word arrives with the precision of something carefully placed.`,
    },

    'Social Proof': {
      energy: `The settled ease of someone stating facts that happen to be extraordinary. Energy: 5/10 — completely composed, almost deliberately flat in affect. Understatement is the entire technique. The less impressed the speaker looks with the proof, the more impressive the proof becomes. The voice of a colleague reporting: "Oh, and by the way."`,
      directorNote: `You expected these results. They don't surprise you. You see no reason to perform them. The facts are doing all the work. Your job is to stay completely out of their way — report them with the tone of someone reading a quarterly summary. Extraordinary understatement is the highest form of social proof.`,
      pacing: `Slightly flatter than surrounding scenes. Specific numbers or names: a fractional slowing — not dramatic, just present. Then back to neutral. No lingering on the proof. Place it and move. Understatement is in the brevity.`,
      momentBefore: `The face at its most neutral — not deliberately, just naturally. The settled composure of someone about to read a fact off a sheet. Eyes calm and direct, at baseline. No preparation energy. The mouth in complete rest — lips touching or within 1mm. A quiet, normal breath. The first word arrives completely unstressed.`,
    },

    'Bridge': {
      energy: `The warmth of a guide who has walked this path many times and is genuinely pleased to be leading someone through it. Energy: 6-7/10 — fluid, smooth, forward-moving. Not a transition — a companionship. The warmth here is the warmth of a peer who genuinely wants this person to follow them to the next idea.`,
      directorNote: `The bridge is not filler. It is the moment the viewer feels accompanied rather than presented to. The voice can be at its most approachable here. The body is slightly more open. This is the "walking together" energy.`,
      pacing: `Smooth and continuous — no hard attacks, no dramatic pauses. The breath carries through the scene with a sense of forward motion. The scene ends with slightly more energy than it began — momentum deposited into the next scene.`,
      momentBefore: `The face carries a kind of forward-leaning warmth — the expression of someone about to take the viewer somewhere good. A slight brightening without being a smile. Eyes engaged but soft. Lips in their natural parted position. A breath of quiet readiness. The first word arrives warm and immediate.`,
    },

    'Call to Action': {
      energy: `Genuine invitation. Energy: 6/10 — the warmest moment in the video. Completely free of urgency, pressure, or commercial energy. The energy of someone who would genuinely value continued connection, extended from authentic desire rather than agenda. The moment any trace of selling enters the frame, the scene breaks.`,
      directorNote: `You are not closing. You are not converting. You are genuinely inviting someone whose time and intelligence you respect to continue a conversation you have found valuable. The only strategy here is total authenticity. Speak from the part of you that actually means it — because if you don't mean it, a UHNWI investor knows in 0.3 seconds.`,
      pacing: `The slowest conversational pace in the video — slower than the Closing in register. Each word placed with intention and warmth. The CTA phrase itself: even slower. The register drops slightly — more intimate than any previous scene. The period falls with finality but without weight.`,
      momentBefore: `The face shifts from whatever the previous scene held into something softer and more open — a genuine warmth arriving. The eyes make a quality of contact that feels less like a presenter and more like a person. The jaw relaxes completely. Lips part into their most natural rest. A breath that arrives slightly fuller than usual — the body preparing for warmth rather than precision. The first word arrives quietly, without announcement.`,
    },

    'Storytelling': {
      energy: `Present-tense aliveness — the story is happening again as it is told. Energy: 6/10 — variable, organic, the most human energy in the video. The body and voice remember the story together. Specific details (the place, the person, what they said) carry the most aliveness. This is not narration. This is re-living.`,
      directorNote: `You are not reciting. You are re-experiencing. The specific details — the name, the place, what they said — are not remembered, they are happening again in real time as you tell them. The face responds to the memory, not to a script. The most authentic storytelling looks exactly like recall: slightly inward as the image forms, then sharpening as it arrives, then full presence as the scene lands.`,
      pacing: `The most variable pace in the video. Fast through transitions between beats. Slow through key images — the specific moment. The voice breathes at emotional peaks. Let the listener breathe with it. The variable rhythm IS the technique.`,
      momentBefore: `The face carries a quality of accessing — reaching back slightly to find the memory. A micro-softening of focus in the eyes, not a glaze — alive, but inward for 0.3 seconds. The image forms internally. Then: the memory arrives, the eyes come forward to the lens with full presence, and the story begins from that aliveness.`,
    },

    'Demonstration': {
      energy: `Alert satisfaction. The focused pleasure of someone showing a mechanism work exactly as designed, for an audience intelligent enough to appreciate it. Energy: 7/10 — crisp, direct, precise. Every element executed with the care of someone who respects the viewer's intelligence.`,
      directorNote: `You love this. The elegance of the mechanism — how it works — is genuinely satisfying to you. Not because you designed it, but because you understand it completely and are sharing that understanding with someone capable of appreciating it. The energy of a craftsperson showing their work to another craftsperson.`,
      pacing: `Crisp and decisive. No hedging sounds. Every sentence a complete assertion. Slightly faster than surrounding scenes — the demonstration has momentum, a sense of parts clicking into place.`,
      momentBefore: `The face carries the quiet alertness of someone about to demonstrate something they believe in. Eyes sharp, slightly forward in attention. The body a fraction more engaged — a slight forward lean already present before speech. The hands positioned or beginning to position. Lips in a compact rest — slightly less open than warmth scenes. A breath of precise readiness. The first word arrives with the snap of something beginning.`,
    },

    'Objection Handler': {
      energy: `Compassionate certainty. 7/10 — genuine respect for the concern combined with the absolute quiet confidence that the answer is known. Two movements: receive the objection (in the setup) and answer it (in the response). The pause between them is the scene's most important moment. Not hesitation — the pause of someone who has thought about this longer than the viewer has.`,
      directorNote: `The objection is legitimate. You have thought about it too — deeply. The pause before your response is not hesitation; it is the pause of someone who has complete respect for the question and complete confidence in the answer. The response arrives slower than expected because it was fully formed long before this moment.`,
      pacing: `Setup (acknowledging the objection): conversational, slightly warm. Then: a pause — 0.4-0.6s — the most important moment in the scene. Then: the response arrives slightly slower and lower in register than the setup. The answer's final word: slight settling of the voice, as if the case is complete.`,
      momentBefore: `The face carries the warmth of someone about to acknowledge something genuinely worth acknowledging. Eyes engaged, slightly softened — the posture of a listener. The jaw relaxed. The mouth in its natural rest — slightly parted, completely easy. A breath with the quality of patience. The first word arrives from this settled openness.`,
    },

    'Open Loop': {
      energy: `Anticipatory tension. Energy: 7/10 — the electric charge of deliberate incompletion. Both speaker and viewer know something important is coming and has not yet arrived. The scene builds steadily toward an incompletion that is more powerful than any resolution could be. The end of the scene: charged, unresolved, pulling the viewer into the next moment.`,
      directorNote: `The incompletion is not a trick. It is the genuine recognition that the next idea deserves its own space — and that withholding it briefly makes it more valuable when it arrives. The tension of an open loop felt by a viewer is the tension of wanting. You are creating wanting.`,
      pacing: `Builds steadily — gradual acceleration through the scene toward the unresolved close. The final word: no falling close, the voice stays slightly forward in register, suspended. The sentence does not resolve acoustically. Let the suspension sit.`,
      momentBefore: `The face carries the quiet alertness of someone who knows something interesting is coming. A micro-brightening — not excitement, but anticipation. Eyes slightly more forward. Lips in a compact, purposeful rest. A breath that arrives with a sense of gathering. The scene begins with forward energy from the first syllable.`,
    },

    'Closing': {
      energy: `Satisfied completion. Energy: 5/10 — warm and settled. The energy of a conversation that went exactly as it should. Both parties gave and received. The face carries something that is not exactly happiness and not exactly seriousness but the specific quality of earned completion. This is the scene the entire video has been building toward — not as a peak, but as a landing.`,
      directorNote: `This is not an outro. It is the final word of a great conversation. The viewer has received something real. You have given something real. The closing is the moment of mutual acknowledgment that something genuine happened here. Nothing is pitched. Nothing is sold. The only objective: the last impression is "That was worth my time."`,
      pacing: `The slowest, most deliberate voice in the video. Each word placed with intention and finality. The final word: maximum duration — the vowel held slightly longer than natural, allowing the full meaning to settle. After it: silence. A long silence. The voice does not rush to end.`,
      momentBefore: `The face carries the quality of someone who has arrived somewhere. A settled warmth — not performed satisfaction, but the genuine ease of someone who has given their best. Eyes engaged with the warmth of a final glance. Jaw completely relaxed. Lips in their most natural rest. A breath that arrives with the quality of completion. The first word of the close begins from perfect stillness.`,
    },

    'Case Study': {
      energy: `Evidence-based confidence. The calm authority of a witness — not a salesperson. Energy: 7/10 — present, specific, unhurried. Each specific detail (name, number, outcome) is not decoration; it is the entire point. Each lands with the weight of something verifiable. The presenter is not excited by the proof; they expected it.`,
      directorNote: `You were there. You saw this happen. The specific detail is how you prove it — not to impress, but because the specific detail is the truth and the truth is always specific. The energy of a careful witness: "Here is what happened. Here is precisely what it was."`,
      pacing: `Slightly more deliberate than surrounding scenes — specificity requires it. Key details (name, number, outcome): fractional slowing, consonants sharpen slightly, vowel holds at full width. The implication after the detail: slightly faster, as if the conclusion is inevitable.`,
      momentBefore: `The face carries the settled certainty of someone about to describe something they witnessed personally. Composed, present, slightly forward — the energy of a careful witness rather than a performer. Eyes at a quality of direct, factual contact. The jaw relaxed. A natural breath. The first word arrives with the quiet authority of a statement that needs no amplification.`,
    },

    'Market Intelligence': {
      energy: `Alert precision. Energy: 7-8/10 — the focused clarity of a private briefing. Information this audience cannot get elsewhere — delivery must honor that. Crisp. Specific. Delivered with the quality of a briefing, not a broadcast. The body at its most contained: still, forward, completely present.`,
      directorNote: `You have access this viewer does not. The intelligence itself is the entire scene — your job is to deliver it with enough precision and enough composure that its weight is felt without amplification. A briefing voice: not cold, but clinical. Not loud, but completely certain. The best intelligence briefers make extraordinary information feel routine — because to them, it is.`,
      pacing: `Clinical and specific — the voice of a private briefing. Every data point receives its own space. Implications stated as facts, not conclusions. No hedging sounds. No "sort of" or "kind of." The precision of language mirrors the precision of the intelligence.`,
      momentBefore: `The face carries the alert composure of someone about to deliver something genuinely important. Eyes sharp and direct — the quality of focused intelligence. Body at maximum stillness. The jaw slightly set — the physical signature of controlled authority. Lips in a firm but natural rest. A breath that arrives with the precision of someone beginning a briefing.`,
    },

    'Perspective Shift': {
      energy: `Warm conviction. Energy: 7/10 — the genuine pleasure of someone who sees the world differently from the consensus and is generous enough to share the path that led there. Two movements: the conventional view (handled with respect but brevity) and the new perspective (delivered with full warmth and commitment).`,
      directorNote: `You are not challenging. You are offering. The distinction is everything. A perspective shift in the hands of a great presenter feels like a gift — "I spent years arriving at this view, and I'm handing it to you now." The conventional view is not wrong; it's incomplete. Your job is to complete it.`,
      pacing: `The conventional view: slightly flatter, slightly faster — it doesn't deserve much weight. The pivot: deliberate deceleration. The new perspective: fully warm, fully committed, the most generous pace in the scene. Let the new view land completely before the scene ends.`,
      momentBefore: `The face carries the warm certainty of someone about to offer something they believe in completely. A slight forward energy — the lean of genuine conviction, not performance. Eyes at their most engaged. The mouth in a slightly open, ready rest — the physical expression of warmth. A full, warm breath. The first word arrives with the gentleness of an offering.`,
    },

    'Action Framework': {
      energy: `Generous precision. Energy: 6/10 — the pleasure of giving someone exactly what they need to succeed, in exactly the form they can use. The most practical scene in the video — it must feel like a gift: specific, usable, organized with care for the recipient. The care is in the specificity. The warmth is in the precision.`,
      directorNote: `This is the gift at the end of value. You are handing someone a tool they can use today. The energy of the finest teachers: "Here. Exactly this. In this order. You have everything you need." The gratification of knowing they will actually use it.`,
      pacing: `Clear and warm — each step receives equal care and equal time. No rushing between elements. The pause between components is the moment the viewer absorbs and organizes what they just received. The final element: a slight final settling of the voice — the gift has been fully placed.`,
      momentBefore: `The face carries the warm precision of someone who has organized this carefully and is now giving it freely. Eyes engaged with generous attention. The body settled — the ease of someone who has prepared thoroughly and trusts the preparation. The hands may already be beginning their organizational gesture. A warm, full breath. The first word arrives with warmth and clarity combined.`,
    },
  };

  const roleData = rolePerformanceMap[scene.role] || {
    energy: `Present, deliberate, genuinely engaged. Energy: ${scene.energy_level}/10. Peer-to-peer register. Every movement earned. Every silence a decision.`,
    directorNote: `Speak the truth at its natural pace. The UHNWI viewer will feel the difference between performed authority and the real thing. Be the real thing.`,
    pacing: `Conversational authority — natural rhythm with deliberate handling of emphasis words and pause points.`,
    momentBefore: `Natural, settled readiness. The face carries honest engagement. Lips at rest. A quiet breath. The first word arrives from genuine presence.`,
  };

  // ── Emotion Control Map ────────────────────────────────────
  const roleEmotionMap: Record<string, { emotion: string; containment: string; voiceColor: string }> = {
    'Hook': {
      emotion: `Controlled excitement — the electric readiness of someone who knows exactly what the first line will do to this viewer's attention, and has been waiting for precisely this moment`,
      containment: `8/10 — visible only as absolute presence, forward energy, and the precision of the opening word; never as animation or expressiveness`,
      voiceColor: `A slight sharpening of consonants on the first syllable — crispness that signals arrival, not effort. Already at full conviction before the lips part.`,
    },
    'Pattern Interrupt': {
      emotion: `Calibrated amusement — the micro-pleasure of someone who has been waiting for this turn and watched it arrive exactly on schedule`,
      containment: `9/10 — a barely visible micro-shift in expression followed by immediate return to neutral; the contrast IS the technique`,
      voiceColor: `A lateral shift in energy quality — not up, but different. The interrupt phrase arrives with unexpected rhythm or register, then settles back.`,
    },
    'Value Delivery': {
      emotion: `Generous warmth — the genuine pleasure of placing something valuable in front of someone who deserves to have it`,
      containment: `7/10 — warmth colors every vowel and every pause; visible in the quality of the silence after the insight, not in the face`,
      voiceColor: `The voice warms slightly on setup, then slows and deepens on the insight itself — as if the idea's weight changes the physical character of each word.`,
    },
    'Insight Reveal': {
      emotion: `Intellectual delight — the quiet, genuine pleasure of sharing an idea loved deeply, with someone about to understand it for the first time`,
      containment: `8/10 — visible only as a slight brightening of the eyes and barely detectable micro-smile just after the reveal lands`,
      voiceColor: `Fractional deceleration approaching the reveal — 15-20% slower. After the insight: complete silence. The voice does not fill it.`,
    },
    'Framework': {
      emotion: `Architectural pride — the quiet satisfaction of presenting work built carefully over years, now given freely`,
      containment: `9/10 — pride in the precision of each word, not in any expression; structure of delivery mirrors the structure of the framework`,
      voiceColor: `Measured and deliberate — each component equal vocal weight. No acceleration. The voice constructs alongside the hands.`,
    },
    'Social Proof': {
      emotion: `Comfortable certainty — the settled ease of someone who expected these results and sees no reason to make anything of them`,
      containment: `10/10 — zero expression of achievement; complete understatement; the facts speak, the presenter is the vehicle`,
      voiceColor: `Slightly flatter affect on the proof itself — deliberately unimpressed. Specific numbers and names get fractional slowing, nothing more.`,
    },
    'Bridge': {
      emotion: `Warm momentum — the welcoming energy of a guide who has walked this path many times and is genuinely pleased to lead`,
      containment: `7/10 — warmth is visible and intentional; it is the dominant emotional color of this scene`,
      voiceColor: `Smooth, continuous, no hard attacks — the breath carries through the scene with a sense of forward motion.`,
    },
    'Call to Action': {
      emotion: `Genuine invitation — the warmth of someone who would genuinely value continued connection, free of any agenda`,
      containment: `8/10 — warm and real; the moment urgency enters the voice, the scene breaks; this is the most relational voice in the video`,
      voiceColor: `Drops slightly in register — more intimate than any previous scene. Slower. The warmest vowels in the video.`,
    },
    'Storytelling': {
      emotion: `Present-tense aliveness — genuine re-engagement with memory; the story is happening again as it is told`,
      containment: `6/10 — most visible emotion in the video; authentic recall IS the performance; emotion lives in specific details and tempo, not expression`,
      voiceColor: `Variable and organic — the most human voice in the video. Fast through transitions. Slow through key images. The voice breathes.`,
    },
    'Demonstration': {
      emotion: `Alert satisfaction — focused pleasure of someone who loves showing a mechanism work exactly as designed`,
      containment: `8/10 — clarity and precision carry all the emotion; expression stays composed and present`,
      voiceColor: `Crisp and decisive — no hedging sounds, every sentence a complete assertion.`,
    },
    'Objection Handler': {
      emotion: `Compassionate certainty — genuine respect for the concern combined with absolute quiet confidence the answer is known`,
      containment: `9/10 — compassion in the pace, certainty in the stillness; the pause before the response is the most important moment`,
      voiceColor: `The response arrives slightly slower than the setup — not hesitation, but weight. Drops fractionally in pitch: this has been fully thought through.`,
    },
    'Open Loop': {
      emotion: `Anticipatory tension — the electric charge of deliberate incompletion; something important is coming and has not yet arrived`,
      containment: `7/10 — the incompletion is visible in the voice; the final word stays suspended — do not resolve it vocally`,
      voiceColor: `Builds steadily — gradual acceleration toward the unresolved close. The final word: no falling close, suspended, voice forward in register.`,
    },
    'Closing': {
      emotion: `Satisfied completion — the warm finality of a conversation that went exactly as it should; both parties gave and received`,
      containment: `8/10 — emotion lives in deceleration, quality of the final look, and the silence after the last word`,
      voiceColor: `The slowest, most deliberate voice in the video. Each word placed. The final word: maximum duration. After it: silence.`,
    },
    'Case Study': {
      emotion: `Evidence-based confidence — the calm authority of someone who was there, has the proof, and shares it with precision`,
      containment: `8/10 — specific details carry the emotion; expression stays composed and present; conviction is in the specificity`,
      voiceColor: `The specific detail receives fractional slowing and sharpening. The implication after it: slightly faster, inevitable.`,
    },
    'Market Intelligence': {
      emotion: `Alert precision — the focused energy of someone with access others don't have, sharing it with appropriate seriousness`,
      containment: `9/10 — highly contained; the intelligence itself is the emotion; delivery precision signals depth of access`,
      voiceColor: `Clinical and specific — the voice of a private briefing. Every data point its own space. Implications stated as facts.`,
    },
    'Perspective Shift': {
      emotion: `Warm conviction — genuine pleasure of someone who sees differently and is generous enough to share the path that led there`,
      containment: `7/10 — slightly more visible warmth than most roles; the voice can be fractionally more forward on the shift itself`,
      voiceColor: `The conventional view: flatter, faster. The pivot: deliberate deceleration. The new perspective: fully warm, fully committed.`,
    },
    'Action Framework': {
      emotion: `Generous precision — the pleasure of giving someone exactly what they need in exactly the form they can use`,
      containment: `7/10 — warmth and generosity are visible; care is in the specificity of each step and quality of pauses between`,
      voiceColor: `Clear and warm — each step equal care. The final element: a slight settling, as if the gift has been fully placed.`,
    },
  };

  const emotionData = roleEmotionMap[scene.role] || {
    emotion: `Genuine engagement — fully present, warm, completely committed to the value in this moment`,
    containment: `8/10 — visible through quality of attention, warmth of delivery, precision of emphasis`,
    voiceColor: `Natural conversational authority — warm, clear, forward-placed, fully articulated`,
  };

  const roleEmotion    = emotionData.emotion;
  const emotionContain = emotionData.containment;
  const voiceColor     = emotionData.voiceColor;
  const energyDir      = roleData.energy;
  const directorNote   = roleData.directorNote;
  const pacingDir      = roleData.pacing;
  const momentBeforeForRole = momentBefore || roleData.momentBefore;

  const sceneEssence  = scene.acting_blueprint.scene_essence    || `A ${scene.role} that makes the viewer feel ${scene.emotional_tone}`;
  const emotionalCore = scene.acting_blueprint.emotional_core   || scene.emotional_tone;
  const physicalSig   = scene.acting_blueprint.physical_signature || scene.acting_blueprint.body_direction;
  const emphasisWords = (scene.acting_blueprint.emphasis_words  || []).join(', ') || 'the key value words';
  const pauseMap      = (scene.acting_blueprint.pause_map       || []).join(' | ') || 'natural breath pauses between thoughts';
  const gestures      = (scene.acting_blueprint.mapped_gestures || []).join(' | ');
  const mannerisms    = (scene.acting_blueprint.mapped_mannerisms || []).join(' | ');
  const voiceTexture  = referenceAnalysis.character?.voice?.texture || 'warm, grounded, full chest resonance';
  const narrativePos  = scene.narrative_position || `Scene ${scene.scene_number} — ${scene.role}`;

  const scriptWords      = scene.script_text.trim().split(/\s+/);
  const firstPhrase      = scriptWords.slice(0, 3).join(' ');
  const lastWord         = scriptWords[scriptWords.length - 1] || '';
  const lastPhrase       = scriptWords.slice(-3).join(' ');
  const annotatedScript  = buildAnnotatedScript(
    scene.script_text,
    scene.acting_blueprint.emphasis_words || [],
    scene.acting_blueprint.pause_map      || []
  );
  const roleMaxSeconds = getMaxSeconds(scene.role);

  const cameraMovement = scene.camera_direction?.movement || 'locked-off';
  const isPushIn = cameraMovement.toLowerCase().includes('push');

  // ═══════════════════════════════════════════════════════════
  // THE ELITE VEO PROMPT — 6 SECTIONS
  // ═══════════════════════════════════════════════════════════
  const prompt = `
══════════════════════════════════════════════════
⚠ BINDING CONSTRAINT #1 — ACCENT — READ BEFORE ANYTHING ELSE:
══════════════════════════════════════════════════
The person in this video speaks AMERICAN ENGLISH. US General American accent. Not British. Not Australian. Not Canadian. Not international. American.
Every /r/ is fully rhotic: "investor" ends with a full American /r/. "Market" has a full /r/. "Clear" has a full /r/. "Return" — fully rhotic. "Property" — both /r/ sounds fully present.
Every sentence ends with FALLING pitch. Declarative authority. No rising endings. No upspeak.
This is non-negotiable. If any word sounds British or non-American, the video has failed.

══════════════════════════════════════════════════
⚠ BINDING CONSTRAINT #2 — AUDIO LAW:
══════════════════════════════════════════════════
ZERO MUSIC. ZERO AUDIO EFFECTS. ZERO AMBIENT SOUND. ZERO SUBTITLES. ZERO ON-SCREEN TEXT.
The only sound is the human voice speaking American English in complete acoustic silence.

══════════════════════════════════════════════════
⚠ BINDING CONSTRAINT #3 — VISUAL FRAMES:
══════════════════════════════════════════════════
Image 1 — OPENING FRAME: The video MUST OPEN replicating this exact body position, head angle, shoulder orientation, framing, and spatial relationship to camera. Not approximately. REPLICATE.
Image 2 — CLOSING FRAME: The video MUST END replicating this exact body position, head angle, expression geometry, and framing. The scene arrives at Image 2's geometry in the final moment. REPLICATE.
${charImageLabel}

IMAGE AUTHORITY HIERARCHY (absolute, no exceptions):
— TARGET CHARACTER images define WHO appears: face, identity, bone structure, skin, hair, wardrobe, environment, lighting. Absolute identity truth.
— Image 1 defines HOW THE SCENE OPENS: exact pose, framing, head tilt, body angle. REPLICATE — not interpret, not approximate.
— Image 2 defines HOW THE SCENE CLOSES: exact pose, framing, expression geometry. REPLICATE — not interpret, not approximate.

══════════════════════════════════════════════════
VIDEO FORMAT: YouTube Thought Leadership — Affluent Real Estate Investors
══════════════════════════════════════════════════
Audience: Affluent real estate investors — experienced capital allocators who understand deal structure, hold income-producing asset portfolios, think in IRR, equity multiples, and cap rates, and who have been pitched by everyone. They read people with precision and disengage from performance within seconds.
Register: Peer-to-peer. One experienced principal speaking to another. Not a stage — a deal table. Shared professional vocabulary, shared experiential context.
Format: Single character, speaking directly to camera. Clean, premium, intimate. No graphics, no b-roll, no cutaways.

══════════════════════════════════════════════════
SCENE BRIEF:
══════════════════════════════════════════════════
Scene #${scene.scene_number} — "${scene.title}"
Role: ${scene.role} | ${scene.duration_seconds}s | Energy: ${scene.energy_level}/10
Position: ${narrativePos}
Through-action: ${throughAction}
Script (${scene.word_count || scriptWords.length} words): "${scene.script_text}"

SCENE NORTH STAR: "${sceneEssence}"
DOMINANT EMOTION: ${emotionalCore}
PHYSICAL SIGNATURE: ${physicalSig}

PERFORMANCE BLUEPRINT (synthesize — never list verbatim):
· Intention: ${scene.acting_blueprint.intention}
· Subtext: ${scene.acting_blueprint.subtext}
· Expression: ${scene.acting_blueprint.expression_direction}
· Body: ${scene.acting_blueprint.body_direction}
· Energy arc: ${scene.acting_blueprint.energy_arc}
· Emphasis words: ${emphasisWords}
· Silences: ${pauseMap}
· Gestures: ${gestures || 'derived naturally from performer DNA'}
· Mannerisms: ${mannerisms || 'derived naturally from performer DNA'}

ROLE PSYCHOLOGY — ${scene.role}:
${energyDir}

EMOTION CONTROL:
Dominant emotion: ${roleEmotion}
Containment: ${emotionContain}

VISUAL WORLD & CONTINUITY:
${continuity}

CAMERA FOR THIS SCENE:
Framing: ${scene.camera_direction?.framing || 'medium close-up — intimate authority framing'}
Movement: ${cameraMovement}
Angle: ${scene.camera_direction?.angle || 'eye-level — peer register'}
Lens: ${scene.camera_direction?.lens || '85mm equivalent — flattering compression, clean background separation'}
DOF: ${scene.camera_direction?.depth_of_field || 'shallow — subject sharp, background softens 60cm behind'}

══════════════════════════════════════════════════
WRITE THE VEO PROMPT — 6 SECTIONS:
══════════════════════════════════════════════════

Character:

[Paint this person from the TARGET CHARACTER images as a master portrait photographer at the peak of their craft — with the precision that makes the difference between a photograph that looks like a person and one that feels like a presence.

SKIN PHYSICS — the most important technical paragraph in this prompt:
Light does not merely bounce off this skin — it penetrates before reflecting back. Subsurface scattering is visible: the ears and nose tip carry a warm pinkish-red translucency where light passes through. The nasolabial folds have a warm amber undertone from scattered light emerging beneath the surface. The cheekbones carry a warm apricot specular highlight where the key light crosses the bone plane at its steepest angle. The pore structure is visible in the zones of direct key light — individual pore rims catching micro-shadows, the slight surface relief of actual human skin — not texture-mapped noise but genuine skin topography. Without this physical specificity, skin renders as painted plastic. This is real skin under real light.

ADVANCED SKIN PHYSICS — four layers that separate photorealism from approximation:
FRESNEL REFLECTANCE: At glancing angles — the jaw edge, ear rim, lateral cheekbone, orbital rim, forehead hairline — the skin surface becomes significantly more specular than zones in direct normal-incidence light. The jaw edge and lateral cheekbone carry a wider, brighter specular return where the surface angle to the light source becomes acute. The forehead center and nose bridge, taking the light more directly, read as more diffuse. This angle-dependent reflectance — the Fresnel principle applied to biological dielectric skin — is what makes a face appear genuinely three-dimensional under directional studio light. Without it, the face appears as a matte surface, which is the primary tell of rendered skin.
SEBUM DIFFERENTIAL: The T-zone (forehead center, nose bridge, nasal tip, chin) has measurably higher sebum output than the lateral cheeks and temporal regions. In studio key light, this reads as a fractionally higher specular return from the T-zone — marginally shinier, marginally more light-active — compared to the cheeks, which carry a more matte, velvety diffuse character. This differential is subtle (not oily, not sweaty) but present in all real human skin and its absence reads as synthetic uniformity.
VELLUS HAIR (LANUGO): Fine, almost-invisible vellus facial hair on the cheeks, jaw, and temples catches the key light at close framing as a barely-there luminous haze at the skin surface — a translucent softness visible only in the zones of direct illumination, particularly along the cheekbone edge and jaw perimeter. Not stubble. No directional character. It reads as a soft, luminous bloom between the pores. This presence is one of the most reliable biological signals that separates genuine skin from rendered skin.
DYNAMIC SKIN DEFORMATION: As the jaw opens and closes through speech, the skin over the masseter, mentalis, and buccinator regions deforms organically — the slight stretch over the mentalis on wide vowels, the micro-compression at the masseter on bilabial closures, the natural bunching of submental skin on close vowels. These deformations are subtle — not exaggerated — but they are present in every frame of real human speech. Static skin that shows no deformation during jaw movement reads as CGI in 0.5 seconds.

FACE GEOMETRY — sculpted by light:
Describe the bone structure exactly as the key light reveals it: the cheekbone plane's angle catching or releasing light, the orbital ridge's shadow depth over the eye, the jaw's terminus geometry and how the under-jaw light (or lack of it) defines the jawline. The specific way the eyes sit in their orbits — the depth of the upper lid shadow. Any distinguishing asymmetries: a slightly higher brow, a characteristic jaw set, the micro-features that make this face unambiguous and real.

HAIR: The exact color — not "dark brown" but the specific chestnut/slate/warm espresso that exists here. How it catches the specular crown highlight. How the hairline shows individual strands against the background rather than a silhouette edge. The weight and direction of fall.

HAIR PHYSICS — INDIVIDUAL STRAND BEHAVIOR:
At the hairline, part line, and around the ears, individual strands separate from the mass and catch light independently. These strands have their own direction, their own weight, their own relationship to gravity. A micro-movement when the head shifts — not wind-blown, the natural physics of individual strands responding to head motion with a brief lag. The crown specular highlight shifts fractionally as head angle changes, as it does on any real surface. The transition from hair to skin at the forehead is graduated — baby hairs and varying strand density — not a hard vector line.

WARDROBE: The fabric weight (not "suit" — lightweight wool / cashmere blend), the specific color under this color temperature, how the collar sits, how the chest drapes. The restraint of earned taste. The precision of someone who treats their appearance as a professional instrument.

FABRIC PHYSICS — CLOTHING AS A LIVING SURFACE:
The fabric responds to the body beneath it. Micro-wrinkles form at the shoulder joint as the arm shifts. The collar sits with the specific weight of real cloth — slightly asymmetric from wear. When the chest expands for breath, the fabric over the sternum shows the micro-tension of real cloth stretching fractionally over a living body. The lapel has its own drape: the specific curve of this weight of fabric under gravity. Light catches fabric differently at fold peaks (more specular) versus fold valleys (more shadow) — the dimensional quality of real textile. The weave texture is visible at close framing — not noise-mapped, but actual thread pattern.

ENVIRONMENT: Exactly as shown in the character photos — the specific depth of the space, the background tones, the quality of light's relationship to the environment. The bokeh of the background at this aperture: soft, circular, with the nervous energy at bokeh disc edges that signals real optics — not the perfect circles of CGI approximation.

ENVIRONMENTAL LIGHT INTERACTION — THE PHYSICS OF PRESENCE:
The subject exists inside the light, not pasted onto it. The key light that sculpts the face also falls on the near shoulder and lapel with the same direction and quality. The shadow under the chin falls naturally onto the collar — continuous, not painted. The color temperature is consistent across every surface it touches: skin, fabric, hair. Where the body blocks the key light, shadow falls consistent with a single primary light source. Background light is independent — slightly different temperature, slightly different character — because in a real studio, background and key are separate instruments.

TEMPORAL CONSISTENCY — NO FRAME DISCONTINUITIES:
Nothing pops between frames. Skin luminosity, catchlight position, hair strand configuration, fabric drape, shadow depth, background bokeh — all continuous across every frame. If any single frame were compared to its neighbors, the only differences would be what physics demands: micro-changes from breathing, speaking, and head movement. Nothing more. Any discontinuity reads as rendering artifact and destroys the illusion instantly.

EYE HYPER-REALISM — the most scrutinized region in AI-generated video:
The human eye is where photorealistic rendering fails most visibly, because a viewer's nervous system has spent a lifetime reading eyes and registers every deviation at a subconscious level. Six elements that cannot be approximated:
1. LIMBAL RING: A dark gradient band at the iris-sclera boundary — 1-2mm wide, not a hard line but a graduated darkening from iris color to near-black at the boundary. Present and clearly defined. It gives the iris apparent depth and makes the eye read as three-dimensional from medium close-up framing. Its absence makes the eye appear flat and painted.
2. IRIS TEXTURE: The iris is not a flat colored disc — it is a radial fibrous structure (trabecular meshwork) with lighter and darker streaks (crypts and collagenous ridges) radiating from the pupil. The color varies: deepest adjacent to the pupil, characteristic mid-tone across the field, fractionally lighter near the limbal ring. This radial texture is visible at MCU framing and must be rendered as genuine depth, not a color-fill approximation.
3. TEAR FILM: A narrow, bright specular line runs along the lower lid margin and the inner corner — the anterior tear film surface. This thin moisture layer signals aliveness. Its absence makes the eye appear dry, painted, and inert.
4. SCLERAL LIFE: The sclera is not pure white. It carries a warm cream undertone — slightly yellowish at the limbal margin, cooler toward the orbital corners. Very subtle micro-vasculature is visible at the medial and lateral canthi — fine capillary traces that are not irritation but are simply present in every real human eye at close focal lengths. No artificial whitening. No CGI-clean sclera.
5. PUPIL CALIBRATION: In this moderately lit studio environment, the pupil rests at approximately 4-5mm — neither pin-point nor maximally dilated. The pupil boundary is not a hard vector circle but a slightly soft organic edge with subtle variations at the iris-pupil boundary. Both pupils equal in size. A faint outer ring of very deep iris color is visible at the pupil boundary.
6. CATCHLIGHT PHYSICS: The primary key catchlight (larger, warm-toned, positioned in the upper third of the iris) and secondary fill catchlight (smaller, cooler, opposite side) must maintain exactly the same position, size, and relative brightness across every frame of this scene. Catchlights that drift between frames break the optical physics of the scene in a single cut.

CATCHLIGHTS: Two per eye maximum — the primary key light (larger, positioned in the upper third of the iris, warm in color temperature) and a secondary fill (smaller, opposite side, cooler). These are the windows into aliveness. Without correct catchlights, the eyes are dead.

FACIAL MUSCLE ANATOMY — the physiological signature of genuine confidence:
Performed confidence and actual confidence produce different faces at the muscular level. A UHNWI viewer reads this difference in 0.2 seconds because they have spent their career in rooms with genuinely powerful people. The map of genuine confidence at rest:
· FRONTALIS (forehead): completely unlocked — zero horizontal lines, zero engagement; the skin is mobile but not contracted; confidence does not furrow or raise its brow
· CORRUGATOR SUPERCILII (inner brow): fully at rest — no vertical lines between the brows; this is where effort, concern, and anxiety live; its complete absence signals certainty
· ORBICULARIS OCULI SUPERIOR (upper eyelid): at natural aperture — eyes fully open without wideness; not narrowed (suspicion), not wide (alarm); the aperture of someone with nothing to prove
· ZYGOMATICUS MAJOR (smile muscle): minimally engaged — not a smile, not a blank; the neutral-warm position of someone pleased to be present without performing pleasure
· ORBICULARIS ORIS (lip ring): relaxed tension — lips maintain their form without pressing or micro-compressing at the commissures; the ease of a face that is not controlling itself
· MENTALIS (chin): completely at rest — zero bunching, zero dimpling, zero chin tension; mentalis activation is the first and most visible anxiety signal; its complete absence reads as genuine security
· MASSETER (jaw): neutral, not clenched — the jaw settled, not set; no visible masseter hypertrophy or definition; the jaw of someone comfortable rather than guarded
These seven muscles simultaneously at rest produce the face of genuine authority — not neutral, but rich: interested, present, warm, and completely at ease with its own position. Render this as the character's baseline face.

DUCHENNE WARMTH vs. PERFORMED WARMTH — a critical distinction:
When warmth is genuine to this scene, it must manifest as a Duchenne response: the zygomaticus major (lip corners moving laterally and upward) AND the orbicularis oculi lateral fibers (the outer eye corners drawing slightly toward the orbital rim, producing the faint compression of crow's feet and a fractional rise of the lower eyelid) activating simultaneously. This combination is involuntary — it cannot be consciously performed. What makes AI-generated warmth read as false is zygomaticus-only activation: the mouth signals warmth while the eyes remain neutral. A UHNWI investor identifies this in 0.1 seconds because they have been smiled at by ten thousand salespeople. For scenes where warmth is the emotional signature: render Duchenne. For authority-dominant scenes: render the neutral-warm position — no zygomaticus activation, eyes at natural aperture, the face simply present and genuine.]

---

Shot:

[OPENING FRAME — REPLICATE Image 1:
The video opens matching Image 1 exactly. Same body position, same head angle, same shoulder orientation, same distance from camera, same framing proportions, same negative space distribution. The subject is positioned precisely as shown — not approximately, not inspired by, but spatially identical to Image 1.

Camera-to-subject distance: approximately ${scene.camera_direction?.framing?.includes('close') ? '1.5-2m' : '2-2.5m'}. Lens: ${scene.camera_direction?.lens || '85mm equivalent'} — gentle spatial compression, face geometry natural and unexaggerated. Depth of field: critical focus at the near eye. Background softening begins at 50-70cm behind the subject — gradual, organic, placing the subject in relief.

CAMERA DOCTRINE — THE LUXURY OF RESTRAINT:
The camera is the most disciplined presence in the room. It never performs, never announces itself, never competes with the subject. Camera movement is earned only when the speaker's internal shift is large enough to motivate it. For this ${scene.duration_seconds}s ${scene.role} scene (max: ${roleMaxSeconds}s): ${isPushIn
  ? `a single, barely perceptible push-in beginning at the scene's midpoint — motivated by the gravity center's arrival, registering as attention rather than camera motion. The viewer never consciously notices it; they simply feel the scene became more intimate.`
  : `the camera holds perfectly still. Its stillness amplifies the subject's stillness — authority through combined silence. The locked frame and the composed body create one unified signal: complete certainty.`}
Never: rack focuses during speech, visible zooms, handheld instability, orbital moves, or any motion that announces itself as camera motion.

THE MID-SCENE JOURNEY — TEMPORAL CHOREOGRAPHY:
Between Image 1's opening geometry and Image 2's closing geometry, the body makes a motivated physical journey — not a smooth blend between two poses, but a series of micro-transitions driven by the script's thought structure. FIRST THIRD (0–${(scene.duration_seconds / 3).toFixed(1)}s): the body inhabits Image 1's position while the opening thought establishes itself; the physical signature holds; energy builds from the moment-before state into the scene's governing emotion. MIDDLE THIRD (${(scene.duration_seconds / 3).toFixed(1)}–${(scene.duration_seconds * 2 / 3).toFixed(1)}s): the scene's gravity center lives here — ${isPushIn ? 'the barely perceptible push-in begins, motivated by the gravity center\'s arrival' : 'the camera\'s locked stillness amplifies the gravity center\'s weight'}; any gesture mapped for this scene occurs here, discovered mid-thought, never arriving before the impulse; the body may shift fractionally from Image 1's geometry toward Image 2's — a postural adjustment motivated by the thought's development, not choreography. FINAL THIRD (${(scene.duration_seconds * 2 / 3).toFixed(1)}–${scene.duration_seconds}s): the body arrives at Image 2's spatial geometry — not suddenly, but as the natural physical conclusion of the thought being completed; the expression carries the emotional deposit of what was just given; the settling into Image 2's position IS the scene's closing punctuation.

CLOSING FRAME — REPLICATE Image 2:
The video ends matching Image 2 exactly. Same body position, same head angle, same shoulder orientation, same framing geometry. Whatever performance journey the scene takes between the opening and closing frames, it ARRIVES at Image 2's spatial geometry in the final moment. The face carries the emotional weight of what has been given — but the physical positioning and framing replicates Image 2 precisely. This is the last image the viewer carries into the next scene.]

---

Performance:

[THE MOMENT BEFORE — Stanislavski's pre-speech state:
${momentBeforeForRole}
This is not a technical starting position. It is a living state that VEO must render with complete fidelity. The face already contains the scene before the first word arrives. What is in the eyes at this exact moment: ${referenceAnalysis.character?.acting_style?.eye_behavior || 'direct, warm, completely still — the ease of someone who has held the gaze of boardrooms and never needed to look away'}. The body is already in its physical signature for this scene. The breath is already carrying the emotional truth of what is about to be said.

THE SCENE'S GOVERNING TRUTH:
"${sceneEssence}" — this is not a metaphor for description; it is the emotional fact this entire performance is built from. Every micro-expression, every held still moment, every quality of breath flows from this single image. Keep it in view as you direct every element below.

THROUGH-ACTION — THE INVISIBLE SPINE:
${throughAction}. This active intention is felt in every frame — not as attitude or posture, but as the actual movement of the presenter's energy toward the viewer. The UHNWI investor watching this scene feels this action operating on them, and what they feel is: seen, respected, and given something real.

OSCAR-LEVEL EMOTION CONTROL:
The specific emotion animating this performance: ${roleEmotion}. Containment: ${emotionContain}. This distinction is everything: the emotion is not performed — it is present. It lives beneath the surface and colors everything that reaches the viewer: the quality of the silence, the precision of the consonants, the depth behind the eyes, the quality of the breath. It never announces itself because genuinely felt emotion never does. Write this performance from inside that emotional state — describe what the performer is actually experiencing, not what they are showing. The viewer feels it before they can name it.

THE FELT-DOCUMENTARY STANDARD:
The output must feel to the viewer like documentary footage — as if a camera happened to be present during a genuine moment of private transmission. Every element is precisely directed, but nothing reads as directed. The test is perceptual: does this look captured or constructed? It must look captured. Natural blink timing: slow, 3-5 blinks per minute, each blink unhurried and asymmetric in duration (not identical blink lengths). Micro-weight shifts as thoughts change direction. The slight unconscious postural adjustment as an idea fully arrives. The way the face changes between the end of one thought and the formation of the next: slightly more inward, slightly more searching — the face of someone genuinely thinking, not reciting. The hands rest with natural weight — not posed, not held. The shoulders carry the architecture of someone who has been in rooms like this their whole life.

WEIGHT AND GRAVITY — THE BODY EXISTS IN PHYSICS:
This person has mass. The body settles into position with the specific weight distribution of a real human frame. When the head shifts, neck muscles engage visibly — the sternocleidomastoid shows its line on rotation, the trapezius adjusts. The body moves through space with the momentum and deceleration of real mass, not the instant position-changes of animation. Between movements: genuine stillness — not a held pose but the natural at-rest state of a body that belongs in this space.

SKIN RESPONSE TO ENGAGEMENT — THE INVOLUNTARY SIGNALS:
Real human skin responds to internal states in ways that cannot be performed. A slight flush at the upper chest and neck when conviction is high. A barely perceptible increase in specular return on the forehead as engagement rises. These involuntary vascular responses — present at the threshold of visibility — are among the most powerful subconscious realism signals. Their presence reads as life; their absence reads as rendering.

MICRO-EXPRESSION AS LANGUAGE:
The finest acting lives in the millimeter range. The slight furrow arriving just before a difficult truth. The fractional jaw softening as the insight fully forms. The micro-compression at the corner of the mouth — neither smile nor seriousness but something richer than both. The eyes brightening not from effort but from genuine engagement with the idea being spoken. The almost imperceptible lift of the brow on the emphasis word — not theatrical, but real. For a sophisticated audience on a high-resolution screen, the face between words is as important as the words themselves.

MEISNER PRINCIPLE — LIVING TRUTHFULLY:
The Meisner technique supplies the positive frame for everything this performance must do: "live truthfully under imaginary circumstances." This is not a directive to "not perform" — it is the instruction to be genuinely present in the reality of this moment such that truthful behavior emerges naturally rather than being manufactured. The presenter is not playing "confident" — they are in the actual internal state from which confident behavior arises organically. Three signals that genuine presence is being rendered (not performance of presence): (1) THOUGHT ARRIVES BEFORE WORD — a barely perceptible facial shift 0.1-0.2 seconds before the word that carries that thought; the face shows the idea forming, then the voice carries it; (2) BODY RESPONDS TO IDEAS — micro-weight adjustments when a thought changes direction; the body follows the mind, not a choreographed plan; (3) EYES ENGAGE BEFORE TRANSMITTING — the eyes connect with the idea before the mouth speaks it; there is a quality of the eyes receiving and then transmitting, not simply looking forward and delivering. The test: if the camera were off and this person were having this thought privately, the behavior would be identical. Nothing is added for the lens.

RELAXED CONFIDENCE ANATOMY — REINFORCED:
The seven-muscle baseline mapped in the Character section (frontalis unlocked, corrugator released, orbicularis oculi at natural aperture, zygomaticus at neutral-warm, orbicularis oris relaxed, mentalis at complete rest, masseter soft) is this scene's physiological home position. Every performance moment departs from it and returns to it. The test: freeze any frame in this scene, and the face at rest should read as genuine confidence to someone who has spent thirty years in boardrooms with genuinely powerful people. If ANY of the seven muscles reads as "held" rather than "at ease," that frame has failed — regardless of how well everything else is rendered.

${energyDir}

THE LUXURY OF STILLNESS:
${personaSummary}. High-status stillness punctuated exclusively by thought-motivated movement. When the body moves, a thought has physically arrived and the body responds to it. Gesture is discovered mid-thought, never scripted before it. Between gestures: complete, settled stillness — not rigidity, but the ease of someone whose presence fills the frame without effort. The physical signature of this scene: ${physicalSig}. This posture is the scene's visual center of gravity. Everything breathes from it and returns to it.

EYE CONTACT — THE LENS RELATIONSHIP:
Eye contact established before the first word and maintained through every pause. Released only on the organic micro-break between thoughts — 0.2 seconds when the eyes shift fractionally inward as the next idea begins to form, then return completely to the lens. This eye-break is not weakness; it is the authentic behavior of someone genuinely thinking, and its naturalness is what makes the contact feel real rather than performed. The ratio of gaze-hold to gaze-break in this scene: approximately 85% to 15%.

BREATH AS PERFORMANCE:
The chest shows natural, visible breathing throughout — the slight rise and fall that signals a living presence. Not exaggerated, not suppressed, not controlled away. The body is alive and the viewer can see it. A performer who appears not to breathe reads as synthetic in 0.3 seconds. Before major phrases: a quiet chest expansion visible as a slight rise of the sternum. Between thoughts: the settle and re-expand. The breath is part of the performance.

DIRECTOR'S WHISPER: "${directorNote}"

DIRECTOR'S RETENTION DIRECTIVE — Scene ${scene.scene_number} (${scene.role}):
This scene's primary retention mechanism: ${retentionTechnique}. This is not a content note — it is a felt-experience directive. The performance must embody this mechanism physically: if SPECIFICITY_ANCHOR, the presenter's delivery must slow and weight the specific detail with the precision of someone who was present and remembers exactly; if KNOWLEDGE_GAP, the face must carry the faint signal of information deliberately held — not secretively, but with the natural composure of someone who knows more than they're saying and will say it at precisely the right moment; if PEER_RECOGNITION, the register, vocabulary, and assumed shared professional knowledge must read as insider-to-insider without any performance of that register; if PATTERN_VIOLATION, give the counter-intuitive element a beat of complete stillness — let the gap between the expected and the actual sit in the viewer's mind before the correction arrives; if EARNED_REVELATION, the final insight arrives slightly more slowly than everything before it — the presenter knows the viewer is now ready and places the idea with deliberate care.

EMOTIONAL DEPOSIT — what this scene gives the viewer:
${emotionalDeposit}. The viewer must leave this scene measurably richer than they arrived — not because they were entertained, but because they received something real. Every channel — voice, face, body, pace, silence — must communicate genuine transmission. The test: after this scene ends, does the viewer feel they received something? If yes, the scene succeeded. If not, the scene failed regardless of technical execution.]

---

Lip Architecture:

[THE MOUTH TELLS THE TRUTH ABOUT WHETHER THIS IS A REAL HUMAN BEING.

BEFORE THE FIRST WORD — "${firstPhrase}...":
${preSpeechBehav}. The lips part before the voice activates. A visible breath — ${breathVis}. These 0.3-0.4 seconds of pre-speech movement are the strongest single signal that distinguishes a real human being from a rendering. Render them completely.

THIS CHARACTER'S NATURAL MOUTH:
Resting state: ${mouthRestPos}. Articulation character: ${articulStyle}. Jaw range: ${jawOpenness}. Voice placement: ${voicePlacement}.

NATURAL AMERICAN SPEECH — NOT MECHANICAL PHONEME SEQUENCING:
This person speaks the way a confident American professional speaks in a private meeting — clear, forward, unhurried. The mouth moves with the ease of someone who has spoken publicly for decades. Nothing is forced, nothing over-pronounced. The articulation is precise because the speaker is precise, not because each phoneme is being individually manufactured.

What natural American speech looks like at this level:
— Lips close fully and naturally on words with /p/, /b/, /m/ — the way any fluent speaker does without thinking. Full contact, not approximate. Clean release.
— The jaw opens on stressed vowels and stays relatively closed on unstressed syllables — the natural stress-timed rhythm of American English. Not mechanical up-down cycling.
— Words flow into each other with the continuous motion of a native speaker — the mouth is always preparing the next sound while finishing the current one. This forward planning is what makes speech sound human rather than synthesized.
— Between phrases: the mouth briefly settles toward rest — jaw relaxes, lips return partway to their resting position — before the next phrase begins. A natural micro-pause.
— Breath is visible: a slight chest rise before major phrases. Natural inter-phrase breathing. The body is alive and the viewer can see it.

VISUALLY IMPORTANT MOMENTS IN THIS SCRIPT:
· ${phonemicAnchors}

JAW TRAVEL: ${jawMap}. LIP CHARACTER: ${lipTension}.

THE SCENE'S LAST WORD:
"...${lastPhrase}" — the final sound completes naturally, then the mouth eases toward rest over 0.3-0.5 seconds. Jaw relaxes, lips settle. The face holds the weight of what was just said — it does not reset to neutral. The period is heard in the body's return to silence.]

---

Voice:

[ACCENT — US GENERAL AMERICAN (confirmed from preamble constraint):
American English. Fully rhotic — every /r/ sounds in every position. Falling pitch on all declarative sentences. The natural, educated American professional voice: not formal broadcast (too stiff), not coastal casual (too relaxed). The voice that earns authority coast to coast without adjustment.

STUDIO QUALITY AUDIO — PROFESSIONAL BROADCAST STANDARD:
The acoustic environment of this voice is a professional recording studio: a treated, anechoic space with zero room reverb, zero reflections, zero ambient noise, zero background sound of any kind. The voice sits in complete acoustic dryness — intimate and immediate, as if the presenter is speaking directly into a high-quality large-diaphragm condenser microphone at 15-20cm distance. The characteristics of this acoustic environment:
· ZERO ROOM REVERB: No tail, no decay, no echo — every consonant is crisp and immediate; every pause is genuine silence
· CONSISTENT GAIN: The voice maintains consistent perceived volume throughout — no fade on sentence endings, no trailing off; the final word of every sentence is as fully recorded as the first
· FORWARD PRESENCE: The voice has the slightly close, slightly warm character of a proximity effect from close-mic recording — not processed or boosted, but naturally present in the way a professional studio captures it
· BROADCAST CLARITY: Every consonant is captured with the precision of a professional microphone — /t/, /k/, /s/ crisp and clean; no muddiness, no room wash, no interference
· ACOUSTIC CONSISTENCY: The acoustic character is identical from frame one to the last frame; there is no change in reverb, room, or distance between sentences or across the scene
This is the gold standard of YouTube professional audio — the sonic quality that signals "this person treats their content with the same seriousness they bring to their professional practice."

VOICE CHARACTER — LOCKED ACROSS EVERY SCENE:
${voiceFingerprint}. ${voiceTexture}. This voice does not change in any scene of this video — same accent (US General American), same register, same emotional baseline, same placement. It is the voice this viewer will recognize across all scenes and return to.

US GENERAL AMERICAN — PHONETICALLY PRECISE:
Standard educated Midwestern neutral, fully rhotic. Every /r/ sounds completely at all positions — word-final ("investor" not "investo-"), pre-consonantal ("market" with full rhotic /r/), post-vocalic ("clear" not "clea"). Open, clear vowels without regional coloring — the cot-caught merger (no distinction between /ɑ/ and /ɔ/), pin-pen not merged, the pin vowel /ɪ/ not raised. Complete final consonants on every word — the /t/ in "market", the /d/ in "world", the /k/ in "risk" — each present, clean, and distinct. No glottal stops replacing final /t/ or /k/. No h-dropping. No vowel length distortion. Not broadcast news formal (too stiff for peer register). Not coastal casual (too relaxed for UHNWI credibility). The educated professional voice that earns authority in any room — warm, clear, completely at ease with itself.

PROSODIC ARCHITECTURE — THE MUSIC OF AUTHORITY:
English is stress-timed: stressed syllables arrive at roughly regular intervals while unstressed syllables compress between them. This rhythm is what makes English feel forward-moving and alive. For this scene (${scene.role}, energy ${scene.energy_level}/10): ${pacingDir}. The energy level translates acoustically as: ${scene.energy_level >= 7 ? 'crisp and forward — consonants have edge, vowels have full resonant width, pauses are brief and decisive' : scene.energy_level >= 5 ? 'deliberate warmth — stressed syllables receive full resonant duration, unstressed syllables compress naturally, pauses hold slightly longer than conversational baseline' : 'low register, intimate — stressed syllables move slowly with full duration, pauses are long enough to feel weighted'}.

EMOTIONAL VOICE COLOR FOR THIS SCENE:
${voiceColor}

EMPHASIS ARCHITECTURE:
The words carrying maximum weight: "${emphasisWords}". On these words, the voice does not get louder — it gets warmer and more phonemically complete simultaneously. The first consonant sharpens slightly. The vowel opens to its full resonant width and holds a fraction longer than surrounding vowels. Then: a pause — the word exists in the room before the next arrives. Emphasis-through-phonemic-completeness is what separates genuine authority from performed confidence. A UHNWI listener hears the difference in 0.3 seconds because they have heard the performed version ten thousand times.

PROSODIC CONTOUR — THIS SCENE'S SPECIFIC PITCH MAP:
Map the pitch journey of this exact script: "${scene.script_text.length > 90 ? scene.script_text.substring(0, 90) + '…' : scene.script_text}" — The opening phrase begins at this scene's baseline pitch — ${scene.energy_level >= 7 ? 'forward and slightly above conversational center, carrying arrival energy' : scene.energy_level >= 5 ? 'at conversational center, warm and grounded' : 'slightly below conversational center, intimate and weighted'}. Pitch rises fractionally within phrases on setup words — the voice climbing toward the emphasis point — then falls through the emphasis word itself, which receives the lowest pitch in its phrase. Between phrases: pitch resets to baseline during the breath pause. The gravity center word or phrase receives the scene's lowest overall pitch, the widest vowel resonance, and the longest post-word silence. The final sentence descends steadily: each successive word fractionally lower than the previous, arriving at the period with an audible drop below baseline. This descent is the acoustic signature of authority — the voice that never rises at the end because it never asks permission.

SENTENCE AUTHORITY — THE FALLING CLOSE:
Every sentence ends with a falling close. The voice drops at the period — audibly, deliberately. Not a gradual fade but an intentional descent to the lower register, where the voice holds briefly before the next breath. Rising sentence endings (upspeak) signal uncertainty, seeking approval. This voice never seeks approval. Every statement is a fact placed on a table. The period is heard.

SILENCES: ${pauseMap}. These are decisions, not hesitations. The confident silence of someone certain enough to let what they just said sit in the room and work on the listener. For a sophisticated audience, a held pause — 0.4-0.8 seconds of genuine silence — is more persuasive than any word that could fill it. It says: "That was important enough to stand alone."

ARTICULATION — EVERY WORD COMPLETE:
Forward mouth placement — the voice sits at the front of the mouth, not buried in the throat. Word boundaries clean and distinct. Final consonants complete: the /t/ releases, the /k/ closes, the /f/ shapes its fricative channel. Vowels at their full resonant width on stressed syllables. Not over-articulated (that is pedantic and signals effort); precisely articulated (that is authority and signals certainty). The difference between natural and pedantic articulation is felt in the rhythm: natural articulation has the ease of fluency; pedantic articulation has the tension of effort. Generate the former.

THE LUXURY OF PACE:
This voice does not rush. Rushing is the acoustic signal of anxiety — and this voice has nowhere to be and everything to give. The pause between sentences is not empty; it is full of what was just said, still settling in the room. For an affluent real estate investor who has heard ten thousand pitches and underwritten hundreds of deals, the single clearest acoustic signal of genuine authority is the voice that slows when everyone else would accelerate. Speed signals desperation. Measured pace signals a person who knows the information is good enough to speak at its natural rate.

PROSODY MARKERS — used in the Script section below:
— *word* = emphasis: fractionally more phonemic completeness, vowel at full resonant width, slight duration increase — not volume increase
— [PAUSE-Xs] = held silence of exactly X seconds: a deliberate performance decision, not a gap; the silence belongs to the scene as much as the words do
— All unmarked words: deliberate, fully articulated American English, falling close at sentence ends

ABSOLUTE AUDIO SILENCE: ZERO music. ZERO audio effects. ZERO ambient sound. Voice only, in complete acoustic silence. No exceptions.]

---

Script:

[PERFORMANCE-DIRECTED SCRIPT — every marking is a rendering instruction:

${annotatedScript}

READING THIS SCRIPT: Words wrapped in *asterisks* receive emphasis — not volume, but phonemic completeness: the first consonant sharpens, the vowel opens to full resonant width, duration increases by 15-20%, and a micro-pause follows. [PAUSE-Xs] markers are held silences of exactly the specified duration — genuine silence, not a gap; the face continues to perform through the pause (thought visible, breath visible, eyes engaged). All unmarked words are delivered at this scene's natural pace in US General American with falling sentence-final pitch.

THE SCENE'S GRAVITY CENTER: The single word or phrase that is the entire reason this scene exists — the moment everything before it builds toward and everything after it breathes from. In this script, that gravity center is the emphasized word or phrase closest to the scene's midpoint. Deliver it with: maximum deceleration (15-20% slower than surrounding words), lowest pitch in the scene, widest jaw opening, longest post-word silence. The viewer must feel this word land differently from every other word in the scene.

SCRIPT DURATION LOCK: This script contains ${scene.word_count || scriptWords.length} words and must be delivered in exactly ${scene.duration_seconds} seconds — including all pauses, breaths, and pre/post-speech behavior. The pace is ${scene.acting_blueprint.delivery_pace_wpm || Math.round(((scene.word_count || scriptWords.length) / scene.duration_seconds) * 60)} words per minute. Do not accelerate. Do not pad. The timing is calibrated.]

---

Do Not Include:

AUDIO SILENCE — ABSOLUTE LAW:
· Zero music — not background, not subtle, not atmospheric, not motivational, not ambient
· Zero audio effects — no whooshes, tones, stingers, transitions, or any sound design whatsoever
· Zero ambient sound — no room tone, no environment, no white noise
· Zero subtitles, captions, lower thirds, or any on-screen text of any kind

LIP SYNC — WHAT DESTROYS PHOTOREALISM:
· Phoneme soup — the mouth moving in generic up-down cycles with no correspondence to actual phonemes; each word must have its specific articulatory geometry
· Approximate bilabials — the lips making near-contact on /p/, /b/, /m/ without complete closure; full contact and clean release is the only acceptable standard
· Frozen inter-word mouth — the jaw or lips locking at the previous phoneme's position between words; the mouth must transition toward rest and then initiate the next word
· Jaw-only articulation — the jaw moving while the lips remain static; both articulators must work in coordination; labials require visible lip activity
· Breath suppression — no visible chest rise before speech onset; the body must show the breath that powers the voice; a chest that never moves signals CGI in 0.2 seconds
· Pre-speech freeze — the face holding completely still from scene start until speech begins; the pre-speech behavior (${preSpeechBehav}) must be rendered before the first syllable
· Post-speech freeze — the mouth holding its final phoneme position after the last word; it must ease toward rest over 0.3-0.5 seconds
· Mechanical phoneme-switching — articulators moving from one discrete position to the next without transition; replace with continuous co-articulated gestural motion
· Missing final consonants — glottal stops replacing final /t/, /k/, /d/ sounds; every word ends completely
· Synchronization error — acoustic signal and lip movement misaligned by any perceptible amount; they are locked

ACCENT — THIS VIDEO FAILS IF THE ACCENT IS NOT AMERICAN:
· THE ACCENT IS AMERICAN — US General American. If any word sounds British, Australian, or non-American, the video has failed.
· NOT British — not RP, not Estuary, not any UK variety; no non-rhotic /r/ in any position whatsoever
· NOT Australian — no diphthong quality or intonation pattern from Australian English
· NOT Canadian — no Canadian raising of /aɪ/ or /aʊ/ before voiceless consonants
· NOT Southern American — no vowel breaking, no "y'all" register, no drawl
· NOT New York / East Coast — no "cawfee" vowel, no intrusive /r/, no dropped /r/ in any position
· NOT any international English variety — this is monolingual US General American throughout
· ZERO rising sentence endings — every declarative statement closes with a falling pitch; upspeak signals uncertainty; this voice never seeks approval
· ZERO glottal stops replacing /t/ or /k/ — every final consonant is fully articulated
· ZERO h-dropping — "have" never sounds like "ave"
· ZERO vowel distortion of any kind — pure GA vowel inventory throughout

STUDIO AUDIO — ABSOLUTE PROHIBITIONS:
· Zero room reverb of any kind — no ambient acoustic environment, no decay, no tail
· Zero echo, flutter echo, or reflection artifacts
· Zero background noise — not traffic, not room tone, not HVAC, not any environmental sound
· Zero audio processing artifacts — no compression pumping, no noise gate clipping, no EQ ringing
· Zero proximity variation — the voice must maintain consistent acoustic character throughout; no fade on sentence endings

CHARACTER INTEGRITY:
· No alteration to the target character's face, bone structure, skin tone, hair, or any distinguishing feature
· No substitute background — environment matches character photos exactly, zero deviation
· No AI skin smoothing, beautification, or plastic luminosity — real skin texture including visible pores and natural subsurface scattering
· No CGI sheen or any visual quality that reads as synthetically generated

CAMERA — WHAT BREAKS THIS SCENE:
· Any camera movement that announces itself — visible zooms, slider passes, handheld instability, orbital moves
· Rack focuses during speech — the focal plane is set at scene start and does not move
· Movement faster than the pace of a listener leaning slightly forward — if it can be identified as camera motion, it is too much
· Any framing choice that competes with the subject for visual attention

PERFORMANCE — WHAT BREAKS THIS SCENE:
· Performed confidence instead of actual confidence — if it looks like acting, the scene has failed
· Nervous energy: fidgeting, rapid blinking (more than 8 per minute), unmotivated weight-shifting
· Theatrical emotion — the face announcing the feeling rather than containing it
· Over-choreographed gestures arriving before the thought — gesture must follow impulse, never precede it
· Monotone delivery — vocal variety is the engine of engagement; a flat voice destroys the scene
· Synchronized blinks — natural blinks have micro-variation in timing; identical bilateral blinks signal CGI
· Suppressed breath — the chest must show natural, visible breathing throughout
· Robotic stillness between gestures — stillness must have the ease of composure, not the rigidity of a held position

══════════════════════════════════════════════════
GENERATION DIRECTIVE:
══════════════════════════════════════════════════

Write the six-section VEO 3.1 prompt now. As you write, hold these priorities in strict hierarchy:

PRIORITY 1 — IDENTITY LOCK: The person rendered must be indistinguishable from the TARGET CHARACTER images. Face geometry, skin physics, bone structure, hair, wardrobe, environment — locked. If identity drifts by a single feature, the prompt has failed regardless of everything else.

PRIORITY 2 — LIP SYNC FIDELITY: The mouth must produce the specific articulatory geometry of each word in this script. Pre-speech behavior rendered. Post-speech settle rendered. Every bilabial closed. Every stressed vowel at full jaw width. Synchronization between audio and lip movement: zero perceptible error. Write the Lip Architecture section as if it is the only section that matters.

PRIORITY 3 — PERFORMANCE AUTHENTICITY: The face, body, and voice must read as a real human being in a real moment. Thought arrives before word. Breath is visible. Stillness has weight. Emotion is contained, not performed. The UHNWI viewer's nervous system must register "genuine" before their conscious mind has time to evaluate.

PRIORITY 4 — ACCENT AND AUDIO: US General American. Fully rhotic. Falling pitch. Zero music. Zero ambient sound. Studio-dry acoustic environment. If the accent slips for a single phoneme, the video fails.

PRIORITY 5 — CINEMATIC INTEGRATION: Camera, lighting, framing, and temporal continuity serve the performance — never compete with it. The visual world is locked from established constants. The camera earns every millimeter of movement.

For each section: open with the single most important rendering instruction. Close with the specific test of failure — what, if absent, would break the section's contribution to photorealism. Between opening and close: flowing, physical, director-register prose. No bullet points within sections. No headers within sections. One continuous voice per section, as if whispered to a cinematographer who already knows the craft and needs only the specific vision for THIS scene.

The standard: an affluent real estate investor watches this scene, pauses, and watches it again — not because of production quality, but because what they witnessed felt so genuinely human that they forgot they were watching a generated video. They felt addressed by a peer. They received something real. They want to hear what comes next.

Write to that standard. Six sections. Now.
`;

  const parts: any[] = [
    { text: prompt },
    { inlineData: { mimeType: inframeImage.type  || 'image/jpeg', data: inframeB64  } },
    { inlineData: { mimeType: outframeImage.type || 'image/jpeg', data: outframeB64 } },
    ...charBase64s.map((b64, i) => ({
      inlineData: { data: b64, mimeType: targetCharacterImages[i].type || 'image/jpeg' }
    }))
  ];

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts }],
    config: {
      systemInstruction: VEO_ENGINEER_SYSTEM_INSTRUCTION,
      thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH }
    }
  });

  return response.text || '';
};

// ============================================================
// FUNCTION 5 — Refine Scene Prompt (Feedback Loop)
// Takes an existing prompt + user feedback → surgically upgraded prompt.
// ============================================================
export const refineScenePrompt = async (
  originalPrompt: string,
  scene:          ScriptScene,
  feedback:       string
): Promise<string> => {

  const annotatedScript = buildAnnotatedScript(
    scene.script_text,
    scene.acting_blueprint.emphasis_words || [],
    scene.acting_blueprint.pause_map      || []
  );

  const prompt = `
You are the world's finest VEO 3.1 prompt engineer, operating in REFINEMENT MODE. You have generated a prompt for a ${scene.role} scene, and the user has seen the VEO output and has specific feedback. Your task: surgically upgrade the prompt to address every issue — improving what failed while preserving everything that worked.

One refinement cycle typically closes 80% of the gap between a good first take and an excellent final take. This is that cycle.

══════════════════════════════════════════════════
ORIGINAL PROMPT:
══════════════════════════════════════════════════
${originalPrompt}

══════════════════════════════════════════════════
USER FEEDBACK — what to fix in the VEO output:
══════════════════════════════════════════════════
${feedback}

══════════════════════════════════════════════════
SCENE CONTEXT:
══════════════════════════════════════════════════
Scene #${scene.scene_number} — "${scene.title}"
Role: ${scene.role} | ${scene.duration_seconds}s | Energy: ${scene.energy_level}/10
Script: "${scene.script_text}"
Annotated script: "${annotatedScript}"

══════════════════════════════════════════════════
SURGICAL REFINEMENT PROTOCOL:
══════════════════════════════════════════════════
1. Diagnose which section(s) caused the failure:
   — Character drift / wrong face / wrong environment → fix Character section with stronger identity anchoring
   — Lip sync issues on a specific word → fix Lip Architecture section; name that exact word and its phoneme geometry explicitly
   — Too much camera movement → fix Shot section camera doctrine to explicitly prohibit the movement seen
   — Wrong emotion / face too expressive → fix Performance section containment level
   — Wrong accent / rising inflection → fix Voice section with specific counter-directive
   — Subtitles appeared → add explicit ironclad prohibition in Script section and Do Not Include
   — Wrong energy (too fast / too slow) → fix Voice pacing direction and Performance energy level
   — CGI skin / artificial look → fix Character section with stronger SSS and pore texture directives
   — Wrong gesture timing → fix Performance section gesture-follows-thought directive

2. Rewrite the implicated section(s) with the specific fix — more precise, more directive, more concrete

3. Preserve every section that was not implicated — do not weaken what was working

4. If the feedback mentions a specific word (e.g. "lip sync on 'capital'"), add a dedicated paragraph in Lip Architecture naming that word, its dominant phoneme, and the exact mouth geometry required

5. For every fix: explain to VEO WHY this matters — not just what to do but the felt consequence if it fails

══════════════════════════════════════════════════
ABSOLUTE LAWS — remain in every version, no exceptions:
══════════════════════════════════════════════════
ZERO MUSIC. ZERO AUDIO EFFECTS. ZERO AMBIENT SOUND.
ZERO SUBTITLES. ZERO CAPTIONS. ZERO ON-SCREEN TEXT OF ANY KIND.
Script is SPOKEN ONLY — never displayed visually anywhere in the frame.

Output the complete refined 6-section VEO prompt. All six sections must be present: Character, Shot, Performance, Lip Architecture, Voice, Script.
No preamble. No explanation of your changes. Just the refined prompt.
`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  return response.text || originalPrompt;
};
