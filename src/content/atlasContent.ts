import type { Register, ZoomLevel } from '../state/urlState';

export interface PresetPrompt {
  id: string;
  label: string;
  text: string;
}

export const PRESET_PROMPTS: PresetPrompt[] = [
  {
    id: 'max-ball',
    label: 'Max & the red ball',
    text: 'Once upon a time, a little dog named Max found a red ball. He',
  },
  {
    id: 'lily-cake',
    label: 'Lily bakes a cake',
    text: 'Lily wanted to bake a cake with her mom, so she',
  },
  {
    id: 'cat-mat',
    label: 'The cat on the mat',
    text: 'The cat sat on the mat and looked at the',
  },
  {
    id: 'confident-mistake',
    label: 'Confident story guess',
    text: 'The capital of France is a little girl named',
  },
];

export interface HonestyMeta {
  kind: 'literal' | 'abstraction';
  label: string;
  detail: string;
  literalLevel?: ZoomLevel;
}

export interface LevelContent {
  level: ZoomLevel;
  crumb: string;
  title: string;
  honesty: HonestyMeta;
  answersQuestion: string | null;
  createsQuestion: string | null;
  nextCta: string | null;
  copy: Record<Register, string>;
  simplificationNote?: string;
}

export const LEVEL_CONTENT: Record<ZoomLevel, LevelContent> = {
  0: {
    level: 0,
    crumb: 'The Machine',
    title: 'A next-word prediction machine',
    honesty: {
      kind: 'abstraction',
      label: 'Outer shell',
      detail:
        'The frosted casing hides 4 real transformer blocks and 862,464 trained numbers. Every floating answer bubble on the right is sized by its real probability.',
      literalLevel: 3,
    },
    answersQuestion: null,
    createsQuestion: 'What happens inside the glass box between the story and the guess?',
    nextCta: 'Open the lid',
    copy: {
      kid: 'Your story breaks into smooth little tiles and glides into the box. Inside, the machine asks one simple question: which word piece is most likely to come next? The biggest bubble on the right is its favorite guess, and the smaller bubbles are its runners-up.',
      curious:
        'A miniature decoder-only transformer takes up to 64 BPE tokens and scores all 512 vocabulary pieces for what comes next. Repeating that single next-token prediction over and over is how a language model writes.',
      deep: 'f_θ: V^T -> Δ^{511} with 862,464 float16 weights trained on TinyStories. Bubble radii map directly to post-softmax probabilities P(x_{T+1} | x_{1..T}) computed live in a browser Web Worker.',
    },
    simplificationNote:
      'The four glowing shapes pulsing inside the glass are real—click the box to lift the lid.',
  },
  1: {
    level: 1,
    crumb: 'Glass Factory',
    title: 'Four chambers sharpening one guess',
    honesty: {
      kind: 'abstraction',
      label: 'Layered flow',
      detail:
        'Each glass chamber wraps a real [T, 128] residual update. The thought-bubbles floating above each chamber are real Logit Lens predictions read directly out of that layer.',
      literalLevel: 3,
    },
    answersQuestion: 'What happens inside the glass box between the story and the guess?',
    createsQuestion: 'What happens inside one of those four glass chambers?',
    nextCta: 'Fly inside chamber',
    copy: {
      kid: 'First, each word tile turns into a glowing stack of 128 numbers. Then it rides along one unbroken glass highway through four chambers. Look at the thought-bubbles floating above each chamber—you can literally watch the machine sharpen its mind from Chamber 0 to Chamber 3.',
      curious:
        'Token and position embeddings enter a shared residual stream of shape [T, 128]. Four stacked transformer blocks refine that stream in place. Peeking at the stream after each block with the final output head (Logit Lens) shows the prediction crystallizing with depth.',
      deep: 'h_0 = wte[x] + wpe[0..T-1] ∈ R^{T×128}; h_{l+1} = Block_l(h_l). Each floating bubble displays argmax softmax(LN_f(h_{l+1, T-1}) wte^T) using the exact tied output matrix.',
    },
    simplificationNote:
      'Every light-bar in the prism represents real magnitudes from this forward pass.',
  },
  2: {
    level: 2,
    crumb: 'Chamber',
    title: 'The shared track, the ring, and the prism',
    honesty: {
      kind: 'abstraction',
      label: 'Pre-norm wiring',
      detail:
        'Shows exact pre-LayerNorm residual wiring. Pulse brightness and prism glow are driven by real L2 vector norms from this block.',
      literalLevel: 3,
    },
    answersQuestion: 'What happens inside one of those four glass chambers?',
    createsQuestion: 'How do words look back at each other inside the Attention Ring?',
    nextCta: 'Enter the Harp of Attention',
    copy: {
      kid: 'Think of the unbroken glowing highway as a shared notebook that never gets erased. First, words step into the circular Attention Ring to talk to earlier words and add a note back onto the highway. Next, each word thinks alone inside the amber Prism and adds one more note.',
      curious:
        'Pre-LayerNorm architecture: x -> x + Attn(LN_1(x)) -> x + MLP(LN_2(x)). Attention mixes information across token positions using four parallel heads (the four orbiting orbs); the MLP expands each token 128 -> 512 -> 128 with GELU independently.',
      deep: 'u_l = h_l + MHA_l(LN_1(h_l)); h_{l+1} = u_l + W_{down} · GELU_{tanh}(LN_2(u_l) W_{up}^T). Click any orbiting orb to enter that head, or unplug it to zero its weighted-value slice live.',
    },
    simplificationNote:
      'Click any glowing Head Orb to inspect it, or pop it out to see what changes.',
  },
  3: {
    level: 3,
    crumb: 'Harp of Attention',
    title: 'Living ribbons of attention',
    honesty: {
      kind: 'literal',
      label: 'Literal data',
      detail:
        'Every ribbon thickness and brightness is the exact post-softmax attention weight computed by this head on your sentence. Unfold numbers to read every float.',
    },
    answersQuestion: 'How do words look back at each other inside the Attention Ring?',
    createsQuestion: 'What happens if we unplug one of these four attention orbs?',
    nextCta: null,
    copy: {
      kid: 'Your words float in an arc. Bright, thick ribbons of light show which earlier words a word is leaning on right now. Hover any word tile to pluck its ribbons, tap the four glowing orbs above to switch heads, or pop an orb out of its socket to watch the answer bubbles wobble.',
      curious:
        'Each arc connects a query token to an earlier key token with thickness equal to its post-softmax attention weight (rows sum to 1.000). Future tokens are causally masked. Popping out an orb zeros that head’s [T, 32] contribution before W_o and recomputes the full pass.',
      deep: 'A_h = softmax(mask(Q_h K_h^T / sqrt(32))) ∈ R^{T×T} for head h ∈ {0..3}. Hover any tile to highlight row/column bindings; toggle "Unfold numbers" to inspect exact post-softmax or raw pre-softmax QK^T / sqrt(32) scores.',
    },
  },
};

export function registerLabel(r: Register): string {
  if (r === 'kid') return 'Kid';
  if (r === 'curious') return 'Curious';
  return 'Deep';
}
