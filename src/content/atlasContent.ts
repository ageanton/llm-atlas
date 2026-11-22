import type { Register, ZoomLevel } from '../state/urlState';

export interface PresetPrompt {
  id: string;
  label: string;
  text: string;
  note: string;
}

export const PRESET_PROMPTS: PresetPrompt[] = [
  {
    id: 'max-ball',
    label: 'Pronoun tracking (Max)',
    text: 'Once upon a time, a little dog named Max found a red ball. He',
    note: 'Shows how attention links "He" back to "Max" and "ball".',
  },
  {
    id: 'lily-cake',
    label: 'Story continuation (Lily)',
    text: 'Lily wanted to bake a cake with her mom, so she',
    note: 'Demonstrates sequential story grammar and action verbs.',
  },
  {
    id: 'cat-mat',
    label: 'Short clause',
    text: 'The cat sat on the mat and looked at the',
    note: 'Compact 10-token sequence ideal for reading every attention cell.',
  },
  {
    id: 'confident-error',
    label: 'Confident mistake (fact vs. pattern)',
    text: 'The capital of France is a little girl named',
    note: 'Trained only on children stories, the model completes a story pattern instead of geography.',
  },
];

export interface HonestyMeta {
  kind: 'literal' | 'abstraction';
  note: string;
  literalLevel?: ZoomLevel;
  mathDetail?: string;
}

export interface LevelExplanationSpec {
  level: ZoomLevel;
  title: string;
  shortName: string;
  honesty: HonestyMeta;
  copy: Record<Register, string>;
  answersQuestion: string | null;
  createsQuestion: string | null;
  nextPromptLabel: string | null;
}

export const LEVEL_SPECS: Record<ZoomLevel, LevelExplanationSpec> = {
  0: {
    level: 0,
    title: 'The Black Box & Text In/Out',
    shortName: 'L0: Whole Model',
    honesty: {
      kind: 'abstraction',
      note: 'Treats the model as a discrete text-in / text-out transformer without revealing vectors.',
      literalLevel: 1,
    },
    copy: {
      simple:
        'You give the model words, and it guesses which word should come next. It does this over and over to write a whole story.',
      sufficient:
        'The model is an autoregressive token predictor. It breaks your sentence into discrete tokens, runs them forward through all layers, and yields a probability distribution over vocabulary tokens for position T+1.',
      complete:
        'A sequence x = (x_1, ..., x_T) is mapped autoregressively to p(x_{T+1} | x_{1:T}) = softmax(W_U · LN(h_T^{(L)})). Generation samples iteratively from this categorical distribution.',
    },
    answersQuestion: null,
    createsQuestion: 'What actually happens inside the box between input words and the next word?',
    nextPromptLabel: 'Zoom into the layer pipeline',
  },
  1: {
    level: 1,
    title: 'The Residual Stream Highway',
    shortName: 'L1: Pipeline',
    honesty: {
      kind: 'literal',
      note: 'Shows the exact sequential order of embeddings, 4 transformer blocks, final LayerNorm, and unembedding.',
      mathDetail: 'x_{l+1} = x_l + Attn(LN(x_l)) + MLP(LN(x_l + Attn(...)))',
    },
    copy: {
      simple:
        'Inside, words ride along an information conveyor belt. Each block reads what is on the belt, writes its own updates back onto it, and leaves the original words intact.',
      sufficient:
        'The residual stream is a 128-dimensional vector highway. Each of the 4 blocks reads from the stream through LayerNorm and adds its attention and MLP outputs back to the highway via residual skip connections.',
      complete:
        'Residual stream x_l ∈ R^{T × 128} evolves additively: x_{l, mid} = x_l + Attn(LN_1(x_l)), x_{l+1} = x_{l, mid} + MLP(LN_2(x_{l, mid})). Skip connections prevent vanishing gradients across all 4 layers.',
    },
    answersQuestion: 'How is work organized between the raw text and the prediction?',
    createsQuestion: 'Why does each block have two distinct halves (Attention vs MLP)?',
    nextPromptLabel: 'Inspect an individual Block',
  },
  2: {
    level: 2,
    title: 'The Anatomy of a Transformer Block',
    shortName: 'L2: Single Block',
    honesty: {
      kind: 'literal',
      note: 'Faithful architectural diagram of Pre-LN Multi-Head Attention and two-layer MLP with GELU.',
      mathDetail: 'Attn moves info across tokens; MLP computes nonlinear features per token.',
    },
    copy: {
      simple:
        'Each block has two jobs: the attention team lets words talk to other words across time, and the MLP thinking room lets each word reflect privately on what it just heard.',
      sufficient:
        'Multi-Head Attention enables tokens to communicate and route context across sequence positions. The subsequent Multi-Layer Perceptron (dim 128 → 512 → 128) applies point-wise nonlinear transformations to each token independently.',
      complete:
        'Attention operates across the sequence dimension T via 4 heads of dimension 32. The feed-forward network applies point-wise affine projections and approximate GELU: MLP(z) = W_{c_proj} · GELU(W_{c_fc} · z + b_{fc}) + b_{proj}.',
    },
    answersQuestion: 'What does each block do to the stream?',
    createsQuestion: 'How does attention decide which earlier words are relevant to the current word?',
    nextPromptLabel: 'Examine Multi-Head Attention',
  },
  3: {
    level: 3,
    title: 'Attention Head Mechanism (Q, K, V & Scores)',
    shortName: 'L3: Attention Head',
    honesty: {
      kind: 'literal',
      note: 'Displays true calculated attention matrices A = softmax(Q K^T / sqrt(d_k) + causal_mask).',
      mathDetail: 'A_{i,j} = exp(q_i · k_j / sqrt(32)) / sum_{m<=i} exp(q_i · k_m / sqrt(32))',
    },
    copy: {
      simple:
        'Every word creates a Question (Query) and a Name Tag (Key). When a Question matches a Name Tag, attention flows and information (Value) gets copied over.',
      sufficient:
        'Each of the 4 attention heads projects tokens into Queries and Keys of dimension 32. The dot product measures relevance. Softmax normalizes row weights with a causal mask so words can never look into the future.',
      complete:
        'For head h, Q_h = x W_Q^{(h)}, K_h = x W_K^{(h)}, V_h = x W_V^{(h)}. Attention weights A_h = softmax(masked(Q_h K_h^T / sqrt(32))). Head output is O_h = A_h V_h, projected by W_O.',
    },
    answersQuestion: 'How does attention decide which earlier words matter?',
    createsQuestion: 'Where do these Query, Key, and Value vectors actually come from?',
    nextPromptLabel: 'Inspect Weight & Activation Tensors',
  },
  4: {
    level: 4,
    title: 'Weight Matrices & Activation Vectors',
    shortName: 'L4: Tensors',
    honesty: {
      kind: 'literal',
      note: 'Direct 2D heatmaps of real float16 parameters and runtime activations straight from the worker.',
      mathDetail: '16 weight tensors totaling 866,304 parameters.',
    },
    copy: {
      simple:
        'Under the hood, all thoughts are grids of real numbers called matrices. When numbers line up, they multiply and add up to create new thoughts.',
      sufficient:
        'Every layer projection is a matrix multiplication. Parameter matrices (e.g. c_attn 128×384, c_proj 128×128, c_fc 128×512) hold learned knowledge; activation matrices hold runtime token states.',
      complete:
        'Parameters θ ∈ R^{866,304} are stored in float16 (1.65 MB). The browser worker evaluates Y = X · W + b using 32-bit floating point accumulation with real memory buffers.',
    },
    answersQuestion: 'Where do the vectors come from?',
    createsQuestion: 'What does an individual number inside a matrix mean, and how did it learn that value?',
    nextPromptLabel: 'Drill down to a single Scalar Weight',
  },
  5: {
    level: 5,
    title: 'Individual Float16 Scalar & Training Trajectory',
    shortName: 'L5: Scalar Weight',
    honesty: {
      kind: 'literal',
      note: 'Every scalar is backed by the real binary weights file and training history from GitHub Actions.',
      mathDetail: 'θ_{t+1} = θ_t - η · m_t / (sqrt(v_t) + ε)',
    },
    copy: {
      simple:
        'This is a single number in the brain of the model. During training, every sentence nudged it slightly up or down so the model could predict better.',
      sufficient:
        'A single scalar parameter θ_{i,j}. During training via AdamW (lr=3e-3), its value was updated by stochastic gradient descent over 1,500 batches to minimize cross-entropy loss.',
      complete:
        'Coordinate-level float16 value with reconstructed AdamW trajectory across training checkpoints. Gradients ∂L/∂θ accumulated across batch tokens drove coordinate motion.',
    },
    answersQuestion: 'What is a single number and how did it learn?',
    createsQuestion: null,
    nextPromptLabel: null,
  },
};

export interface StationExplorationInfo {
  id: string;
  name: string;
  role: string;
  shapeNote: string;
}

export const STATION_INFO: Record<string, StationExplorationInfo> = {
  model: {
    id: 'model',
    name: 'Complete Transformer Model',
    role: '866K-parameter causal language model trained on TinyStories.',
    shapeNote: '4 blocks · 4 heads · 128 hidden dim · 512 MLP dim',
  },
  tokenizer: {
    id: 'tokenizer',
    name: 'Byte-Pair Encoding (BPE) Tokenizer',
    role: 'Converts raw string text into discrete integer token IDs.',
    shapeNote: 'Vocabulary size: 2,048 tokens',
  },
  embed: {
    id: 'embed',
    name: 'Token + Position Embedding',
    role: 'Looks up 128-dimensional semantic vector and adds learned positional encoding.',
    shapeNote: 'wte: 2,048 × 128 · wpe: 128 × 128',
  },
  block0: {
    id: 'block0',
    name: 'Transformer Block 0',
    role: 'Early features: local bigrams, punctuation linking, and raw surface syntax.',
    shapeNote: 'Residual stream dim: 128',
  },
  block1: {
    id: 'block1',
    name: 'Transformer Block 1',
    role: 'Syntactic parsing, noun-phrase grouping, and subject agreement.',
    shapeNote: 'Residual stream dim: 128',
  },
  block2: {
    id: 'block2',
    name: 'Transformer Block 2',
    role: 'Semantic binding: linking pronouns to character entities and action verbs.',
    shapeNote: 'Residual stream dim: 128',
  },
  block3: {
    id: 'block3',
    name: 'Transformer Block 3',
    role: 'High-level narrative completion and next-token probability distribution prep.',
    shapeNote: 'Residual stream dim: 128',
  },
  ln_f: {
    id: 'ln_f',
    name: 'Final Layer Normalization',
    role: 'Normalizes the residual stream to zero mean and unit variance before unembedding.',
    shapeNote: 'Gamma / Beta dim: 128',
  },
  softmax: {
    id: 'softmax',
    name: 'Unembedding & Softmax',
    role: 'Projects 128-dim stream back to 2,048 logits and computes probabilities.',
    shapeNote: 'Tied with wte: 128 × 2,048',
  },
};

export interface ScaleStop {
  name: string;
  params: string;
  layers: number;
  dModel: number;
  heads: number;
  context: number;
  note: string;
}

export const SCALE_LADDER_STOPS: ScaleStop[] = [
  {
    name: 'TinyStories-4L-128D (This explainer)',
    params: '866K',
    layers: 4,
    dModel: 128,
    heads: 4,
    context: 64,
    note: 'Real miniature model executing live in your browser web worker.',
  },
  {
    name: 'GPT-2 Small (2019)',
    params: '124M',
    layers: 12,
    dModel: 768,
    heads: 12,
    context: 1024,
    note: '143× larger than this atlas model. Uses identical Pre-LN transformer block architecture.',
  },
  {
    name: 'Llama 3 8B (2024)',
    params: '8.03B',
    layers: 32,
    dModel: 4096,
    heads: 32,
    context: 8192,
    note: '9,268× larger than this atlas model. Adds RoPE, SwiGLU, and Grouped-Query Attention (GQA).',
  },
  {
    name: 'Llama 3 70B (2024)',
    params: '70.6B',
    layers: 80,
    dModel: 8192,
    heads: 64,
    context: 8192,
    note: '81,495× larger than this atlas model. Requires multi-GPU cluster for inference.',
  },
  {
    name: 'Frontier LLM (e.g. Gemini 1.5 Pro / GPT-4)',
    params: '~1T+',
    layers: 128,
    dModel: 16384,
    heads: 128,
    context: 1000000,
    note: 'Over 1,159,000× larger than this atlas model. Often uses Mixture-of-Experts routing, yet still predicts the next token through stacked attention and MLP residual updates.',
  },
];

export function formatRegisterName(r: Register): string {
  if (r === 'simple') return 'Simple';
  if (r === 'sufficient') return 'Sufficient';
  return 'Complete';
}

export function verifyQuestionChain(): boolean {
  for (let lv = 0; lv < 5; lv++) {
    const cur = LEVEL_SPECS[lv as ZoomLevel];
    const next = LEVEL_SPECS[(lv + 1) as ZoomLevel];
    if (!cur.createsQuestion || cur.createsQuestion !== next.answersQuestion) {
      return false;
    }
  }
  return true;
}
