import type { Register, StageId } from '../state/atlasState';

export interface PresetPrompt {
  id: string;
  label: string;
  text: string;
}

export const PRESET_PROMPTS: PresetPrompt[] = [
  {
    id: 'max-ball',
    label: 'Pronoun link (Max)',
    text: 'Once upon a time, a little dog named Max found a red ball. He',
  },
  {
    id: 'lily-cake',
    label: 'Story clause (Lily)',
    text: 'Lily wanted to bake a cake with her mom, so she',
  },
  {
    id: 'cat-mat',
    label: 'Short clause',
    text: 'The cat sat on the mat and looked at the',
  },
  {
    id: 'confident-error',
    label: 'Pattern vs. fact',
    text: 'The capital of France is a little girl named',
  },
];

export interface StageCurriculum {
  id: StageId;
  navLabel: string;
  title: string;
  honestyKind: 'literal' | 'structural';
  honestyLabel: string;
  nextStage: StageId | null;
  nextQuestion: string | null;
  copy: Record<Register, string>;
}

export const CURRICULUM: Record<StageId, StageCurriculum> = {
  overview: {
    id: 'overview',
    navLabel: '1. Full Model',
    title: 'End-to-End Decoder-Only Transformer',
    honestyKind: 'structural',
    honestyLabel: 'Real tensors on 3D plates',
    nextStage: 'embed',
    nextQuestion: 'How does raw text become the [T, 128] matrix entering Block 0?',
    copy: {
      intuitive:
        'This is the entire 862,464-parameter miniature GPT laid out left to right. Every colored plate shows real activation numbers computed on your sentence (blue is negative, white is zero, amber is positive). Follow the unbroken central Residual Stream highway through 4 identical Transformer blocks—notice the Logit Lens probes above each block sharpening the next-token guess from Block 0 to Block 3.',
      technical:
        'A 4-layer Pre-LayerNorm decoder-only Transformer (d_model = 128, 4 heads of d_head = 32, d_mlp = 512, 512-token BPE vocab, max context 64). Each block reads from the shared [T, 128] residual stream via Multi-Head Self-Attention and a GELU MLP and writes additive updates back to it. Click any section in 3D to fly in and inspect its exact tensors.',
      math: 'h_0 = W_E[x] + W_P[0..T-1] ∈ R^{T×128}; h_{l+1} = Block_l(h_l) for l ∈ {0..3}; logits = LN_f(h_4) W_E^T ∈ R^{T×512}. Logit Lens probes compute argmax softmax(LN_f(h_{l+1,T-1}) W_E^T) at every depth.',
    },
  },
  embed: {
    id: 'embed',
    navLabel: '2. Embeddings',
    title: 'Tokenization, Token Embeddings & Position',
    honestyKind: 'literal',
    honestyLabel: 'Literal [T, 128] slabs',
    nextStage: 'block',
    nextQuestion: 'How does a single Transformer block transform this [T, 128] matrix?',
    copy: {
      intuitive:
        'An LLM never sees raw letters. First, BPE tokenization slices your text into integer IDs from its 512-entry dictionary (including leading spaces). Each ID looks up a row of 128 learned numbers (Token Embedding). Because attention has no built-in sense of word order, a second 128-number vector marking slot 0, 1, 2... (Position Embedding) is added element-by-element to create the starting Residual Stream.',
      technical:
        'Discrete token IDs index rows of wte [512, 128], mapping symbols into continuous semantic space. Learned absolute positional embeddings wpe [64, 128] are added at each position t < T. Every subsequent layer operates strictly on this [T, 128] slab. (Frontier models typically replace additive wpe with RoPE rotations inside Q and K.)',
      math: 'H_0[t] = W_E[x_t] + W_P[t] ∈ R^{128} for t = 0..T-1. Parameters: |W_E| = 65,536 (weight-tied with lm_head), |W_P| = 8,192.',
    },
  },
  block: {
    id: 'block',
    navLabel: '3. Block & Residual',
    title: 'Inside One Transformer Block (Pre-LN + Residual Bus)',
    honestyKind: 'structural',
    honestyLabel: 'Exact Pre-LN wiring',
    nextStage: 'attention',
    nextQuestion: 'Inside Multi-Head Attention, how do tokens compute who to attend to?',
    copy: {
      intuitive:
        'The wide glowing ribbon running straight across the bottom is the Residual Stream—a shared memory bus that is never erased. First, LayerNorm 1 rescales the numbers to keep magnitudes stable. Multi-Head Self-Attention then lets tokens exchange context and adds its update back at the first "+" junction. Next, LayerNorm 2 stabilizes the stream and sends it into the 512-neuron MLP, which processes each token independently and adds a second update at the second "+".',
      technical:
        'Pre-LayerNorm Transformer block: x -> x + MHA(LN_1(x)) -> x + MLP(LN_2(x)). Notice the division of labor: Attention is the ONLY stage where token positions communicate; LayerNorm and MLP act strictly per-token. The unbroken residual path also gives gradients a clean highway during backpropagation.',
      math: 'u_l = h_l + MHA_l(LN_1(h_l)); h_{l+1} = u_l + W_{down} · GELU(LN_2(u_l) W_{up}^T), with LN(v) = ((v - μ) / sqrt(σ^2 + 1e-5)) ⊙ γ + β.',
    },
  },
  attention: {
    id: 'attention',
    navLabel: '4. Attention (QKV)',
    title: 'Multi-Head Self-Attention Unrolled',
    honestyKind: 'literal',
    honestyLabel: 'Literal Q, K, V & A matrices',
    nextStage: 'mlp',
    nextQuestion: 'After tokens mix context in Attention, what does the 512-neuron MLP do?',
    copy: {
      intuitive:
        'Each token is projected into three 32-number vectors per head: Query (what I am looking for), Key (what I advertise), and Value (the information I will pass along). Multiplying Q against Kᵀ produces a raw compatibility score for every pair of tokens. The matte barrier seals off future columns (Causal Mask) so tokens cannot peek ahead. Row-wise Softmax turns each row into weights summing to 1.000, which then blend the Value vectors into A·V before all 4 heads recombine through W_o.',
      technical:
        'Inspecting one head (d_head = 32) of 4 parallel heads. Height and color on the center [T, T] 3D grid show exact post-softmax weights (toggle to inspect raw QKᵀ / √32 scores). Hover any token pill below to highlight its query row, key column, and strongest attention arcs simultaneously.',
      math: 'S_h = Q_h K_h^T / √32; M_{i,j} = S_{i,j} if j <= i else -1e9; A_h = softmax(M_h); Z_h = A_h V_h ∈ R^{T×32}; Out = Concat(Z_0..Z_3) W_O^T.',
    },
  },
  mlp: {
    id: 'mlp',
    navLabel: '5. MLP & GELU',
    title: 'Feed-Forward Network (128 → 512 GELU → 128)',
    honestyKind: 'literal',
    honestyLabel: 'Literal [T, 512] activations',
    nextStage: 'output',
    nextQuestion: 'How does the final [T, 128] vector become a probability over 512 tokens?',
    copy: {
      intuitive:
        'Two-thirds of a Transformer’s parameters live right here in the MLP. Every token’s 128-number vector is multiplied by W_up to fan out 4× wider into 512 feature-detector neurons. The non-linear GELU curve suppresses negative/weak matches (creating sparsity) while letting strong pattern matches blaze bright amber. Finally, W_down folds those surviving features back to 128 numbers to add onto the residual stream.',
      technical:
        'The MLP operates identically and independently on every token position—acting as a learned key-value pattern memory. Without non-linear GELU, stacked linear layers would collapse into a single matrix multiply. Watch the 3D activation wall below: only a fraction of the 512 neurons fire strongly for the final token.',
      math: 'pre = LN_2(u) W_{up}^T ∈ R^{T×512}; hidden = 0.5 · pre · (1 + tanh(√(2/π)(pre + 0.044715 · pre^3))); out = hidden · W_{down}^T ∈ R^{T×128}.',
    },
  },
  output: {
    id: 'output',
    navLabel: '6. Output & Loop',
    title: 'Final Norm, Unembedding, Softmax & Autoregressive Sampling',
    honestyKind: 'literal',
    honestyLabel: 'Literal logits & probabilities',
    nextStage: null,
    nextQuestion: null,
    copy: {
      intuitive:
        'After Block 3, the final token vector passes through Final LayerNorm and multiplies against the 512-row vocabulary matrix (tied to wte) to score every possible next token—these raw scores are Logits. Softmax turns them into probabilities summing to 100%. The network itself is 100% deterministic; randomness happens strictly here via Temperature, Top-k, and Top-p. Click "Step +1 Token" to pick one token, append it to the sentence, and watch the autoregressive loop run again.',
      technical:
        'Final logits = LN_f(h_4[T-1]) W_E^T ∈ R^{512}. Blue 3D bars are inside your active Top-k / Top-p sampling nucleus; muted bars are zeroed out by the cutoff. Lowering temperature sharpens the distribution toward argmax; raising it flattens probability mass toward lower-ranked tokens.',
      math: 'P_τ(v) = exp(logit_v / τ) / Σ_{u ∈ Nucleus(k,p)} exp(logit_u / τ). Cross-entropy loss over the prompt: L = -(1/(T-1)) Σ_{t=0}^{T-2} log P(x_{t+1} | x_{<=t}).',
    },
  },
};

export interface ScaleStop {
  label: string;
  params: number;
  layers: number;
  dModel: number;
  heads: number;
  context: number;
  note: string;
}

export const SCALE_STOPS: ScaleStop[] = [
  {
    label: 'This Atlas Model (TinyStories)',
    params: 862_464,
    layers: 4,
    dModel: 128,
    heads: 4,
    context: 64,
    note: '1.65 MB float16. Same core Transformer math; writes simple story clauses, no encyclopedia knowledge.',
  },
  {
    label: 'GPT-2 Small (2019)',
    params: 124_000_000,
    layers: 12,
    dModel: 768,
    heads: 12,
    context: 1024,
    note: '144× larger. Holds multi-paragraph coherence and basic factual patterns.',
  },
  {
    label: '8B Open-Weight Class',
    params: 8_000_000_000,
    layers: 32,
    dModel: 4096,
    heads: 32,
    context: 8192,
    note: '9,275× larger. Uses RoPE, GQA, RMSNorm, and SwiGLU; capable of coding and broad world recall.',
  },
  {
    label: '70B Class',
    params: 70_000_000_000,
    layers: 80,
    dModel: 8192,
    heads: 64,
    context: 128000,
    note: '81,163× larger. Reliable multi-step reasoning and nuanced instruction following emerge.',
  },
  {
    label: 'Frontier Scale (~1T+)',
    params: 1_000_000_000_000,
    layers: 128,
    dModel: 16384,
    heads: 128,
    context: 1000000,
    note: 'Over 1.15M× larger. Typically uses sparse Mixture-of-Experts routing while preserving the exact residual attention + FFN loop.',
  },
];

export interface FrontierBridge {
  concept: string;
  here: string;
  frontier: string;
}

export const FRONTIER_BRIDGES: FrontierBridge[] = [
  {
    concept: 'Positional Encoding',
    here: 'Learned additive wpe [64, 128]',
    frontier: 'RoPE (rotates Q & K vectors by relative angle for long context)',
  },
  {
    concept: 'Normalization',
    here: 'Pre-LayerNorm (mean + variance)',
    frontier: 'Pre-RMSNorm (root-mean-square only; faster, equally stable)',
  },
  {
    concept: 'Attention Heads',
    here: 'Standard MHA (4 Q, 4 K, 4 V heads)',
    frontier: 'Grouped-Query Attention (GQA) + KV Cache to cut memory bandwidth',
  },
  {
    concept: 'MLP / FFN',
    here: 'Dense GELU (128 → 512 → 128)',
    frontier: 'SwiGLU gated FFN + Mixture-of-Experts (MoE) sparse routing',
  },
  {
    concept: 'Training & Alignment',
    here: 'Next-token pretraining only',
    frontier: 'Pretraining + Supervised Fine-Tuning (SFT) + RLHF / RL reasoning',
  },
];

export function registerLabel(r: Register): string {
  if (r === 'intuitive') return 'Intuitive';
  if (r === 'technical') return 'Technical';
  return 'Math';
}
