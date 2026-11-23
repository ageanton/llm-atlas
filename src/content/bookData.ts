import type { Edition } from '../state/bookState';

export interface BeatRegister {
  title: string;
  summary: string;
  mathFormula?: string;
  codeSnippet?: string;
  bulletPoints: string[];
  intuitionTrap?: string;
}

export interface PredictChallenge {
  id: string;
  question: string;
  options: { id: string; label: string }[];
  correctId: string;
  explanation: string;
}

export interface ChapterBeat {
  id: string;
  beatIndex: number;
  title: string;
  subtitle: string;
  stageFocus:
    | 'attn-1-qkv-projection'
    | 'attn-2-dot-product'
    | 'attn-3-scale-by-root-d'
    | 'attn-4-mask-softmax'
    | 'attn-5-value-mixer'
    | 'attn-6-multihead-full';
  registers: Record<Edition, BeatRegister>;
  predictChallenge?: PredictChallenge;
  interactiveGoal: string;
}

export interface Chapter {
  id: number;
  title: string;
  subtitle: string;
  status: 'ready' | 'upcoming';
  openingMystery: string;
  beats: ChapterBeat[];
}

export const CHAPTER_4_BEATS: ChapterBeat[] = [
  {
    id: 'attn-1-qkv-projection',
    beatIndex: 0,
    title: 'The Three Roles: Query, Key, and Value',
    subtitle: 'Why one vector must split into three distinct search vectors',
    stageFocus: 'attn-1-qkv-projection',
    interactiveGoal:
      'Click through the prompt tokens below to observe how each word projects into three parallel 16-number vectors.',
    predictChallenge: {
      id: 'p1-qkv-roles',
      question:
        'If a token used the exact same vector for both Query and Key, what pathological behavior would always happen?',
      options: [
        {
          id: 'opt-a',
          label: 'Every token would attend almost entirely to itself.',
        },
        {
          id: 'opt-b',
          label: 'The output tensor would collapse to pure zeros.',
        },
        {
          id: 'opt-c',
          label: 'The causal mask would fail to block future tokens.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'Because dot-product measures alignment, a vector’s dot product with itself (||v||²) is virtually always much larger than with different words, causing every token to drown out its context and speak only to itself.',
    },
    registers: {
      foundations: {
        title: 'Asking, Advertising, and Delivering',
        summary:
          'A single word cannot do three jobs at once. When "He" enters attention, it needs to ask questions about the past, advertise its own attributes to future words, and carry the content to be blended.',
        bulletPoints: [
          'Query (Q): "What am I looking for?" (e.g. He asks: Who is the masculine character?)',
          'Key (K): "What attributes do I advertise?" (e.g. Max advertises: I am a little male dog)',
          'Value (V): "What factual content do I deliver if matched?" (e.g. delivers the dog/pet concept)',
          'Notice how 64 numbers in the residual stream get multiplied by weight matrices to produce three 16-number vectors per head.',
        ],
        intuitionTrap:
          'Do not think of Q, K, and V as distinct words. Every single token produces its OWN Query, its OWN Key, and its OWN Value.',
      },
      practitioner: {
        title: 'Linear Projections: Wq, Wk, Wv',
        summary:
          'Input tensor x ∈ ℝ^{B×T×64} is multiplied by three weight matrices Wq, Wk, Wv ∈ ℝ^{64×64}. In our model, 64 dimensions split across 4 heads of dimension d_head = 16.',
        codeSnippet: `# PyTorch equivalent verified against our TS engine:
q = self.q_proj(x).view(B, T, 4, 16).transpose(1, 2)  # [B, 4, T, 16]
k = self.k_proj(x).view(B, T, 4, 16).transpose(1, 2)  # [B, 4, T, 16]
v = self.v_proj(x).view(B, T, 4, 16).transpose(1, 2)  # [B, 4, T, 16]`,
        bulletPoints: [
          'Input dim d_model = 64, number of heads = 4, per-head dim d_head = 16.',
          'Weights: W_q, W_k, W_v each have 64 × 64 = 4,096 fp16 parameters per block.',
          'Inference memory: intermediate activations for Q, K, V are retained for backprop during training, but during generation K and V are appended to the KV cache.',
        ],
      },
      formal: {
        title: 'Affine Subspace Mapping and Geometric Decoupling',
        summary:
          'Let x_t ∈ ℝ^{d} denote the representation of token t in the residual stream. Self-attention projects x_t into three distinct linear subspaces: q_t = W_q x_t, k_t = W_k x_t, v_t = W_v x_t.',
        mathFormula:
          '\\mathbf{q}_t = \\mathbf{W}_q \\mathbf{x}_t, \\quad \\mathbf{k}_t = \\mathbf{W}_k \\mathbf{x}_t, \\quad \\mathbf{v}_t = \\mathbf{W}_v \\mathbf{x}_t \\quad \\in \\mathbb{R}^{d_{head}}',
        bulletPoints: [
          'Rank and expressivity: Linear maps decouple search addressing (q·k) from semantic content transfer (v).',
          'Coordinate invariance: The metric space for addressing need not match the metric space of the residual manifold.',
          'Parameter count: 3 × d_model² parameters per attention layer, totaling 12,288 weights per block.',
        ],
      },
    },
  },
  {
    id: 'attn-2-dot-product',
    beatIndex: 1,
    title: 'The Dot Product: Measuring Alignment',
    subtitle: 'Multiplying matching dimensions to score relevance between tokens',
    stageFocus: 'attn-2-dot-product',
    interactiveGoal:
      'Select Query token "He" and compare it against Key tokens "Max", "dog", and "ball" to see the elementwise multiplications sum into a score.',
    predictChallenge: {
      id: 'p2-dot-product',
      question:
        'If Query q and Key k have opposite signs on almost all 16 coordinates, what will the dot product be?',
      options: [
        {
          id: 'opt-a',
          label: 'A large negative number, signaling strong incompatibility.',
        },
        {
          id: 'opt-b',
          label: 'Zero, signaling that the tokens are completely independent.',
        },
        {
          id: 'opt-c',
          label: 'A positive number because squared magnitudes dominate.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'The dot product is the sum of elementwise products. If coordinates have opposite signs (+ times -), every term is negative, summing to a large negative score that softmax turns into nearly 0% attention.',
    },
    registers: {
      foundations: {
        title: 'Comparing Notes Across Time',
        summary:
          'To determine how much "He" cares about "Max", the model takes the 16 numbers of Query("He") and multiplies them coordinate-by-coordinate with the 16 numbers of Key("Max"), then sums them up.',
        bulletPoints: [
          'High positive score: The two tokens are looking for each other (mutual match).',
          'Zero score: Orthogonal, no semantic relation on this specific head.',
          'Negative score: Active suppression or grammatical mismatch.',
          'Observe the 3D bars: blue bars are positive matches, coral bars are negative penalties.',
        ],
      },
      practitioner: {
        title: 'All-Pairs Affinity Matrix: S = Q Kᵀ',
        summary:
          'Batch matrix multiplication computes all dot products simultaneously in a single BLAS GEMM call.',
        codeSnippet: `# Matrix multiplication over sequence length T:
scores = torch.matmul(q, k.transpose(-2, -1))  # [B, 4, T, T]
# scores[b, h, i, j] = sum_d (q[b, h, i, d] * k[b, h, j, d])`,
        bulletPoints: [
          'Complexity: O(T² · d_head) arithmetic operations.',
          'FLOPs: For sequence length T=15 and d_head=16, exactly 15 × 15 × 16 = 3,600 multiply-adds per head.',
          'Hardware friendliness: GEMM runs at peak tensor-core efficiency on modern GPUs.',
        ],
      },
      formal: {
        title: 'Bilinear Form & Unnormalized Affinity Kernel',
        summary:
          'The attention score between query i and key j is a bilinear form over the residual vectors: a(x_i, x_j) = x_iᵀ (W_qᵀ W_k) x_j.',
        mathFormula:
          'S_{i,j} = \\langle \\mathbf{q}_i, \\mathbf{k}_j \\rangle = \\sum_{c=1}^{d_{head}} q_{i,c} \\cdot k_{j,c}',
        bulletPoints: [
          'Symmetry: Even if W_q = W_k, S is non-symmetric across token positions i and j because x_i ≠ x_j.',
          'Gram Matrix: S represents an unnormalized kernel Gram matrix in the projected embedding space.',
        ],
      },
    },
  },
  {
    id: 'attn-3-scale-by-root-d',
    beatIndex: 2,
    title: 'The √d Scale: Preventing Softmax Saturation',
    subtitle: 'Why dividing by √16 = 4 keeps gradients alive',
    stageFocus: 'attn-3-scale-by-root-d',
    interactiveGoal:
      'Toggle between unscaled dot products and scaled dot products to see how dividing by 4 pulls extreme numbers back into the sensitive gradient zone.',
    predictChallenge: {
      id: 'p3-scale-variance',
      question:
        'If two independent vectors have components with mean 0 and variance 1, what is the variance of their 16-dimensional dot product?',
      options: [
        {
          id: 'opt-a',
          label: '16. The variance grows linearly with dimension d.',
        },
        {
          id: 'opt-b',
          label: '4. The standard deviation equals dimension d.',
        },
        {
          id: 'opt-c',
          label: '1. The dot product normalizes variance automatically.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'The dot product is the sum of d independent products, each having variance 1. The variance of a sum of independent random variables is the sum of variances: 16 × 1 = 16. Dividing by √16 = 4 restores the variance back to 1!',
    },
    registers: {
      foundations: {
        title: 'The Taming of Large Numbers',
        summary:
          'When you add up 16 multiplied numbers, the total naturally swings wildly. If numbers get too big (+20 or -20), softmax turns into an all-or-nothing switch.',
        bulletPoints: [
          'Without scaling: A single large product dominates, causing the model to pick only one word with 99.99% confidence.',
          'When softmax saturates: Learning grinds to a halt because gradients vanish to zero.',
          'Dividing by √16 = 4 cools down the temperature and allows balanced consideration of multiple context words.',
        ],
      },
      practitioner: {
        title: 'Variance Normalization & Softmax Gradients',
        summary:
          'For d_head = 16, 1 / √d = 0.25. Multiplying by 0.25 scales the variance from 16 down to 1.0, preserving healthy ∂Softmax/∂z gradients.',
        codeSnippet: `scale = 1.0 / math.sqrt(16)  # exactly 0.25
scaled_scores = scores * scale`,
        bulletPoints: [
          'Implementation: Multiplied as a constant scalar before causal masking and softmax.',
          'Scale factor for frontier models: Llama-3 uses d_head = 128, so 1 / √128 ≈ 0.088388.',
          'Numerical stability: Keeps exponential inputs inside the well-behaved [-10, 10] float range.',
        ],
      },
      formal: {
        title: 'Stochastic Stability of High-Dimensional Inner Products',
        summary:
          'Let q_c, k_c ∼ i.i.d. 𝒩(0, 1). Then 𝔼[q·k] = 0 and Var(q·k) = d. By dividing by √d, Var((q·k)/√d) = 1.',
        mathFormula:
          '\\widetilde{S}_{i,j} = \\frac{\\mathbf{q}_i^T \\mathbf{k}_j}{\\sqrt{d_{head}}} = \\frac{1}{4} \\sum_{c=1}^{16} q_{i,c} k_{j,c}',
        bulletPoints: [
          'Softmax temperature: As d → ∞, unscaled inner products push softmax towards a degenerate Dirac delta distribution.',
          'Vanishing gradients: The derivative of softmax(z)_i with respect to z_j decays exponentially as max(z) - min(z) diverges.',
        ],
      },
    },
  },
  {
    id: 'attn-4-mask-softmax',
    beatIndex: 3,
    title: 'The Causal Mask & Softmax Normalization',
    subtitle: 'Enforcing the arrow of time (-∞) and turning affinities into exact percentages',
    stageFocus: 'attn-4-mask-softmax',
    interactiveGoal:
      'Observe the upper-right triangle covered in dark slate tiles (-∞). Toggle the button to inspect raw pre-softmax logits vs normalized probabilities.',
    predictChallenge: {
      id: 'p4-future-mask',
      question:
        'What would happen during autoregressive text generation if the causal mask were removed?',
      options: [
        {
          id: 'opt-a',
          label:
            'During training the model would cheat by reading the answers from the future, failing completely when generating text alone.',
        },
        {
          id: 'opt-b',
          label: 'Generation would run twice as fast because of symmetry.',
        },
        {
          id: 'opt-c',
          label: 'The model would only be able to generate palindromes.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'If future positions are unmasked during training, token 5 simply copies token 6. At test time, token 6 does not exist yet, so the model produces incoherent gibberish.',
    },
    registers: {
      foundations: {
        title: 'The Wall of Time and 100% Budgets',
        summary:
          'Language moves in one direction. A word can only look back at the past. The future is masked out with negative infinity, ensuring e^{-∞} = 0%.',
        bulletPoints: [
          'Lower triangle: Allowed to attend (past and self).',
          'Upper triangle: Strictly forbidden (future). Set to -10,000 (represented as -∞).',
          'Softmax converts raw scores into positive percentages that sum to exactly 1.00 (100%) across every row.',
        ],
      },
      practitioner: {
        title: 'Triangular Causal Masking & Numerically Stable Softmax',
        summary:
          'A lower triangular boolean mask replaces future positions with -inf, followed by subtraction of row-max before exponentiation to prevent float overflow.',
        codeSnippet: `mask = torch.tril(torch.ones(T, T)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
probs = torch.softmax(scores, dim=-1)  # sums to 1.0 per row`,
        bulletPoints: [
          'Float16 precision: In fp16, we use -65,504 or -10,000 to avoid IEEE 754 NaN overflows.',
          'Memory footprint: In flash-attention, the T×T attention matrix is tiled in SRAM and never written to main VRAM.',
        ],
      },
      formal: {
        title: 'Simplex Projection on the Causal Cone',
        summary:
          'Causal self-attention restricts information propagation to the directed acyclic graph defined by the causal order t ≤ i.',
        mathFormula:
          'A_{i,j} = \\frac{\\exp(\\widetilde{S}_{i,j} + M_{i,j})}{\\sum_{k=1}^T \\exp(\\widetilde{S}_{i,k} + M_{i,k})}, \\quad M_{i,j} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}',
        bulletPoints: [
          'Probability Simplex: For all i, A_{i,:} ∈ Δ^{T-1}, where ∑_{j=1}^T A_{i,j} = 1 and A_{i,j} ≥ 0.',
          'Causal Invariance: A_{i,j} = 0 for all j > i guarantees ∂output_i / ∂input_j = 0 for j > i.',
        ],
      },
    },
  },
  {
    id: 'attn-5-value-mixer',
    beatIndex: 4,
    title: 'Value Mixing: Blending Memories into Context',
    subtitle: 'Using attention weights to take a weighted average of Value vectors',
    stageFocus: 'attn-5-value-mixer',
    interactiveGoal:
      'Hover over token "He" to see its 100% attention budget distributed across past tokens, pulling chunks of their Value vectors into its new representation.',
    predictChallenge: {
      id: 'p5-value-blending',
      question:
        'If token "He" assigns 62% attention to "Max" and 25% to "dog", what does its output vector contain?',
      options: [
        {
          id: 'opt-a',
          label:
            'A blended vector that is mostly Max’s Value vector plus a quarter of dog’s Value vector.',
        },
        {
          id: 'opt-b',
          label: 'The concatenation of Max and dog side-by-side.',
        },
        {
          id: 'opt-c',
          label: 'A randomly chosen token sampled with 62% probability.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'Attention is a linear weighted sum (convex combination). The output vector at position i is literally 0.62·V_Max + 0.25·V_dog + ... blending their information into one new vector.',
    },
    registers: {
      foundations: {
        title: 'The Semantic Smoothie',
        summary:
          'Attention weights are a recipe. If "He" has a 62% weight on "Max" and 25% on "dog", the kitchen blends 62% of Max’s Value vector and 25% of dog’s Value vector into a rich new concept: a male pet dog.',
        bulletPoints: [
          'Query and Key only decided the percentages (the recipe).',
          'Value carries the actual substance that gets moved.',
          'The result is a context-enriched vector ready for the MLP private thinking stage.',
        ],
      },
      practitioner: {
        title: 'Context Matrix GEMM: Out = A · V',
        summary:
          'The attention probability tensor [B, 4, T, T] multiplies the value tensor [B, 4, T, 16] to produce per-head context vectors [B, 4, T, 16].',
        codeSnippet: `# Weighted combination of Value vectors:
head_out = torch.matmul(probs, v)  # [B, 4, T, 16]`,
        bulletPoints: [
          'Convex combination: Every output vector lies within the convex hull of the input Value vectors.',
          'Information routing: Notice that unlike convolution or RNNs, distance in time does not degrade information transfer.',
        ],
      },
      formal: {
        title: 'Convex Hull Operator in the Value Manifold',
        summary:
          'Let V ∈ ℝ^{T × d_head}. The output vector for token i is y_i = ∑_{j=1}^i A_{i,j} v_j, representing a non-local linear aggregation.',
        mathFormula:
          '\\mathbf{y}_i = \\sum_{j=1}^i A_{i,j} \\mathbf{v}_j = [\\mathbf{A} \\mathbf{V}]_i \\quad \\in \\mathbb{R}^{d_{head}}',
        bulletPoints: [
          'Contractive mapping: ||y_i|| ≤ max_j ||v_j|| when measured under convex norms.',
          'Permutation equivariance (unmasked): Attention without positional encoding is fully symmetric under permutation of the input set.',
        ],
      },
    },
  },
  {
    id: 'attn-6-multihead-full',
    beatIndex: 5,
    title: 'The 4-Head Orchestra & Output Projection',
    subtitle: 'Concatenating 4 specialized viewpoints and projecting back with Wo',
    stageFocus: 'attn-6-multihead-full',
    interactiveGoal:
      'Switch between Heads 0, 1, 2, and 3. Disable a head using the Ablation button to see how the downstream prediction changes in real time!',
    predictChallenge: {
      id: 'p6-multihead-purpose',
      question:
        'Why do modern transformers use multiple small heads (4 × 16 = 64) instead of one giant 64-dimensional attention head?',
      options: [
        {
          id: 'opt-a',
          label:
            'Multiple heads allow the model to track several relationships simultaneously (e.g. grammar, subject, and punctuation).',
        },
        {
          id: 'opt-b',
          label: 'Multiple heads use 4× fewer mathematical operations.',
        },
        {
          id: 'opt-c',
          label: 'A single 64-dimensional head cannot compute dot products.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'A single head can only produce one attention distribution per token. Four heads allow token "He" to simultaneously track: Head 0 (gender/subject: Max), Head 1 (previous word: .), Head 2 (action verb), and Head 3 (topic: ball).',
    },
    registers: {
      foundations: {
        title: 'Four Specialists in Every Room',
        summary:
          'Instead of one generalist trying to understand everything, four distinct heads examine the sentence simultaneously from different perspectives.',
        bulletPoints: [
          'Head 0: Grammar detective (who is the subject?).',
          'Head 1: Previous-token tracker (what just happened?).',
          'Head 2: Semantic connector (connects ball to red).',
          'Head 3: Narrative forward looker.',
          'All 4 outputs (4 × 16 = 64 numbers) are glued together and mixed through the Output Matrix W_o.',
        ],
      },
      practitioner: {
        title: 'Concat & Output Projection: MultiHead(Q,K,V) · Wo',
        summary:
          'The 4 heads are concatenated along the head dimension into a 64-wide vector, then multiplied by Wo ∈ ℝ^{64×64} to mix information across heads.',
        codeSnippet: `# Transpose and flatten 4 heads of 16 into single 64-d vector:
out = head_out.transpose(1, 2).contiguous().view(B, T, 64)
# Project back to residual stream:
attn_out = self.c_proj(out)  # [B, T, 64]`,
        bulletPoints: [
          'Parameter count: W_o has 64 × 64 = 4,096 fp16 parameters.',
          'Residual connection: attn_out is added directly to input x: x = x + attn_out.',
          'Surgical ablation: If you disable Head 2, you zero its 16-d slice before W_o projection, observing immediate behavioral drift.',
        ],
      },
      formal: {
        title: 'Multi-Head Subspace Ensemble & Linear Re-projection',
        summary:
          'Multi-head attention projects queries, keys, and values into h distinct representation subspaces, executing attention in parallel and combining via W_o.',
        mathFormula:
          '\\text{MHA}(\\mathbf{x}) = \\left[ \\text{head}_0 \\mathbin{\\Vert} \\text{head}_1 \\mathbin{\\Vert} \\text{head}_2 \\mathbin{\\Vert} \\text{head}_3 \\right] \\mathbf{W}_o',
        bulletPoints: [
          'Subspace rank: While each head has rank at most d_head = 16, the ensemble spans a full rank-64 subspace.',
          'Residual addition: The final block update x^{(l)} = x^{(l-1)} + MHA(LN(x^{(l-1)})) preserves identity routing.',
        ],
      },
    },
  },
];

export const ALL_CHAPTERS: Chapter[] = [
  {
    id: 1,
    title: 'From Words to Numbers',
    subtitle: 'Byte-Pair Encoding, Token Embeddings & Positional Vectors',
    status: 'upcoming',
    openingMystery:
      'How does a computer that only does arithmetic understand what a "dog" or a "red ball" is?',
    beats: [],
  },
  {
    id: 2,
    title: 'The Information Highway',
    subtitle: 'The 64-Dimensional Residual Stream & LayerNorm Gates',
    status: 'upcoming',
    openingMystery:
      'How does a deep stack of layers keep adding new ideas without erasing earlier memory or letting numbers explode?',
    beats: [],
  },
  {
    id: 3,
    title: 'Preparing for the Dance',
    subtitle: 'Layer Normalization, Variance Calibration & Clean Inputs',
    status: 'upcoming',
    openingMystery:
      'Why does every single layer force vectors onto a hypersphere before letting them interact?',
    beats: [],
  },
  {
    id: 4,
    title: 'How Words Talk — Self-Attention',
    subtitle: 'Q/K/V, Scaled Dot Products, Causal Mask, Softmax, Value Mixing & 4 Heads',
    status: 'ready',
    openingMystery:
      'How does the word "He" figure out it refers to a little dog named Max who found a red ball?',
    beats: CHAPTER_4_BEATS,
  },
  {
    id: 5,
    title: 'Private Thinking & Memory',
    subtitle: 'Feed-Forward Network (MLP), 4× Expansion, GELU, SwiGLU & MoE',
    status: 'upcoming',
    openingMystery:
      'If Attention only averages existing vectors, where does the model think privately and store learned facts?',
    beats: [],
  },
  {
    id: 6,
    title: 'Sharpening a Guess & Writing a Story',
    subtitle: 'Logit Lens, Unembedding, Temperature/Top-k/Top-p, Autoregression & KV Cache',
    status: 'upcoming',
    openingMystery:
      'If the math is 100% deterministic, why does the exact same prompt give different answers?',
    beats: [],
  },
  {
    id: 7,
    title: 'Where Every Number Came From',
    subtitle: 'Cross-Entropy Loss, Manual Backprop, AdamW, Training Replay & Hallucination',
    status: 'upcoming',
    openingMystery:
      'Nobody programmed grammar into this AI—how did 862,464 random numbers organize themselves into English?',
    beats: [],
  },
  {
    id: 8,
    title: 'From Miniature Toy to Frontier AI',
    subtitle: 'Surgical Ablation, Scaling Laws, Modern Upgrades & Post-Training (SFT / RLHF)',
    status: 'upcoming',
    openingMystery:
      'What stays identical and what fundamentally changes across a 1,000,000× jump to trillion-parameter LLMs?',
    beats: [],
  },
];

export function editionLabel(e: Edition): string {
  if (e === 'foundations') return 'Foundations';
  if (e === 'practitioner') return 'Practitioner';
  return 'Formal';
}
