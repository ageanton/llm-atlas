import type { Edition } from '../state/bookState';

export type SceneBeatId =
  | 'attn-1-problem'
  | 'attn-2-dot-product'
  | 'attn-3-scan-row'
  | 'attn-4-mask-softmax'
  | 'attn-5-value-mixer'
  | 'attn-6-multihead-full';

export interface PredictChallenge {
  id: string;
  question: string;
  options: { id: string; label: string }[];
  correctId: string;
  explanation: string;
}

export interface BookBeat {
  id: string;
  beatNumber: number;
  shortName: string;
  title: string;
  subtitle: string;
  sceneBeat: SceneBeatId;
  questionPrompt: string;
  summary: Record<Edition, string>;
  deepDive: Record<Edition, string>;
  equationLatex?: string;
  codeBlock?: string;
  prediction?: PredictChallenge;
  inspectInstructions: string;
}

export interface ChapterMeta {
  id: number;
  title: string;
  subtitle: string;
  readingTimeMinutes: number;
  status: 'published' | 'coming-soon';
  teaser: string;
  beats: BookBeat[];
}

export const PRESET_PROMPTS = [
  'Once upon a time, a little dog named Max found a red ball. He',
  'The girl put the letter inside the blue box, and then she',
  'In a sunny kitchen, the chef picked up a sharp knife and',
  'The small green frog hopped across the quiet pond until it',
];

export const CHAPTER_4_BEATS: BookBeat[] = [
  {
    id: 'c4-b1-problem',
    beatNumber: 1,
    shortName: 'The Addressing Problem',
    title: 'The Addressing Problem: Q, K, and V',
    subtitle:
      'Why one vector must split into three distinct communication roles',
    sceneBeat: 'attn-1-problem',
    questionPrompt:
      'How does the word "He" at token 14 know which past word it should talk to?',
    prediction: {
      id: 'p-qkv-roles',
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
    summary: {
      foundations:
        'In a sentence like "Once upon a time, a little dog named Max found a red ball. He", every token enters block 2 as a 64-number vector in the residual stream. But a single vector cannot simultaneously ask questions, advertise its identity, and carry content. The layer multiplies the residual vector by three weight matrices (Wq, Wk, Wv) to create three 16-number vectors for head 0: Query ("Who is the masculine subject?"), Key ("I am Max, a masculine subject"), and Value ("dog, friendly, small").',
      practitioner:
        'Input x ∈ ℝ^{T×64} is linearly projected by W_q, W_k, W_v ∈ ℝ^{64×64}. In our 4-head architecture, each projection splits into 4 slices of d_head = 16: q, k, v ∈ ℝ^{4×T×16}. W_q projects into query space, W_k into key space, and W_v into value space.',
      formal:
        'Let x_t ∈ ℝ^{d_{model}} be the residual stream representation at position t. The multi-head projection defines three affine maps:\n\nq_{t,h} = W_{q,h} x_t, \\quad k_{t,h} = W_{k,h} x_t, \\quad v_{t,h} = W_{v,h} x_t\n\nwhere W_{q,h}, W_{k,h}, W_{v,h} ∈ ℝ^{d_{head} × d_{model}} (here d_{model}=64, d_{head}=16, h ∈ {0..3}). Decoupling query and key subspaces permits non-symmetric relational addressing.',
    },
    deepDive: {
      foundations:
        'Think of Query as a job opening, Key as a resume, and Value as the employee showing up to work. When "He" enters the room, its Query reads: "Looking for an animate character of male gender." "Max" holds up a Key: "Male character named Max." Because the Query and Key match, "Max" sends its Value vector over to update "He".',
      practitioner:
        'Parameter count for the QKV projections: 3 × (64 × 64) = 12,288 fp16 parameters per block. Across 4 layers, QKV consumes 49,152 parameters. In inference, intermediate Q values are discarded after dot-product computation, while K and V can be cached across generation steps (the KV cache).',
      formal:
        'Decoupling addressing from value transport is mathematically essential. If Q = K = V, attention collapses into a kernel density estimator on the residual manifold itself. By learning distinct projections, the geometry of retrieval (determined by the metric tensor W_q^T W_k) is completely independent of the semantics of transport (governed by W_v).',
    },
    equationLatex:
      '\\mathbf{q}_t = \\mathbf{W}_q \\mathbf{x}_t, \\quad \\mathbf{k}_t = \\mathbf{W}_k \\mathbf{x}_t, \\quad \\mathbf{v}_t = \\mathbf{W}_v \\mathbf{x}_t',
    codeBlock: `# PyTorch equivalent verified against our TypeScript engine:
q = self.q_proj(x).view(B, T, 4, 16).transpose(1, 2)  # [B, 4, T, 16]
k = self.k_proj(x).view(B, T, 4, 16).transpose(1, 2)  # [B, 4, T, 16]
v = self.v_proj(x).view(B, T, 4, 16).transpose(1, 2)  # [B, 4, T, 16]`,
    inspectInstructions:
      'Click token "He" in the bar below. Observe how its 64 residual numbers split into three distinct 16-number vectors for Head 0.',
  },
  {
    id: 'c4-b2-dot-product',
    beatNumber: 2,
    shortName: 'The Dot Product',
    title: 'The Dot Product: Scoring Relevance',
    subtitle:
      'Multiplying 16 coordinates pair-by-pair to quantify semantic alignment',
    sceneBeat: 'attn-2-dot-product',
    questionPrompt:
      'How does math turn two 16-dimensional vectors into a single number representing match quality?',
    prediction: {
      id: 'p-dot-product',
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
    summary: {
      foundations:
        'To see how much "He" cares about "Max", the model multiplies the 1st number of Q("He") with the 1st number of K("Max"), the 2nd with the 2nd, and so on through all 16 dimensions. It sums those 16 products together into a single total. If both vectors point in the same direction, the sum is high and positive. If they disagree, the sum is negative.',
      practitioner:
        'The raw affinity score between query position i and key position j is the Euclidean inner product: score_{i,j} = ∑_{d=0}^{15} q_{i,d} · k_{j,d}. In parallel, this is batch matrix multiplication: S = Q K^T ∈ ℝ^{T×T}.',
      formal:
        'The unscaled compatibility score S_{i,j} is a bilinear form over the projected vectors: S_{i,j} = \\langle q_i, k_j \\rangle = q_i^T k_j = x_i^T (W_q^T W_k) x_j. The matrix M = W_q^T W_k ∈ ℝ^{64×64} acts as a learned bilinear metric tensor.',
    },
    deepDive: {
      foundations:
        'Notice the 16 product bars in the center of the 3D scene. When "He" queries "Max", coordinate 3 produces a strong positive product because both are positive. Coordinate 7 produces another positive because both are negative (- × - = +!). Summing all 16 yields the total alignment score.',
      practitioner:
        'Because d_head = 16, computing all pairs for a 15-token sequence requires 15 × 15 × 16 = 3,600 multiply-accumulate operations per head. Across all 4 heads, that is 14,400 FLOPs per layer.',
      formal:
        'Because W_q and W_k are generally not identical, M is not symmetric (M ≠ M^T). Therefore, S_{i,j} ≠ S_{j,i}: the affinity of token i for token j is completely distinct from the affinity of token j for token i.',
    },
    equationLatex:
      'S_{i,j} = \\langle \\mathbf{q}_i, \\mathbf{k}_j \\rangle = \\sum_{c=1}^{16} q_{i,c} \\cdot k_{j,c}',
    codeBlock: `# Compute all dot products simultaneously:
scores = torch.matmul(q, k.transpose(-2, -1))  # [B, 4, T, T]`,
    inspectInstructions:
      'Compare Query "He" against Key "Max", then against Key "dog" and Key "ball". Look at the 16 products summing up.',
  },
  {
    id: 'c4-b3-scan-row',
    beatNumber: 3,
    shortName: 'Row of Affinities',
    title: 'Scanning the Sentence: The 1/sqrt(d) Scale',
    subtitle: 'Why dividing by 4 is the secret that keeps gradients alive',
    sceneBeat: 'attn-3-scan-row',
    questionPrompt:
      'If Query "He" computes dot products with all 15 past tokens, why must every score be divided by 4?',
    prediction: {
      id: 'p-scale-variance',
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
    summary: {
      foundations:
        'When you add 16 random multiplications together, the sum naturally swings from -20 to +20. If numbers get that large, softmax turns into an all-or-nothing switch: one token gets 99.99% of the attention and all others get 0%. Dividing by √16 = 4 keeps the scores gentle, allowing the model to consider several context words at once.',
      practitioner:
        'Without the 1/√d_head scale factor (here 1/√16 = 0.25), dot product variance scales linearly with d_head. Large logits push softmax into regions where gradients are virtually zero, causing vanishing gradients during backpropagation.',
      formal:
        'Under standard i.i.d. Gaussian assumptions q_{i,c}, k_{j,c} ∼ 𝒩(0, 1), 𝔼[q_i^T k_j] = 0 and Var(q_i^T k_j) = d_{head}. Dividing by √d_{head} normalizes the variance: Var((q_i^T k_j) / √d_{head}) = 1.0, preserving healthy gradient flow through the softmax Jacobian.',
    },
    deepDive: {
      foundations:
        'Imagine turning up the volume on a microphone until it squeals with feedback. That is what unscaled attention does to neural networks. Dividing by 4 acts like an automatic volume limiter that keeps every instrument audible.',
      practitioner:
        'In frontier models like Llama-3 (d_head = 128), 1/√d is 1/√128 ≈ 0.08839. In our model (d_head = 16), 1/√d = 0.25. Notice how multiplying raw scores by 0.25 compresses a raw score of 12.0 down to a manageable 3.0.',
      formal:
        'The softmax gradient satisfies ∂Softmax(z)_i / ∂z_j = S_i (δ_{ij} - S_j). When any single logit z_k dominates, S_k → 1 and S_{j≠k} → 0, driving the entire Jacobian matrix to 0. The 1/√d scale is precisely calibrated to prevent this saturation.',
    },
    equationLatex:
      '\\widetilde{S}_{i,j} = \\frac{\\mathbf{q}_i^T \\mathbf{k}_j}{\\sqrt{d_{head}}} = \\frac{1}{4} \\sum_{c=1}^{16} q_{i,c} k_{j,c}',
    codeBlock: `scale = 1.0 / math.sqrt(16)  # exactly 0.25
scaled_scores = scores * scale`,
    inspectInstructions:
      'Observe the row of 15 affinity bars for Query "He". Toggle the "Show Raw Scores" button to compare unscaled vs scaled numbers.',
  },
  {
    id: 'c4-b4-mask-softmax',
    beatNumber: 4,
    shortName: 'The Causal Mask & Softmax',
    title: 'The Arrow of Time: Causal Masking & Softmax',
    subtitle:
      'Blocking the future with -inf and converting affinities into exact percentages',
    sceneBeat: 'attn-4-mask-softmax',
    questionPrompt:
      'How does the model guarantee that token 5 cannot cheat by reading token 6 during generation?',
    prediction: {
      id: 'p-future-mask',
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
    summary: {
      foundations:
        'Language only moves forward in time. When reading token 5 ("Max"), the model is not allowed to peek at token 10 ("red"). To enforce this, any position in the future is replaced with negative infinity (-∞). Because e^{-∞} = 0, the future receives exactly 0.00% of the attention. Softmax then turns the remaining numbers into percentages that sum to exactly 100%.',
      practitioner:
        'An upper-triangular mask where j > i is filled with -1e4 (or -inf in fp32). Softmax is applied row-wise: probs_{i,j} = exp(S_{i,j}) / ∑_{k≤i} exp(S_{i,k}). In our engine, row-max subtraction is applied first for numerical stability.',
      formal:
        'Let M ∈ {0, -∞}^{T×T} be the causal mask matrix where M_{i,j} = 0 for j ≤ i and M_{i,j} = -∞ for j > i. The attention weight matrix A is defined as:\n\nA = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_{head}}} + M\\right)\n\nEvery row A_{i,:} lies on the probability simplex Δ^{T-1} with support restricted to {1..i}.',
    },
    deepDive: {
      foundations:
        'In the 3D scene, look at the dark slate tiles covering the upper-right triangle. Those are the masked future positions. Notice that row 0 ("Once") can only look at itself (100%), while row 14 ("He") can look across all 15 tokens.',
      practitioner:
        'In fp16, setting masked entries to -1e4 rather than -1e30 avoids float16 underflow/overflow bugs. In modern FlashAttention, the upper triangle is simply never loaded from memory, saving 50% of memory bandwidth.',
      formal:
        'Causal masking guarantees the autoregressive property: ∂x_t / ∂x_s = 0 for all s > t. This ensures the model factorizes the joint probability as a product of conditionals: P(x_1..x_T) = ∏_{t=1}^T P(x_t | x_{<t}).',
    },
    equationLatex:
      'A_{i,j} = \\frac{\\exp(\\widetilde{S}_{i,j} + M_{i,j})}{\\sum_{k=1}^T \\exp(\\widetilde{S}_{i,k} + M_{i,k})}, \\quad M_{i,j} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}',
    codeBlock: `mask = torch.tril(torch.ones(T, T)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
probs = torch.softmax(scores, dim=-1)  # each row sums to 1.0`,
    inspectInstructions:
      'Look at the 3D attention matrix. Notice the dark triangular wall of time. Click different tokens to see their 100% attention distribution.',
  },
  {
    id: 'c4-b5-value-mixer',
    beatNumber: 5,
    shortName: 'Value Mixing',
    title: 'Value Mixing: Blending Context Vectors',
    subtitle:
      'Using attention percentages to compute a weighted sum of Value vectors',
    sceneBeat: 'attn-5-value-mixer',
    questionPrompt:
      'Once token "He" knows it wants 62% Max and 25% dog, how does it actually absorb their information?',
    prediction: {
      id: 'p-value-blending',
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
    summary: {
      foundations:
        'Query and Key only decided the percentages. Value carries the actual substance! If token "He" assigns 62% attention to "Max" and 25% to "dog", the model blends 62% of Max’s Value vector and 25% of dog’s Value vector together. The result is a brand-new 16-number vector for "He" that is packed with knowledge about Max the dog.',
      practitioner:
        'The head output is a matrix multiplication between attention probabilities and values: Out_h = A_h V_h ∈ ℝ^{T×16}. For token i, out_{i,h} = ∑_{j≤i} A_{i,j,h} v_{j,h}.',
      formal:
        'The attention operation is a convex combination of value vectors in ℝ^{d_{head}}: y_i = ∑_{j=1}^i A_{i,j} v_j. Because A_{i,:} ∈ Δ^{i-1}, y_i lies in the convex hull of {v_1..v_i}: y_i ∈ \\text{conv}(v_1, \\dots, v_i).',
    },
    deepDive: {
      foundations:
        'Think of the attention weights as a chef’s recipe: 62% Max flavor, 25% dog flavor, 10% ball flavor. The kitchen blends them into a single smoothie. When "He" moves to the next layer, it no longer acts like an empty pronoun—it carries the full concept of Max.',
      practitioner:
        'Value mixing is a dense GEMM: [T, T] × [T, 16] → [T, 16]. Notice that unlike recurrent networks (RNNs) where information must squeeze through many sequential steps, attention transports information from token 2 to token 14 in a single hop with zero path length.',
      formal:
        'The convex hull property bounds the spectral norm of the aggregation operator: ||A||_∞ = 1. Value mixing does not amplify activations; any dimensional growth or feature synthesis must be performed by the downstream MLP layer.',
    },
    equationLatex:
      '\\mathbf{y}_i = \\sum_{j=1}^i A_{i,j} \\mathbf{v}_j = [\\mathbf{A} \\mathbf{V}]_i \\quad \\in \\mathbb{R}^{16}',
    codeBlock: `# Compute context vector as weighted sum:
head_out = torch.matmul(probs, v)  # [B, 4, T, 16]`,
    inspectInstructions:
      'Watch the glowing transfer arcs carrying Value numbers from "Max" and "dog" into the new representation of "He".',
  },
  {
    id: 'c4-b6-multihead-full',
    beatNumber: 6,
    shortName: 'The 4-Head Orchestra',
    title: 'The 4-Head Orchestra & Output Projection',
    subtitle:
      'Combining 4 specialized heads and projecting back into the 64-d residual stream',
    sceneBeat: 'attn-6-multihead-full',
    questionPrompt:
      'Why have 4 separate 16-number heads instead of one single 64-number head?',
    prediction: {
      id: 'p-multihead-purpose',
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
    summary: {
      foundations:
        'A single attention head can only focus on one thing at a time. By splitting 64 dimensions into 4 heads of 16, the model runs 4 parallel search engines: Head 0 tracks who "He" refers to (Max), Head 1 tracks grammar and punctuation, Head 2 tracks objects (ball), and Head 3 anticipates what happens next. The 4 heads are glued together and mixed by the Output Matrix (W_o) to update the residual stream.',
      practitioner:
        'The outputs of the 4 heads [T, 4, 16] are concatenated along the head dimension into a single [T, 64] matrix, then projected by W_o ∈ ℝ^{64×64}. The result is added directly back into the input residual stream: x_{out} = x_{in} + (Concat(head_0..3) · W_o).',
      formal:
        '\\text{MultiHead}(Q, K, V) = \\left[ \\text{head}_0 \\mathbin{\\Vert} \\text{head}_1 \\mathbin{\\Vert} \\text{head}_2 \\mathbin{\\Vert} \\text{head}_3 \\right] W_o\n\nwhere each head_h = \\text{Attention}(x W_{q,h}, x W_{k,h}, x W_{v,h}) and W_o ∈ ℝ^{64×64}. The residual connection preserves identity routing while attention adds low-rank updates.',
    },
    deepDive: {
      foundations:
        'Try disabling Head 0 using the Ablation button in the top bar! Because Head 0 is the character-resolution head, disabling it immediately causes the model’s prediction for the next word to shift.',
      practitioner:
        'Total parameters in the attention block: W_q (4096), W_k (4096), W_v (4096), W_o (4096) = 16,384 fp16 parameters per block. Across 4 layers, attention accounts for 65,536 parameters (~7.6% of the 862,464 total model weights).',
      formal:
        'Multi-head attention can be interpreted as an ensemble of low-rank relational operators. While individual heads have rank at most 16, their concatenation and linear mixing via W_o spans a full-rank 64-dimensional subspace, providing rich cross-subspace feature interaction.',
    },
    equationLatex:
      '\\mathbf{x}_{\\text{out}} = \\mathbf{x}_{\\text{in}} + \\left[ \\text{head}_0 \\mathbin{\\Vert} \\dots \\mathbin{\\Vert} \\text{head}_3 \\right] \\mathbf{W}_o',
    codeBlock: `# Concatenate 4 heads and project:
out = head_out.transpose(1, 2).contiguous().view(B, T, 64)
attn_out = self.c_proj(out)  # [B, T, 64]
x = x + attn_out  # residual connection`,
    inspectInstructions:
      'Switch between Heads 0, 1, 2, and 3 using the top HUD buttons. Click "Disable H0" to ablate Head 0 and see the prediction change.',
  },
];

export const TEXTBOOK_CHAPTERS: ChapterMeta[] = [
  {
    id: 1,
    title: 'From Words to Numbers',
    subtitle: 'Byte-Pair Encoding, Token Embeddings & Positional Vectors',
    readingTimeMinutes: 12,
    status: 'coming-soon',
    teaser:
      'How does a computer that only does arithmetic understand what a "dog" or a "red ball" is?',
    beats: [],
  },
  {
    id: 2,
    title: 'The Information Highway',
    subtitle: 'The 64-Dimensional Residual Stream & LayerNorm Gates',
    readingTimeMinutes: 15,
    status: 'coming-soon',
    teaser:
      'How does a deep stack of layers keep adding new ideas without erasing earlier memory or letting numbers explode?',
    beats: [],
  },
  {
    id: 3,
    title: 'Preparing for the Dance',
    subtitle: 'Layer Normalization, Variance Calibration & Clean Inputs',
    readingTimeMinutes: 14,
    status: 'coming-soon',
    teaser:
      'Why does every single layer force vectors onto a hypersphere before letting them interact?',
    beats: [],
  },
  {
    id: 4,
    title: 'How Words Talk: Self-Attention',
    subtitle:
      'Q/K/V, Scaled Dot Products, Causal Mask, Softmax, Value Mixing & 4 Heads',
    readingTimeMinutes: 25,
    status: 'published',
    teaser:
      'How does the word "He" figure out it refers to a little dog named Max who found a red ball?',
    beats: CHAPTER_4_BEATS,
  },
  {
    id: 5,
    title: 'Private Thinking & Memory',
    subtitle: 'Feed-Forward Network (MLP), 4× Expansion, GELU, SwiGLU & MoE',
    readingTimeMinutes: 20,
    status: 'coming-soon',
    teaser:
      'If Attention only averages existing vectors, where does the model think privately and store learned facts?',
    beats: [],
  },
  {
    id: 6,
    title: 'Sharpening a Guess & Writing a Story',
    subtitle:
      'Logit Lens, Unembedding, Temperature/Top-k/Top-p, Autoregression & KV Cache',
    readingTimeMinutes: 18,
    status: 'coming-soon',
    teaser:
      'If the math is 100% deterministic, why does the exact same prompt give different answers?',
    beats: [],
  },
  {
    id: 7,
    title: 'Where Every Number Came From',
    subtitle:
      'Cross-Entropy Loss, Manual Backprop, AdamW, Training Replay & Hallucination',
    readingTimeMinutes: 22,
    status: 'coming-soon',
    teaser:
      'Nobody programmed grammar into this AI—how did 862,464 random numbers organize themselves into English?',
    beats: [],
  },
  {
    id: 8,
    title: 'From Miniature Toy to Frontier AI',
    subtitle:
      'Surgical Ablation, Scaling Laws, Modern Upgrades & Post-Training (SFT / RLHF)',
    readingTimeMinutes: 30,
    status: 'coming-soon',
    teaser:
      'What stays identical and what fundamentally changes across a 1,000,000× jump to trillion-parameter LLMs?',
    beats: [],
  },
];

export function editionLabel(e: Edition): string {
  if (e === 'foundations') return 'Foundations';
  if (e === 'practitioner') return 'Practitioner';
  return 'Formal';
}
