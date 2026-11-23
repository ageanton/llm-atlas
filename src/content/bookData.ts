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
  shortTitle: string;
  heading: string;
  sceneBeat: SceneBeatId;
  visualFocusNote: string;
  paragraphs: Record<Edition, string[]>;
  keyTakeaway: Record<Edition, string>;
  predict?: PredictChallenge;
}

export interface ChapterMeta {
  id: number;
  title: string;
  subtitle: string;
  status: 'ready' | 'upcoming';
  openingMystery: string;
  beats: BookBeat[];
}

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

export const CHAPTER_4_BEATS: BookBeat[] = [
  {
    id: 'c4-b1-problem',
    beatNumber: 1,
    shortTitle: '1. The Bottleneck',
    heading: 'The Addressing Bottleneck & The Three Roles (Q, K, V)',
    sceneBeat: 'attn-1-problem',
    visualFocusNote:
      'Scene shows the query token projecting into three distinct 16-dimensional vectors for Head 3: Query, Key, and Value.',
    paragraphs: {
      foundations: [
        'Before tokens can exchange information, each token sits isolated inside its own 64-dimensional vector. At the prompt end, the pronoun "He" needs to know what noun it refers to. But a single vector cannot simultaneously declare what it is searching for, advertise its own attributes to future tokens, and supply factual content.',
        'Attention solves this by giving every token three distinct projection heads per attention block: Query ("what kind of information am I looking for?"), Key ("what attributes do I advertise to other words?"), and Value ("what content will I transmit if selected?").',
        'Look at the three projected strips below. Even though they all originate from the same token embedding, the learned weight matrices project them into completely different geometric roles.',
      ],
      practitioner: [
        'Input activations x ∈ ℝ^{T × 64} are multiplied by W_q, W_k, W_v ∈ ℝ^{64 × 64} to produce the full multi-head Q, K, V tensors. For Head h, the projection produces slices q_h, k_h, v_h ∈ ℝ^{T × 16}.',
        'Decoupling Query and Key projections is essential: without distinct W_q and W_k, the similarity matrix Q K^T would be symmetric, making it mathematically impossible for token i to attend strongly to token j without token j attending equally back to token i.',
        'In the 3D studio, the gold, emerald, and amber columns show the 16 real scalar activations for the active Query token under the verified miniature weights.',
      ],
      formal: [
        'Let x_t ∈ ℝ^{d_{model}} denote the residual stream activation at position t, with d_{model} = 64. For each head h ∈ {0..3}, linear transformations W_{q,h}, W_{k,h}, W_{v,h} ∈ ℝ^{d_{head} × d_{model}} (with d_{head} = 16) define affine embeddings into distinct metric subspaces.',
        'The query vector q_{t,h} = W_{q,h} x_t and key vector k_{t,h} = W_{k,h} x_t span an asymmetric bilinear retrieval manifold. Information transmission is governed strictly by the value manifold v_{t,h} = W_{v,h} x_t.',
        'Because rank(W_{q,h}^T W_{k,h}) ≤ d_{head} = 16 ≪ d_{model}, each head isolates an independent 16-dimensional semantic query subspace, bypassing interference in the remaining 48 dimensions.',
      ],
    },
    keyTakeaway: {
      foundations:
        'A token cannot use the same vector to ask questions, answer questions, and deliver payload. Q searches, K matches, V delivers.',
      practitioner:
        'Asymmetric Q and K projections allow directed, non-symmetric information flow between past and future positions.',
      formal:
        'Decoupled linear maps W_q and W_k induce a non-symmetric bilinear form q^T k = x_i^T (W_q^T W_k) x_j over the residual stream.',
    },
    predict: {
      id: 'predict-c4-b1',
      question:
        'What would happen if a model used the EXACT same projection weights for Query and Key (W_q = W_k)?',
      options: [
        {
          id: 'opt-a',
          label: 'The similarity matrix would be symmetric, forcing A(i→j) = A(j→i).',
        },
        {
          id: 'opt-b',
          label: 'Attention weights would instantly collapse to NaN.',
        },
        {
          id: 'opt-c',
          label: 'The model could no longer generate words autoregressively.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'If W_q = W_k, then q_i · k_j = x_i^T W^T W x_j = (x_j^T W^T W x_i)^T = q_j · k_i. Token "He" could never attend to "Max" without "Max" attending equally back to "He".',
    },
  },
  {
    id: 'c4-b2-dot-product',
    beatNumber: 2,
    shortTitle: '2. One Comparison',
    heading: 'The Dot Product: Testing Alignment Across 16 Dimensions',
    sceneBeat: 'attn-2-dot-product',
    visualFocusNote:
      'Scene isolates the 16 coordinate-wise products q_c · k_c between Query ("He") and the selected Key token.',
    paragraphs: {
      foundations: [
        'How does the model decide whether "He" should pay attention to "Max" versus "ball"? It takes the 16 numbers of Q("He") and multiplies each coordinate with the corresponding number of K("Max"), then sums all 16 products together.',
        'When two coordinates have the same sign (both positive or both negative), their product is positive—a vote for relevance. When coordinates have opposite signs, their product is negative—a vote against relevance.',
        'In the 3D scene below, notice the coordinate beams. Positive products rise in blue; negative products drop in amber. The sum of these 16 products is the raw dot product score.',
      ],
      practitioner: [
        'The unscaled compatibility score is S_{i,j} = ∑_{c=0}^{15} q_{i,c} k_{j,c}. Across a sequence of length T, this is computed as a batched matrix multiplication Q K^T ∈ ℝ^{T × T}.',
        'Click on different key tokens in the bottom bar ("Max", "dog", "ball", "."). Notice how Head 3 produces a strong positive dot product for animate masculine antecedents ("Max", "dog") and heavily suppresses punctuation or future positions.',
        'The breakdown panel displays the exact arithmetic: raw dot product, scale divisor √16 = 4, and the resulting scaled score.',
      ],
      formal: [
        'Let q_i, k_j ∈ ℝ^{d_{head}}. The inner product ⟨q_i, k_j⟩ = q_i^T k_j represents the directional cosine similarity scaled by vector magnitudes ||q_i||_2 ||k_j||_2.',
        'In our architecture, d_{head} = 16. Under independent identically distributed components with zero mean and unit variance, 𝔼[q_i^T k_j] = 0 and Var(q_i^T k_j) = d_{head} = 16.',
        'Without rescaling, the variance of the logits scales linearly with dimension, pushing values to extreme magnitudes where softmax saturates.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Dot products add up matching coordinates. If coordinates agree in sign, the score shoots up; if they disagree, the score plummets.',
      practitioner:
        'High-dimensional dot products act as soft feature detectors, matching search queries against advertised keys.',
      formal:
        'The variance of the unnormalized dot product scales directly with d_{head}, requiring normalization prior to exponentiation.',
    },
    predict: {
      id: 'predict-c4-b2',
      question:
        'If two 16-dimensional vectors have opposite signs on 14 out of 16 coordinates, what will the dot product most likely be?',
      options: [
        {
          id: 'opt-a',
          label: 'Negative, because negative products dominate the sum.',
        },
        {
          id: 'opt-b',
          label: 'Zero, because opposite signs cancel out.',
        },
        {
          id: 'opt-c',
          label: 'Positive, because squaring occurs in vector multiplication.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'The dot product is a linear sum of products. Each coordinate pair with opposite signs contributes a negative number (+ × - = -), driving the overall score strongly negative.',
    },
  },
  {
    id: 'c4-b3-scan-row',
    beatNumber: 3,
    shortTitle: '3. Row Scan & 1/√d',
    heading: 'Scanning the Entire Past & The 1/√d Scale Factor',
    sceneBeat: 'attn-3-scan-row',
    visualFocusNote:
      'Scene expands from a single pair to the entire horizontal row of dot products that Query "He" casts across all tokens.',
    paragraphs: {
      foundations: [
        'Query "He" does not look at only one token—it scans every single word that came before it simultaneously, producing a full row of scores.',
        'Notice the scale factor 1/√16 = 1/4 = 0.25. Why must we divide every score by 4? Because adding 16 multiplications causes numbers to grow 4 times wider in variance. If numbers get too large (+12 or -15), softmax becomes an all-or-nothing switch that ignores all nuance.',
        'Dividing by √d keeps scores in a gentle range where multiple tokens can share attention weights proportionately.',
      ],
      practitioner: [
        'In PyTorch: `scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)`. In our verified miniature transformer, d_k = 16, so the divisor is exactly 4.0.',
        'Toggle the "Raw Scores" HUD button. Without 1/√d, scores swing between -8 and +14. After dividing by 4, they stay neatly between -2.0 and +3.5, keeping softmax gradients healthy during training.',
        'The 3D stage renders the complete horizontal row of scores cast by the active Query token across all sequence positions.',
      ],
      formal: [
        'Let z_j = (q_i^T k_j) / √{d_{head}}. For i.i.d. components q_{i,c}, k_{j,c} ∼ 𝒩(0, 1), the scaled logit satisfies Var(z_j) = 1.',
        'Preserving unit variance prevents exponential divergence in the softmax denominator ∑_k exp(z_k), guaranteeing non-vanishing gradients: ∂(softmax)/∂z_j = s_j (δ_{jk} - s_k).',
        'When Var(z) ≫ 1, the maximum logit dominates, forcing s_{max} → 1 and all other probabilities to 0, which collapses the gradient tensor to 0.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Dividing by √16 = 4 prevents extreme numbers from blowing up the softmax exponentiation.',
      practitioner:
        '1/√d_k standardizes logit variance to 1.0, protecting backpropagation from vanishing gradients.',
      formal:
        'Rescaling stabilizes the spectral radius of the attention Jacobian matrix around unit magnitude.',
    },
    predict: {
      id: 'predict-c4-b3',
      question:
        'If a frontier LLM uses attention heads of dimension d_head = 128, what is its exact scale factor?',
      options: [
        {
          id: 'opt-a',
          label: '1 / √128 ≈ 0.0884',
        },
        {
          id: 'opt-b',
          label: '1 / 128 = 0.0078',
        },
        {
          id: 'opt-c',
          label: '1 / √64 = 0.125',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'The formula is always 1 / √{d_{head}}. For d_head = 128, √128 ≈ 11.3137, giving a scale factor of approximately 0.0884.',
    },
  },
  {
    id: 'c4-b4-mask-softmax',
    beatNumber: 4,
    shortTitle: '4. Mask & Softmax',
    heading: 'The Causal Mask & Softmax Normalization (100% Budget)',
    sceneBeat: 'attn-4-mask-softmax',
    visualFocusNote:
      'Scene visualizes the full N×N attention matrix with the upper triangular future masked to -∞ (slate tiles) and lower rows normalized to 100%.',
    paragraphs: {
      foundations: [
        'Time in language only moves forward. When computing representations for token 3 ("a little dog"), the model is not allowed to peek at token 10 ("red ball").',
        'To enforce this physical law, the upper-right half of the attention matrix is covered by a causal mask: any connection to a future token is forced to -∞. Because exp(-∞) = 0, future positions receive strictly 0% attention.',
        'Softmax then exponentiates the remaining past scores and divides each by their sum. Every row becomes an exact probability distribution that sums to 1.00 (100%).',
      ],
      practitioner: [
        'The causal mask is a lower-triangular matrix: M_{i,j} = 0 for j ≤ i, and -10000.0 (or -inf) for j > i. In our verified FP16 engine, -10000.0 is used to avoid NaN underflow.',
        'In the 3D grid below, dark slate tiles represent masked future tokens. Notice how each token i can only look at positions 0 through i.',
        'The glowing blue cells show the normalized attention weights A_{i,j}. Notice how row 14 ("He") distributes its 100% budget predominantly across "Max" and "dog".',
      ],
      formal: [
        'Let M ∈ ℝ^{T × T} with M_{i,j} = 0 if j ≤ i and M_{i,j} = -∞ if j > i. The attention matrix A is defined as A = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_{head}}} + M \\right).',
        'By construction, A_{i,j} = 0 for all j > i, preserving the autoregressive factorization P(x) = ∏_{t=1}^T P(x_t | x_{<t}).',
        'Row stochasticity is strictly preserved: A 1_T = 1_T and A_{i,j} ≥ 0, meaning A acts as a Markov transition kernel over the sequence graph.',
      ],
    },
    keyTakeaway: {
      foundations:
        'The causal mask blocks the future with -∞ so the AI cannot cheat. Softmax turns raw scores into exact percentages summing to 100%.',
      practitioner:
        'Causal masking converts attention into a lower-triangular row-stochastic matrix where each token distributes a unit probability budget.',
      formal:
        'A ∈ ℝ^{T × T} belongs to the directed acyclic graph convolution manifold, enforcing non-anticipative causality in the model.',
    },
    predict: {
      id: 'predict-c4-b4',
      question:
        'What would happen during autoregressive text generation if the causal mask were omitted during training?',
      options: [
        {
          id: 'opt-a',
          label: 'The model would learn to look ahead to cheat during training, and collapse at test time.',
        },
        {
          id: 'opt-b',
          label: 'Generation speed would double due to bidirectional parallelism.',
        },
        {
          id: 'opt-c',
          label: 'Nothing; position embeddings already prevent looking into the future.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'Without a causal mask, token t can simply attend to token t+1 to copy the answer. At inference time, token t+1 does not exist yet, so the model hallucinates or outputs gibberish.',
    },
  },
  {
    id: 'c4-b5-value-mixer',
    beatNumber: 5,
    shortTitle: '5. Value Mixing',
    heading: 'Value Mixing: Constructing the New Context Vector',
    sceneBeat: 'attn-5-value-mixer',
    visualFocusNote:
      'Scene shows the attention weights pulling proportional slices of past Value vectors into a single new 16-dimensional representation.',
    paragraphs: {
      foundations: [
        'Now comes the payoff. Query and Key were just the matchmaking system—they determined the percentages. Value is the actual cargo being delivered.',
        'If "He" assigned 62% attention to "Max", 25% to "dog", and 13% to "ball", the model computes a weighted average: 0.62 × V("Max") + 0.25 × V("dog") + 0.13 × V("ball").',
        'The result is a brand-new 16-dimensional vector for "He". Before attention, "He" was just a generic pronoun. After attention, its vector is enriched with the actual traits of Max the dog.',
      ],
      practitioner: [
        'Context representation for head h: HeadOut_h = A_h V_h ∈ ℝ^{T × 16}. For token i, the output is out_{i,h} = ∑_{j=0}^i A_{i,j,h} v_{j,h}.',
        'Because ∑_j A_{i,j} = 1, the output vector lies strictly inside the convex hull of the input Value vectors.',
        'Observe the glowing streams in the 3D scene converging from past tokens into the output node for "He".',
      ],
      formal: [
        'The head context mapping is a non-local linear aggregation operator: y_{i,h} = ∑_{j=1}^i A_{i,j,h} v_{j,h} = [A_h V_h]_i.',
        'Because A_{i,:} ∈ Δ^{T-1} (the probability simplex), y_{i,h} is a convex combination of {v_{1,h}, ..., v_{i,h}}.',
        'This proves that self-attention is a contractive operator on the value manifold under the convex hull metric: ||y_i||_2 ≤ \\max_j ||v_j||_2.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Attention percentages are a recipe: the model takes a weighted smoothie of past Value vectors to enrich the current token.',
      practitioner:
        'The attention output is a batched GEMM: HeadOut = A · V. It routes factual features across arbitrary distances in a single hop.',
      formal:
        'Self-attention computes a data-dependent convex combination on the value manifold, bounding output norms within the convex hull.',
    },
    predict: {
      id: 'predict-c4-b5',
      question:
        'If a token assigns 100% of its attention weight to a single past word, what is its resulting head output vector?',
      options: [
        {
          id: 'opt-a',
          label: 'An exact copy of that past word’s Value vector.',
        },
        {
          id: 'opt-b',
          label: 'An average of the past word and the current word.',
        },
        {
          id: 'opt-c',
          label: 'Zero, because attention requires multiple words to blend.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'Because the output is ∑_j A_{i,j} v_j, if A_{i,k} = 1.0 and all other weights are 0, the sum reduces to 1.0 × v_k = v_k exactly.',
    },
  },
  {
    id: 'c4-b6-multihead-full',
    beatNumber: 6,
    shortTitle: '6. 4-Head Orchestra',
    heading: 'The 4-Head Orchestra & Output Projection (W_o)',
    sceneBeat: 'attn-6-multihead-full',
    visualFocusNote:
      'Scene pulls back to show all 4 attention heads operating in parallel, concatenating into 64 dimensions, and mixing via W_o with surgical ablation.',
    paragraphs: {
      foundations: [
        'A sentence has many things happening at once: grammar, meaning, gender, and rhythm. A single attention head cannot focus on all of them simultaneously.',
        'That is why our model uses 4 separate heads running in parallel. Head 0 tracks gender and subjects ("He" → "Max"); Head 1 tracks recent punctuation; Head 2 tracks verbs; Head 3 tracks narrative themes.',
        'All 4 heads (4 × 16 = 64 numbers) are concatenated side-by-side, multiplied by the Output Matrix W_o, and added back into the residual stream via the skip connection.',
      ],
      practitioner: [
        'MultiHead output: Concat(Head_0, Head_1, Head_2, Head_3) ∈ ℝ^{T × 64}. Multiplying by W_o ∈ ℝ^{64 × 64} projects the concatenated features back into residual coordinates.',
        'Residual addition: x_{next} = x + MultiHead(x) W_o. The residual connection ensures identity gradient flow across deep transformer stacks.',
        'Click the "Ablate Head" button in the HUD. Notice how surgically disabling Head 3 shifts the next-token prediction immediately, proving that specific heads specialize in distinct roles.',
      ],
      formal: [
        'The complete multi-head attention map is defined as \\text{MHA}(x) = [\\text{head}_0 \\Vert \\text{head}_1 \\Vert \\text{head}_2 \\Vert \\text{head}_3] W_o where \\text{head}_h = A_h (x W_{v,h}).',
        'While each head possesses rank at most d_{head} = 16, their concatenation spans a full rank-64 subspace: rank(MHA(x)) ≤ \\min(64, 4 × 16) = 64.',
        'W_o ∈ ℝ^{64 × 64} provides full cross-head linear mixing before adding the result back to the residual stream x^{(l)} = x^{(l-1)} + \\text{MHA}(\\text{LN}(x^{(l-1)})).',
      ],
    },
    keyTakeaway: {
      foundations:
        'Multiple heads divide and conquer: each head becomes a specialist, and W_o fuses their findings back into the main residual stream.',
      practitioner:
        'Multi-head attention combines multiple low-rank projections into a full-rank update, fused by W_o and stabilized by residual addition.',
      formal:
        'Multi-head attention constructs an ensemble of rank-16 relational operators whose direct sum spans the full 64-dimensional residual space.',
    },
    predict: {
      id: 'predict-c4-b6',
      question:
        'Why not replace the 4 heads of size 16 with a single giant head of size 64?',
      options: [
        {
          id: 'opt-a',
          label: 'A single head can only produce ONE attention pattern per token, preventing simultaneous multi-role tracking.',
        },
        {
          id: 'opt-b',
          label: 'A 64-dimensional head requires 4× more FLOPs to compute.',
        },
        {
          id: 'opt-c',
          label: 'Softmax cannot normalize vectors with dimension greater than 32.',
        },
      ],
      correctId: 'opt-a',
      explanation:
        'A single head only produces one set of weights per token. Token "He" would have to choose between attending to its gender antecedent ("Max") OR its syntactic context, unable to do both at once.',
    },
  },
];

export const TEXTBOOK_CHAPTERS: ChapterMeta[] = [
  {
    id: 1,
    title: 'From Characters to Thought Vectors',
    subtitle: 'Byte-Pair Encoding, Token Embeddings, and the Geometry of Words',
    status: 'upcoming',
    openingMystery:
      'How does a computer that only adds and multiplies numbers know that "dog" and "puppy" are practically siblings, while "dog" and "fog" are total strangers?',
    beats: [],
  },
  {
    id: 2,
    title: 'The Residual Stream Highway',
    subtitle: 'Layer Stacking, Residual Streams, and Additive Memory Accumulation',
    status: 'upcoming',
    openingMystery:
      'How do 4 deep transformer layers keep refining an idea across time without earlier memories getting overwritten or erased?',
    beats: [],
  },
  {
    id: 3,
    title: 'Keeping Numbers Sane: LayerNorm',
    subtitle: 'RMSNorm, Variance Explosion, and Normalized Activation Spheres',
    status: 'upcoming',
    openingMystery:
      'Why do neural networks go completely berserk and output NaNs if you do not force every vector onto a standardized sphere before every layer?',
    beats: [],
  },
  {
    id: 4,
    title: 'How Words Talk: Self-Attention',
    subtitle: 'Q, K, V Projections, Scaled Dot Products, Causal Masking, and the 4-Head Orchestra',
    status: 'ready',
    openingMystery:
      'How does the tiny word "He" at the end of a sentence reach back across 14 tokens to figure out it is referring to Max the dog?',
    beats: CHAPTER_4_BEATS,
  },
  {
    id: 5,
    title: 'Private Thinking: The MLP Engine',
    subtitle: '4× Dimension Expansion, GELU Activation, and Fact Retrieval Storage',
    status: 'upcoming',
    openingMystery:
      'Attention only routes information between tokens. Where does the model actually store its facts and think quietly to itself?',
    beats: [],
  },
  {
    id: 6,
    title: 'Turning Math Back Into English',
    subtitle: 'Unembedding, Logit Lens, Softmax Temperature, and Autoregressive Sampling',
    status: 'upcoming',
    openingMystery:
      'If the final vector is just 64 numbers, how does the model turn it into an exact probability distribution over all 2,048 English vocabulary tokens?',
    beats: [],
  },
  {
    id: 7,
    title: 'Where the Weights Came From',
    subtitle: 'Cross-Entropy Loss, Backpropagation, and the AdamW Optimization Journey',
    status: 'upcoming',
    openingMystery:
      'Nobody programmed grammar or story plots into this file. How did 862,464 completely random numbers teach themselves English?',
    beats: [],
  },
  {
    id: 8,
    title: 'From Miniature Toy to Frontier AI',
    subtitle: 'Scaling Laws, Architectural Evolution, and the Leap from 862K to 1 Trillion Parameters',
    status: 'upcoming',
    openingMystery:
      'What stays identical, and what changes when you scale an 862,464-parameter toy up to a 1,000,000× larger frontier model?',
    beats: [],
  },
];

export function editionLabel(edition: Edition): string {
  switch (edition) {
    case 'foundations':
      return 'Foundations (Intuition)';
    case 'practitioner':
      return 'Practitioner (Code & Tensors)';
    case 'formal':
      return 'Formal (Mathematical Proof)';
  }
}
