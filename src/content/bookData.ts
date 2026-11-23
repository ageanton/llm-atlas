import {
  CHAPTER_1_BEATS,
  CHAPTER_2_BEATS,
  CHAPTER_3_BEATS,
} from './chapters123';
import {
  CHAPTER_5_BEATS,
  CHAPTER_6_BEATS,
  CHAPTER_7_BEATS,
  CHAPTER_8_BEATS,
} from './chapters5678';
import type { Edition } from '../state/bookState';

export type SceneBeatId =
  | 'ch1-1-extremes'
  | 'ch1-2-bpe-merges'
  | 'ch1-3-spaces-blindspot'
  | 'ch1-4-token-ids'
  | 'ch1-5-context-window'
  | 'ch2-1-wte-lookup'
  | 'ch2-2-pca-cloud'
  | 'ch2-3-order-blind'
  | 'ch2-4-wpe-add'
  | 'ch2-5-rope'
  | 'ch3-1-residual-bus'
  | 'ch3-2-cut-highway'
  | 'ch3-3-layernorm'
  | 'ch3-4-rmsnorm'
  | 'attn-1-problem'
  | 'attn-2-dot-product'
  | 'attn-3-scan-row'
  | 'attn-4-mask-softmax'
  | 'attn-5-value-mixer'
  | 'attn-6-multihead-full'
  | 'ch5-1-mlp-role'
  | 'ch5-2-expansion'
  | 'ch5-3-gelu'
  | 'ch5-4-wdown'
  | 'ch5-5-swiglu-moe'
  | 'ch6-1-logit-lens'
  | 'ch6-2-unembed'
  | 'ch6-3-temperature'
  | 'ch6-4-nucleus'
  | 'ch6-5-autoregression'
  | 'ch6-6-kv-quant'
  | 'ch7-1-step0'
  | 'ch7-2-cross-entropy'
  | 'ch7-3-manual-backprop'
  | 'ch7-4-training-replay'
  | 'ch7-5-hallucination'
  | 'ch8-1-ablation'
  | 'ch8-2-scale'
  | 'ch8-3-frontier-diff'
  | 'ch8-4-post-training'
  | 'ch8-5-capstone';

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
    id: 'attn-beat-1',
    beatNumber: 1,
    shortTitle: '1. The Problem',
    heading: 'Why Words Cannot Think Alone',
    sceneBeat: 'attn-1-problem',
    visualFocusNote:
      'Only your sentence is on stage. Notice the highlighted Query token sitting in isolation with no arrows connecting it to earlier context.',
    paragraphs: {
      foundations: [
        'Look at the final token highlighted in blue on the 3D stage. Right now, the model only has that token’s generic dictionary entry plus its slot number in the sentence.',
        'If the token is "He", its dictionary entry has no idea whether "He" is a dog named Max, a pirate captain, or a toaster. Every other part of a Transformer only looks at one word at a time. To predict what happens next in the story, "He" needs a way to reach backward and pull information from earlier words.',
        'That single job—letting each token look back at earlier tokens and borrow their meaning—is what Self-Attention was invented to solve.',
      ],
      practitioner: [
        'Entering Block l, every token t holds a residual vector h_t of shape [128] that initially only encodes its own token embedding plus positional offset (and any updates from earlier layers).',
        'LayerNorm and the Feed-Forward MLP operate strictly row-wise on each token in isolation. Multi-Head Self-Attention is the only operation in the entire Transformer architecture where information can cross between different sequence positions t_1 and t_2.',
        'Before we look at any [T, T] matrix, click Next Beat to watch the model compare just ONE query token against ONE earlier key token.',
      ],
      formal: [
        'Let H ∈ R^{T × d_model} denote the normalized residual stream entering layer l. Position-wise maps (LN, MLP, unembedding) act identically across rows t ∈ {0, ..., T-1}, leaving tokens statistically decoupled.',
        'Self-Attention introduces a data-dependent mixing operator over the sequence axis, computing a dynamic weighted graph whose edge strengths depend on the content and position of the input tokens themselves rather than fixed convolutional offsets.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Self-Attention is the only meeting room in an LLM where words can talk to earlier words.',
      practitioner:
        'Cross-token communication happens exclusively inside Self-Attention; all other layers are strictly per-token.',
      formal:
        'Self-Attention supplies the sequence-axis coupling required for contextual representations H_t = f(x_0, ..., x_t).',
    },
    predict: {
      id: 'ch4-b1-predict',
      question:
        'To predict the word right after "He" in Max’s story, why isn’t the dictionary embedding of "He" enough by itself?',
      options: [
        {
          id: 'a',
          label: 'Because "He" has not yet been linked to earlier context like "dog", "Max", and "ball".',
        },
        {
          id: 'b',
          label: 'Because the embedding table does not store pronouns.',
        },
        {
          id: 'c',
          label: 'Because pronouns are always masked out.',
        },
      ],
      correctId: 'a',
      explanation:
        'Exactly. The embedding of "He" is identical in every story until Attention pulls clues from earlier tokens.',
    },
  },
  {
    id: 'attn-beat-2',
    beatNumber: 2,
    shortTitle: '2. One Comparison (Q · K)',
    heading: 'One Single Pair: How Query Meets Key',
    sceneBeat: 'attn-2-dot-product',
    visualFocusNote:
      'The matrix is still hidden. On stage are ONLY two real 32-number strips from your model: the Query vector of one token, the Key vector of one earlier token, and the 32 element-wise products between them.',
    paragraphs: {
      foundations: [
        'Instead of comparing every word at once, let’s zoom all the way into ONE pair of words. Before comparing, the model turns each word into two 32-number lists: a Query ("what kind of clue am I looking for?") and a Key ("what kind of name-tag do I advertise?").',
        'Look at the two 32-bar strips facing each other on the 3D stage. To check if they match, the model multiplies bar #0 by bar #0, bar #1 by bar #1, all the way to bar #31. Wherever both bars point the same way (both amber or both blue), the product bar in the middle shoots upward in warm amber (+). Wherever they disagree, it drops in cool blue (-).',
        'Add all 32 product bars together and divide by √32 (about 5.66) to keep the scale gentle. That entire calculation produces ONE single real number: the match score between these two words. Try clicking different Key tokens on the controls below to watch the real 32-bar match change instantly.',
      ],
      practitioner: [
        'Inside each head (d_head = 32), the normalized input [T, 128] is projected by learned matrices W_q and W_k into Q_h [T, 32] and K_h [T, 32].',
        'The compatibility between query position i and key position j is simply the scaled dot product: dot(Q[h, i], K[h, j]) / sqrt(32). Separating W_q and W_k makes matching asymmetric—so "He" can search for nouns while "Max" advertises itself as a subject without having to match itself equally in reverse.',
        'Why divide by sqrt(32)? Summing 32 random products increases the variance by 32×. Without 1/sqrt(d_head), raw scores grow huge, forcing Softmax into saturated 0/1 cliffs where training gradients die.',
      ],
      formal: [
        'For head h ∈ {0..3}, let q_i = (LN_1(h_i) W_{Q,h}^T) ∈ R^{32} and k_j = (LN_1(h_j) W_{K,h}^T) ∈ R^{32}. The raw pre-softmax attention logit is the bilinear form S_{h,i,j} = (q_i^T k_j) / √d_k = h_i^T (W_{Q,h}^T W_{K,h} / √32) h_j.',
        'Under zero-mean, unit-variance independent coordinates, E[q_i^T k_j] = 0 and Var(q_i^T k_j) = d_k. Dividing by √d_k restores unit variance Var(S_{h,i,j}) = 1, keeping the downstream Softmax Jacobian well-conditioned.',
      ],
    },
    keyTakeaway: {
      foundations:
        'A dot product is just 32 pair-by-pair alignment votes added together into one match score.',
      practitioner:
        'S[i, j] = dot(Q[i], K[j]) / sqrt(32) measures asymmetric query-to-key alignment with unit-variance scaling.',
      formal:
        'Scaled bilinear QK circuits compute S_{i,j} = q_i^T k_j / √d_k to prevent Softmax Jacobian saturation.',
    },
    predict: {
      id: 'ch4-b2-predict',
      question:
        'What happens in one dimension d when BOTH q_i[d] and k_j[d] are negative numbers (for example, -0.8 and -0.5)?',
      options: [
        {
          id: 'a',
          label: 'They multiply to a positive vote (+0.40), increasing the match score because both vectors agreed on that feature.',
        },
        {
          id: 'b',
          label: 'They multiply to a negative penalty (-0.40), lowering the match score.',
        },
        {
          id: 'c',
          label: 'Negative values are ignored before the dot product.',
        },
      ],
      correctId: 'a',
      explanation:
        'Correct! Negative × negative = positive. A dot product rewards alignment in either direction, not just positive numbers.',
    },
  },
  {
    id: 'attn-beat-3',
    beatNumber: 3,
    shortTitle: '3. Scanning One Row',
    heading: 'One Word Scans Every Earlier Word',
    sceneBeat: 'attn-3-scan-row',
    visualFocusNote:
      'Now the Query token runs that exact dot product against every token in the sentence, producing a single 1D row of raw signed scores.',
    paragraphs: {
      foundations: [
        'Now that you know how one pair produces one score, repeat that exact test across the whole sentence. Our chosen Query word slides its Query strip past every word’s Key strip—producing one raw signed bar above each word.',
        'Tall amber bars mean strong agreement; downward blue bars mean "not what I am looking for." Notice we still do NOT have a scary 2D table—just one single line of match scores for one word.',
        'Look closely at the raw numbers: some are negative, some are positive, and they don’t add up to 100% yet. We cannot use raw negative numbers to mix word meanings cleanly—so we need two more steps next.',
      ],
      practitioner: [
        'Fixing a single query row i and dotting q_i against all key vectors K_h[0..T-1] produces one 1D vector of length T: S_h[i, :] = (q_i @ K_h^T) / sqrt(32).',
        'These raw logits are unbounded real numbers (typically ranging from -4 to +4 in our trained miniature model). Try clicking different tokens in the bottom bar to inspect each token’s live raw score profile across the sentence.',
      ],
      formal: [
        'Row i of the pre-softmax score tensor is s_i = K_h q_i / √32 ∈ R^T. Notice that at this stage s_i contains entries for j > i as well—which would violate autoregressive causality if left unmasked during parallel training.',
      ],
    },
    keyTakeaway: {
      foundations:
        'One word testing every word in the sentence creates a single 1D row of raw positive and negative scores.',
      practitioner:
        'One row of S = QKᵀ/√32 holds T unbounded real compatibility scores before masking and normalization.',
      formal:
        'Unmasked row scores s_i ∈ R^T must be causally restricted and mapped onto the probability simplex Δ^{T-1}.',
    },
  },
  {
    id: 'attn-beat-4',
    beatNumber: 4,
    shortTitle: '4. Causal Mask & Softmax',
    heading: 'Blocking the Future & Turning Scores Into a 100% Budget',
    sceneBeat: 'attn-4-mask-softmax',
    visualFocusNote:
      'Watch the matte Causal Mask shutter seal off any future tokens to the right of your chosen word, then toggle Raw Scores vs. Softmax Weights.',
    paragraphs: {
      foundations: [
        'First, look at any tokens to the right of our chosen word. Because the model’s job is to guess what comes NEXT, a word is never allowed to peek at the future. The matte gray barrier seals off every future slot by setting its score to negative infinity (-∞).',
        'Second, we pass the surviving scores through Softmax. Softmax turns any messy list of positive and negative scores into clean percentages between 0% and 100% that ALWAYS add up to exactly 100%.',
        'Because Softmax uses exponents, big positive scores grab most of the spotlight while negative scores shrink near zero—and masked future words become an exact 0%. Try clicking an earlier word in the middle of the sentence to see the future shutter slide right up beside it.',
      ],
      practitioner: [
        'During training, all T tokens run in one parallel matrix multiply. Without the Causal Mask setting S[i, j] = -1e9 for all j > i, token 5 could trivially cheat by reading token 6. After exp(-1e9) = 0, future positions receive strictly zero attention weight.',
        'Row-wise Softmax creates competition: raising one key’s raw score automatically pulls probability mass away from every other key so sum_j A[i, j] = 1.000000. Use the Raw vs. Softmax toggle on stage to compare before and after on the exact same row.',
      ],
      formal: [
        'Causal masking enforces M_{i,j} = S_{i,j} for j ≤ i and M_{i,j} = -∞ (implemented as -1e9 in src/engine/transformer.ts) for j > i. Row-wise Softmax A_{i,j} = exp(M_{i,j} - m_i) / Σ_r exp(M_{i,r} - m_i) yields a lower-triangular row-stochastic matrix ensuring P(x_{1..T}) = ∏_t P(x_t | x_{<t}).',
      ],
    },
    keyTakeaway: {
      foundations:
        'The Causal Mask blindfolds future words, and Softmax turns the remaining scores into percentages summing to 100%.',
      practitioner:
        'Masking j > i to -∞ before row-wise Softmax enforces left-to-right causality and turns scores into a convex weight distribution.',
      formal:
        'Lower-triangular row-stochastic A_h guarantees autoregressive factorizability across parallel teacher-forced passes.',
    },
    predict: {
      id: 'ch4-b4-predict',
      question:
        'Why do we overwrite future tokens with -∞ BEFORE Softmax instead of simply replacing them with 0 before Softmax?',
      options: [
        {
          id: 'a',
          label: 'Because exp(0) = 1, so setting raw scores to 0 would STILL give future tokens positive attention weight! Only exp(-∞) becomes 0.',
        },
        {
          id: 'b',
          label: 'Because computers cannot multiply by zero.',
        },
        {
          id: 'c',
          label: 'There is no difference; 0 and -∞ give the exact same Softmax output.',
        },
      ],
      correctId: 'a',
      explanation:
        'Spot on! Since e⁰ = 1, a raw score of 0 actually gets real weight in Softmax. You must push masked scores to -∞ so e^(-∞) = 0.',
    },
  },
  {
    id: 'attn-beat-5',
    beatNumber: 5,
    shortTitle: '5. Mixing Values (A · V)',
    heading: 'Copying Meaning: The Value Mixing Bowl',
    sceneBeat: 'attn-5-value-mixer',
    visualFocusNote:
      'Watch the top attended tokens’ 32-number Value strips pour into the blended output vector A·V in exact proportion to their Softmax weights.',
    paragraphs: {
      foundations: [
        'Here is the step almost every tutorial forgets to explain clearly: Attention does NOT stop at highlighting which earlier words matter. Highlighting alone wouldn’t change what "He" means!',
        'Every word also created a third 32-number list called its Value (V)—the actual packet of meaning it wants to share if someone pays attention to it. Now, our chosen word builds a blended message by pouring together every earlier word’s Value strip multiplied by its Softmax percentage.',
        'Look at the 3D mixing stage: if "He" puts 40% weight on "Max" and 20% weight on "ball", its new 32-number output literally becomes 40% of Max’s Value plus 20% of ball’s Value (plus smaller splashes of the rest). When that blend gets added back onto "He" in the Residual Stream, the vector for "He" now physically carries Max and the ball inside it.',
      ],
      practitioner: [
        'Once A_h[i, :] is computed, the head forms a weighted linear combination of the Value matrix V_h [T, 32]: Z_h[i] = Σ_{j=0}^i A_h[i, j] · V_h[j].',
        'Notice the clean functional split: the QK circuit decides WHERE to read from, while V (paired with W_o) is the OV circuit deciding WHAT information gets copied into the destination token’s residual stream.',
        'When you disable a head in Ablation Mode, our engine zeroes out this exact Z_h tensor right before concatenation.',
      ],
      formal: [
        'For head h, Z_h = A_h V_h ∈ R^{T × 32} computes a convex combination of Value vectors v_j = LN_1(h_j) W_{V,h}^T for j ≤ i. Combined with the head slice of W_O, the linear map W_{OV,h} = W_{O,h} W_{V,h} controls how source features write into the residual stream subspace.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Attention isn’t just highlighting words—it physically blends earlier words’ Value vectors into the current word.',
      practitioner:
        'Z_h = A_h V_h copies weighted Value payloads across token positions; QK decides where, OV decides what.',
      formal:
        'The output of head h is the convex mixture Z_{h,i} = Σ_{j≤i} A_{h,i,j} (x_j W_{V,h}^T) projected via W_{O,h}.',
    },
  },
  {
    id: 'attn-beat-6',
    beatNumber: 6,
    shortTitle: '6. Full Matrix & 4 Heads',
    heading: 'Stacking Every Row & Running 4 Heads in Parallel',
    sceneBeat: 'attn-6-multihead-full',
    visualFocusNote:
      'Only now do all T rows stack into the classic lower-triangular [T, T] attention matrix—alongside all 4 parallel heads feeding into W_o.',
    paragraphs: {
      foundations: [
        'Now step back! Everything you just watched for ONE word happens simultaneously for EVERY word in the sentence. Stack all those 1D rows one behind another, and suddenly that famous staircase-shaped Attention Matrix makes complete sense—each row is one word looking back, and the flat gray half is the Causal Mask blocking the future.',
        'Even better: why stop at one spotlight? A word often needs to track multiple clues at once—one spotlight for the previous word, one for punctuation, one for the character’s name. So every block runs FOUR independent Attention Heads side-by-side (H0, H1, H2, H3).',
        'Switch between Head 0, 1, 2, and 3 below to see how each head looks at totally different words, or click "Disable Head" to unplug one head live and watch how the model’s prediction shifts. Finally, all four 32-number head outputs snap back together into 128 numbers and pass through W_o onto the Residual Highway.',
      ],
      practitioner: [
        'Stacking all T query rows gives the full tensor A_h [T, T], executed across all 4 heads in parallel as [4, T, T]. Concatenating Z_0..Z_3 restores [T, 128], which multiplies W_o [128, 128] to write back into the residual stream.',
        'Notice redundancy: try disabling different heads below—some heads are critical for the next-token prediction while others barely move the loss. In frontier LLMs (Llama, Gemini, Mistral), Grouped-Query Attention (GQA) shares K and V heads across multiple Q heads to cut KV-cache memory by 4–8× with minimal quality loss.',
      ],
      formal: [
        'MHA(X) = Concat(Z_0, ..., Z_{H-1}) W_O^T = Σ_{h=0}^{H-1} A_h X W_{V,h}^T W_{O,h}^T, where W_O ∈ R^{128×128} is partitioned column-wise into H=4 head output projections of shape [128, 32]. Because the sum is additive across heads, each head acts as an independent low-rank residual stream writer.',
      ],
    },
    keyTakeaway: {
      foundations:
        'The full matrix is just every word’s row stacked together—and 4 parallel heads let the model track 4 clues at once.',
      practitioner:
        'Four [T, 32] heads run in parallel, concatenate to [T, 128], and project through W_o back onto the residual stream.',
      formal:
        'Multi-Head Attention decomposes into an additive sum of H independent rank-32 QK/OV circuits writing to R^{d_model}.',
    },
  },
];

export const TEXTBOOK_CHAPTERS: ChapterMeta[] = [
  {
    id: 1,
    title: 'Breaking Language Into Atoms',
    subtitle: 'Tokenization, BPE Merges, Vocabulary IDs & Context Window',
    status: 'ready',
    openingMystery:
      'Why can’t computers read raw letters—and why do LLMs miscount the R’s in "strawberry"?',
    beats: CHAPTER_1_BEATS,
  },
  {
    id: 2,
    title: 'The Geometry of Meaning',
    subtitle: 'Embeddings (wte), 3D PCA Space, Positional Encoding & RoPE',
    status: 'ready',
    openingMystery:
      'How do meaningless integer IDs become directions in 128-dimensional space—and how does the model know word order?',
    beats: CHAPTER_2_BEATS,
  },
  {
    id: 3,
    title: 'The Unbroken Highway',
    subtitle: 'The Residual Stream, Additive Updates, Pre-LayerNorm & RMSNorm',
    status: 'ready',
    openingMystery:
      'How does a deep stack of layers keep adding new ideas without erasing earlier memory or letting numbers explode?',
    beats: CHAPTER_3_BEATS,
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
    status: 'ready',
    openingMystery:
      'If Attention only averages existing vectors, where does the model think privately and store learned facts?',
    beats: CHAPTER_5_BEATS,
  },
  {
    id: 6,
    title: 'Sharpening a Guess & Writing a Story',
    subtitle: 'Logit Lens, Unembedding, Temperature/Top-k/Top-p, Autoregression & KV Cache',
    status: 'ready',
    openingMystery:
      'If the math is 100% deterministic, why does the exact same prompt give different answers?',
    beats: CHAPTER_6_BEATS,
  },
  {
    id: 7,
    title: 'Where Every Number Came From',
    subtitle: 'Cross-Entropy Loss, Manual Backprop, AdamW, Training Replay & Hallucination',
    status: 'ready',
    openingMystery:
      'Nobody programmed grammar into this AI—how did 862,464 random numbers organize themselves into English?',
    beats: CHAPTER_7_BEATS,
  },
  {
    id: 8,
    title: 'From Miniature Toy to Frontier AI',
    subtitle: 'Surgical Ablation, Scaling Laws, Modern Upgrades & Post-Training (SFT / RLHF)',
    status: 'ready',
    openingMystery:
      'What stays identical and what fundamentally changes across a 1,000,000× jump to trillion-parameter LLMs?',
    beats: CHAPTER_8_BEATS,
  },
];

export function editionLabel(e: Edition): string {
  if (e === 'foundations') return 'Foundations';
  if (e === 'practitioner') return 'Practitioner';
  return 'Formal';
}
