import {
  CHAPTER_1_BEATS,
  CHAPTER_2_BEATS,
  CHAPTER_3_BEATS,
} from './chapters123';
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
          label:
            'Because "He" has not yet been linked to earlier context like "dog", "Max", and "ball".',
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
        'Without attention, "He" is just an isolated pronoun with no context. Attention allows it to pull the identity of "dog named Max" from earlier tokens.',
    },
  },
  {
    id: 'attn-beat-2',
    beatNumber: 2,
    shortTitle: '2. The Dot Product',
    heading: 'One Word Asks, One Word Answers',
    sceneBeat: 'attn-2-dot-product',
    visualFocusNote:
      'Look at the two spotlighted tokens and their 32-element Query (blue) and Key (emerald) bars. Notice the glowing bridge calculating their similarity.',
    paragraphs: {
      foundations: [
        'How do two words decide if they care about each other? Imagine the word "He" is looking for clues. It creates a Query vector: a list of 32 numbers describing what it wants to find.',
        'Earlier words like "dog" create a Key vector: a list of 32 numbers describing what kind of information they have to offer.',
        'To test if they match, the computer multiplies each of the 32 query numbers by the corresponding key number and adds them all up. This is a dot product. If the vectors point in the same direction, the sum is a big positive number (high match). If they are unrelated, the sum is near zero.',
      ],
      practitioner: [
        'For layer l and head h, Query q_t = W_Q h_t and Key k_s = W_K h_s each have head dimension d_k = 32 (since d_model = 128 and H = 4 heads).',
        'The unscaled score is dot(q_t, k_s) = sum_{i=0}^{31} q_{t,i} k_{s,i}. We divide by sqrt{d_k} = sqrt{32} ≈ 5.657 to keep the variance at 1.0, preventing extreme values from pushing softmax into saturation.',
        'The Live Card on the left displays the real numbers from our model: the raw dot product, the division by 5.657, and the final attention weight for this exact pair.',
      ],
      formal: [
        'Given linear projections W_Q, W_K ∈ R^{d_model × d_k}, let q_t = H_t W_Q and k_s = H_s W_K. Under standard assumptions (zero-mean unit-variance coordinates), Var(q_t^T k_s) = d_k.',
        'The scaling factor 1 / sqrt{d_k} stabilizes the variance of the logits: Var(q_t^T k_s / sqrt{d_k}) = 1, ensuring softmax gradients do not vanish during backpropagation.',
      ],
    },
    keyTakeaway: {
      foundations:
        'A dot product is a matching score: big positive score means "these two words belong together in this story".',
      practitioner:
        'Dot products compute query-key alignment; scaling by 1/sqrt(32) keeps softmax gradients healthy.',
      formal:
        'Scaled dot products implement an empirical kernel similarity metric whose variance is normalized to unity.',
    },
    predict: {
      id: 'ch4-b2-predict',
      question:
        'In our model where head dimension d_k = 32, why do we divide the dot product by sqrt(32) ≈ 5.657?',
      options: [
        {
          id: 'a',
          label:
            'To keep dot products from getting too large, which would make Softmax peak into extreme 1s and 0s and kill gradients.',
        },
        {
          id: 'b',
          label: 'Because the model has 32 layers.',
        },
        {
          id: 'c',
          label: 'To convert float32 numbers into integers.',
        },
      ],
      correctId: 'a',
      explanation:
        'Without scaling by sqrt(d_k), large dot products drive softmax into near-one-hot distributions with zero gradients, freezing learning.',
    },
  },
  {
    id: 'ch4-beat-3',
    beatNumber: 3,
    shortTitle: '3. Scanning the Row',
    heading: 'One Word Against Every Earlier Word',
    sceneBeat: 'attn-3-scan-row',
    visualFocusNote:
      'Look at the horizontal row of attention bars arching back from "He" to every earlier token in the sentence. Taller bars indicate stronger attention.',
    paragraphs: {
      foundations: [
        'Now watch "He" do this for the entire sentence at once! Its single query vector is compared against the key vector of EVERY token in the story so far.',
        'Look at the horizontal row of bars on stage. "He" gets a score with "Once", a score with "upon", a score with "dog", a score with "Max", and so on.',
        'Notice which bars stand tall: in our trained model, Head 3 of Layer 2 places strong attention on "dog" and "Max" because it is learning to resolve pronouns to their antecedents!',
      ],
      practitioner: [
        'This row represents the logit vector for token t: a_t = (q_t K^T) / sqrt{d_k} of length T, where K ∈ R^{T × 32} stacks all key vectors up to sequence length T.',
        'Use the Token Inspector above to click different query tokens in your prompt. Watch how different words look backward at different grammatical partners.',
      ],
      formal: [
        'For sequence position t, the pre-mask attention row is a_t = q_t K_{0:t}^T / sqrt{d_k} ∈ R^{t+1}.',
        'This computes the continuous affinity spectrum over all preceding temporal locations.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Each word produces a row of match scores showing how strongly it looks at every earlier word.',
      practitioner:
        'The row vector a_t = (q_t K^T) / sqrt(d_k) measures one token’s raw affinity against all available keys.',
      formal:
        'Attention affinity rows instantiate an instantaneous directed coupling density across the causal prefix.',
    },
  },
  {
    id: 'ch4-beat-4',
    beatNumber: 4,
    shortTitle: '4. Mask & Softmax',
    heading: 'No Peeking into the Future, Then Normalizing to 100%',
    sceneBeat: 'attn-4-mask-softmax',
    visualFocusNote:
      'Inspect the full [T × T] matrix. Notice the upper right triangle is completely blocked out by gray causal mask tiles.',
    paragraphs: {
      foundations: [
        'Two critical rules govern this score table. First: an LLM generates text one word at a time, so it must NEVER look into the future! We enforce this with a Causal Mask: every cell where a word tries to look ahead is set to minus infinity.',
        'Second: raw scores can be any number (+4.2, -1.8, 0.0). We need them to act like percentages that add up to 100%. We pass each row through Softmax: exponentiate each score (turning negatives into positive fractions) and divide by the sum.',
        'Toggle the "Pre-Softmax Logits" button in the HUD above to see the raw scores turn into normalized probabilities right before your eyes!',
      ],
      practitioner: [
        'The causal mask sets M_{i,j} = -inf for j > i and 0 otherwise. After adding M, Softmax computes: A_{i,j} = exp(S_{i,j}) / sum_k exp(S_{i,k}).',
        'Because exp(-inf) = 0, future positions receive an exact attention weight of 0.000, mathematically guaranteeing strict causality.',
        'Look at the live matrix heatmap on stage: the lower triangular structure is the visual hallmark of autoregressive generative models.',
      ],
      formal: [
        'Attention weights A = Softmax((Q K^T) / sqrt{d_k} + M), where M_{i,j} = 0 if j <= i and -inf if j > i.',
        'This projects each row onto the probability simplex Delta^{t}, preserving causality: d y_t / d x_s = 0 for all s > t.',
      ],
    },
    keyTakeaway: {
      foundations:
        'The causal mask blocks the future with -infinity; Softmax turns the remaining scores into percentages that sum to 1.0.',
      practitioner:
        'Causal masking M_{i,j} = -inf enforces zero attention to future tokens; softmax normalizes the prefix into a valid distribution.',
      formal:
        'The masked softmax operator confines dynamic coupling to the causal lower-triangular cone.',
    },
    predict: {
      id: 'ch4-b4-predict',
      question:
        'What would happen during autoregressive generation if we removed the causal mask?',
      options: [
        {
          id: 'a',
          label:
            'The model would cheat during training by copying answers from future tokens, then fail completely during real generation.',
        },
        {
          id: 'b',
          label: 'The model would run 2x faster.',
        },
        {
          id: 'c',
          label: 'The output text would become palindromes.',
        },
      ],
      correctId: 'a',
      explanation:
        'Without causal masking, training allows tokens to see future ground-truth words, preventing the model from learning autoregressive prediction.',
    },
  },
  {
    id: 'ch4-beat-5',
    beatNumber: 5,
    shortTitle: '5. The Value Mixer',
    heading: 'Borrowing Meaning as a Weighted Average',
    sceneBeat: 'attn-5-value-mixer',
    visualFocusNote:
      'Look at the convergence of glowing colored ribbons from earlier tokens into the final output column, weighted by the attention percentages.',
    paragraphs: {
      foundations: [
        'Now comes the payout! The attention percentages from Softmax are not just for display—they are a recipe for mixing words together.',
        'Every token also created a third vector called a Value vector: the actual information payload it is willing to share.',
        'If "He" gave 45% attention to "Max" and 35% attention to "dog", the computer takes 0.45 × (Max’s Value) + 0.35 × (dog’s Value) + 0.20 × (other words). The result is a brand-new vector for "He" that is now deeply infused with the meaning of a dog named Max!',
      ],
      practitioner: [
        'The attention head output is the matrix multiplication: Head = A V, where V = H W_V ∈ R^{T × 32}.',
        'For each position t, the output vector out_t = sum_{s=0}^t A_{t,s} v_s is a convex combination of value vectors.',
        'Notice the distinction: Query and Key determine WHERE to look; Value determines WHAT information is carried across the sequence.',
      ],
      formal: [
        'Let V = H W_V ∈ R^{T × d_v}. The single-head output is O_h = A V = Softmax((Q K^T)/sqrt{d_k} + M) V.',
        'Because each row of A is in the probability simplex Delta^T, each output vector lies in the convex hull of the input value projections {v_0, ..., v_t}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Softmax percentages are a blending recipe: the new word is a weighted average of earlier word value vectors.',
      practitioner:
        'Head output is a convex combination of Value vectors: out_t = sum_s A_{t,s} v_s.',
      formal:
        'The value aggregation operator maps the convex hull of V into the head representation space.',
    },
  },
  {
    id: 'ch4-beat-6',
    beatNumber: 6,
    shortTitle: '6. Four Heads Working Together',
    heading: 'Specialized Perspectives and the Output Projection',
    sceneBeat: 'attn-6-multihead-full',
    visualFocusNote:
      'Look at the 4 distinct colored heads on stage (Cyan, Emerald, Violet, Amber). Each head focuses on a different relationship in your sentence.',
    paragraphs: {
      foundations: [
        'Why have only one attention mechanism when you can have four? Our miniature model has 4 independent attention heads running in parallel in every layer.',
        'Think of them as four detectives reading the story through different glasses: Head 0 tracks the previous word, Head 1 looks back at sentence boundaries, Head 2 monitors verbs and actions, and Head 3 links pronouns to characters.',
        'At the end of the layer, the four 32-number answers are glued side-by-side into a single 128-number vector, multiplied by an Output matrix (W_O), and added back into the main residual highway!',
      ],
      practitioner: [
        'Multi-Head Attention computes 4 parallel heads: Head_h = A_h (H W_V^{(h)}). Their outputs are concatenated along the feature dimension: Concat(Head_0, ..., Head_3) ∈ R^{T × 128}.',
        'The concatenated tensor is multiplied by the output projection W_O ∈ R^{128 × 128}: MHA(H) = Concat(Head_0..3) W_O.',
        'Use the Head Selector in the HUD (Heads 0, 1, 2, 3) to inspect each head’s distinct attention pattern on your prompt.',
      ],
      formal: [
        'MHA(H) = Concat(head_0, ..., head_{H-1}) W_O, where head_h = Attention(H W_Q^{(h)}, H W_K^{(h)}, H W_V^{(h)}).',
        'Multi-head projection partitions the representation space into H orthogonal subspaces of dimension d_k = d_model / H = 32, allowing the model to simultaneously attend to information from different representation subspaces at different positions.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Four heads look at the sentence through four different lenses; their answers are glued together and added back to the main highway.',
      practitioner:
        'Concat(Head_0..3) @ W_O projects the four 32-dim subspace heads back into the 128-dim residual stream.',
      formal:
        'Multi-head attention combines H rank-d_k subspace attention maps into a full-rank linear representation in R^{d_model}.',
    },
    predict: {
      id: 'ch4-b6-predict',
      question:
        'If our model has d_model = 128 and 4 attention heads, what is the dimension d_k of each head, and what is the shape of W_O?',
      options: [
        {
          id: 'a',
          label: 'd_k = 32 (128 / 4), and W_O is [128, 128].',
        },
        {
          id: 'b',
          label: 'd_k = 128, and W_O is [512, 128].',
        },
        {
          id: 'c',
          label: 'd_k = 4, and W_O is [4, 4].',
        },
      ],
      correctId: 'a',
      explanation:
        'd_k = d_model / n_heads = 128 / 4 = 32. Concatenating 4 heads of size 32 reconstructs a 128-dim vector, which W_O [128, 128] projects.',
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

export function getChapterMeta(chapterId: number): ChapterMeta {
  return (
    TEXTBOOK_CHAPTERS.find((c) => c.id === chapterId) ?? TEXTBOOK_CHAPTERS[0]
  );
}

export function editionLabel(edition: Edition): string {
  switch (edition) {
    case 'foundations':
      return 'Foundations (Intuition)';
    case 'practitioner':
      return 'Practitioner (Code & Tensors)';
    case 'formal':
      return 'Formal (Math Proof)';
  }
}
