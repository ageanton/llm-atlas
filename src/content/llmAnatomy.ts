import type { Register } from '../state/atlasState';

export type NodeSceneView =
  | 'overview'
  | 'embed'
  | 'block'
  | 'attention'
  | 'mlp'
  | 'output';

export type NodeLabKind =
  | 'none'
  | 'tokenizer'
  | 'attention-matrix'
  | 'weight-matrix'
  | 'scalar'
  | 'sampling'
  | 'replay'
  | 'ablation'
  | 'scale'
  | 'logit-lens';

export interface AnatomyNode {
  id: string;
  parentId: string | null;
  title: string;
  shortTitle: string;
  oneLiner: string;
  status: 'live' | 'extension';
  shapeTag?: string;
  sceneView: NodeSceneView;
  sceneFocus?: string;
  labKind: NodeLabKind;
  /** Optional parameter template where {L} is replaced by the active layer index */
  paramTemplate?: string;
  children: string[];
  copy: Record<Register, string>;
  simplificationNote?: string;
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

export const ANATOMY_NODES: Record<string, AnatomyNode> = {
  root: {
    id: 'root',
    parentId: null,
    title: 'Large Language Model (Complete System)',
    shortTitle: 'LLM',
    oneLiner: 'A real 862,464-parameter miniature Transformer plus every component of modern LLMs.',
    status: 'live',
    shapeTag: '862,464 params · 4 blocks',
    sceneView: 'overview',
    labKind: 'none',
    children: [
      'input-stage',
      'embeddings',
      'transformer-stack',
      'output-stage',
      'training-lifecycle',
      'experiments',
    ],
    copy: {
      simple:
        'A Large Language Model is a machine that reads a piece of text and predicts which small piece of text is most likely to come next. Laid out left to right on screen is the entire real miniature model running in your browser. Click any part in 3D—or open any folder below—to expand it and see exactly what it is made of.',
      sufficient:
        'This atlas exposes a real, trained 4-layer decoder-only Transformer (862,464 parameters, d_model=128, 4 heads, 512-token BPE vocabulary) alongside the architectural and training components used in frontier LLMs. Every colored plate in the 3D cutaway displays real activations computed live on your prompt. Select any stage to explode it into its exact sub-components.',
      complete:
        'An autoregressive Transformer defines P_θ(x_{1..T}) = ∏_{t=1}^T P_θ(x_t | x_{<t}) via tokenization, embedding lookup, L stacked Pre-LN residual blocks (MHA + FFN), final normalization, and tied unembedding. Every displayed activation and parameter is decoded verbatim from public/model/weights.fp16.bin and verified against PyTorch to <1e-4.',
    },
    simplificationNote:
      'You are zoomed all the way out. Click any component below or in 3D to peel it open layer by layer.',
  },

  // 1. INPUT & TOKENIZATION
  'input-stage': {
    id: 'input-stage',
    parentId: 'root',
    title: '1. Input, Chat Formatting & Tokenization',
    shortTitle: 'Input & Tokens',
    oneLiner: 'Converts raw human text and conversation roles into integer token IDs.',
    status: 'live',
    shapeTag: 'string → [T] ints',
    sceneView: 'embed',
    sceneFocus: 'input',
    labKind: 'tokenizer',
    children: ['chat-templates', 'tokenizer'],
    copy: {
      simple:
        'Computers cannot do math directly on English sentences. Before the neural network can read anything, the input is wrapped in any conversation markers and chopped into small numbered chunks called tokens.',
      sufficient:
        'The input stage prepares raw text for numerical computation. Production assistants wrap messages in a Chat Template with special role tokens, then pass the resulting string through a subword Tokenizer (usually BPE) to produce an integer sequence of length T ≤ T_max.',
      complete:
        'Input preprocessing maps a UTF-8 string s through deterministic role serialization and a finite vocabulary encoder E: Σ* → {0, ..., |V|-1}^T, truncated to the model context length T_max (64 in this miniature model; 128k–1M+ in frontier LLMs).',
    },
  },

  'chat-templates': {
    id: 'chat-templates',
    parentId: 'input-stage',
    title: 'System & Chat Templates',
    shortTitle: 'Chat Templates',
    oneLiner: 'Special control tokens that turn a raw text completer into a turn-by-turn assistant.',
    status: 'extension',
    shapeTag: '<|system|> <|user|> <|assistant|>',
    sceneView: 'embed',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'At its core, an LLM only continues text like finishing a story. To make it act like a helpful chatbot, apps quietly wrap your message in invisible tags like "User:" and "Assistant:" so the model knows it is its turn to answer—and knows when to stop talking.',
      sufficient:
        'Base pretrained models only predict continuations. Chat models introduce reserved special tokens (e.g., <|system|>, <|user|>, <|assistant|>, <|eot|>) during fine-tuning. Everything you type—plus system instructions and tool definitions—is serialized into one single token sequence before entering the Transformer.',
      complete:
        'Chat templates format multi-turn dialogue state [(r_1, m_1), ..., (r_k, m_k)] into a flat token array punctuated by dedicated vocabulary IDs outside normal BPE merges. Generation halts when the sampler emits an end-of-turn stop token ID.',
    },
  },

  tokenizer: {
    id: 'tokenizer',
    parentId: 'input-stage',
    title: 'BPE Tokenizer & Vocabulary',
    shortTitle: 'Tokenizer (BPE)',
    oneLiner: 'Slices text into subword chunks from a fixed dictionary of 512 tokens.',
    status: 'live',
    shapeTag: '|V| = 512 tokens',
    sceneView: 'embed',
    labKind: 'tokenizer',
    children: ['bpe-merges', 'vocab-ids'],
    copy: {
      simple:
        'Instead of reading letter by letter or memorizing every word in the dictionary, the model uses common word pieces (like " the", "ing", or " dog"). Common words stay whole; rare words split into smaller pieces. Every piece gets a fixed ID number.',
      sufficient:
        'Byte-Pair Encoding (BPE) starts with individual characters and iteratively merges the most frequent adjacent pairs in the training corpus until reaching vocabulary size |V| (512 here; 32k–128k+ in frontier LLMs). Leading spaces are attached to words, which is why " cat" and "cat" have different IDs.',
      complete:
        'Deterministic greedy rank-ordered BPE over pre-tokenized regex spans (` ?[A-Za-z]+| ?[0-9]+|...`). This model uses 98 base tokens (<|eos|>, <|unk|>, newline, printable ASCII) plus 414 learned merges learned on TinyStories. Exact character spans [start, end) map every token back to source characters.',
    },
  },

  'bpe-merges': {
    id: 'bpe-merges',
    parentId: 'tokenizer',
    title: 'Text Slicing & Byte-Pair Merges',
    shortTitle: 'BPE Merges',
    oneLiner: 'Why LLMs see word chunks rather than individual letters (and struggle with letter counting).',
    status: 'live',
    shapeTag: '414 merge rules',
    sceneView: 'embed',
    labKind: 'tokenizer',
    children: [],
    copy: {
      simple:
        'When you type a word, the tokenizer glues frequent letter pairs together step by step until no more merges apply. Because whole chunks become a single ID number, an LLM does not automatically see individual letters—which is why LLMs can make silly mistakes when counting letters in a word.',
      sufficient:
        'During encoding, each pre-token starts as single characters and repeatedly merges whichever adjacent pair has the lowest rank in merges[]. This balances sequence length (fewer tokens = cheaper attention) against vocabulary size (smaller embedding table).',
      complete:
        'Given merge table M = [(a_0, b_0), ..., (a_{k-1}, b_{k-1})], encoding greedily contracts argmin_i rank(s_i, s_{i+1}) until no adjacent pair appears in M. Out-of-vocabulary codepoints map to <|unk|> (ID 1).',
    },
  },

  'vocab-ids': {
    id: 'vocab-ids',
    parentId: 'tokenizer',
    title: 'Token IDs & Context Window Limit',
    shortTitle: 'IDs & Context',
    oneLiner: 'The hard window of tokens (T ≤ 64 here) the model can see at one time.',
    status: 'live',
    shapeTag: 'T ≤ 64 slots',
    sceneView: 'embed',
    labKind: 'tokenizer',
    children: [],
    copy: {
      simple:
        'Every token chunk is swapped for its ID number from 0 to 511. The model also has a maximum desk size—called its context window. Anything that falls off the edge of the desk simply does not exist to the model.',
      sufficient:
        'The output of tokenization is a 1D integer vector x ∈ {0..511}^T. Our miniature model has a context window T_max = 64 tokens (fixed by its 64-row positional table). Frontier LLMs extend T_max to 128,000–1,000,000+ tokens.',
      complete:
        'Integer indices carry no ordinal semantics (ID 104 is unrelated to ID 105). Sequence truncation slices x[:64] prior to embedding lookup; attention compute and KV memory scale with T.',
    },
  },

  // 2. EMBEDDINGS & POSITION
  embeddings: {
    id: 'embeddings',
    parentId: 'root',
    title: '2. Embedding Layer & Positional Encoding',
    shortTitle: 'Embeddings',
    oneLiner: 'Turns each discrete token ID and slot number into a 128-number vector.',
    status: 'live',
    shapeTag: '[T, 128] slab',
    sceneView: 'embed',
    labKind: 'none',
    children: ['wte', 'wpe', 'rope', 'embed-sum'],
    copy: {
      simple:
        'An ID number like #104 tells the model nothing about what a word means. So the model looks up a list of 128 learned numbers for each token (its meaning), adds another list of 128 numbers for where it sits in the sentence (its position), and hands that grid of numbers to Block 0.',
      sufficient:
        'Embeddings map discrete token IDs into continuous d_model-dimensional geometry (128 here; 4,096–16,384 in frontier LLMs) where semantically related tokens cluster together. Adding positional vectors injects word order because self-attention itself is order-blind.',
      complete:
        'H_0 = W_E[x] + W_P[0..T-1] ∈ R^{T×128}. W_E ∈ R^{512×128} (65,536 params, weight-tied to lm_head) and W_P ∈ R^{64×128} (8,192 params) are jointly trained via backpropagation.',
    },
  },

  wte: {
    id: 'wte',
    parentId: 'embeddings',
    title: 'Token Embedding Matrix (wte)',
    shortTitle: 'wte [512×128]',
    oneLiner: 'A learned 512-row table giving every vocabulary token its own 128-number coordinate.',
    status: 'live',
    shapeTag: '[512, 128] · 65,536 params',
    sceneView: 'embed',
    sceneFocus: 'wte',
    labKind: 'weight-matrix',
    paramTemplate: 'wte.weight',
    children: [],
    copy: {
      simple:
        'Think of this matrix as a giant coordinate book with 512 rows—one row per token in the vocabulary. Words that play similar roles in stories gradually get nudged during training so their 128 numbers point in similar directions. Click any pixel in the matrix below to zoom into one single number.',
      sufficient:
        'Row lookup into wte [512, 128] replaces each integer token ID with a dense vector of 128 floats. In our model (and GPT-2), this exact same matrix is reused at the very end of the network to score next-token predictions (weight tying).',
      complete:
        'W_E ∈ R^{|V|×d_model} initialized from N(0, 0.02^2). Indexing W_E[x_t] is algebraically identical to multiplying a one-hot vector e_{x_t} ∈ R^{512} by W_E. Click any cell (r, c) below to inspect W_E[r, c] at the Single Scalar level.',
    },
  },

  wpe: {
    id: 'wpe',
    parentId: 'embeddings',
    title: 'Learned Positional Embeddings (wpe)',
    shortTitle: 'wpe [64×128]',
    oneLiner: 'Learned vectors marking slot 0, slot 1, ..., slot 63 so the model knows word order.',
    status: 'live',
    shapeTag: '[64, 128] · 8,192 params',
    sceneView: 'embed',
    sceneFocus: 'wpe',
    labKind: 'weight-matrix',
    paramTemplate: 'wpe.weight',
    children: [],
    copy: {
      simple:
        '"The dog chased the cat" and "The cat chased the dog" use the exact same words in a different order. Without position numbers, attention would treat a sentence like a jumbled bag of words. This table gives each slot (0 to 63) its own 128-number tag.',
      sufficient:
        'Learned absolute positional embeddings store one trainable 128-dim vector per sequence slot 0..63. Simple and effective for short contexts, though it hard-caps sequence length at 64 tokens.',
      complete:
        'W_P ∈ R^{T_max×d_model} indexed by arange(T). Added directly to W_E[x] prior to Block 0. Click any pixel below to inspect an exact positional weight scalar.',
    },
  },

  rope: {
    id: 'rope',
    parentId: 'embeddings',
    title: 'RoPE — Rotary Position Embeddings',
    shortTitle: 'RoPE (Modern)',
    oneLiner: 'How modern LLMs encode relative distance by rotating Q and K vectors inside Attention.',
    status: 'extension',
    shapeTag: 'Rotation in 2D pairs',
    sceneView: 'embed',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Instead of adding a fixed slot tag at the beginning like our tiny model does, modern LLMs like Llama and Gemini gently spin each word’s Query and Key arrows by an angle based on its position. That way, two words automatically know how far apart they are—even in a 100-page document.',
      sufficient:
        'Rotary Position Embedding (RoPE) pairs up dimensions of Q and K inside every attention layer and rotates each 2D pair by an angle m·θ_i proportional to token position m. Because rotation preserves norms and depends only on (m - n) in dot products, models extrapolate cleanly to huge context windows.',
      complete:
        'For position m and dimension pair (2i, 2i+1), RoPE applies R_Θ,m^(d) with θ_i = base^(-2i/d). Consequently (R_m q)^T (R_n k) = q^T R_{n-m} k, injecting exact relative positional decay into attention logits without extra learned parameters.',
    },
  },

  'embed-sum': {
    id: 'embed-sum',
    parentId: 'embeddings',
    title: 'Initial Residual Stream (h₀ = wte + wpe)',
    shortTitle: 'h₀ Stream Start',
    oneLiner: 'Element-wise sum of token meaning and slot position entering Block 0.',
    status: 'live',
    shapeTag: '[T, 128] float32',
    sceneView: 'embed',
    sceneFocus: 'h0',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'We add each token’s meaning list and position list number-by-number to create one combined 128-number strip per token. Stack all tokens together and you get the exact 2D grid of numbers that rides through the entire model.',
      sufficient:
        'Adding wte[x] + wpe[pos] produces h_0 ∈ R^{T×128}. Even though addition blends both vectors into one 128-dim space, high-dimensional geometry lets downstream heads read semantic and positional subspaces independently.',
      complete:
        'h_0[t, d] = W_E[x_t, d] + W_P[t, d]. From this point until Final LayerNorm, every hidden state in the main spine retains invariant shape [T, 128].',
    },
  },

  // 3. TRANSFORMER STACK
  'transformer-stack': {
    id: 'transformer-stack',
    parentId: 'root',
    title: '3. Transformer Block Stack (Blocks 0–3)',
    shortTitle: 'Transformer Blocks',
    oneLiner: 'Four identical Pre-LN blocks stacked in sequence, each refining the Residual Stream.',
    status: 'live',
    shapeTag: '4× blocks · 788,480 params',
    sceneView: 'block',
    labKind: 'none',
    children: [
      'residual-stream',
      'ln1',
      'attention',
      'ln2',
      'mlp',
      'logit-lens',
    ],
    copy: {
      simple:
        'This is the heart of the LLM: four identical rooms lined up in a row (frontier LLMs stack 32 to 128+ rooms). Every room has the same layout—first words talk to each other in Attention, then each word thinks on its own in the MLP—but each room learns its own private set of weights.',
      sufficient:
        'Each Transformer block reads from the [T, 128] residual stream, computes a context-mixing update via Multi-Head Self-Attention, adds it back, then computes a per-token feature update via the MLP and adds it back. Early blocks handle local patterns; deeper blocks handle longer-range grammar and prediction formatting.',
      complete:
        'Each block l ∈ {0..3} holds 197,120 parameters (LN1: 256, MHA: 65,536, LN2: 256, MLP: 131,072) executing h_{l+1} = h_l + MHA_l(LN1(h_l)) + MLP_l(LN2(h_l + MHA_l(LN1(h_l)))). Use the Block 0–3 selector on the 3D stage to inspect any depth.',
    },
  },

  'residual-stream': {
    id: 'residual-stream',
    parentId: 'transformer-stack',
    title: 'The Residual Stream (Shared Memory Bus)',
    shortTitle: 'Residual Stream',
    oneLiner: 'The unbroken central highway that layers read from and add updates onto without erasing.',
    status: 'live',
    shapeTag: '[T, 128] unbroken',
    sceneView: 'block',
    sceneFocus: 'residual',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Look at the solid blue highway running straight underneath every block. The model never erases what it already knows. Every Attention and MLP block simply reads a copy of the highway, figures out a helpful note, and adds (+) that note right back onto the highway.',
      sufficient:
        'Residual connections (skip connections: x + SubLayer(x)) turn the Transformer into a shared communication bus. Earlier blocks can write features that later blocks read directly, and during training, gradients flow cleanly straight back through the addition nodes without vanishing.',
      complete:
        'Because h_L = h_0 + Σ_{l=0}^{L-1} (Δ_attn^(l) + Δ_mlp^(l)), every block operates in a common d_model coordinate basis. Disabling residual bypasses in the Ablation Lab demonstrates why deep networks collapse without them.',
    },
  },

  ln1: {
    id: 'ln1',
    parentId: 'transformer-stack',
    title: 'Pre-LayerNorm 1 (ln_1) & RMSNorm',
    shortTitle: 'Pre-LayerNorm 1',
    oneLiner: 'Re-centers and rescales each token vector before Attention so numbers never explode.',
    status: 'live',
    shapeTag: '[128] γ + [128] β',
    sceneView: 'block',
    sceneFocus: 'ln1',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.ln_1.weight',
    children: ['ln1-scale-bias', 'rmsnorm'],
    copy: {
      simple:
        'After adding lots of numbers together, some token vectors can get way too huge or tiny. LayerNorm acts like an automatic volume leveler before Attention—keeping every token’s 128 numbers in a steady, healthy range.',
      sufficient:
        'Pre-LayerNorm normalizes each token’s 128 numbers independently to mean 0 and variance 1 before entering Multi-Head Attention, then applies learned per-dimension scale (γ) and bias (β). Placing norm before the sub-layer (Pre-LN) makes training far more stable than the original 2017 Post-LN design.',
      complete:
        'LN_1(v) = ((v - μ) / √(σ² + 1e-5)) ⊙ γ + β applied along the last dimension per token t. No cross-token mixing occurs here.',
    },
  },

  'ln1-scale-bias': {
    id: 'ln1-scale-bias',
    parentId: 'ln1',
    title: 'Learned Scale (γ) & Bias (β)',
    shortTitle: 'Scale γ & Bias β',
    oneLiner: '256 trainable numbers that let LayerNorm stretch or shift each channel back after normalizing.',
    status: 'live',
    shapeTag: '256 params per LN',
    sceneView: 'block',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.ln_1.weight',
    children: [],
    copy: {
      simple:
        'Forcing every channel to have the exact same volume might accidentally silence an important feature. So LayerNorm includes 128 learned volume knobs (scale γ) and 128 offset knobs (bias β) that training adjusts automatically.',
      sufficient:
        'Initialized to γ = 1 and β = 0, these affine parameters restore representational freedom after standardization so the layer can emphasize specific residual stream dimensions.',
      complete:
        'γ, β ∈ R^{128} stored as blocks.{L}.ln_1.weight and blocks.{L}.ln_1.bias. Click any cell in the vector strip below to inspect one scalar weight.',
    },
  },

  rmsnorm: {
    id: 'rmsnorm',
    parentId: 'ln1',
    title: 'RMSNorm (Root Mean Square Norm)',
    shortTitle: 'RMSNorm (Modern)',
    oneLiner: 'Modern simplified normalization that skips mean subtraction for faster GPU execution.',
    status: 'extension',
    shapeTag: 'v / RMS(v) ⊙ γ',
    sceneView: 'block',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Researchers discovered you do not even need to calculate the average (mean) to keep numbers steady—just scaling by the overall size of the vector works just as well and runs faster. Almost all modern LLMs use RMSNorm.',
      sufficient:
        'RMSNorm drops mean-centering and the bias β, dividing each vector purely by its root-mean-square magnitude √(mean(v²) + ε) and multiplying by learned scale γ. Used in Llama, Gemma, Mistral, and DeepSeek.',
      complete:
        'RMSNorm(v) = (v / √((1/d) Σ_i v_i² + ε)) ⊙ γ. Eliminates one reduction pass per token while preserving re-scaling invariance.',
    },
  },

  // MULTI-HEAD SELF-ATTENTION
  attention: {
    id: 'attention',
    parentId: 'transformer-stack',
    title: 'Multi-Head Self-Attention',
    shortTitle: 'Self-Attention',
    oneLiner: 'The ONLY part of a Transformer where tokens communicate and pull context from earlier tokens.',
    status: 'live',
    shapeTag: '4 heads · 65,536 params',
    sceneView: 'attention',
    labKind: 'attention-matrix',
    children: [
      'qkv-proj',
      'head-split',
      'dot-product',
      'causal-mask',
      'attn-softmax',
      'weighted-v',
      'w-o',
    ],
    copy: {
      simple:
        'Every other part of an LLM looks at one word at a time. Self-Attention is the meeting room where each word looks back at earlier words to figure out context—like realizing that "He" refers to "Max" and "ball". Click any step below to see exactly how that lookup works.',
      sufficient:
        'Multi-Head Self-Attention projects each token into Query (what I want), Key (what I offer), and Value (what information I carry) vectors across 4 parallel heads (d_head = 32). Each head computes causal-masked attention weights A and forms a weighted sum A·V before recombining via W_o.',
      complete:
        'MHA(X) = Concat_{h=0}^3(softmax(mask(Q_h K_h^T / √32)) V_h) W_O^T with Q_h, K_h, V_h ∈ R^{T×32}. Hover tokens at the bottom of the 3D view to cross-highlight query rows, key columns, and live attention arcs.',
    },
  },

  'qkv-proj': {
    id: 'qkv-proj',
    parentId: 'attention',
    title: 'Q, K, V Linear Projections',
    shortTitle: 'Q, K, V Projections',
    oneLiner: 'Three separate learned weight matrices turning each token into a Query, a Key, and a Value.',
    status: 'live',
    shapeTag: '3 × [128, 128] = 49,152 params',
    sceneView: 'attention',
    sceneFocus: 'qkv',
    labKind: 'none',
    children: ['w-q', 'w-k', 'w-v'],
    copy: {
      simple:
        'Every token gets asked three different questions using three grids of learned numbers: Query ("what am I looking for?"), Key ("what label do I advertise to others?"), and Value ("if someone picks me, what information should I hand over?").',
      sufficient:
        'Before comparing tokens, the normalized [T, 128] stream is multiplied by three distinct learned matrices W_q, W_k, and W_v (each [128, 128], no bias). Separating Q, K, and V lets a token search for something totally different from what it advertises.',
      complete:
        'Q = X W_Q^T, K = X W_K^T, V = X W_V^T ∈ R^{T×128}. Open W_q, W_k, or W_v below to inspect the literal float16 matrices and click any single scalar weight.',
    },
  },

  'w-q': {
    id: 'w-q',
    parentId: 'qkv-proj',
    title: 'Query Weight Matrix (W_q)',
    shortTitle: 'W_q [128×128]',
    oneLiner: 'Projects each token into the "search question" it asks of earlier tokens.',
    status: 'live',
    shapeTag: '[128, 128] · 16,384 params',
    sceneView: 'attention',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.attn.W_q.weight',
    children: [],
    copy: {
      simple:
        'This 128×128 wall of learned numbers translates each token into a search query—like a pronoun asking, "Where is the most recent character name?" Click any colored pixel to zoom all the way into one real weight.',
      sufficient:
        'W_q [128, 128] transforms LN_1(h) into 4 stacked 32-dim query vectors per token. Each 32-row horizontal band belongs to one of the 4 attention heads.',
      complete:
        'blocks.{L}.attn.W_q.weight ∈ R^{128×128}, bias=False. Rows [32h : 32(h+1)] compute Q_h = X (W_q[h])^T.',
    },
  },

  'w-k': {
    id: 'w-k',
    parentId: 'qkv-proj',
    title: 'Key Weight Matrix (W_k)',
    shortTitle: 'W_k [128×128]',
    oneLiner: 'Projects each token into the "searchable label" it advertises to later tokens.',
    status: 'live',
    shapeTag: '[128, 128] · 16,384 params',
    sceneView: 'attention',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.attn.W_k.weight',
    children: [],
    copy: {
      simple:
        'This 128×128 grid creates a name-tag for every token so later tokens can find it—like the word "Max" advertising "I am a dog’s name!" Click any pixel to inspect a single learned number.',
      sufficient:
        'W_k [128, 128] generates key vectors K_h [T, 32] per head. Together, W_q and W_k form a low-rank bilinear matching circuit (the QK circuit) determining *where* attention flows.',
      complete:
        'blocks.{L}.attn.W_k.weight ∈ R^{128×128}, bias=False. Matching scores arise from X (W_q^T W_k) X^T / √32.',
    },
  },

  'w-v': {
    id: 'w-v',
    parentId: 'qkv-proj',
    title: 'Value Weight Matrix (W_v)',
    shortTitle: 'W_v [128×128]',
    oneLiner: 'Projects each token into the actual payload of information copied when attended to.',
    status: 'live',
    shapeTag: '[128, 128] · 16,384 params',
    sceneView: 'attention',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.attn.W_v.weight',
    children: [],
    copy: {
      simple:
        'Finding the right earlier word is only half the job—you also need to decide *what information* to borrow from it. W_v packs each word’s useful details into a 32-number message ready to be blended.',
      sufficient:
        'W_v [128, 128] computes value vectors V_h [T, 32] per head. Paired with W_o, it forms the OV circuit determining *what* information gets copied into the residual stream once attention selects a source token.',
      complete:
        'blocks.{L}.attn.W_v.weight ∈ R^{128×128}, bias=False. Downstream weighted blend computes Z_h = A_h (X W_v[h]^T).',
    },
  },

  'head-split': {
    id: 'head-split',
    parentId: 'attention',
    title: 'Parallel Heads (H0–H3) & GQA / MQA',
    shortTitle: 'Multi-Head & GQA',
    oneLiner: 'Splits 128 channels into 4 independent 32-dim heads so a token tracks 4 clues at once.',
    status: 'live',
    shapeTag: '4 × 32 = 128 dims',
    sceneView: 'attention',
    labKind: 'attention-matrix',
    children: [],
    copy: {
      simple:
        'Why have 4 heads instead of 1? Because a single word often needs to pay attention to several things at once—Head 0 might look at the word right before it, Head 1 at punctuation, and Head 3 back at a character’s name. Switch between H0–H3 in the 3D view to see each head’s distinct pattern.',
      sufficient:
        'Reshaping [T, 128] into [4, T, 32] lets 4 independent attention patterns run in parallel at the same total parameter budget. Modern frontier LLMs often use Grouped-Query Attention (GQA) or Multi-Query Attention (MQA), sharing K and V across groups of Q heads to shrink KV-cache memory by 4–8×.',
      complete:
        'Standard MHA uses H_q = H_kv = 4 with d_head = d_model / H = 32. GQA generalizes to H_kv | H_q (e.g., 64 Q heads sharing 8 KV heads), while DeepSeek’s Multi-Head Latent Attention (MLA) compresses KV into a shared low-rank latent vector.',
    },
  },

  'dot-product': {
    id: 'dot-product',
    parentId: 'attention',
    title: 'Scaled Dot-Product Scores (QKᵀ / √32)',
    shortTitle: 'Scores QKᵀ/√32',
    oneLiner: 'Measures alignment between every Query vector and every Key vector, scaled by 1/√d_head.',
    status: 'live',
    shapeTag: '[T, T] raw scores',
    sceneView: 'attention',
    labKind: 'attention-matrix',
    children: [],
    copy: {
      simple:
        'To see how well two words match, the model multiplies their 32 Query and Key numbers pair-by-pair and adds them up. A high positive total means a strong match; a negative total means "not relevant." Dividing by √32 (~5.66) keeps the scores from growing too extreme.',
      sufficient:
        'Every cell (i, j) is the dot product q_i · k_j divided by √32. Without the 1/√d_head scaling factor, variance grows linearly with head dimension, saturating Softmax into near-one-hot peaks whose gradients vanish during training.',
      complete:
        'S_{h,i,j} = (1 / √32) Σ_{d=0}^{31} Q_{h,i,d} K_{h,j,d}. Toggle "Pre-Softmax" in the matrix below to inspect exact unmasked S_{h,i,j} values.',
    },
  },

  'causal-mask': {
    id: 'causal-mask',
    parentId: 'attention',
    title: 'Causal Mask (Enforcing the Arrow of Time)',
    shortTitle: 'Causal Mask',
    oneLiner: 'Blocks every token from peeking at future tokens (j > i) by setting those scores to -∞.',
    status: 'live',
    shapeTag: 'j > i → -1e9 (-∞)',
    sceneView: 'attention',
    sceneFocus: 'mask',
    labKind: 'attention-matrix',
    children: [],
    copy: {
      simple:
        'The model’s job is to guess the next word, so it is never allowed to peek ahead at words that haven’t happened yet. The matte gray triangle blocking the upper-right half of the 3D grid is a literal blindfold over future words—every word can only look at itself and earlier words.',
      sufficient:
        'Because a Transformer processes all T tokens simultaneously in one matrix multiply during training, token i would otherwise cheat by reading token i+1. Overwriting every upper-right cell (j > i) with -∞ ensures e^{-∞} = 0 after Softmax, strictly enforcing left-to-right causality.',
      complete:
        'M_{i,j} = S_{i,j} for j ≤ i and M_{i,j} = -1e9 for j > i (src/engine/transformer.ts). Guarantees autoregressive factorizability P(x_{1..T}) = ∏_t P(x_t | x_{<t}) across parallel teacher-forced batches.',
    },
  },

  'attn-softmax': {
    id: 'attn-softmax',
    parentId: 'attention',
    title: 'Row-Wise Softmax (Attention Weights A)',
    shortTitle: 'Softmax Weights A',
    oneLiner: 'Turns raw compatibility scores into percentages between 0% and 100% summing to 1.000 per row.',
    status: 'live',
    shapeTag: 'Σ_j A[i,j] = 1.000',
    sceneView: 'attention',
    labKind: 'attention-matrix',
    children: [],
    copy: {
      simple:
        'Raw match scores can be negative or huge. Softmax converts every row into a clean 100% budget of attention: strong matches grab a big slice of the bar height and glow dark cobalt, weak matches shrink near zero, and masked future words become an exact 0%.',
      sufficient:
        'Applying Softmax row-wise turns M_h into probability weights A_h ∈ [0, 1]^{T×T}. Because every row must sum to 1.0, tokens compete for attention—raising one key’s score automatically reduces weight on others.',
      complete:
        'A_{h,i,j} = exp(M_{h,i,j} - max_m M_{h,i,m}) / Σ_{m=0}^{T-1} exp(M_{h,i,m} - max_r M_{h,i,r}). Numerically stabilized via max-shift; every row sums to 1.000000 (±1e-6).',
    },
  },

  'weighted-v': {
    id: 'weighted-v',
    parentId: 'attention',
    title: 'Weighted Sum of Values (A · V)',
    shortTitle: 'Weighted Blend A·V',
    oneLiner: 'Copies and blends earlier tokens’ Value vectors proportionally to the attention weights.',
    status: 'live',
    shapeTag: '[T, 32] per head',
    sceneView: 'attention',
    sceneFocus: 'av',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Attention does not stop at highlighting which earlier word matters—it actually mixes the information from those words together. If "He" puts 60% weight on "Max" and 25% on "dog", its output vector becomes a 60/25 blend of their Value messages.',
      sufficient:
        'Multiplying the [T, T] attention weight matrix A_h by the [T, 32] Value matrix V_h computes a convex combination of Value vectors for every token position. This is the physical mechanism that moves context across positions.',
      complete:
        'Z_{h,i} = Σ_{j=0}^i A_{h,i,j} V_{h,j} ∈ R^{32}. When a head is ablated in our Ablation Lab, Z_h is zeroed out right here before concatenation.',
    },
  },

  'w-o': {
    id: 'w-o',
    parentId: 'attention',
    title: 'Concatenation & Output Projection (W_o)',
    shortTitle: 'W_o [128×128]',
    oneLiner: 'Stitches all 4 heads back into 128 channels and translates them into the Residual Stream.',
    status: 'live',
    shapeTag: '[128, 128] · 16,384 params',
    sceneView: 'attention',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.attn.W_o.weight',
    children: [],
    copy: {
      simple:
        'Each of the 4 heads finishes with a 32-number summary. We glue all four summaries side-by-side back into 128 numbers, pass them through W_o to blend their findings together, and add the result onto the central Residual Stream.',
      sufficient:
        'Concat(Z_0, Z_1, Z_2, Z_3) forms [T, 128], which is linearly projected by W_o [128, 128]. Mathematically, this is equivalent to each head having its own [128, 32] output slice writing an independent additive update into the residual stream.',
      complete:
        'blocks.{L}.attn.W_o.weight ∈ R^{128×128}, initialized with residual scaling std = 0.02 / √(2·n_layers) to control residual variance across depth.',
    },
  },

  ln2: {
    id: 'ln2',
    parentId: 'transformer-stack',
    title: 'Pre-LayerNorm 2 (ln_2)',
    shortTitle: 'Pre-LayerNorm 2',
    oneLiner: 'Re-normalizes the residual stream after Attention and right before the MLP.',
    status: 'live',
    shapeTag: '[128] γ + [128] β',
    sceneView: 'block',
    sceneFocus: 'ln2',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.ln_2.weight',
    children: [],
    copy: {
      simple:
        'After Attention adds its context note to the highway, LayerNorm 2 levels the numbers once more so the MLP receives clean, steady inputs.',
      sufficient:
        'Standardizes u_l = h_l + Attn(LN_1(h_l)) per token before the Feed-Forward Network using its own learned scale and bias vectors.',
      complete:
        'blocks.{L}.ln_2.weight and .bias (256 params). Click any element below to inspect a single LayerNorm scale value.',
    },
  },

  // MLP / FEED-FORWARD NETWORK
  mlp: {
    id: 'mlp',
    parentId: 'transformer-stack',
    title: 'Feed-Forward Network / MLP (128 → 512 → 128)',
    shortTitle: 'MLP / FFN',
    oneLiner: 'Per-token computation and pattern memory—holding two-thirds of every block’s parameters.',
    status: 'live',
    shapeTag: '131,072 params / block',
    sceneView: 'mlp',
    labKind: 'none',
    children: ['w-up', 'gelu', 'w-down', 'swiglu-moe'],
    copy: {
      simple:
        'If Attention is where words talk to each other, the MLP is where each word thinks privately about what it just learned. Each token’s 128 numbers expand 4× wider into 512 specialist neurons, pass through the GELU filter, and fold back down to 128 numbers to add onto the highway.',
      sufficient:
        'The MLP processes every token independently—no cross-token interaction happens here. Research shows MLP layers act like massive key-value memories storing learned facts, idioms, and feature combinations across 4× hidden dimensions (512 here).',
      complete:
        'MLP(u) = W_down · GELU(LN_2(u) W_up^T), with W_up ∈ R^{512×128} and W_down ∈ R^{128×512} (131,072 params per block, 66.5% of block weights).',
    },
  },

  'w-up': {
    id: 'w-up',
    parentId: 'mlp',
    title: 'MLP Up-Projection (W_up)',
    shortTitle: 'W_up [512×128]',
    oneLiner: 'Expands each token 4× wider into 512 pattern-detector neurons.',
    status: 'live',
    shapeTag: '[512, 128] · 65,536 params',
    sceneView: 'mlp',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.mlp.W_up.weight',
    children: [],
    copy: {
      simple:
        'Each of the 512 rows in this matrix acts like a detector checking for one specific clue (like "is this a past-tense story verb?"). Multiplying against the token gives 512 raw detector scores.',
      sufficient:
        'Projects each 128-dim token into a wider 512-dim workspace where overlapping features can separate cleanly before non-linear activation.',
      complete:
        'blocks.{L}.mlp.W_up.weight ∈ R^{512×128}, bias=False. Click any pixel below to jump straight to that scalar weight.',
    },
  },

  gelu: {
    id: 'gelu',
    parentId: 'mlp',
    title: 'GELU Non-Linear Activation (512 Neurons)',
    shortTitle: 'GELU Activation',
    oneLiner: 'Smoothly suppresses negative/weak neurons and creates the non-linearity neural nets need.',
    status: 'live',
    shapeTag: '[T, 512] sparse activations',
    sceneView: 'mlp',
    sceneFocus: 'gelu',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Without a curve like GELU, stacking a hundred matrix layers would mathematically collapse into just one simple grid—unable to learn real language logic. GELU gently turns off negative neurons (gray bars in 3D) while letting strongly matched neurons blaze bright amber.',
      sufficient:
        'Gaussian Error Linear Unit (GELU) weights inputs by their Gaussian CDF probability, smoothly zeroing out negative pre-activations while keeping positive signals nearly linear. Notice in the 3D bar profile how only a subset of the 512 neurons fire strongly for any given token.',
      complete:
        'GELU_tanh(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))). Exact tanh approximation matched bit-for-bit between PyTorch and browser TypeScript.',
    },
  },

  'w-down': {
    id: 'w-down',
    parentId: 'mlp',
    title: 'MLP Down-Projection (W_down)',
    shortTitle: 'W_down [128×512]',
    oneLiner: 'Writes each firing neuron’s learned output vector back into the 128-dim Residual Stream.',
    status: 'live',
    shapeTag: '[128, 512] · 65,536 params',
    sceneView: 'mlp',
    labKind: 'weight-matrix',
    paramTemplate: 'blocks.{L}.mlp.W_down.weight',
    children: [],
    copy: {
      simple:
        'Every neuron that survived GELU now gets to cast its vote. W_down collects all 512 votes, shrinks them back to 128 numbers, and adds that final update onto the central Residual Stream.',
      sufficient:
        'Each column k of W_down [128, 512] is a 128-dim value vector added into the residual stream scaled by neuron k’s GELU activation—completing the MLP key-value memory lookup.',
      complete:
        'blocks.{L}.mlp.W_down.weight ∈ R^{128×512}, initialized with std = 0.02 / √(2·n_layers). Click any cell below to inspect a single scalar.',
    },
  },

  'swiglu-moe': {
    id: 'swiglu-moe',
    parentId: 'mlp',
    title: 'SwiGLU & Mixture of Experts (MoE)',
    shortTitle: 'SwiGLU & MoE (Modern)',
    oneLiner: 'How frontier LLMs gate MLPs (SwiGLU) and route tokens to sparse specialist experts (MoE).',
    status: 'extension',
    shapeTag: 'Router → Top-k Experts',
    sceneView: 'mlp',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Modern giant LLMs upgrade this MLP in two big ways: SwiGLU adds a second "dimmer switch" path that controls how much of each neuron gets through, and Mixture of Experts (MoE) replaces one giant MLP with dozens of smaller specialist MLPs—so each word only wakes up the 2 specialists it needs.',
      sufficient:
        'SwiGLU computes (SiLU(x W_gate) ⊙ (x W_up)) W_down, consistently outperforming GELU MLPs per FLOP. Mixture of Experts places a tiny learned Router before 8–256 separate FFNs so a trillion-parameter model only activates ~5–15% of its parameters per token—giving massive capacity at fast inference speed.',
      complete:
        'Sparse MoE layer: y = Σ_{e ∈ TopK(softmax(x W_r))} g_e(x) · SwiGLU_e(x), trained with auxiliary load-balancing loss (or router bias updates) to prevent expert collapse.',
    },
  },

  'logit-lens': {
    id: 'logit-lens',
    parentId: 'transformer-stack',
    title: 'Logit Lens (Layer-by-Layer Prediction)',
    shortTitle: 'Logit Lens',
    oneLiner: 'Peeks at the model’s evolving best guess after Block 0, 1, 2, and 3.',
    status: 'live',
    shapeTag: '4 live probes',
    sceneView: 'overview',
    labKind: 'logit-lens',
    children: [],
    copy: {
      simple:
        'Because the Residual Stream stays in the exact same 128-number language from start to finish, we can sneak a peek right after each block to see what the model would guess if we stopped early. Watch the guess sharpen from blurry to confident across the 4 blocks.',
      sufficient:
        'Applying Final LayerNorm and the tied output matrix wte^T directly to the residual stream after each intermediate block reveals iterative refinement: early layers often guess generic grammatical continuations, while later layers sharpen onto context-specific tokens.',
      complete:
        'LogitLens(l) = softmax(LN_f(h_{l+1, T-1}) W_E^T) for l ∈ {0..3}. Computed without altering forward activations.',
    },
  },

  // 4. OUTPUT, SAMPLING & DECODING
  'output-stage': {
    id: 'output-stage',
    parentId: 'root',
    title: '4. Output Head, Sampling & Inference Engine',
    shortTitle: 'Output & Decoding',
    oneLiner: 'Turns the final 128-number vector into scores over 512 tokens, picks one, and loops.',
    status: 'live',
    shapeTag: '[128] → [512] probs',
    sceneView: 'output',
    labKind: 'sampling',
    children: ['ln-f', 'lm-head', 'sampling', 'autoregressive'],
    copy: {
      simple:
        'After the 4th block finishes, the model turns the last word’s 128 numbers into a score for every single word piece in its 512-piece dictionary, converts those scores into percentages, picks one winner, and repeats the whole machine to write the next word.',
      sufficient:
        'The output stage applies Final LayerNorm, multiplies by the unembedding matrix (lm_head) to yield 512 raw Logits, converts logits to probabilities with Temperature/Top-k/Top-p sampling, and appends the sampled token back into the input for the next autoregressive pass.',
      complete:
        'logits = LN_f(h_4[T-1]) W_E^T ∈ R^{512}. The Transformer forward pass is 100% deterministic; stochasticity enters exclusively during decoding.',
    },
  },

  'ln-f': {
    id: 'ln-f',
    parentId: 'output-stage',
    title: 'Final LayerNorm (ln_f)',
    shortTitle: 'Final Norm (ln_f)',
    oneLiner: 'Final stabilization of the residual stream right before scoring the 512 vocabulary tokens.',
    status: 'live',
    shapeTag: '[128] γ + [128] β',
    sceneView: 'output',
    labKind: 'weight-matrix',
    paramTemplate: 'ln_f.weight',
    children: [],
    copy: {
      simple:
        'Because each block only normalizes its *side branches*, the main Residual Stream highway itself has never been normalized since the start. Final LayerNorm cleans up the highway one last time right before the finish line.',
      sufficient:
        'Required in Pre-LN architectures to bring the accumulated residual stream sum back to unit variance before taking dot products against all 512 vocabulary embedding rows.',
      complete:
        'ln_f.weight and ln_f.bias (256 params). Click any entry below to inspect a single scale weight.',
    },
  },

  'lm-head': {
    id: 'lm-head',
    parentId: 'output-stage',
    title: 'Unembedding Head (lm_head) & Raw Logits',
    shortTitle: 'Unembedding & Logits',
    oneLiner: 'Computes 512 raw signed scores (logits)—one for every token in the vocabulary.',
    status: 'live',
    shapeTag: 'Tied wteᵀ → [512] logits',
    sceneView: 'output',
    labKind: 'sampling',
    children: [],
    copy: {
      simple:
        'The model takes the final 128-number summary and compares it against all 512 rows of its dictionary. Tokens that match well get high positive scores (logits); tokens that make no sense get negative scores.',
      sufficient:
        'Unembedding multiplies LN_f(h_last) [128] against W_U [512, 128] (weight-tied directly to W_E). The resulting 512 raw unbounded numbers are called Logits—not probabilities yet.',
      complete:
        'z_v = LN_f(h_{4,T-1}) · W_E[v] for v ∈ {0..511}. Weight tying saves 65,536 parameters and aligns input/output embedding geometry.',
    },
  },

  sampling: {
    id: 'sampling',
    parentId: 'output-stage',
    title: 'Softmax, Temperature, Top-k & Top-p',
    shortTitle: 'Softmax & Sampling',
    oneLiner: 'Converts logits to probabilities and controls how predictable or creative the chosen token is.',
    status: 'live',
    shapeTag: 'τ · top-k · top-p',
    sceneView: 'output',
    labKind: 'sampling',
    children: [],
    copy: {
      simple:
        'Why can the exact same prompt give different answers? Because of this final step. Low temperature makes the model play it safe and always pick the tallest blue bar. High temperature flattens the bars so surprise runners-up get picked more often. Try the sliders below and watch the 3D bars react live.',
      sufficient:
        'Dividing logits by Temperature τ before Softmax sharpens (τ < 1) or flattens (τ > 1) the distribution. Top-k keeps only the k highest bars; Top-p (Nucleus sampling) keeps only the smallest group of top tokens whose probabilities add up to p (e.g., 90%), cutting off the noisy tail.',
      complete:
        'P_τ(v) = exp(z_v / τ) / Σ_{u ∈ Nucleus(k, p)} exp(z_u / τ). Excluded bars render muted gray on the 3D skyline.',
    },
  },

  autoregressive: {
    id: 'autoregressive',
    parentId: 'output-stage',
    title: 'Autoregressive Loop, KV Cache & Quantization',
    shortTitle: 'Loop & Inference',
    oneLiner: 'How one next-token prediction loops to write full paragraphs—and how real servers make it fast.',
    status: 'live',
    shapeTag: 'Append & repeat',
    sceneView: 'output',
    labKind: 'sampling',
    children: ['kv-cache', 'quantization'],
    copy: {
      simple:
        'An LLM never writes a whole paragraph in one shot. It predicts ONE token, glues that token onto the end of your sentence, and runs the whole machine all over again for the next token. Click "Step +1 Token" below to watch our real model write one step at a time.',
      sufficient:
        'Text generation is an autoregressive loop: feed x_{1..T}, sample x_{T+1}, append it to form x_{1..T+1}, and repeat until reaching an <|eos|> stop token or max length. Production engines accelerate this loop using KV Caching and low-bit Quantization.',
      complete:
        'Sequential decoding is memory-bandwidth bound on modern accelerators because every generated token must stream all model weights from HBM unless batched.',
    },
  },

  'kv-cache': {
    id: 'kv-cache',
    parentId: 'autoregressive',
    title: 'Prefill vs. Decode & KV Cache',
    shortTitle: 'KV Cache (Modern)',
    oneLiner: 'Stores previously computed Key and Value vectors so generation does not redo past work.',
    status: 'extension',
    shapeTag: 'O(T) instead of O(T²)',
    sceneView: 'attention',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'When you send a long prompt, the model reads all those words at once (Prefill). Then, as it writes its reply one word at a time (Decode), all the earlier words’ Keys and Values never change—so the server saves them in a notepad called the KV Cache instead of recalculating them over and over.',
      sufficient:
        'Because causal masking prevents earlier tokens from attending to newly generated tokens, their K and V tensors at every layer stay frozen. Caching [layers, heads, seq, d_head] turns each new decode step from O(T²) recomputation into O(T) attention against the cache.',
      complete:
        'KV cache memory = 2 · n_layers · n_kv_heads · d_head · T · bytes_per_elem per active sequence. At 128k+ context, KV cache often rivals or exceeds model weights in GPU VRAM, motivating GQA, MLA, and PagedAttention.',
    },
  },

  quantization: {
    id: 'quantization',
    parentId: 'autoregressive',
    title: 'Numeric Precision & Quantization',
    shortTitle: 'Precision (FP16/INT4)',
    oneLiner: 'Why storing weights in 16-bit, 8-bit, or 4-bit numbers shrinks models without ruining intelligence.',
    status: 'live',
    shapeTag: 'This model: FP16 (1.65 MB)',
    sceneView: 'overview',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Instead of storing every weight with 15 decimal digits, we can round each number slightly to take half or a quarter as much memory. Your miniature model right now is stored in 16-bit half precision (FP16)—just 1.65 MB for all 862,464 weights!',
      sufficient:
        'Models typically train in BF16/FP32 mixed precision, then ship for inference in FP16, FP8, INT8, or INT4 quantization. Because neural network weights are redundant and resilient to tiny rounding noise, 4-bit quantization cuts memory by ~4× with minimal quality loss.',
      complete:
        'This app decodes little-endian IEEE 754 binary16 (weights.fp16.bin, 2 bytes/param) into Float32Array in src/engine/fp16.ts. Parity verification proves max output deviation <1e-4 against PyTorch.',
    },
  },

  // 5. TRAINING LIFECYCLE
  'training-lifecycle': {
    id: 'training-lifecycle',
    parentId: 'root',
    title: '5. How an LLM Learns (Training & Alignment)',
    shortTitle: 'How It Learns',
    oneLiner: 'Where all 862,464 weights actually came from—and how raw models become helpful assistants.',
    status: 'live',
    shapeTag: 'Step 0 → Step 2,000',
    sceneView: 'overview',
    labKind: 'replay',
    children: ['pretraining', 'loss-perplexity', 'backprop', 'post-training'],
    copy: {
      simple:
        'Nobody hand-typed a single rule of English grammar into this model. At Step 0, all 862,464 numbers were pure random static and the model babbled nonsense. By guessing the next word across thousands of short stories and nudging its numbers to be slightly less wrong each time, language emerged automatically.',
      sufficient:
        'Building a real LLM happens in two major chapters: Pre-Training (self-supervised next-token prediction via Cross-Entropy Loss, Backpropagation, and AdamW across massive corpora) followed by Post-Training (Supervised Fine-Tuning and RLHF / Reasoning RL to align behavior).',
      complete:
        'Pre-training minimizes empirical risk L(θ) = E[-log P_θ(x_t | x_{<t})] via stochastic gradient descent (AdamW + cosine LR decay). Scrub the real checkpoints from this model’s training run below.',
    },
  },

  pretraining: {
    id: 'pretraining',
    parentId: 'training-lifecycle',
    title: 'Pre-Training & Real Training Replay',
    shortTitle: 'Pre-Training Replay',
    oneLiner: 'Scrub through our actual recorded training checkpoints from random gibberish to stories.',
    status: 'live',
    shapeTag: 'TinyStories · 10 checkpoints',
    sceneView: 'overview',
    labKind: 'replay',
    children: [],
    copy: {
      simple:
        'Drag the slider below from Step 0 to the final step. You are watching real snapshots saved while this exact miniature AI trained on GitHub’s cloud server: at Step 0 it outputs random letters, and step by step the loss drops and real children’s stories appear.',
      sufficient:
        'Every training step samples a batch of 64-token windows from TinyStories, runs a forward pass, measures next-token error at all 63 positions in parallel, backpropagates gradients, and updates all 862,464 parameters.',
      complete:
        'Loaded directly from public/model/training_replay.json: validation loss starts near ln(512) ≈ 6.24 (uniform random baseline) and descends smoothly under cosine-decayed AdamW.',
    },
  },

  'loss-perplexity': {
    id: 'loss-perplexity',
    parentId: 'training-lifecycle',
    title: 'Cross-Entropy Loss & Perplexity',
    shortTitle: 'Loss & Perplexity',
    oneLiner: 'The exact scorecard measuring how surprised the model was by the true next token.',
    status: 'live',
    shapeTag: 'L = -log P(true)',
    sceneView: 'output',
    labKind: 'replay',
    children: [],
    copy: {
      simple:
        'If the true next word was "ran" and the model gave "ran" a 90% chance, its penalty (loss) is tiny. If it only gave "ran" a 1% chance, its penalty is huge! Perplexity is simply how many equally-likely words the model is effectively torn between.',
      sufficient:
        'Cross-entropy loss L = -log P(x_true) penalizes confident wrong predictions exponentially. Perplexity = exp(L): at Step 0 with 512 tokens, L ≈ 6.24 gives perplexity ≈ 512 (blind guessing); at L = 2.0, perplexity drops to ~7.4.',
      complete:
        'Teacher-forced sequence loss L = -(1/(T-1)) Σ_{t=0}^{T-2} log softmax(logits_t)[x_{t+1}]. Live prompt loss is computed directly in src/engine/transformer.ts.',
    },
  },

  backprop: {
    id: 'backprop',
    parentId: 'training-lifecycle',
    title: 'Backpropagation, Gradients & AdamW',
    shortTitle: 'Backprop & AdamW',
    oneLiner: 'How error flows backward through the chain rule to nudge every single scalar weight.',
    status: 'live',
    shapeTag: '∂L / ∂w → nudge',
    sceneView: 'overview',
    labKind: 'replay',
    children: [],
    copy: {
      simple:
        'Once the model measures how wrong its guess was, it works backward from the finish line to the start, asking every single weight: "If I nudge you slightly up or down, does the mistake get smaller?" The AdamW optimizer then gives all 862,464 numbers a tiny, careful nudge in the winning direction.',
      sufficient:
        'Backpropagation applies reverse-mode automatic differentiation (the calculus chain rule) from the scalar loss back through every matmul, GELU, Softmax, and LayerNorm to compute ∂L/∂w for every weight. AdamW tracks momentum and adaptive per-weight step sizes with weight decay and gradient clipping.',
      complete:
        'Trained with AdamW (lr=3e-3, β_1=0.9, β_2=0.95, weight_decay=0.05, grad_clip=1.0, 5% warmup + cosine decay to 0.1×). Click any tracked scalar below to inspect its exact Step 0 → Final path.',
    },
  },

  'post-training': {
    id: 'post-training',
    parentId: 'training-lifecycle',
    title: 'Post-Training: SFT, RLHF & Reasoning RL',
    shortTitle: 'SFT & RLHF (Modern)',
    oneLiner: 'How a raw internet text predictor is shaped into a safe, instruction-following thinker.',
    status: 'extension',
    shapeTag: 'SFT → RLHF / DPO / RL',
    sceneView: 'output',
    labKind: 'none',
    children: [],
    copy: {
      simple:
        'Pre-training teaches a model how language works, like reading a library—but it doesn’t yet know how to answer questions politely or check its math. Post-training shows it thousands of great Q&A examples (SFT), then rewards helpful, truthful, well-reasoned answers using reinforcement learning (RLHF and Reasoning RL).',
      sufficient:
        'Modern frontier LLMs undergo multi-stage post-training after pretraining: (1) Supervised Fine-Tuning (SFT) on high-quality dialogue/instruction demonstrations, (2) Preference Alignment (RLHF / DPO) using human or AI preference pairs for helpfulness and safety, and (3) Verifiable-Reward RL to train multi-step chain-of-thought reasoning.',
      complete:
        'Post-training updates the same Transformer weights θ without changing architecture—shifting the output distribution from raw corpus mimicry toward policy π_θ(y|x) maximizing expected reward R(x, y) under a KL-divergence constraint β·D_KL(π_θ || π_ref).',
    },
  },

  // 6. INTERACTIVE LABS
  experiments: {
    id: 'experiments',
    parentId: 'root',
    title: '6. Interactive Labs: Ablation & Frontier Scale',
    shortTitle: 'Labs & Scale',
    oneLiner: 'Break parts of the live model to prove cause-and-effect, and compare to trillion-parameter scale.',
    status: 'live',
    shapeTag: 'Causal tests + 1T ladder',
    sceneView: 'overview',
    labKind: 'none',
    children: ['ablation', 'scale-ladder'],
    copy: {
      simple:
        'The fastest way to know what a machine part really does is to turn it off and see what breaks. Use the Ablation Lab to unplug heads, MLPs, or shortcuts live, and use the Scale Ladder to see how this tiny model compares to giant frontier AI.',
      sufficient:
        'Interactive causal interventions prove every component in this atlas is genuinely load-bearing, while the Scale Ladder honestly maps what stays identical vs. what changes between 862,464 parameters and frontier scale.',
      complete:
        'Live counterfactual forward passes execute in ~15ms inside the browser Web Worker with exact probability and cross-entropy loss deltas.',
    },
  },

  ablation: {
    id: 'ablation',
    parentId: 'experiments',
    title: 'Surgical Counterfactual Ablation Lab',
    shortTitle: 'Ablation Lab',
    oneLiner: 'Turn off any head, Attention block, MLP, Residual bypass, LayerNorm, or Layer live.',
    status: 'live',
    shapeTag: 'Live before/after diff',
    sceneView: 'block',
    labKind: 'ablation',
    children: [],
    copy: {
      simple:
        'Flip any switch below to unplug that exact part of the running miniature AI. Watch the original blue bars vs. the new orange bars shift instantly. Notice how unplugging one attention head barely changes the answer (redundancy), while unplugging a Residual shortcut or LayerNorm wrecks it.',
      sufficient:
        'Causal ablation zeroes selected head outputs, sub-blocks, or bypasses LayerNorm/residual additions and re-runs the full forward pass live. Comparing before/after probabilities and input cross-entropy loss reveals head specialization and architectural load-bearing paths.',
      complete:
        'Interventions execute directly inside runForwardTrace() in src/engine/transformer.ts, verified against PyTorch ablation traces in scripts/verify-parity.ts.',
    },
  },

  'scale-ladder': {
    id: 'scale-ladder',
    parentId: 'experiments',
    title: 'Scale Ladder (862k → 1T+ Parameters)',
    shortTitle: 'Scale Ladder',
    oneLiner: 'Compare this 862,464-parameter miniature Transformer against GPT-2, 8B, 70B, and Frontier LLMs.',
    status: 'live',
    shapeTag: '1× → 1,159,000×',
    sceneView: 'overview',
    labKind: 'scale',
    children: [],
    copy: {
      simple:
        'This miniature AI runs on the exact same heartbeat as the largest AI systems on Earth—just shrunk down more than a million times so you can see every gear. Drag the slider below to feel the jump in size, and see honestly what new abilities wake up at huge scale.',
      sufficient:
        'Scaling laws show predictable loss drops as parameters N, training tokens D, and compute C grow together. While the residual Attention + FFN backbone remains identical, scale unlocks broad factual recall, coding, in-context learning, and multi-step reasoning absent at 862k params.',
      complete:
        'Compares exact architectural dimensions (layers, d_model, heads, context window) and modern upgrades (RoPE, RMSNorm, GQA, SwiGLU, MoE, post-training) side-by-side.',
    },
  },

  // SINGLE SCALAR LEAF (Reached by clicking any cell in any weight matrix)
  scalar: {
    id: 'scalar',
    parentId: 'w-k',
    title: 'Single Scalar Parameter (The Atom of an LLM)',
    shortTitle: 'Single Scalar',
    oneLiner: 'One real floating-point number decoded directly from weights.fp16.bin—proving nothing is faked.',
    status: 'live',
    shapeTag: '1 scalar float',
    sceneView: 'block',
    labKind: 'scalar',
    children: [],
    copy: {
      simple:
        'This is one single number inside the model. Nobody typed it in. At Step 0 it started as a random guess near zero. During training, the model tried to predict thousands of short stories, measured its mistake, and nudged this number—along with 862,463 others—slightly in the direction that made it less wrong. That is the entire origin story of every number in an LLM.',
      sufficient:
        'One exact learned scalar parameter θ_{r,c} read verbatim from weights.fp16.bin. The histogram shows where it sits among sibling weights in the same matrix, and for the 12 checkpoint-tracked coordinates, the curve plots its actual AdamW trajectory across training.',
      complete:
        'Scalar parameter θ_{r,c} initialized from N(0, 0.02²) (or 0.02/√8 on residual projections), updated via AdamW under next-token cross-entropy, quantized to IEEE 754 float16 for static browser delivery.',
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
    note: '1.65 MB FP16. Same core Transformer math; writes simple story clauses, no world encyclopedia.',
  },
  {
    label: 'GPT-2 Small (2019)',
    params: 124_000_000,
    layers: 12,
    dModel: 768,
    heads: 12,
    context: 1024,
    note: '144× larger. Holds coherent multi-paragraph topics and basic factual patterns.',
  },
  {
    label: '8B Open-Weight Class',
    params: 8_000_000_000,
    layers: 32,
    dModel: 4096,
    heads: 32,
    context: 8192,
    note: '9,275× larger. Uses RoPE, GQA, RMSNorm, SwiGLU; capable of coding and broad world recall.',
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
    note: 'Over 1.15M× larger. Typically uses sparse Mixture-of-Experts routing while preserving the residual Attention + FFN heartbeat.',
  },
];

export function getAnatomyNode(id: string): AnatomyNode {
  return ANATOMY_NODES[id] ?? ANATOMY_NODES.root;
}

export function getBreadcrumbChain(nodeId: string): AnatomyNode[] {
  const chain: AnatomyNode[] = [];
  let cur: AnatomyNode | undefined = getAnatomyNode(nodeId);
  const seen = new Set<string>();
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    chain.unshift(cur);
    cur = cur.parentId ? ANATOMY_NODES[cur.parentId] : undefined;
  }
  return chain;
}

export function resolveParamName(
  node: AnatomyNode,
  layer: number,
  fallback: string
): string {
  if (!node.paramTemplate) return fallback;
  return node.paramTemplate.replace('{L}', String(layer));
}

export function registerDisplayName(r: Register): string {
  if (r === 'simple') return 'Extremely Simplified';
  if (r === 'sufficient') return 'Sufficient';
  return 'Complete';
}
