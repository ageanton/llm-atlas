import type { BookBeat } from './bookData';

export const CHAPTER_1_BEATS: BookBeat[] = [
  {
    id: 'ch1-beat-1',
    beatNumber: 1,
    shortTitle: '1. Two Extremes',
    heading: 'Why Neither Letters Nor Whole Words Work',
    sceneBeat: 'ch1-1-extremes',
    visualFocusNote:
      'Compare the two stacked rows on the 3D stage: tiny single characters in the back versus subword tokens in the front.',
    paragraphs: {
      foundations: [
        'A neural network is pure arithmetic—it can only multiply and add numbers, never raw English letters. So before an LLM can read your sentence, we have to chop the text into pieces and give each piece a number.',
        'At first you might try two obvious extremes. Extreme 1: chop text into individual letters ("O", "n", "c", "e", ...). That keeps the dictionary tiny, but even one short sentence turns into 60 tiny meaningless pieces, forcing the model to do far more work to see whole ideas.',
        'Extreme 2: give every whole word in English its own number. That immediately breaks on typos ("dogg"), names ("Maxzilla"), code, or compound words. Modern LLMs invent the sweet spot in between: subword tokens.',
      ],
      practitioner: [
        'Sequence length T directly governs Self-Attention cost, which scales as O(T²) in compute and O(T) in KV-cache memory per layer. Character-level tokenization inflates T by ~4×, starving context length.',
        'Word-level vocabularies suffer from severe out-of-vocabulary (<|unk|>) collapse and bloat the embedding matrix W_E [|V|, d_model] into hundreds of millions of rarely updated rows. Subword tokenization (BPE / Unigram) compresses frequent words into single IDs while gracefully falling back to smaller stems or characters for unseen strings.',
      ],
      formal: [
        'A tokenizer defines a deterministic lossless encoder E: Σ* → V^T trading off average sequence compression ratio (chars/token) against vocabulary cardinality |V|. Optimal subword segmentation minimizes sequence length T subject to bounded embedding parameter budget |V| · d_model and zero OOV rate over base alphabet Σ_0.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Subword tokens keep common words whole while splitting rare or new words into familiar pieces.',
      practitioner:
        'Subword tokenization balances O(T²) sequence length against |V| · d_model embedding memory with zero OOV failures.',
      formal:
        'Subword vocabularies bridge character-level coverage and word-level semantic density under fixed |V|.',
    },
  },
  {
    id: 'ch1-beat-2',
    beatNumber: 2,
    shortTitle: '2. Live BPE Merges',
    heading: 'Watch Byte-Pair Encoding Glue Letters Together',
    sceneBeat: 'ch1-2-bpe-merges',
    visualFocusNote:
      'Use the BPE Merge Step slider on stage to scrub from Step 0 (single characters) forward through our model’s real 414 learned merge rules.',
    paragraphs: {
      foundations: [
        'How does an LLM decide which chunks belong in its dictionary? Before training the neural network, we run a simple counting recipe called Byte-Pair Encoding (BPE) over our training stories.',
        'Start by chopping every story into single letters. Find the two neighbors that appear side-by-side most often—for example, "t" and "h"—and glue them into a brand-new tile "th". Repeat that merge hundreds of times (414 merges in our miniature model; ~100,000 in frontier LLMs).',
        'Drag the Merge Step slider on the 3D stage from Step 0 to the final step to watch your own sentence compress in real time, highlighting the exact newly glued tiles in warm amber at every step.',
      ],
      practitioner: [
        'Our miniature model uses a 512-token BPE vocabulary trained on TinyStories: 98 base characters/special tokens plus 414 rank-ordered merge rules stored in public/model/tokenizer.json.',
        'During encoding, each pre-tokenized regex chunk starts as single characters and greedily contracts whichever adjacent pair has the lowest merge rank until no valid pairs remain. Notice how fast the token count drops in the first dozen merges.',
      ],
      formal: [
        'Given pre-tokenized strings decomposed into base symbols Σ_0 and ordered merge table M = [(a_r, b_r)]_{r=0}^{K-1}, greedy BPE iteratively contracts argmin_{i} rank(s_i, s_{i+1}) until no adjacent pair belongs to M. Our TinyBPETokenizer matches the Python reference encoder token-for-token.',
      ],
    },
    keyTakeaway: {
      foundations:
        'BPE builds a dictionary automatically by repeatedly gluing the most common neighboring letters together.',
      practitioner:
        'Rank-ordered greedy BPE merges compress raw characters into subword IDs deterministically before the forward pass.',
      formal:
        'Iterative frequency-ranked pair contraction constructs a compact prefix-free-like subword code over Σ*.',
    },
  },
  {
    id: 'ch1-beat-3',
    beatNumber: 3,
    shortTitle: '3. The "Strawberry" Blindspot',
    heading: 'Spaces, Chunks, and Why LLMs Miscount Letters',
    sceneBeat: 'ch1-3-spaces-blindspot',
    visualFocusNote:
      'Each 3D tile shows exact visible space markers (·) and character widths—tall amber tiles hide multiple letters inside a single atomic token.',
    paragraphs: {
      foundations: [
        'Look closely at the leading dots ("·") on the 3D tiles. A tokenizer does not throw spaces away—it glues the leading space onto the front of the word! That means "·dog" (with a space) and "dog" (at the start of a sentence) are two completely different tokens.',
        'This also explains one of the most famous quirks of modern AI: why LLMs often fail to count the letter "r" in "strawberry" or spell words backward. Once "straw" and "berry" become single solid integer blocks, the neural network literally never receives the individual letters—it only sees two solid IDs.',
        'Try typing "strawberry" or "unbelievable" into the top prompt bar right now to watch how our real miniature tokenizer slices it.',
      ],
      practitioner: [
        'Regex pre-tokenization (` ?[A-Za-z]+|...`) binds leading whitespace to words so decoding is lossless without post-hoc spacing heuristics, while preventing merges from crossing word/punctuation boundaries.',
        'Because multi-character tokens enter the network as atomic embedding lookups, character-level tasks (spelling, counting substrings, cipher decoding) require the model to memorize spelling facts inside its weights rather than reading characters directly.',
      ],
      formal: [
        'Subword tokenization acts as a lossy inductive bias with respect to character-level morphology: once substring u ∈ Σ^m maps to atomic index v ∈ V, character-composition properties of u must be recovered indirectly from the learned representation W_E[v].',
      ],
    },
    keyTakeaway: {
      foundations:
        'Leading spaces are part of tokens, and multi-letter chunks hide individual letters from the model.',
      practitioner:
        'Atomic subword IDs explain whitespace sensitivity and character-counting failure modes in LLMs.',
      formal:
        'Subword atomicity decouples surface orthography from the input basis seen by the Transformer.',
    },
    predict: {
      id: 'ch1-b3-predict',
      question:
        'Why do "cat" (at the start of a sentence) and " cat" (in the middle of a sentence) usually get different Token IDs?',
      options: [
        {
          id: 'a',
          label: 'Because BPE attaches the leading space to the word so the tokenizer can reconstruct exact spacing when decoding.',
        },
        {
          id: 'b',
          label: 'Because LLMs treat the first word of every sentence as a special uppercase command.',
        },
        {
          id: 'c',
          label: 'They always share the exact same ID; spaces are deleted before tokenization.',
        },
      ],
      correctId: 'a',
      explanation:
        'Exactly right. Keeping the leading space attached (" cat") makes text-to-tokens-to-text cleanly reversible.',
    },
  },
  {
    id: 'ch1-beat-4',
    beatNumber: 4,
    shortTitle: '4. Integer IDs',
    heading: 'Numbers Without Meaning (Yet)',
    sceneBeat: 'ch1-4-token-ids',
    visualFocusNote:
      'Every token tile now reveals its exact integer ID (0–511) from our vocabulary dictionary, plotted by raw ID height.',
    paragraphs: {
      foundations: [
        'Each chunk on your sentence strip is now replaced by its row number in the 512-entry dictionary—an integer like #104 or #277.',
        'Notice a critical trap: the bar heights on stage show raw ID numbers, and they look totally jagged and random! Token #104 is NOT "slightly smaller" than token #105, and #200 is not "twice as big" as #100. An ID is purely an arbitrary locker number—just like house #104 and house #105 have nothing to do with how similar the families inside are.',
        'That exposes our next big bottleneck, which Chapter 2 will solve: how do we turn meaningless locker numbers into real math where similar words live close together?',
      ],
      practitioner: [
        'The tokenizer outputs a 1D integer tensor x of shape [T] with entries in {0, ..., |V|-1}. Feeding raw scalar IDs directly into linear layers would impose a bogus ordinal relationship based purely on BPE merge frequency.',
        'Instead, x is used strictly as a categorical lookup index into the learned embedding matrix W_E in Chapter 2.',
      ],
      formal: [
        'Integer indices x_t ∈ {0, ..., |V|-1} live in a discrete nominal set lacking metric or algebraic structure. Treating x_t as a real scalar induces false monotonicity; lifting x_t via one-hot basis projection e_{x_t} W_E maps symbols into an unconstrained Euclidean space R^{d_model}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Token IDs are arbitrary dictionary locker numbers—neighboring IDs have no mathematical similarity.',
      practitioner:
        'Token IDs [T] are categorical lookup keys, never ordinal numeric features.',
      formal:
        'Discrete token indices carry no Euclidean geometry prior to embedding projection W_E.',
    },
  },
  {
    id: 'ch1-beat-5',
    beatNumber: 5,
    shortTitle: '5. Context Window',
    heading: 'The Hard Desk Limit: Context Length (T_max)',
    sceneBeat: 'ch1-5-context-window',
    visualFocusNote:
      'Your tokens sit inside our miniature model’s physical 64-slot desk. Filled slots glow blue; empty slots wait ahead; anything past slot 64 drops off the edge.',
    paragraphs: {
      foundations: [
        'Every LLM has a fixed maximum desk size called its Context Window—the total number of token slots it can see at once. Our miniature model’s desk holds exactly 64 tokens.',
        'Look at the 64-slot tray on the 3D stage: your current sentence fills the blue slots on the left, leaving the open slots free for the model to keep writing. If you paste a long paragraph with 80 tokens, anything past slot 64 literally falls off the desk and does not exist to the model.',
        'Modern frontier models expand this exact same concept from 64 slots to 128,000 or 1,000,000+ slots—enough to place an entire novel or codebase on the desk at once.',
      ],
      practitioner: [
        'In our GPT-2-style architecture, T_max = 64 is hard-capped by the 64 rows of wpe.weight [64, 128]. Exceeding 64 tokens forces truncation (`tokens.slice(0, 64)`) in src/engine/transformer.ts.',
        'In production LLMs, the context window must fit the System Prompt, Chat Template control tokens (<|user|>, <|assistant|>), conversation history, retrieved documents, AND the model’s generated reply.',
      ],
      formal: [
        'Absolute positional tables strictly bound input length T ≤ T_max. Even with length-extrapolating relative encodings (RoPE + YaRN), effective context remains constrained by quadratic prefill FLOPs O(L · T² · d_model) and linear KV-cache VRAM O(L · T · n_kv · d_head).',
      ],
    },
    keyTakeaway: {
      foundations:
        'The context window is the model’s entire working desk—tokens beyond the limit are completely invisible.',
      practitioner:
        'Our model caps T ≤ 64 via wpe [64, 128]; frontier models scale T to 1M+ within compute and KV-cache limits.',
      formal:
        'Context window T_max defines the finite horizon of causal conditioning x_{1..T}.',
    },
    predict: {
      id: 'ch1-b5-predict',
      question:
        'If you paste a 90-token story into our 64-token miniature model, what happens to tokens 65 through 90?',
      options: [
        {
          id: 'a',
          label: 'They are cut off before entering the model—the Transformer never sees them at all.',
        },
        {
          id: 'b',
          label: 'The model automatically compresses them into smaller letters so all 90 fit.',
        },
        {
          id: 'c',
          label: 'They are saved into the MLP weights for later.',
        },
      ],
      correctId: 'a',
      explanation:
        'Correct. Any token past the context window ceiling is truncated (or slid out of the window) and is completely invisible to the forward pass.',
    },
  },
];

export const CHAPTER_2_BEATS: BookBeat[] = [
  {
    id: 'ch2-beat-1',
    beatNumber: 1,
    shortTitle: '1. Coordinate Book (wte)',
    heading: 'Turning Locker Numbers Into 128-Number Vectors',
    sceneBeat: 'ch2-1-wte-lookup',
    visualFocusNote:
      'On the left stands the full 512×128 Token Embedding table (wte); on the right, your highlighted token’s exact 128-bar vector slides out.',
    paragraphs: {
      foundations: [
        'Chapter 1 left us with meaningless locker numbers like #104. Now the model opens a giant coordinate book called the Token Embedding Matrix (wte). It has 512 rows—one row for every token in the dictionary—and each row holds 128 learned numbers.',
        'Click any token in the bottom strip: its exact row lights up on the tall wte wall and slides out as a 128-bar strip of real floats (amber = positive, blue = negative).',
        'Think of those 128 numbers as 128 subtle sliders describing how that token behaves in stories—animal vs. object, action vs. name, past vs. present—all learned automatically during training without any human labeling them.',
      ],
      practitioner: [
        'Embedding lookup indexes W_E[x_t] from wte.weight [512, 128] (65,536 parameters, 7.6% of our entire model). In frontier LLMs, d_model grows from 128 to 4,096–16,384.',
        'Stacked across all T tokens in your sentence, this lookup produces a dense floating-point tensor tokEmb of shape [T, 128]. Notice this same wte matrix is weight-tied to the final output head at the end of the network.',
      ],
      formal: [
        'The token embedding map Φ_E(x_t) = e_{x_t}^T W_E ∈ R^{128} embeds the discrete vocabulary into a continuous distributed representation. Because W_E is trained end-to-end by next-token cross-entropy, tokens sharing predictive distributions acquire collinear embedding directions.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Each token ID looks up its own learned list of 128 numbers from the wte coordinate book.',
      practitioner:
        'wte[x] converts integer IDs [T] into a continuous activation slab of shape [T, 128].',
      formal:
        'Token embedding lookup lifts categorical indices into distributed Euclidean coordinates in R^{d_model}.',
    },
  },
  {
    id: 'ch2-beat-2',
    beatNumber: 2,
    shortTitle: '2. Real 3D Embedding Cloud',
    heading: 'Meaning as Geometry: Inside Our Model’s Real PCA Space',
    sceneBeat: 'ch2-2-pca-cloud',
    visualFocusNote:
      'Every sphere is one of our model’s REAL 512 vocabulary tokens, projected from 128D to 3D via exact PCA! Click any token in the bottom strip to spotlight its top-5 real Cosine Similarity neighbors in warm amber.',
    paragraphs: {
      foundations: [
        'We cannot picture 128 dimensions directly—but we can project our trained model’s REAL 512 token vectors down into 3D space using Principal Component Analysis (PCA) right here on stage!',
        'Orbit around this constellation of 512 real learned points. The blue glowing sphere is your selected token, and the warm amber spheres connected to it are its closest neighbors measured by Cosine Similarity (how closely two arrows point in the same direction in full 128D space).',
        'Nobody told the model that nouns, verbs, or character names belong together—yet purely from predicting children’s stories, words with similar jobs naturally drifted into the same neighborhoods.',
      ],
      practitioner: [
        'This 3D scatter cloud is computed live in your browser from public/model/weights.fp16.bin via centered covariance eigendecomposition (PC1, PC2, PC3).',
        'Crucially, the neighbor rankings printed below are computed in the full unprojected 128-dimensional space using exact cosine similarity: cos(u, v) = (u · v) / (||u|| ||v||), avoiding 3D projection distortion.',
      ],
      formal: [
        'Let X_c = W_E - 1 μ^T ∈ R^{512×128}. We extract the leading three principal eigenvectors v_1, v_2, v_3 of (1/N) X_c^T X_c via deflated power iteration to visualize global embedding topology, while evaluating exact semantic proximity via angular metric cos(w_i, w_j) in R^{128}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'In an LLM, "similar meaning" literally means pointing in a similar direction in high-dimensional space.',
      practitioner:
        'Training organizes wte into geometric clusters measurable by full-dimensional cosine similarity.',
      formal:
        'Distributional semantics emerges as angular alignment in R^{d_model}, visible globally through top-3 PCA projection.',
    },
    predict: {
      id: 'ch2-b2-predict',
      question:
        'Why do ML engineers usually measure embedding similarity with Cosine Similarity (angle) rather than raw Euclidean distance alone?',
      options: [
        {
          id: 'a',
          label: 'Because meaning is primarily encoded by direction, whereas raw vector length (norm) often varies with token frequency.',
        },
        {
          id: 'b',
          label: 'Because Euclidean distance cannot be computed in more than 3 dimensions.',
        },
        {
          id: 'c',
          label: 'Because all embedding vectors are forced to have length 1.0 by the tokenizer.',
        },
      ],
      correctId: 'a',
      explanation:
        'Spot on! High-frequency tokens and rare tokens often settle at different lengths, while semantic similarity aligns along direction (angle).',
    },
  },
  {
    id: 'ch2-beat-3',
    beatNumber: 3,
    shortTitle: '3. Order-Blind Problem',
    heading: 'Why Embeddings Alone Cannot Tell Word Order',
    sceneBeat: 'ch2-3-order-blind',
    visualFocusNote:
      'Look at the two sentences floating side-by-side: identical token vectors rearranged in a different order look like the exact same unordered bag to Attention.',
    paragraphs: {
      foundations: [
        'Pause and spot a sneaky flaw we just created. Compare "The dog chased the ball" with "The ball chased the dog".',
        'Both sentences contain the exact same five tokens—so looking them up in wte produces the EXACT same five 128-number vectors! If we hand those vectors straight to Self-Attention, Attention compares every pair symmetrically and has zero clue which word came first and which came last.',
        'To understand grammar, cause and effect, and storytelling, the model must stamp every token vector with its position in the sentence.',
      ],
      practitioner: [
        'Without positional information, the Transformer block (MHA + per-token MLP + LayerNorm) is strictly permutation-equivariant: permuting the input rows simply permutes the output rows identically.',
        'Unlike RNNs (which baked order into sequential hidden-state recurrence), Transformers process all T tokens in parallel—so positional geometry must be injected explicitly.',
      ],
      formal: [
        'For any permutation matrix P ∈ {0,1}^{T×T}, an unmasked Transformer without positional encoding satisfies F(P X) = P F(X). Breaking permutation equivariance requires injecting index-dependent structure into either the input stream (absolute PE) or the attention kernel (relative PE / RoPE).',
      ],
    },
    keyTakeaway: {
      foundations:
        'Without position stamps, a Transformer treats a sentence like a shuffled bag of words.',
      practitioner:
        'Self-Attention is inherently permutation-equivariant, so order must be injected explicitly.',
      formal:
        'Breaking permutation symmetry F(PX) = PF(X) is mandatory for sequence modeling.',
    },
  },
  {
    id: 'ch2-beat-4',
    beatNumber: 4,
    shortTitle: '4. Stamping Position (wpe)',
    heading: 'Merging Meaning + Position Into the Starting Stream (h₀)',
    sceneBeat: 'ch2-4-wpe-add',
    visualFocusNote:
      'Watch our model’s real Token Embedding slab (wte) and Positional Embedding slab (wpe) add element-by-element into the starting Residual Stream h₀ [T, 128].',
    paragraphs: {
      foundations: [
        'Our miniature model solves word order the exact way GPT-2 did: it keeps a second tiny coordinate book called Positional Embeddings (wpe), with 64 learned rows—one 128-number tag for Slot 0, Slot 1, Slot 2, all the way to Slot 63.',
        'For every word in your sentence, the model simply adds its 128 meaning numbers + its 128 slot numbers element-by-element. That combined grid of numbers—h₀—is the official starting slab that rides into Block 0.',
        'Wait: doesn’t adding meaning and position together scramble them? Surprisingly, no! In 128 dimensions (or 4,096 in big LLMs), there is so much elbow room that different subspaces carry meaning and position side-by-side without stepping on each other.',
      ],
      practitioner: [
        'Computing h_0 = wte[x] + wpe[0..T-1] adds [T, 128] + [T, 128] → [T, 128] with zero extra width (unlike concatenation, which would expand d_model).',
        'The limitation of learned absolute wpe is clear: wpe has only 64 trained rows, so the model cannot generalize cleanly to slot 65, and "Max" at slot 5 vs. slot 50 gets completely different learned offsets.',
      ],
      formal: [
        'H_0[t] = W_E[x_t] + W_P[t] ∈ R^{128}, where W_P ∈ R^{64×128} (8,192 params). Additive superposition exploits near-orthogonality of random subspaces in high dimensions while rigidly bounding T ≤ T_max.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Adding each word’s meaning vector and slot vector creates the [T, 128] starting slab entering Block 0.',
      practitioner:
        'h₀ = wte[x] + wpe[pos] injects absolute order in-place, but hard-caps context length at 64.',
      formal:
        'Additive absolute positional encoding creates H_0 ∈ R^{T×d_model} at the cost of fixed horizon T_max.',
    },
  },
  {
    id: 'ch2-beat-5',
    beatNumber: 5,
    shortTitle: '5. Modern Upgrade: RoPE',
    heading: 'Frontier Position Encoding: Rotating Vectors With RoPE',
    sceneBeat: 'ch2-5-rope',
    visualFocusNote:
      'Drag the Slot A and Slot B sliders on stage to spin the Query and Key arrows on the 2D RoPE clock face—notice the angle between them depends ONLY on their distance (B − A)!',
    paragraphs: {
      foundations: [
        'Modern giant LLMs (like Llama, Gemini, Mistral, and DeepSeek) replaced our tiny model’s additive wpe table with an ingenious geometric trick called RoPE — Rotary Position Embeddings.',
        'Instead of adding a slot number at the start, RoPE pairs up dimensions inside Attention and spins each word’s Query and Key arrows like clock hands—1 click for Slot 1, 2 clicks for Slot 2, 10 clicks for Slot 10.',
        'Try dragging both Slot sliders on the 3D clock face! Whether the two words sit at Slots (2, 5) or Slots (12, 15), the angle between their arrows stays IDENTICAL (distance = 3). That lets modern LLMs understand relative distance naturally across hundreds of pages.',
      ],
      practitioner: [
        'RoPE applies a position-dependent 2D rotation matrix to consecutive coordinate pairs (2i, 2i+1) of Q and K inside every attention layer, using multiple frequencies (fast-spinning hands for nearby words, slow-spinning hands for distant paragraphs).',
        'Because rotation is orthogonal, vector norms stay untouched and the dot product (R_m q) · (R_n k) depends strictly on relative offset (m − n), enabling clean long-context scaling (128k–1M+ tokens).',
      ],
      formal: [
        'For slot m and pair index i ∈ {0, ..., d_k/2 - 1} with angular frequency θ_i = base^{-2i/d_k}, RoPE rotates q^{(i)} by m θ_i via R_{m,i} ∈ SO(2). By orthogonality R_m^T R_n = R_{n-m}, inner products satisfy ⟨R_m q, R_n k⟩ = q^T R_{n-m} k, injecting pure relative positional structure directly into QK logits.',
      ],
    },
    keyTakeaway: {
      foundations:
        'RoPE rotates Q and K like clock hands so two words always know their exact distance apart—no matter how long the document is.',
      practitioner:
        'RoPE encodes relative position (m − n) directly in Q·K dot products with zero learned parameters and clean long-context extrapolation.',
      formal:
        'Block-diagonal SO(2) rotations over Q and K convert absolute indices into exact relative-displacement kernels R_{n-m}.',
    },
    predict: {
      id: 'ch2-b5-predict',
      question:
        'In RoPE, if Token A moves from Slot 2 to Slot 22, and Token B moves from Slot 5 to Slot 25, what happens to the angle between their rotated Q and K arrows?',
      options: [
        {
          id: 'a',
          label: 'Nothing! Both rotated by +20 steps, so the relative angle between them (distance = 3) stays exactly the same.',
        },
        {
          id: 'b',
          label: 'The angle becomes 10× larger because 25 is much bigger than 5.',
        },
        {
          id: 'c',
          label: 'The arrows shrink to zero length past slot 20.',
        },
      ],
      correctId: 'a',
      explanation:
        'Exactly! Because rotating two clock hands by the same amount preserves the angle between them, RoPE naturally captures relative distance.',
    },
  },
];

export const CHAPTER_3_BEATS: BookBeat[] = [
  {
    id: 'ch3-beat-1',
    beatNumber: 1,
    shortTitle: '1. The Residual Highway',
    heading: 'Never Erase the Notebook: The Residual Stream',
    sceneBeat: 'ch3-1-residual-bus',
    visualFocusNote:
      'Follow the unbroken blue Residual Stream beam running straight through all 4 Transformer blocks while each Attention and MLP branch adds (+) a smaller update onto it.',
    paragraphs: {
      foundations: [
        'We now have our starting [T, 128] slab h₀ ready to enter the 4 Transformer blocks. Here is the single most important architectural secret of modern AI: a Transformer block NEVER overwrites its input.',
        'Look at the solid blue beam running unbroken across the entire 3D machine. That is the Residual Stream—think of it as a shared notebook passing down an assembly line. Each Attention block reads a copy of the notebook, computes a helpful note (Δ_attn), and ADDS (+) it onto the notebook. Then the MLP reads a copy and ADDS (+) one more note (Δ_mlp).',
        'Because nothing is erased, early clues (like the exact original word or its slot) stay safely readable even 100 layers later.',
      ],
      practitioner: [
        'Residual skip connections (`x = x + Attn(LN1(x))`, `x = x + MLP(LN2(x))`) turn the deep stack into an additive communication bus in a shared R^{128} coordinate space.',
        'Just as critically for training: during backpropagation, the derivative of `x + F(x)` with respect to `x` is `I + ∂F/∂x`. That identity term `I` creates an unobstructed gradient superhighway straight from the final loss all the way back to Block 0.',
      ],
      formal: [
        'Unrolling L residual blocks yields h_L = h_0 + Σ_{l=0}^{L-1} (Δ_{attn}^{(l)} + Δ_{mlp}^{(l)}). Residual linearity explains both why deep Transformers train without vanishing gradients and why the Logit Lens (projecting intermediate h_l directly through W_E^T) produces meaningful early predictions.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Transformer blocks add small updates onto an unbroken shared highway instead of erasing earlier information.',
      practitioner:
        'x + SubLayer(x) preserves early features forward and gives gradients a clean identity highway backward.',
      formal:
        'Additive residual composition h_L = h_0 + Σ Δ_l establishes a shared linear communication channel across depth.',
    },
  },
  {
    id: 'ch3-beat-2',
    beatNumber: 2,
    shortTitle: '2. Break-It: Cut Highway',
    heading: 'Break-It Experiment #1: What Happens If We Cut the Skip Connections?',
    sceneBeat: 'ch3-2-cut-highway',
    visualFocusNote:
      'Click "Cut Residual Highway" on stage to disable all skip connections live in your real browser model and watch the next-token probabilities and loss react immediately.',
    paragraphs: {
      foundations: [
        'Don’t take a textbook’s word that skip connections are load-bearing—let’s break them right now on our real running AI.',
        'Click the orange "Cut Residual Highway" button at the top of the 3D stage. Instead of adding updates onto the shared highway, each block is now forced to throw away the highway and pass forward ONLY its raw sub-block output.',
        'Watch the 3D bridge break and look at the live probability bars and loss meter below: the model’s confident story prediction immediately collapses and cross-entropy loss spikes dramatically! Click again to reconnect the highway and watch intelligence snap straight back.',
      ],
      practitioner: [
        'Cutting residual bypasses (`disabledResiduals` in src/engine/transformer.ts) replaces `res + attnOut` with `attnOut` and `res + mlpOut` with `mlpOut` across all 4 blocks.',
        'Compare this catastrophic loss spike against disabling a single attention head in Chapter 4: individual heads are redundant, whereas the residual stream is the structural backbone of the entire network.',
      ],
      formal: [
        'Removing the identity skip term turns h_L = h_0 + Σ Δ_l into pure compositional application (MLP_3 ∘ Attn_3 ∘ ... ∘ Attn_0)(h_0), destroying the shared residual basis that downstream weight matrices and tied unembedding W_E^T were trained to read.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Cutting the residual highway instantly breaks the model—proving the bypass path carries vital memory.',
      practitioner:
        'Live counterfactual ablation proves residual skip connections are structurally load-bearing, not optional.',
      formal:
        'Severing identity skips collapses the shared residual coordinate frame and spikes cross-entropy loss.',
    },
    predict: {
      id: 'ch3-b2-predict',
      question:
        'Before clicking the button: which do you predict hurts the model’s cross-entropy loss MORE—disabling ONE Attention Head, or cutting the Residual Highway?',
      options: [
        {
          id: 'a',
          label: 'Cutting the Residual Highway hurts dramatically more, because every layer relies on adding updates onto the preserved stream.',
        },
        {
          id: 'b',
          label: 'Disabling one Attention Head hurts more, because the residual highway is only an empty backup wire.',
        },
        {
          id: 'c',
          label: 'Both cause the exact same tiny 1% change.',
        },
      ],
      correctId: 'a',
      explanation:
        'Right! Click "Cut Residual Highway" above to verify live how steeply the loss jumps when the shared stream is broken.',
    },
  },
  {
    id: 'ch3-beat-3',
    beatNumber: 3,
    shortTitle: '3. Volume & LayerNorm',
    heading: 'Taming Exploding Numbers: Pre-LayerNorm (LN₁)',
    sceneBeat: 'ch3-3-layernorm',
    visualFocusNote:
      'Back row: your selected token’s jagged raw 128-number vector entering Block l. Front row: the exact same 128 numbers right after LayerNorm 1 stabilizes their spread.',
    paragraphs: {
      foundations: [
        'Adding updates onto a highway over and over creates a new danger: volume drift! Some tokens’ numbers gradually grow huge while others shrink tiny, making later matrix multiplications unstable.',
        'Before entering Attention or the MLP, each token passes through an automatic volume leveler called LayerNorm. Look at the two 128-bar strips on the 3D stage: the back strip is the token’s raw vector straight off the highway; the front strip is right after LayerNorm shifts its average to 0 and tightens its spread to a steady range.',
        'Finally, LayerNorm includes 128 tiny learned volume knobs (γ) and 128 offset knobs (β) so the model can still turn specific important channels louder or quieter whenever it needs to.',
      ],
      practitioner: [
        'Pre-LayerNorm applies `LN(v) = ((v - mean(v)) / sqrt(var(v) + 1e-5)) * gamma + beta` independently to each token row [128].',
        'Why Pre-LN instead of the original 2017 Post-LN? Putting LayerNorm on the side branch BEFORE Attention/MLP leaves the main residual highway completely untouched by division—making deep networks far easier to train without fragile learning-rate warmups.',
      ],
      formal: [
        'For token vector v ∈ R^{128}, LN_1 standardizes μ = (1/d) Σ v_i and σ² = (1/d) Σ (v_i - μ)², producing unit-variance coordinates prior to affine rescaling γ ⊙ v̂ + β (256 parameters per norm). Pre-LN preserves exact identity Jacobian flow along the residual spine.',
      ],
    },
    keyTakeaway: {
      foundations:
        'LayerNorm acts like an automatic volume leveler before each sub-block so token numbers stay steady and balanced.',
      practitioner:
        'Pre-LN standardizes each token’s 128 channels to mean 0, std 1 (plus learned γ, β) while keeping the residual spine clean.',
      formal:
        'Per-token affine standardization stabilizes activation variance and preserves unattenuated Pre-LN gradient paths.',
    },
  },
  {
    id: 'ch3-beat-4',
    beatNumber: 4,
    shortTitle: '4. Final Norm & RMSNorm',
    heading: 'Break-It #2 (Final LN_f) & Modern RMSNorm',
    sceneBeat: 'ch3-4-rmsnorm',
    visualFocusNote:
      'Compare our model’s live LayerNorm output (blue/amber back row) against modern RMSNorm (front row)—and click "Disable Final LN_f" to test why normalization matters at the exit.',
    paragraphs: {
      foundations: [
        'Because Pre-LayerNorm only sits on the side branches inside each block, the main Residual Highway itself builds up steam across all 4 blocks. So right before predicting the next word, one last Final LayerNorm (ln_f) levels the highway one final time. Try clicking "Disable Final LN_f" above to watch how much the predictions and loss wobble without it!',
        'And here is a neat modern upgrade used in Llama, Gemma, Mistral, and DeepSeek: engineers discovered you don’t even need to subtract the average (mean) to keep vectors stable. Simply dividing by the vector’s overall size—called RMSNorm—produces nearly identical bars (compare the two rows in 3D!) while running noticeably faster on GPUs.',
      ],
      practitioner: [
        'RMSNorm drops mean-centering and the bias parameter β, computing `v / sqrt(mean(v^2) + eps) * gamma`. That saves one full reduction kernel pass per token across every layer.',
        'Toggle "Disable Final LN_f" on stage to observe unnormalized residual magnitudes hitting the tied unembedding matrix W_E^T directly.',
      ],
      formal: [
        'RMSNorm(v) = (v / √((1/d) ||v||_2² + ε)) ⊙ γ retains re-scaling invariance while omitting shift invariance (mean subtraction). On our live token vector below, notice how closely RMSNorm tracks full LayerNorm whenever residual channel means remain near zero.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Final LayerNorm steadies the highway before the exit—and modern LLMs speed this up with RMSNorm by skipping mean subtraction.',
      practitioner:
        'Pre-LN architectures require a final ln_f before unembedding; modern LLMs replace LayerNorm with bias-free RMSNorm for GPU speed.',
      formal:
        'RMSNorm preserves scale invariance at lower memory-bandwidth cost, matching LayerNorm stability across frontier scale.',
    },
    predict: {
      id: 'ch3-b4-predict',
      question:
        'What is the main mathematical step that modern RMSNorm SKIPS compared to classic LayerNorm?',
      options: [
        {
          id: 'a',
          label: 'Subtracting the mean (mean-centering) and adding a learned bias β.',
        },
        {
          id: 'b',
          label: 'Dividing by the vector’s magnitude.',
        },
        {
          id: 'c',
          label: 'Normalizing each token independently.',
        },
      ],
      correctId: 'a',
      explanation:
        'Correct! RMSNorm keeps magnitude scaling (dividing by RMS) and learned γ, while skipping mean-centering and β for faster GPU kernels.',
    },
  },
];
