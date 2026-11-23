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
        'Extreme 2: make every unique whole word its own dictionary entry. That makes sentences short, but English has millions of words, names, typos, and conjugations. Your dictionary would explode into millions of entries, and the moment a user types a new word like "strawberryish", the model would crash.',
        'Modern LLMs solve this with Byte Pair Encoding (BPE): a clever middle path that starts with individual characters and merges only the most frequent pairs into subword chunks.',
      ],
      practitioner: [
        'Pure character-level tokenization produces sequence lengths T that scale with raw character count, exploding the O(T^2) cost of self-attention and diluting credit assignment across dozens of meaningless intermediate steps.',
        'Pure word-level tokenization suffers from vocabulary unboundedness |V| -> inf, huge embedding parameter footprint |V| * d_model, and catastrophic Out-Of-Vocabulary (OOV) failure whenever morphological variants or neologisms appear.',
        'Byte Pair Encoding (Sennrich et al., 2016) constructs a finite subword vocabulary |V| (here exactly 512 entries) by greedily merging the most frequent adjacent bigrams in training corpora, guaranteeing zero OOV while compressing sequence length by ~3-4x.',
      ],
      formal: [
        'Character tokenization maximizes sequence length T while minimizing vocabulary size |V| = Sigma. Word tokenization minimizes T but induces unbounded parameter matrices W_e in R^{|V| x d_model} with severe sample sparsity.',
        'BPE realizes an optimal rate-distortion trade-off: an entropy-guided greedy compression that yields a subword vocabulary V with bounded capacity |V| = 512, preserving deterministic invertibility while reducing sequence length T.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Tokens are neither whole words nor raw letters; they are statistical syllables that balance dictionary size against sentence length.',
      practitioner:
        'BPE compresses sequence length T to curb attention quadratic cost while bounding vocabulary parameter size |V| * d_model with zero OOV.',
      formal:
        'Subword segmentation establishes an empirical prefix-free compression code over text that bounds vocabulary cardinality |V| at finite sequence length T.',
    },
    predict: {
      id: 'ch1-b1-predict',
      question:
        'If we used pure character tokenization instead of BPE, what would happen to the computation cost inside Self-Attention?',
      options: [
        {
          id: 'a',
          label:
            'Self-Attention cost would explode quadratically because the sequence length T would be 4x to 5x longer.',
        },
        {
          id: 'b',
          label: 'Self-Attention cost would drop because each token is simpler.',
        },
        {
          id: 'c',
          label: 'Nothing changes because attention does not depend on sequence length.',
        },
      ],
      correctId: 'a',
      explanation:
        'Self-Attention compares every token against every other token in a [T, T] matrix. Quadrupling T increases attention matrix cells by 16x (T^2).',
    },
  },
  {
    id: 'ch1-beat-2',
    beatNumber: 2,
    shortTitle: '2. BPE Merges',
    heading: 'Watching Character Pairs Fuse into Subwords',
    sceneBeat: 'ch1-2-bpe-merges',
    visualFocusNote:
      'Use the merge step slider above the 3D stage. Watch how adjacent letter tiles highlight in amber and snap together into a single token block.',
    paragraphs: {
      foundations: [
        'How does the computer decide which pairs to glue together? Before training the language model, we run a simple counting tournament over millions of storybooks.',
        'We begin with every word split into raw letters: "O n c e   u p o n". Then we find the single most frequent adjacent pair across the entire library (for instance, "i" + "n" -> "in") and replace all occurrences with a new token.',
        'Drag the Merge Step slider on the 3D stage right now. Watch step 0 (raw letters) fuse step by step into the exact tokens our miniature model sees. Notice how frequent prefixes and small common words become single tiles while rarer combinations remain split.',
      ],
      practitioner: [
        'Our model vocabulary was trained with 256 byte/character base symbols followed by 256 learned merge operations, resulting in an exact vocabulary size of |V| = 512.',
        'Inference tokenization applies these exact merge rules deterministically in ascending order of their training merge rank.',
        'Scrub the merge slider to inspect real merge events on your prompt. Observe how common substrings fuse into single token IDs while rare words stay decomposed into multi-token fragments.',
      ],
      formal: [
        'Given alphabet Sigma and base vocabulary V_0 = Sigma, each merge step k defines a pair (u, v) = argmax_{(u,v)} count(u, v) in Corpus and sets V_{k} = V_{k-1} union {uv}.',
        'At tokenization runtime, text is greedily contracted according to the total ordering induced by merge rank r(u, v), producing the unique minimal parse under the fixed merge hierarchy.',
      ],
    },
    keyTakeaway: {
      foundations:
        'BPE is simply a greedy tournament of letter pairs: the most popular pairs get glued together first.',
      practitioner:
        'Tokenization greedily applies the training merge rank table to partition input text into vocabulary indices.',
      formal:
        'Deterministic greedy contraction under the rank-ordered merge table generates the canonical subword parse.',
    },
    predict: {
      id: 'ch1-b2-predict',
      question:
        'Why does BPE merge "th" + "e" into "the" very early, but leaves "strawberry" split into multiple pieces?',
      options: [
        {
          id: 'a',
          label:
            '"the" is among the most frequent byte pairs in English, so its merge rank was learned early, whereas rare words run out of merge budget.',
        },
        {
          id: 'b',
          label: 'Because the dictionary has a rule against long words.',
        },
        {
          id: 'c',
          label: 'Because fruits are intentionally filtered during pre-training.',
        },
      ],
      correctId: 'a',
      explanation:
        'BPE merge rank is purely determined by frequency in the training corpus. Ultra-frequent words fuse in the first few merges; rare words exhaust the vocabulary budget before fusing.',
    },
  },
  {
    id: 'ch1-beat-3',
    beatNumber: 3,
    shortTitle: '3. Spaces & Typos',
    heading: 'The Leading-Space Blindspot',
    sceneBeat: 'ch1-3-spaces-blindspot',
    visualFocusNote:
      'Look at the two side-by-side token ribbons. Notice how the leading space is baked into the token itself (marked with a visible middle-dot dot symbol).',
    paragraphs: {
      foundations: [
        'Here is one of the most famous quirks in all of AI: in modern language models, a leading space is NOT a separate token. It is glued to the start of the word!',
        'That means " dog" (with a space) and "dog" (without a space) are two completely different token IDs in the model’s vocabulary, with completely different numbers and vectors.',
        'This is why LLMs sometimes stumble when you ask them to spell words backwards, count letters, or solve rhyming riddles. The model never saw raw letters; it saw opaque numeric tokens with spaces pre-baked into them.',
      ],
      practitioner: [
        'GPT-2 and LLaMA tokenizers treat whitespace as a regular byte. The space character is mapped to a visible byte character (like "Ġ" in GPT-2 or " " in SentencePiece).',
        'Consequently, the token ID for " dog" (ID 184) and "dog" (ID 212) share zero identity in the embedding matrix. Their weights are updated independently during backprop.',
        'This explains why prompt engineering is so sensitive to trailing spaces: appending a space changes the next-token distribution from word-continuations to new-word starts.',
      ],
      formal: [
        'The tokenizer bijection maps byte streams including 0x20 to vocabulary IDs: f(" dog") != f("dog"). The embedding lookup returns orthogonal basis rows W_{" dog"} != W_{"dog"}.',
        'Spatial and morphological reasoning errors in LLMs originate from this tokenization boundary asymmetry: characters lose topological identity once absorbed into subword tokens.',
      ],
    },
    keyTakeaway: {
      foundations:
        'A word with a space in front is a completely different token than the word alone.',
      practitioner:
        '" dog" and "dog" are distinct vocabulary IDs with distinct embedding rows; trailing spaces alter the input state.',
      formal:
        'Token boundaries break character-level topological continuity, making character-counting tasks non-trivial for autoregressive decoders.',
    },
  },
  {
    id: 'ch1-beat-4',
    beatNumber: 4,
    shortTitle: '4. Vocabulary IDs',
    heading: 'From Text Chunks to Discrete Integers',
    sceneBeat: 'ch1-4-token-ids',
    visualFocusNote:
      'Examine the token ribbon on stage. Each token displays its human text chunk above and its exact integer vocabulary ID [0-511] below.',
    paragraphs: {
      foundations: [
        'Once BPE finishes merging, the letters are gone. The computer replaces each subword tile with an integer from 0 to 511.',
        'In our miniature model, vocabulary ID 0 is "<|unk|>" (unknown character), ID 1 is "<|pad|>" (padding), and the remaining 510 IDs map to single characters, punctuation, and common English subwords.',
        'These integers are NOT grades or measurements—they are just barcode numbers! Token 400 is not "twice as large" or "twice as good" as token 200. They are purely categorical lookup keys.',
      ],
      practitioner: [
        'The output of the tokenizer is a 1D tensor of discrete indices x in {0, ..., |V|-1}^T of sequence length T.',
        'Because discrete categorical indices cannot support continuous gradient flow, they serve solely as row addresses into the weight matrix W_e in R^{|V| x d_model}.',
        'Inspect the live numbers card on the left: see the exact integer sequence generated for your current prompt, with its min, max, and sequence length T.',
      ],
      formal: [
        'Tokenization yields a discrete sequence X = (x_0, x_1, ..., x_{T-1}) in V^T, where V subset N and |V| = 512.',
        'Categorical indices are non-differentiable; continuous computation begins only after one-hot projection X @ W_e in R^{T x d_model}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Tokens become barcode IDs. The numbers carry no intrinsic magnitude—they are just addresses.',
      practitioner:
        'The discrete token sequence x in {0..|V|-1}^T acts as row pointers for the subsequent continuous embedding lookup.',
      formal:
        'Discrete token IDs parameterize a one-hot sequence operator onto the continuous d_model representation manifold.',
    },
  },
  {
    id: 'ch1-beat-5',
    beatNumber: 5,
    shortTitle: '5. Context Window',
    heading: 'The Hard Limit of Sequence Length',
    sceneBeat: 'ch1-5-context-window',
    visualFocusNote:
      'Look at the sequence tray on the 3D stage. Active tokens light up in cyan; the remaining slots up to our model limit of 64 are outlined in dim gray.',
    paragraphs: {
      foundations: [
        'Every LLM has a hard limit on how many tokens it can read at once: its context window. For frontier models it might be 128,000 tokens; for our miniature model, it is exactly 64 tokens.',
        'Why does a hard limit exist? Because inside every Transformer layer, every token has to compare itself to every other token. A sentence of 64 tokens requires 4,096 comparisons; 1,000 tokens requires 1,000,000 comparisons!',
        'If your text exceeds 64 tokens, the model cannot see the surplus—it must either truncate the beginning or slide a window forward.',
      ],
      practitioner: [
        'Our miniature architecture fixes max_seq_len T_max = 64. Memory consumption in attention scales as O(L * H * T^2).',
        'Positional embedding tables (wpe) are allocated up to shape [T_max, d_model] = [64, 128]. Sequence inputs longer than 64 tokens trigger runtime out-of-bounds index errors.',
        'Observe the capacity bar in the live card: your sentence occupies T of 64 slots, leaving exactly (64 - T) slots available for autoregressive generation.',
      ],
      formal: [
        'The input manifold is constrained to X in V^T with T <= T_max = 64. Attention activation footprint scales as Theta(T^2 * d_k * H).',
        'Learned absolute positional encodings W_p in R^{T_max x d_model} strictly bound sequence capacity, preventing zero-shot length extrapolation beyond T_max.',
      ],
    },
    keyTakeaway: {
      foundations:
        'The context window is the model’s working memory tray: 64 slots in our miniature LLM, strictly bounded by attention math.',
      practitioner:
        'Sequence length is hard-capped at T_max = 64 by the learned positional table and the O(T^2) attention memory footprint.',
      formal:
        'The operational sequence domain is bounded by T <= T_max, governed by W_p capacity and quadratic attention tensor scaling.',
    },
  },
];

export const CHAPTER_2_BEATS: BookBeat[] = [
  {
    id: 'ch2-beat-1',
    beatNumber: 1,
    shortTitle: '1. The Lookup Wall',
    heading: 'Turning Barcodes into 128 Continuous Numbers',
    sceneBeat: 'ch2-1-wte-lookup',
    visualFocusNote:
      'Look at the massive embedding wall on the 3D stage (512 rows by 128 columns). Notice the highlighted rows pulled out for each token in your sentence.',
    paragraphs: {
      foundations: [
        'An integer barcode like 184 cannot be multiplied or added to anything meaningful. We need to turn that integer into a rich list of numbers—a vector.',
        'Our model has a giant lookup table called `wte` (Word Token Embeddings). It has exactly 512 rows (one for each token in our vocabulary) and 128 columns.',
        'When the word "dog" arrives with token ID 184, the model simply looks up row 184 and copies out all 128 numbers. That 128-number vector is the word’s initial coordinate in the model’s internal universe.',
      ],
      practitioner: [
        'The token embedding layer is a linear lookup table W_e of shape [vocab_size, d_model] = [512, 128], totaling 65,536 trainable float32 parameters.',
        'Mathematically, lookup is equivalent to one-hot multiplying: h_t^0 = e_{x_t}^T W_e. No matrix multiplication is executed at runtime; it is an O(1) memory pointer dereference.',
        'Inspect the live card: see the exact L2 norm, min, and max coordinate values of the embedding vector fetched for your current active token.',
      ],
      formal: [
        'Let x_t in {0, ..., |V|-1}. The embedding map psi: V -> R^{d_model} extracts row x_t from parameter tensor W_e in R^{|V| x d_model}: h_t^{(0)} = W_{e}[x_t, :].',
        'Weights are initialized under Gaussian N(0, 0.02^2) and optimized via backpropagation, embedding discrete symbols into a Riemannian representation space.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Embedding is just looking up a row in a 512x128 table. Every token gets its own list of 128 numbers.',
      practitioner:
        'W_e [512, 128] transforms discrete indices into continuous activation vectors via direct row indexing.',
      formal:
        'The embedding operator maps categorical tokens into a continuous 128-dimensional latent vector space via W_e in R^{512 x 128}.',
    },
  },
  {
    id: 'ch2-beat-2',
    beatNumber: 2,
    shortTitle: '2. 3D PCA Space',
    heading: 'Exploring the 512-Token Geometry in 3D',
    sceneBeat: 'ch2-2-pca-cloud',
    visualFocusNote:
      'The 3D stage displays our real trained 512-token embedding space projected down to 3 principal axes. Rotate the camera to see semantic clusters of words and names.',
    paragraphs: {
      foundations: [
        'How do 128 numbers represent meaning? In high-dimensional space, geometry is meaning. Words that appear in similar situations learn similar lists of numbers.',
        'Because human eyes cannot see 128 dimensions at once, we use a mathematical camera called Principal Component Analysis (PCA) to find the three widest directions of variation and project all 512 tokens into a 3D cloud.',
        'Look around the 3D stage! Notice how character names (Max, Lily) cluster together, punctuation forms its own outer ring, and action verbs gather in their own territory.',
      ],
      practitioner: [
        'We compute exact centered covariance C = (1/N) (W_e - mu)^T (W_e - mu) and extract the top 3 eigenvectors via power iteration with Hotelling deflation.',
        'The live card displays the percentage of total variance explained by PC1, PC2, and PC3 across all 512 vocabulary entries.',
        'Cosine similarity cos(u, v) = (u . v) / (||u|| ||v||) measures angular proximity in R^128. The top cosine neighbors for your active token are displayed in the live telemetry panel.',
      ],
      formal: [
        'PCA projects X in R^{512 x 128} onto the subspace spanned by the top 3 eigenvectors of Sigma = (1/N) tilde{X}^T tilde{X}.',
        'Semantic similarity corresponds to inner-product density under the learned metric tensor G = W_e W_e^T.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Words with similar meanings live near each other in the 128-dimensional embedding cloud.',
      practitioner:
        'PCA projects the 128D space onto 3 principal axes, preserving maximum semantic variance for visual exploration.',
      formal:
        'Eigenspectrum decomposition reveals clustering geometry governed by the distributional hypothesis in R^{128}.',
    },
    predict: {
      id: 'ch2-b2-predict',
      question:
        'In the trained embedding space, what do you expect the cosine similarity between "dog" and "cat" to be compared to "dog" and "banana"?',
      options: [
        {
          id: 'a',
          label:
            '"dog" and "cat" have much higher cosine similarity because they share contexts (pets, animals, story protagonists).',
        },
        {
          id: 'b',
          label: 'They all have identical similarity because weights are random.',
        },
        {
          id: 'c',
          label: '"dog" and "banana" are more similar because both have three vowels.',
        },
      ],
      correctId: 'a',
      explanation:
        'Embeddings organize by contextual substitution: words that appear in identical sentence frames receive correlated gradient updates and cluster together.',
    },
  },
  {
    id: 'ch2-beat-3',
    beatNumber: 3,
    shortTitle: '3. Order-Blind Bag',
    heading: 'Why Embeddings Alone Cannot Tell Word Order',
    sceneBeat: 'ch2-3-order-blind',
    visualFocusNote:
      'Look at the scrambled token cloud on stage. Without position data, "Max bit the dog" and "The dog bit Max" look 100% identical to the model!',
    paragraphs: {
      foundations: [
        'Imagine someone hands you three cards: "dog", "bit", and "man". Who bit whom? You cannot tell unless you know the order of the cards!',
        'Here is the shock: raw token embeddings contain ZERO order information. If you feed the sentence "Max chased the cat" versus "The cat chased Max", the embedding lookup produces the exact same set of vectors.',
        'Self-Attention itself is completely order-blind (permutation equivariant). If we do not inject position information, the model has no way to know whether a word came first, middle, or last.',
      ],
      practitioner: [
        'Self-Attention without position encoding is strictly permutation equivariant: Attention(P X) = P Attention(X) for any permutation matrix P.',
        'This means an order-blind Transformer is merely a continuous "bag of words", incapable of distinguishing syntax, causal narrative, or directional dependencies.',
        'To break this symmetry, modern LLMs add or multiply positional coordinates into the representation.',
      ],
      formal: [
        'Let pi in S_T be a permutation operator. For standard self-attention, Softmax((pi X W_Q)(pi X W_K)^T / sqrt{d_k}) pi X W_V = pi Attention(X).',
        'Without symmetry breaking, the representation space lacks the capacity to distinguish directed graph structure in language.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Without position markers, an LLM cannot tell "dog bit man" from "man bit dog".',
      practitioner:
        'Transformer operations are permutation equivariant; position information must be explicitly injected to model grammar and syntax.',
      formal:
        'Permutation equivariance of dot-product attention necessitates explicit symmetry-breaking coordinates.',
    },
  },
  {
    id: 'ch2-beat-4',
    beatNumber: 4,
    shortTitle: '4. Absolute Position (wpe)',
    heading: 'Adding Position Coordinates Vector-by-Vector',
    sceneBeat: 'ch2-4-wpe-add',
    visualFocusNote:
      'Watch the two parallel plates on the 3D stage: the token embedding plate (wte) on the left and the position plate (wpe) on the right fusing into the residual stream.',
    paragraphs: {
      foundations: [
        'The original GPT-2 solution is beautifully simple: make a second lookup table called `wpe` (Word Position Embeddings) of size 64 rows by 128 columns.',
        'Token 0 at slot 0 gets position vector 0 added to it. Token 1 at slot 1 gets position vector 1 added to it. We literally add the two 128-number vectors together: `vector = wte + wpe`.',
        'Now every vector knows both WHAT word it is and WHERE it sits in the sentence! Slot 0 has a different fingerprint than slot 15 even if both hold the word "dog".',
      ],
      practitioner: [
        'Our model implements GPT-2 style learned absolute positional embeddings W_p in R^{T_max x d_model} = [64, 128], adding 8,192 parameters.',
        'The input representation entering Block 0 is the elementwise sum: h_t^{(0)} = W_e[x_t, :] + W_p[t, :].',
        'Inspect the live card: see the exact magnitude of the wte vector, the wpe vector, and their cosine alignment before entering the first transformer block.',
      ],
      formal: [
        'The input layer computes h_t^{(0)} = psi_{wte}(x_t) + psi_{wpe}(t) for t in {0, ..., T-1}.',
        'Learned absolute encodings occupy distinct subspace directions within R^{d_model}, allowing the attention mechanism to attend based on both content and linear sequence index.',
      ],
    },
    keyTakeaway: {
      foundations:
        'GPT-2 solves word order by adding a position vector to each word vector: word + slot = complete token.',
      practitioner:
        'Learned absolute positional embeddings W_p [64, 128] are added directly: h_t = wte[x_t] + wpe[t].',
      formal:
        'Additive combination h_t^{(0)} = W_e[x_t] + W_p[t] breaks permutation symmetry by linearly superimposing position basis vectors.',
    },
  },
  {
    id: 'ch2-beat-5',
    beatNumber: 5,
    shortTitle: '5. Modern Upgrade — RoPE',
    heading: 'Rotary Position Embeddings & Clock Angles',
    sceneBeat: 'ch2-5-rope',
    visualFocusNote:
      'Interact with the RoPE clock dials on stage. Scrub Slot A and Slot B to watch the 2D feature coordinates rotate by angles proportional to relative distance (t_A - t_B).',
    paragraphs: {
      foundations: [
        'Absolute position has a flaw: it memorizes specific slot numbers (slot 5, slot 6) rather than relative distances ("3 words ago").',
        'Modern frontier models (LLaMA, Mistral, Gemma) replace learned position tables with Rotary Position Embeddings (RoPE). Instead of adding vectors, RoPE pairs up the 128 numbers into 64 dials and rotates each pair by an angle proportional to its slot index.',
        'Use the RoPE sliders on the 3D stage right now. Set Slot A and Slot B. Notice how the angle between the two hands depends ONLY on the difference (B - A), allowing the model to naturally generalize to long texts!',
      ],
      practitioner: [
        'RoPE (Su et al., 2021) encodes relative position by applying a 2D rotation matrix R_{Theta, t} to every 2-dimensional chunk of the Query and Key vectors: q_t = R_{Theta, t} (W_Q h_t).',
        'The inner product then satisfies <R_m q, R_n k> = g(q, k, m - n), making attention dot products depend strictly on relative token distance (m - n).',
        'RoPE adds zero learnable parameters and exhibits superior length extrapolation compared to absolute learned positional embeddings.',
      ],
      formal: [
        'Let R_{Theta, t}^{(d)} = diag(R_1, ..., R_{d/2}) where each block rotates by angle theta_i * t with theta_i = 10000^{-2(i-1)/d}.',
        'The bilinear form satisfies <R_{Theta, m} q, R_{Theta, n} k> = q^T R_{Theta, n - m} k, embedding relative distance directly into the affine invariant of dot-product attention.',
      ],
    },
    keyTakeaway: {
      foundations:
        'RoPE rotates feature pairs like clock hands. The angle between two words depends only on how far apart they are.',
      practitioner:
        'RoPE rotates Q and K slices by frequency angles, making attention dot products purely relative (m - n) with zero added parameters.',
      formal:
        'Orthogonal rotation groups SO(2)^{d/2} preserve vector L2 norm while encoding relative displacement in dot-product attention.',
    },
  },
];

export const CHAPTER_3_BEATS: BookBeat[] = [
  {
    id: 'ch3-beat-1',
    beatNumber: 1,
    shortTitle: '1. The Residual Bus',
    heading: 'The Central Information Trunk of the Transformer',
    sceneBeat: 'ch3-1-residual-bus',
    visualFocusNote:
      'Follow the glowing gold residual highway running through the center of the 3D stage. Notice how Attention and MLP are merely branch loops off to the side.',
    paragraphs: {
      foundations: [
        'A Transformer is often drawn as a tall tower of layers. But that picture is misleading! A Transformer is actually a high-speed central bus highway with side workshops.',
        'The 128-number vector for each word starts on the highway. Layer 1 Attention reads from the highway, computes a small update, and ADDS its answer back to the highway. Layer 1 MLP reads the highway, computes another update, and ADDS it back.',
        'Because every block adds its result (`highway = highway + update`), the original word meaning is never erased or trapped—it flows directly from the input to the output!',
      ],
      practitioner: [
        'The residual stream (He et al., 2016) enforces the state recurrence: h^{(l+1)} = h^{(l)} + F_l(LayerNorm(h^{(l)})).',
        'Instead of forcing layer l to learn an entire full representation from scratch, the layer only has to learn a differential residual perturbation Delta h.',
        'Inspect the live card: compare the L2 norm of the residual stream entering Layer 2 versus the norm of the Attention and MLP branch updates.',
      ],
      formal: [
        'Let h^{(l)} in R^{T x d_model}. The residual recurrence h^{(l+1)} = h^{(l)} + Delta h^{(l)} decomposes the final representation into an explicit sum over all paths: h^{(L)} = h^{(0)} + sum_{l=0}^{L-1} (Attn_l + MLP_l).',
        'This prevents vanishing gradients by providing an unimpeded identity gradient highway: d L / d h^{(0)} = d L / d h^{(L)} (I + sum d Delta h / d h^{(0)}).',
      ],
    },
    keyTakeaway: {
      foundations:
        'The residual stream is an unbroken highway: layers only add small notes to the stream, never erase it.',
      practitioner:
        'Additive updates h_{l+1} = h_l + F(h_l) allow deep layers to compute differential updates without destroying earlier features.',
      formal:
        'The residual formulation transforms deep non-linear networks into ensembles of shallow gradient paths.',
    },
  },
  {
    id: 'ch3-beat-2',
    beatNumber: 2,
    shortTitle: '2. Break-It Lab',
    heading: 'What Happens When You Cut the Highway?',
    sceneBeat: 'ch3-2-cut-highway',
    visualFocusNote:
      'Look at the fractured red barrier cutting across the residual bus on stage. Toggle the "Cut Residual Stream" button to see the model break.',
    paragraphs: {
      foundations: [
        'What would happen if we deleted the residual connections and forced each layer to pass its output directly into the next layer (`highway = update`)?',
        'Let’s test it! Click the "Cut Residuals" button in the HUD above the 3D stage right now.',
        'Watch what happens to the output probabilities. Without the residual highway, the model loses the ability to remember earlier words. The numbers either blow up into infinities or collapse into gibberish.',
      ],
      practitioner: [
        'Ablating residual connections forces each block to act as an unconditioned map h_{l+1} = F(h_l). In deep networks, this causes exponential degradation of gradient signal.',
        'In our 4-layer model, cutting all residual connections causes output top-1 prediction probability to collapse and perplexity to explode by over 100x.',
        'Inspect the live telemetry card: observe the catastrophic drop in confidence for the next token when the residual highway is severed.',
      ],
      formal: [
        'Removing identity shortcuts contracts the gradient operator to prod_{l=0}^{L-1} J_l, whose singular values exponentially vanish or explode depending on ||J_l||.',
        'Residual bypass maintains an identity component in the Jacobian, guaranteeing well-conditioned backpropagation.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Cutting the residual highway destroys the model: gradients vanish and words lose their identities.',
      practitioner:
        'Residual ablation causes catastrophic signal loss and prediction collapse; identity connections are non-optional in modern LLMs.',
      formal:
        'The identity skip connection prevents exponential spectrum collapse of the layer-to-layer Jacobian.',
    },
    predict: {
      id: 'ch3-b2-predict',
      question:
        'If you cut the residual connection around an Attention block, what happens to the original word embedding vector?',
      options: [
        {
          id: 'a',
          label:
            'It is completely discarded and replaced by the Attention value-mix output alone.',
        },
        {
          id: 'b',
          label: 'It is duplicated and stored in the KV cache.',
        },
        {
          id: 'c',
          label: 'It gets amplified by 4x.',
        },
      ],
      correctId: 'a',
      explanation:
        'Without the residual add (x + Attention(x)), the output is purely Attention(x), entirely replacing the input vector with whatever the attention head gathered.',
    },
  },
  {
    id: 'ch3-beat-3',
    beatNumber: 3,
    shortTitle: '3. Pre-LayerNorm',
    heading: 'Keeping Numbers Tamed Before Every Operation',
    sceneBeat: 'ch3-3-layernorm',
    visualFocusNote:
      'Compare the two side-by-side vector strips: raw unbounded numbers on the left versus normalized (zero-mean, unit-variance) numbers on the right.',
    paragraphs: {
      foundations: [
        'Because every block ADDS numbers to the residual highway, what stops the numbers from growing larger and larger until the computer runs out of digits?',
        'Enter LayerNorm: the model’s thermostat. Before any block (Attention or MLP) reads from the highway, LayerNorm measures all 128 numbers, subtracts their average (so the center is zero), and divides by their spread (so the variance is 1.0).',
        'Modern LLMs use Pre-LayerNorm: the highway stays untouched and free to grow, but every sub-block reads a cleanly scaled, predictable slice.',
      ],
      practitioner: [
        'LayerNorm computes: y = (x - mu) / sqrt{sigma^2 + eps} * gamma + beta, where gamma and beta are learned 128-dimensional affine parameter vectors.',
        'In Pre-LN architectures (unlike early Post-LN Transformers), normalization is placed on the branch input before Attention/MLP rather than after the residual addition: h_{l+1} = h_l + F(LN(h_l)).',
        'Pre-LN eliminates the need for delicate warm-up learning rate schedules because activations and gradients at initialization stay scale-invariant across all layers.',
      ],
      formal: [
        'Let x in R^{d_model}. LayerNorm maps x -> hat{x} = (x - mu 1) / sqrt{sigma^2 + eps}, then applies affine scale gamma and shift beta.',
        'Pre-LN bounds the variance of branch inputs while allowing the residual stream norm ||h_l|| to grow sublinearly as O(sqrt{L}), stabilizing deep gradient dynamics.',
      ],
    },
    keyTakeaway: {
      foundations:
        'LayerNorm is a thermostat: it centers and rescales the 128 numbers so operations never deal with runaway values.',
      practitioner:
        'Pre-LayerNorm normalizes branch inputs before Attention/MLP, guaranteeing stable gradients without tricky learning rate warmups.',
      formal:
        'Pre-LN enforces scale invariance of sub-block inputs, bounding Jacobian singular values across arbitrary layer depths.',
    },
  },
  {
    id: 'ch3-beat-4',
    beatNumber: 4,
    shortTitle: '4. Modern Upgrade — RMSNorm',
    heading: 'Root Mean Square Normalization & Modern Efficiency',
    sceneBeat: 'ch3-4-rmsnorm',
    visualFocusNote:
      'Examine the live side-by-side comparison on stage: standard LayerNorm (mean subtraction + variance scaling) versus RMSNorm (pure root-mean-square scaling).',
    paragraphs: {
      foundations: [
        'LayerNorm works brilliantly, but subtracting the mean requires two separate passes over the 128 numbers: one pass to find the mean, and a second pass to compute the variance.',
        'Researchers realized a surprising truth: you do NOT need to subtract the mean! Scaling by the Root Mean Square (RMS) alone gives 99.9% of the benefits while running noticeably faster.',
        'Frontier models like LLaMA, Mistral, and Gemma have completely replaced LayerNorm with RMSNorm, saving memory bandwidth across billions of tokens.',
      ],
      practitioner: [
        'RMSNorm (Zhang and Sennrich, 2019) simplifies LayerNorm by enforcing: y = x / RMS(x) * gamma, where RMS(x) = sqrt{(1/d) sum_{i=1}^d x_i^2 + eps}.',
        'By dispensing with mean centering (mu = 0), RMSNorm reduces GPU memory reads/writes by ~7% and eliminates the learned beta bias vector entirely.',
        'Inspect the live card: compare the exact output vector produced by our model’s final LN_f against a simulated RMSNorm on the exact same residual stream.',
      ],
      formal: [
        'RMSNorm replaces the affine Gaussian standardization with a pure L2-norm scaling: hat{x}_i = x_i / sqrt{(1/d) ||x||_2^2 + eps} * gamma_i.',
        'The hypothesis that shift-invariance (mu centering) is essential to Transformer stability is refuted empirically; scale regularization alone suffices for optimization stability.',
      ],
    },
    keyTakeaway: {
      foundations:
        'RMSNorm skips centering the average and only rescales the spread, giving the same stability with less computer work.',
      practitioner:
        'RMSNorm computes x / RMS(x) * gamma without mean subtraction, saving memory bandwidth and dropping the bias parameter beta.',
      formal:
        'RMSNorm demonstrates that regularizing activation magnitude alone is sufficient for numerical stability across deep decoders.',
    },
  },
];
