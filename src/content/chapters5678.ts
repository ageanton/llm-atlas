import type { BookBeat } from './bookData';

export const CHAPTER_5_BEATS: BookBeat[] = [
  {
    id: 'ch5-beat-1',
    beatNumber: 1,
    shortTitle: '1. Attention vs. MLP',
    heading: 'Attention Moves Information; the MLP Thinks About It',
    sceneBeat: 'ch5-1-mlp-role',
    visualFocusNote:
      'Compare the parameter mass inside one Transformer block: two-thirds of all learned weights (131,072 of 197,120) live inside the Feed-Forward MLP.',
    paragraphs: {
      foundations: [
        'Chapter 4 showed how Self-Attention blends clues from earlier words into the current word. But notice what Attention CANNOT do: taking weighted averages is purely linear mixing. It cannot run an "IF-THEN" rule like "IF this pronoun refers to a dog AND he found a ball, THEN predict an action like ran or played."',
        'That is the job of the second half of every block—the Feed-Forward Network (MLP). Unlike Attention, tokens never talk to each other inside the MLP. Every token steps into a private thinking booth alone, holding two-thirds of the entire block’s learned memory.',
      ],
      practitioner: [
        'In our miniature model, each block contains 65,536 attention weights versus 131,072 MLP weights (66.5% of the block). The MLP runs identically and independently on every row of LN_2(u) [T, 128].',
        'Mechanistic interpretability research shows MLP layers operate as distributed key-value pattern memories: early MLPs recognize n-grams and syntax, while deeper MLPs retrieve factual associations and narrative rules.',
      ],
      formal: [
        'Position-wise FFN(u) = W_down · σ(u W_up^T) supplies the non-linear computational expressivity absent from linear residual mixing. With expansion factor 4 (d_mlp = 512), MLP parameter count 2 · 4 d² = 8 d² doubles Multi-Head Attention’s 4 d².',
      ],
    },
    keyTakeaway: {
      foundations:
        'Attention gathers context across words; the MLP is where each word thinks privately using two-thirds of the model’s weights.',
      practitioner:
        'Position-wise MLPs hold 66.5% of block parameters and provide non-linear per-token computation and pattern memory.',
      formal:
        'Alternating sequence-axis mixing (MHA) with channel-axis non-linear FFN transforms gives Transformers universal sequence expressivity.',
    },
  },
  {
    id: 'ch5-beat-2',
    beatNumber: 2,
    shortTitle: '2. 4× Expansion',
    heading: 'Fanning Out 4× Wider: W_up (128 → 512)',
    sceneBeat: 'ch5-2-expansion',
    visualFocusNote:
      'Watch your token’s 128-number vector on the left fan outward through W_up into a 4× wider wall of 512 individual pattern-detector neurons.',
    paragraphs: {
      foundations: [
        'Inside the MLP, the very first step multiplies each token’s 128 numbers by a tall matrix called W_up, expanding 128 numbers into 512 separate neurons—four times wider!',
        'Why spread out into a bigger room? Imagine trying to sort 500 overlapping story clues inside a cramped 128-slot closet. Expanding to 512 slots gives individual specialist detectors space to separate cleanly—one neuron might watch for "dog + ball", another for "past-tense verb", another for "end of sentence".',
      ],
      practitioner: [
        'Linear projection `pre = LN_2(u) @ W_up^T` maps [T, 128] → [T, 512] using 65,536 learned weights. Each row k of W_up [512, 128] acts as a learned linear feature detector (a "key" vector) whose dot product with the token measures how strongly pattern k is present.',
      ],
      formal: [
        'Up-projection z_t = W_up · LN_2(u_t) ∈ R^{512} lifts representations into an overcomplete basis (4× d_model), mitigating polysemantic superposition before element-wise gating.',
      ],
    },
    keyTakeaway: {
      foundations:
        'W_up expands 128 numbers into 512 specialist pattern detectors so overlapping ideas can separate cleanly.',
      practitioner:
        'Each of W_up’s 512 rows scores how strongly one learned feature pattern matches the current token.',
      formal:
        '4× channel expansion maps R^{128} → R^{512} to expose disentangled linear pre-activations prior to GELU.',
    },
  },
  {
    id: 'ch5-beat-3',
    beatNumber: 3,
    shortTitle: '3. GELU & Sparsity',
    heading: 'Why Non-Linearity Is Mandatory: Interactive GELU Curve',
    sceneBeat: 'ch5-3-gelu',
    visualFocusNote:
      'Drag the GELU Input slider on stage to roll the dot along the real GELU curve—and inspect the live 512-neuron firing profile right behind it (amber = firing, gray = suppressed).',
    paragraphs: {
      foundations: [
        'Here is a wild math secret: if you multiply by W_up and immediately multiply by W_down without a curved bend in the middle, two matrix multiplies mathematically collapse into ONE plain matrix—unable to learn real "IF-THEN" logic!',
        'So every one of the 512 neurons passes through a smooth curve called GELU. Drag the slider on the 3D curve: negative inputs get gently squashed toward zero (turning unmatched detectors off), while positive matches pass right through and blaze bright amber.',
        'Look at the live 512-neuron profile for your chosen word: only a small fraction of the 512 neurons fire strongly at once—that selective firing is called sparsity.',
      ],
      practitioner: [
        'GELU (Gaussian Error Linear Unit) weights each pre-activation x by Φ(x), smoothly suppressing negative values without ReLU’s hard zero-gradient dead zone at x < 0. Our engine uses the exact tanh approximation matched to PyTorch.',
        'Inspect the live count below to see how many of Block l’s 512 neurons fire above 0.05 for your selected token.',
      ],
      formal: [
        'Without non-linearity σ, W_down(W_up u) = (W_down W_up) u collapses to a rank-bounded linear map. GELU_tanh(x) = 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³))) provides smooth C^∞ gating that yields sparse, high-selectivity hidden activations h_mlp ∈ R^{T×512}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'GELU bends the math so the model can learn real logic, shutting off unmatched neurons and letting strong matches blaze.',
      practitioner:
        'Non-linear GELU prevents linear collapse and induces sparse feature activation across the 512 hidden units.',
      formal:
        'Element-wise GELU breaks linearity, enabling universal function approximation over the residual stream.',
    },
    predict: {
      id: 'ch5-b3-predict',
      question:
        'What would happen mathematically if we deleted GELU from every MLP so the model only ran W_up followed directly by W_down?',
      options: [
        {
          id: 'a',
          label: 'The two matrices would collapse into a single linear matrix (W_down · W_up), losing non-linear reasoning power.',
        },
        {
          id: 'b',
          label: 'The model would immediately double in speed and become smarter.',
        },
        {
          id: 'c',
          label: 'The 512 neurons would automatically turn into attention heads.',
        },
      ],
      correctId: 'a',
      explanation:
        'Exactly! Linear × linear is always just another linear map. Non-linear curves like GELU or SiLU are what give neural networks real computational power.',
    },
  },
  {
    id: 'ch5-beat-4',
    beatNumber: 4,
    shortTitle: '4. Write-Back (W_down)',
    heading: 'Key-Value Memory Write-Back: W_down (512 → 128)',
    sceneBeat: 'ch5-4-wdown',
    visualFocusNote:
      'Watch the surviving amber GELU neurons compress through W_down back into a [T, 128] update slab that adds (+) onto the central Residual Highway.',
    paragraphs: {
      foundations: [
        'Every neuron that stayed lit after GELU now gets to cast its vote back onto the Residual Highway. The second matrix, W_down, holds one 128-number "memory response" column for every single one of those 512 neurons.',
        'If neuron #214 fired strongly, its learned 128-number column gets scaled up and added straight onto the highway; if neuron #12 was dimmed to zero by GELU, its column contributes nothing. Together, W_up (checking the pattern) + GELU (deciding if it matched) + W_down (writing the answer) work like a giant associative memory.',
      ],
      practitioner: [
        'Computing `mlpOut = hidden @ W_down^T` is algebraically a sum over 512 rank-1 updates: Σ_{k=0}^{511} hidden[k] · W_down[:, k]. Residual addition `resAfterMlp = resAfterAttn + mlpOut` completes the Transformer block.',
      ],
      formal: [
        'Interpreted as neural key-value memory (Geva et al.), rows of W_up act as pattern keys k_m and columns of W_down act as value vectors v_m, writing Δ_mlp(u) = Σ_{m=1}^{512} GELU(u · k_m) v_m into R^{128}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'W_down folds the 512 firing neurons back to 128 numbers, writing their learned memories onto the Residual Highway.',
      practitioner:
        'W_up checks keys, GELU gates matches, and W_down adds matching value columns back to the residual stream.',
      formal:
        'W_down projects sparse R^{512} activations into additive residual updates via residual-scaled initialization 0.02/√(2L).',
    },
  },
  {
    id: 'ch5-beat-5',
    beatNumber: 5,
    shortTitle: '5. SwiGLU & MoE',
    heading: 'Frontier Upgrades: SwiGLU Gating & Mixture of Experts (MoE)',
    sceneBeat: 'ch5-5-swiglu-moe',
    visualFocusNote:
      'See how frontier LLMs replace one dense MLP with a learned Router sending each token to only its top-2 specialist Expert FFNs (glowing amber) while inactive experts stay asleep.',
    paragraphs: {
      foundations: [
        'Modern frontier LLMs upgrade this exact MLP booth in two major ways. First, instead of plain GELU, they use SwiGLU—adding a second "dimmer switch" path that multiplies against the neurons for sharper control.',
        'Second, once models hit hundreds of billions of parameters, running one giant dense MLP for every single word becomes painfully slow. So Mixture of Experts (MoE) splits the MLP into 8 to 256 smaller specialist booths (one great at math, one at dialogue) behind a tiny Router. Each word wakes up ONLY the 2 specialists it needs—giving trillion-parameter knowledge at a fraction of the compute cost!',
      ],
      practitioner: [
        'SwiGLU computes `(SiLU(x W_gate) ⊙ (x W_up)) W_down` (shrinking hidden width slightly to ~8/3 d_model to keep parameter count matched). Sparse MoE routes each token x to Top-k experts via `softmax(x W_router)`, activating only ~5–15% of total parameters per token.',
      ],
      formal: [
        'Sparse MoE layer y = Σ_{e ∈ TopK(softmax(x W_r))} g_e(x) · SwiGLU_e(x) decouples total parameter capacity N from per-token FLOPs C_tok, trained with auxiliary load-balancing regularization to prevent router collapse.',
      ],
    },
    keyTakeaway: {
      foundations:
        'SwiGLU sharpens neuron gating, and MoE routes each word to only its top specialist experts to stay fast at huge scale.',
      practitioner:
        'SwiGLU improves quality per FLOP; sparse MoE lets trillion-parameter models run with ~10% active weights per token.',
      formal:
        'Gated FFNs and Top-k sparse routing decouple memory capacity from per-token arithmetic intensity.',
    },
  },
];

export const CHAPTER_6_BEATS: BookBeat[] = [
  {
    id: 'ch6-beat-1',
    beatNumber: 1,
    shortTitle: '1. Depth & Logit Lens',
    heading: 'Four Rounds of Thinking: Watching the Guess Sharpen',
    sceneBeat: 'ch6-1-logit-lens',
    visualFocusNote:
      'Four live Logit Lens towers stand over Blocks 0 → 1 → 2 → 3—showing our real model’s top guess and confidence sharpening at every layer.',
    paragraphs: {
      foundations: [
        'Why does an LLM stack 4 blocks (or 128 in frontier AI) instead of just 1 giant block? Because understanding a sentence takes multiple rounds of connecting clues!',
        'Because the Residual Highway keeps the exact same 128-number format from start to finish, we can sneak a peek right after Block 0, Block 1, Block 2, and Block 3—a trick called the Logit Lens. Look at the four towers on stage: watch how early blocks make a rough grammatical guess while deeper blocks sharpen onto the exact story continuation.',
      ],
      practitioner: [
        'Applying Final LayerNorm and the tied output matrix W_E^T to intermediate residual states h_{l+1, T-1} reveals progressive refinement across depth without altering the forward pass. Early layers resolve syntax/bigrams; deep layers format the final next-token distribution.',
      ],
      formal: [
        'Logit Lens evaluates p^{(l)} = softmax(LN_f(h_{l+1, T-1}) W_E^T) for l ∈ {0..3}, tracking KL divergence D_KL(p^{(3)} || p^{(l)}) contraction as residual updates align with the target unembedding row.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Stacking blocks lets the model refine its guess step by step—visible live through the Logit Lens.',
      practitioner:
        'Projecting intermediate residual streams through ln_f + wte^T exposes layer-by-layer prediction sharpening.',
      formal:
        'Iterative residual refinement progressively rotates h_l toward the target token’s unembedding direction.',
    },
  },
  {
    id: 'ch6-beat-2',
    beatNumber: 2,
    shortTitle: '2. Unembedding & Logits',
    heading: 'Turning 128 Numbers Into 512 Raw Scores (Logits)',
    sceneBeat: 'ch6-2-unembed',
    visualFocusNote:
      'The final normalized 128-number vector sweeps across the 512 vocabulary rows (tied to wte) to produce 512 raw signed Logits—positive in amber, negative in blue.',
    paragraphs: {
      foundations: [
        'After Block 3 finishes and Final LayerNorm steadies the highway, we have ONE final 128-number summary for the last token. How does the model turn 128 numbers back into an English word?',
        'It takes that 128-number arrow and computes a dot product against ALL 512 word rows in the dictionary (reusing the exact same wte coordinate book from Chapter 2!). Words that align get big positive scores; words that clash get negative scores. Those 512 raw, unbounded scores are called Logits.',
      ],
      practitioner: [
        'Unembedding (`lm_head`) computes `logits = LN_f(h_4[T-1]) @ W_E^T`, returning a [512] vector of raw real numbers (typically -8 to +12). Weight tying shares W_E [512, 128] between input lookup and output projection, saving 65,536 parameters.',
      ],
      formal: [
        'Linear readout z = LN_f(h_{L, T-1}) W_U^T ∈ R^{|V|} with tied weights W_U = W_E maps the final hidden state to log-unnormalized probabilities (logits) over the entire vocabulary.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Unembedding compares the final 128 numbers against all 512 words in the dictionary to produce raw scores called Logits.',
      practitioner:
        'lm_head (tied to wte^T) maps [128] → [512] raw unbounded logits prior to Softmax.',
      formal:
        'Tied unembedding aligns input and output token geometries via z_v = ⟨LN_f(h_L), W_E[v]⟩.',
    },
  },
  {
    id: 'ch6-beat-3',
    beatNumber: 3,
    shortTitle: '3. Temperature (τ)',
    heading: 'Determinism vs. Chance: Softmax & Temperature',
    sceneBeat: 'ch6-3-temperature',
    visualFocusNote:
      'Drag the Temperature (τ) slider on stage to watch the 3D probability bars sharpen into a single spike or flatten out—then click "Sample 10" to draw 10 real random rolls!',
    paragraphs: {
      foundations: [
        'People often wonder: if an LLM is just math, why does asking the exact same question twice give different answers? Here is the exact place randomness enters—nowhere else!',
        'Up to the Logits, the neural network is 100% deterministic. Before Softmax turns logits into probabilities, we divide every logit by Temperature (τ). Low temperature (0.2) exaggerates the leader so the model plays it safe; high temperature (1.6) flattens the bars so creative runners-up win rolls too. Try dragging τ and clicking "Sample 10"!',
      ],
      practitioner: [
        'Temperature scaling computes `softmax(logits / τ)`. As τ → 0, the distribution converges to greedy argmax (deterministic). As τ → ∞, it approaches a uniform 1/|V| distribution. Notice entropy rises smoothly with τ.',
      ],
      formal: [
        'Boltzmann / Gibbs distribution P_τ(v) = exp(z_v / τ) / Σ_u exp(z_u / τ) controls Shannon entropy H(P_τ) monotonically with inverse temperature β = 1/τ without altering rank order.',
      ],
    },
    keyTakeaway: {
      foundations:
        'The neural network is 100% deterministic; Temperature controls how safe or adventurous the final random draw is.',
      practitioner:
        'Dividing logits by τ before Softmax tunes distribution entropy without changing token rankings.',
      formal:
        'Temperature scaling modulates Gibbs entropy H(P_τ), interpolating between argmax (τ→0) and uniform sampling (τ→∞).',
    },
    predict: {
      id: 'ch6-b3-predict',
      question:
        'If you run the exact same prompt twice with Temperature = 0.1 vs. Temperature = 1.6 and click "Sample 10", which produces MORE unique words?',
      options: [
        {
          id: 'a',
          label: 'Temperature = 1.6 produces more unique words because dividing logits by 1.6 flattens the probabilities across runners-up.',
        },
        {
          id: 'b',
          label: 'Temperature = 0.1 produces more unique words because smaller numbers make the model more creative.',
        },
        {
          id: 'c',
          label: 'Both always pick the exact same single word 10 times.',
        },
      ],
      correctId: 'a',
      explanation:
        'Right! Try dragging the Temperature slider to 0.1 and clicking "Sample 10", then drag it to 1.6 and click "Sample 10" again to see it happen live.',
    },
  },
  {
    id: 'ch6-beat-4',
    beatNumber: 4,
    shortTitle: '4. Top-k & Top-p',
    heading: 'Trimming the Noisy Tail: The Nucleus Cutoff Curtain',
    sceneBeat: 'ch6-4-nucleus',
    visualFocusNote:
      'Drag Top-k and Top-p on stage: a translucent cutoff curtain sweeps across the 3D skyline—blue bars stay in the active pool while cut tail tokens turn muted gray.',
    paragraphs: {
      foundations: [
        'Even at a sensible temperature, Softmax gives tiny 0.01% chances to hundreds of bizarre, off-topic words. Roll the dice enough times and a story about Max the dog could suddenly print "umbrella"!',
        'To prevent gibberish without killing variety, LLMs use two clean scissors: Top-k keeps ONLY the top k tallest bars, and Top-p (Nucleus Sampling) keeps only the smallest group of leading words whose chances add up to p (like 90%). Watch the 3D cutoff curtain sweep across the skyline as you slide Top-k and Top-p.',
      ],
      practitioner: [
        'Top-k truncates rigidly at rank k regardless of certainty, whereas Top-p (Holtzman et al.) adapts dynamically: when the model is confident (top token = 92%), Top-p = 0.90 keeps just 1 token; when uncertain, it widens to dozens. Surviving probabilities are renormalized to sum to 1.0.',
      ],
      formal: [
        'Nucleus filtering retains minimal prefix V_p = {v_(1), ..., v_(m)} such that Σ_{i≤m} P_τ(v_(i)) ≥ p, intersected with top-k indices, zeroing the long tail before renormalizing P̃(v) = P_τ(v) / Z_{k,p}.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Top-k and Top-p slice off weird low-probability words before rolling the dice so output stays coherent.',
      practitioner:
        'Top-p dynamically adapts candidate pool size to model entropy, cutting off tail hallucinations.',
      formal:
        'Truncated nucleus re-normalization prevents tail sampling errors while preserving local diversity.',
    },
  },
  {
    id: 'ch6-beat-5',
    beatNumber: 5,
    shortTitle: '5. Autoregressive Loop',
    heading: 'One Word at a Time: The Live Autoregressive Writer',
    sceneBeat: 'ch6-5-autoregression',
    visualFocusNote:
      'Click "Step +1 Token" on stage to sample a real token from your active pool, watch it loop back onto your sentence, and trigger a fresh live forward pass!',
    paragraphs: {
      foundations: [
        'Now you know how the model predicts ONE next token. How does an LLM write an entire story, poem, or essay? It simply loops!',
        'Click the blue "Step +1 Token" button at the top of the 3D stage. Our miniature AI rolls the dice on its real distribution, glues the winning token onto the end of your sentence, and feeds that longer sentence right back into Block 0 to predict the next token. Click it 5 or 6 times and watch your real 862k-parameter model write a story live.',
      ],
      practitioner: [
        'Autoregressive decoding generates x_{T+1} ~ P(· | x_{1..T}), appends it to the context, and repeats until emitting `<|eos|>` or hitting `max_seq_len = 64`. Notice why generation errors compound: once a weird token is sampled, it becomes permanent ground-truth input for every future step.',
      ],
      formal: [
        'Sequential ancestral sampling draws x_{t} ~ P̃_θ(· | x_{<t}) for t = T+1, ..., T_max. Because each step depends on the previous discrete sample, decoding is inherently serial along the time axis.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Writing a whole story is just predicting one token, gluing it onto the prompt, and running the machine again.',
      practitioner:
        'Autoregressive looping appends each sampled token to the context window until <|eos|> or T_max.',
      formal:
        'Serial ancestral sampling factorizes sequence generation into T_gen sequential forward evaluations.',
    },
  },
  {
    id: 'ch6-beat-6',
    beatNumber: 6,
    shortTitle: '6. KV Cache & Precision',
    heading: 'How Real Servers Stay Fast: Prefill, KV Cache & Quantization',
    sceneBeat: 'ch6-6-kv-quant',
    visualFocusNote:
      'Blue frosted columns represent cached past Keys & Values frozen in GPU memory; only the single glowing amber column for the newest token is computed on each decode step.',
    paragraphs: {
      foundations: [
        'If you clicked "Step +1 Token" six times, you might notice something wasteful: every time we added ONE new word, our simple browser engine re-calculated all the earlier words from scratch! In a 50-page chat, that would take forever.',
        'Because of the Causal Mask (Chapter 4), earlier words can NEVER look forward at new words—so their Keys (K) and Values (V) never change! Real AI servers process your whole prompt in parallel once (Prefill), freeze those past K and V columns in a notepad called the KV Cache, and during reply writing (Decode) only compute the ONE newest column. Combined with shrinking weight precision (Quantization: FP32 → our 1.65 MB FP16 → FP8 / INT4), that is how modern AI streams replies at 100+ tokens per second.',
      ],
      practitioner: [
        'KV Caching reduces per-token decode attention from O(T²) to O(T), shifting inference from compute-bound (Prefill) to memory-bandwidth-bound (Decode). Storing all 862,464 weights in IEEE 754 float16 (2 bytes/param = 1.65 MB) cuts size 2× vs. FP32; 4-bit quantization (INT4/NF4) shrinks frontier weights ~4× further.',
      ],
      formal: [
        'Caching K^{(l)}_{1..t-1}, V^{(l)}_{1..t-1} trades O(L · T · n_kv · d_head) HBM capacity to avoid redundant O(T²) projections during decode. Our browser engine decodes little-endian FP16 weights to FP32 in src/engine/fp16.ts with verified parity <1e-4.',
      ],
    },
    keyTakeaway: {
      foundations:
        'KV Caching saves past Keys & Values so generation never repeats old work, while Quantization shrinks model weights to fit in memory.',
      practitioner:
        'Prefill is parallel and compute-bound; Decode uses KV Caching (O(T)) and FP16/FP8/INT4 quantization to relieve memory bandwidth.',
      formal:
        'Causal invariance enables O(T) incremental KV-cached decoding, bounded primarily by HBM bandwidth and weight/cache bit-width.',
    },
  },
];

export const CHAPTER_7_BEATS: BookBeat[] = [
  {
    id: 'ch7-beat-1',
    beatNumber: 1,
    shortTitle: '1. Step 0 Static',
    heading: 'Before Training: 862,464 Random Numbers & Blind Guessing',
    sceneBeat: 'ch7-1-step0',
    visualFocusNote:
      'Compare Step 0 (flat gray bars: every one of 512 tokens gets ~0.2% chance, loss ≈ 6.24) against our trained model’s sharp blue bars right in front of it.',
    paragraphs: {
      foundations: [
        'So far we have taken one miraculous fact for granted: all 862,464 weights inside our miniature AI already knew English story patterns. Nobody typed a single rule of grammar into those matrices!',
        'At Step 0—the moment the model was born—every single weight was filled with tiny random noise near zero. Look at the flat gray back row on stage: with random weights, the model assigns almost the exact same ~0.2% chance to all 512 tokens, spitting out pure gibberish. How did random static organize itself into stories?',
      ],
      practitioner: [
        'Weights initialize from N(0, 0.02²) (and 0.02/√(2L) on W_o and W_down). Before a single gradient step, the model outputs near-uniform logits, yielding theoretical initial cross-entropy loss ln(|V|) = ln(512) ≈ 6.238—confirmed verbatim in Checkpoint 0 of public/model/training_replay.json.',
      ],
      formal: [
        'Under isotropic small-variance initialization θ_0 ~ N(0, σ² I), initial predictive entropy H(P_{θ_0}) ≈ ln |V| = 6.238 nats (perplexity ≈ 512). All linguistic structure must be extracted strictly from gradient updates on the training corpus.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Every LLM starts at Step 0 as pure random static guessing blindly among all tokens.',
      practitioner:
        'Random initialization yields uniform predictions with initial cross-entropy loss ≈ ln(|V|) ≈ 6.24.',
      formal:
        'Initial maximum-entropy state L(θ_0) ≈ ln |V| provides a clean, symmetric baseline for stochastic optimization.',
    },
  },
  {
    id: 'ch7-beat-2',
    beatNumber: 2,
    shortTitle: '2. Loss & Perplexity',
    heading: 'Measuring Surprise: Cross-Entropy Loss & Perplexity',
    sceneBeat: 'ch7-2-cross-entropy',
    visualFocusNote:
      'Every bar above your sentence is our real model’s Cross-Entropy Loss (-log P) on predicting that exact next token—tall amber bars mark tokens that surprised the model most!',
    paragraphs: {
      foundations: [
        'To learn from mistakes, the model needs an exact scorecard. As it reads a real story, we ask it to predict EVERY next token along the sentence and grade each guess using Cross-Entropy Loss.',
        'If the true next word was "ball" and the model gave "ball" a 90% chance, its surprise penalty is tiny (0.11). If it gave "ball" only a 1% chance, the penalty shoots way up (4.61)! Look at the 3D bars above your sentence: predictable words have short blue bars, while surprising names or plot twists spike tall in amber. Perplexity simply translates average loss into "how many equally-likely words the model is torn between."',
      ],
      practitioner: [
        'Teacher-forced Cross-Entropy Loss evaluates `L = -(1/(T-1)) Σ log P(x_{t+1} | x_{≤t})` in parallel across every slot of your prompt. Perplexity `PPL = exp(L)` gives an intuitive branching factor: Step 0 has PPL ≈ 512; a well-trained TinyStories model drops below PPL ≈ 6.',
      ],
      formal: [
        'Negative log-likelihood L(θ) = -E_{(x_t, x_{<t})}[ln P_θ(x_t | x_{<t})] equals cross-entropy H(P_data, P_θ), upper-bounding KL divergence from the true data distribution up to constant source entropy. Perplexity exp(L) measures effective vocabulary uncertainty.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Cross-Entropy Loss punishes confident wrong guesses sharply; Perplexity tells you how many words the model is torn between.',
      practitioner:
        'Every prompt token yields a parallel next-token loss term L_t = -log P(x_{t+1}); lowering mean L is the entire training objective.',
      formal:
        'Minimizing teacher-forced cross-entropy directly minimizes D_KL(P_data || P_θ) across all sequence positions.',
    },
  },
  {
    id: 'ch7-beat-3',
    beatNumber: 3,
    shortTitle: '3. Be Backprop Yourself',
    heading: 'Signature Lab: Nudge One Real Weight Down the Loss Valley',
    sceneBeat: 'ch7-3-manual-backprop',
    visualFocusNote:
      'The 3D curve is the REAL Cross-Entropy Loss landscape of ONE single weight in your running model! Drag the "Nudge Weight" slider on stage to roll the amber ball and feel what a gradient is.',
    paragraphs: {
      foundations: [
        'This is the moment training clicks forever. Right now, we isolated ONE real weight inside your running miniature AI’s output matrix—and handed you a manual slider to nudge it left or right.',
        'Drag the "Nudge Weight" slider on stage! As you push that single number up or down, your browser recomputes the EXACT real Cross-Entropy Loss on your sentence in real time, rolling the amber ball along the 3D U-shaped valley. Notice the green slope arrow: that arrow is the GRADIENT (∂L/∂w)—it always points uphill, so training simply steps in the OPPOSITE direction to roll downhill into the valley floor!',
        'Notice something amazing: our trained model starts at Δ = 0.00 already sitting right near the bottom of the valley—that is why it is so hard to beat by hand!',
      ],
      practitioner: [
        'Perturbing a single unembedding weight W_U[target, d] shifts logit z_t[target] by `Δw · lnFOut[t, d]`, yielding an exact convex cross-entropy valley with analytical gradient `∂L/∂w = mean_t((P_t[target] - 1_{y_t=target}) · lnFOut[t, d])`.',
        'Try nudging the weight away from 0.00 to watch the gradient arrow grow steeper, pointing back toward lower loss. Backpropagation computes this exact downhill slope for ALL 862,464 weights simultaneously in one reverse pass.',
      ],
      formal: [
        'Because cross-entropy is convex in the output head weights W_U, holding upstream activations fixed produces a smooth convex 1D slice L(w + δ). Backpropagation generalizes this single-coordinate partial derivative ∂L/∂w to all parameters via reverse-mode automatic differentiation (chain rule) in O(1) forward-pass cost.',
      ],
    },
    keyTakeaway: {
      foundations:
        'A gradient is just the slope of the loss valley for one weight—training rolls every weight downhill toward smaller mistakes.',
      practitioner:
        'Backpropagation computes exact partial derivatives ∂L/∂w for all 862,464 weights in one reverse sweep via the chain rule.',
      formal:
        'Reverse-mode autodiff evaluates ∇_θ L efficiently so AdamW can descend the high-dimensional loss surface.',
    },
    predict: {
      id: 'ch7-b3-predict',
      question:
        'When you nudge the trained weight away from Δ = 0.00, why does the loss usually go UP in either direction instead of getting dramatically lower?',
      options: [
        {
          id: 'a',
          label: 'Because 2,000 steps of AdamW training already steered this weight close to a minimum of the loss valley.',
        },
        {
          id: 'b',
          label: 'Because changing a single weight has zero effect on the logits.',
        },
        {
          id: 'c',
          label: 'Because Cross-Entropy Loss is always a flat horizontal line.',
        },
      ],
      correctId: 'a',
      explanation:
        'Exactly! At convergence, the gradient ∂L/∂w sits near zero at the bottom of the valley—meaning training already found a strong setting for that number.',
    },
  },
  {
    id: 'ch7-beat-4',
    beatNumber: 4,
    shortTitle: '4. Real Training Replay',
    heading: '862,464 Nudges at Once: AdamW & Real Training Replay',
    sceneBeat: 'ch7-4-training-replay',
    visualFocusNote:
      'Scrub the Checkpoint Step slider on stage to replay our model’s REAL recorded cloud training run—watching validation loss plunge while gibberish turns into English stories.',
    paragraphs: {
      foundations: [
        'Imagine doing what you just did with that manual slider—except calculus (Backpropagation) measures the downhill slope for ALL 862,464 weights in one sweep, and an optimizer called AdamW gives every single number a tiny, careful nudge downhill.',
        'Drag the "Training Step" scrubber on stage from Step 0 to the final step. Every number and story sample below is a REAL snapshot saved while this miniature AI trained on GitHub’s cloud server! Watch the blue loss curve sweep down, watch tracked individual weights wiggle from random guesses into their final values, and watch babbling nonsense turn into real stories.',
      ],
      practitioner: [
        'Loaded verbatim from public/model/training_replay.json. Our training loop used AdamW (lr=3e-3, β_1=0.9, β_2=0.95, weight_decay=0.05, grad_clip=1.0, 5% linear warmup + cosine LR decay) over 64-token windows of TinyStories.',
        'Watch the amber vertical cursor move along the real validation loss curve while the recorded completions below evolve from random tokens to fluent syntax.',
      ],
      formal: [
        'AdamW decouples L2 weight decay from adaptive first- and second-moment gradient estimates m_t, v_t, stabilizing Transformer training across heterogeneous parameter scales (embeddings, QKV projections, LayerNorms).',
      ],
    },
    keyTakeaway: {
      foundations:
        'Repeating tiny downhill nudges thousands of times transforms 862,464 random numbers into working language.',
      practitioner:
        'AdamW + cosine decay + gradient clipping drove validation loss from ~6.24 down to fluent TinyStories completions.',
      formal:
        'Stochastic adaptive optimization over next-token NLL self-organizes attention circuits and MLP memories without explicit linguistic rules.',
    },
  },
  {
    id: 'ch7-beat-5',
    beatNumber: 5,
    shortTitle: '5. Why Models Hallucinate',
    heading: 'The Mechanics of Hallucination: Pattern vs. Truth',
    sceneBeat: 'ch7-5-hallucination',
    visualFocusNote:
      'Click "Load Hallucination Specimen" on stage to feed "The capital of France is a little girl named" into our TinyStories model and watch confident pattern completion override geography.',
    paragraphs: {
      foundations: [
        'Now you can finally understand WHY AI sometimes "hallucinates" confident falsehoods—with zero mystery.',
        'Click "Load Hallucination Specimen" on stage! Our miniature AI was trained ONLY on short children’s stories—it has never read an encyclopedia. When we type "The capital of France is a little girl named", it does NOT stop and say "I don’t know geography." Instead, it calmly assigns high probability to a story character’s name—because next-token training rewards plausible text patterns from its training data, not checking a built-in truth database.',
      ],
      practitioner: [
        'Pretraining optimizes next-token likelihood P_data(x_{t+1} | x_{≤t}), not factual verification. Whenever a prompt falls outside the training distribution—or mixes a strong local syntactic pattern ("a little girl named") with an unseen fact ("capital of France")—the model smoothly extrapolates the strongest learned pattern with high confidence.',
      ],
      formal: [
        'Hallucination is a structural consequence of density estimation under distribution shift and finite parameter capacity: minimizing cross-entropy assigns non-zero smooth likelihood mass over plausible continuations without an intrinsic epistemic uncertainty channel unless explicitly aligned via post-training or retrieval (RAG).',
      ],
    },
    keyTakeaway: {
      foundations:
        'LLMs predict plausible patterns from their training data—they do not automatically know when they are guessing outside their knowledge.',
      practitioner:
        'Confident hallucinations emerge mechanically from next-token likelihood optimization under out-of-distribution prompts.',
      formal:
        'Next-token NLL calibrates syntactic plausibility rather than factual grounding, motivating RAG and post-training calibration.',
    },
  },
];

export const CHAPTER_8_BEATS: BookBeat[] = [
  {
    id: 'ch8-beat-1',
    beatNumber: 1,
    shortTitle: '1. Surgical Ablation Lab',
    heading: 'Causal Proof: Be a Neural Neuroscientist',
    sceneBeat: 'ch8-1-ablation',
    visualFocusNote:
      'Use the surgical switches on stage to turn off individual Heads, an entire Attention sub-block, an entire MLP, or a full Layer—and compare live orange bars against wireframe blue baselines.',
    paragraphs: {
      foundations: [
        'You have now seen every single gear inside a real Transformer. Before we zoom out to trillion-parameter AI, let’s run one final scientific proof: surgical ablation.',
        'Use the switches at the top of the 3D stage to unplug different parts of Block 2 (or any block) live! Notice what happens: unplugging ONE attention head barely nudges the prediction (neural redundancy), whereas unplugging the entire MLP or skipping a whole block causes a dramatic probability wobble and cross-entropy loss jump.',
      ],
      practitioner: [
        'Counterfactual ablation proves which sub-circuits are causally load-bearing on your exact prompt. Every intervention executes live inside src/engine/transformer.ts—compare Δloss across single-head ablation, full-MHA ablation, full-MLP ablation, and residual bypass removal.',
      ],
      formal: [
        'Zero-ablation interventions isolate marginal causal contributions ΔL(S) = L(θ_{-S}) - L(θ) across architectural partitions S, revealing distributed redundancy across heads alongside sharp fragility in shared residual and MLP pathways.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Breaking pieces live proves neural networks have backup redundancy across heads, while MLPs and residual highways are deeply load-bearing.',
      practitioner:
        'Live causal ablation measures exact ΔP and ΔL per component directly in the browser.',
      formal:
        'Counterfactual zero-ablation quantifies component necessity and circuit redundancy empirically.',
    },
    predict: {
      id: 'ch8-b1-predict',
      question:
        'In Block 2, which single switch usually causes a BIGGER jump in prompt cross-entropy loss: disabling ONE head (H3) or disabling the entire MLP?',
      options: [
        {
          id: 'a',
          label: 'Disabling the entire MLP usually hurts much more—it removes two-thirds of the block’s parameters at once, whereas other heads partially cover for H3.',
        },
        {
          id: 'b',
          label: 'Disabling H3 always destroys the model completely while disabling the MLP has zero effect.',
        },
        {
          id: 'c',
          label: 'Neither switch changes the output probabilities.',
        },
      ],
      correctId: 'a',
      explanation:
        'Test both switches right above the 3D stage to verify the exact Δloss numbers on your current sentence!',
    },
  },
  {
    id: 'ch8-beat-2',
    beatNumber: 2,
    shortTitle: '2. Scale Cathedral',
    heading: 'Scaling Laws: From 862,464 to 1,000,000,000,000 Parameters',
    sceneBeat: 'ch8-2-scale',
    visualFocusNote:
      'Drag the Scale Ladder slider on stage to watch our miniature model sit beside GPT-2, 8B, 70B, and Frontier (~1T+) architectures across logarithmic towers.',
    paragraphs: {
      foundations: [
        'If our tiny 862,464-parameter model uses the exact same Transformer heartbeat as Gemini, GPT, and Claude, why can’t it write complex software or pass a medical exam?',
        'Drag the Scale slider on stage! Across a 1,000,000× jump in size—growing from 4 layers to 128 layers, from width 128 to 16,384, and training on trillions of words instead of a few thousand short stories—loss drops smoothly along predictable Scaling Laws, and rich abilities like coding, broad factual recall, and multi-step reasoning wake up that a tiny toy can never hold.',
      ],
      practitioner: [
        'Chinchilla scaling laws show optimal loss requires scaling model parameters N and training tokens D together (~20 tokens per parameter for compute-optimal pretraining, and far higher token ratios for inference-optimized small models).',
        'Honesty note: architecture alone does not grant intelligence—scale of parameters, data diversity, and compute unlocks capabilities absent at 862k parameters.',
      ],
      formal: [
        'Empirical power-law scaling L(N, D) ≈ E + A/N^α + B/D^β governs pretraining loss across 6+ orders of magnitude in compute C ≈ 6 N D, driving smooth reductions in perplexity that translate into sharp downstream benchmark phase transitions.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Giant AI runs on the same core Transformer heartbeat—scaled up 1,000,000× in parameters, context, and training data.',
      practitioner:
        'Chinchilla scaling laws link parameters N, tokens D, and compute C; scale unlocks reasoning and recall impossible at 862k params.',
      formal:
        'Power-law loss scaling across N, D, and C bridges miniature reference Transformers to frontier capabilities.',
    },
  },
  {
    id: 'ch8-beat-3',
    beatNumber: 3,
    shortTitle: '3. Modern Frontier Diff',
    heading: 'The Complete Upgrade Map: Miniature GPT → Modern Frontier LLM',
    sceneBeat: 'ch8-3-frontier-diff',
    visualFocusNote:
      'Five paired architectural columns map every component in our verified miniature Transformer (blue) directly to its modern frontier counterpart (amber).',
    paragraphs: {
      foundations: [
        'Everything you learned across Chapters 1 through 7 transfers directly to modern AI—with five clean engineering upgrades designed to handle million-word prompts and faster chips:',
        '1. Learned wpe → RoPE rotations (Ch. 2). 2. LayerNorm → faster RMSNorm (Ch. 3). 3. Standard MHA → Grouped-Query Attention + KV Cache (Ch. 4 & 6). 4. Dense GELU MLP → SwiGLU + Mixture of Experts (Ch. 5). 5. Raw next-token pretraining → Post-Training Alignment (next beat!).',
      ],
      practitioner: [
        'Every single frontier upgrade targets one of two real hardware walls: long-context quadratic cost / KV-cache VRAM (RoPE, GQA, MLA, PagedAttention) or FLOPs/memory-bandwidth efficiency (RMSNorm, SwiGLU, sparse MoE, FP8/INT4 quantization).',
      ],
      formal: [
        'Modern decoder-only Transformers preserve the Pre-Norm residual attention + FFN topology while optimizing hardware arithmetic intensity and length generalization through SO(2) RoPE, bias-free RMSNorm, low-rank/grouped KV sharing, gated FFNs, and sparse MoE routing.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Modern LLMs keep the exact same core blueprint, swapping in 5 targeted upgrades for long context, speed, and alignment.',
      practitioner:
        'RoPE, RMSNorm, GQA+KV Cache, SwiGLU+MoE, and Post-Training directly bridge our miniature model to production LLMs.',
      formal:
        'Frontier architectures refine normalization, positional kernels, KV memory, and sparse routing atop the identical residual Transformer spine.',
    },
  },
  {
    id: 'ch8-beat-4',
    beatNumber: 4,
    shortTitle: '4. Post-Training & RLHF',
    heading: 'Turning a Text Completer Into a Helpful Thinker: SFT & RL',
    sceneBeat: 'ch8-4-post-training',
    visualFocusNote:
      'Watch the 4 pillars show how Post-Training reshapes the exact same Transformer weights from raw internet mimicry into a helpful, safe, step-by-step reasoner.',
    paragraphs: {
      foundations: [
        'Even after pretraining on trillions of words, a raw "base model" still only behaves like an autocomplete engine—if you ask it "What is photosynthesis?", it might reply with a list of more quiz questions instead of answering!',
        'To turn a raw text completer into a helpful assistant, engineers run Post-Training on the EXACT same Transformer weights: first Supervised Fine-Tuning (SFT) shows it thousands of helpful Q&A dialogues with Chat Templates; next Preference Alignment (RLHF / DPO) rewards clear, honest, safe replies; finally Reasoning RL trains it to spend extra autoregressive tokens thinking step-by-step before answering tough math or coding problems.',
      ],
      practitioner: [
        'Post-training never changes the Transformer architecture—it shifts the learned probability policy π_θ(y | x): (1) SFT teaches assistant formatting and tool use, (2) RLHF / DPO aligns tone, refusal boundaries, and calibration against human/AI preference pairs, and (3) RL with verifiable rewards (math/code unit tests) induces emergent chain-of-thought self-verification.',
      ],
      formal: [
        'Post-training optimizes policy π_θ(y|x) over the pretrained initialization π_ref via SFT cross-entropy followed by KL-regularized reward maximization max_θ E[R(x,y)] - β D_KL(π_θ || π_ref) (PPO / GRPO / DPO), steering generation toward calibrated, verifiable multi-step reasoning.',
      ],
    },
    keyTakeaway: {
      foundations:
        'Post-training (SFT, RLHF, and Reasoning RL) reshapes the exact same weights from raw autocomplete into a helpful, reasoning assistant.',
      practitioner:
        'SFT instills chat format, RLHF/DPO aligns preferences and safety, and verifiable-reward RL unlocks test-time reasoning.',
      formal:
        'KL-constrained policy optimization shifts the pretrained distribution toward helpfulness, truthfulness, and verifiable reasoning chains.',
    },
  },
  {
    id: 'ch8-beat-5',
    beatNumber: 5,
    shortTitle: '5. Capstone Workbench',
    heading: 'You Now Understand the Complete Machine',
    sceneBeat: 'ch8-5-capstone',
    visualFocusNote:
      'All 4 blocks, the unbroken Residual Highway, live Logit Lens probes, and the final probability skyline sit together in one unified 3D cutaway—ready for free exploration or classroom lectures.',
    paragraphs: {
      foundations: [
        'Step back and look at the entire sunlit machine on stage. Nothing inside it is a mystery box anymore! You watched raw text slice into BPE tokens, map into real 3D embedding geometry, ride an unbroken Residual Highway through Pre-LayerNorm, mix context through 6 beats of Self-Attention, think privately through 512 GELU neurons, sample next tokens in an autoregressive loop, and learn every single one of its 862,464 weights by rolling downhill along Cross-Entropy Loss.',
        'Try typing any custom sentence into the top bar, toggle Lecture Mode for full-screen classroom projection, or click Share to send anyone a deep link straight to any chapter, beat, or experiment.',
      ],
      practitioner: [
        'Every number across all 8 chapters was computed live in your browser from real float16 weights verified to <1e-4 against PyTorch. Use the Contents menu to jump directly to any lab—PCA cloud, 6-Beat Attention, GELU sparsity, Autoregressive Writer, Manual Backprop, Training Replay, or Surgical Ablation—for teaching or debugging intuition.',
      ],
      formal: [
        'You have traversed the full lifecycle of an autoregressive Transformer: tokenization E, distributed embeddings W_E + W_P, Pre-LN residual blocks (MHA + FFN), tied unembedding, stochastic decoding, NLL/AdamW optimization, counterfactual ablation, scaling laws, and post-training policy alignment.',
      ],
    },
    keyTakeaway: {
      foundations:
        'An LLM is not magic—it is a transparent, understandable symphony of simple mathematical steps working together at scale.',
      practitioner:
        'Use Lecture Mode and Share links to bring any live beat or ablation experiment straight into classrooms and workshops.',
      formal:
        'End-to-end mastery of Transformer mechanics—grounded in exact, auditable tensors from token IDs to scalar gradients.',
    },
  },
];
