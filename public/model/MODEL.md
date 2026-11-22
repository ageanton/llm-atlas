# Miniature Transformer Audit Card (`MODEL.md`)

- **Architecture:** Decoder-only Pre-LN GPT (weight-tied input/output embeddings, `bias=False` on linear layers, `tanh`-approx GELU)
- **Exact Parameter Count:** `862,464`
- **Exported Weights:** `weights.fp16.bin` (`1,724,928` bytes, sha256 prefix `c05e870e742f9f65`)
- **Vocabulary:** `512` tokens (BPE trained on TinyStories)
- **Dimensions:** `4` layers, `d_model = 128`, `4` heads (`d_head = 32`), `d_mlp = 512`, context length `64`
- **Training Corpus:** `roneneldan/TinyStories` (`489,260` train tokens, `25,383` val tokens)
- **Training Steps:** `300` (batch size `64`, AdamW + cosine decay)
- **Final Loss:** train `3.0757` / validation `3.0317` (untrained step 0 was `6.2613`)

## Unedited Greedy Samples (Post-Quantization)

- **Prompt:** `"Once upon a time, a little dog named Max found a red ball. He"`
  - **Greedy completion:** `" was a big, and he was very happy.

One day, the park, the park,"`
- **Prompt:** `"Lily wanted to bake a cake with her mom, so she"`
  - **Greedy completion:** `" had a biggry. She was very happy and said, "It't want to"`
- **Prompt:** `"The cat sat on the mat and looked at the"`
  - **Greedy completion:** `" peached.

The little girl was so happy and the little girl was so happy"`
