# Workspace Rules for llm-atlas

These rules are non-negotiable across all sessions and milestones:

1. **No Fake Numbers:** Never hardcode, mock, or synthesize attention weights, logits, probabilities, or parameter tensors. Every displayed number must come from the real trained weights (`public/model/weights.fp16.bin`). If weights fail to load, show an explicit error state.
2. **No Inline Explanatory Copy:** All three-register explanations (`simple`, `sufficient`, `complete`), honesty badges (`literal` / `abstraction`), and `createsQuestion` / `answersQuestion` links must live in `src/content/`, never hardcoded inside UI components.
3. **Never Modify Verified Math:** Once the TypeScript inference engine passes the parity test against `public/model/reference_trace.json` (max error < 1e-4), do not refactor or alter its math unless explicitly instructed.
4. **No Local Python Training:** All model training runs in GitHub Actions (`.github/workflows/train-model.yml`). Never install PyTorch locally.
5. **Verbatim Task Packets:** When given exact file contents to write, write them verbatim without shortening or adding unrequested features, gradients, or decorative illustrations.
