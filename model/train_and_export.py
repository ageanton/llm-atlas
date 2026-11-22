from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset

from tokenizer import TinyBPETokenizer
from transformer import AblationConfig, ModelConfig, TinyTransformer

REPLAY_PROMPTS = [
    "Once upon a time, a little dog named Max found a red ball. He",
    "Lily wanted to bake a cake with her mom, so she",
    "The cat sat on the mat and looked at the",
]

TRACKED_SCALARS: List[Tuple[str, Tuple[int, ...], str]] = [
    ("wte.weight", (100, 0), "Token embedding feature 0 for a common word token"),
    ("wpe.weight", (0, 0), "Position 0 embedding dimension 0"),
    ("blocks.0.attn.W_q.weight", (0, 0), "Block 0 Query projection entry (0, 0)"),
    ("blocks.0.attn.W_k.weight", (4, 12), "Block 0 Key projection entry (4, 12)"),
    ("blocks.0.attn.W_v.weight", (16, 8), "Block 0 Value projection entry (16, 8)"),
    ("blocks.1.attn.W_q.weight", (32, 7), "Block 1 Head 1 Query weight (32, 7)"),
    ("blocks.1.attn.W_o.weight", (11, 45), "Block 1 Output projection weight (11, 45)"),
    ("blocks.2.attn.W_k.weight", (64, 19), "Block 2 Head 2 Key weight (64, 19)"),
    ("blocks.2.mlp.W_up.weight", (120, 3), "Block 2 MLP expansion neuron 120, dim 3"),
    ("blocks.2.mlp.W_down.weight", (3, 120), "Block 2 MLP projection dim 3 from neuron 120"),
    ("blocks.3.attn.W_o.weight", (96, 14), "Block 3 Output projection weight (96, 14)"),
    ("ln_f.weight", (0,), "Final LayerNorm learned scale for dimension 0"),
]


def describe_param(name: str) -> str:
    if name == "wte.weight":
        return "Token embedding lookup table (also tied as final vocabulary output head)."
    if name == "wpe.weight":
        return "Learned positional offsets added to each token based on its slot (0..63)."
    if name == "ln_f.weight":
        return "Final LayerNorm learned per-dimension scale before vocabulary logits."
    if name == "ln_f.bias":
        return "Final LayerNorm learned per-dimension shift before vocabulary logits."
    parts = name.split(".")
    layer = parts[1]
    sub = ".".join(parts[2:])
    roles = {
        "ln_1.weight": f"Block {layer} pre-attention LayerNorm scale.",
        "ln_1.bias": f"Block {layer} pre-attention LayerNorm bias.",
        "attn.W_q.weight": f"Block {layer} Query matrix: projects tokens into what they are looking for.",
        "attn.W_k.weight": f"Block {layer} Key matrix: projects tokens into what they advertise to others.",
        "attn.W_v.weight": f"Block {layer} Value matrix: projects the information a token passes along.",
        "attn.W_o.weight": f"Block {layer} Output projection: blends all 4 attention heads back into the residual stream.",
        "ln_2.weight": f"Block {layer} pre-MLP LayerNorm scale.",
        "ln_2.bias": f"Block {layer} pre-MLP LayerNorm bias.",
        "mlp.W_up.weight": f"Block {layer} MLP expansion matrix (128 -> 512) before GELU.",
        "mlp.W_down.weight": f"Block {layer} MLP contraction matrix (512 -> 128) back into the residual stream.",
    }
    return roles.get(sub, f"Parameter {name}")


def load_stories(num_train: int, num_val: int, local_corpus: Path | None) -> Tuple[List[str], List[str]]:
    if local_corpus is not None:
        raw = local_corpus.read_text(encoding="utf-8")
        chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 40]
        split = max(1, int(len(chunks) * 0.95))
        return chunks[:split], chunks[split:]

    print(f"Streaming {num_train} train + {num_val} val stories from roneneldan/TinyStories...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    train_stories = [row["text"] for _, row in zip(range(num_train), train_ds)]
    val_stories = [row["text"] for _, row in zip(range(num_val), val_ds)]
    return train_stories, val_stories


def encode_corpus(tokenizer: TinyBPETokenizer, stories: List[str]) -> np.ndarray:
    ids: List[int] = []
    for story in stories:
        ids.extend(tokenizer.encode(story))
        ids.append(tokenizer.eos_id)
    return np.array(ids, dtype=np.int64)


def get_batch(
    data: np.ndarray, batch_size: int, seq_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = np.random.randint(0, len(data) - seq_len - 1, size=(batch_size,))
    x = np.stack([data[i : i + seq_len] for i in idx])
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in idx])
    return (
        torch.from_numpy(x).to(device, dtype=torch.long),
        torch.from_numpy(y).to(device, dtype=torch.long),
    )


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    iters: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(data, batch_size, seq_len, device)
        _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def capture_checkpoint(
    step: int,
    train_loss: float,
    val_loss: float,
    model: TinyTransformer,
    tokenizer: TinyBPETokenizer,
) -> Dict[str, Any]:
    model.eval()
    state = dict(model.named_parameters())

    scalars = {}
    for pname, coord, _ in TRACKED_SCALARS:
        key = f"{pname}{list(coord)}"
        scalars[key] = round(float(state[pname][coord].item()), 6)

    prompt_snapshots = []
    for prompt in REPLAY_PROMPTS:
        ids = tokenizer.encode(prompt)
        gen_ids = model.generate_greedy(ids, max_new_tokens=20, eos_id=tokenizer.eos_id)
        completion = tokenizer.decode(gen_ids[len(ids) :])

        logits, _ = model(
            torch.tensor([ids], dtype=torch.long, device=model.wte.weight.device)
        )
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_vals, top_idx = torch.topk(probs, k=5)
        prompt_snapshots.append(
            {
                "prompt": prompt,
                "completion": completion,
                "top5": [
                    {
                        "id": int(idx),
                        "token": tokenizer.vocab[int(idx)],
                        "prob": round(float(val), 5),
                    }
                    for idx, val in zip(top_idx.tolist(), top_vals.tolist())
                ],
            }
        )

    model.train()
    return {
        "step": step,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "scalars": scalars,
        "prompts": prompt_snapshots,
    }


def export_fp16_weights(
    model: TinyTransformer, out_dir: Path
) -> Tuple[Dict[str, Any], bytes]:
    """Export all parameters to a contiguous little-endian float16 binary + manifest."""
    chunks: List[bytes] = []
    tensors_meta: List[Dict[str, Any]] = []
    offset = 0

    for name, param in model.named_parameters():
        arr_f32 = param.detach().cpu().numpy().astype(np.float32)
        arr_f16 = arr_f32.astype("<f2")
        raw_bytes = arr_f16.tobytes()

        deq = arr_f16.astype(np.float32)
        hist_counts, hist_edges = np.histogram(deq, bins=20)

        tensors_meta.append(
            {
                "name": name,
                "shape": list(arr_f16.shape),
                "count": int(arr_f16.size),
                "dtype": "float16",
                "offset_bytes": offset,
                "byte_length": len(raw_bytes),
                "role": describe_param(name),
                "stats": {
                    "min": round(float(deq.min()), 6),
                    "max": round(float(deq.max()), 6),
                    "mean": round(float(deq.mean()), 6),
                    "std": round(float(deq.std()), 6),
                },
                "histogram": {
                    "counts": [int(c) for c in hist_counts.tolist()],
                    "edges": [round(float(e), 5) for e in hist_edges.tolist()],
                },
            }
        )
        chunks.append(raw_bytes)
        offset += len(raw_bytes)

    blob = b"".join(chunks)
    sha256 = hashlib.sha256(blob).hexdigest()[:16]
    (out_dir / "weights.fp16.bin").write_bytes(blob)

    manifest = {
        "version": 1,
        "weights_file": "weights.fp16.bin",
        "weights_sha256": sha256,
        "total_bytes": len(blob),
        "total_params": model.param_count(),
        "config": {
            "vocab_size": model.cfg.vocab_size,
            "max_seq_len": model.cfg.max_seq_len,
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_heads": model.cfg.n_heads,
            "d_head": model.cfg.d_head,
            "d_mlp": model.cfg.d_mlp,
            "ln_eps": model.cfg.ln_eps,
            "gelu": "tanh_approx",
            "tied_embeddings": True,
        },
        "tensors": tensors_meta,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest, blob


def load_dequantized_fp16_into_model(
    model: TinyTransformer, manifest: Dict[str, Any], blob: bytes
) -> None:
    """Reload exact dequantized fp16 weights so reference_trace matches the browser bit-for-bit."""
    state = dict(model.named_parameters())
    with torch.no_grad():
        for meta in manifest["tensors"]:
            start = meta["offset_bytes"]
            end = start + meta["byte_length"]
            arr = np.frombuffer(blob[start:end], dtype="<f2").astype(np.float32)
            arr = arr.reshape(meta["shape"])
            state[meta["name"]].copy_(torch.from_numpy(arr))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export the TinyStories miniature GPT."
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-stories", type=int, default=12000)
    parser.add_argument("--val-stories", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("../public/model"))
    parser.add_argument("--local-corpus", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Quick 300-step smoke test")
    args = parser.parse_args()

    if args.fast:
        args.steps = 300
        args.train_stories = 1500
        args.val_stories = 100

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_stories, val_stories = load_stories(
        args.train_stories, args.val_stories, args.local_corpus
    )
    print("Training 512-token BPE tokenizer...")
    tokenizer = TinyBPETokenizer.train(train_stories[:5000], vocab_size=512)
    tokenizer.save(args.out_dir / "tokenizer.json")
    print(f"Tokenizer ready ({len(tokenizer.vocab)} tokens). Encoding corpus...")

    train_tokens = encode_corpus(tokenizer, train_stories)
    val_tokens = encode_corpus(tokenizer, val_stories)
    print(
        f"Encoded {len(train_tokens):,} train tokens and {len(val_tokens):,} val tokens."
    )

    cfg = ModelConfig(vocab_size=len(tokenizer.vocab))
    model = TinyTransformer(cfg).to(device)
    print(f"Initialized TinyTransformer with exactly {model.param_count():,} parameters.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    raw_ckpts = np.unique(
        np.round(np.geomspace(1, args.steps, num=9)).astype(int)
    ).tolist()
    checkpoint_steps = set([0] + raw_ckpts + [args.steps])

    checkpoints: List[Dict[str, Any]] = []
    loss_curve: List[Dict[str, Any]] = []

    t0 = time.time()
    init_train = estimate_loss(
        model, train_tokens, args.batch_size, cfg.max_seq_len, device
    )
    init_val = estimate_loss(
        model, val_tokens, args.batch_size, cfg.max_seq_len, device
    )
    checkpoints.append(capture_checkpoint(0, init_train, init_val, model, tokenizer))
    loss_curve.append(
        {"step": 0, "train_loss": round(init_train, 4), "val_loss": round(init_val, 4)}
    )
    print(f"Step    0 | train {init_train:.4f} | val {init_val:.4f}")

    warmup = max(20, int(args.steps * 0.05))
    for step in range(1, args.steps + 1):
        if step < warmup:
            lr = args.lr * (step / warmup)
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch(train_tokens, args.batch_size, cfg.max_seq_len, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step in checkpoint_steps:
            val_loss = estimate_loss(
                model, val_tokens, args.batch_size, cfg.max_seq_len, device
            )
            tr_loss = float(loss.item())
            loss_curve.append(
                {
                    "step": step,
                    "train_loss": round(tr_loss, 4),
                    "val_loss": round(val_loss, 4),
                }
            )

            if step in checkpoint_steps:
                checkpoints.append(
                    capture_checkpoint(step, tr_loss, val_loss, model, tokenizer)
                )
                elapsed = time.time() - t0
                print(
                    f"Step {step:4d} | train {tr_loss:.4f} | val {val_loss:.4f} | lr {lr:.5f} | {elapsed:.1f}s"
                )

    model.cpu()
    manifest, blob = export_fp16_weights(model, args.out_dir)
    load_dequantized_fp16_into_model(model, manifest, blob)

    replay_payload = {
        "version": 1,
        "tracked_scalars_meta": [
            {
                "key": f"{p}{list(c)}",
                "param": p,
                "coord": list(c),
                "description": desc,
            }
            for p, c, desc in TRACKED_SCALARS
        ],
        "loss_curve": loss_curve,
        "checkpoints": checkpoints,
    }
    (args.out_dir / "training_replay.json").write_text(
        json.dumps(replay_payload, indent=2), encoding="utf-8"
    )

    ref_prompt = REPLAY_PROMPTS[0]
    ref_tokens = tokenizer.encode_with_spans(ref_prompt)
    ref_ids = [t.id for t in ref_tokens]

    baseline_trace = model.trace_forward(ref_ids, ablation=None)
    ablated_trace = model.trace_forward(
        ref_ids, ablation=AblationConfig(disabled_heads={(2, 3)})
    )

    ref_payload = {
        "prompt": ref_prompt,
        "tokens": [
            {"id": t.id, "text": t.text, "start": t.start, "end": t.end}
            for t in ref_tokens
        ],
        "baseline": baseline_trace,
        "ablation_disable_L2_H3": ablated_trace,
    }
    (args.out_dir / "reference_trace.json").write_text(
        json.dumps(ref_payload, indent=2), encoding="utf-8"
    )

    final_ckpt = checkpoints[-1]
    sample_lines = []
    for item in final_ckpt["prompts"]:
        sample_lines.append(
            f'- **Prompt:** `"{item["prompt"]}"`\n  - **Greedy completion:** `"{item["completion"]}"`'
        )

    model_md = f"""# Miniature Transformer Audit Card (`MODEL.md`)

- **Architecture:** Decoder-only Pre-LN GPT (weight-tied input/output embeddings, `bias=False` on linear layers, `tanh`-approx GELU)
- **Exact Parameter Count:** `{model.param_count():,}`
- **Exported Weights:** `weights.fp16.bin` (`{len(blob):,}` bytes, sha256 prefix `{manifest["weights_sha256"]}`)
- **Vocabulary:** `{cfg.vocab_size}` tokens (BPE trained on TinyStories)
- **Dimensions:** `{cfg.n_layers}` layers, `d_model = {cfg.d_model}`, `{cfg.n_heads}` heads (`d_head = {cfg.d_head}`), `d_mlp = {cfg.d_mlp}`, context length `{cfg.max_seq_len}`
- **Training Corpus:** `{"local file" if args.local_corpus else "roneneldan/TinyStories"}` (`{len(train_tokens):,}` train tokens, `{len(val_tokens):,}` val tokens)
- **Training Steps:** `{args.steps}` (batch size `{args.batch_size}`, AdamW + cosine decay)
- **Final Loss:** train `{final_ckpt["train_loss"]}` / validation `{final_ckpt["val_loss"]}` (untrained step 0 was `{checkpoints[0]["val_loss"]}`)

## Unedited Greedy Samples (Post-Quantization)

{chr(10).join(sample_lines)}
"""
    (args.out_dir / "MODEL.md").write_text(model_md, encoding="utf-8")
    print(f"\nDone. All Milestone 1 artifacts written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
