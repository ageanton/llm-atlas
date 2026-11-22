from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 512
    max_seq_len: int = 64
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    ln_eps: float = 1e-5

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class AblationConfig:
    disabled_heads: Set[Tuple[int, int]] = field(default_factory=set)
    disabled_attn_blocks: Set[int] = field(default_factory=set)
    disabled_mlp_blocks: Set[int] = field(default_factory=set)
    disabled_residuals: Set[Tuple[str, int]] = field(default_factory=set)
    disabled_layer_norms: Set[Tuple[str, int]] = field(default_factory=set)
    disabled_layers: Set[int] = field(default_factory=set)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Explicit tanh-approximation GELU for exact cross-language parity with TypeScript."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


class ExplicitLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        ablation: Optional[AblationConfig] = None,
        record: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        B, T, C = x.shape
        H = self.cfg.n_heads
        D = self.cfg.d_head

        q = self.W_q(x).view(B, T, H, D).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        masked_scores = raw_scores.masked_fill(causal_mask, -1e9)
        attn_weights = F.softmax(masked_scores, dim=-1)
        weighted_v = torch.matmul(attn_weights, v)

        if ablation:
            for h in range(H):
                if (layer_idx, h) in ablation.disabled_heads:
                    weighted_v[:, h, :, :] = 0.0

        concat = weighted_v.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(concat)

        if ablation and layer_idx in ablation.disabled_attn_blocks:
            out = torch.zeros_like(out)

        trace = None
        if record:
            trace = {
                "q": q[0].tolist(),
                "k": k[0].tolist(),
                "v": v[0].tolist(),
                "raw_scores": raw_scores[0].tolist(),
                "masked_scores": masked_scores[0].tolist(),
                "attn_weights": attn_weights[0].tolist(),
                "weighted_v": weighted_v[0].tolist(),
                "output": out[0].tolist(),
            }
        return out, trace


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.W_up = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.W_down = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        ablation: Optional[AblationConfig] = None,
        record: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        pre_act = self.W_up(x)
        hidden = gelu_tanh(pre_act)
        out = self.W_down(hidden)

        if ablation and layer_idx in ablation.disabled_mlp_blocks:
            out = torch.zeros_like(out)

        trace = None
        if record:
            trace = {
                "hidden": hidden[0].tolist(),
                "output": out[0].tolist(),
            }
        return out, trace


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = ExplicitLayerNorm(cfg.d_model, eps=cfg.ln_eps)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln_2 = ExplicitLayerNorm(cfg.d_model, eps=cfg.ln_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        ablation: Optional[AblationConfig] = None,
        record: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if ablation and layer_idx in ablation.disabled_layers:
            if not record:
                return x, None
            return x, {"skipped": True, "residual_out": x[0].tolist()}

        res_before_attn = x
        if ablation and ("ln_1", layer_idx) in ablation.disabled_layer_norms:
            ln1_out = x
        else:
            ln1_out = self.ln_1(x)

        attn_out, attn_trace = self.attn(ln1_out, layer_idx, ablation, record)

        if ablation and ("attn", layer_idx) in ablation.disabled_residuals:
            res_after_attn = attn_out
        else:
            res_after_attn = res_before_attn + attn_out

        if ablation and ("ln_2", layer_idx) in ablation.disabled_layer_norms:
            ln2_out = res_after_attn
        else:
            ln2_out = self.ln_2(res_after_attn)

        mlp_out, mlp_trace = self.mlp(ln2_out, layer_idx, ablation, record)

        if ablation and ("mlp", layer_idx) in ablation.disabled_residuals:
            res_after_mlp = mlp_out
        else:
            res_after_mlp = res_after_attn + mlp_out

        trace = None
        if record:
            trace = {
                "skipped": False,
                "residual_before_attn": res_before_attn[0].tolist(),
                "ln_1_out": ln1_out[0].tolist(),
                "attn": attn_trace,
                "residual_after_attn": res_after_attn[0].tolist(),
                "ln_2_out": ln2_out[0].tolist(),
                "mlp": mlp_trace,
                "residual_after_mlp": res_after_mlp[0].tolist(),
            }
        return res_after_mlp, trace


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = ExplicitLayerNorm(cfg.d_model, eps=cfg.ln_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if name.endswith("W_o") or name.endswith("W_down"):
                    std = 0.02 / math.sqrt(2.0 * self.cfg.n_layers)
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.wte(token_ids) + self.wpe(pos)
        for i, block in enumerate(self.blocks):
            x, _ = block(x, layer_idx=i, ablation=None, record=False)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def trace_forward(
        self,
        token_ids: List[int],
        ablation: Optional[AblationConfig] = None,
    ) -> Dict[str, Any]:
        """Run a single-sequence forward pass capturing every intermediate tensor + Logit Lens."""
        self.eval()
        token_ids = token_ids[: self.cfg.max_seq_len]
        device = self.wte.weight.device
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
        T = ids_t.shape[1]
        pos = torch.arange(T, device=device).unsqueeze(0)

        tok_emb = self.wte(ids_t)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        layer_traces: List[Dict[str, Any]] = []
        logit_lens: List[Dict[str, Any]] = []

        for i, block in enumerate(self.blocks):
            x, b_trace = block(x, layer_idx=i, ablation=ablation, record=True)
            layer_traces.append(b_trace)

            lens_norm = self.ln_f(x[:, -1:, :])
            lens_logits = F.linear(lens_norm, self.wte.weight)[0, 0]
            lens_probs = F.softmax(lens_logits, dim=-1)
            top_vals, top_idx = torch.topk(lens_probs, k=5)
            logit_lens.append(
                {
                    "after_layer": i,
                    "top_tokens": [
                        {"id": int(idx), "prob": float(val)}
                        for idx, val in zip(top_idx.tolist(), top_vals.tolist())
                    ],
                }
            )

        if ablation and ("ln_f", -1) in ablation.disabled_layer_norms:
            ln_f_out = x
        else:
            ln_f_out = self.ln_f(x)

        logits = F.linear(ln_f_out, self.wte.weight)[0]
        next_logits = logits[-1]
        next_probs = F.softmax(next_logits, dim=-1)

        input_loss = None
        if T > 1:
            input_loss = float(F.cross_entropy(logits[:-1], ids_t[0, 1:]).item())

        return {
            "token_ids": token_ids,
            "tok_emb": tok_emb[0].tolist(),
            "pos_emb": pos_emb[0].tolist(),
            "layers": layer_traces,
            "logit_lens": logit_lens,
            "ln_f_out": ln_f_out[0].tolist(),
            "final_logits": next_logits.tolist(),
            "final_probs": next_probs.tolist(),
            "greedy_next_token_id": int(torch.argmax(next_probs).item()),
            "input_loss": input_loss,
        }

    @torch.no_grad()
    def generate_greedy(
        self, token_ids: List[int], max_new_tokens: int = 32, eos_id: int = 0
    ) -> List[int]:
        self.eval()
        device = self.wte.weight.device
        out = list(token_ids)
        for _ in range(max_new_tokens):
            ctx = out[-self.cfg.max_seq_len :]
            logits, _ = self.forward(torch.tensor([ctx], dtype=torch.long, device=device))
            next_id = int(torch.argmax(logits[0, -1]).item())
            out.append(next_id)
            if next_id == eos_id:
                break
        return out
