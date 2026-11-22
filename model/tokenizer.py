from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Identical regex used in browser TypeScript for 100% parity
PRETOKEN_REGEX = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
PRETOKEN_RE = re.compile(PRETOKEN_REGEX)

EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"


def normalize_text(text: str) -> str:
    """Normalize smart punctuation to clean ASCII."""
    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


@dataclass
class EncodedToken:
    id: int
    text: str
    start: int
    end: int


class TinyBPETokenizer:
    def __init__(
        self,
        vocab: List[str],
        merges: List[Tuple[str, str]],
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.merge_ranks: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 512) -> "TinyBPETokenizer":
        # Base vocabulary: special tokens + newline + printable ASCII (32..126) = 98 tokens
        base_vocab = [EOS_TOKEN, UNK_TOKEN, "\n"] + [chr(i) for i in range(32, 127)]
        base_set = set(base_vocab)

        word_freqs: Counter[Tuple[str, ...]] = Counter()
        for raw in texts:
            text = normalize_text(raw)
            for match in PRETOKEN_RE.finditer(text):
                piece = match.group(0)
                symbols = tuple(ch if ch in base_set else UNK_TOKEN for ch in piece)
                word_freqs[symbols] += 1

        vocab = list(base_vocab)
        merges: List[Tuple[str, str]] = []
        num_merges = vocab_size - len(vocab)

        for _ in range(num_merges):
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for symbols, freq in word_freqs.items():
                if len(symbols) < 2:
                    continue
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    if UNK_TOKEN in pair or EOS_TOKEN in pair:
                        continue
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair, best_count = min(
                pair_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if best_count < 2:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab.append(merged_symbol)

            new_word_freqs: Counter[Tuple[str, ...]] = Counter()
            a, b = best_pair
            for symbols, freq in word_freqs.items():
                if len(symbols) < 2:
                    new_word_freqs[symbols] += freq
                    continue
                merged: List[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        merged.append(merged_symbol)
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                new_word_freqs[tuple(merged)] += freq
            word_freqs = new_word_freqs

        return cls(vocab=vocab, merges=merges)

    def encode_with_spans(self, text: str) -> List[EncodedToken]:
        """Encode string into tokens with exact [start, end) character spans."""
        text = normalize_text(text)
        result: List[EncodedToken] = []

        for match in PRETOKEN_RE.finditer(text):
            piece = match.group(0)
            cursor = match.start()

            items: List[Tuple[str, int]] = [
                (ch if ch in self.token_to_id else UNK_TOKEN, 1) for ch in piece
            ]

            while len(items) > 1:
                best_rank = None
                best_pair = None
                for i in range(len(items) - 1):
                    pair = (items[i][0], items[i + 1][0])
                    rank = self.merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = pair

                if best_pair is None:
                    break

                a, b = best_pair
                next_items: List[Tuple[str, int]] = []
                i = 0
                while i < len(items):
                    if (
                        i < len(items) - 1
                        and items[i][0] == a
                        and items[i + 1][0] == b
                    ):
                        next_items.append((a + b, items[i][1] + items[i + 1][1]))
                        i += 2
                    else:
                        next_items.append(items[i])
                        i += 1
                items = next_items

            for sym, char_len in items:
                result.append(
                    EncodedToken(
                        id=self.token_to_id.get(sym, self.unk_id),
                        text=sym,
                        start=cursor,
                        end=cursor + char_len,
                    )
                )
                cursor += char_len

        return result

    def encode(self, text: str) -> List[int]:
        return [tok.id for tok in self.encode_with_spans(text)]

    def decode(self, ids: List[int]) -> str:
        out: List[str] = []
        for idx in ids:
            tok = self.vocab[idx]
            if tok == EOS_TOKEN:
                break
            out.append(tok)
        return "".join(out)

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "vocab_size": len(self.vocab),
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "pretokenize_regex": PRETOKEN_REGEX,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TinyBPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vocab=data["vocab"],
            merges=[(a, b) for a, b in data["merges"]],
        )
