import type { EncodedToken, TokenizerData } from './types';

const PRETOKEN_RE = / ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
const EOS_TOKEN = '<|eos|>';
const UNK_TOKEN = '<|unk|>';

export function normalizeText(text: string): string {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\u2018/g, "'")
    .replace(/\u2019/g, "'")
    .replace(/\u201c/g, '"')
    .replace(/\u201d/g, '"')
    .replace(/\u2014/g, '-')
    .replace(/\u2013/g, '-')
    .replace(/\u2026/g, '...');
}

export class TinyBPETokenizer {
  readonly vocab: string[];
  readonly eosId: number;
  readonly unkId: number;
  private readonly tokenToId: Map<string, number>;
  private readonly mergeRanks: Map<string, number>;

  constructor(data: TokenizerData) {
    this.vocab = data.vocab;
    this.eosId = data.eos_id;
    this.unkId = data.unk_id;
    this.tokenToId = new Map();
    for (let i = 0; i < data.vocab.length; i++) {
      this.tokenToId.set(data.vocab[i], i);
    }
    this.mergeRanks = new Map();
    for (let rank = 0; rank < data.merges.length; rank++) {
      const [a, b] = data.merges[rank];
      this.mergeRanks.set(`${a}\u0000${b}`, rank);
    }
  }

  encodeWithSpans(rawText: string): EncodedToken[] {
    const text = normalizeText(rawText);
    const result: EncodedToken[] = [];

    for (const match of text.matchAll(PRETOKEN_RE)) {
      const piece = match[0];
      let cursor = match.index ?? 0;

      let items: [string, number][] = [];
      for (let i = 0; i < piece.length; i++) {
        const ch = piece[i];
        items.push([this.tokenToId.has(ch) ? ch : UNK_TOKEN, 1]);
      }

      while (items.length > 1) {
        let bestRank: number | null = null;
        let bestA = '';
        let bestB = '';

        for (let i = 0; i < items.length - 1; i++) {
          const a = items[i][0];
          const b = items[i + 1][0];
          const rank = this.mergeRanks.get(`${a}\u0000${b}`);
          if (rank !== undefined && (bestRank === null || rank < bestRank)) {
            bestRank = rank;
            bestA = a;
            bestB = b;
          }
        }

        if (bestRank === null) break;

        const nextItems: [string, number][] = [];
        let i = 0;
        while (i < items.length) {
          if (
            i < items.length - 1 &&
            items[i][0] === bestA &&
            items[i + 1][0] === bestB
          ) {
            nextItems.push([bestA + bestB, items[i][1] + items[i + 1][1]]);
            i += 2;
          } else {
            nextItems.push(items[i]);
            i += 1;
          }
        }
        items = nextItems;
      }

      for (const [sym, charLen] of items) {
        result.push({
          id: this.tokenToId.get(sym) ?? this.unkId,
          text: sym,
          start: cursor,
          end: cursor + charLen,
        });
        cursor += charLen;
      }
    }

    return result;
  }

  encode(text: string): number[] {
    return this.encodeWithSpans(text).map((t) => t.id);
  }

  decode(ids: number[]): string {
    const parts: string[] = [];
    for (const id of ids) {
      const tok = this.vocab[id];
      if (tok === EOS_TOKEN) break;
      parts.push(tok);
    }
    return parts.join('');
  }

  tokenText(id: number): string {
    return this.vocab[id] ?? UNK_TOKEN;
  }
}
