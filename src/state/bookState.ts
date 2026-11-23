import type { AblationConfig } from '../engine/types';

export type Edition = 'foundations' | 'practitioner' | 'formal';

export interface BookState {
  prompt: string;
  chapterId: number;
  beatIndex: number;
  edition: Edition;
  layer: number;
  head: number;
  queryIdx: number | null;
  keyIdx: number | null;
  showPreSoftmax: boolean;
  ablation: AblationConfig;
  lectureMode: boolean;
}

export const DEFAULT_PROMPT =
  'Once upon a time, a little dog named Max found a red ball. He';

export function clampInt(n: number, min: number, max: number): number {
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, Math.round(n)));
}

export function hasActiveAblation(ablation: AblationConfig): boolean {
  return (
    (ablation.disabledHeads?.length ?? 0) > 0 ||
    (ablation.disabledAttnBlocks?.length ?? 0) > 0 ||
    (ablation.disabledMlpBlocks?.length ?? 0) > 0 ||
    (ablation.disabledResiduals?.length ?? 0) > 0 ||
    (ablation.disabledLayerNorms?.length ?? 0) > 0 ||
    (ablation.disabledLayers?.length ?? 0) > 0
  );
}

export function isHeadDisabled(
  ablation: AblationConfig,
  layer: number,
  head: number
): boolean {
  return (ablation.disabledHeads ?? []).some(
    ([l, h]) => l === layer && h === head
  );
}

export function toggleHeadDisabled(
  ablation: AblationConfig,
  layer: number,
  head: number
): AblationConfig {
  const cur = ablation.disabledHeads ?? [];
  const exists = cur.some(([l, h]) => l === layer && h === head);
  return {
    ...ablation,
    disabledHeads: exists
      ? cur.filter(([l, h]) => !(l === layer && h === head))
      : [...cur, [layer, head]],
  };
}

export function hasResidualAblation(ablation: AblationConfig): boolean {
  return (ablation.disabledResiduals?.length ?? 0) > 0;
}

export function toggleAllResiduals(ablation: AblationConfig): AblationConfig {
  if (hasResidualAblation(ablation)) {
    return { ...ablation, disabledResiduals: [] };
  }
  const all: ['attn' | 'mlp', number][] = [];
  for (let l = 0; l < 4; l++) {
    all.push(['attn', l], ['mlp', l]);
  }
  return { ...ablation, disabledResiduals: all };
}

export function isFinalLnDisabled(ablation: AblationConfig): boolean {
  return (ablation.disabledLayerNorms ?? []).some(([kind]) => kind === 'ln_f');
}

export function toggleFinalLn(ablation: AblationConfig): AblationConfig {
  const cur = ablation.disabledLayerNorms ?? [];
  const exists = cur.some(([kind]) => kind === 'ln_f');
  return {
    ...ablation,
    disabledLayerNorms: exists
      ? cur.filter(([kind]) => kind !== 'ln_f')
      : [...cur, ['ln_f', -1]],
  };
}

export function toggleAttnBlock(ablation: AblationConfig, layer: number): AblationConfig {
  const cur = ablation.disabledAttnBlocks ?? [];
  return {
    ...ablation,
    disabledAttnBlocks: cur.includes(layer)
      ? cur.filter((x) => x !== layer)
      : [...cur, layer],
  };
}

export function toggleMlpBlock(ablation: AblationConfig, layer: number): AblationConfig {
  const cur = ablation.disabledMlpBlocks ?? [];
  return {
    ...ablation,
    disabledMlpBlocks: cur.includes(layer)
      ? cur.filter((x) => x !== layer)
      : [...cur, layer],
  };
}

export function toggleLayerDisabled(ablation: AblationConfig, layer: number): AblationConfig {
  const cur = ablation.disabledLayers ?? [];
  return {
    ...ablation,
    disabledLayers: cur.includes(layer)
      ? cur.filter((x) => x !== layer)
      : [...cur, layer],
  };
}

export function ablationToParam(ablation: AblationConfig): string {
  const parts: string[] = [];
  for (const [l, h] of ablation.disabledHeads ?? []) parts.push(`h${l}_${h}`);
  for (const l of ablation.disabledLayers ?? []) parts.push(`l${l}`);
  for (const l of ablation.disabledAttnBlocks ?? []) parts.push(`a${l}`);
  for (const l of ablation.disabledMlpBlocks ?? []) parts.push(`m${l}`);
  for (const [kind, l] of ablation.disabledResiduals ?? []) {
    parts.push(`r${kind}_${l}`);
  }
  for (const [kind, l] of ablation.disabledLayerNorms ?? []) {
    parts.push(`n${kind}_${l}`);
  }
  return parts.length > 0 ? parts.join(',') : 'none';
}

export function paramToAblation(raw: string | null): AblationConfig {
  const cfg: AblationConfig = {};
  if (!raw || raw === 'none') return cfg;
  for (const token of raw.split(',')) {
    if (token.startsWith('h')) {
      const [l, h] = token.slice(1).split('_').map(Number);
      if (Number.isFinite(l) && Number.isFinite(h)) {
        cfg.disabledHeads = [
          ...(cfg.disabledHeads ?? []),
          [clampInt(l, 0, 3), clampInt(h, 0, 3)],
        ];
      }
    } else if (token.startsWith('l')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledLayers = [
          ...(cfg.disabledLayers ?? []),
          clampInt(l, 0, 3),
        ];
      }
    } else if (token.startsWith('a')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledAttnBlocks = [
          ...(cfg.disabledAttnBlocks ?? []),
          clampInt(l, 0, 3),
        ];
      }
    } else if (token.startsWith('m')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledMlpBlocks = [
          ...(cfg.disabledMlpBlocks ?? []),
          clampInt(l, 0, 3),
        ];
      }
    } else if (token.startsWith('r')) {
      const [kind, lStr] = token.slice(1).split('_');
      const l = Number(lStr);
      if ((kind === 'attn' || kind === 'mlp') && Number.isFinite(l)) {
        cfg.disabledResiduals = [
          ...(cfg.disabledResiduals ?? []),
          [kind, clampInt(l, 0, 3)],
        ];
      }
    } else if (token.startsWith('n')) {
      const idx = token.lastIndexOf('_');
      const kind = token.slice(1, idx);
      const l = Number(token.slice(idx + 1));
      if (
        (kind === 'ln_1' || kind === 'ln_2' || kind === 'ln_f') &&
        Number.isFinite(l)
      ) {
        cfg.disabledLayerNorms = [
          ...(cfg.disabledLayerNorms ?? []),
          [kind, kind === 'ln_f' ? -1 : clampInt(l, 0, 3)],
        ];
      }
    }
  }
  return cfg;
}

function parseEdition(raw: string | null): Edition {
  if (raw === 'practitioner' || raw === 'sufficient') return 'practitioner';
  if (raw === 'formal' || raw === 'complete') return 'formal';
  return 'foundations';
}

export function stateFromUrl(search: string): BookState {
  const params = new URLSearchParams(search);
  const prompt = params.get('in') || DEFAULT_PROMPT;
  const chapterId = clampInt(Number(params.get('ch') ?? '1'), 1, 8);
  const beatIndex = clampInt(Number(params.get('beat') ?? '0'), 0, 5);
  const edition = parseEdition(params.get('ed'));
  const layer = clampInt(Number(params.get('block') ?? '2'), 0, 3);
  const head = clampInt(Number(params.get('head') ?? '3'), 0, 3);
  const qRaw = params.get('q');
  const kRaw = params.get('k');
  const queryIdx =
    qRaw !== null && Number.isFinite(Number(qRaw)) ? Number(qRaw) : null;
  const keyIdx =
    kRaw !== null && Number.isFinite(Number(kRaw)) ? Number(kRaw) : null;
  const showPreSoftmax = params.get('raw') === '1';
  const ablation = paramToAblation(params.get('abl'));
  const lectureMode = params.get('mode') === 'lecture';

  return {
    prompt,
    chapterId,
    beatIndex,
    edition,
    layer,
    head,
    queryIdx,
    keyIdx,
    showPreSoftmax,
    ablation,
    lectureMode,
  };
}

export function stateToSearch(state: BookState): string {
  const params = new URLSearchParams();
  params.set('in', state.prompt);
  params.set('ch', String(state.chapterId));
  params.set('beat', String(state.beatIndex));
  params.set('ed', state.edition);
  params.set('block', String(state.layer));
  params.set('head', String(state.head));
  if (state.queryIdx !== null) params.set('q', String(state.queryIdx));
  if (state.keyIdx !== null) params.set('k', String(state.keyIdx));
  if (state.showPreSoftmax) params.set('raw', '1');
  if (hasActiveAblation(state.ablation)) {
    params.set('abl', ablationToParam(state.ablation));
  }
  if (state.lectureMode) params.set('mode', 'lecture');
  return `?${params.toString()}`;
}
