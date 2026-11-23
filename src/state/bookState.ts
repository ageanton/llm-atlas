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

export function ablationToParam(ablation: AblationConfig): string {
  const parts: string[] = [];
  for (const [l, h] of ablation.disabledHeads ?? []) parts.push(`h${l}_${h}`);
  for (const l of ablation.disabledLayers ?? []) parts.push(`l${l}`);
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
  const chapterId = clampInt(Number(params.get('ch') ?? '4'), 1, 8);
  const beatIndex = clampInt(Number(params.get('beat') ?? '0'), 0, 5);
  const edition = parseEdition(params.get('ed'));
  const layer = clampInt(Number(params.get('block') ?? '2'), 0, 3);
  const head = clampInt(Number(params.get('head') ?? '0'), 0, 3);
  const queryIdxRaw = params.get('q');
  const queryIdx =
    queryIdxRaw !== null && queryIdxRaw !== ''
      ? clampInt(Number(queryIdxRaw), 0, 63)
      : null;
  const keyIdxRaw = params.get('k');
  const keyIdx =
    keyIdxRaw !== null && keyIdxRaw !== ''
      ? clampInt(Number(keyIdxRaw), 0, 63)
      : null;
  const showPreSoftmax = params.get('presoft') === '1';
  const lectureMode = params.get('mode') === 'lecture';
  const ablation = paramToAblation(params.get('abl'));

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
  if (state.prompt !== DEFAULT_PROMPT) params.set('in', state.prompt);
  params.set('ch', String(state.chapterId));
  params.set('beat', String(state.beatIndex));
  if (state.edition !== 'foundations') params.set('ed', state.edition);
  if (state.layer !== 2) params.set('block', String(state.layer));
  if (state.head !== 0) params.set('head', String(state.head));
  if (state.queryIdx !== null) params.set('q', String(state.queryIdx));
  if (state.keyIdx !== null) params.set('k', String(state.keyIdx));
  if (state.showPreSoftmax) params.set('presoft', '1');
  if (hasActiveAblation(state.ablation)) {
    params.set('abl', ablationToParam(state.ablation));
  }
  if (state.lectureMode) params.set('mode', 'lecture');
  return `?${params.toString()}`;
}
