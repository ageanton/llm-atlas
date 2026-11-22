import type { AblationConfig } from '../engine/types';

export type Register = 'kid' | 'curious' | 'deep';
export type ZoomLevel = 0 | 1 | 2 | 3;

export interface AtlasState {
  prompt: string;
  level: ZoomLevel;
  layer: number;
  head: number;
  register: Register;
  ablation: AblationConfig;
  showNumbers: boolean;
}

export const DEFAULT_PROMPT =
  'Once upon a time, a little dog named Max found a red ball. He';

export function clampLevel(n: number): ZoomLevel {
  if (n <= 0) return 0;
  if (n >= 3) return 3;
  return Math.round(n) as ZoomLevel;
}

export function clampIndex(n: number, max = 3): number {
  if (!Number.isFinite(n) || n < 0) return 0;
  if (n > max) return max;
  return Math.round(n);
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

export function toggleHeadAblation(
  ablation: AblationConfig,
  layer: number,
  head: number
): AblationConfig {
  const cur = ablation.disabledHeads ?? [];
  const exists = cur.some(([l, h]) => l === layer && h === head);
  const next = exists
    ? cur.filter(([l, h]) => !(l === layer && h === head))
    : [...cur, [layer, head] as [number, number]];
  return { ...ablation, disabledHeads: next };
}

export function ablationToParam(ablation: AblationConfig): string {
  const parts: string[] = [];
  for (const [l, h] of ablation.disabledHeads ?? []) parts.push(`h${l}_${h}`);
  for (const l of ablation.disabledLayers ?? []) parts.push(`l${l}`);
  for (const l of ablation.disabledMlpBlocks ?? []) parts.push(`m${l}`);
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
          [clampIndex(l), clampIndex(h)],
        ];
      }
    } else if (token.startsWith('l')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledLayers = [...(cfg.disabledLayers ?? []), clampIndex(l)];
      }
    } else if (token.startsWith('m')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledMlpBlocks = [
          ...(cfg.disabledMlpBlocks ?? []),
          clampIndex(l),
        ];
      }
    }
  }
  return cfg;
}

function parseRegister(raw: string | null): Register {
  if (raw === 'kid' || raw === 'simp' || raw === 'simple') return 'kid';
  if (raw === 'deep' || raw === 'comp' || raw === 'complete') return 'deep';
  return 'curious';
}

export function stateFromUrl(search: string): AtlasState {
  const params = new URLSearchParams(search);
  const prompt = params.get('in') || DEFAULT_PROMPT;
  const lvRaw = params.get('lv');
  const level =
    lvRaw !== null && Number.isFinite(Number(lvRaw))
      ? clampLevel(Number(lvRaw))
      : 0;
  const layer = clampIndex(Number(params.get('block') ?? '2'));
  const head = clampIndex(Number(params.get('head') ?? '3'));
  const register = parseRegister(params.get('reg'));
  const ablation = paramToAblation(params.get('abl'));
  const showNumbers = params.get('nums') === '1';

  return { prompt, level, layer, head, register, ablation, showNumbers };
}

export function stateToSearch(state: AtlasState): string {
  const params = new URLSearchParams();
  params.set('in', state.prompt);
  params.set('lv', String(state.level));
  params.set('block', String(state.layer));
  params.set('head', String(state.head));
  params.set('reg', state.register);
  params.set('abl', ablationToParam(state.ablation));
  if (state.showNumbers) params.set('nums', '1');
  return `?${params.toString()}`;
}
