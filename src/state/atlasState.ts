import type { AblationConfig } from '../engine/types';

export type Register = 'simple' | 'sufficient' | 'complete';

export interface AtlasState {
  prompt: string;
  nodeId: string;
  layer: number;
  head: number;
  register: Register;
  ablation: AblationConfig;
  showPreSoftmax: boolean;
  paramName: string;
  paramRow: number;
  paramCol: number;
}

export const DEFAULT_PROMPT =
  'Once upon a time, a little dog named Max found a red ball. He';

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

export function ablationToParam(ablation: AblationConfig): string {
  const parts: string[] = [];
  for (const [l, h] of ablation.disabledHeads ?? []) parts.push(`h${l}_${h}`);
  for (const l of ablation.disabledAttnBlocks ?? []) parts.push(`a${l}`);
  for (const l of ablation.disabledMlpBlocks ?? []) parts.push(`m${l}`);
  for (const [kind, l] of ablation.disabledResiduals ?? [])
    parts.push(`r${kind}_${l}`);
  for (const [kind, l] of ablation.disabledLayerNorms ?? [])
    parts.push(`n${kind}_${l}`);
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
          [clampIndex(l), clampIndex(h)],
        ];
      }
    } else if (token.startsWith('a')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledAttnBlocks = [
          ...(cfg.disabledAttnBlocks ?? []),
          clampIndex(l),
        ];
      }
    } else if (token.startsWith('m')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledMlpBlocks = [
          ...(cfg.disabledMlpBlocks ?? []),
          clampIndex(l),
        ];
      }
    } else if (token.startsWith('r')) {
      const [kind, lStr] = token.slice(1).split('_');
      const l = Number(lStr);
      if ((kind === 'attn' || kind === 'mlp') && Number.isFinite(l)) {
        cfg.disabledResiduals = [
          ...(cfg.disabledResiduals ?? []),
          [kind, clampIndex(l)],
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
          [kind, kind === 'ln_f' ? -1 : clampIndex(l)],
        ];
      }
    } else if (token.startsWith('l')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledLayers = [...(cfg.disabledLayers ?? []), clampIndex(l)];
      }
    }
  }
  return cfg;
}

function parseRegister(raw: string | null): Register {
  if (raw === 'simple' || raw === 'simp' || raw === 'intuitive') return 'simple';
  if (raw === 'complete' || raw === 'comp' || raw === 'math') return 'complete';
  return 'sufficient';
}

export function stateFromUrl(search: string): AtlasState {
  const params = new URLSearchParams(search);
  const prompt = params.get('in') || DEFAULT_PROMPT;
  const nodeId = params.get('node') || 'root';
  const layer = clampIndex(Number(params.get('block') ?? '2'));
  const head = clampIndex(Number(params.get('head') ?? '3'));
  const register = parseRegister(params.get('reg'));
  const ablation = paramToAblation(params.get('abl'));
  const showPreSoftmax = params.get('raw') === '1';
  const paramName = params.get('param') || 'blocks.2.attn.W_k.weight';
  const paramRow = Math.max(0, Number(params.get('r') ?? '64') || 0);
  const paramCol = Math.max(0, Number(params.get('c') ?? '19') || 0);

  return {
    prompt,
    nodeId,
    layer,
    head,
    register,
    ablation,
    showPreSoftmax,
    paramName,
    paramRow,
    paramCol,
  };
}

export function stateToSearch(state: AtlasState): string {
  const params = new URLSearchParams();
  params.set('in', state.prompt);
  params.set('node', state.nodeId);
  params.set('block', String(state.layer));
  params.set('head', String(state.head));
  params.set('reg', state.register);
  params.set('abl', ablationToParam(state.ablation));
  if (state.showPreSoftmax) params.set('raw', '1');
  if (state.nodeId === 'scalar') {
    params.set('param', state.paramName);
    params.set('r', String(state.paramRow));
    params.set('c', String(state.paramCol));
  }
  return `?${params.toString()}`;
}
