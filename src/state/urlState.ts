import type { AblationConfig } from '../engine/types';

export type Register = 'simple' | 'sufficient' | 'complete';
export type ZoomLevel = 0 | 1 | 2 | 3 | 4 | 5;
export type TeachingMode =
  | 'explore'
  | 'ablate'
  | 'replay'
  | 'sampling'
  | 'inspector'
  | 'scale';

export interface SelectionTarget {
  station: 'model' | 'tokenizer' | 'embed' | 'block' | 'ln_f' | 'head' | 'softmax';
  layer: number;
  sub: 'block' | 'ln_1' | 'attn' | 'ln_2' | 'mlp';
  head: number;
  matrix: string;
  row: number;
  col: number;
}

export interface AtlasState {
  prompt: string;
  level: ZoomLevel;
  sel: string;
  register: Register;
  ablation: AblationConfig;
  mode: TeachingMode;
  tourId: string | null;
  tourStep: number;
}

export const DEFAULT_PROMPT =
  'Once upon a time, a little dog named Max found a red ball. He';

export const DEFAULT_SELECTION = 'block2.attn.head3.K';

export function clampLevel(n: number): ZoomLevel {
  if (n <= 0) return 0;
  if (n >= 5) return 5;
  return Math.round(n) as ZoomLevel;
}

export function parseSelection(sel: string): SelectionTarget {
  const target: SelectionTarget = {
    station: 'block',
    layer: 0,
    sub: 'block',
    head: 0,
    matrix: 'blocks.0.attn.c_attn.weight',
    row: 0,
    col: 0,
  };

  if (!sel || sel === 'model') {
    target.station = 'model';
    return target;
  }
  if (sel === 'tokenizer') {
    target.station = 'tokenizer';
    return target;
  }
  if (sel === 'embed') {
    target.station = 'embed';
    target.matrix = 'wte.weight';
    return target;
  }
  if (sel === 'ln_f') {
    target.station = 'ln_f';
    target.matrix = 'ln_f.weight';
    return target;
  }
  if (sel === 'softmax') {
    target.station = 'softmax';
    target.matrix = 'wte.weight';
    return target;
  }

  const parts = sel.split('.');
  const blockMatch = parts[0]?.match(/^block([0-3])$/);
  if (blockMatch) {
    target.station = 'block';
    target.layer = Number.parseInt(blockMatch[1], 10);
    target.matrix = `blocks.${target.layer}.attn.c_attn.weight`;

    if (parts.length === 1) {
      target.sub = 'block';
      return target;
    }

    const sub = parts[1];
    if (sub === 'ln_1') {
      target.sub = 'ln_1';
      target.matrix = `blocks.${target.layer}.ln_1.weight`;
      return target;
    }
    if (sub === 'ln_2') {
      target.sub = 'ln_2';
      target.matrix = `blocks.${target.layer}.ln_2.weight`;
      return target;
    }
    if (sub === 'mlp') {
      target.sub = 'mlp';
      target.matrix = `blocks.${target.layer}.mlp.c_fc.weight`;
      return target;
    }

    if (sub === 'attn') {
      target.sub = 'attn';
      if (parts[2]?.startsWith('head')) {
        target.station = 'head';
        target.head = Math.max(0, Math.min(3, Number.parseInt(parts[2].replace('head', ''), 10) || 0));
        const mat = parts[3];
        if (mat === 'Q' || mat === 'K' || mat === 'V') {
          target.matrix = `blocks.${target.layer}.attn.c_attn.weight`;
        } else if (mat === 'O' || mat === 'proj') {
          target.matrix = `blocks.${target.layer}.attn.c_proj.weight`;
        }
      }
      return target;
    }
  }

  if (sel.startsWith('param:')) {
    const raw = sel.replace('param:', '');
    const [name, rc] = raw.split('@');
    target.matrix = name || target.matrix;
    if (rc) {
      const [r, c] = rc.split(',').map((x) => Number.parseInt(x, 10));
      target.row = Number.isFinite(r) ? r : 0;
      target.col = Number.isFinite(c) ? c : 0;
    }
    const m = target.matrix.match(/blocks\.([0-3])\.(attn|mlp|ln_[12])/);
    if (m) {
      target.station = 'block';
      target.layer = Number.parseInt(m[1], 10);
      if (m[2] === 'attn') target.sub = 'attn';
      else if (m[2] === 'mlp') target.sub = 'mlp';
      else if (m[2] === 'ln_1') target.sub = 'ln_1';
      else if (m[2] === 'ln_2') target.sub = 'ln_2';
    }
    return target;
  }

  return target;
}

export function buildParamSelection(
  paramName: string,
  row?: number,
  col?: number
): string {
  if (typeof row === 'number' && typeof col === 'number') {
    return `param:${paramName}@${row},${col}`;
  }
  return `param:${paramName}`;
}

export function targetToSelectionString(target: SelectionTarget): string {
  if (target.station === 'model') return 'model';
  if (target.station === 'tokenizer') return 'tokenizer';
  if (target.station === 'embed') return 'embed';
  if (target.station === 'ln_f') return 'ln_f';
  if (target.station === 'softmax') return 'softmax';

  if (target.station === 'head') {
    return `block${target.layer}.attn.head${target.head}.K`;
  }

  if (target.sub && target.sub !== 'block') {
    return `block${target.layer}.${target.sub}`;
  }

  return `block${target.layer}`;
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

export function ablationToParam(ablation: AblationConfig): string {
  const parts: string[] = [];
  for (const [l, h] of ablation.disabledHeads ?? []) parts.push(`h${l}_${h}`);
  for (const l of ablation.disabledAttnBlocks ?? []) parts.push(`a${l}`);
  for (const l of ablation.disabledMlpBlocks ?? []) parts.push(`m${l}`);
  for (const [kind, l] of ablation.disabledResiduals ?? []) parts.push(`r${kind}_${l}`);
  for (const [kind, l] of ablation.disabledLayerNorms ?? []) parts.push(`n${kind}_${l}`);
  for (const l of ablation.disabledLayers ?? []) parts.push(`l${l}`);
  return parts.length > 0 ? parts.join(';') : '';
}

export function paramToAblation(raw: string | null): AblationConfig {
  const cfg: AblationConfig = {};
  if (!raw || raw === 'none') return cfg;
  for (const token of raw.split(';')) {
    if (!token) continue;
    if (token.startsWith('h')) {
      const [l, h] = token.slice(1).split('_').map(Number);
      if (Number.isFinite(l) && Number.isFinite(h)) {
        cfg.disabledHeads = [...(cfg.disabledHeads ?? []), [l, h]];
      }
    } else if (token.startsWith('a')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledAttnBlocks = [...(cfg.disabledAttnBlocks ?? []), l];
      }
    } else if (token.startsWith('m')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledMlpBlocks = [...(cfg.disabledMlpBlocks ?? []), l];
      }
    } else if (token.startsWith('r')) {
      const [kind, lStr] = token.slice(1).split('_');
      const l = Number(lStr);
      if ((kind === 'attn' || kind === 'mlp') && Number.isFinite(l)) {
        cfg.disabledResiduals = [...(cfg.disabledResiduals ?? []), [kind, l]];
      }
    } else if (token.startsWith('n')) {
      const idx = token.lastIndexOf('_');
      const kind = token.slice(1, idx);
      const l = Number(token.slice(idx + 1));
      if (['ln_1', 'ln_2', 'ln_f'].includes(kind) && Number.isFinite(l)) {
        cfg.disabledLayerNorms = [
          ...(cfg.disabledLayerNorms ?? []),
          [kind as 'ln_1' | 'ln_2' | 'ln_f', l],
        ];
      }
    } else if (token.startsWith('l')) {
      const l = Number(token.slice(1));
      if (Number.isFinite(l)) {
        cfg.disabledLayers = [...(cfg.disabledLayers ?? []), l];
      }
    }
  }
  return cfg;
}

function registerToShort(r: Register): string {
  if (r === 'simple') return 's';
  if (r === 'sufficient') return 'm';
  return 'c';
}

function shortToRegister(v: string | null): Register {
  if (v === 's' || v === 'simple') return 'simple';
  if (v === 'c' || v === 'complete') return 'complete';
  return 'sufficient';
}

export function stateToSearch(state: AtlasState): string {
  const params = new URLSearchParams();
  if (state.prompt !== DEFAULT_PROMPT) params.set('p', state.prompt);
  if (state.level !== 0) params.set('z', String(state.level));
  if (state.sel && state.sel !== 'model') params.set('sel', state.sel);
  if (state.register !== 'sufficient') params.set('r', registerToShort(state.register));

  const abParam = ablationToParam(state.ablation);
  if (abParam) params.set('ab', abParam);

  if (state.mode !== 'explore') params.set('mode', state.mode);
  if (state.tourId) {
    params.set('tour', state.tourId);
    if (state.tourStep > 0) params.set('step', String(state.tourStep));
  }

  const query = params.toString();
  return query ? `?${query}` : '';
}

export function stateFromUrl(search: string): AtlasState {
  const params = new URLSearchParams(search);

  const prompt = params.get('p') || DEFAULT_PROMPT;
  const levelRaw = Number.parseInt(params.get('z') || '0', 10);
  const level = clampLevel(Number.isFinite(levelRaw) ? levelRaw : 0);
  const sel = params.get('sel') || DEFAULT_SELECTION;
  const register = shortToRegister(params.get('r'));
  const ablation = paramToAblation(params.get('ab'));
  const modeRaw = params.get('mode') as TeachingMode | null;
  const mode: TeachingMode =
    modeRaw && ['explore', 'ablate', 'replay', 'sampling', 'inspector', 'scale'].includes(modeRaw)
      ? modeRaw
      : 'explore';
  const tourId = params.get('tour') || null;
  const tourStep = Math.max(0, Number.parseInt(params.get('step') || '0', 10) || 0);

  return {
    prompt,
    level,
    sel,
    register,
    ablation,
    mode,
    tourId,
    tourStep,
  };
}
