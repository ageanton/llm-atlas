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
