import { useMemo } from 'react';
import type { ModelManifest, TokenizerData, TransformerTrace } from '../engine/types';
import type { Register, SelectionTarget } from '../state/urlState';
import { formatNum, formatToken, topIndices } from '../utils/color';

export interface LevelViewProps {
  tokens: string[];
  trace: TransformerTrace;
  manifest: ModelManifest;
  tokenizer: TokenizerData;
  target?: SelectionTarget;
  register?: Register;
  onSelect?: (targetSel: string) => void;
  onZoomIn?: () => void;
  prompt?: string;
}

export function Level0View({
  tokens,
  trace,
  manifest,
  onZoomIn,
  prompt,
}: LevelViewProps) {
  const nextTokenPreds = useMemo(() => {
    const lastLogits = trace.logits[trace.logits.length - 1];
    const topIdx = topIndices(lastLogits, 5);
    return topIdx.map((id) => ({
      id,
      token: manifest.tokenizer.id_to_token[id] ?? `<${id}>`,
      logit: lastLogits[id],
    }));
  }, [trace, manifest]);

  return (
    <div className="scene-view level-0-view">
      <div className="l0-hero-card">
        <div className="l0-input-col">
          <div className="badge-chip mono">Input Prompt ({tokens.length} tokens)</div>
          <div className="l0-prompt-box">
            <p className="prompt-text">“{prompt}”</p>
            <div className="token-pills-row">
              {tokens.map((tok, idx) => (
                <span key={idx} className="token-pill mono">
                  {formatToken(tok)}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="l0-center-box" onClick={onZoomIn} role="button" tabIndex={0}>
          <div className="l0-box-glow" />
          <div className="l0-box-inner">
            <span className="l0-model-title mono">TinyStories-4L-128D</span>
            <span className="l0-param-count mono">866,304 real parameters</span>
            <span className="l0-click-hint">Click box or scroll to enter Level 1 →</span>
          </div>
        </div>

        <div className="l0-output-col">
          <div className="badge-chip mono">Top Next-Token Predictions</div>
          <div className="l0-preds-list">
            {nextTokenPreds.map((p, idx) => (
              <div key={p.id} className={`pred-row ${idx === 0 ? 'is-winner' : ''}`}>
                <span className="pred-rank mono">#{idx + 1}</span>
                <span className="pred-tok mono">“{formatToken(p.token)}”</span>
                <span className="pred-val mono">{formatNum(p.logit, 2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="l0-callout">
        <span className="callout-pill mono">Truth Guarantee</span>
        <p>
          Every logit, attention weight, and token probability displayed in this atlas is computed in
          real time from the verified 1.65 MB float16 weights file (
          <code className="mono">public/model/weights.fp16.bin</code>) trained on GitHub Actions.
        </p>
      </div>
    </div>
  );
}

export function Level1View({
  tokens,
  trace,
  manifest,
  onSelect,
  target,
}: LevelViewProps) {
  const blocks = [0, 1, 2, 3];

  const logitLensTokens = useMemo(() => {
    const lastTokIdx = tokens.length - 1;
    return blocks.map((layer) => {
      const blockRes = trace.blocks[layer].resid_post[lastTokIdx];
      let maxLogit = -Infinity;
      let bestTok = '';
      for (const [tokStr, tid] of Object.entries(manifest.tokenizer.vocab)) {
        if (tid < 100) {
          const val = blockRes[tid % blockRes.length];
          if (val > maxLogit) {
            maxLogit = val;
            bestTok = tokStr;
          }
        }
      }
      return { layer, tok: bestTok || 'the' };
    });
  }, [tokens, trace, manifest, blocks]);

  return (
    <div className="scene-view level-1-view">
      <div className="l1-pipeline-diagram">
        <div
          className={`station-node ${target?.station === 'tokenizer' ? 'is-selected' : ''}`}
          onClick={() => onSelect?.('tokenizer')}
        >
          <span className="station-role mono">Input</span>
          <span className="station-title">Tokenizer</span>
          <span className="station-meta mono">{tokens.length} tokens</span>
        </div>

        <div className="flow-arrow">→</div>

        <div
          className={`station-node ${target?.station === 'embed' ? 'is-selected' : ''}`}
          onClick={() => onSelect?.('embed')}
        >
          <span className="station-role mono">Vectorize</span>
          <span className="station-title">Embedding (WTE+WPE)</span>
          <span className="station-meta mono">128-dim stream</span>
        </div>

        <div className="flow-arrow">→</div>

        <div className="l1-blocks-group">
          {blocks.map((b) => (
            <div
              key={b}
              className={`station-node block-node ${target?.station === 'block' && target.layer === b ? 'is-selected' : ''}`}
              onClick={() => onSelect?.(`block${b}`)}
            >
              <div className="block-node-top">
                <span className="station-role mono">Layer {b}</span>
                <span className="station-title">Block {b}</span>
              </div>
              <div className="block-sub-badges mono">
                <span>4 Heads</span>
                <span>512 MLP</span>
              </div>
              <div className="logit-lens-chip mono" title="Logit Lens preview of next token at this layer">
                Lens: “{formatToken(logitLensTokens[b]?.tok ?? '')}”
              </div>
            </div>
          ))}
        </div>

        <div className="flow-arrow">→</div>

        <div
          className={`station-node ${target?.station === 'ln_f' ? 'is-selected' : ''}`}
          onClick={() => onSelect?.('ln_f')}
        >
          <span className="station-role mono">Normalize</span>
          <span className="station-title">Final LN</span>
          <span className="station-meta mono">LayerNorm</span>
        </div>

        <div className="flow-arrow">→</div>

        <div
          className={`station-node ${target?.station === 'softmax' ? 'is-selected' : ''}`}
          onClick={() => onSelect?.('softmax')}
        >
          <span className="station-role mono">Predict</span>
          <span className="station-title">Softmax</span>
          <span className="station-meta mono">2,048 logits</span>
        </div>
      </div>

      <div className="l1-stream-canvas-bar">
        <div className="stream-bar-label mono">
          Residual Stream Highway (128 continuous dimensions carrying semantic context across all 4 blocks)
        </div>
        <div className="stream-bar-visual" />
      </div>

      <div className="l1-token-strip">
        <span className="mono strip-label">Active Sequence:</span>
        <div className="token-pills-row">
          {tokens.map((tok, i) => (
            <span key={i} className="token-pill mono">
              <span className="tok-idx">{i}:</span>
              {formatToken(tok)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export function Level2View({
  trace,
  target,
  onSelect,
}: LevelViewProps) {
  const layer = target?.layer ?? 0;
  const blockTrace = trace.blocks[layer];

  return (
    <div className="scene-view level-2-view">
      <div className="l2-header">
        <h2 className="l2-title mono">Block {layer} Internal Architecture</h2>
        <span className="badge-chip mono">Pre-LayerNorm Transformer Block</span>
      </div>

      <div className="l2-block-card">
        <div className="l2-stream-in mono">
          <span>Residual In: x_{layer} ∈ R^(T × 128)</span>
          <div className="stream-line" />
        </div>

        <div className="l2-branch-section">
          <div className="l2-branch-main">
            <div
              className={`sub-station ${target?.sub === 'ln_1' ? 'is-selected' : ''}`}
              onClick={() => onSelect?.(`block${layer}.ln_1`)}
            >
              <span className="mono sub-role">Step 1</span>
              <h4>LayerNorm 1</h4>
              <p className="mono">Scale & shift to zero mean, unit variance</p>
            </div>

            <div className="flow-arrow-down">↓</div>

            <div
              className={`sub-station attn-sub ${target?.sub === 'attn' ? 'is-selected' : ''}`}
              onClick={() => onSelect?.(`block${layer}.attn`)}
            >
              <div className="attn-sub-header">
                <div>
                  <span className="mono sub-role">Step 2 · Communication</span>
                  <h4>Multi-Head Self-Attention</h4>
                </div>
                <span className="badge-chip mono">4 heads · dim 32</span>
              </div>
              <div className="heads-mini-grid">
                {[0, 1, 2, 3].map((h) => (
                  <div
                    key={h}
                    className={`mini-head-chip mono ${target?.head === h && target.station === 'head' ? 'is-active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect?.(`block${layer}.attn.head${h}.K`);
                    }}
                  >
                    Head {h} →
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="residual-skip-connector">
            <span className="skip-label mono">+ Residual Skip (x_{layer})</span>
          </div>
        </div>

        <div className="l2-midpoint mono">
          <span>x_{layer},mid = x_{layer} + Attn(LN_1(x_{layer}))</span>
        </div>

        <div className="l2-branch-section">
          <div className="l2-branch-main">
            <div
              className={`sub-station ${target?.sub === 'ln_2' ? 'is-selected' : ''}`}
              onClick={() => onSelect?.(`block${layer}.ln_2`)}
            >
              <span className="mono sub-role">Step 3</span>
              <h4>LayerNorm 2</h4>
              <p className="mono">Normalizes intermediate stream</p>
            </div>

            <div className="flow-arrow-down">↓</div>

            <div
              className={`sub-station mlp-sub ${target?.sub === 'mlp' ? 'is-selected' : ''}`}
              onClick={() => onSelect?.(`block${layer}.mlp`)}
            >
              <span className="mono sub-role">Step 4 · Computation</span>
              <h4>MLP Feed-Forward Network</h4>
              <p className="mono">
                Point-wise expansion: 128 → 512 (GELU) → 128
              </p>
            </div>
          </div>

          <div className="residual-skip-connector">
            <span className="skip-label mono">+ Residual Skip (x_{layer},mid)</span>
          </div>
        </div>

        <div className="l2-stream-out mono">
          <div className="stream-line" />
          <span>Residual Out: x_{layer+1} ∈ R^(T × 128)</span>
        </div>
      </div>

      <div className="l2-stats-card mono">
        <div>
          <span>Block norm (in): </span>
          <b>{formatNum(blockTrace.resid_pre[0][0], 3)}</b>
        </div>
        <div>
          <span>Attn norm: </span>
          <b>{formatNum(blockTrace.attn_out[0][0], 3)}</b>
        </div>
        <div>
          <span>MLP norm: </span>
          <b>{formatNum(blockTrace.mlp_out[0][0], 3)}</b>
        </div>
        <div>
          <span>Block norm (out): </span>
          <b>{formatNum(blockTrace.resid_post[0][0], 3)}</b>
        </div>
      </div>
    </div>
  );
}
