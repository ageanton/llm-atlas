import type {
  EncodedToken,
  ModelManifest,
  TokenizerData,
  Trace,
} from '../engine/types';
import type { Register, SelectionTarget } from '../state/urlState';
import { formatToken, vectorL2Norm } from '../utils/color';

interface Level0Props {
  prompt: string;
  tokens: EncodedToken[];
  trace: Trace;
  tokenizer: TokenizerData;
  manifest: ModelManifest;
  register: Register;
  onZoomIn: () => void;
}

export function Level0View({
  prompt,
  tokens,
  trace,
  tokenizer,
  manifest,
  register,
  onZoomIn,
}: Level0Props) {
  const topProbs = trace.finalProbs
    .map((prob, id) => ({ id, prob, text: tokenizer.vocab[id] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);
  const top = topProbs[0];

  return (
    <div className="scene-stage level0-stage">
      <div className="l0-flow">
        <div className="l0-io-card">
          <div className="l0-caption">Input text</div>
          <div className="l0-sentence">{prompt}</div>
          <div className="l0-meta mono">
            {tokens.length} tokens of {manifest.config.max_seq_len} max
          </div>
        </div>

        <div className="l0-connector" aria-hidden="true" />

        <button
          type="button"
          className="l0-model-box"
          onClick={onZoomIn}
          aria-label="Miniature transformer model box. Activate to zoom into Level 1 pipeline."
        >
          <span className="l0-box-title">Miniature Transformer</span>
          <span className="l0-box-params mono">
            {manifest.total_params.toLocaleString()} parameters
          </span>
          {register !== 'simple' && (
            <span className="l0-box-spec mono">
              {manifest.config.n_layers} layers · {manifest.config.n_heads} heads · dim {manifest.config.d_model}
            </span>
          )}
          <span className="l0-box-action">Click to inspect internal highway →</span>
        </button>

        <div className="l0-connector" aria-hidden="true" />

        <div className="l0-io-card">
          <div className="l0-caption">Next token output</div>
          <div className="l0-winner mono">
            “{formatToken(top.text)}”
            <span className="l0-prob">{(top.prob * 100).toFixed(1)}%</span>
          </div>
          <div className="l0-runnerups mono">
            {topProbs.slice(1, 4).map((p) => (
              <span key={p.id}>
                “{formatToken(p.text)}” {(p.prob * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

interface Level1Props {
  tokens: EncodedToken[];
  trace: Trace;
  manifest: ModelManifest;
  tokenizer: TokenizerData;
  target: SelectionTarget;
  register: Register;
  onSelect: (sel: string, zoomIn?: boolean) => void;
}

export function Level1View({
  tokens,
  trace,
  manifest,
  tokenizer,
  target,
  register,
  onSelect,
}: Level1Props) {
  return (
    <div className="scene-stage level1-stage">
      <div className="highway-strip" aria-label="Residual stream highway across 4 blocks">
        <button
          type="button"
          className={`station-pill ${target.station === 'tokenizer' ? 'is-selected' : ''}`}
          onClick={() => onSelect('tokenizer')}
        >
          <span className="station-role mono">Input</span>
          <span className="station-title">Tokenizer</span>
          <span className="station-sub mono">{tokens.length} tokens</span>
        </button>

        <div className="highway-link" aria-hidden="true" />

        <button
          type="button"
          className={`station-pill ${target.station === 'embed' ? 'is-selected' : ''}`}
          onClick={() => onSelect('embed')}
        >
          <span className="station-role mono">Vectorize</span>
          <span className="station-title">Embed (WTE+WPE)</span>
          <span className="station-sub mono">dim {manifest.config.d_model}</span>
        </button>

        <div className="highway-link" aria-hidden="true" />

        {trace.layers.map((_, idx) => {
          const lens = trace.logitLens[idx];
          const topLens = lens?.topTokens[0];
          const lensWord = topLens ? tokenizer.vocab[topLens.id] : '—';
          const lensProb = topLens ? Math.round(topLens.prob * 100) : 0;
          const isSelected = target.station === 'block' && target.layer === idx;

          return (
            <div key={idx} className="block-station-wrap">
              <button
                type="button"
                className={`station-pill block-station ${isSelected ? 'is-selected' : ''}`}
                onClick={() => onSelect(`block${idx}`, true)}
              >
                <span className="station-role mono">Layer {idx}</span>
                <span className="station-title">Block {idx}</span>
                <span className="station-sub mono">Attn + MLP</span>
              </button>

              <div
                className="lens-tag mono"
                title={`Logit Lens guess after block ${idx}: "${lensWord}" (${lensProb}%)`}
              >
                Lens: “{formatToken(lensWord)}” {lensProb}%
              </div>
            </div>
          );
        })}

        <div className="highway-link" aria-hidden="true" />

        <button
          type="button"
          className={`station-pill ${target.station === 'ln_f' ? 'is-selected' : ''}`}
          onClick={() => onSelect('ln_f')}
        >
          <span className="station-role mono">Norm</span>
          <span className="station-title">Final LN</span>
          <span className="station-sub mono">zero mean, unit var</span>
        </button>

        <div className="highway-link" aria-hidden="true" />

        <button
          type="button"
          className={`station-pill ${target.station === 'softmax' ? 'is-selected' : ''}`}
          onClick={() => onSelect('softmax')}
        >
          <span className="station-role mono">Unembed</span>
          <span className="station-title">Logits & Softmax</span>
          <span className="station-sub mono">vocab {manifest.config.vocab_size}</span>
        </button>
      </div>

      <div className="highway-vector-indicator mono">
        <span className="indicator-label">Residual Stream vector norm:</span>
        <span className="indicator-value">
          ||x_0|| = {vectorL2Norm(trace.tokEmb[tokens.length - 1] ?? []).toFixed(2)} →{' '}
          ||x_final|| = {vectorL2Norm(trace.lnFOut[tokens.length - 1] ?? []).toFixed(2)}
        </span>
        {register !== 'simple' && (
          <span className="indicator-help">
            Skip connections preserve representation stability from token 0 to token {tokens.length - 1}.
          </span>
        )}
      </div>
    </div>
  );
}

interface Level2Props {
  tokens: EncodedToken[];
  trace: Trace;
  manifest: ModelManifest;
  target: SelectionTarget;
  register: Register;
  onSelect: (sel: string, zoomIn?: boolean) => void;
}

export function Level2View({
  tokens,
  trace,
  manifest,
  register: _register,
  target,
  onSelect,
}: Level2Props) {
  const layerIdx = target.layer;
  const layerTrace = trace.layers[layerIdx];
  const lastTok = tokens.length - 1;

  if (layerTrace.skipped) {
    return (
      <div className="scene-stage">
        <div className="empty-notice">
          Block {layerIdx} is disabled in Ablation Mode. Toggle it back on in the Ablation panel to inspect its components.
        </div>
      </div>
    );
  }

  const normIn = vectorL2Norm(layerTrace.residualBeforeAttn[lastTok] ?? []);
  const normAttn = vectorL2Norm(layerTrace.attn.output[lastTok] ?? []);
  const normMid = vectorL2Norm(layerTrace.residualAfterAttn[lastTok] ?? []);
  const normMlp = vectorL2Norm(layerTrace.mlp.output[lastTok] ?? []);
  const normOut = vectorL2Norm(layerTrace.residualAfterMlp[lastTok] ?? []);

  return (
    <div className="scene-stage level2-stage">
      <div className="l2-header">
        <h2>Block {layerIdx} Architecture</h2>
        <span className="mono l2-meta">
          Pre-LayerNorm · {manifest.config.n_heads} attention heads of dim {manifest.config.d_head} · MLP expansion to {manifest.config.d_mlp}
        </span>
      </div>

      <div className="l2-block-diagram">
        <div className="l2-stream-marker mono">
          Residual stream in: x_{layerIdx} (||x|| = {normIn.toFixed(2)})
        </div>

        <div className="l2-sublayer attn-sublayer">
          <div className="l2-main-path">
            <button
              type="button"
              className={`l2-station-btn ${target.sub === 'ln_1' ? 'is-selected' : ''}`}
              onClick={() => onSelect(`block${layerIdx}.ln_1`)}
            >
              <span className="btn-role mono">Norm</span>
              <span className="btn-title">LN 1</span>
            </button>

            <span className="l2-arrow" aria-hidden="true">→</span>

            <button
              type="button"
              className={`l2-station-btn broad ${target.sub === 'attn' ? 'is-selected' : ''}`}
              onClick={() => onSelect(`block${layerIdx}.attn.head0.K`, true)}
            >
              <span className="btn-role mono">Communication (Token mixing across time)</span>
              <span className="btn-title">Multi-Head Attention (4 heads)</span>
              <span className="btn-sub mono">Output norm = {normAttn.toFixed(2)}</span>
            </button>
          </div>

          <div className="l2-skip-path mono" aria-label="Residual skip addition">
            + skip addition (x_{layerIdx} + attn_out)
          </div>
        </div>

        <div className="l2-stream-marker mono">
          Intermediate stream: x_{layerIdx},mid (||x|| = {normMid.toFixed(2)})
        </div>

        <div className="l2-sublayer mlp-sublayer">
          <div className="l2-main-path">
            <button
              type="button"
              className={`l2-station-btn ${target.sub === 'ln_2' ? 'is-selected' : ''}`}
              onClick={() => onSelect(`block${layerIdx}.ln_2`)}
            >
              <span className="btn-role mono">Norm</span>
              <span className="btn-title">LN 2</span>
            </button>

            <span className="l2-arrow" aria-hidden="true">→</span>

            <button
              type="button"
              className={`l2-station-btn broad ${target.sub === 'mlp' ? 'is-selected' : ''}`}
              onClick={() => onSelect(`block${layerIdx}.mlp`)}
            >
              <span className="btn-role mono">Computation (Per-token nonlinear feature map)</span>
              <span className="btn-title">MLP (128 → 512 → 128)</span>
              <span className="btn-sub mono">Output norm = {normMlp.toFixed(2)}</span>
            </button>
          </div>

          <div className="l2-skip-path mono" aria-label="Residual skip addition">
            + skip addition (x_{layerIdx},mid + mlp_out)
          </div>
        </div>

        <div className="l2-stream-marker mono">
          Residual stream out: x_{layerIdx + 1} (||x|| = {normOut.toFixed(2)})
        </div>
      </div>

      <div className="l2-head-pills">
        <span className="mono pills-label">Direct head drill-down:</span>
        {[0, 1, 2, 3].map((h) => (
          <button
            key={h}
            type="button"
            className="seg-btn mono"
            onClick={() => onSelect(`block${layerIdx}.attn.head${h}.K`, true)}
          >
            Inspect Head {h} →
          </button>
        ))}
      </div>
    </div>
  );
}
