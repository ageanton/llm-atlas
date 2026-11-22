import { useEffect, useMemo, useRef, useState } from 'react';
import { TransformerEngineClient } from './engine/client';
import type { EngineInitResult, EngineTraceResult } from './engine/types';

const PRESETS = [
  'Once upon a time, a little dog named Max found a red ball. He',
  'Lily wanted to bake a cake with her mom, so she',
  'The cat sat on the mat and looked at the',
];

function formatTokenLabel(raw: string): string {
  if (raw === '\n') return '\\n';
  return raw.replace(/^ /, '·');
}

export function App() {
  const clientRef = useRef<TransformerEngineClient | null>(null);
  const [initInfo, setInitInfo] = useState<EngineInitResult | null>(null);
  const [prompt, setPrompt] = useState(PRESETS[0]);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);
  const [ablateL2H3, setAblateL2H3] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const runPrompt = async (text: string, withAblation: boolean) => {
    const client = clientRef.current;
    if (!client) return;
    setBusy(true);
    setError(null);
    try {
      const baseRes = await client.trace(text);
      setBaseline(baseRes);
      if (withAblation) {
        const ablRes = await client.trace(text, { disabledHeads: [[2, 3]] });
        setAblated(ablRes);
      } else {
        setAblated(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const client = new TransformerEngineClient();
    clientRef.current = client;
    const modelBaseUrl = new URL(
      `${import.meta.env.BASE_URL}model/`,
      window.location.origin
    ).href;

    client
      .init(modelBaseUrl)
      .then((info) => {
        setInitInfo(info);
        return runPrompt(PRESETS[0], false);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      client.dispose();
      clientRef.current = null;
    };
  }, []);

  const topBase = useMemo(() => {
    if (!baseline || !initInfo) return [];
    return baseline.trace.finalProbs
      .map((prob, id) => ({
        id,
        text: initInfo.tokenizer.vocab[id],
        prob,
        ablProb: ablated ? ablated.trace.finalProbs[id] : null,
      }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 6);
  }, [baseline, ablated, initInfo]);

  return (
    <main className="page">
      <header className="header">
        <div>
          <h1>LLM Atlas — Live Truth Engine</h1>
          <div className="muted">
            Milestone 2 verification shell. Every value below is computed live in your browser from real trained weights.
          </div>
        </div>
        {initInfo && (
          <div className="badge mono">
            {initInfo.manifest.total_params.toLocaleString()} params · sha{' '}
            {initInfo.manifest.weights_sha256.slice(0, 8)} · loaded in{' '}
            {initInfo.loadMs}ms
          </div>
        )}
      </header>

      {error && <div className="error-box">{error}</div>}

      <section className="card">
        <h2>Input text (max 64 tokens)</h2>
        <form
          className="prompt-row"
          onSubmit={(e) => {
            e.preventDefault();
            void runPrompt(prompt, ablateL2H3);
          }}
        >
          <input
            className="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            aria-label="Model input text"
          />
          <button className="btn btn-primary" type="submit" disabled={busy || !initInfo}>
            {busy ? 'Running...' : 'Run forward pass'}
          </button>
          <button
            type="button"
            className={`btn ${ablateL2H3 ? 'btn-active' : ''}`}
            disabled={busy || !initInfo}
            onClick={() => {
              const next = !ablateL2H3;
              setAblateL2H3(next);
              void runPrompt(prompt, next);
            }}
          >
            {ablateL2H3 ? 'Ablation ON: Layer 2 Head 3 off' : 'Test ablation (disable L2 Head 3)'}
          </button>
        </form>
        <div className="presets">
          {PRESETS.map((p) => (
            <button
              key={p}
              type="button"
              className="btn"
              onClick={() => {
                setPrompt(p);
                void runPrompt(p, ablateL2H3);
              }}
            >
              {p.slice(0, 38)}...
            </button>
          ))}
        </div>
      </section>

      {baseline && initInfo && (
        <>
          <section className="card">
            <h2>
              Real BPE tokens ({baseline.tokens.length}) · forward pass in{' '}
              <span className="mono">{baseline.elapsedMs}ms</span>
              {baseline.trace.inputLoss !== null && (
                <>
                  {' '}· input loss <span className="mono">{baseline.trace.inputLoss.toFixed(3)}</span>
                </>
              )}
            </h2>
            <div className="token-strip">
              {baseline.tokens.map((tok, i) => (
                <span key={`${tok.start}-${i}`} className="token-pill">
                  <span className="mono">{formatTokenLabel(tok.text)}</span>
                  <span className="token-id mono">#{tok.id}</span>
                </span>
              ))}
            </div>
          </section>

          <div className="grid-2">
            <section className="card">
              <h2>Next-token probability distribution</h2>
              {topBase.map((item) => (
                <div key={item.id} className="prob-row">
                  <span className="mono">{formatTokenLabel(item.text)}</span>
                  <div>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${Math.min(100, item.prob * 100)}%` }}
                      />
                    </div>
                    {item.ablProb !== null && (
                      <div className="bar-track" style={{ marginTop: 3 }}>
                        <div
                          className="bar-fill-ablated"
                          style={{
                            width: `${Math.min(100, item.ablProb * 100)}%`,
                          }}
                        />
                      </div>
                    )}
                  </div>
                  <span className="mono">
                    {(item.prob * 100).toFixed(1)}%
                    {item.ablProb !== null &&
                      ` → ${(item.ablProb * 100).toFixed(1)}%`}
                  </span>
                </div>
              ))}
              {ablated &&
                baseline.trace.inputLoss !== null &&
                ablated.trace.inputLoss !== null && (
                  <p className="muted mono" style={{ marginTop: 12 }}>
                    Loss shift with Layer 2 Head 3 disabled:{' '}
                    {baseline.trace.inputLoss.toFixed(3)} →{' '}
                    {ablated.trace.inputLoss.toFixed(3)} (
                    {(
                      ablated.trace.inputLoss - baseline.trace.inputLoss
                    ).toFixed(3)}
                    )
                  </p>
                )}
            </section>

            <section className="card">
              <h2>Logit Lens across the 4 transformer blocks</h2>
              {baseline.trace.logitLens.map((entry) => {
                const top = entry.topTokens[0];
                const tokText = initInfo.tokenizer.vocab[top.id];
                return (
                  <div key={entry.afterLayer} className="prob-row">
                    <span className="mono">Block {entry.afterLayer}</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${Math.min(100, top.prob * 100)}%` }}
                      />
                    </div>
                    <span className="mono">
                      {formatTokenLabel(tokText)} ({(top.prob * 100).toFixed(0)}%)
                    </span>
                  </div>
                );
              })}
            </section>
          </div>
        </>
      )}
    </main>
  );
}
