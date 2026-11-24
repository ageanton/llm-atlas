import { useMemo, useState } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import type { SceneBeatId } from '../content/bookData';
import type { Edition } from '../state/bookState';

interface BeatFormulaSpec {
  tex: string;
  legend?: { color: string; label: string }[];
  workedExample?: {
    title: string;
    steps: string[];
  };
}

const BEAT_FORMULAS: Partial<Record<SceneBeatId, BeatFormulaSpec>> = {
  'ch2-2-pca-cloud': {
    tex: String.raw`\cos(\textcolor{#1d4ed8}{\mathbf{u}}, \textcolor{#b45309}{\mathbf{v}}) = \frac{\textcolor{#1d4ed8}{\mathbf{u}} \cdot \textcolor{#b45309}{\mathbf{v}}}{\|\textcolor{#1d4ed8}{\mathbf{u}}\|_2 \, \|\textcolor{#b45309}{\mathbf{v}}\|_2}`,
    legend: [
      { color: '#1d4ed8', label: 'Selected token vector' },
      { color: '#b45309', label: 'Candidate neighbor' },
    ],
    workedExample: {
      title: '3-Number Worked Example: Cosine Similarity',
      steps: [
        'Suppose dog = [0.8, 0.6, 0.0] and cat = [0.6, 0.8, 0.0] (both length 1.0).',
        'Dot product = (0.8 × 0.6) + (0.6 × 0.8) + (0 × 0) = 0.48 + 0.48 = 0.96.',
        'Cosine = 0.96 / (1.0 × 1.0) = +0.96 → nearly identical direction!',
      ],
    },
  },
  'ch2-4-wpe-add': {
    tex: String.raw`\textcolor{#0f766e}{H_0}[t] = \textcolor{#1d4ed8}{W_E}[x_t] + \textcolor{#b45309}{W_P}[t] \in \mathbb{R}^{128}`,
    legend: [
      { color: '#1d4ed8', label: 'Token embedding' },
      { color: '#b45309', label: 'Slot position' },
      { color: '#0f766e', label: 'Starting stream h₀' },
    ],
  },
  'ch2-5-rope': {
    tex: String.raw`\langle R_{\textcolor{#1d4ed8}{m}} \mathbf{q},\, R_{\textcolor{#b45309}{n}} \mathbf{k} \rangle = \mathbf{q}^\top R_{\textcolor{#0f766e}{n - m}} \mathbf{k}`,
    legend: [
      { color: '#1d4ed8', label: 'Slot A (m)' },
      { color: '#b45309', label: 'Slot B (n)' },
      { color: '#0f766e', label: 'Pure relative distance (n − m)' },
    ],
  },
  'ch3-1-residual-bus': {
    tex: String.raw`h_{l+1} = \textcolor{#1d4ed8}{h_l} + \textcolor{#2563eb}{\Delta_{\text{Attn}}}(\text{LN}_1(h_l)) + \textcolor{#b45309}{\Delta_{\text{MLP}}}(\text{LN}_2(u_l))`,
  },
  'ch3-3-layernorm': {
    tex: String.raw`\text{LN}(\mathbf{v}) = \frac{\mathbf{v} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta, \qquad \text{RMSNorm}(\mathbf{v}) = \frac{\mathbf{v}}{\sqrt{\frac{1}{d}\sum v_i^2 + \epsilon}} \odot \gamma`,
  },
  'ch3-4-rmsnorm': {
    tex: String.raw`\text{RMSNorm}(\mathbf{v}) = \frac{\mathbf{v}}{\sqrt{\frac{1}{d}\|\mathbf{v}\|_2^2 + \epsilon}} \odot \gamma`,
  },
  'attn-2-dot-product': {
    tex: String.raw`S_{i,j} = \frac{\textcolor{#1d4ed8}{\mathbf{q}_i} \cdot \textcolor{#0f766e}{\mathbf{k}_j}}{\sqrt{d_k}} = \frac{1}{\sqrt{32}} \sum_{d=0}^{31} \textcolor{#1d4ed8}{q_{i,d}}\,\textcolor{#0f766e}{k_{j,d}}`,
    legend: [
      { color: '#1d4ed8', label: 'Query vector q' },
      { color: '#0f766e', label: 'Key vector k' },
    ],
    workedExample: {
      title: '3-Number Worked Example: Scaled Dot Product',
      steps: [
        'Let q = [0.8, -0.5, 0.2] and k = [0.6, -0.4, -0.5] (dimension d = 3).',
        'Pairwise products: (0.8 × 0.6) = +0.48, (-0.5 × -0.4) = +0.20, (0.2 × -0.5) = -0.10.',
        'Sum = 0.48 + 0.20 - 0.10 = +0.58. Divide by √3 (1.732) → score = +0.335.',
      ],
    },
  },
  'attn-4-mask-softmax': {
    tex: String.raw`A_{i,j} = \text{softmax}\!\left(\frac{\textcolor{#1d4ed8}{\mathbf{q}_i} \cdot \textcolor{#0f766e}{\mathbf{k}_j}}{\sqrt{32}} + \textcolor{#dc2626}{M_{i,j}}\right), \quad \textcolor{#dc2626}{M_{i,j}} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}`,
    legend: [
      { color: '#dc2626', label: 'Causal mask (e⁻∞ = 0, but e⁰ = 1!)' },
    ],
  },
  'attn-5-value-mixer': {
    tex: String.raw`\textcolor{#0f766e}{\mathbf{z}_i} = \sum_{j=0}^{i} \textcolor{#1d4ed8}{A_{i,j}} \, \textcolor{#b45309}{\mathbf{v}_j} \in \mathbb{R}^{32}`,
    legend: [
      { color: '#1d4ed8', label: 'Attention weight A[i,j]' },
      { color: '#b45309', label: 'Source Value v_j' },
      { color: '#0f766e', label: 'Blended output A·V' },
    ],
    workedExample: {
      title: '3-Number Worked Example: Mixing Value Vectors',
      steps: [
        'Suppose "He" gives 70% weight to V("Max") = [1.0, 0.0, 0.4] and 30% to V("ball") = [0.0, 1.0, 0.2].',
        'Scale each vector: 0.7 × [1.0, 0.0, 0.4] = [0.70, 0.00, 0.28]; 0.3 × [0.0, 1.0, 0.2] = [0.00, 0.30, 0.06].',
        'Add them together → blended output = [0.70, 0.30, 0.34] (carries BOTH Max and ball!).',
      ],
    },
  },
  'attn-6-multihead-full': {
    tex: String.raw`\text{MHA}(X) = \text{Concat}\!\left(\text{softmax}\!\left(\frac{\textcolor{#1d4ed8}{Q_h}\textcolor{#0f766e}{K_h}^\top}{\sqrt{32}} + \textcolor{#dc2626}{M}\right)\textcolor{#b45309}{V_h}\right)_{h=0}^{3} W_O^\top`,
  },
  'ch5-3-gelu': {
    tex: String.raw`\text{GELU}(x) \approx \frac{1}{2}x\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}(x + 0.044715\,x^3)\right)\right)`,
    workedExample: {
      title: '3-Number Worked Example: Why GELU Matters',
      steps: [
        'Try x = -2.0 → GELU(-2.0) = -0.046 (strong negative signal squashed near zero).',
        'Try x = 0.0 → GELU(0.0) = 0.000 (neutral sits cleanly at zero).',
        'Try x = +2.0 → GELU(+2.0) = +1.954 (strong positive pattern passes nearly untouched).',
      ],
    },
  },
  'ch6-3-temperature': {
    tex: String.raw`P_{\textcolor{#b45309}{\tau}}(v) = \frac{\exp(z_v / \textcolor{#b45309}{\tau})}{\sum_{u=1}^{512} \exp(z_u / \textcolor{#b45309}{\tau})}`,
    legend: [{ color: '#b45309', label: 'Temperature τ (low = sharp, high = flat)' }],
  },
  'ch7-2-cross-entropy': {
    tex: String.raw`\mathcal{L} = -\frac{1}{T-1}\sum_{t=0}^{T-2} \log P(x_{t+1} \mid x_{\le t}), \qquad \text{Perplexity} = e^{\mathcal{L}}`,
    workedExample: {
      title: '3-Number Worked Example: Cross-Entropy Surprise',
      steps: [
        'If the model gives the true next word P = 0.90 → Loss = -ln(0.90) = 0.105 (tiny penalty).',
        'If it gives the true word P = 0.10 → Loss = -ln(0.10) = 2.303 (22× bigger penalty!).',
        'If it gives P = 0.01 → Loss = -ln(0.01) = 4.605 (severe shock drives a huge gradient update).',
      ],
    },
  },
  'ch7-3-manual-backprop': {
    tex: String.raw`w_{\text{new}} = w - \eta\,\frac{\partial \mathcal{L}}{\partial w}, \qquad \frac{\partial \mathcal{L}}{\partial w_{v,d}} = \frac{1}{T-1}\sum_t \left(P_t(v) - \mathbf{1}_{y_t=v}\right) h_{t,d}`,
  },
};

interface MathFormulaProps {
  beatId: SceneBeatId;
  edition: Edition;
}

export function MathFormula({ beatId, edition }: MathFormulaProps) {
  const [showWorked, setShowWorked] = useState(false);
  const spec = BEAT_FORMULAS[beatId];

  const html = useMemo(() => {
    if (!spec) return '';
    return katex.renderToString(spec.tex, {
      displayMode: true,
      throwOnError: false,
    });
  }, [spec]);

  if (!spec) return null;

  return (
    <div className="math-block">
      {edition !== 'foundations' && (
        <div className="katex-card">
          <div
            className="katex-render"
            dangerouslySetInnerHTML={{ __html: html }}
          />
          {spec.legend && spec.legend.length > 0 && (
            <div className="math-legend">
              {spec.legend.map((item) => (
                <span key={item.label} className="math-legend-item">
                  <span
                    className="math-dot"
                    style={{ backgroundColor: item.color }}
                  />
                  {item.label}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {spec.workedExample && (
        <div className="worked-card">
          <button
            type="button"
            className="worked-toggle"
            onClick={() => setShowWorked((v) => !v)}
            aria-expanded={showWorked}
          >
            <span>{spec.workedExample.title}</span>
            <span className="mono">{showWorked ? 'Hide −' : 'Verify by hand +'}</span>
          </button>
          {showWorked && (
            <ol className="worked-steps">
              {spec.workedExample.steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
