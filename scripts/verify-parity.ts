import fs from 'node:fs';
import path from 'node:path';
import { TinyBPETokenizer } from '../src/engine/tokenizer';
import { loadWeights, runForwardTrace } from '../src/engine/transformer';
import type {
  ActiveLayerTrace,
  ModelManifest,
  TokenizerData,
} from '../src/engine/types';

interface RefToken {
  id: number;
  text: string;
  start: number;
  end: number;
}

interface RefLayer {
  skipped: boolean;
  attn?: { attn_weights: number[][][] };
  mlp?: { hidden: number[][] };
}

interface RefTrace {
  token_ids: number[];
  tok_emb: number[][];
  pos_emb: number[][];
  layers: RefLayer[];
  final_logits: number[];
  final_probs: number[];
  greedy_next_token_id: number;
  input_loss: number | null;
}

interface ReferencePayload {
  prompt: string;
  tokens: RefToken[];
  baseline: RefTrace;
  ablation_disable_L2_H3: RefTrace;
}

function maxDiff1D(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`1D length mismatch: ${a.length} vs ${b.length}`);
  }
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function maxDiff2D(a: number[][], b: number[][]): number {
  if (a.length !== b.length) {
    throw new Error(`2D row mismatch: ${a.length} vs ${b.length}`);
  }
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = maxDiff1D(a[i], b[i]);
    if (d > m) m = d;
  }
  return m;
}

function maxDiff3D(a: number[][][], b: number[][][]): number {
  if (a.length !== b.length) {
    throw new Error(`3D depth mismatch: ${a.length} vs ${b.length}`);
  }
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = maxDiff2D(a[i], b[i]);
    if (d > m) m = d;
  }
  return m;
}

function main(): void {
  const modelDir = path.resolve(process.cwd(), 'public/model');
  const refPath = path.join(modelDir, 'reference_trace.json');
  if (!fs.existsSync(refPath)) {
    console.error(
      'ERROR: public/model/reference_trace.json not found. Run the "Train Miniature Transformer" GitHub Action first.'
    );
    process.exit(1);
  }

  const manifest = JSON.parse(
    fs.readFileSync(path.join(modelDir, 'manifest.json'), 'utf8')
  ) as ModelManifest;
  const tokData = JSON.parse(
    fs.readFileSync(path.join(modelDir, 'tokenizer.json'), 'utf8')
  ) as TokenizerData;
  const ref = JSON.parse(fs.readFileSync(refPath, 'utf8')) as ReferencePayload;

  const rawBuf = fs.readFileSync(path.join(modelDir, manifest.weights_file));
  const arrayBuf = rawBuf.buffer.slice(
    rawBuf.byteOffset,
    rawBuf.byteOffset + rawBuf.byteLength
  );

  const tokenizer = new TinyBPETokenizer(tokData);
  const store = loadWeights(manifest, arrayBuf);

  // 1. Tokenizer parity
  const tsTokens = tokenizer.encodeWithSpans(ref.prompt);
  if (tsTokens.length !== ref.tokens.length) {
    throw new Error(
      `Token count mismatch: TS=${tsTokens.length}, Py=${ref.tokens.length}`
    );
  }
  for (let i = 0; i < tsTokens.length; i++) {
    const a = tsTokens[i];
    const b = ref.tokens[i];
    if (
      a.id !== b.id ||
      a.text !== b.text ||
      a.start !== b.start ||
      a.end !== b.end
    ) {
      throw new Error(
        `Token ${i} mismatch: TS=${JSON.stringify(a)} vs Py=${JSON.stringify(b)}`
      );
    }
  }
  console.log(`Tokenizer parity PASS (${tsTokens.length} tokens exact match).`);

  const TOL = 1e-4;

  // 2. Baseline forward trace parity
  const tsBase = runForwardTrace(store, ref.baseline.token_ids);
  const embDiff = Math.max(
    maxDiff2D(tsBase.tokEmb, ref.baseline.tok_emb),
    maxDiff2D(tsBase.posEmb, ref.baseline.pos_emb)
  );

  let maxAttnDiff = 0;
  let maxMlpDiff = 0;
  for (let l = 0; l < tsBase.layers.length; l++) {
    const tsLayer = tsBase.layers[l] as ActiveLayerTrace;
    const pyLayer = ref.baseline.layers[l];
    if (pyLayer.attn) {
      maxAttnDiff = Math.max(
        maxAttnDiff,
        maxDiff3D(tsLayer.attn.attnWeights, pyLayer.attn.attn_weights)
      );
    }
    if (pyLayer.mlp) {
      maxMlpDiff = Math.max(
        maxMlpDiff,
        maxDiff2D(tsLayer.mlp.hidden, pyLayer.mlp.hidden)
      );
    }
  }

  const baseLogitDiff = maxDiff1D(
    tsBase.finalLogits,
    ref.baseline.final_logits
  );
  const baseProbDiff = maxDiff1D(tsBase.finalProbs, ref.baseline.final_probs);
  const baseLossDiff =
    tsBase.inputLoss !== null && ref.baseline.input_loss !== null
      ? Math.abs(tsBase.inputLoss - ref.baseline.input_loss)
      : 0;

  if (tsBase.greedyNextTokenId !== ref.baseline.greedy_next_token_id) {
    throw new Error(
      `Baseline greedy token mismatch: TS=${tsBase.greedyNextTokenId} vs Py=${ref.baseline.greedy_next_token_id}`
    );
  }

  // 3. Ablated forward trace parity (Disable Layer 2, Head 3)
  const tsAbl = runForwardTrace(store, ref.ablation_disable_L2_H3.token_ids, {
    disabledHeads: [[2, 3]],
  });
  const ablLogitDiff = maxDiff1D(
    tsAbl.finalLogits,
    ref.ablation_disable_L2_H3.final_logits
  );
  const ablProbDiff = maxDiff1D(
    tsAbl.finalProbs,
    ref.ablation_disable_L2_H3.final_probs
  );
  if (
    tsAbl.greedyNextTokenId !== ref.ablation_disable_L2_H3.greedy_next_token_id
  ) {
    throw new Error(
      `Ablated greedy token mismatch: TS=${tsAbl.greedyNextTokenId} vs Py=${ref.ablation_disable_L2_H3.greedy_next_token_id}`
    );
  }

  const worst = Math.max(
    embDiff,
    maxAttnDiff,
    maxMlpDiff,
    baseLogitDiff,
    baseProbDiff,
    baseLossDiff,
    ablLogitDiff,
    ablProbDiff
  );

  console.log(
    [
      `Baseline parity:`,
      `  embeddings max diff : ${embDiff.toExponential(3)}`,
      `  attnWeights max diff: ${maxAttnDiff.toExponential(3)}`,
      `  mlpHidden max diff  : ${maxMlpDiff.toExponential(3)}`,
      `  finalLogits max diff: ${baseLogitDiff.toExponential(3)}`,
      `  finalProbs max diff : ${baseProbDiff.toExponential(3)}`,
      `  inputLoss diff      : ${baseLossDiff.toExponential(3)}`,
      `Ablation (L2 H3 off) parity:`,
      `  finalLogits max diff: ${ablLogitDiff.toExponential(3)}`,
      `  finalProbs max diff : ${ablProbDiff.toExponential(3)}`,
    ].join('\n')
  );

  if (worst > TOL) {
    console.error(`PARITY FAILED: max diff ${worst} exceeded ${TOL}`);
    process.exit(1);
  }
  console.log(`ALL PARITY CHECKS PASSED (worst error ${worst.toExponential(3)} <= ${TOL}).`);
}

main();
