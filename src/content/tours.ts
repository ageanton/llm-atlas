import type { Register, TeachingMode, ZoomLevel } from '../state/urlState';

export interface TourStep {
  level: ZoomLevel;
  sel: string;
  register: Register;
  mode: TeachingMode;
  prompt?: string;
  message: string;
}

export interface GuidedTour {
  id: string;
  title: string;
  summary: string;
  steps: TourStep[];
}

export const GUIDED_TOURS: GuidedTour[] = [
  {
    id: 'predict-next',
    title: 'How does it predict the next word?',
    summary: 'Follow one sentence from the whole box down to a single weight.',
    steps: [
      {
        level: 0,
        sel: 'model',
        register: 'simple',
        mode: 'explore',
        prompt: 'Once upon a time, a little dog named Max found a red ball. He',
        message:
          'Start at Level 0: one story fragment enters on the left and a single highest-probability token exits on the right.',
      },
      {
        level: 1,
        sel: 'block2',
        register: 'simple',
        mode: 'explore',
        message:
          'At Level 1, watch the Logit Lens badges beneath Blocks 0 through 3—the model refines its next-token guess step by step.',
      },
      {
        level: 2,
        sel: 'block2',
        register: 'sufficient',
        mode: 'explore',
        message:
          'Zooming into Block 2: Multi-Head Attention mixes tokens across time, and the MLP reflects on the gathered information.',
      },
      {
        level: 3,
        sel: 'block2.attn.head3.K',
        register: 'sufficient',
        mode: 'explore',
        message:
          'In Head 3, the attention heatmap shows the pronoun "He" attending back to "Max" and "ball".',
      },
      {
        level: 4,
        sel: 'param:blocks.2.attn.c_attn.weight',
        register: 'complete',
        mode: 'explore',
        message:
          'At Level 4, inspect the 128×384 parameter matrix c_attn.weight that projects tokens into Q, K, and V.',
      },
      {
        level: 5,
        sel: 'param:blocks.2.attn.c_attn.weight@14,32',
        register: 'complete',
        mode: 'explore',
        message:
          'At Level 5, meet a single float16 number and look at its recorded training trajectory across 1,500 training steps.',
      },
    ],
  },
  {
    id: 'attention-tracking',
    title: 'Why attention connects "He" to "Max"',
    summary: 'See how self-attention resolves pronoun references in children stories.',
    steps: [
      {
        level: 3,
        sel: 'block2.attn.head3.K',
        register: 'sufficient',
        mode: 'explore',
        prompt: 'Once upon a time, a little dog named Max found a red ball. He',
        message:
          'Notice row 14 (" He"): look at columns 7 (" Max") and 12 (" ball"). Attention routes both entity context and toy context into the pronoun.',
      },
      {
        level: 3,
        sel: 'block1.attn.head1.K',
        register: 'sufficient',
        mode: 'explore',
        message:
          'Switch to Block 1 Head 1: notice the strong diagonal attention pattern. This head is primarily an adjacent-token grammar tracker.',
      },
    ],
  },
  {
    id: 'residual-highway',
    title: 'The Residual Stream Highway',
    summary: 'Understand why skip connections keep representations intact across depth.',
    steps: [
      {
        level: 1,
        sel: 'block0',
        register: 'sufficient',
        mode: 'explore',
        prompt: 'Lily wanted to bake a cake with her mom, so she',
        message:
          'The horizontal blue bar is the residual highway. Tokens travel along this stream, receiving additive updates at each block.',
      },
      {
        level: 1,
        sel: 'block3',
        register: 'complete',
        mode: 'explore',
        message:
          'Because blocks add rather than overwrite (x + f(x)), early token representations remain accessible all the way to Block 3.',
      },
    ],
  },
  {
    id: 'ablation-experiment',
    title: 'Ablation: Knocking out Block 2',
    summary: 'Test causality by zeroing out an entire transformer layer.',
    steps: [
      {
        level: 1,
        sel: 'block2',
        register: 'sufficient',
        mode: 'ablate',
        prompt: 'Once upon a time, a little dog named Max found a red ball. He',
        message:
          'We switched to Ablation mode. Toggle the zero checkbox on Block 2 to see the top predicted tokens shift immediately in real time.',
      },
    ],
  },
  {
    id: 'training-evolution',
    title: 'From Babble to Grammar: Training Replay',
    summary: 'Step through checkpoints from step 0 to step 1,500 to witness learning.',
    steps: [
      {
        level: 0,
        sel: 'model',
        register: 'sufficient',
        mode: 'replay',
        prompt: 'Once upon a time, a little dog named Max found a red ball. He',
        message:
          'Drag the checkpoint slider to step 0: loss is ~7.6 and generated text is random noise. Drag to step 300 to see punctuation appear, and step 1,500 for coherent grammar.',
      },
    ],
  },
];
