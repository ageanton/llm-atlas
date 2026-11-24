interface PipelineMinimapProps {
  activeChapterId: number;
  onSelectChapter: (chapterId: number) => void;
}

const PIPELINE_NODES: { ch: number; label: string }[] = [
  { ch: 1, label: '1. Tokens' },
  { ch: 2, label: '2. Embed' },
  { ch: 3, label: '3. Highway' },
  { ch: 4, label: '4. Attn' },
  { ch: 5, label: '5. MLP' },
  { ch: 6, label: '6. Output' },
  { ch: 7, label: '7. Learn' },
  { ch: 8, label: '8. Scale' },
];

export function PipelineMinimap({
  activeChapterId,
  onSelectChapter,
}: PipelineMinimapProps) {
  return (
    <nav className="pipeline-minimap" aria-label="Where you are in the Transformer">
      {PIPELINE_NODES.map((node, i) => (
        <span key={node.ch} className="minimap-item">
          {i > 0 && <span className="minimap-arrow">→</span>}
          <button
            type="button"
            className={`minimap-node ${
              node.ch === activeChapterId ? 'is-active' : ''
            }`}
            onClick={() => onSelectChapter(node.ch)}
          >
            {node.label}
          </button>
        </span>
      ))}
    </nav>
  );
}
