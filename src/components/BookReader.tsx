import {
  editionLabel,
  TEXTBOOK_CHAPTERS,
  type BookBeat,
  type ChapterMeta,
} from '../content/bookData';
import { PredictCard } from './PredictCard';
import type { BookState, Edition } from '../state/bookState';
import { formatToken } from '../utils/math';
import type { EngineTraceResult } from '../engine/types';

interface BookReaderProps {
  state: BookState;
  trace: EngineTraceResult | null;
  onUpdateState: (fn: (prev: BookState) => BookState) => void;
  onOpenContents?: () => void;
}

export function BookReader({
  state,
  trace,
  onUpdateState,
  onOpenContents,
}: BookReaderProps) {
  const currentChapter: ChapterMeta =
    TEXTBOOK_CHAPTERS.find((c) => c.id === state.chapterId) ??
    TEXTBOOK_CHAPTERS[3];

  const currentBeat: BookBeat | undefined =
    currentChapter.beats[state.beatIndex] ?? currentChapter.beats[0];

  const edition = state.edition;

  const handleSelectEdition = (ed: Edition) => {
    onUpdateState((prev) => ({ ...prev, edition: ed }));
  };

  const handleSelectBeat = (idx: number) => {
    onUpdateState((prev) => ({
      ...prev,
      beatIndex: idx,
    }));
  };

  return (
    <aside className="book-reader-pane">
      <header className="reader-header">
        <div className="reader-chapter-badge">
          <button
            type="button"
            className="chapter-indicator-btn"
            onClick={onOpenContents}
            title="Open Table of Contents"
          >
            Chapter {currentChapter.id}: {currentChapter.title} ▾
          </button>
        </div>

        <div className="edition-selector" role="tablist" aria-label="Book Edition">
          {(['foundations', 'practitioner', 'formal'] as Edition[]).map(
            (ed) => (
              <button
                key={ed}
                role="tab"
                aria-selected={edition === ed}
                className={`edition-tab ${edition === ed ? 'is-active' : ''}`}
                onClick={() => handleSelectEdition(ed)}
              >
                {editionLabel(ed)}
              </button>
            )
          )}
        </div>
      </header>

      {/* Beat Stepper Ribbon */}
      <nav className="beat-nav" aria-label="Chapter Beats">
        {currentChapter.beats.map((b, idx) => (
          <button
            key={b.id}
            type="button"
            className={`beat-step ${state.beatIndex === idx ? 'is-active' : ''}`}
            onClick={() => handleSelectBeat(idx)}
          >
            <span className="beat-num">{b.beatNumber}</span>
            <span className="beat-name">{b.shortName}</span>
          </button>
        ))}
      </nav>

      {/* Main Beat Content */}
      <div className="reader-scroll-body">
        {currentBeat && (
          <article className="beat-content">
            <header className="beat-hero">
              <span className="beat-kicker">
                Beat {currentBeat.beatNumber} of {currentChapter.beats.length}
              </span>
              <h1 className="beat-title">{currentBeat.title}</h1>
              <p className="beat-subtitle">{currentBeat.subtitle}</p>
            </header>

            <div className="interactive-cue-card">
              <span className="cue-label">Direct Inspection</span>
              <p className="cue-text">{currentBeat.inspectInstructions}</p>
            </div>

            {/* Prediction Challenge */}
            {currentBeat.prediction && (
              <PredictCard challenge={currentBeat.prediction} />
            )}

            {/* Register Summary */}
            <section className="register-section">
              <h2 className="register-heading">
                {edition === 'foundations' && 'Intuition & Mechanics'}
                {edition === 'practitioner' && 'Implementation & Tensors'}
                {edition === 'formal' && 'Formal Definition & Math'}
              </h2>
              <p className="register-body-text">
                {currentBeat.summary[edition]}
              </p>
            </section>

            {/* LaTeX Math Formula */}
            {edition === 'formal' && currentBeat.equationLatex && (
              <div className="math-display">
                <code>{currentBeat.equationLatex}</code>
              </div>
            )}

            {/* PyTorch Code */}
            {edition === 'practitioner' && currentBeat.codeBlock && (
              <pre className="code-display">
                <code>{currentBeat.codeBlock}</code>
              </pre>
            )}

            {/* Deep Dive */}
            <section className="deep-dive-section">
              <h3 className="deep-dive-heading">Under the Hood</h3>
              <p className="deep-dive-text">{currentBeat.deepDive[edition]}</p>
            </section>
          </article>
        )}
      </div>

      {/* Footer Navigation */}
      <div className="reader-footer-nav">
        <button
          type="button"
          className="nav-btn"
          disabled={state.beatIndex <= 0}
          onClick={() => handleSelectBeat(state.beatIndex - 1)}
        >
          ← Previous Beat
        </button>
        <span className="nav-counter">
          {state.beatIndex + 1} / {currentChapter.beats.length}
        </span>
        <button
          type="button"
          className="nav-btn is-primary"
          disabled={state.beatIndex >= currentChapter.beats.length - 1}
          onClick={() => handleSelectBeat(state.beatIndex + 1)}
        >
          Next Beat →
        </button>
      </div>
    </aside>
  );
}
