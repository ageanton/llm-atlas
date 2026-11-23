import {
  ALL_CHAPTERS,
  type Chapter,
  type ChapterBeat,
  editionLabel,
} from '../content/bookData';
import { PredictCard } from './PredictCard';
import type { BookState, Edition } from '../state/bookState';
import { formatToken } from '../utils/math';
import type { ModelTrace } from '../engine/types';

interface BookReaderProps {
  state: BookState;
  trace: ModelTrace | null;
  onUpdateState: (fn: (prev: BookState) => BookState) => void;
}

export function BookReader({ state, trace, onUpdateState }: BookReaderProps) {
  const currentChapter: Chapter =
    ALL_CHAPTERS.find((c) => c.id === state.chapterId) ?? ALL_CHAPTERS[3];

  const currentBeat: ChapterBeat | undefined =
    currentChapter.beats[state.beatIndex] ?? currentChapter.beats[0];

  const edition = state.edition;
  const registerContent = currentBeat ? currentBeat.registers[edition] : null;

  const handleSelectChapter = (id: number) => {
    onUpdateState((prev) => ({
      ...prev,
      chapterId: id,
      beatIndex: 0,
    }));
  };

  const handleSelectBeat = (idx: number) => {
    onUpdateState((prev) => ({
      ...prev,
      beatIndex: idx,
    }));
  };

  const handleSelectEdition = (ed: Edition) => {
    onUpdateState((prev) => ({
      ...prev,
      edition: ed,
    }));
  };

  return (
    <div className="book-reader-container">
      {/* Chapter Dropdown / Nav */}
      <header className="book-header">
        <div className="book-nav-row">
          <label htmlFor="chapter-select" className="book-label">
            Chapter:
          </label>
          <select
            id="chapter-select"
            className="chapter-select"
            value={state.chapterId}
            onChange={(e) => handleSelectChapter(Number(e.target.value))}
          >
            {ALL_CHAPTERS.map((ch) => (
              <option
                key={ch.id}
                value={ch.id}
                disabled={ch.status === 'upcoming'}
              >
                {ch.id}. {ch.title} {ch.status === 'upcoming' ? '(Upcoming)' : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="book-meta-bar">
          <div className="edition-tabs" role="tablist" aria-label="Book Edition">
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
        </div>
      </header>

      {/* Chapter Opening Mystery */}
      <section className="mystery-banner">
        <div className="mystery-badge">Core Mystery</div>
        <div className="mystery-text">"{currentChapter.openingMystery}"</div>
      </section>

      {/* Beat Stepper Navigation */}
      {currentChapter.beats.length > 0 && (
        <nav className="beat-stepper" aria-label="Chapter Beats">
          {currentChapter.beats.map((b, idx) => (
            <button
              key={b.id}
              type="button"
              className={`beat-step-btn ${
                state.beatIndex === idx ? 'is-active' : ''
              }`}
              onClick={() => handleSelectBeat(idx)}
            >
              <span className="step-num">{idx + 1}</span>
              <span className="step-name">{b.title.split(':')[0]}</span>
            </button>
          ))}
        </nav>
      )}

      {/* Main Beat Reading Section */}
      {currentBeat && registerContent && (
        <article className="beat-article">
          <div className="beat-title-area">
            <span className="beat-eyebrow">
              Beat {currentBeat.beatIndex + 1} of {currentChapter.beats.length}
            </span>
            <h2 className="beat-heading">{currentBeat.title}</h2>
            <p className="beat-subtitle">{currentBeat.subtitle}</p>
          </div>

          {/* Interactive Goal Box */}
          <div className="interactive-goal-box">
            <strong className="goal-tag">Interactive Goal</strong>
            <span>{currentBeat.interactiveGoal}</span>
          </div>

          {/* Predictive Question Challenge */}
          {currentBeat.predictChallenge && (
            <PredictCard challenge={currentBeat.predictChallenge} />
          )}

          {/* Multi-Register Content Area */}
          <div className="register-body">
            <h3 className="register-subheading">{registerContent.title}</h3>
            <p className="register-summary">{registerContent.summary}</p>

            {/* LaTeX Math Formula in Formal Edition */}
            {edition === 'formal' && registerContent.mathFormula && (
              <div className="math-block">
                <code className="math-code">{registerContent.mathFormula}</code>
              </div>
            )}

            {/* PyTorch Code in Practitioner Edition */}
            {edition === 'practitioner' && registerContent.codeSnippet && (
              <pre className="code-block">
                <code>{registerContent.codeSnippet}</code>
              </pre>
            )}

            {/* Bullet Points */}
            <ul className="beat-bullets">
              {registerContent.bulletPoints.map((pt, i) => (
                <li key={i}>{pt}</li>
              ))}
            </ul>

            {/* Intuition Trap Warning */}
            {registerContent.intuitionTrap && (
              <aside className="trap-box">
                <strong className="trap-title">Common Intuition Trap:</strong>
                <p>{registerContent.intuitionTrap}</p>
              </aside>
            )}
          </div>
        </article>
      )}

      {/* Navigation Footer */}
      <footer className="beat-footer-nav">
        <button
          type="button"
          className="btn-beat-nav"
          disabled={state.beatIndex <= 0}
          onClick={() => handleSelectBeat(state.beatIndex - 1)}
        >
          ← Previous Beat
        </button>
        <span className="beat-counter">
          {state.beatIndex + 1} / {currentChapter.beats.length}
        </span>
        <button
          type="button"
          className="btn-beat-nav is-primary"
          disabled={state.beatIndex >= currentChapter.beats.length - 1}
          onClick={() => handleSelectBeat(state.beatIndex + 1)}
        >
          Next Beat →
        </button>
      </footer>
    </div>
  );
}
