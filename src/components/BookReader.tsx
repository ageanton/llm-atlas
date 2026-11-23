import {
  editionLabel,
  TEXTBOOK_CHAPTERS,
  type BookBeat,
  type ChapterMeta,
} from '../content/bookData';
import type { Edition } from '../state/bookState';
import { PredictCard } from './PredictCard';

const EDITIONS: Edition[] = ['foundations', 'practitioner', 'formal'];

interface BookReaderProps {
  chapter: ChapterMeta;
  beat: BookBeat | null;
  beatIndex: number;
  edition: Edition;
  showContents: boolean;
  liveCardLines: string[];
  onToggleContents: () => void;
  onSelectChapter: (chapterId: number) => void;
  onSelectBeat: (beatIndex: number) => void;
  onChangeEdition: (edition: Edition) => void;
}

export function BookReader({
  chapter,
  beat,
  beatIndex,
  edition,
  showContents,
  liveCardLines,
  onToggleContents,
  onSelectChapter,
  onSelectBeat,
  onChangeEdition,
}: BookReaderProps) {
  const totalBeats = chapter.beats.length;

  return (
    <aside className="book-pane" aria-label="Interactive LLM Textbook">
      <div className="book-header">
        <div className="book-header-top">
          <button
            type="button"
            className={`outline-btn ${showContents ? 'is-active' : ''}`}
            onClick={onToggleContents}
          >
            {showContents ? 'Back to Chapter' : `Contents (Ch. ${chapter.id} of 8)`}
          </button>

          <div className="edition-switch" role="group" aria-label="Academic edition">
            {EDITIONS.map((ed) => (
              <button
                key={ed}
                type="button"
                className={`ed-btn ${edition === ed ? 'is-active' : ''}`}
                onClick={() => onChangeEdition(ed)}
              >
                {editionLabel(ed)}
              </button>
            ))}
          </div>
        </div>

        {!showContents && totalBeats > 0 && (
          <nav className="beat-stepper" aria-label="Chapter beats">
            {chapter.beats.map((b, idx) => (
              <button
                key={b.id}
                type="button"
                className={`beat-pill ${idx === beatIndex ? 'is-current' : ''}`}
                onClick={() => onSelectBeat(idx)}
              >
                {b.shortTitle}
              </button>
            ))}
          </nav>
        )}
      </div>

      <div className="book-scroll">
        {showContents ? (
          <div className="toc-list">
            <div className="toc-intro">
              <div className="chapter-kicker">8-Chapter Interactive Curriculum</div>
              <h1 className="toc-heading">Inventing the Transformer from Scratch</h1>
              <p className="book-para">
                Every chapter opens with a concrete bottleneck on a real sentence, lets you test why the machine needs a new mechanism, and drives the live 3D simulation one step at a time.
              </p>
            </div>

            {TEXTBOOK_CHAPTERS.map((ch) => (
              <button
                key={ch.id}
                type="button"
                className={`toc-item ${ch.id === chapter.id ? 'is-current' : ''}`}
                onClick={() => onSelectChapter(ch.id)}
              >
                <div className="toc-item-top">
                  <span className="toc-num mono">Ch. {ch.id}</span>
                  <span
                    className={`toc-status ${
                      ch.status === 'ready' ? 'is-live' : 'is-upcoming'
                    }`}
                  >
                    {ch.status === 'ready' ? 'Live Interactive' : 'Pass 2 / 3'}
                  </span>
                </div>
                <div className="toc-title">{ch.title}</div>
                <div className="toc-sub">{ch.subtitle}</div>
              </button>
            ))}
          </div>
        ) : chapter.status === 'upcoming' || !beat ? (
          <div className="upcoming-chapter">
            <div className="chapter-kicker">Chapter {chapter.id} of 8</div>
            <h1 className="chapter-title">{chapter.title}</h1>
            <p className="chapter-sub">{chapter.subtitle}</p>
            <div className="mystery-box">
              <div className="mystery-label">Opening Mystery</div>
              <p>{chapter.openingMystery}</p>
            </div>
            <p className="book-para">
              This chapter is scheduled in Build Pass 2 &amp; 3 once we lock the teaching quality of our flagship Chapter 4 engine.
            </p>
            <button
              type="button"
              className="primary-btn"
              onClick={() => onSelectChapter(4)}
            >
              Open Flagship Chapter 4: Self-Attention
            </button>
          </div>
        ) : (
          <article className="beat-article" aria-live="polite">
            <div className="chapter-kicker">
              Chapter {chapter.id} · Beat {beat.beatNumber} of {totalBeats}
            </div>
            <h1 className="beat-heading">{beat.heading}</h1>

            <div className="stage-cue">{beat.visualFocusNote}</div>

            {beat.paragraphs[edition].map((para, i) => (
              <p key={i} className="book-para">
                {para}
              </p>
            ))}

            {liveCardLines.length > 0 && (
              <div className="live-proof-box mono">
                <div className="live-proof-title">Live Numbers From Your Sentence</div>
                {liveCardLines.map((line, i) => (
                  <div key={i} className="live-proof-line">
                    {line}
                  </div>
                ))}
              </div>
            )}

            <div className="takeaway-box">
              <span className="takeaway-label">Core Takeaway</span>
              <p>{beat.keyTakeaway[edition]}</p>
            </div>

            {beat.predict && <PredictCard challenge={beat.predict} />}

            <div className="beat-footer-nav">
              <button
                type="button"
                className="secondary-btn"
                disabled={beatIndex <= 0}
                onClick={() => onSelectBeat(beatIndex - 1)}
              >
                ← Previous Beat
              </button>
              <button
                type="button"
                className="primary-btn"
                disabled={beatIndex >= totalBeats - 1}
                onClick={() => onSelectBeat(beatIndex + 1)}
              >
                {beatIndex < totalBeats - 1
                  ? `Next: ${chapter.beats[beatIndex + 1].shortTitle} →`
                  : 'End of Chapter 4'}
              </button>
            </div>
          </article>
        )}
      </div>
    </aside>
  );
}
