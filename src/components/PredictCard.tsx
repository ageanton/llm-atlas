import { useEffect, useState } from 'react';
import type { PredictChallenge } from '../content/bookData';

interface PredictCardProps {
  challenge: PredictChallenge;
}

export function PredictCard({ challenge }: PredictCardProps) {
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    setSelected(null);
  }, [challenge.id]);

  const hasAnswered = selected !== null;
  const isCorrect = selected === challenge.correctId;

  return (
    <div className="predict-card" aria-label="Prediction Challenge">
      <div className="predict-tag">Prediction Challenge · Think First</div>
      <div className="predict-question">{challenge.question}</div>

      <div className="predict-options" role="radiogroup">
        {challenge.options.map((opt) => {
          const isChosen = selected === opt.id;
          let btnClass = 'predict-option-btn';
          if (hasAnswered) {
            if (opt.id === challenge.correctId) {
              btnClass += ' is-correct';
            } else if (isChosen) {
              btnClass += ' is-wrong';
            }
          }

          return (
            <button
              key={opt.id}
              type="button"
              className={btnClass}
              onClick={() => setSelected(opt.id)}
            >
              <span className="opt-marker">
                {opt.id.replace('opt-', '').toUpperCase()}
              </span>
              <span className="opt-label">{opt.label}</span>
            </button>
          );
        })}
      </div>

      {hasAnswered && (
        <div
          className={`predict-feedback ${
            isCorrect ? 'is-correct' : 'is-wrong'
          }`}
        >
          <div className="feedback-verdict">
            {isCorrect ? 'Correct Intuition!' : 'Not Quite — Keep Looking:'}
          </div>
          <div className="feedback-explanation">{challenge.explanation}</div>
        </div>
      )}
    </div>
  );
}
