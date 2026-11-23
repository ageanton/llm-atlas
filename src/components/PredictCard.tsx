import { useEffect, useState } from 'react';
import type { PredictChallenge } from '../content/bookData';

interface PredictCardProps {
  challenge: PredictChallenge;
}

export function PredictCard({ challenge }: PredictCardProps) {
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    setPicked(null);
  }, [challenge.id]);

  const isCorrect = picked === challenge.correctId;

  return (
    <div className="predict-card">
      <div className="predict-eyebrow">Predict Before You Peek</div>
      <div className="predict-question">{challenge.question}</div>

      <div className="predict-options">
        {challenge.options.map((opt) => {
          const isSelected = picked === opt.id;
          const revealRight = picked !== null && opt.id === challenge.correctId;
          const revealWrong = isSelected && !isCorrect;
          return (
            <button
              key={opt.id}
              type="button"
              className={`predict-opt ${revealRight ? 'is-correct' : ''} ${
                revealWrong ? 'is-wrong' : ''
              }`}
              onClick={() => setPicked(opt.id)}
            >
              {opt.label}
            </button>
          );
        })}
      </div>

      {picked !== null && (
        <div
          className={`predict-feedback ${isCorrect ? 'is-correct' : 'is-wrong'}`}
        >
          <strong>{isCorrect ? 'Right intuition — ' : 'Not quite — '}</strong>
          {challenge.explanation}
        </div>
      )}
    </div>
  );
}
