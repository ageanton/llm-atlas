import { LEVEL_SPECS } from '../content/atlasContent';
import type { Register, ZoomLevel } from '../state/urlState';

interface ExplanationPanelProps {
  level: ZoomLevel;
  register: Register;
  mobileOpen: boolean;
  onToggleMobile: () => void;
  onJumpLevel: (lv: ZoomLevel) => void;
}

export function ExplanationPanel({
  level,
  register,
  mobileOpen,
  onToggleMobile,
  onJumpLevel,
}: ExplanationPanelProps) {
  const spec = LEVEL_SPECS[level];
  const bodyText = spec.copy[register];

  return (
    <aside
      className={`explanation-panel ${mobileOpen ? 'is-open' : ''}`}
      aria-live="polite"
      aria-label="Level explanation"
    >
      <button
        type="button"
        className="sheet-handle"
        onClick={onToggleMobile}
        aria-expanded={mobileOpen}
      >
        <span>
          {spec.shortName}: {spec.title}
        </span>
        <span className="sheet-icon">{mobileOpen ? '▼' : '▲'}</span>
      </button>

      <div className="panel-body">
        <div className="panel-badge-row">
          <span className={`honesty-badge ${spec.honesty.kind}`}>
            {spec.honesty.kind === 'literal'
              ? '● Literal Math'
              : '◌ Pedagogical Abstraction'}
          </span>
          <span className="register-label mono">{register} register</span>
        </div>

        <h3 className="panel-title">{spec.title}</h3>

        <div className="panel-prose">
          <p>{bodyText}</p>
        </div>

        {spec.honesty.mathDetail && (
          <div className="panel-math mono">
            <code>{spec.honesty.mathDetail}</code>
          </div>
        )}

        <p className="honesty-note">{spec.honesty.note}</p>

        <div className="panel-pedagogy-links">
          {spec.answersQuestion && (
            <div className="pedagogy-card answers">
              <span className="pedagogy-kicker mono">Answers prior question:</span>
              <p>{spec.answersQuestion}</p>
            </div>
          )}

          {spec.createsQuestion && (
            <div className="pedagogy-card creates">
              <span className="pedagogy-kicker mono">Creates new question:</span>
              <p>{spec.createsQuestion}</p>
              {spec.nextPromptLabel && level < 5 && (
                <button
                  type="button"
                  className="btn-inline"
                  onClick={() => onJumpLevel(((level + 1) as ZoomLevel))}
                >
                  {spec.nextPromptLabel} →
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
