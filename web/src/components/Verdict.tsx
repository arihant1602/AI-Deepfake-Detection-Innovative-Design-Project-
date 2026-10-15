import type { Verdict as V } from "../lib/gates";
import "./Verdict.css";

/** Grain-textured colour swatch, one-word verdict, and a plain-language line. */
export function Verdict({ verdict, compact = false }: { verdict: V; compact?: boolean }) {
  return (
    <div className={`verdict verdict--${verdict.status} ${compact ? "verdict--compact" : ""}`} aria-live="polite">
      <span className="verdict__swatch" aria-hidden="true" />
      <div className="verdict__body">
        <p className="verdict__word">{verdict.word}</p>
        <p className="verdict__title">{verdict.title}</p>
        {verdict.line && <p className="verdict__line">{verdict.line}</p>}
      </div>
    </div>
  );
}
