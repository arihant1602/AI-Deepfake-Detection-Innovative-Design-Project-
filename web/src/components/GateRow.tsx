import type { GateState } from "../lib/gates";
import "./GateRow.css";

const STATUS_WORD = { pass: "passed", block: "failed", flag: "needs a look", wait: "in progress", off: "did not run" };

/** The four gates, in the order the pipeline runs them. */
export function GateRow({ gates, live = false }: { gates: GateState[]; live?: boolean }) {
  return (
    <ol className="gates">
      {gates.map((g) => (
        <li key={g.gate} className={`gate gate--${g.status}`}>
          <span className="gate__num">{g.gate}</span>
          <h3 className="gate__name">
            {g.name}
            {live ? "?" : ""}
          </h3>
          <p className="gate__label">
            <span className="gate__dot" />
            <span className="visually-hidden">{STATUS_WORD[g.status]}: </span>
            {g.label}
          </p>
          {(g.value || g.ms !== undefined) && (
            <p className="gate__meta">
              {[g.value, g.ms !== undefined ? `${Math.round(g.ms).toLocaleString()} ms` : null].filter(Boolean).join(", ")}
            </p>
          )}
        </li>
      ))}
    </ol>
  );
}
