import type { GateState } from "../lib/gates";
import "./Pipeline.css";

const STATUS_WORD = { pass: "pass", block: "fail", flag: "check", wait: "…", off: "—" };

/**
 * The four gates in pipeline order, joined by a line. Below the first failed gate the line turns
 * coral: in the gated architecture a session stops there. `gatedLabel` marks the later gates
 * (live mode keeps measuring them, so their readings are still shown, dimmed).
 */
export function Pipeline({ gates, gatedLabel }: { gates: GateState[]; gatedLabel?: string }) {
  const firstFail = gates.findIndex((g) => g.status === "block");
  return (
    <ol className="pipeline">
      {gates.map((g, i) => {
        const gated = firstFail >= 0 && i > firstFail;
        return (
          <li
            key={g.gate}
            className={`stage stage--${g.status} ${gated ? "is-gated" : ""} ${firstFail >= 0 && i >= firstFail ? "is-after-fail" : ""}`}
          >
            <span className="stage__node" aria-hidden="true" />
            <div className="stage__row">
              <span className="stage__id">{g.gate}</span>
              <span className="stage__name">{g.name}</span>
              <span className={`chip chip--${g.status}`}>{STATUS_WORD[g.status]}</span>
            </div>
            <p className="stage__label">{g.label}</p>
            {(g.value || g.ms !== undefined || (gated && gatedLabel)) && (
              <p className="stage__meta">
                {[gated ? gatedLabel : null, g.value, g.ms !== undefined ? `${Math.round(g.ms).toLocaleString()} ms` : null]
                  .filter(Boolean)
                  .join("  ·  ")}
              </p>
            )}
          </li>
        );
      })}
    </ol>
  );
}
