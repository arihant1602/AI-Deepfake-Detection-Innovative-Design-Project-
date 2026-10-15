import type { LiveEvent } from "../lib/api";
import "./EventLog.css";

const TAG = { block: "FAIL", warn: "WARN", ok: " OK ", info: "INFO" };

/** Newest first, terminal-style. */
export function EventLog({ events }: { events: LiveEvent[] }) {
  if (!events.length) return <p className="log__empty">No events yet.</p>;
  return (
    <ol className="log" aria-live="polite">
      {events.slice(0, 30).map((e) => (
        <li key={`${e.t}-${e.text}`} className={`log__row log__row--${e.level}`}>
          <time className="log__time" dateTime={new Date(e.t * 1000).toISOString()}>
            {new Date(e.t * 1000).toLocaleTimeString([], { hour12: false })}
          </time>
          <span className="log__tag">{TAG[e.level]}</span>
          <span className="log__text">{e.text}</span>
        </li>
      ))}
    </ol>
  );
}
