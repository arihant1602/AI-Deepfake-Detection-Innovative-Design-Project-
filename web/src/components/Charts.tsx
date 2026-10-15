import { useLayoutEffect, useRef, useState } from "react";
import type { ChallengeAttempt } from "../lib/api";
import "./Charts.css";

const HISTORY_S = 60;
const PAD = { l: 34, r: 8, t: 10, b: 22 };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export interface TimePoint {
  t: number;
  v: number;
  bad?: boolean;    // failed the check: drawn as a coral dot
  muted?: boolean;  // no decision (e.g. no face): drawn as a hollow dot
}

interface TimelineProps {
  points: TimePoint[];
  threshold: number;
  yMax: number;
  label: string;
  format?: (v: number) => string;
  now: number;
  height?: number;
}

/** A value over the last minute, with the pass/fail threshold as a dashed rule. */
export function Timeline({ points, threshold, yMax, label, format = (v) => v.toFixed(1), now, height = 150 }: TimelineProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const recent = points.filter((p) => now - p.t <= HISTORY_S);
  const w = Math.max(0, width - PAD.l - PAD.r);
  const h = height - PAD.t - PAD.b;
  const x = (t: number) => PAD.l + w * (1 - (now - t) / HISTORY_S);
  const y = (v: number) => PAD.t + h * (1 - Math.min(v, yMax) / yMax);
  const path = recent.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
  const last = recent.at(-1);

  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={label}>
          {[0, yMax / 2, yMax].map((v) => (
            <g key={v}>
              <line className="chart__grid" x1={PAD.l} x2={PAD.l + w} y1={y(v)} y2={y(v)} />
              <text className="chart__tick" x={PAD.l - 6} y={y(v) + 3} textAnchor="end">
                {format(v)}
              </text>
            </g>
          ))}
          {[60, 45, 30, 15, 0].map((s) => (
            <text key={s} className="chart__tick" x={x(now - s)} y={height - 6} textAnchor={s === 0 ? "end" : "middle"}>
              {s === 0 ? "now" : `−${s}s`}
            </text>
          ))}
          <line className="chart__threshold" x1={PAD.l} x2={PAD.l + w} y1={y(threshold)} y2={y(threshold)} />
          <path className="chart__line" d={path} />
          {recent.map((p) =>
            p.bad || p.muted ? (
              <circle key={p.t} className={p.bad ? "chart__bad" : "chart__muted"} cx={x(p.t)} cy={y(p.v)} r={p.bad ? 4 : 3} />
            ) : null,
          )}
          {last && <circle className="chart__head" cx={x(last.t)} cy={y(last.v)} r={4.5} />}
        </svg>
      )}
      {recent.length === 0 && <p className="chart__empty">
          <span>waiting for the first reading</span>
        </p>}
    </div>
  );
}

/** The last liveness challenge: the brightness pattern we sent, over what the video did. */
export function FlashChart({ attempt, height = 130 }: { attempt: ChallengeAttempt; height?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const n = attempt.luma.length;
  const w = Math.max(0, width - PAD.r - 4);
  const x = (i: number) => 4 + (w * i) / Math.max(1, n - 1);
  const cmdH = 26;
  const lumaTop = cmdH + 14;
  const lumaH = height - lumaTop - 6;
  const lo = Math.min(...attempt.luma);
  const hi = Math.max(...attempt.luma);
  const yl = (v: number) => lumaTop + lumaH * (1 - (v - lo) / Math.max(1e-6, hi - lo));
  const cmd = attempt.command
    .map((c, i) => {
      const yy = c > 0 ? 2 : cmdH;
      return `${i ? `H${x(i).toFixed(1)}V` : `M${x(0)},`}${yy}`;
    })
    .join("");
  const luma = attempt.luma.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${yl(v).toFixed(1)}`).join("");

  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Brightness pattern sent to the camera, and the brightness of the video frames">
          <path className="chart__command" d={`${cmd}H${x(n - 1)}`} />
          <line className="chart__grid" x1={0} x2={width} y1={cmdH + 7} y2={cmdH + 7} />
          <path className="chart__line" d={luma} />
        </svg>
      )}
    </div>
  );
}
