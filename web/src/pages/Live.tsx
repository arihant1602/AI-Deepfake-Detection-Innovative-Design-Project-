import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type FlashPlan, type LiveSession, type Meta, type Snapshot } from "../lib/api";
import { liveGates, liveVerdict, type GateState } from "../lib/gates";
import { TopBarControls } from "../components/Header";
import { Verdict } from "../components/Verdict";
import { Pipeline } from "../components/Pipeline";
import { FlashChart, Timeline } from "../components/Charts";
import { EventLog } from "../components/EventLog";
import "./Live.css";

const POLL_MS = 1000;
const CHART_H = 112;

// What each check does, shown before the camera starts.
const IDLE_GATES: GateState[] = [
  { gate: "1a", name: "Real camera connected", status: "off", label: "A physical camera, not OBS, an emulator or a hooked driver." },
  { gate: "1b", name: "Camera is live", status: "off", label: "We change brightness, colour and contrast at random; only a live sensor follows." },
  { gate: "2", name: "Has real camera grain", status: "off", label: "Fresh grain in every frame, from the same sensor that passed the live test." },
  { gate: "3", name: "Face looks real", status: "off", label: "GenD looks for deepfake artefacts in the face." },
];

/** Full-screen colour sequence for the screen-flash check; `offset` maps client to server time. */
function ScreenFlashOverlay({ plan, offset, onDone }: { plan: FlashPlan; offset: number; onDone: () => void }) {
  const [colour, setColour] = useState<string | null>(null);
  useEffect(() => {
    let raf = 0;
    const frame = () => {
      const k = Math.floor((Date.now() / 1000 + offset - plan.t0) / plan.slot_s);
      if (k >= plan.seq.length) return onDone();
      setColour(k < 0 ? null : plan.seq[k] > 0 ? plan.colours[0] : plan.colours[1]);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [plan, offset, onDone]);
  return colour ? <div className="screen-flash" style={{ background: colour }} aria-hidden="true" /> : null;
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function Live() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [node, setNode] = useState("");
  const [session, setSession] = useState<LiveSession | null>(null);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<{ plan: FlashPlan; offset: number } | null>(null);
  const tokenRef = useRef<string | null>(null);

  const running = Boolean(snap?.running);

  useEffect(() => {
    api
      .meta()
      .then((m) => {
        setMeta(m);
        setNode((n) => n || m.devices.find((d) => !d.is_virtual)?.node || m.devices[0]?.node || "");
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  // Poll while a session exists. Each poll is also the heartbeat that keeps the camera open.
  useEffect(() => {
    if (!session) return;
    let alive = true;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const s = await api.state(session.token);
        if (!alive) return;
        setSnap(s);
        if (s.running) timer = window.setTimeout(tick, POLL_MS);
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [session]);

  // Leaving the page or closing the tab releases the camera.
  useEffect(() => {
    const release = () => {
      if (tokenRef.current) api.stopBeacon(tokenRef.current);
    };
    window.addEventListener("pagehide", release);
    return () => {
      window.removeEventListener("pagehide", release);
      release();
    };
  }, []);

  const start = useCallback(async () => {
    if (!node) return;
    setBusy(true);
    setError(null);
    setSnap(null);
    try {
      const s = await api.start(node);
      tokenRef.current = s.token;
      setSession(s);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [node]);

  const stop = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    try {
      await api.stop(session.token);
      setSnap(await api.state(session.token));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      tokenRef.current = null;
      setBusy(false);
    }
  }, [session]);

  const screenFlash = async () => {
    if (!session) return;
    try {
      const r = await api.flash(session.token);
      setFlash({ plan: r.plan, offset: r.now - Date.now() / 1000 });
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const flashDone = useCallback(() => setFlash(null), []);

  const attack = async (mode: string) => {
    if (!session) return;
    try {
      await api.attack(session.token, mode);
      setSnap(await api.state(session.token));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const gates = snap ? liveGates(snap) : null;
  const verdict = liveVerdict(gates, snap);
  const thr = snap?.deepfake_threshold ?? 0.5;
  const noiseT = meta?.noise_threshold ?? 0.6;
  const noiseMax = Math.max(2, ...(snap?.noise.map((n) => (n.sigma ?? 0) * 1.2) ?? []));
  const lastFlash = snap?.challenges[0];
  const attacked = Boolean(snap?.injection && snap.injection !== "none");

  return (
    <main className="console">
      {flash && <ScreenFlashOverlay plan={flash.plan} offset={flash.offset} onDone={flashDone} />}
      <TopBarControls>
        {meta && meta.devices.length > 0 ? (
          <label className="console__camera">
            <span className="visually-hidden">Camera</span>
            <select className="select" value={node} disabled={running || busy} onChange={(e) => setNode(e.target.value)}>
              {meta.devices.map((d) => (
                <option key={d.node} value={d.node}>
                  {d.node} · {d.name}
                  {d.is_virtual ? " (software camera)" : ""}
                </option>
              ))}
            </select>
          </label>
        ) : (
          meta && <span className="error-text">No camera found</span>
        )}
        {running && snap ? (
          <span className="chip chip--live console__status">
            LIVE {snap.fps.toFixed(0)} fps · {clock(snap.now - snap.started)}
          </span>
        ) : (
          <span className="chip console__status">{snap?.stop_reason ? "stopped" : "idle"}</span>
        )}
        {running ? (
          <button className="btn btn--alert" onClick={stop} disabled={busy}>
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <rect x="2" y="2" width="10" height="10" fill="currentColor" />
            </svg>
            Stop
          </button>
        ) : (
          <button className="btn btn--primary" onClick={start} disabled={busy || !node}>
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <path d="M3 1.5v11l9-5.5z" fill="currentColor" />
            </svg>
            {busy ? "Opening…" : "Start"}
          </button>
        )}
      </TopBarControls>

      <div className="console-grid console__grid">
        {/* ---------------------------------------------------------------- video */}
        <section className="panel console__video" aria-label="Camera">
          <div className="panel__head">
            <h2 className="label">Camera feed</h2>
            <span className="panel__aside">{snap?.camera_id ?? session?.node ?? node}</span>
          </div>
          <div className={`feed ${attacked ? "feed--attacked" : ""} ${snap?.in_challenge ? "feed--flash" : ""}`}>
            {running && session ? (
              <img
                key={session.token}
                src={api.streamUrl(session.token)}
                width={session.width}
                height={session.height}
                alt="Live camera, with the detected face outlined"
              />
            ) : (
              <div className="feed__off">
                <span className="mono">{snap?.stop_reason ? `stopped: ${snap.stop_reason}` : "camera off"}</span>
                {!snap && <span className="hint">Press Start. The camera stays on only while this page is open.</span>}
              </div>
            )}
          </div>
          {error && (
            <p className="error-text" role="alert">
              {error}
            </p>
          )}
        </section>

        {/* ---------------------------------------------------------------- verdict + pipeline */}
        <aside className="panel console__side" aria-label="Verdict">
          <h2 className="label">Verdict</h2>
          <Verdict verdict={verdict} />
          <div className="console__rule" />
          <div className="panel__head">
            <h2 className="label">Pipeline</h2>
            <span className="panel__aside">runs top to bottom</span>
          </div>
          <Pipeline gates={gates ?? IDLE_GATES} gatedLabel="gated off" />
        </aside>

        {/* ---------------------------------------------------------------- attack dock */}
        <section className="panel console__dock" aria-label="Test controls">
          <div className="dock__group">
            <h2 className="label">Simulate an attack</h2>
            <div className="segmented segmented--attack" role="group" aria-label="Simulated attack">
              {Object.entries(meta?.injections ?? { none: "Off" }).map(([key, label]) => (
                <button
                  key={key}
                  data-key={key}
                  aria-pressed={(snap?.injection ?? "none") === key}
                  disabled={!running}
                  onClick={() => attack(key)}
                  title={label}
                >
                  {{ none: "off", replay: "replay", ai: "AI video", photo: "animated photo" }[key] ?? label}
                </button>
              ))}
            </div>
            <p className="hint">Swaps frames after the camera driver, like a hooked capture call.</p>
          </div>
          <div className="dock__group dock__group--end">
            <h2 className="label">Live test</h2>
            <button
              className="btn"
              onClick={() => session && api.challenge(session.token)}
              disabled={!running || snap?.in_challenge || Boolean(flash)}
              title="Changes brightness and two random camera controls"
            >
              Camera
            </button>
            <button
              className="btn"
              onClick={screenFlash}
              disabled={!running || snap?.in_challenge || Boolean(flash || snap?.flash_plan)}
              title="Flashes the screen in two colours for 5 seconds and looks for the light on your face"
            >
              Screen flash
            </button>
            <p className="mono dock__next">
              {running && snap ? (snap.in_challenge ? "running" : `next camera test in ${Math.round(snap.next_challenge_in)} s`) : "—"}
            </p>
          </div>
        </section>

        {/* ---------------------------------------------------------------- signal strip */}
        <figure className="panel console__signal">
          <figcaption className="panel__head">
            <span className="label">Camera grain</span>
            <span className="panel__aside">
              {snap?.noise.at(-1)?.sigma != null ? `σ ${snap.noise.at(-1)!.sigma!.toFixed(2)}` : "—"} / min {noiseT}
            </span>
          </figcaption>
          <Timeline
            now={snap?.now ?? 0}
            height={CHART_H}
            label="Live sensor noise level over the last minute; real cameras stay above the dashed line"
            threshold={noiseT}
            yMax={noiseMax}
            points={(snap?.noise ?? [])
              .filter((n) => n.sigma !== null)
              .map((n) => ({ t: n.t, v: n.sigma as number, bad: n.verdict === "ABSENT" }))}
          />
        </figure>

        <figure className="panel console__signal">
          <figcaption className="panel__head">
            <span className="label">AI-face score</span>
            <span className="panel__aside">
              {snap?.deepfake_smoothed != null ? `${Math.round(snap.deepfake_smoothed * 100)}%` : "—"} / max{" "}
              {Math.round(thr * 100)}%
            </span>
          </figcaption>
          <Timeline
            now={snap?.now ?? 0}
            height={CHART_H}
            label="Probability the face is AI-generated over the last minute; above the dashed line looks fake"
            threshold={thr}
            yMax={1}
            format={(v) => `${Math.round(v * 100)}%`}
            points={(snap?.deepfake ?? [])
              .filter((d) => d.p !== null || d.abstained)
              .map((d) => ({ t: d.t, v: d.p ?? 0, bad: d.p !== null && d.p >= thr, muted: d.abstained }))}
          />
        </figure>

        <figure className="panel console__signal">
          <figcaption className="panel__head">
            <span className="label">Last camera test</span>
            <span className="panel__aside" title={lastFlash?.control?.replaceAll("_", " ")}>
              {lastFlash?.corr !== undefined ? `r ${lastFlash.corr.toFixed(2)} / min 0.85` : "—"}
            </span>
          </figcaption>
          {lastFlash?.attempt ? (
            <FlashChart attempt={lastFlash.attempt} height={CHART_H} />
          ) : (
            <p className="signal__empty mono">dashed: brightness sent · line: brightness seen</p>
          )}
        </figure>

        <section className="panel console__log" aria-label="Event log">
          <div className="panel__head">
            <h2 className="label">Event log</h2>
            <span className="panel__aside">{snap ? `${snap.events.length} events` : ""}</span>
          </div>
          <EventLog events={snap?.events ?? []} />
        </section>
      </div>
    </main>
  );
}
