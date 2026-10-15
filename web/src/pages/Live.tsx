import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type LiveSession, type Meta, type Snapshot } from "../lib/api";
import { liveGates, liveVerdict, type GateState } from "../lib/gates";
import { StackWord } from "../components/StackWord";
import { GrainPanel } from "../components/GrainPanel";
import { GateRow } from "../components/GateRow";
import { FlashChart, Timeline } from "../components/Charts";
import { EventLog } from "../components/EventLog";
import "./Live.css";

const POLL_MS = 1000;

// What each check does, shown before the camera starts.
const IDLE_GATES: GateState[] = [
  { gate: "1a", name: "Real camera connected", status: "off", label: "A physical camera, not OBS, an emulator or a hooked driver" },
  { gate: "1b", name: "Camera is live", status: "off", label: "Every 10 s we flicker the camera's brightness. Only a live sensor follows" },
  { gate: "2", name: "Has real camera grain", status: "off", label: "Real sensors add fresh grain to every frame. Generated video is too clean" },
  { gate: "3", name: "Face looks real", status: "off", label: "An AI detector (GenD) looks for deepfake artefacts in the face" },
];

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 5.5v13l10.5-6.5z" fill="currentColor" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="6.5" y="6.5" width="11" height="11" fill="currentColor" />
    </svg>
  );
}

export function Live() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [node, setNode] = useState("");
  const [session, setSession] = useState<LiveSession | null>(null);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
  const device = meta?.devices.find((d) => d.node === node);
  const latestAttempt = snap?.challenges[0]?.attempt;
  const thr = snap?.deepfake_threshold ?? 0.5;
  const noiseT = meta?.noise_threshold ?? 0.6;
  const noiseMax = Math.max(2, ...(snap?.noise.map((n) => (n.sigma ?? 0) * 1.2) ?? []));

  return (
    <main className="page live">
      <section className="live-hero" aria-labelledby="verdict-title">
        <div className="live-hero__copy">
          <p className="live-hero__intro">
            A live camera leaves fresh grain in every frame and answers when you flicker its light. A fake can't do
            either.
          </p>
          <StackWord word={verdict.word} status={verdict.status} />
          <div className="live-hero__verdict" aria-live="polite">
            <h2 id="verdict-title" className="live-hero__title">
              {verdict.title}
            </h2>
            {verdict.line && <p className="live-hero__line">{verdict.line}</p>}
          </div>

          <div className="live-hero__controls">
            {running ? (
              <button className="orb orb--stop" onClick={stop} disabled={busy}>
                <span className="orb__ring">
                  <StopIcon />
                </span>
                <span className="orb__label">
                  Stop checking
                  <span className="orb__sub">releases {snap?.node}</span>
                </span>
              </button>
            ) : (
              <button className="orb" onClick={start} disabled={busy || !node}>
                <span className="orb__ring">
                  <PlayIcon />
                </span>
                <span className="orb__label">
                  {busy ? "Opening camera" : "Start checking"}
                  <span className="orb__sub">{device ? device.name : "no camera found"}</span>
                </span>
              </button>
            )}
            {meta && meta.devices.length > 1 && (
              <label className="field live-hero__camera">
                <span className="visually-hidden">Camera</span>
                <select className="select" value={node} disabled={running || busy} onChange={(e) => setNode(e.target.value)}>
                  {meta.devices.map((d) => (
                    <option key={d.node} value={d.node}>
                      {d.name} ({d.node}){d.is_virtual ? ", software camera" : ""}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          {meta && meta.devices.length === 0 && (
            <p className="error-text">No video-capture device found. Plug in a webcam and reload.</p>
          )}
          {error && <p className="error-text" role="alert">{error}</p>}
        </div>

        <GrainPanel className="live-hero__panel">
          <p className="live-hero__side">
            {running ? `${snap?.fps.toFixed(0)} fps` : "Camera off"}
            <br />
            {snap?.camera_id ?? device?.node ?? ""}
          </p>
          <p className="live-hero__foot mono">
            {running && snap
              ? snap.in_challenge
                ? "Flash test running"
                : `Next flash test in ${Math.round(snap.next_challenge_in)} s`
              : "Video stays on this machine"}
          </p>
        </GrainPanel>

        <figure className={`live-hero__video ${snap?.injection && snap.injection !== "none" ? "is-attacked" : ""}`}>
          {running && session ? (
            <img
              key={session.token}
              src={api.streamUrl(session.token)}
              width={session.width}
              height={session.height}
              alt="Live camera, with the detected face outlined"
            />
          ) : (
            <div className="live-hero__off" style={{ aspectRatio: "4 / 3" }}>
              <span>{snap?.stop_reason ? `Stopped: ${snap.stop_reason}` : "Camera off"}</span>
            </div>
          )}
        </figure>
      </section>

      {running && meta && (
        <section className="live-tools" aria-label="Test controls">
          <div className="live-tools__group">
            <h2 className="live-tools__title">Simulate an attack</h2>
            <p className="field__hint">
              Swaps the frames after the camera driver, like a hooked capture call. The real camera still gets the
              flash test.
            </p>
            <div className="choices" role="group" aria-label="Simulated attack">
              {Object.entries(meta.injections).map(([key, label]) => (
                <button
                  key={key}
                  className={`choice ${key === "none" ? "" : "choice--attack"}`}
                  aria-pressed={snap?.injection === key}
                  onClick={() => attack(key)}
                >
                  {key === "none" ? "No attack" : label}
                </button>
              ))}
            </div>
          </div>
          <div className="live-tools__group live-tools__group--end">
            <button className="btn" onClick={() => session && api.challenge(session.token)} disabled={snap?.in_challenge}>
              Run the flash test now
            </button>
            <p className="field__hint">The picture flickers for a moment. That is the test.</p>
          </div>
        </section>
      )}

      <section className="live-section" aria-labelledby="checks-title">
        <h2 id="checks-title" className="section-title">
          {snap ? "The four checks, right now" : "Four checks, run continuously"}
        </h2>
        <GateRow gates={gates ?? IDLE_GATES} live={Boolean(gates)} />
      </section>

      {snap && (
        <section className="live-section live-readings" aria-labelledby="readings-title">
          <h2 id="readings-title" className="section-title">
            The last minute
          </h2>
          <div className="readings">
            <figure className="reading">
              <figcaption>
                <span className="reading__name">Camera grain</span>
                <span className="reading__how">Real cameras stay above the dashed line. Coral dots failed.</span>
              </figcaption>
              <Timeline
                now={snap.now}
                label="Live sensor noise level over the last minute"
                threshold={noiseT}
                yMax={noiseMax}
                points={snap.noise
                  .filter((n) => n.sigma !== null)
                  .map((n) => ({ t: n.t, v: n.sigma as number, bad: n.verdict === "ABSENT" }))}
              />
            </figure>
            <figure className="reading">
              <figcaption>
                <span className="reading__name">AI-face score</span>
                <span className="reading__how">Above the dashed line looks fake. Hollow dots: no face in view.</span>
              </figcaption>
              <Timeline
                now={snap.now}
                label="Probability the face is AI-generated, over the last minute"
                threshold={thr}
                yMax={1}
                format={(v) => `${Math.round(v * 100)}%`}
                points={snap.deepfake
                  .filter((d) => d.p !== null || d.abstained)
                  .map((d) => ({ t: d.t, v: d.p ?? 0, bad: d.p !== null && d.p >= thr, muted: d.abstained }))}
              />
            </figure>
            <figure className="reading">
              <figcaption>
                <span className="reading__name">Last flash test</span>
                <span className="reading__how">Dashed: the brightness we sent. Blue: what the video did.</span>
              </figcaption>
              {latestAttempt ? (
                <FlashChart attempt={latestAttempt} />
              ) : (
                <p className="reading__wait">The first test runs a second after Start.</p>
              )}
              {snap.challenges[0]?.corr !== undefined && (
                <p className="mono reading__value">r = {snap.challenges[0].corr.toFixed(2)}</p>
              )}
            </figure>
          </div>

          <div className="live-log">
            <h3 className="live-log__title">What happened</h3>
            <EventLog events={snap.events} />
          </div>
        </section>
      )}
    </main>
  );
}
