import { useEffect, useMemo, useState, type FormEvent } from "react";
import { api, type AnalysisResult, type GateDetails, type Meta } from "../lib/api";
import { fileGates, fileVerdict, type GateState } from "../lib/gates";
import { Verdict } from "../components/Verdict";
import { Pipeline } from "../components/Pipeline";
import "./Analyze.css";

const FRAME_OPTIONS = [45, 60, 90, 150];
const SAMPLE_NAMES: Record<string, string> = { "real_sim.mp4": "simulated real", "fake_sim.mp4": "simulated fake" };
const TABS = ["Gate 1", "Gate 2", "Gate 3", "Frames"] as const;
type Tab = (typeof TABS)[number];

const IDLE_GATES: GateState[] = [
  { gate: "1a", name: "Real camera", status: "off", label: "Not checked for files: there is no device to attest." },
  { gate: "1b", name: "Live sensor", status: "off", label: "Not possible for files: nothing to flash." },
  { gate: "2", name: "Sensor noise", status: "off", label: "Decides only with the claimed camera's fingerprint." },
  { gate: "3", name: "Face", status: "off", label: "GenD deepfake model on sampled face crops." },
];

function GateTab({ tab, result }: { tab: Tab; result: AnalysisResult }) {
  const d = result.details ?? {};
  const hw: GateDetails = result.gate === 1 ? d : d.attestation ?? result.attestation ?? {};
  const prnu: GateDetails | undefined = d.prnu ?? result.prnu ?? (result.gate === 2 ? d : undefined);
  const face: GateDetails | undefined = d.temporal ?? (result.gate === 3 ? d : undefined);

  if (tab === "Gate 1")
    return (
      <div className="tab-body">
        <p className="explain">{hw.details}</p>
        {hw.diagnostics_summary && hw.diagnostics_summary.length > 0 && (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Check</th>
                  <th>Result</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {hw.diagnostics_summary.map((r) => (
                  <tr key={r.name}>
                    <td className="text">{r.name}</td>
                    <td>{r.status}</td>
                    <td>{r.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {hw.is_file_upload && (
          <dl className="kv">
            <dt>container encoder</dt>
            <dd>{hw.encoder}</dd>
          </dl>
        )}
      </div>
    );
  if (tab === "Gate 2")
    return prnu ? (
      <div className="tab-body">
        <p className="explain">{prnu.explanation}</p>
        <dl className="kv">
          <dt>mode</dt>
          <dd>{prnu.mode}</dd>
          <dt>decision</dt>
          <dd>{prnu.enforced ? "enforced" : "advisory"}</dd>
          <dt>frames</dt>
          <dd>{prnu.frames_analyzed ?? "?"}</dd>
          <dt>time</dt>
          <dd>{Math.round(prnu.latency_ms ?? 0)} ms</dd>
        </dl>
      </div>
    ) : (
      <p className="tab-empty mono">Gate 2 did not run: an earlier gate stopped the session.</p>
    );
  if (tab === "Gate 3")
    return face ? (
      <div className="tab-body">
        <p className="explain">{face.explanation}</p>
        <dl className="kv">
          <dt>model</dt>
          <dd>{face.model ?? "GenD"}</dd>
          <dt>AI score</dt>
          <dd>{face.score !== undefined ? face.score.toFixed(2) : "—"}</dd>
          <dt>flagged at</dt>
          <dd>{face.threshold ?? 0.5}</dd>
          <dt>faces found</dt>
          <dd>
            {face.frames_with_face} of {face.frames_analyzed} sampled frames
          </dd>
        </dl>
      </div>
    ) : (
      <p className="tab-empty mono">Gate 3 did not run: an earlier gate stopped the session.</p>
    );
  return result.thumbs.length ? (
    <div className="thumbs">
      {result.thumbs.map((t) => (
        <figure key={t.index}>
          <img src={t.src} alt={`Analysed frame ${t.index}`} />
          <figcaption className="mono">frame {t.index}</figcaption>
        </figure>
      ))}
    </div>
  ) : (
    <p className="tab-empty mono">No frames kept: the session stopped before frame analysis.</p>
  );
}

export function Analyze() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [source, setSource] = useState<"upload" | "sample">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState("real_sim.mp4");
  const [reference, setReference] = useState<File | null>(null);
  const [frames, setFrames] = useState(60);
  const [attestation, setAttestation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [tab, setTab] = useState<Tab>("Gate 1");

  useEffect(() => {
    api.meta().then(setMeta).catch((e: Error) => setError(e.message));
  }, []);

  const preview = useMemo(() => {
    if (source === "upload") return file ? URL.createObjectURL(file) : null;
    return meta?.samples.includes(sample) ? api.sampleUrl(sample) : null;
  }, [source, file, sample, meta]);

  useEffect(
    () => () => {
      if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const form = new FormData();
    if (source === "upload" && file) form.append("video", file);
    else if (source === "sample") form.append("sample", sample);
    if (reference) form.append("reference", reference);
    form.append("max_frames", String(frames));
    form.append("attestation", attestation);
    setBusy(true);
    setError(null);
    try {
      const r = await api.analyze(form);
      setResult(r);
      setTab(r.gate ? (`Gate ${r.gate}` as Tab) : "Gate 1");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const verdict = result ? fileVerdict(result) : null;
  const canRun = !busy && (source === "upload" ? Boolean(file) : Boolean(meta?.samples.includes(sample)));

  return (
    <main className="console-grid filepage">
      <form className="panel filepage__side" onSubmit={submit}>
        <div className="field">
          <span className="label">Source</span>
          <div className="segmented" role="group" aria-label="Video source">
            <button type="button" aria-pressed={source === "upload"} onClick={() => setSource("upload")}>
              upload
            </button>
            <button type="button" aria-pressed={source === "sample"} onClick={() => setSource("sample")}>
              sample clip
            </button>
          </div>
        </div>

        {source === "upload" ? (
          <label className="drop">
            <input
              type="file"
              accept=".mp4,.mov,.avi,.mkv,.webm,video/*"
              className="visually-hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <span className="drop__title">{file ? file.name : "Choose a video file"}</span>
            <span className="mono drop__hint">{file ? `${(file.size / 1e6).toFixed(1)} MB` : "mp4 · mov · avi · mkv · webm"}</span>
          </label>
        ) : (
          <div className="segmented" role="group" aria-label="Sample clip">
            {(meta?.samples ?? []).map((s) => (
              <button type="button" key={s} aria-pressed={sample === s} onClick={() => setSample(s)}>
                {SAMPLE_NAMES[s] ?? s}
              </button>
            ))}
            {meta && meta.samples.length === 0 && (
              <p className="hint">No sample clips. Run python generate_test_videos.py.</p>
            )}
          </div>
        )}

        <div className="filepage__preview">
          {preview ? (
            <video key={preview} src={preview} controls muted playsInline />
          ) : (
            <span className="mono">no video selected</span>
          )}
        </div>

        <label className="field">
          <span className="label">Claimed camera's fingerprint (.npy)</span>
          <input type="file" accept=".npy" className="file-input" onChange={(e) => setReference(e.target.files?.[0] ?? null)} />
          <span className="hint">Without it, the grain check is advisory.</span>
        </label>

        <div className="field">
          <span className="label">Frames to analyse</span>
          <div className="segmented" role="group" aria-label="Frames to analyse">
            {FRAME_OPTIONS.map((n) => (
              <button type="button" key={n} aria-pressed={frames === n} onClick={() => setFrames(n)}>
                {n}
              </button>
            ))}
          </div>
        </div>

        <label className="field">
          <span className="label">Simulated device report</span>
          <select className="select" value={attestation} onChange={(e) => setAttestation(e.target.value)}>
            <option value="">none (origin unknown)</option>
            {(meta?.presets ?? []).map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>

        <button type="submit" className="btn btn--primary btn--block" disabled={!canRun}>
          {busy ? "Analysing…" : "Run analysis"}
        </button>
        {error && (
          <p className="error-text" role="alert">
            {error}
          </p>
        )}
      </form>

      <section className="panel filepage__report" aria-label="Report" aria-busy={busy}>
        <div className="panel__head">
          <h1 className="label">Report</h1>
          <span className="panel__aside">
            {result?.timings.total_ms !== undefined ? `${(result.timings.total_ms / 1000).toFixed(1)} s` : ""}
          </span>
        </div>

        {verdict && result ? (
          <Verdict verdict={verdict} />
        ) : (
          <p className="explain">
            {busy
              ? "Reading frames and running the gates."
              : "A file has no live camera behind it, so the device and flash checks can't run. Choose a video and run the analysis."}
          </p>
        )}

        {result?.limitations && result.limitations.length > 0 && (
          <ul className="limits">
            {result.limitations.map((l) => (
              <li key={l} className="mono">
                {l}
              </li>
            ))}
          </ul>
        )}

        <div className="filepage__pipeline">
          <h2 className="label">Pipeline</h2>
          <Pipeline gates={result && result.verdict !== "ERROR" ? fileGates(result) : IDLE_GATES} />
        </div>

        {result && result.verdict !== "ERROR" && (
          <div className="filepage__tabs">
            <div className="tabs" role="tablist" aria-label="Gate details">
              {TABS.map((t) => (
                <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
                  {t}
                </button>
              ))}
            </div>
            <div role="tabpanel" aria-label={tab}>
              <GateTab tab={tab} result={result} />
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
