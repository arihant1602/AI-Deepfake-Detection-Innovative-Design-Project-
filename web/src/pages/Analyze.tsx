import { useEffect, useMemo, useState, type FormEvent } from "react";
import { api, type AnalysisResult, type GateDetails, type Meta } from "../lib/api";
import { fileGates, fileVerdict } from "../lib/gates";
import { StackWord } from "../components/StackWord";
import { GrainPanel } from "../components/GrainPanel";
import { GateRow } from "../components/GateRow";
import "./Analyze.css";

const FRAME_OPTIONS = [45, 60, 90, 150];

function Details({ result }: { result: AnalysisResult }) {
  const d = result.details ?? {};
  const hw: GateDetails = result.gate === 1 ? d : d.attestation ?? result.attestation ?? {};
  const prnu: GateDetails | undefined = d.prnu ?? result.prnu ?? (result.gate === 2 ? d : undefined);
  const face: GateDetails | undefined = d.temporal ?? (result.gate === 3 ? d : undefined);
  return (
    <div className="file-details">
      <details className="fold">
        <summary>Gate 1: device and environment</summary>
        <div className="fold__body">
          <p className="prose">{hw.details}</p>
          {hw.diagnostics_summary && hw.diagnostics_summary.length > 0 && (
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
                    <td>{r.name}</td>
                    <td>{r.status}</td>
                    <td>{r.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {hw.is_file_upload && <p className="mono">Container encoder: {hw.encoder}</p>}
        </div>
      </details>
      {prnu && (
        <details className="fold">
          <summary>Gate 2: sensor fingerprint</summary>
          <div className="fold__body">
            <p className="prose">{prnu.explanation}</p>
            <p className="mono">
              {prnu.mode} mode, {prnu.enforced ? "enforced" : "advisory"}, {prnu.frames_analyzed ?? "?"} frames,{" "}
              {Math.round(prnu.latency_ms ?? 0)} ms
            </p>
          </div>
        </details>
      )}
      {face && (
        <details className="fold">
          <summary>Gate 3: face deepfake model</summary>
          <div className="fold__body">
            <p className="prose">{face.explanation}</p>
            <p className="mono">
              {face.model ?? "GenD"}: AI score {face.score?.toFixed(2)}, flagged at {face.threshold ?? 0.5}. Face found in{" "}
              {face.frames_with_face} of {face.frames_analyzed} sampled frames.
            </p>
          </div>
        </details>
      )}
      {result.thumbs.length > 0 && (
        <details className="fold">
          <summary>Frames analysed</summary>
          <div className="fold__body thumbs">
            {result.thumbs.map((t) => (
              <figure key={t.index}>
                <img src={t.src} alt={`Analysed frame ${t.index}`} />
                <figcaption className="mono">frame {t.index}</figcaption>
              </figure>
            ))}
          </div>
        </details>
      )}
    </div>
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

  useEffect(() => {
    api.meta().then(setMeta).catch((e: Error) => setError(e.message));
  }, []);

  const preview = useMemo(() => {
    if (source === "upload") return file ? URL.createObjectURL(file) : null;
    return meta?.samples.includes(sample) ? api.sampleUrl(sample) : null;
  }, [source, file, sample, meta]);

  useEffect(() => () => {
    if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
  }, [preview]);

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
    setResult(null);
    try {
      setResult(await api.analyze(form));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const verdict = result ? fileVerdict(result) : null;
  const canRun = !busy && (source === "upload" ? Boolean(file) : Boolean(meta?.samples.includes(sample)));

  return (
    <main className="page file">
      <header className="file-head">
        <StackWord word={verdict?.word ?? "File"} status={verdict?.status ?? "off"} size="page" />
        <p className="prose file-head__note">
          A file has no live camera behind it, so the device and flash checks can't run. The grain check can only
          decide when you supply the claimed camera's enrolled fingerprint.
        </p>
      </header>

      <div className="file-grid">
        <form className="file-form" onSubmit={submit}>
          <div className="field">
            <span className="field__label">Video</span>
            <div className="choices" role="group" aria-label="Video source">
              <button type="button" className="choice" aria-pressed={source === "upload"} onClick={() => setSource("upload")}>
                Upload a file
              </button>
              <button type="button" className="choice" aria-pressed={source === "sample"} onClick={() => setSource("sample")}>
                Use a sample clip
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
              <span className="drop__title">{file ? file.name : "Choose a video"}</span>
              <span className="drop__hint mono">
                {file ? `${(file.size / 1e6).toFixed(1)} MB` : "mp4, mov, avi, mkv or webm"}
              </span>
            </label>
          ) : (
            <div className="choices" role="group" aria-label="Sample clip">
              {(meta?.samples ?? []).map((s) => (
                <button type="button" key={s} className="choice" aria-pressed={sample === s} onClick={() => setSample(s)}>
                  {s === "real_sim.mp4" ? "Simulated real camera" : s === "fake_sim.mp4" ? "Simulated fake" : s}
                </button>
              ))}
              {meta && meta.samples.length === 0 && (
                <p className="field__hint">No sample clips yet. Run python generate_test_videos.py to make them.</p>
              )}
            </div>
          )}

          <details className="fold">
            <summary>Options</summary>
            <div className="fold__body">
              <label className="field">
                <span className="field__label">Claimed camera's fingerprint</span>
                <span className="field__hint">An enrolled .npy file. Without it the grain check is advisory.</span>
                <input type="file" accept=".npy" className="file-input mono" onChange={(e) => setReference(e.target.files?.[0] ?? null)} />
              </label>
              <div className="field">
                <span className="field__label">Frames to analyse</span>
                <div className="choices" role="group" aria-label="Frames to analyse">
                  {FRAME_OPTIONS.map((n) => (
                    <button type="button" key={n} className="choice" aria-pressed={frames === n} onClick={() => setFrames(n)}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <label className="field">
                <span className="field__label">Attach a simulated device report</span>
                <select className="select" value={attestation} onChange={(e) => setAttestation(e.target.value)}>
                  <option value="">None, the file's origin is unknown</option>
                  {(meta?.presets ?? []).map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </details>

          <button type="submit" className="btn btn--solid file-form__go" disabled={!canRun}>
            {busy ? "Analysing" : "Analyse video"}
          </button>
          {error && <p className="error-text" role="alert">{error}</p>}
        </form>

        <GrainPanel className="file-preview">
          {preview ? (
            <video key={preview} src={preview} controls muted playsInline className="file-preview__video" />
          ) : (
            <p className="file-preview__empty">The video plays here once you choose one.</p>
          )}
        </GrainPanel>
      </div>

      {busy && <p className="file-busy">Reading frames and running the gates. This takes a few seconds.</p>}

      {result && verdict && (
        <section className="file-result" aria-labelledby="result-title" aria-live="polite">
          <div className="file-result__head">
            <h2 id="result-title" className="section-title">
              {verdict.title}
            </h2>
            <p className="prose">{verdict.line}</p>
            {result.limitations?.map((l) => (
              <p key={l} className="prose file-result__limit">
                {l}
              </p>
            ))}
            {result.timings.total_ms !== undefined && (
              <p className="mono">Total {(result.timings.total_ms / 1000).toFixed(1)} s, mostly reading frames.</p>
            )}
          </div>
          {result.verdict !== "ERROR" && (
            <>
              <GateRow gates={fileGates(result)} />
              <Details result={result} />
            </>
          )}
        </section>
      )}
    </main>
  );
}
