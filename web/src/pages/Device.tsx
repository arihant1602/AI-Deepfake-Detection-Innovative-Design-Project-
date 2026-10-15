import { useEffect, useState } from "react";
import { api, type DeviceReport } from "../lib/api";
import type { Status, Verdict as V } from "../lib/gates";
import { TopBarControls } from "../components/Header";
import { Verdict } from "../components/Verdict";
import "./Device.css";

const DEFAULT_PAYLOAD = '{\n  "isRooted": false,\n  "isEmulator": false,\n  "isVirtualCamera": false\n}';

const statusOf = (verdict: string): Status =>
  verdict === "PASS" ? "pass" : verdict === "FLAG_FOR_REVIEW" ? "flag" : "block";

const plain = (verdict: string) => verdict.replace(/_/g, " ").toLowerCase();

function toVerdict(r: DeviceReport): V {
  const status = statusOf(r.verdict);
  return {
    status,
    word: { pass: "Clean", flag: "Review", block: "Blocked", wait: "", off: "" }[status],
    title: { pass: "Nothing suspicious on this machine", flag: "Suspicious software is running", block: "The camera would be refused" }[
      status as "pass" | "flag" | "block"
    ],
    line: r.details,
  };
}

export function Device() {
  const [report, setReport] = useState<DeviceReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [check, setCheck] = useState<{ verdict: string; details: string } | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    setLoading(true);
    api
      .device()
      .then(setReport)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  const evaluate = async () => {
    setCheck(null);
    setCheckError(null);
    try {
      setCheck(await api.attestation(payload));
    } catch (e) {
      setCheckError((e as Error).message);
    }
  };

  return (
    <main className="console-grid devpage">
      <TopBarControls>
        {report && <span className="chip">probed in {report.latency_ms.toFixed(1)} ms</span>}
        <button className="btn" onClick={load} disabled={loading}>
          {loading ? "Checking…" : "Check again"}
        </button>
      </TopBarControls>

      <div className="devpage__main">
        <section className="panel">
          <h1 className="label">This machine</h1>
          {report ? (
            <Verdict verdict={toVerdict(report)} />
          ) : (
            <p className="mono">{error ?? "probing…"}</p>
          )}
          {error && report && (
            <p className="error-text" role="alert">
              {error}
            </p>
          )}
        </section>

        {report && (
          <>
            <section className="panel">
              <div className="panel__head">
                <h2 className="label">Video devices</h2>
                <span className="panel__aside">{report.devices.length} nodes</span>
              </div>
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Node</th>
                      <th>Name</th>
                      <th>Kind</th>
                      <th>Driver</th>
                      <th>Bus</th>
                      <th>Identity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.devices.map((d) => (
                      <tr key={d.node} className={d.is_virtual ? "is-bad" : ""}>
                        <td>{d.node}</td>
                        <td className="text">{d.name}</td>
                        <td>
                          {d.kind}
                          {d.why && <span className="table__why">{d.why}</span>}
                        </td>
                        <td>{d.driver}</td>
                        <td>{d.bus}</td>
                        <td>{d.camera_id ?? "none"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="panel">
              <div className="panel__head">
                <h2 className="label">Environment</h2>
                <span className="panel__aside">{report.diagnostics.length} checks</span>
              </div>
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
                    {report.diagnostics.map((d) => (
                      <tr key={d.name} className={/fail|block/i.test(d.status) ? "is-bad" : ""}>
                        <td className="text">{d.name}</td>
                        <td>{d.status}</td>
                        <td>{d.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>

      <aside className="devpage__side">
        <section className="panel">
          <div className="panel__head">
            <h2 className="label">Enrolled fingerprints</h2>
            <span className="panel__aside">{report?.fingerprints.length ?? 0}</span>
          </div>
          {report?.fingerprints.length ? (
            <ul className="prints">
              {report.fingerprints.map((f) => (
                <li key={f} className="mono">
                  {f}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mono prints__empty">none on this machine</p>
          )}
        </section>

        <section className="panel">
          <h2 className="label">Check an Android report</h2>
          <label className="field">
            <span className="hint">Paste the JSON the Android SDK produced.</span>
            <textarea className="textarea" value={payload} onChange={(e) => setPayload(e.target.value)} spellCheck={false} />
          </label>
          <button className="btn btn--primary" onClick={evaluate}>
            Check report
          </button>
          {check && (
            <p className={`attest attest--${statusOf(check.verdict)}`} role="status">
              <span className={`chip chip--${statusOf(check.verdict)}`}>{plain(check.verdict)}</span>
              <span>{check.details}</span>
            </p>
          )}
          {checkError && (
            <p className="error-text" role="alert">
              {checkError}
            </p>
          )}
        </section>
      </aside>
    </main>
  );
}
