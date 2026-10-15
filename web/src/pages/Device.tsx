import { useEffect, useState } from "react";
import { api, type DeviceReport } from "../lib/api";
import { StackWord } from "../components/StackWord";
import type { Status } from "../lib/gates";
import "./Device.css";

const DEFAULT_PAYLOAD = '{\n  "isRooted": false,\n  "isEmulator": false,\n  "isVirtualCamera": false\n}';

const statusOf = (verdict: string): Status =>
  verdict === "PASS" ? "pass" : verdict === "FLAG_FOR_REVIEW" ? "flag" : "block";

const plain = (verdict: string) => verdict.replace(/_/g, " ").toLowerCase();

export function Device() {
  const [report, setReport] = useState<DeviceReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [check, setCheck] = useState<{ verdict: string; details: string } | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    api.device().then(setReport).catch((e: Error) => setError(e.message));
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

  const status = report ? statusOf(report.verdict) : "wait";
  const word = report ? ({ pass: "Clean", flag: "Check", block: "Risky" } as const)[status as "pass" | "flag" | "block"] : "Device";

  return (
    <main className="page device">
      <header className="device-head">
        <StackWord word={word} status={status} size="page" />
        <div className="device-head__copy">
          <p className="lede">What the device check sees on this machine right now.</p>
          {report && (
            <>
              <p className="prose">{report.details}</p>
              <p className="mono">Checked in {report.latency_ms.toFixed(1)} ms</p>
            </>
          )}
          {error && <p className="error-text" role="alert">{error}</p>}
          <button className="btn" onClick={load}>
            Check again
          </button>
        </div>
      </header>

      {report && (
        <>
          <section className="device-section" aria-labelledby="cams-title">
            <h2 id="cams-title" className="section-title">
              Video devices
            </h2>
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
                      <td className="mono">{d.node}</td>
                      <td>{d.name}</td>
                      <td>
                        {d.kind}
                        {d.why && <span className="table__why">{d.why}</span>}
                      </td>
                      <td className="mono">{d.driver}</td>
                      <td className="mono">{d.bus}</td>
                      <td className="mono">{d.camera_id ?? "none"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="device-section" aria-labelledby="env-title">
            <h2 id="env-title" className="section-title">
              Environment
            </h2>
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
                      <td>{d.name}</td>
                      <td className="mono">{d.status}</td>
                      <td className="mono">{d.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="device-section device-split">
            <div>
              <h2 className="section-title">Enrolled fingerprints</h2>
              {report.fingerprints.length ? (
                <ul className="prints">
                  {report.fingerprints.map((f) => (
                    <li key={f} className="mono">
                      {f}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="prose">None on this machine yet.</p>
              )}
            </div>
            <div className="device-attest">
              <h2 className="section-title">Check an Android report</h2>
              <label className="field">
                <span className="field__hint">Paste the JSON the Android SDK produced.</span>
                <textarea className="textarea" value={payload} onChange={(e) => setPayload(e.target.value)} spellCheck={false} />
              </label>
              <button className="btn btn--solid" onClick={evaluate}>
                Check report
              </button>
              {check && (
                <p className={`attest-result attest-result--${statusOf(check.verdict)}`} role="status">
                  <strong>{plain(check.verdict)}.</strong> {check.details}
                </p>
              )}
              {checkError && <p className="error-text" role="alert">{checkError}</p>}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
