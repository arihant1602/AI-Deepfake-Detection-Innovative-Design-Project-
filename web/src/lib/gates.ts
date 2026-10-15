// Turns live snapshots and file-analysis results into the four gate cards and the overall verdict.
import type { AnalysisResult, GateDetails, Snapshot } from "./api";

export type Status = "pass" | "block" | "flag" | "wait" | "off";

export interface GateState {
  gate: string;
  name: string;
  status: Status;
  label: string;
  value?: string;
  ms?: number;
}

export interface Verdict {
  status: Status;
  word: string; // the stacked display word
  title: string;
  line: string;
}

const PLAIN_NOISE: Record<string, string> = {
  "noise present": "video is too clean for a live camera",
  "no codec copying": "video was compressed like a file",
  "sensor-like texture": "grain looks digitally added",
  "sensor-like colour": "grain colours don't match a camera",
};

const pct = (p: number) => `${Math.round(p * 100)}%`;

export function liveGates(s: Snapshot): GateState[] {
  const now = s.now;
  const v = s.hw.verdict ?? "PASS";
  const g1a: GateState = {
    gate: "1a",
    name: "Real camera connected",
    status: v === "PASS" ? "pass" : v === "FLAG_FOR_REVIEW" ? "flag" : "block",
    label: v === "PASS" ? "Yes" : v === "FLAG_FOR_REVIEW" ? "Yes, but suspicious software is running" : "No",
    value: s.camera_id ?? undefined,
  };

  let g1b: GateState;
  const ch = s.challenges[0];
  const flash = s.screen_flashes[0];
  if (s.in_challenge) {
    g1b = { gate: "1b", name: "Camera is live", status: "flag", label: "Camera test running" };
  } else if (s.flash_plan) {
    g1b = { gate: "1b", name: "Camera is live", status: "flag", label: "Screen flash running" };
  } else if (ch) {
    const passed = ch.verdict === "PASS";
    const missed = Object.entries(ch.channels ?? {})
      .filter(([, c]) => !c.passed)
      .map(([name]) => name.replace("_", " "));
    g1b = {
      gate: "1b",
      name: "Camera is live",
      status: passed ? "pass" : ch.verdict === "FAIL" ? "block" : "off",
      label: passed
        ? "Yes"
        : ch.verdict === "FAIL"
          ? `No, the video ignored the ${missed.join(" and ") || "camera"} change`
          : "Can't test this camera",
      value: [ch.controls?.length ? `${ch.controls.length} controls` : null, `tested ${Math.max(0, Math.round(now - ch.t))} s ago`]
        .filter(Boolean)
        .join(" · "),
    };
    // The screen flash is advisory: a bright room can drown the reflection, so a miss only flags.
    if (passed && flash?.verdict === "NO_REFLECTION")
      g1b = { ...g1b, status: "flag", label: "Yes, but the screen flash didn't light the face" };
    else if (passed && flash?.verdict === "PASS") g1b = { ...g1b, value: `screen reflection seen · ${g1b.value}` };
  } else {
    g1b = { gate: "1b", name: "Camera is live", status: "wait", label: "Testing" };
  }

  let g2: GateState = { gate: "2", name: "Has real camera grain", status: "wait", label: "Warming up" };
  const n = s.noise.at(-1);
  if (n?.verdict === "PRESENT") g2 = { ...g2, status: "pass", label: "Yes" };
  else if (n?.verdict === "ABSENT")
    g2 = { ...g2, status: "block", label: "No", value: n.failed[0] ? PLAIN_NOISE[n.failed[0]] ?? n.failed[0] : undefined };
  else if (n) g2 = { ...g2, label: "Hold still a moment" };
  const same = s.same_sensor.at(-1);
  if (same?.verdict === "MISMATCH")
    g2 = { ...g2, status: "block", label: "No, the sensor changed after the live test", value: `PCE ${Math.round(same.pce ?? 0)}` };
  else if (same?.verdict === "MATCH" && g2.status === "pass") g2 = { ...g2, value: "same sensor as the live test" };

  const thr = s.deepfake_threshold;
  let g3: GateState = {
    gate: "3",
    name: "Face looks real",
    status: "wait",
    label: thr === null ? "Loading the AI detector" : "Checking",
  };
  const last = s.deepfake.at(-1);
  const p = s.deepfake_smoothed;
  if (last && (last.abstained || last.p === null)) g3 = { ...g3, label: "Move closer and face the camera" };
  else if (last && p !== null && thr !== null) {
    const fake = p >= thr;
    const unsure = !fake && p >= 0.8 * thr;
    g3 = {
      ...g3,
      status: fake ? "block" : unsure ? "flag" : "pass",
      label: fake ? "No, it looks AI-generated" : unsure ? "Unsure" : "Yes",
      value: `AI score ${pct(p)}`,
    };
  }
  return [g1a, g1b, g2, g3];
}

export function liveVerdict(gates: GateState[] | null, s: Snapshot | null): Verdict {
  if (!s) return { status: "off", word: "Idle", title: "Camera off", line: "Pick a camera and start checking." };
  const blocked = (gates ?? []).filter((g) => g.status === "block");
  if (blocked.length)
    return {
      status: "block",
      word: "Fake",
      title: "Fake video detected",
      line: `Failed: ${blocked.map((g) => g.name.toLowerCase()).join(", ")}.`,
    };
  if (!s.running) return { status: "off", word: "Off", title: "Camera off", line: s.stop_reason ?? "" };
  if (gates?.every((g) => g.status === "pass"))
    return { status: "pass", word: "Real", title: "Real person, real camera", line: "All four checks pass." };
  const waiting = (gates ?? []).filter((g) => g.status !== "pass").map((g) => `${g.name}: ${g.label.toLowerCase()}`);
  return {
    status: gates?.some((g) => g.status === "flag") ? "flag" : "wait",
    word: "Checking",
    title: "Still checking",
    line: waiting.join(". ") + ".",
  };
}

// ----------------------------------------------------------------------------- file analysis

export function fileGates(r: AnalysisResult): GateState[] {
  const { gate, timings: t } = r;
  const d = r.details ?? {};
  const hw: GateDetails = gate === 1 ? d : d.attestation ?? r.attestation ?? {};
  const ch = hw.challenge ?? null;
  const chFailed = ch?.verdict === "FAIL";

  const g1a: GateState =
    gate === 1 && !chFailed
      ? { gate: "1a", name: "Real camera", status: "block", label: "Blocked" }
      : hw.verdict === "FLAG_FOR_REVIEW"
        ? { gate: "1a", name: "Real camera", status: "flag", label: "Review" }
        : hw.verdict === "UNATTESTED_ORIGIN"
          ? { gate: "1a", name: "Real camera", status: "off", label: "Not checked for files" }
          : { gate: "1a", name: "Real camera", status: "pass", label: "Physical camera" };
  g1a.ms = t.gate1_ms ?? t.gate1_passive_ms;

  const g1b: GateState =
    gate === 1 && !chFailed
      ? { gate: "1b", name: "Live sensor", status: "off", label: "Not run" }
      : ch?.verdict === "PASS"
        ? { gate: "1b", name: "Live sensor", status: "pass", label: "Responded", value: `r = ${(ch.corr ?? 0).toFixed(2)}` }
        : chFailed
          ? { gate: "1b", name: "Live sensor", status: "block", label: "Did not respond", value: `r = ${(ch?.corr ?? 0).toFixed(2)}` }
          : { gate: "1b", name: "Live sensor", status: "off", label: ch ? "Unsupported" : "Not possible for files" };

  const prnu: GateDetails | undefined = d.prnu ?? r.prnu ?? (gate === 2 ? d : undefined);
  let g2: GateState;
  if (gate === 1 || !prnu) g2 = { gate: "2", name: "Sensor noise", status: "off", label: "Not run" };
  else if (prnu.same_sensor?.verdict === "MISMATCH")
    g2 = {
      gate: "2",
      name: "Sensor noise",
      status: "block",
      label: "Different sensor than the live test",
      value: `PCE ${Math.round(prnu.same_sensor.pce ?? 0)}`,
    };
  else if (prnu.mode === "reference" && (prnu.verdict === "PRESENT" || prnu.verdict === "ABSENT")) {
    const match = prnu.verdict === "PRESENT";
    g2 = {
      gate: "2",
      name: "Sensor noise",
      status: match ? "pass" : "block",
      label: match ? "Fingerprint matches" : "Fingerprint doesn't match",
      value: `PCE ${Math.round(prnu.pce_score ?? 0).toLocaleString()}`,
    };
  } else if (prnu.verdict === "PRESENT")
    g2 = { gate: "2", name: "Sensor noise", status: "pass", label: "Live noise", value: `level ${(prnu.noise_sigma ?? 0).toFixed(2)}` };
  else if (prnu.verdict === "ABSENT")
    g2 = {
      gate: "2",
      name: "Sensor noise",
      status: prnu.enforced ? "block" : "flag",
      label: prnu.enforced ? "No live noise" : "No live noise (advisory)",
      value: (prnu.failed_checks ?? []).join(", ") || undefined,
    };
  else g2 = { gate: "2", name: "Sensor noise", status: "off", label: "Inconclusive" };
  g2.ms = t.gate2_ms;

  const tmp: GateDetails | undefined = d.temporal ?? (gate === 3 ? d : undefined);
  let g3: GateState;
  if (gate === 1 || gate === 2 || !tmp) g3 = { gate: "3", name: "Face", status: "off", label: "Not run" };
  else if (gate === 3)
    g3 = { gate: "3", name: "Face", status: "block", label: "Synthetic face", value: `AI score ${pct(tmp.score ?? 0)}` };
  else if (tmp.abstained) g3 = { gate: "3", name: "Face", status: "off", label: "No face found" };
  else g3 = { gate: "3", name: "Face", status: "pass", label: "Real face", value: `AI score ${pct(tmp.score ?? 0)}` };
  g3.ms = t.gate3_ms;

  return [g1a, g1b, g2, g3];
}

export function fileVerdict(r: AnalysisResult): Verdict {
  if (r.verdict === "ERROR")
    return { status: "block", word: "Error", title: "The file could not be analysed", line: r.details?.explanation ?? "" };
  if (r.verdict === "AUTHENTIC LIVE STREAM")
    return { status: "pass", word: "Real", title: "Authentic", line: "Every check ran and passed." };
  if (r.verdict === "NO ANOMALY DETECTED")
    return {
      status: "flag",
      word: "Clear",
      title: "No attack detected",
      line: "Nothing failed, but not every check could run on a file.",
    };
  return {
    status: "block",
    word: "Fake",
    title: r.verdict.charAt(0) + r.verdict.slice(1).toLowerCase(),
    line: `Stopped at gate ${r.gate}; later gates did not run.`,
  };
}
