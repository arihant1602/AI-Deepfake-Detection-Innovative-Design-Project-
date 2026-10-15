// Typed client for server.py's JSON API.

export interface Device {
  node: string;
  name: string;
  kind: string;
  is_virtual: boolean;
  driver: string;
  bus: string;
  camera_id: string | null;
  why: string;
}

export interface Meta {
  devices: Device[];
  injections: Record<string, string>;
  noise_threshold: number;
  samples: string[];
  presets: string[];
  challenge_interval: number;
}

export interface ChallengeAttempt {
  luma: number[];
  command: number[];
  levels: [number, number];
  corr: number;
}

export interface Challenge {
  t: number;
  verdict: "PASS" | "FAIL" | "UNSUPPORTED";
  corr?: number;
  details?: string;
  attempt?: ChallengeAttempt;
}

export interface NoisePoint {
  t: number;
  sigma: number | null;
  verdict: "PRESENT" | "ABSENT" | "INCONCLUSIVE" | string;
  failed: string[];
}

export interface DeepfakePoint {
  t: number;
  p: number | null;
  abstained: boolean;
}

export type EventLevel = "block" | "warn" | "ok" | "info";

export interface LiveEvent {
  t: number;
  level: EventLevel;
  text: string;
}

export interface Snapshot {
  node: string;
  running: boolean;
  stop_reason: string | null;
  fps: number;
  in_challenge: boolean;
  next_challenge_in: number;
  hw: { verdict?: string; details?: string; latency_ms?: number };
  camera_id: string | null;
  noise: NoisePoint[];
  deepfake: DeepfakePoint[];
  deepfake_threshold: number | null;
  deepfake_smoothed: number | null;
  challenges: Challenge[];
  events: LiveEvent[];
  injection: string;
  started: number;
  now: number;
}

export interface LiveSession {
  token: string;
  node: string;
  width: number;
  height: number;
}

export interface Diagnostic {
  name: string;
  status: string;
  value: string;
}

export interface DeviceReport {
  verdict: string;
  details: string;
  latency_ms: number;
  devices: Device[];
  diagnostics: Diagnostic[];
  fingerprints: string[];
}

// Pipeline results are loosely shaped dicts; the fields the UI reads are typed here.
export interface GateDetails {
  verdict?: string;
  details?: string;
  explanation?: string;
  diagnostics_summary?: Diagnostic[];
  challenge?: { verdict?: string; corr?: number; control?: string; details?: string } | null;
  is_file_upload?: boolean;
  encoder?: string;
  mode?: string;
  enforced?: boolean;
  pce_score?: number;
  noise_sigma?: number;
  failed_checks?: string[];
  frames_analyzed?: number;
  latency_ms?: number;
  score?: number;
  threshold?: number;
  abstained?: boolean;
  model?: string;
  frames_with_face?: number;
  attestation?: GateDetails;
  prnu?: GateDetails;
  temporal?: GateDetails;
}

export interface AnalysisResult {
  verdict: string;
  gate: number | null;
  limitations?: string[];
  details: GateDetails;
  attestation?: GateDetails;
  prnu?: GateDetails;
  timings: Record<string, number>;
  thumbs: { index: number; src: string }[];
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the server. Is `python server.py` running?");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(body?.error ?? `The server answered ${res.status}.`);
  return body as T;
}

const post = <T>(path: string, data: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const api = {
  meta: () => request<Meta>("/api/meta"),
  start: (node: string) => post<LiveSession>("/api/live/start", { node }),
  state: (token: string) => request<Snapshot>(`/api/live/state?token=${encodeURIComponent(token)}`),
  stop: (token: string) => post<{ stopped: boolean }>("/api/live/stop", { token }),
  challenge: (token: string) => post<{ requested: boolean }>("/api/live/challenge", { token }),
  attack: (token: string, mode: string) => post<{ injection: string }>("/api/live/attack", { token, mode }),
  streamUrl: (token: string) => `/api/live.mjpg?token=${encodeURIComponent(token)}`,
  // Releases the camera even while the tab is closing.
  stopBeacon: (token: string) => navigator.sendBeacon(`/api/live/stop?token=${encodeURIComponent(token)}`),
  analyze: (form: FormData) => request<AnalysisResult>("/api/analyze", { method: "POST", body: form }),
  sampleUrl: (name: string) => `/api/samples/${encodeURIComponent(name)}`,
  device: () => request<DeviceReport>("/api/device"),
  attestation: (payload: string) => post<{ verdict: string; details: string }>("/api/attestation", { payload }),
};
