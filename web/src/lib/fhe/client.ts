/**
 * HTTP API of the local rubikfhe client (fhe/client), proxied under /api by
 * the dev server. The browser only ever talks to the client, which holds the
 * secret key; the client talks to the server.
 */
import { parseMoves, type Move } from "../cube";

/** The client's CKKS setup: keys are generated, then uploaded to the server. */
export type SetupPhase =
  | { phase: "preparing" }
  | { phase: "generating_keys" }
  | { phase: "uploading"; sent_bytes: number }
  | { phase: "ready"; key_id: number; key_bytes: number; upload_ms: number; server_prepare_ms: number }
  | { phase: "waiting_for_server"; error: string }
  | { phase: "failed"; error: string };

export interface Status {
  client: {
    version: string;
    scheme: string;
    backend: string;
    params: string | null;
    setup: SetupPhase | null;
    keygen_ms: number | null;
    stages: number | null;
    output_tokens: number | null;
  };
  server: {
    addr: string;
    reachable: boolean;
    info: {
      backend: string;
      ckks_params: string;
      key_id: number | null;
      stages: number;
      output_tokens: number;
    } | null;
    error: string | null;
  };
}

export interface SolveStats {
  scheme: string;
  output_slots: number;
  raw_moves: number;
  moves: number;
  upload_bytes: number;
  download_bytes: number;
  solve_ms: number;
  network_ms: number;
  total_ms: number;
  // CKKS only.
  input_bytes?: number;
  output_bytes?: number;
  encrypt_ms?: number;
  decrypt_ms?: number;
  refreshes?: number;
  cleanings?: number;
}

export interface Progress {
  step: "idle" | "encrypting" | "evaluating" | "decrypting";
  server: {
    stage: number;
    stages: number;
    step: "decoding" | "stage" | "bootstrapping" | "cleaning" | "output";
    fraction: number;
    elapsed_ms: number;
  } | null;
}

export interface SolveResult {
  moves: Move[];
  stats: SolveStats;
}

/** The client process is not running (or not reachable through the proxy). */
export class ClientUnavailable extends Error {
  constructor() {
    super("the rubikfhe client is not running (start everything with `npm run dev`)");
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  let body: unknown;
  try {
    res = await fetch(path, init);
    body = await res.json();
  } catch {
    throw new ClientUnavailable();
  }
  if (!res.ok) {
    const message = (body as { error?: string } | null)?.error;
    if (!message) throw new ClientUnavailable();
    throw new Error(message);
  }
  return body as T;
}

export function fetchProgress(): Promise<Progress> {
  return call<Progress>("/api/progress");
}

export function fetchStatus(): Promise<Status> {
  return call<Status>("/api/status");
}

export async function solve(facelets: string): Promise<SolveResult> {
  const body = await call<{ moves: string[]; stats: SolveStats }>("/api/solve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ facelets }),
  });
  return { moves: parseMoves(body.moves.join(" ")), stats: body.stats };
}
