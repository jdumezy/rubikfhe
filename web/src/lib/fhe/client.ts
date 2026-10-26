/**
 * HTTP API of the local rubikfhe client (fhe/client), proxied under /api by
 * the dev server. The browser only ever talks to the client, which holds the
 * secret key; the client talks to the server.
 */
import { parseMoves, type Move } from "../cube";

export interface Status {
  client: {
    version: string;
    scheme: string;
    keygen_ms: number;
    eval_key_bytes: number;
  };
  server: {
    addr: string;
    reachable: boolean;
    solver: string | null;
    phases: string[];
    phase_slots: number[];
    error: string | null;
  };
}

export interface SolveStats {
  scheme: string;
  slots: number;
  phase_slots: number[];
  phase_moves: number[];
  raw_moves: number;
  moves: number;
  eval_key_bytes: number;
  state_bytes: number;
  solution_bytes: number;
  upload_bytes: number;
  download_bytes: number;
  encrypt_ms: number;
  solve_ms: number;
  network_ms: number;
  decrypt_ms: number;
  total_ms: number;
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
