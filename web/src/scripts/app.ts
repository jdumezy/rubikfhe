import {
  FACES,
  LAYERS,
  formatMove,
  formatMoves,
  isSolved,
  randomScramble,
  solved,
  toFaceletString,
  toFacelets,
  type Face,
  type Layer,
  type Move,
} from "../lib/cube";
import * as fhe from "../lib/fhe/client";
import { CubeView, STICKER_COLORS, type MoveEventDetail } from "../lib/render/CubeView";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const view = new CubeView($("cube"));
const status = $("status");
const solutionLine = $("solution");
const solveButton = $<HTMLButtonElement>("solve");

let scramble: Move[] = [];
let userMoves: Move[] = [];

// --- Solving ----------------------------------------------------------------------

type SolvePhase = "idle" | "requesting" | "playing";

let solvePhase: SolvePhase = "idle";
let solution: { moves: Move[]; played: number } | null = null;
let solveError = "";
let lastStats: fhe.SolveStats | null = null;
/** Bumped whenever the cube changes under a pending solve, to drop stale replies. */
let solveToken = 0;

const idle = () =>
  view.busy ? new Promise<void>((r) => view.addEventListener("idle", () => r(), { once: true })) : Promise.resolve();

function setSolvePhase(phase: SolvePhase) {
  solvePhase = phase;
  view.setTurnsLocked(phase !== "idle");
  solveButton.disabled = phase !== "idle";
}

function cancelSolve() {
  solveToken++;
  solution = null;
  solveError = "";
  setSolvePhase("idle");
}

async function solve() {
  if (solvePhase !== "idle") return;
  const token = ++solveToken;
  solution = null;
  solveError = "";
  setSolvePhase("requesting");
  refresh();

  await idle();
  if (token !== solveToken) return;
  try {
    const result = await fhe.solve(toFaceletString(view.getState()));
    if (token !== solveToken) return;
    lastStats = result.stats;
    solution = { moves: result.moves, played: 0 };
    if (result.moves.length > 0) {
      setSolvePhase("playing");
      view.enqueue(result.moves, "solver");
    } else {
      setSolvePhase("idle");
    }
  } catch (e) {
    if (token !== solveToken) return;
    solveError = e instanceof Error ? e.message : String(e);
    setSolvePhase("idle");
    refreshClientStatus();
  }
  refresh();
}

// --- Controls -----------------------------------------------------------------

$("scramble").addEventListener("click", () => {
  cancelSolve();
  view.setState(view.getState()); // drop anything still queued
  scramble = randomScramble();
  userMoves = [];
  view.enqueue(scramble, "scramble");
  refresh();
});

solveButton.addEventListener("click", solve);

$("reset").addEventListener("click", () => {
  cancelSolve();
  view.setState(solved());
  scramble = [];
  userMoves = [];
  refresh();
});

view.addEventListener("move", (e) => {
  const { move, source } = (e as CustomEvent<MoveEventDetail>).detail;
  if (source === "user") {
    userMoves.push(move);
    if (solvePhase === "idle") solution = null;
  } else if (source === "solver" && solution) {
    solution.played++;
    if (solution.played === solution.moves.length) setSolvePhase("idle");
  }
  refresh();
});

// Match the typed character, not the physical key, so non-QWERTY layouts work.
const LAYER_BY_KEY = Object.fromEntries(LAYERS.map((l) => [l, l])) as Record<string, Layer>;

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  if (e.target instanceof HTMLElement && e.target.closest("input, textarea, [contenteditable]")) return;
  const layer = LAYER_BY_KEY[e.key.toUpperCase()];
  if (layer) {
    if (solvePhase === "idle") view.enqueue([{ layer, turns: e.shiftKey ? 3 : 1 }], "user");
  } else if (e.key === "`") {
    setPanelOpen(!panelOpen);
  } else if (e.key === "Escape" && panelOpen) {
    setPanelOpen(false);
  } else {
    return;
  }
  e.preventDefault();
});

// --- Theme ----------------------------------------------------------------------

$("theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try {
    localStorage.setItem("theme", dark ? "dark" : "light");
  } catch {}
});

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let stored = null;
  try {
    stored = localStorage.getItem("theme");
  } catch {}
  if (!stored) document.documentElement.classList.toggle("dark", e.matches);
});

// --- Status and stats for nerds ------------------------------------------------

const panel = $("nerd-panel");
const toggle = $("nerd-toggle");
let panelOpen = false;
let statsTimer: number | undefined;

function setPanelOpen(open: boolean) {
  panelOpen = open;
  panel.toggleAttribute("data-open", open);
  panel.inert = !open;
  toggle.setAttribute("aria-pressed", String(open));
  // On wide screens, make room for the panel instead of covering the cube.
  document.body.classList.toggle("lg:pr-96", open);
  clearInterval(statsTimer);
  if (open) {
    refresh();
    refreshClientStatus();
    statsTimer = window.setInterval(refreshRenderStats, 250);
  }
}

toggle.addEventListener("click", () => setPanelOpen(!panelOpen));
$("nerd-close").addEventListener("click", () => setPanelOpen(false));

const stat = (name: string) => panel.querySelector<HTMLElement>(`[data-stat="${name}"]`)!;

// Unfolded net: U on top, L F R B in the middle row, D at the bottom.
const NET_ORIGIN: Record<Face, [number, number]> = { U: [3, 0], L: [0, 3], F: [3, 3], R: [6, 3], B: [9, 3], D: [3, 6] };
const netCells = Array.from({ length: 54 }, (_, i) => {
  const [col, row] = NET_ORIGIN[FACES[Math.floor(i / 9)]];
  const cell = document.createElement("div");
  cell.className = "size-3 rounded-[2px]";
  cell.style.gridColumn = String(col + (i % 3) + 1);
  cell.style.gridRow = String(row + Math.floor((i % 9) / 3) + 1);
  $("nerd-net").appendChild(cell);
  return cell;
});

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return `${(n / 1024 ** 3).toFixed(2)} GiB`;
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

async function refreshClientStatus() {
  let s: fhe.Status | null = null;
  try {
    s = await fhe.fetchStatus();
  } catch {}
  if (!s) {
    for (const name of ["fheScheme", "fheServer", "fheSolver", "fheKey"]) stat(name).textContent = "–";
    stat("fheClient").textContent = "not running";
    return;
  }
  const { client, server } = s;
  stat("fheScheme").textContent = client.scheme;
  stat("fheClient").textContent = `running, v${client.version}`;
  stat("fheServer").textContent = `${server.reachable ? "reachable" : "unreachable"} at ${server.addr}`;
  stat("fheSolver").textContent = server.solver
    ? `${server.solver}, ${server.phase_slots.join("+")} slots`
    : "–";
  stat("fheKey").textContent = `${formatBytes(client.eval_key_bytes)} (keygen ${formatMs(client.keygen_ms)})`;
}

function refreshRenderStats() {
  const s = view.stats();
  stat("fps").textContent = s.fps.toFixed(0);
  stat("drawCalls").textContent = String(s.drawCalls);
  stat("triangles").textContent = s.triangles.toLocaleString("en");
  stat("pixelRatio").textContent = String(s.pixelRatio);
  stat("queued").textContent = String(s.queued);
}

function refreshStatusLine(done: boolean) {
  if (solvePhase === "requesting")
    status.textContent = view.busy ? "finishing moves…" : "encrypting, solving, decrypting…";
  else if (solvePhase === "playing") status.textContent = `solution: ${solution!.moves.length} moves`;
  else if (solveError) status.textContent = `error: ${solveError}`;
  else if (solution && done)
    status.textContent = solution.moves.length ? `solved by rubikfhe in ${solution.moves.length} moves` : "already solved";
  else if (view.busy && userMoves.length === 0 && scramble.length > 0) status.textContent = "scrambling…";
  else if (done && userMoves.length > 0) status.textContent = `solved in ${userMoves.length} moves`;
  else if (userMoves.length > 0) status.textContent = `${userMoves.length} moves`;
  else status.textContent = "";
}

/** The decrypted solution, with played moves dimmed and the current one highlighted. */
function refreshSolutionLine() {
  solutionLine.replaceChildren(
    ...(solution?.moves ?? []).map((m, i) => {
      const span = document.createElement("span");
      span.textContent = formatMove(m);
      span.className =
        i < solution!.played
          ? "text-zinc-400 dark:text-zinc-600"
          : i === solution!.played && solvePhase === "playing"
            ? "font-semibold text-emerald-600 dark:text-emerald-400"
            : "";
      return span;
    }),
  );
  // Plain spaces between spans keep the line wrappable.
  for (const span of [...solutionLine.children].slice(1)) span.before(" ");
}

function refreshLastSolve() {
  const s = lastStats;
  if (!s) return;
  stat("fheCiphertexts").textContent = `${formatBytes(s.state_bytes)} / ${formatBytes(s.solution_bytes)}`;
  stat("fheTraffic").textContent = `${formatBytes(s.upload_bytes)} / ${formatBytes(s.download_bytes)}`;
  stat("fheEncrypt").textContent = formatMs(s.encrypt_ms);
  stat("fheSolve").textContent = formatMs(s.solve_ms);
  stat("fheDecrypt").textContent = formatMs(s.decrypt_ms);
  stat("fheTotal").textContent = `${formatMs(s.network_ms)} / ${formatMs(s.total_ms)}`;
  stat("fheMoves").textContent = `${s.raw_moves} / ${s.moves} (${s.slots} slots)`;
  stat("fhePhases").textContent = s.phase_moves.map((n, i) => `${n}/${s.phase_slots[i]}`).join(" ");
}

function refresh() {
  const state = view.getState();
  const done = isSolved(state);
  refreshStatusLine(done);
  refreshSolutionLine();

  if (!panelOpen) return;
  refreshRenderStats();
  refreshLastSolve();
  stat("solved").textContent = String(done);
  stat("moveCount").textContent = String(userMoves.length);
  toFacelets(state).forEach((c, i) => (netCells[i].style.backgroundColor = STICKER_COLORS[c]));
  stat("facelets").textContent = toFaceletString(state);
  stat("cp").textContent = `[${state.cp.join(" ")}]`;
  stat("co").textContent = `[${state.co.join(" ")}]`;
  stat("ep").textContent = `[${state.ep.join(" ")}]`;
  stat("eo").textContent = `[${state.eo.join(" ")}]`;
  stat("ctr").textContent = `[${state.ctr.map((c) => FACES[c]).join(" ")}]`;
  stat("scramble").textContent = formatMoves(scramble) || "–";
  stat("history").textContent = formatMoves(userMoves) || "–";
}

refresh();
