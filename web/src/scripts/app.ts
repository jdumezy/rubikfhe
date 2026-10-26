import {
  FACES,
  LAYERS,
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
import { CubeView, STICKER_COLORS, type MoveEventDetail } from "../lib/render/CubeView";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const view = new CubeView($("cube"));
const status = $("status");

let scramble: Move[] = [];
let userMoves: Move[] = [];

// --- Controls -----------------------------------------------------------------

$("scramble").addEventListener("click", () => {
  view.setState(view.getState()); // drop anything still queued
  scramble = randomScramble();
  userMoves = [];
  view.enqueue(scramble, "scramble");
  refresh();
});

$("reset").addEventListener("click", () => {
  view.setState(solved());
  scramble = [];
  userMoves = [];
  refresh();
});

view.addEventListener("move", (e) => {
  const { move, source } = (e as CustomEvent<MoveEventDetail>).detail;
  if (source === "user") userMoves.push(move);
  refresh();
});

// Match the typed character, not the physical key, so non-QWERTY layouts work.
const LAYER_BY_KEY = Object.fromEntries(LAYERS.map((l) => [l, l])) as Record<string, Layer>;

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  if (e.target instanceof HTMLElement && e.target.closest("input, textarea, [contenteditable]")) return;
  const layer = LAYER_BY_KEY[e.key.toUpperCase()];
  if (layer) {
    view.enqueue([{ layer, turns: e.shiftKey ? 3 : 1 }], "user");
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

function refreshRenderStats() {
  const s = view.stats();
  stat("fps").textContent = s.fps.toFixed(0);
  stat("drawCalls").textContent = String(s.drawCalls);
  stat("triangles").textContent = s.triangles.toLocaleString("en");
  stat("pixelRatio").textContent = String(s.pixelRatio);
  stat("queued").textContent = String(s.queued);
}

function refresh() {
  const state = view.getState();
  const done = isSolved(state);

  if (view.busy && userMoves.length === 0 && scramble.length > 0) status.textContent = "scrambling…";
  else if (done && userMoves.length > 0) status.textContent = `solved in ${userMoves.length} moves`;
  else if (userMoves.length > 0) status.textContent = `${userMoves.length} moves`;
  else status.textContent = "";

  if (!panelOpen) return;
  refreshRenderStats();
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
