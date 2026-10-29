// Builds the Rust binaries, then runs the server, the client and the web dev
// server together. Ctrl-C stops all three.
import { spawn, spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

// The server runs on the GPU whenever there is one: CUDA on NVIDIA cards (with
// their driver), HIP on AMD cards (with ROCm), wgpu on the others (Metal on
// macOS, Vulkan elsewhere). The CPU only without a GPU. RUBIKFHE_GPU (cuda,
// hip, wgpu or none) overrides the choice.
function gpuFeature() {
  const forced = process.env.RUBIKFHE_GPU;
  if (forced) return forced === "none" ? null : forced;
  const runs = (cmd, args) => spawnSync(cmd, args, { stdio: "ignore" }).status === 0;
  if (runs("nvidia-smi", ["-L"])) return "cuda";
  if (runs("rocminfo", [])) return "hip";
  if (process.platform === "darwin" || process.platform === "win32") return "wgpu";
  // Linux lists its display adapters, whatever their drivers.
  const hasGpu = () => {
    try {
      return readdirSync("/sys/class/drm").some((d) => /^card\d+$/.test(d));
    } catch {
      return false;
    }
  };
  if (hasGpu() || runs("vulkaninfo", ["--summary"])) return "wgpu";
  return null;
}

const gpu = gpuFeature();
console.log(gpu ? `server on the GPU (${gpu})` : "server on the CPU");

// Cargo runs inside fhe/ so its pinned toolchain and build flags apply.
const build = spawnSync(
  "cargo",
  ["build", "--release", ...(gpu ? ["--features", `rubikfhe-ckks/${gpu}`] : [])],
  { cwd: `${root}/fhe`, stdio: "inherit" },
);
if (build.status !== 0) process.exit(build.status ?? 1);

const processes = [
  { name: "server", color: 35, cmd: "fhe/target/release/rubikfhe-server", args: [] },
  { name: "client", color: 36, cmd: "fhe/target/release/rubikfhe-client", args: [] },
  { name: "web", color: 32, cmd: "npm", args: ["--prefix", "web", "run", "dev"] },
];

const children = processes.map(({ name, color, cmd, args }) => {
  const child = spawn(cmd, args, { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
  const label = `\x1b[${color}m${name.padEnd(6)}\x1b[0m│ `;
  for (const stream of [child.stdout, child.stderr]) {
    let pending = "";
    stream.on("data", (chunk) => {
      const lines = (pending + chunk).split("\n");
      pending = lines.pop();
      for (const line of lines) process.stdout.write(label + line + "\n");
    });
  }
  child.on("exit", (code, signal) => {
    process.stdout.write(`${label}exited (${signal ?? code})\n`);
    shutdown(code ?? 0);
  });
  return child;
});

let stopping = false;
function shutdown(code) {
  if (stopping) return;
  stopping = true;
  for (const child of children) if (child.exitCode === null) child.kill("SIGINT");
  setTimeout(() => process.exit(code), 500).unref();
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
