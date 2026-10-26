// Builds the Rust binaries, then runs the server, the client and the web dev
// server together. Ctrl-C stops all three.
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

// Cargo runs inside fhe/ so its pinned toolchain and build flags apply.
const build = spawnSync("cargo", ["build", "--release"], {
  cwd: `${root}/fhe`,
  stdio: "inherit",
});
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
