# rubikfhe

A Rubik's cube solver that runs under fully homomorphic encryption.

## Requirements

- [Rust](https://rustup.rs) (rustup installs the pinned toolchain)
- Node.js 22 or later
- 16 GB of RAM
- Optional GPU: the CUDA toolkit on NVIDIA, ROCm on AMD

## Run

```sh
npm --prefix web install
npm run dev
```

Then open http://localhost:4321.

The server picks the best hardware it finds: CUDA on NVIDIA, HIP on AMD, Metal or Vulkan on other GPUs, otherwise the fastest CPU backend (`RUBIKFHE_GPU=none` forces the CPU).
