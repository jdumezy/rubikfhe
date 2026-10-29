//! The poulpy backend: the fastest NTT backend the build target supports,
//! chosen at compile time (the workspace builds with `-C target-cpu=native`).

#[cfg(target_arch = "aarch64")]
mod chosen {
    pub type Be = poulpy_cpu_arm::NTT4x30NeonRayon;
    pub const NAME: &str = "NTT4x30NeonRayon";
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma", target_feature = "avx512vl"))]
mod chosen {
    pub type Be = poulpy_cpu_avx512::NTT3x42IfmaRayon;
    pub const NAME: &str = "NTT3x42IfmaRayon";
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    not(all(target_feature = "avx512ifma", target_feature = "avx512vl"))
))]
mod chosen {
    pub type Be = poulpy_cpu_avx512::NTT4x30Avx512Rayon;
    pub const NAME: &str = "NTT4x30Avx512Rayon";
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", not(target_feature = "avx512f")))]
mod chosen {
    pub type Be = poulpy_cpu_avx::NTT4x30AvxRayon;
    pub const NAME: &str = "NTT4x30AvxRayon";
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
)))]
mod chosen {
    pub type Be = poulpy_cpu_portable::NTT4x30Portable;
    pub const NAME: &str = "NTT4x30Portable";
}

pub use chosen::{Be, NAME};

/// The backend the server evaluates on: the GPU runtime the build enables, at
/// most one (CUDA first, then HIP, then wgpu), or else the CPU backend.
#[cfg(feature = "cuda")]
mod device {
    pub type Dev = poulpy_gpu_portable::NTT4x30Cuda;
    pub const DEVICE_NAME: &str = "NTT4x30Cuda";
    pub const ON_GPU: bool = true;
}

#[cfg(all(feature = "hip", not(feature = "cuda")))]
mod device {
    pub type Dev = poulpy_gpu_portable::NTT4x30Hip;
    pub const DEVICE_NAME: &str = "NTT4x30Hip";
    pub const ON_GPU: bool = true;
}

#[cfg(all(feature = "wgpu", not(any(feature = "cuda", feature = "hip"))))]
mod device {
    pub type Dev = poulpy_gpu_portable::NTT4x30Wgpu;
    pub const DEVICE_NAME: &str =
        if cfg!(target_os = "macos") { "NTT4x30Wgpu (Metal)" } else { "NTT4x30Wgpu (Vulkan)" };
    pub const ON_GPU: bool = true;
}

#[cfg(not(any(feature = "cuda", feature = "hip", feature = "wgpu")))]
mod device {
    pub type Dev = super::Be;
    pub const DEVICE_NAME: &str = super::NAME;
    pub const ON_GPU: bool = false;
}

pub use device::{DEVICE_NAME, Dev, ON_GPU};
