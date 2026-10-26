//! Machine resources: worker threads and memory.

use anyhow::{Result, bail};

/// Worker threads on x86 CPUs; ARM CPUs use all their cores.
const X86_MAX_THREADS: usize = 16;

/// Machine memory the server asks for with the secure parameters. Measured on
/// an M5 Pro: it peaks at 21.5 GB of footprint while receiving and preparing
/// the keys (each key is prepared as it arrives), then holds 19 GB.
pub const SERVER_MIN_MEMORY: u64 = 24 << 30;

/// Machine memory the client asks for: it peaks at 8.6 GB while generating its
/// keys, and drops them once they are uploaded. Running both processes and a
/// browser on one machine, 32 GiB are comfortable.
pub const CLIENT_MIN_MEMORY: u64 = 12 << 30;

/// Sizes the global Rayon pool the poulpy backends run on, unless
/// `RAYON_NUM_THREADS` is set. Returns the number of worker threads.
pub fn init_threads() -> usize {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = if cfg!(target_arch = "aarch64") { available } else { available.min(X86_MAX_THREADS) };
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }
    rayon::current_num_threads()
}

/// Physical memory of the machine, if the platform reports it.
pub fn total_memory() -> Option<u64> {
    if cfg!(target_os = "macos") {
        let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        String::from_utf8(out.stdout).ok()?.trim().parse().ok()
    } else {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib: u64 = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
}

/// Refuses to run on a machine with less than `required` bytes of memory.
/// Linux reports a little less than the installed memory, so 5% of slack is allowed.
pub fn ensure_memory(required: u64) -> Result<()> {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    match total_memory() {
        Some(total) if total < required - required / 20 => bail!(
            "this machine has {:.1} GiB of memory; the secure parameters need {:.0} GiB \
             (pass --allow-low-memory to try anyway, or --params test for small insecure parameters)",
            gib(total),
            gib(required)
        ),
        Some(_) => Ok(()),
        None => {
            eprintln!("warning: could not read the machine's memory size; {:.0} GiB are needed", gib(required));
            Ok(())
        }
    }
}
