//! CKKS side of the client: the secret key, key generation and upload, and
//! the encrypt → solve → decrypt round trip.

use std::io::{BufReader, BufWriter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rubikfhe_circuit::packed::PackedPlan;
use rubikfhe_circuit::tables::Tables;
use rubikfhe_ckks::wire::{ciphertext_from_bytes, ciphertext_to_bytes, write_eval_keys};
use rubikfhe_ckks::{EvalKeys, Params, Ring, SecretKey};
use rubikfhe_protocol::{Counting, Request, Response, Scheme, read_frame, write_frame};
use rubikfhe_solver::Thistlethwaite;
use serde::Serialize;

use crate::{connect, ms};

pub struct Keyed {
    pub ring: Ring,
    sk: SecretKey,
    pub plan: PackedPlan,
    pub output_tokens: usize,
    /// Evaluation keys until they are uploaded; dropped then (7 GB), and
    /// regenerated from the secret key if the server ever needs them again.
    keys: Mutex<Option<EvalKeys>>,
}

impl Keyed {
    fn take_keys(&self) -> Result<EvalKeys> {
        if let Some(keys) = self.keys.lock().unwrap().take() {
            return Ok(keys);
        }
        let mut seed = [0u8; 32];
        getrandom(&mut seed)?;
        self.ring.generate_eval_keys(&self.sk, &self.ring.circuit_galois_elements(&self.plan), seed)
    }
}

#[derive(Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    Preparing,
    GeneratingKeys,
    Uploading { sent_bytes: u64 },
    Ready { key_id: u64, key_bytes: u64, upload_ms: f64, server_prepare_ms: f64 },
    WaitingForServer { error: String },
    Failed { error: String },
}

pub struct CkksClient {
    pub params_name: String,
    keyed: OnceLock<Keyed>,
    phase: Mutex<Phase>,
    upload_progress: Arc<AtomicU64>,
    pub keygen_ms: Mutex<Option<f64>>,
    /// What the running solve is doing: idle, encrypting, evaluating or decrypting.
    pub step: Mutex<&'static str>,
}

/// Puts the client's step back to idle when a solve ends, however it ends.
struct IdleOnDrop<'a>(&'a Mutex<&'static str>);

impl Drop for IdleOnDrop<'_> {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = "idle";
    }
}

#[derive(Serialize)]
pub struct CkksStats {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub encrypt_ms: f64,
    pub decrypt_ms: f64,
    pub refreshes: u32,
    pub cleanings: u32,
}

impl CkksClient {
    /// Starts preparing the circuit and keys, then uploads the keys to the server.
    pub fn start(params: Params, server_addr: String) -> Arc<CkksClient> {
        let client = Arc::new(CkksClient {
            params_name: params.name.clone(),
            keyed: OnceLock::new(),
            phase: Mutex::new(Phase::Preparing),
            upload_progress: Arc::new(AtomicU64::new(0)),
            keygen_ms: Mutex::new(None),
            step: Mutex::new("idle"),
        });
        let c = client.clone();
        std::thread::spawn(move || {
            if let Err(e) = c.prepare(params) {
                eprintln!("ckks setup failed: {e:#}");
                *c.phase.lock().unwrap() = Phase::Failed { error: format!("{e:#}") };
                return;
            }
            c.upload_until_done(&server_addr);
        });
        client
    }

    fn prepare(&self, params: Params) -> Result<()> {
        let tables = Tables::new(&Thistlethwaite::new());
        let plan = PackedPlan::new(&tables, params.slots());
        let ring = Ring::new(params)?;
        *self.phase.lock().unwrap() = Phase::GeneratingKeys;
        let start = Instant::now();
        let mut seed = [0u8; 32];
        getrandom(&mut seed)?;
        let sk = ring.generate_secret(seed);
        getrandom(&mut seed)?;
        let keys = ring.generate_eval_keys(&sk, &ring.circuit_galois_elements(&plan), seed)?;
        *self.keygen_ms.lock().unwrap() = Some(ms(start));
        eprintln!("ckks keys generated in {:.1} s", start.elapsed().as_secs_f64());
        let output_tokens = tables.output_len();
        let _ = self.keyed.set(Keyed { ring, sk, plan, output_tokens, keys: Mutex::new(Some(keys)) });
        Ok(())
    }

    fn upload_until_done(&self, server_addr: &str) {
        loop {
            match self.upload(server_addr) {
                Ok(()) => return,
                Err(e) => {
                    *self.phase.lock().unwrap() = Phase::WaitingForServer { error: format!("{e:#}") };
                    std::thread::sleep(Duration::from_secs(2));
                }
            }
        }
    }

    fn upload(&self, server_addr: &str) -> Result<()> {
        let keyed = self.keyed.get().context("keys are not ready")?;
        let stream = connect(server_addr)?;
        if self.keyed.get().is_some_and(|k| k.keys.lock().unwrap().is_none()) {
            *self.phase.lock().unwrap() = Phase::GeneratingKeys;
        }
        let keys = keyed.take_keys()?;
        self.upload_progress.store(0, Ordering::Relaxed);
        *self.phase.lock().unwrap() = Phase::Uploading { sent_bytes: 0 };
        let start = Instant::now();
        let mut writer = Counting {
            inner: BufWriter::with_capacity(1 << 20, stream.try_clone()?),
            bytes: self.upload_progress.clone(),
        };
        write_frame(&mut writer, &Request::UploadKeys)?;
        write_eval_keys(&mut writer, &keys)?;
        // The server now holds them: free the memory while it prepares them.
        drop(writer);
        drop(keys);
        let (response, _) = read_frame::<_, Response>(&mut BufReader::new(stream))?;
        match response {
            Response::KeysInstalled { key_id, bytes, prepare_ms, .. } => {
                eprintln!(
                    "uploaded {:.2} GB of evaluation keys in {:.1} s",
                    bytes as f64 / 1e9,
                    start.elapsed().as_secs_f64()
                );
                *self.phase.lock().unwrap() =
                    Phase::Ready { key_id, key_bytes: bytes, upload_ms: ms(start), server_prepare_ms: prepare_ms };
                Ok(())
            }
            Response::Error(e) => bail!("server: {e}"),
            _ => bail!("unexpected response to the key upload"),
        }
    }

    pub fn phase(&self) -> Phase {
        let mut phase = self.phase.lock().unwrap().clone();
        if let Phase::Uploading { sent_bytes } = &mut phase {
            *sent_bytes = self.upload_progress.load(Ordering::Relaxed);
        }
        phase
    }

    pub fn keyed(&self) -> Option<&Keyed> {
        self.keyed.get()
    }

    /// Encrypts the sticker colors, has the server evaluate the circuit, and
    /// decrypts the move tokens.
    pub fn solve(
        self: &Arc<Self>,
        server_addr: &str,
        colors: &[u8; 54],
    ) -> Result<(Vec<i64>, CkksStats, crate::Exchange)> {
        let Phase::Ready { key_id, .. } = self.phase() else {
            bail!("the evaluation keys are not on the server yet ({})", phase_label(&self.phase()));
        };
        let keyed = self.keyed().context("keys are not ready")?;

        let _idle = IdleOnDrop(&self.step);
        *self.step.lock().unwrap() = "encrypting";
        let t = Instant::now();
        let (u, v) = keyed.plan.pack_input(colors);
        let mut seed = [0u8; 32];
        let mut inputs = Vec::with_capacity(u.len() + v.len());
        for slots in u.iter().chain(&v) {
            getrandom(&mut seed)?;
            inputs.push(ciphertext_to_bytes(&keyed.ring.encrypt(&keyed.sk, slots, seed)?)?);
        }
        let encrypt_ms = ms(t);
        let input_bytes = inputs.iter().map(Vec::len).sum();

        *self.step.lock().unwrap() = "evaluating";
        let ex = crate::exchange(server_addr, &Request::Solve { scheme: Scheme::Ckks, key_id, inputs })?;
        *self.step.lock().unwrap() = "decrypting";
        let (outputs, refreshes, cleanings) = match &ex.response {
            Response::Solution { outputs, refreshes, cleanings, .. } => (outputs, *refreshes, *cleanings),
            Response::Error(e) => {
                if e.contains("unknown evaluation keys") {
                    // The server restarted: upload the keys again in the background.
                    let (c, addr) = (self.clone(), server_addr.to_string());
                    std::thread::spawn(move || c.upload_until_done(&addr));
                }
                bail!("server: {e}")
            }
            _ => bail!("unexpected response"),
        };

        let t = Instant::now();
        let slots = outputs
            .iter()
            .map(|b| keyed.ring.decrypt(&keyed.sk, &ciphertext_from_bytes(&keyed.ring, b)?))
            .collect::<Result<Vec<_>>>()?;
        let (cols, rows) = slots.split_at(keyed.plan.token_cts[0]);
        let tokens = keyed.plan.unpack_tokens(cols, rows);
        let decrypt_ms = ms(t);
        let output_bytes = outputs.iter().map(Vec::len).sum();
        Ok((tokens, CkksStats { input_bytes, output_bytes, encrypt_ms, decrypt_ms, refreshes, cleanings }, ex))
    }
}

pub fn phase_label(phase: &Phase) -> &'static str {
    match phase {
        Phase::Preparing => "preparing the circuit",
        Phase::GeneratingKeys => "generating keys",
        Phase::Uploading { .. } => "uploading keys",
        Phase::Ready { .. } => "ready",
        Phase::WaitingForServer { .. } => "waiting for the server",
        Phase::Failed { .. } => "failed",
    }
}

/// Fresh randomness for secret keys, encryption and key seeds.
fn getrandom(buf: &mut [u8]) -> Result<()> {
    use std::io::Read;
    std::fs::File::open("/dev/urandom")?.read_exact(buf)?;
    Ok(())
}
