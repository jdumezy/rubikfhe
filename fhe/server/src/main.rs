//! rubikfhe server: receives evaluation keys and encrypted sticker colors,
//! evaluates the oblivious solver circuit on them. It never holds a secret key.

use std::io::{self, BufReader, BufWriter, Read};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use rubikfhe_circuit::packed::{Event, PackedPlan, Prepared, evaluate_inspected};
use rubikfhe_circuit::tables::Tables;
use rubikfhe_ckks::wire::{ciphertext_from_bytes, ciphertext_to_bytes};
use rubikfhe_ckks::{CkksEngine, Params, Ring, backend, system};
use rubikfhe_protocol::{
    DEFAULT_SERVER_ADDR, PROTOCOL_VERSION, Request, Response, Scheme, ServerInfo, SolveProgress, Step, plaintext,
    read_frame, write_frame,
};
use rubikfhe_solver::Thistlethwaite;

struct Server {
    solver: Thistlethwaite,
    ckks: Ckks,
    progress: Mutex<Option<SolveProgress>>,
    /// Bootstraps and cleanings of the last solve, to estimate the next one's length.
    last_report: Mutex<(u32, u32)>,
}

/// Rough cost of each step, in units of one circuit stage.
const DECODE_UNITS: f64 = 0.5;
const BOOTSTRAP_UNITS: f64 = 3.0;
const CLEAN_UNITS: f64 = 0.3;

struct Ckks {
    plan: PackedPlan,
    prepared: Prepared<CkksEngine>,
    engine: RwLock<CkksEngine>,
    key_id: Mutex<Option<u64>>,
    output_tokens: usize,
}

impl Server {
    fn info(&self) -> ServerInfo {
        ServerInfo {
            version: PROTOCOL_VERSION,
            backend: backend::NAME.to_string(),
            ckks_params: self.ckks.engine.read().unwrap().ring.params.name.clone(),
            key_id: *self.ckks.key_id.lock().unwrap(),
            stages: self.ckks.plan.stages.len() as u32,
            output_tokens: self.ckks.output_tokens as u32,
        }
    }

    fn install_keys(&self, reader: &mut impl Read) -> Result<Response> {
        let start = Instant::now();
        // Drop the current keys first: two prepared sets would not fit side by side.
        *self.ckks.key_id.lock().unwrap() = None;
        self.ckks.engine.write().unwrap().set_keys(None);

        // Each key is prepared as it arrives; receiving and preparing overlap.
        let mut counting = CountingReader { inner: reader, bytes: 0 };
        let keys = self.ckks.engine.read().unwrap().prepare_keys_from(&mut counting).context("receiving keys")?;
        self.ckks.engine.write().unwrap().set_keys(Some(keys));
        let key_id = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_nanos() as u64;
        *self.ckks.key_id.lock().unwrap() = Some(key_id);
        let elapsed = ms(start);
        eprintln!(
            "installed evaluation keys {key_id:x}: {:.2} GB received and prepared in {elapsed:.0} ms",
            counting.bytes as f64 / 1e9
        );
        Ok(Response::KeysInstalled { key_id, bytes: counting.bytes, receive_ms: elapsed, prepare_ms: elapsed })
    }

    fn solve(&self, scheme: Scheme, key_id: u64, inputs: &[Vec<u8>]) -> Result<Response> {
        let start = Instant::now();
        match scheme {
            Scheme::Plaintext => {
                let state = plaintext::decode_state(inputs.first().context("missing state")?)?;
                let moves = plaintext::encode_moves(&self.solver.solve_padded(&state));
                Ok(Response::Solution { outputs: vec![moves], solve_ms: ms(start), refreshes: 0, cleanings: 0 })
            }
            Scheme::Ckks => {
                anyhow::ensure!(
                    *self.ckks.key_id.lock().unwrap() == Some(key_id),
                    "unknown evaluation keys: upload them again"
                );
                let engine = self.ckks.engine.read().unwrap();
                let n = self.ckks.plan.state_cts;
                anyhow::ensure!(inputs.len() == 2 * n, "expected {} input ciphertexts", 2 * n);
                let cts =
                    inputs.iter().map(|b| ciphertext_from_bytes(&engine.ring, b)).collect::<io::Result<Vec<_>>>()?;

                let stages = self.ckks.plan.stages.len() as u32;
                let (refreshes, cleanings) = *self.last_report.lock().unwrap();
                let total =
                    DECODE_UNITS + stages as f64 + BOOTSTRAP_UNITS * refreshes as f64 + CLEAN_UNITS * cleanings as f64;
                let mut done = 0.0;
                let mut report_step = |stage: u32, step: Step, units: f64| {
                    let fraction = (done / total).min(0.99);
                    done += units;
                    *self.progress.lock().unwrap() =
                        Some(SolveProgress { stage, stages, step, fraction, elapsed_ms: ms(start) });
                };
                report_step(0, Step::Decoding, DECODE_UNITS);
                let result = evaluate_inspected(&*engine, &self.ckks.prepared, &cts[..n], &cts[n..], |e| match e {
                    Event::Refresh { stage } => report_step(stage as u32, Step::Bootstrapping, BOOTSTRAP_UNITS),
                    Event::Clean { stage } => report_step(stage as u32, Step::Cleaning, CLEAN_UNITS),
                    Event::Stage { stage, .. } => report_step(stage as u32, Step::Stage, 1.0),
                });
                report_step(stages, Step::Output, 0.0);
                let finished = result.and_then(|(out, report)| {
                    let outputs =
                        out.cols.iter().chain(&out.rows).map(ciphertext_to_bytes).collect::<io::Result<_>>()?;
                    Ok((outputs, report))
                });
                *self.progress.lock().unwrap() = None;
                let (outputs, report) = finished?;
                *self.last_report.lock().unwrap() = (report.refreshes as u32, report.cleanings as u32);
                eprintln!(
                    "ckks solve in {:.0} ms ({} refreshes, {} cleanings)",
                    ms(start),
                    report.refreshes,
                    report.cleanings
                );
                Ok(Response::Solution {
                    outputs,
                    solve_ms: ms(start),
                    refreshes: report.refreshes as u32,
                    cleanings: report.cleanings as u32,
                })
            }
        }
    }
}

struct CountingReader<R> {
    inner: R,
    bytes: u64,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

fn handle(server: &Server, stream: TcpStream) -> io::Result<()> {
    let mut reader = BufReader::with_capacity(1 << 20, stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    loop {
        let request = match read_frame::<_, Request>(&mut reader) {
            Ok((r, _)) => r,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        let response = match request {
            Request::Info => Ok(Response::Info(server.info())),
            Request::UploadKeys => server.install_keys(&mut reader),
            Request::Solve { scheme, key_id, inputs } => server.solve(scheme, key_id, &inputs),
            Request::Progress => Ok(Response::Progress(server.progress.lock().unwrap().clone())),
        };
        let response = response.unwrap_or_else(|e| {
            eprintln!("error: {e:#}");
            Response::Error(format!("{e:#}"))
        });
        write_frame(&mut writer, &response)?;
    }
}

fn main() -> Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());
    let params = match arg("--params").as_deref() {
        Some("test") => Params::insecure_test(12),
        _ => {
            if !flag("--allow-low-memory") {
                system::ensure_memory(system::SERVER_MIN_MEMORY)?;
            }
            Params::secure()?
        }
    };
    let threads = system::init_threads();

    let start = Instant::now();
    let solver = Thistlethwaite::new();
    let tables = Tables::new(&solver);
    let plan = PackedPlan::new(&tables, params.slots());
    let name = params.name.clone();
    let engine = CkksEngine::new(Ring::new(params)?);
    let prepared = Prepared::new(&engine, &plan)?;
    let output_tokens = tables.output_len();
    eprintln!(
        "{} backend ({threads} threads), {name} parameters, circuit ready in {:.1} s",
        backend::NAME,
        start.elapsed().as_secs_f64()
    );

    let server = Arc::new(Server {
        solver,
        ckks: Ckks { plan, prepared, engine: RwLock::new(engine), key_id: Mutex::new(None), output_tokens },
        progress: Mutex::new(None),
        // The secure parameters bootstrap twice; their bootstraps also clean.
        last_report: Mutex::new((2, 0)),
    });
    let listener = TcpListener::bind(&addr)?;
    eprintln!("listening on {addr}");
    for stream in listener.incoming() {
        let (stream, server) = (stream?, server.clone());
        std::thread::spawn(move || {
            if let Err(e) = handle(&server, stream) {
                eprintln!("connection error: {e}");
            }
        });
    }
    Ok(())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
