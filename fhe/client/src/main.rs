//! rubikfhe client: holds the secret key. The browser sends it the cube; it
//! encrypts the stickers, has the server evaluate the solver circuit, and
//! decrypts the moves.

mod ckks;

use std::io::{BufReader, BufWriter};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use ckks::{CkksClient, CkksStats, Phase};
use rubikfhe_circuit::reference::tokens_to_moves;
use rubikfhe_ckks::Params;
use rubikfhe_cube::{Move, facelet_colors, from_facelets, simplify};
use rubikfhe_protocol::{
    DEFAULT_CLIENT_ADDR, DEFAULT_SERVER_ADDR, Request, Response, Scheme, ServerInfo, SolveProgress, plaintext,
    read_frame, write_frame,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

struct App {
    scheme: Scheme,
    server_addr: String,
    ckks: Option<Arc<CkksClient>>,
}

type ApiError = (StatusCode, Json<Value>);

fn error(status: StatusCode, message: impl std::fmt::Display) -> ApiError {
    (status, Json(json!({ "error": format!("{message:#}") })))
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

fn connect(addr: &str) -> Result<TcpStream> {
    let socket = addr.to_socket_addrs()?.next().context("no address")?;
    TcpStream::connect_timeout(&socket, CONNECT_TIMEOUT).with_context(|| format!("server unreachable at {addr}"))
}

/// One request/response round trip with the server, with sizes on the wire.
pub struct Exchange {
    response: Response,
    sent: usize,
    received: usize,
    elapsed_ms: f64,
}

fn exchange(addr: &str, request: &Request) -> Result<Exchange> {
    let start = Instant::now();
    let stream = connect(addr)?;
    let sent = write_frame(&mut BufWriter::new(stream.try_clone()?), request)?;
    let (response, received) = read_frame(&mut BufReader::new(stream))?;
    Ok(Exchange { response, sent, received, elapsed_ms: ms(start) })
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.expect("blocking task panicked")
}

// --- GET /api/status -------------------------------------------------------------

#[derive(Serialize)]
struct Status {
    client: ClientStatus,
    server: ServerStatus,
}

#[derive(Serialize)]
struct ClientStatus {
    version: &'static str,
    scheme: &'static str,
    backend: &'static str,
    params: Option<String>,
    setup: Option<Phase>,
    keygen_ms: Option<f64>,
    stages: Option<usize>,
    output_tokens: Option<usize>,
}

#[derive(Serialize)]
struct ServerStatus {
    addr: String,
    reachable: bool,
    info: Option<ServerInfo>,
    error: Option<String>,
}

async fn status(State(app): State<Arc<App>>) -> Json<Status> {
    let keyed = app.ckks.as_ref().and_then(|c| c.keyed());
    let client = ClientStatus {
        version: env!("CARGO_PKG_VERSION"),
        scheme: app.scheme.name(),
        backend: rubikfhe_ckks::backend::NAME,
        params: app.ckks.as_ref().map(|c| c.params_name.clone()),
        setup: app.ckks.as_ref().map(|c| c.phase()),
        keygen_ms: app.ckks.as_ref().and_then(|c| *c.keygen_ms.lock().unwrap()),
        stages: keyed.map(|k| k.plan.stages.len()),
        output_tokens: keyed.map(|k| k.output_tokens),
    };
    let addr = app.server_addr.clone();
    let server = blocking(move || match exchange(&addr, &Request::Info) {
        Ok(Exchange { response: Response::Info(info), .. }) => {
            ServerStatus { addr, reachable: true, info: Some(info), error: None }
        }
        Ok(_) => ServerStatus { addr, reachable: true, info: None, error: Some("unexpected response".into()) },
        Err(e) => ServerStatus { addr, reachable: false, info: None, error: Some(format!("{e:#}")) },
    })
    .await;
    Json(Status { client, server })
}

// --- POST /api/solve -------------------------------------------------------------

#[derive(Deserialize)]
struct SolveRequest {
    facelets: String,
}

#[derive(Serialize)]
struct SolveResponse {
    moves: Vec<String>,
    stats: SolveStats,
}

#[derive(Serialize)]
struct SolveStats {
    scheme: &'static str,
    output_slots: usize,
    raw_moves: usize,
    moves: usize,
    upload_bytes: usize,
    download_bytes: usize,
    solve_ms: f64,
    network_ms: f64,
    total_ms: f64,
    #[serde(flatten)]
    ckks: Option<CkksStats>,
}

fn solve_blocking(app: &App, facelets: &str) -> Result<SolveResponse, ApiError> {
    let start = Instant::now();
    let state = from_facelets(facelets).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let gateway = |e: anyhow::Error| error(StatusCode::BAD_GATEWAY, e);

    let (raw, ckks_stats, ex): (Vec<Move>, Option<CkksStats>, Exchange) = match (app.scheme, &app.ckks) {
        (Scheme::Ckks, Some(ckks)) => {
            let colors = facelet_colors(facelets).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
            let (tokens, stats, ex) = ckks.solve(&app.server_addr, &colors).map_err(gateway)?;
            (tokens_to_moves(&tokens), Some(stats), ex)
        }
        _ => {
            let request =
                Request::Solve { scheme: Scheme::Plaintext, key_id: 0, inputs: vec![plaintext::encode_state(&state)] };
            let ex = exchange(&app.server_addr, &request).map_err(gateway)?;
            let moves = match &ex.response {
                Response::Solution { outputs, .. } => {
                    plaintext::decode_moves(&outputs[0]).map_err(|e| gateway(e.into()))?
                }
                Response::Error(e) => return Err(gateway(anyhow::anyhow!("server: {e}"))),
                _ => return Err(gateway(anyhow::anyhow!("unexpected response"))),
            };
            (moves.into_iter().flatten().collect(), None, ex)
        }
    };
    let Response::Solution { outputs, solve_ms, .. } = &ex.response else { unreachable!() };
    if !state.apply_all(&raw).is_solved() {
        return Err(gateway(anyhow::anyhow!("the decrypted solution does not solve the cube")));
    }
    let moves = simplify(&raw);
    let output_slots = match app.scheme {
        Scheme::Ckks => app.ckks.as_ref().and_then(|c| c.keyed()).map_or(0, |k| k.output_tokens),
        Scheme::Plaintext => outputs[0].len(),
    };
    let stats = SolveStats {
        scheme: app.scheme.name(),
        output_slots,
        raw_moves: raw.len(),
        moves: moves.len(),
        upload_bytes: ex.sent,
        download_bytes: ex.received,
        solve_ms: *solve_ms,
        network_ms: (ex.elapsed_ms - solve_ms).max(0.0),
        total_ms: ms(start),
        ckks: ckks_stats,
    };
    eprintln!("solved {facelets} in {} moves ({:.0} ms)", moves.len(), stats.total_ms);
    Ok(SolveResponse { moves: moves.iter().map(Move::to_string).collect(), stats })
}

async fn solve(State(app): State<Arc<App>>, Json(req): Json<SolveRequest>) -> Result<Json<SolveResponse>, ApiError> {
    blocking(move || solve_blocking(&app, &req.facelets)).await.map(Json)
}

// --- GET /api/progress -----------------------------------------------------------

#[derive(Serialize)]
struct Progress {
    step: &'static str,
    server: Option<SolveProgress>,
}

async fn progress(State(app): State<Arc<App>>) -> Json<Progress> {
    let step = app.ckks.as_ref().map_or("idle", |c| *c.step.lock().unwrap());
    let server = if step == "evaluating" {
        let addr = app.server_addr.clone();
        blocking(move || match exchange(&addr, &Request::Progress) {
            Ok(Exchange { response: Response::Progress(p), .. }) => p,
            _ => None,
        })
        .await
    } else {
        None
    };
    Json(Progress { step, server })
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_CLIENT_ADDR.to_string());
    let server_addr = arg("--server").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());
    let scheme = match arg("--scheme").as_deref() {
        None | Some("ckks") => Scheme::Ckks,
        Some("plaintext") => Scheme::Plaintext,
        Some(other) => bail!("unknown scheme {other:?} (ckks or plaintext)"),
    };
    let ckks = (scheme == Scheme::Ckks).then(|| -> Result<_> {
        let params = match arg("--params").as_deref() {
            Some("test") => Params::insecure_test(12),
            _ => {
                if !std::env::args().any(|a| a == "--allow-low-memory") {
                    rubikfhe_ckks::system::ensure_memory(rubikfhe_ckks::system::CLIENT_MIN_MEMORY)?;
                }
                Params::secure()?
            }
        };
        let threads = rubikfhe_ckks::system::init_threads();
        eprintln!(
            "{} backend ({threads} threads), {} parameters: preparing keys",
            rubikfhe_ckks::backend::NAME,
            params.name
        );
        Ok(CkksClient::start(params, server_addr.clone()))
    });
    let ckks = ckks.transpose()?;

    let app = Arc::new(App { scheme, server_addr, ckks });
    let router = Router::new()
        .route("/api/status", get(status))
        .route("/api/solve", post(solve))
        .route("/api/progress", get(progress))
        .with_state(app.clone());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}, server at {}, scheme {}", app.server_addr, app.scheme.name());
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await?;
    Ok(())
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
