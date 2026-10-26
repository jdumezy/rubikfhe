//! rubikfhe client: holds the secret key. The browser sends it the cube, it
//! encrypts the state, asks the server for a solution and decrypts the moves.

mod scheme;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rubikfhe_cube::{Move, from_facelets, simplify};
use rubikfhe_protocol::{
    DEFAULT_CLIENT_ADDR, DEFAULT_SERVER_ADDR, Request, Response, ServerInfo, read_frame, write_frame,
};
use scheme::{ClientScheme, Plaintext};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpStream;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

struct App {
    scheme: Box<dyn ClientScheme>,
    server_addr: String,
    keygen_ms: f64,
}

type ApiError = (StatusCode, Json<Value>);

fn error(status: StatusCode, message: impl ToString) -> ApiError {
    (status, Json(json!({ "error": message.to_string() })))
}

/// One request/response round trip with the server, with sizes on the wire.
struct Exchange {
    response: Response,
    sent: usize,
    received: usize,
    elapsed: Duration,
}

async fn exchange(addr: &str, request: &Request) -> std::io::Result<Exchange> {
    let start = Instant::now();
    let mut stream = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr))
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timed out"))??;
    let sent = write_frame(&mut stream, request).await?;
    let (response, received) = read_frame(&mut stream).await?;
    Ok(Exchange { response, sent, received, elapsed: start.elapsed() })
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
    keygen_ms: f64,
    eval_key_bytes: usize,
}

#[derive(Serialize)]
struct ServerStatus {
    addr: String,
    reachable: bool,
    solver: Option<String>,
    phases: Vec<String>,
    phase_slots: Vec<u32>,
    error: Option<String>,
}

async fn status(State(app): State<Arc<App>>) -> Json<Status> {
    let client = ClientStatus {
        version: env!("CARGO_PKG_VERSION"),
        scheme: app.scheme.scheme().name(),
        keygen_ms: app.keygen_ms,
        eval_key_bytes: app.scheme.eval_key().len(),
    };
    let mut server = ServerStatus {
        addr: app.server_addr.clone(),
        reachable: false,
        solver: None,
        phases: vec![],
        phase_slots: vec![],
        error: None,
    };
    match exchange(&app.server_addr, &Request::Info).await {
        Ok(Exchange { response: Response::Info(ServerInfo { solver, phases, phase_slots, .. }), .. }) => {
            server.reachable = true;
            server.solver = Some(solver);
            server.phases = phases;
            server.phase_slots = phase_slots;
        }
        Ok(_) => server.error = Some("unexpected response".into()),
        Err(e) => server.error = Some(e.to_string()),
    }
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
    slots: usize,
    phase_slots: Vec<u32>,
    phase_moves: Vec<usize>,
    raw_moves: usize,
    moves: usize,
    eval_key_bytes: usize,
    state_bytes: usize,
    solution_bytes: usize,
    upload_bytes: usize,
    download_bytes: usize,
    encrypt_ms: f64,
    solve_ms: f64,
    network_ms: f64,
    decrypt_ms: f64,
    total_ms: f64,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

async fn solve(State(app): State<Arc<App>>, Json(req): Json<SolveRequest>) -> Result<Json<SolveResponse>, ApiError> {
    let start = Instant::now();
    let state = from_facelets(&req.facelets).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;

    let t = Instant::now();
    let ciphertext = app.scheme.encrypt_state(&state);
    let encrypt = t.elapsed();
    let state_bytes = ciphertext.len();

    let request =
        Request::Solve { scheme: app.scheme.scheme(), eval_key: app.scheme.eval_key().to_vec(), state: ciphertext };
    let ex = exchange(&app.server_addr, &request)
        .await
        .map_err(|e| error(StatusCode::BAD_GATEWAY, format!("server unreachable at {}: {e}", app.server_addr)))?;
    let (encrypted_moves, phase_slots, solve_micros) = match ex.response {
        Response::Solution { moves, phase_slots, solve_micros } => (moves, phase_slots, solve_micros),
        Response::Error(e) => return Err(error(StatusCode::BAD_GATEWAY, format!("server: {e}"))),
        Response::Info(_) => return Err(error(StatusCode::BAD_GATEWAY, "unexpected response")),
    };

    let t = Instant::now();
    let slots = app.scheme.decrypt_moves(&encrypted_moves).map_err(|e| error(StatusCode::BAD_GATEWAY, e))?;
    let decrypt = t.elapsed();

    let raw: Vec<Move> = slots.iter().flatten().copied().collect();
    if !state.apply_all(&raw).is_solved() {
        return Err(error(StatusCode::BAD_GATEWAY, "the decrypted solution does not solve the cube"));
    }
    let mut offset = 0;
    let phase_moves = phase_slots
        .iter()
        .map(|&n| {
            let used = slots[offset..offset + n as usize].iter().flatten().count();
            offset += n as usize;
            used
        })
        .collect();
    let moves = simplify(&raw);
    let solve = Duration::from_micros(solve_micros);

    let stats = SolveStats {
        scheme: app.scheme.scheme().name(),
        slots: slots.len(),
        phase_slots,
        phase_moves,
        raw_moves: raw.len(),
        moves: moves.len(),
        eval_key_bytes: app.scheme.eval_key().len(),
        state_bytes,
        solution_bytes: encrypted_moves.len(),
        upload_bytes: ex.sent,
        download_bytes: ex.received,
        encrypt_ms: ms(encrypt),
        solve_ms: ms(solve),
        network_ms: ms(ex.elapsed.saturating_sub(solve)),
        decrypt_ms: ms(decrypt),
        total_ms: ms(start.elapsed()),
    };
    eprintln!("solved {} in {} moves ({:.1} ms)", req.facelets, moves.len(), stats.total_ms);
    Ok(Json(SolveResponse { moves: moves.iter().map(Move::to_string).collect(), stats }))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_CLIENT_ADDR.to_string());
    let server_addr = arg("--server").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());

    let t = Instant::now();
    let scheme: Box<dyn ClientScheme> = Box::new(Plaintext);
    let keygen_ms = ms(t.elapsed());
    eprintln!("{} scheme ready, evaluation key: {} B", scheme.scheme().name(), scheme.eval_key().len());

    let app = Arc::new(App { scheme, server_addr, keygen_ms });
    let router =
        Router::new().route("/api/status", get(status)).route("/api/solve", post(solve)).with_state(app.clone());

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}, server at {}", app.server_addr);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
