//! rubikfhe server: receives an evaluation key and an encrypted cube state,
//! computes an encrypted solution. It never holds a secret key.

use std::sync::Arc;
use std::time::Instant;

use rubikfhe_protocol::{
    DEFAULT_SERVER_ADDR, PROTOCOL_VERSION, Request, Response, Scheme, ServerInfo, plaintext, read_frame, write_frame,
};
use rubikfhe_solver::Thistlethwaite;
use tokio::net::{TcpListener, TcpStream};

struct Server {
    solver: Thistlethwaite,
}

impl Server {
    fn info(&self) -> ServerInfo {
        ServerInfo {
            version: PROTOCOL_VERSION,
            schemes: vec![Scheme::Plaintext],
            solver: "thistlethwaite".into(),
            phases: self.solver.phases.iter().map(|p| p.name.to_string()).collect(),
            phase_slots: self.phase_slots(),
        }
    }

    fn phase_slots(&self) -> Vec<u32> {
        self.solver.phases.iter().map(|p| p.max_depth as u32).collect()
    }

    fn solve(&self, scheme: Scheme, _eval_key: &[u8], state: &[u8]) -> Response {
        let start = Instant::now();
        let moves = match scheme {
            Scheme::Plaintext => match plaintext::decode_state(state) {
                Ok(s) => plaintext::encode_moves(&self.solver.solve_padded(&s)),
                Err(e) => return Response::Error(e.to_string()),
            },
        };
        Response::Solution { moves, phase_slots: self.phase_slots(), solve_micros: start.elapsed().as_micros() as u64 }
    }
}

async fn handle(server: Arc<Server>, mut stream: TcpStream) -> std::io::Result<()> {
    let peer = stream.peer_addr()?;
    loop {
        let (request, size) = match read_frame::<_, Request>(&mut stream).await {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        let response = match request {
            Request::Info => Response::Info(server.info()),
            Request::Solve { scheme, eval_key, state } => {
                let server = server.clone();
                let response = tokio::task::spawn_blocking(move || server.solve(scheme, &eval_key, &state))
                    .await
                    .map_err(std::io::Error::other)?;
                if let Response::Solution { solve_micros, .. } = &response {
                    eprintln!("{peer}: {} solve, {size} B in, {solve_micros} µs", scheme.name());
                }
                response
            }
        };
        write_frame(&mut stream, &response).await?;
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());

    let start = Instant::now();
    let server = Arc::new(Server { solver: Thistlethwaite::new() });
    let sizes: Vec<String> = server.solver.phases.iter().map(|p| p.size().to_string()).collect();
    eprintln!("solver tables ready in {:.2?} (cosets per phase: {})", start.elapsed(), sizes.join(", "));

    let listener = TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}");
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let server = server.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle(server, stream).await {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            _ = tokio::signal::ctrl_c() => return Ok(()),
        }
    }
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
