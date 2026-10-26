//! Messages exchanged between the client and the server.
//!
//! Each message is a postcard-encoded frame prefixed by its length as a
//! little-endian u32. Cube states and solutions travel as opaque byte strings
//! whose meaning depends on the scheme: the server never needs more than the
//! evaluation keys and the ciphertexts. Evaluation keys, several gigabytes
//! with bootstrapping, follow an [`Request::UploadKeys`] frame as a raw stream.

use std::io::{Read, Write};

use rubikfhe_cube::{CubeError, CubieState, Move, STATE_BYTES};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 2;
pub const DEFAULT_CLIENT_ADDR: &str = "127.0.0.1:8787";
pub const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8788";

const MAX_FRAME_BYTES: usize = 1 << 30;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// No encryption: a plaintext Thistlethwaite solve, for comparison.
    Plaintext,
    /// CKKS: the oblivious stabilizer-chain circuit on encrypted stickers.
    Ckks,
}

impl Scheme {
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Plaintext => "plaintext",
            Scheme::Ckks => "ckks",
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Request {
    Info,
    /// Followed on the same connection by the evaluation-key stream.
    UploadKeys,
    /// `inputs` are scheme-specific byte strings: the plaintext state, or the
    /// packed input ciphertexts.
    Solve {
        scheme: Scheme,
        key_id: u64,
        inputs: Vec<Vec<u8>>,
    },
    /// Where the running solve is, asked on a separate connection.
    Progress,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Info(ServerInfo),
    KeysInstalled { key_id: u64, bytes: u64, receive_ms: f64, prepare_ms: f64 },
    Solution { outputs: Vec<Vec<u8>>, solve_ms: f64, refreshes: u32, cleanings: u32 },
    Progress(Option<SolveProgress>),
    Error(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Decoding,
    Stage,
    Bootstrapping,
    Cleaning,
    Output,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SolveProgress {
    pub stage: u32,
    pub stages: u32,
    pub step: Step,
    /// Estimated fraction of the evaluation done, from the cost of each step
    /// and the bootstraps and cleanings of the previous solve.
    pub fraction: f64,
    pub elapsed_ms: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerInfo {
    pub version: u32,
    /// The poulpy backend the server was built with.
    pub backend: String,
    pub ckks_params: String,
    /// Identifier of the installed evaluation keys, if any.
    pub key_id: Option<u64>,
    pub stages: u32,
    pub output_tokens: u32,
}

/// Writes one frame and returns its size on the wire.
pub fn write_frame<W: Write>(w: &mut W, msg: &impl Serialize) -> std::io::Result<usize> {
    let body = postcard::to_stdvec(msg).map_err(std::io::Error::other)?;
    let len = u32::try_from(body.len()).map_err(std::io::Error::other)?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()?;
    Ok(4 + body.len())
}

/// Reads one frame and returns it with its size on the wire.
pub fn read_frame<R: Read, T: DeserializeOwned>(r: &mut R) -> std::io::Result<(T, usize)> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(std::io::Error::other(format!("frame too large: {len} bytes")));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    let msg = postcard::from_bytes(&body).map_err(std::io::Error::other)?;
    Ok((msg, 4 + len))
}

/// A writer that counts the bytes going through it.
pub struct Counting<W> {
    pub inner: W,
    pub bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<W: Write> Write for Counting<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Encodings used by the plaintext scheme.
pub mod plaintext {
    use super::*;

    /// Marks an unused move slot.
    pub const NO_MOVE: u8 = 18;

    pub fn encode_state(s: &CubieState) -> Vec<u8> {
        s.to_bytes().to_vec()
    }

    pub fn decode_state(bytes: &[u8]) -> Result<CubieState, CubeError> {
        debug_assert_eq!(bytes.len(), STATE_BYTES);
        CubieState::from_bytes(bytes)
    }

    pub fn encode_moves(slots: &[Option<Move>]) -> Vec<u8> {
        slots.iter().map(|m| m.map_or(NO_MOVE, Move::index)).collect()
    }

    pub fn decode_moves(bytes: &[u8]) -> Result<Vec<Option<Move>>, CubeError> {
        bytes
            .iter()
            .map(|&b| match b {
                NO_MOVE => Ok(None),
                _ => Move::from_index(b).map(Some).ok_or_else(|| CubeError::InvalidMove(format!("slot {b}"))),
            })
            .collect()
    }
}
