//! Messages exchanged between the client and the server.
//!
//! Each message is a postcard-encoded frame prefixed by its length as a
//! little-endian u32. Cube states and solutions travel as opaque byte strings
//! whose meaning depends on the scheme: the server never needs more than the
//! evaluation key and the ciphertexts.

use rubikfhe_cube::{CubeError, CubieState, Move, STATE_BYTES};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTOCOL_VERSION: u32 = 1;
pub const DEFAULT_CLIENT_ADDR: &str = "127.0.0.1:8787";
pub const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8788";

/// Frames can carry evaluation keys, which get large under FHE.
const MAX_FRAME_BYTES: usize = 1 << 30;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// No encryption: the "ciphertexts" are the plaintext encodings.
    Plaintext,
}

impl Scheme {
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Plaintext => "plaintext",
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Request {
    Info,
    Solve { scheme: Scheme, eval_key: Vec<u8>, state: Vec<u8> },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Info(ServerInfo),
    Solution {
        moves: Vec<u8>,
        /// Number of move slots used by each phase; public, input independent.
        phase_slots: Vec<u32>,
        solve_micros: u64,
    },
    Error(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerInfo {
    pub version: u32,
    pub schemes: Vec<Scheme>,
    pub solver: String,
    pub phases: Vec<String>,
    pub phase_slots: Vec<u32>,
}

/// Writes one frame and returns its size on the wire.
pub async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, msg: &impl Serialize) -> std::io::Result<usize> {
    let body = postcard::to_stdvec(msg).map_err(std::io::Error::other)?;
    let len = u32::try_from(body.len()).map_err(std::io::Error::other)?;
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await?;
    Ok(4 + body.len())
}

/// Reads one frame and returns it with its size on the wire.
pub async fn read_frame<R: AsyncRead + Unpin, T: DeserializeOwned>(r: &mut R) -> std::io::Result<(T, usize)> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(std::io::Error::other(format!("frame too large: {len} bytes")));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).await?;
    let msg = postcard::from_bytes(&body).map_err(std::io::Error::other)?;
    Ok((msg, 4 + len))
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
