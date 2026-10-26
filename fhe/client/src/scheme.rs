//! Client side of an encryption scheme: owns the secret key.

use rubikfhe_cube::{CubeError, CubieState, Move};
use rubikfhe_protocol::{Scheme, plaintext};

pub trait ClientScheme: Send + Sync {
    fn scheme(&self) -> Scheme;

    /// Public key material the server needs to compute on ciphertexts.
    fn eval_key(&self) -> &[u8];

    fn encrypt_state(&self, state: &CubieState) -> Vec<u8>;

    fn decrypt_moves(&self, ciphertext: &[u8]) -> Result<Vec<Option<Move>>, CubeError>;
}

/// No encryption at all; stands in until the FHE scheme is wired in.
pub struct Plaintext;

impl ClientScheme for Plaintext {
    fn scheme(&self) -> Scheme {
        Scheme::Plaintext
    }

    fn eval_key(&self) -> &[u8] {
        &[]
    }

    fn encrypt_state(&self, state: &CubieState) -> Vec<u8> {
        plaintext::encode_state(state)
    }

    fn decrypt_moves(&self, ciphertext: &[u8]) -> Result<Vec<Option<Move>>, CubeError> {
        plaintext::decode_moves(ciphertext)
    }
}
