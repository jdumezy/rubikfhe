//! Byte formats of evaluation keys and ciphertexts between the processes.
//!
//! Keys are written as a stream of length-prefixed blobs, so that several
//! gigabytes of bootstrapping keys never sit in one buffer; the reader
//! allocates every key from the layouts both sides derive from the parameters.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use poulpy_ckks::prelude::*;
use poulpy_core::layouts::{
    GLWEAutomorphismKey, GLWECIEmbedKey, GLWECITraceKey, GLWESwitchingKey, GLWETensorKey, LWEInfos, ModuleCoreAlloc,
};
use poulpy_hal::layouts::{Backend, ReaderFrom, WriterTo};

use crate::backend::Be;
use crate::engine::{Ciphertext, EvalKeys, Ring};

const MAGIC: &[u8; 8] = b"RFHEKEY1";

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_blob<W: Write, T: WriterTo>(w: &mut W, value: &T) -> io::Result<()> {
    let mut bytes = Vec::new();
    value.write_to(&mut bytes)?;
    write_u64(w, bytes.len() as u64)?;
    w.write_all(&bytes)
}

fn read_blob<R: Read, T: ReaderFrom>(r: &mut R, mut value: T) -> io::Result<T> {
    let len = read_u64(r)?;
    let mut blob = r.take(len);
    value.read_from(&mut blob)?;
    if blob.limit() != 0 {
        return Err(invalid("key blob longer than its layout"));
    }
    Ok(value)
}

fn write_flag<W: Write>(w: &mut W, present: bool) -> io::Result<()> {
    w.write_all(&[present as u8])
}

fn read_flag<R: Read>(r: &mut R) -> io::Result<bool> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0] != 0)
}

pub fn write_eval_keys<W: Write>(w: &mut W, keys: &EvalKeys) -> io::Result<()> {
    w.write_all(MAGIC)?;
    write_blob(w, &keys.tensor_key)?;
    write_u64(w, keys.rotation_keys.len() as u64)?;
    for (&g, key) in &keys.rotation_keys {
        write_u64(w, g as u64)?;
        write_blob(w, key)?;
    }
    write_flag(w, keys.bootstrap.is_some())?;
    if let Some(b) = &keys.bootstrap {
        let rotations: BTreeMap<_, _> = b.rotation_keys.iter().collect();
        write_u64(w, rotations.len() as u64)?;
        for (&g, key) in rotations {
            write_u64(w, g as u64)?;
            write_blob(w, key)?;
        }
        write_blob(w, &b.tensor_key)?;
        write_flag(w, b.encapsulation_keys.is_some())?;
        if let Some((dense_to_sparse, sparse_to_dense)) = &b.encapsulation_keys {
            write_blob(w, dense_to_sparse)?;
            write_flag(w, sparse_to_dense.is_some())?;
            if let Some(key) = sparse_to_dense {
                write_blob(w, key)?;
            }
        }
        let first_factor: BTreeMap<_, _> = b.first_factor_keys.iter().collect();
        write_u64(w, first_factor.len() as u64)?;
        for (&rotation, key) in first_factor {
            write_u64(w, rotation as u64)?;
            write_blob(w, key)?;
        }
        write_flag(w, b.conversion_keys.is_some())?;
        if let Some((trace, embed)) = &b.conversion_keys {
            write_blob(w, trace)?;
            write_blob(w, embed)?;
        }
    }
    w.flush()
}

type Buf = <Be as Backend>::OwnedBuf;
type Word = <Be as Backend>::ZnxWord;

/// Receives the keys of a stream one by one, in stream order, so a reader can
/// prepare and drop each before the next arrives.
pub trait KeySink {
    fn tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> anyhow::Result<()>;
    fn rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> anyhow::Result<()>;
    fn bootstrap_rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> anyhow::Result<()>;
    fn bootstrap_tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> anyhow::Result<()>;
    fn encapsulation_keys(
        &mut self,
        dense_to_sparse: GLWESwitchingKey<Buf, Word>,
        sparse_to_dense: Option<GLWESwitchingKey<Buf, Word>>,
    ) -> anyhow::Result<()>;
    fn first_factor_key(&mut self, rotation: i64, key: GLWESwitchingKey<Buf, Word>) -> anyhow::Result<()>;
    fn conversion_keys(
        &mut self,
        trace: GLWECITraceKey<Buf, Word>,
        embed: GLWECIEmbedKey<Buf, Word>,
    ) -> anyhow::Result<()>;
}

/// Reads a key stream, handing every key to `sink` as soon as it is read.
/// Returns whether the stream carried bootstrapping keys.
pub fn read_eval_keys_into<R: Read>(r: &mut R, ring: &Ring, sink: &mut impl KeySink) -> anyhow::Result<bool> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not an evaluation key stream").into());
    }
    let (module, p) = (&ring.module, &ring.params);
    sink.tensor_key(read_blob(r, module.glwe_tensor_key_alloc_from_infos(&p.tensor_key_layout()))?)?;
    for _ in 0..read_u64(r)? {
        let g = read_u64(r)? as i64;
        sink.rotation_key(g, read_blob(r, module.glwe_automorphism_key_alloc_from_infos(&p.rotation_key_layout()))?)?;
    }
    if !read_flag(r)? {
        return Ok(false);
    }
    let Some(preset) = &p.bootstrap else {
        return Err(invalid("bootstrapping keys for parameters without bootstrapping").into());
    };
    let (m, layout) = (module, &preset.keys_layout);
    for _ in 0..read_u64(r)? {
        let g = read_u64(r)? as i64;
        sink.bootstrap_rotation_key(
            g,
            read_blob(r, m.glwe_automorphism_key_alloc_from_infos(&layout.rotation_key(g)))?,
        )?;
    }
    sink.bootstrap_tensor_key(read_blob(r, m.glwe_tensor_key_alloc_from_infos(&layout.tensor_key))?)?;
    if read_flag(r)? {
        let enc = layout.encapsulation.as_ref().ok_or_else(|| invalid("unexpected encapsulation keys"))?;
        let dense_to_sparse = read_blob(r, m.glwe_switching_key_alloc_from_infos(&enc.dense_to_sparse))?;
        let sparse_to_dense = if read_flag(r)? {
            let l = enc.sparse_to_dense.as_ref().ok_or_else(|| invalid("unexpected sparseToDense key"))?;
            Some(read_blob(r, m.glwe_switching_key_alloc_from_infos(l))?)
        } else {
            None
        };
        sink.encapsulation_keys(dense_to_sparse, sparse_to_dense)?;
    }
    for _ in 0..read_u64(r)? {
        let rotation = read_u64(r)? as i64;
        let l = layout.first_factor_key.as_ref().ok_or_else(|| invalid("unexpected first-factor keys"))?;
        sink.first_factor_key(rotation, read_blob(r, m.glwe_switching_key_alloc_from_infos(l))?)?;
    }
    if read_flag(r)? {
        let (trace, embed) = layout.conversion_keys.as_ref().ok_or_else(|| invalid("unexpected conversion keys"))?;
        let trace = read_blob(r, m.glwe_ci_trace_key_alloc_from_infos(trace))?;
        sink.conversion_keys(trace, read_blob(r, m.glwe_ci_embed_key_alloc_from_infos(embed))?)?;
    }
    Ok(true)
}

/// A ciphertext with the width and scale its GLWE bytes do not carry.
pub fn ciphertext_to_bytes(ct: &Ciphertext) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    write_u64(&mut out, ct.k().as_usize() as u64)?;
    write_u64(&mut out, ct.log_delta() as u64)?;
    ct.write_to(&mut out)?;
    Ok(out)
}

pub fn ciphertext_from_bytes(ring: &Ring, mut bytes: &[u8]) -> io::Result<Ciphertext> {
    let k = read_u64(&mut bytes)? as usize;
    let log_delta = read_u64(&mut bytes)? as usize;
    let mut layout = ring.params.glwe_layout();
    if k > layout.k.as_usize() {
        return Err(invalid("ciphertext wider than the parameters allow"));
    }
    layout.k = k.into();
    let mut ct = ring.module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
    ct.read_from(&mut bytes)?;
    ct.set_meta_checked(GLWEPlaintextMeta { scale: Scale::Log(log_delta), slots: SlotsKind::Complex, log_sparsity: 0 })
        .map_err(|e| invalid(&e.to_string()))?;
    Ok(ct)
}
