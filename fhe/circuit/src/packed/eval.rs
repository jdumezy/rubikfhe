//! Evaluation of the packed circuit on any [`Engine`].

use anyhow::Result;

use super::engine::Engine;
use super::layout::{LOG_BLOCK, LOG_SIDE, Orientation};
use super::plan::{PackedPlan, family_index};

/// A [`PackedPlan`] with every plaintext encoded for an engine.
pub struct Prepared<E: Engine> {
    pub log_blocks_per_ct: usize,
    pub state_cts: usize,
    pub token_cts: [usize; 2],
    stages: Vec<PreparedStage<E>>,
}

struct PreparedStage<E: Engine> {
    orientation: Orientation,
    isolate: (usize, E::Pt),
    codes: Vec<(usize, E::Pt)>,
    gather: Vec<E::Lt>,
    keep_state: Vec<E::Pt>,
    keep_next: Option<(usize, E::Pt)>,
}

impl<E: Engine> Prepared<E> {
    pub fn new(engine: &E, plan: &PackedPlan) -> Result<Prepared<E>> {
        let stages = plan
            .stages
            .iter()
            .map(|s| -> Result<PreparedStage<E>> {
                Ok(PreparedStage {
                    orientation: s.orientation,
                    isolate: (s.isolate.0, engine.encode(&s.isolate.1)?),
                    codes: s.codes.iter().map(|(c, v)| Ok((*c, engine.encode(v)?))).collect::<Result<_>>()?,
                    gather: s.gather.iter().map(|d| engine.encode_lt(d)).collect::<Result<_>>()?,
                    keep_state: s.keep_state.iter().map(|v| engine.encode(v)).collect::<Result<_>>()?,
                    keep_next: s
                        .keep_next
                        .as_ref()
                        .map(|(c, v)| Ok::<_, anyhow::Error>((*c, engine.encode(v)?)))
                        .transpose()?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Prepared {
            log_blocks_per_ct: plan.layout.log_blocks_per_ct(),
            state_cts: plan.state_cts,
            token_cts: plan.token_cts,
            stages,
        })
    }
}

/// Encrypted outputs: the folded token ciphertexts of both families.
pub struct Outputs<C> {
    pub cols: Vec<C>,
    pub rows: Vec<C>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub refreshes: usize,
    pub cleanings: usize,
}

/// Errors on the one-hot vectors double every stage (the selector and the
/// state carry the same error into a bilinear map), so the state is cleaned
/// this often, unless the engine's refreshes clean it.
pub const CLEAN_EVERY: usize = 4;

/// Smoothstep `3x² − 2x³`: fixes 0 and 1 with zero slope, squaring small errors.
fn clean<E: Engine>(engine: &E, x: &E::Ct) -> Result<E::Ct> {
    let x2 = engine.mul(x, x)?;
    let x3 = engine.mul(&x2, x)?;
    let mut h = engine.copy(&x2)?;
    engine.add_assign(&mut h, &x2)?;
    engine.add_assign(&mut h, &x2)?;
    engine.sub_assign(&mut h, &x3)?;
    engine.sub_assign(&mut h, &x3)?;
    Ok(h)
}

/// Runs the circuit on the client's packed inputs (see [`PackedPlan::pack_input`]).
pub fn evaluate<E: Engine>(
    engine: &E,
    plan: &Prepared<E>,
    u: &[E::Ct],
    v: &[E::Ct],
) -> Result<(Outputs<E::Ct>, Report)> {
    evaluate_inspected(engine, plan, u, v, |_| {})
}

/// What the evaluator is doing, for progress reports and tests.
pub enum Event<'a, C> {
    /// Bootstrapping the state before `stage`.
    Refresh { stage: usize },
    /// Cleaning the noise of the state before `stage`.
    Clean { stage: usize },
    /// `stage` starts, its selector copied into every block.
    Stage { stage: usize, selector: &'a C, state: &'a [C] },
}

/// [`evaluate`], reporting every [`Event`] to `inspect`.
pub fn evaluate_inspected<E: Engine>(
    engine: &E,
    plan: &Prepared<E>,
    u: &[E::Ct],
    v: &[E::Ct],
    mut inspect: impl FnMut(Event<'_, E::Ct>),
) -> Result<(Outputs<E::Ct>, Report)> {
    let mut report = Report::default();
    let mut state: Vec<E::Ct> = u.iter().zip(v).map(|(a, b)| engine.mul(a, b)).collect::<Result<_>>()?;
    let mut next_selector: Option<E::Ct> = None;
    let mut acc: [Vec<Option<E::Ct>>; 2] = plan.token_cts.map(|n| (0..n).map(|_| None).collect());

    for (i, stage) in plan.stages.iter().enumerate() {
        let o = stage.orientation;
        let cleaning = !engine.refresh_cleans() && i > 0 && i % CLEAN_EVERY == 0 && !stage.gather.is_empty();
        // The last stage only emits tokens: one plaintext product of the selector.
        let stage_cost = if stage.gather.is_empty() { engine.plaintext_cost() } else { engine.stage_cost() };
        let cost = stage_cost + if cleaning { engine.clean_cost() } else { 0 };
        if engine.log_budget(&state[0]) < cost + engine.refresh_reserve() {
            inspect(Event::Refresh { stage: i });
            state = state.iter().map(|ct| engine.refresh(ct)).collect::<Result<_>>()?;
            next_selector = None;
            report.refreshes += 1;
        }
        if cleaning {
            inspect(Event::Clean { stage: i });
            state = state.iter().map(|ct| clean(engine, ct)).collect::<Result<_>>()?;
            next_selector = None;
            report.cleanings += 1;
        }

        // The selector, copied into every block of a ciphertext.
        let isolated = match next_selector.take() {
            Some(s) => s,
            None => engine.mul_pt(&state[stage.isolate.0], &stage.isolate.1)?,
        };
        let selector = if plan.log_blocks_per_ct > 0 {
            engine.inner_sum(&isolated, LOG_BLOCK, plan.log_blocks_per_ct)?
        } else {
            isolated
        };
        inspect(Event::Stage { stage: i, selector: &selector, state: &state });

        for (c, codes) in &stage.codes {
            let tokens = engine.mul_pt(&selector, codes)?;
            match &mut acc[family_index(o)][*c] {
                Some(a) => engine.add_assign(a, &tokens)?,
                slot => *slot = Some(tokens),
            }
        }

        if stage.gather.is_empty() {
            continue;
        }
        let mut next_state = Vec::with_capacity(state.len());
        for (c, x) in state.iter().enumerate() {
            let transition = engine.apply_lt(&selector, &stage.gather[c])?;
            let products = engine.mul(&transition, x)?;
            let folded = engine.inner_sum(&products, o.log_batch(), LOG_SIDE)?;
            let kept = engine.mul_pt(&folded, &stage.keep_state[c])?;
            next_state.push(engine.replicate(&kept, o.log_batch(), LOG_SIDE)?);
            if let Some((sc, keep)) = &stage.keep_next
                && *sc == c
            {
                let kept = engine.mul_pt(&folded, keep)?;
                next_selector = Some(engine.replicate(&kept, o.log_batch(), LOG_SIDE)?);
            }
        }
        state = next_state;
    }

    let fold = |family: Orientation, cts: Vec<Option<E::Ct>>| -> Result<Vec<E::Ct>> {
        cts.into_iter()
            .map(|ct| {
                engine.inner_sum(&ct.expect("every token ciphertext receives codes"), family.log_batch(), LOG_SIDE)
            })
            .collect()
    };
    let [cols, rows] = acc;
    Ok((Outputs { cols: fold(Orientation::Cols, cols)?, rows: fold(Orientation::Rows, rows)? }, report))
}
