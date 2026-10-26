//! Stabilizer chain of the cube group for a prescribed base (Schreier–Sims).

use std::collections::BTreeMap;

use crate::perm::Perm;

/// Level `i` of the chain: the group `H_i` fixing `base[..i]`, through a
/// strong generating set, and the orbit of `base[i]` under it.
pub struct Level {
    pub base: u8,
    pub gens: Vec<Perm>,
    /// For each orbit point `a`, a transversal element `u ∈ H_i` with `u(base) = a`.
    pub transversal: BTreeMap<u8, Perm>,
}

impl Level {
    fn recompute_orbit(&mut self) {
        self.transversal.clear();
        self.transversal.insert(self.base, Perm::IDENTITY);
        let mut queue = vec![self.base];
        while let Some(p) = queue.pop() {
            let u = self.transversal[&p];
            for g in &self.gens {
                let q = g.image(p);
                if let std::collections::btree_map::Entry::Vacant(e) = self.transversal.entry(q) {
                    e.insert(g.after(&u));
                    queue.push(q);
                }
            }
        }
    }
}

pub struct StabChain {
    pub levels: Vec<Level>,
}

impl StabChain {
    /// Deterministic Schreier–Sims. Panics if `base` is not a base of `<gens>`.
    pub fn new(gens: &[Perm], base: &[u8]) -> StabChain {
        let mut levels: Vec<Level> =
            base.iter().map(|&b| Level { base: b, gens: Vec::new(), transversal: BTreeMap::new() }).collect();
        for g in gens {
            for (i, level) in levels.iter_mut().enumerate() {
                if g.fixes(&base[..i]) {
                    level.gens.push(*g);
                }
            }
        }
        for level in &mut levels {
            level.recompute_orbit();
        }

        let mut chain = StabChain { levels };
        let mut i = base.len();
        'outer: while i > 0 {
            let level = i - 1;
            let schreier: Vec<Perm> = {
                let l = &chain.levels[level];
                l.transversal
                    .values()
                    .flat_map(|u| {
                        l.gens.iter().map(move |s| {
                            let su = s.after(u);
                            l.transversal[&su.image(l.base)].inverse().after(&su)
                        })
                    })
                    .collect()
            };
            for g in schreier {
                let (h, j) = chain.strip(g, level + 1);
                if !h.is_identity() {
                    assert!(j < chain.levels.len(), "the given points do not form a base");
                    for l in level + 1..=j {
                        chain.levels[l].gens.push(h);
                        chain.levels[l].recompute_orbit();
                    }
                    i = j + 1;
                    continue 'outer;
                }
            }
            i -= 1;
        }
        chain
    }

    /// Sifts `g` through the levels from `from`; returns the residue and the
    /// level where sifting stopped (`levels.len()` if it went through).
    fn strip(&self, mut g: Perm, from: usize) -> (Perm, usize) {
        for (i, level) in self.levels.iter().enumerate().skip(from) {
            match level.transversal.get(&g.image(level.base)) {
                Some(u) => g = u.inverse().after(&g),
                None => return (g, i),
            }
        }
        (g, self.levels.len())
    }

    /// Group order: the product of the orbit sizes.
    pub fn order(&self) -> u128 {
        self.levels.iter().map(|l| l.transversal.len() as u128).product()
    }

    pub fn orbit_sizes(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.transversal.len()).collect()
    }
}
