//! Noether's theorem — symmetry implies conservation.
//!
//! For every continuous symmetry of the action there exists a conserved
//! quantity.  This module provides tools to:
//!
//! 1. Define symmetry transformations.
//! 2. Numerically test whether a Lagrangian is invariant.
//! 3. Compute the associated conserved Noether charge.

use crate::lagrangian::{AgentState, Lagrangian};
use crate::Scalar;

/// A continuous one-parameter symmetry transformation `q ↦ q'(q, ε)`.
///
/// The user supplies the transformed coordinates and (optionally) velocities.
pub trait Symmetry<S: Scalar, const N: usize> {
    /// Apply the transformation with parameter `epsilon`.
    fn transform(&self, state: &AgentState<S, N>, epsilon: S) -> AgentState<S, N>;
    /// Infinitesimal generator `δqᵢ = ∂q'/∂ε |_{ε=0}`.
    fn generator_q(&self, state: &AgentState<S, N>) -> [S; N];
    /// Name of the symmetry for diagnostics.
    fn name(&self) -> &'static str;

    /// Surface term `F` such that the Lagrangian changes by a total derivative
    /// `δL = ε dF/dt` under the symmetry.
    ///
    /// For internal symmetries (translation, rotation, ...) `δL = 0` and this
    /// returns zero. For time translation `F = L`, so the full conserved
    /// Noether charge becomes `Σ pᵢ δqᵢ − F = H`, i.e. the energy.
    fn surface_term(&self, _lagrangian: &dyn Lagrangian<S, N>, _state: &AgentState<S, N>) -> S {
        S::zero()
    }
}

/// Spatial translation symmetry along axis `axis`.
pub struct TranslationSymmetry<const N: usize> {
    pub axis: usize,
}

impl<S: Scalar, const N: usize> Symmetry<S, N> for TranslationSymmetry<N> {
    fn transform(&self, state: &AgentState<S, N>, epsilon: S) -> AgentState<S, N> {
        let mut q = state.q;
        q[self.axis] = q[self.axis] + epsilon;
        AgentState::new(q, state.q_dot)
    }

    fn generator_q(&self, _state: &AgentState<S, N>) -> [S; N] {
        let mut gen = [S::zero(); N];
        gen[self.axis] = S::one();
        gen
    }

    fn name(&self) -> &'static str {
        "translation"
    }
}

/// Time-translation symmetry `t ↦ t + ε`.
///
/// Implemented by evolving the state forward by `ε` using the equations of
/// motion.  For autonomous systems this is always a symmetry.
pub struct TimeTranslationSymmetry;

impl<S: Scalar, const N: usize> Symmetry<S, N> for TimeTranslationSymmetry {
    fn transform(&self, state: &AgentState<S, N>, epsilon: S) -> AgentState<S, N> {
        // First-order approximation: q(t+ε) ≈ q + ε q̇.
        // We keep q̇ unchanged because the trait does not have access to the
        // equations of motion needed to compute q̈; this is sufficient for the
        // generator and the surface-term correction below.
        let mut q_new = [S::zero(); N];
        for (i, q_new_i) in q_new.iter_mut().enumerate().take(N) {
            *q_new_i = state.q[i] + epsilon * state.q_dot[i];
        }
        AgentState::new(q_new, state.q_dot)
    }

    fn generator_q(&self, state: &AgentState<S, N>) -> [S; N] {
        state.q_dot
    }

    fn name(&self) -> &'static str {
        "time_translation"
    }

    fn surface_term(&self, lagrangian: &dyn Lagrangian<S, N>, state: &AgentState<S, N>) -> S {
        // For time translation δL = ε dL/dt, so F = L and the conserved
        // Noether charge is Q = Σ pᵢ δqᵢ − L = H = T + V.
        lagrangian.lagrangian(state)
    }
}

/// Rotation symmetry in the `(i, j)` plane by angle `ε`.
pub struct RotationSymmetry {
    pub i: usize,
    pub j: usize,
}

impl<S: Scalar, const N: usize> Symmetry<S, N> for RotationSymmetry {
    fn transform(&self, state: &AgentState<S, N>, epsilon: S) -> AgentState<S, N> {
        let cos = epsilon.cos();
        let sin = epsilon.sin();
        let mut q = state.q;
        let mut q_dot = state.q_dot;
        let qi = q[self.i];
        let qj = q[self.j];
        q[self.i] = cos * qi - sin * qj;
        q[self.j] = sin * qi + cos * qj;
        let vi = q_dot[self.i];
        let vj = q_dot[self.j];
        q_dot[self.i] = cos * vi - sin * vj;
        q_dot[self.j] = sin * vi + cos * vj;
        AgentState::new(q, q_dot)
    }

    fn generator_q(&self, state: &AgentState<S, N>) -> [S; N] {
        let mut gen = [S::zero(); N];
        gen[self.i] = -state.q[self.j];
        gen[self.j] = state.q[self.i];
        gen
    }

    fn name(&self) -> &'static str {
        "rotation"
    }
}

/// Result of a Noether invariance test.
#[derive(Debug, Clone, PartialEq)]
pub struct InvarianceResult<S: Scalar> {
    pub invariant: bool,
    pub delta_lagrangian: S,
    pub epsilon: S,
}

/// Numerically test whether a Lagrangian is invariant under a symmetry.
///
/// Returns `InvarianceResult` with `invariant = true` when
/// `|L(q') − L(q)| < tolerance` for the given `epsilon`.
pub fn test_invariance<S: Scalar, const N: usize, L: Lagrangian<S, N>>(
    lagrangian: &L,
    symmetry: &dyn Symmetry<S, N>,
    state: &AgentState<S, N>,
    epsilon: S,
    tolerance: S,
) -> InvarianceResult<S> {
    let l0 = lagrangian.lagrangian(state);
    let state_prime = symmetry.transform(state, epsilon);
    let l_prime = lagrangian.lagrangian(&state_prime);
    let delta = (l_prime - l0).abs();
    InvarianceResult {
        invariant: delta < tolerance,
        delta_lagrangian: delta,
        epsilon,
    }
}

/// Compute the Noether charge `Q = Σᵢ pᵢ δqᵢ` where `pᵢ = ∂L/∂q̇ᵢ`.
///
/// For a mechanical Lagrangian `L = ½m q̇² − V(q)` this reduces to
/// `Q = m Σᵢ q̇ᵢ δqᵢ`.
pub fn noether_charge<S: Scalar, const N: usize>(
    mass: S,
    state: &AgentState<S, N>,
    generator_q: &[S; N],
) -> S {
    let mut q = S::zero();
    for (i, &gen_i) in generator_q.iter().enumerate().take(N) {
        q = q + mass * state.q_dot[i] * gen_i;
    }
    q
}

/// A conserved-quantity monitor that tracks a charge along a trajectory.
#[derive(Debug, Clone)]
pub struct ChargeMonitor<S: Scalar> {
    pub values: Vec<S>,
    pub tolerance: S,
}

impl<S: Scalar> ChargeMonitor<S> {
    pub fn new(tolerance: S) -> Self {
        Self {
            values: Vec::new(),
            tolerance,
        }
    }

    pub fn push(&mut self, value: S) {
        self.values.push(value);
    }

    /// Return `true` if the charge stays within tolerance of its initial value.
    pub fn is_conserved(&self) -> bool {
        if self.values.is_empty() {
            return true;
        }
        let q0 = self.values[0];
        self.values.iter().all(|&v| (v - q0).abs() < self.tolerance)
    }

    pub fn max_drift(&self) -> S {
        if self.values.is_empty() {
            return S::zero();
        }
        let q0 = self.values[0];
        self.values
            .iter()
            .map(|&v| (v - q0).abs())
            .fold(S::zero(), S::max)
    }
}

/// Verify Noether's theorem for a given system, symmetry, and trajectory:
///
/// 1. Check Lagrangian invariance (within tolerance).
/// 2. Compute the Noether charge at each step.
/// 3. Assert the charge is conserved.
pub fn verify_noether<S: Scalar, const N: usize, L: Lagrangian<S, N>>(
    lagrangian: &L,
    symmetry: &dyn Symmetry<S, N>,
    trajectory: &[AgentState<S, N>],
    mass: S,
    epsilon: S,
    tolerance: S,
) -> Result<ChargeMonitor<S>, String> {
    let inv = test_invariance(lagrangian, symmetry, &trajectory[0], epsilon, tolerance);
    if !inv.invariant {
        return Err(format!(
            "Lagrangian not invariant under {} (ΔL = {:?})",
            symmetry.name(),
            inv.delta_lagrangian
        ));
    }

    let mut monitor = ChargeMonitor::new(tolerance);
    for state in trajectory {
        let gen = symmetry.generator_q(state);
        let q = noether_charge(mass, state, &gen) - symmetry.surface_term(lagrangian, state);
        monitor.push(q);
    }

    if !monitor.is_conserved() {
        return Err(format!(
            "Noether charge not conserved for {} (max drift = {:?})",
            symmetry.name(),
            monitor.max_drift()
        ));
    }

    Ok(monitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::{total_energy, MechanicalLagrangian, SymplecticIntegrator};
    use approx::assert_relative_eq;

    #[test]
    fn translation_invariance_free_particle() {
        let potential = |_: &[f64; 2]| 0.0_f64;
        let lagrangian = MechanicalLagrangian {
            mass: 1.0,
            potential_fn: potential,
        };
        let state = AgentState::new([1.0, 2.0], [3.0, 4.0]);
        let sym = TranslationSymmetry::<2> { axis: 0 };

        let inv = test_invariance(&lagrangian, &sym, &state, 1e-3, 1e-10);
        assert!(
            inv.invariant,
            "free particle should be translation invariant"
        );
    }

    #[test]
    fn translation_not_invariant_harmonic() {
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass: 1.0,
            potential_fn: potential,
        };
        let state = AgentState::new([1.0], [0.0]);
        let sym = TranslationSymmetry::<1> { axis: 0 };

        let inv = test_invariance(&lagrangian, &sym, &state, 1e-3, 1e-10);
        assert!(!inv.invariant, "harmonic potential breaks translation");
    }

    #[test]
    fn rotation_invariance_central_potential() {
        let potential = |q: &[f64; 2]| {
            let r2 = q[0] * q[0] + q[1] * q[1];
            r2.sqrt() // V(r) = r
        };
        let lagrangian = MechanicalLagrangian {
            mass: 2.0,
            potential_fn: potential,
        };
        let state = AgentState::new([1.0, 0.5], [-0.2, 0.3]);
        let sym = RotationSymmetry { i: 0, j: 1 };

        let inv = test_invariance(&lagrangian, &sym, &state, 1e-4, 1e-10);
        assert!(
            inv.invariant,
            "central potential should be rotation invariant"
        );
    }

    #[test]
    fn angular_momentum_conservation() {
        // Central potential → angular momentum conserved
        let m = 1.0_f64;
        let potential = |q: &[f64; 2]| {
            let r2 = q[0] * q[0] + q[1] * q[1];
            0.5 * r2
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0, 0.0], [0.0, 1.0]);
        let traj = integrator.integrate(m, &potential, &initial, 5000).unwrap();

        let sym = RotationSymmetry { i: 0, j: 1 };
        let monitor = verify_noether(
            &MechanicalLagrangian {
                mass: m,
                potential_fn: potential,
            },
            &sym,
            &traj,
            m,
            1e-6,
            1e-8,
        )
        .expect("Noether verification should pass");

        assert!(monitor.max_drift() < 1e-6);
    }

    #[test]
    fn linear_momentum_conservation_free_particle() {
        let m = 2.0_f64;
        let potential = |_: &[f64; 3]| 0.0_f64;
        let dt = 0.01;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        let traj = integrator.integrate(m, &potential, &initial, 100).unwrap();

        let sym = TranslationSymmetry::<3> { axis: 1 };
        let monitor = verify_noether(
            &MechanicalLagrangian {
                mass: m,
                potential_fn: potential,
            },
            &sym,
            &traj,
            m,
            1e-6,
            1e-10,
        )
        .expect("Noether verification should pass");

        assert!(monitor.is_conserved());
        // Charge should equal p_y = m * v_y
        assert_relative_eq!(monitor.values[0], m * 2.0, epsilon = 1e-10);
    }

    #[test]
    fn energy_conservation_via_time_translation() {
        let m = 1.0_f64;
        let k = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass: m,
            potential_fn: potential,
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0], [0.0]);
        let traj = integrator.integrate(m, &potential, &initial, 1000).unwrap();

        // Energy is the Noether charge for time translation
        let mut e_monitor = ChargeMonitor::new(1e-5);
        for state in &traj {
            e_monitor.push(total_energy(&lagrangian, state));
        }
        assert!(e_monitor.is_conserved());
    }

    #[test]
    fn verify_noether_time_translation_yields_energy() {
        // Regression check: TimeTranslationSymmetry must produce the Hamiltonian
        // H = T + V as the conserved Noether charge, not 2T.
        let m = 1.0_f64;
        let k = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass: m,
            potential_fn: potential,
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        // Start away from the turning point so q_dot is non-zero along the orbit.
        let initial = AgentState::new([0.8], [0.6]);
        let traj = integrator.integrate(m, &potential, &initial, 2000).unwrap();

        let e0 = total_energy(&lagrangian, &initial);
        let sym = TimeTranslationSymmetry;
        let monitor = verify_noether(&lagrangian, &sym, &traj, m, 1e-4, 1e-4)
            .expect("energy should be conserved under time translation");

        assert!(
            monitor.is_conserved(),
            "Noether charge for time translation should be conserved (max drift = {})",
            monitor.max_drift()
        );
        // The conserved charge must equal the physical energy E = T + V.
        assert_relative_eq!(monitor.values[0], e0, epsilon = 1e-10);
    }

    #[test]
    fn verify_noether_rejects_non_invariant_symmetry() {
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass: 1.0,
            potential_fn: potential,
        };
        let initial = AgentState::new([1.0], [0.0]);
        let traj = vec![initial.clone()];
        let sym = TranslationSymmetry::<1> { axis: 0 };

        let result = verify_noether(&lagrangian, &sym, &traj, 1.0, 1e-3, 1e-10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not invariant"));
    }

    #[test]
    fn verify_noether_rejects_non_conserved_charge() {
        // Use a free-particle Lagrangian (translation invariant) but supply an
        // invalid trajectory where velocity changes, so the Noether charge
        // (linear momentum) drifts. This exercises the "charge not conserved"
        // error path independently of the "Lagrangian not invariant" path.
        let potential = |_q: &[f64; 1]| 0.0_f64;
        let lagrangian = MechanicalLagrangian {
            mass: 1.0,
            potential_fn: potential,
        };
        let traj = vec![
            AgentState::new([0.0], [1.0]),
            AgentState::new([0.1], [2.0]),
            AgentState::new([0.3], [3.0]),
        ];
        let sym = TranslationSymmetry::<1> { axis: 0 };

        let result = verify_noether(&lagrangian, &sym, &traj, 1.0, 1e-3, 1e-10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not conserved"));
    }

    #[test]
    fn charge_monitor_empty_is_vacuously_conserved() {
        let monitor = ChargeMonitor::<f64>::new(1e-6);
        assert!(monitor.is_conserved());
        assert_eq!(monitor.max_drift(), 0.0);
    }
}
