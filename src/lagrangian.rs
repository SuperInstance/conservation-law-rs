//! Lagrangian mechanics for agent dynamics.
//!
//! Implements the Euler–Lagrange equations:
//!
//! ```text
//! d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = 0
//! ```
//!
//! where `L(q, q̇, t) = T(q̇) − V(q)` is the Lagrangian.

use crate::Scalar;
use std::fmt;

/// State of an agent in N-dimensional configuration space.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentState<S, const N: usize> {
    /// Generalised coordinates.
    pub q: [S; N],
    /// Generalised velocities.
    pub q_dot: [S; N],
}

impl<S: Scalar, const N: usize> AgentState<S, N> {
    pub fn new(q: [S; N], q_dot: [S; N]) -> Self {
        Self { q, q_dot }
    }
}

/// A Lagrangian system defines kinetic and potential energy.
pub trait Lagrangian<S: Scalar, const N: usize> {
    /// Kinetic energy `T(q, q̇)`.
    fn kinetic(&self, state: &AgentState<S, N>) -> S;
    /// Potential energy `V(q)`.
    fn potential(&self, state: &AgentState<S, N>) -> S;
    /// Full Lagrangian `L = T − V`.
    fn lagrangian(&self, state: &AgentState<S, N>) -> S {
        self.kinetic(state) - self.potential(state)
    }
}

/// A simple mechanical Lagrangian: `T = ½ m q̇²` and `V(q)` is user-supplied.
pub struct MechanicalLagrangian<S, V, const N: usize>
where
    S: Scalar,
    V: Fn(&[S; N]) -> S,
{
    pub mass: S,
    pub potential_fn: V,
}

impl<S, V, const N: usize> Lagrangian<S, N> for MechanicalLagrangian<S, V, N>
where
    S: Scalar,
    V: Fn(&[S; N]) -> S,
{
    fn kinetic(&self, state: &AgentState<S, N>) -> S {
        let half = S::one() / (S::one() + S::one());
        let sum = state.q_dot.iter().map(|&v| v * v).fold(S::zero(), |a, b| a + b);
        half * self.mass * sum
    }

    fn potential(&self, state: &AgentState<S, N>) -> S {
        (self.potential_fn)(&state.q)
    }
}

/// Errors that can arise during simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicsError {
    StepSizeTooSmall,
    IntegrationDiverged,
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicsError::StepSizeTooSmall => write!(f, "step size too small"),
            DynamicsError::IntegrationDiverged => write!(f, "integration diverged"),
        }
    }
}

impl std::error::Error for DynamicsError {}

/// Numerical integrator for Euler–Lagrange dynamics.
///
/// Uses a symplectic Stormer–Verlet (leapfrog) scheme which preserves
/// energy much better than naive explicit Euler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymplecticIntegrator<S: Scalar, const N: usize> {
    pub dt: S,
}

impl<S: Scalar, const N: usize> SymplecticIntegrator<S, N> {
    pub fn new(dt: S) -> Result<Self, DynamicsError> {
        if dt <= S::zero() {
            return Err(DynamicsError::StepSizeTooSmall);
        }
        Ok(Self { dt })
    }

    /// Compute the generalised force `Fᵢ = −∂V/∂qᵢ` via central differences.
    pub fn generalised_force<V: Fn(&[S; N]) -> S>(
        &self,
        potential: &V,
        q: &[S; N],
    ) -> [S; N] {
        let h = S::epsilon().sqrt();
        let mut force = [S::zero(); N];
        let two = S::one() + S::one();
        for (i, force_i) in force.iter_mut().enumerate().take(N) {
            let mut q_plus = *q;
            let mut q_minus = *q;
            q_plus[i] = q_plus[i] + h;
            q_minus[i] = q_minus[i] - h;
            *force_i = -(potential(&q_plus) - potential(&q_minus)) / (two * h);
        }
        force
    }

    /// Perform one Stormer–Verlet step for a mechanical system.
    ///
    /// `mass` must be positive.
    pub fn step<V: Fn(&[S; N]) -> S>(
        &self,
        mass: S,
        potential: &V,
        state: &AgentState<S, N>,
    ) -> Result<AgentState<S, N>, DynamicsError> {
        if !mass.is_finite() || mass <= S::zero() {
            return Err(DynamicsError::IntegrationDiverged);
        }

        let half = S::one() / (S::one() + S::one());
        let dt = self.dt;

        // Half-step velocity
        let f = self.generalised_force(potential, &state.q);
        let mut q_dot_half = [S::zero(); N];
        for (i, qdh) in q_dot_half.iter_mut().enumerate().take(N) {
            *qdh = state.q_dot[i] + f[i] / mass * dt * half;
        }

        // Full-step position
        let mut q_new = [S::zero(); N];
        for (i, qn) in q_new.iter_mut().enumerate().take(N) {
            *qn = state.q[i] + q_dot_half[i] * dt;
        }

        // Half-step velocity with new force
        let f_new = self.generalised_force(potential, &q_new);
        let mut q_dot_new = [S::zero(); N];
        for (i, qdn) in q_dot_new.iter_mut().enumerate().take(N) {
            *qdn = q_dot_half[i] + f_new[i] / mass * dt * half;
        }

        Ok(AgentState::new(q_new, q_dot_new))
    }

    /// Integrate over `steps` steps and return the trajectory.
    pub fn integrate<V: Fn(&[S; N]) -> S>(
        &self,
        mass: S,
        potential: &V,
        initial: &AgentState<S, N>,
        steps: usize,
    ) -> Result<Vec<AgentState<S, N>>, DynamicsError> {
        let mut traj = Vec::with_capacity(steps + 1);
        let mut state = initial.clone();
        traj.push(state.clone());
        for _ in 0..steps {
            state = self.step(mass, potential, &state)?;
            traj.push(state.clone());
        }
        Ok(traj)
    }
}

/// Total mechanical energy of a state under a given Lagrangian.
pub fn total_energy<S: Scalar, const N: usize, L: Lagrangian<S, N>>(
    lagrangian: &L,
    state: &AgentState<S, N>,
) -> S {
    lagrangian.kinetic(state) + lagrangian.potential(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn harmonic_oscillator_period() {
        // V = ½ k x², k = 1, m = 1  →  ω = 1, T = 2π
        let k = 1.0_f64;
        let m = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0], [0.0]);

        // Integrate one full period (≈ 6.283185)
        let steps = ((std::f64::consts::TAU / dt) as usize) + 1;
        let traj = integrator.integrate(m, &potential, &initial, steps).unwrap();

        // Position should return close to initial value
        let final_state = traj.last().unwrap();
        assert_relative_eq!(final_state.q[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(final_state.q_dot[0], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn energy_conservation_symplectic() {
        let k = 2.0_f64;
        let m = 1.5_f64;
        let potential = |q: &[f64; 2]| 0.5 * k * (q[0] * q[0] + q[1] * q[1]);
        let lagrangian = MechanicalLagrangian {
            mass: m,
            potential_fn: potential,
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0, -0.5], [0.2, 0.3]);

        let e0 = total_energy(&lagrangian, &initial);
        let traj = integrator.integrate(m, &potential, &initial, 10_000).unwrap();

        for state in &traj[1..] {
            let e = total_energy(&lagrangian, state);
            // Symplectic integrator should keep energy bounded (no drift)
            assert_relative_eq!(e, e0, epsilon = 1e-4);
        }
    }

    #[test]
    fn free_particle_motion() {
        let m = 1.0_f64;
        let potential = |_: &[f64; 3]| 0.0_f64; // free particle
        let dt = 0.01;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let v0 = [1.0, 2.0, -0.5];
        let initial = AgentState::new([0.0, 0.0, 0.0], v0);

        let traj = integrator.integrate(m, &potential, &initial, 100).unwrap();
        let final_state = traj.last().unwrap();

        // Position should be x = v0 * t
        assert_relative_eq!(final_state.q[0], v0[0] * 1.0, epsilon = 1e-10);
        assert_relative_eq!(final_state.q[1], v0[1] * 1.0, epsilon = 1e-10);
        assert_relative_eq!(final_state.q[2], v0[2] * 1.0, epsilon = 1e-10);
    }

    #[test]
    fn rejects_non_positive_step_size() {
        assert_eq!(
            SymplecticIntegrator::<f64, 1>::new(0.0).unwrap_err(),
            DynamicsError::StepSizeTooSmall
        );
        assert_eq!(
            SymplecticIntegrator::<f64, 1>::new(-0.01).unwrap_err(),
            DynamicsError::StepSizeTooSmall
        );
    }

    #[test]
    fn rejects_non_positive_or_non_finite_mass() {
        let integrator = SymplecticIntegrator::new(0.01).unwrap();
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let state = AgentState::new([1.0], [0.0]);

        assert_eq!(
            integrator.step(0.0, &potential, &state).unwrap_err(),
            DynamicsError::IntegrationDiverged
        );
        assert_eq!(
            integrator.step(-1.0, &potential, &state).unwrap_err(),
            DynamicsError::IntegrationDiverged
        );
        assert_eq!(
            integrator.step(f64::NAN, &potential, &state).unwrap_err(),
            DynamicsError::IntegrationDiverged
        );
        assert_eq!(
            integrator.step(f64::INFINITY, &potential, &state).unwrap_err(),
            DynamicsError::IntegrationDiverged
        );
    }

    #[test]
    fn integrate_zero_steps_returns_initial_state() {
        let integrator = SymplecticIntegrator::new(0.01).unwrap();
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let initial = AgentState::new([1.0], [0.0]);
        let traj = integrator.integrate(1.0, &potential, &initial, 0).unwrap();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj[0], initial);
    }
}
