//! Conserved quantity detection and verification from trajectories.
//!
//! Given a trajectory of agent states, automatically:
//! - Detect candidate conserved quantities (linear and quadratic).
//! - Verify conservation along the trajectory.
//! - Test for energy, momentum, and angular momentum conservation.

use crate::lagrangian::{
    total_energy, AgentState, Lagrangian, MechanicalLagrangian, SymplecticIntegrator,
};

use crate::Scalar;

/// A candidate conserved quantity detected from trajectory data.
#[derive(Debug, Clone)]
pub struct ConservedQuantity<S: Scalar> {
    pub name: String,
    pub initial_value: S,
    pub max_drift: S,
    pub is_conserved: bool,
}

impl<S: Scalar> ConservedQuantity<S> {
    /// Check if the quantity is conserved within tolerance.
    pub fn verify(&self, tolerance: S) -> bool {
        self.max_drift < tolerance
    }
}

/// Detect and verify conserved quantities from a trajectory.
///
/// Tests for:
/// 1. Total energy conservation.
/// 2. Linear momentum (if potential is translation-invariant).
/// 3. Angular momentum (if potential is rotation-invariant).
/// 4. Custom user-supplied quantities.
pub struct ConservationDetector<S: Scalar, const N: usize> {
    pub tolerance: S,
    pub quantities: Vec<ConservedQuantity<S>>,
}

impl<S: Scalar, const N: usize> ConservationDetector<S, N> {
    pub fn new(tolerance: S) -> Self {
        Self {
            tolerance,
            quantities: Vec::new(),
        }
    }

    /// Check energy conservation along a trajectory.
    pub fn check_energy<L: Lagrangian<S, N>>(
        &mut self,
        lagrangian: &L,
        trajectory: &[AgentState<S, N>],
    ) {
        if trajectory.is_empty() {
            return;
        }
        let e0 = total_energy(lagrangian, &trajectory[0]);
        let mut max_drift = S::zero();
        for state in &trajectory[1..] {
            let e = total_energy(lagrangian, state);
            let drift = (e - e0).abs();
            if drift > max_drift {
                max_drift = drift;
            }
        }
        self.quantities.push(ConservedQuantity {
            name: "energy".to_string(),
            initial_value: e0,
            max_drift,
            is_conserved: max_drift < self.tolerance,
        });
    }

    /// Check linear momentum along axis `axis` for a given mass.
    pub fn check_linear_momentum(&mut self, mass: S, trajectory: &[AgentState<S, N>], axis: usize) {
        if trajectory.is_empty() {
            return;
        }
        let p0 = mass * trajectory[0].q_dot[axis];
        let mut max_drift = S::zero();
        for state in &trajectory[1..] {
            let p = mass * state.q_dot[axis];
            let drift = (p - p0).abs();
            if drift > max_drift {
                max_drift = drift;
            }
        }
        self.quantities.push(ConservedQuantity {
            name: format!("momentum_axis_{axis}"),
            initial_value: p0,
            max_drift,
            is_conserved: max_drift < self.tolerance,
        });
    }

    /// Check angular momentum L = Σᵢ m(qᵢ q̇ⱼ − qⱼ q̇ᵢ) in the (i,j) plane.
    pub fn check_angular_momentum(
        &mut self,
        mass: S,
        trajectory: &[AgentState<S, N>],
        i: usize,
        j: usize,
    ) {
        if trajectory.is_empty() {
            return;
        }
        let l0 = mass
            * (trajectory[0].q[i] * trajectory[0].q_dot[j]
                - trajectory[0].q[j] * trajectory[0].q_dot[i]);
        let mut max_drift = S::zero();
        for state in &trajectory[1..] {
            let l = mass * (state.q[i] * state.q_dot[j] - state.q[j] * state.q_dot[i]);
            let drift = (l - l0).abs();
            if drift > max_drift {
                max_drift = drift;
            }
        }
        self.quantities.push(ConservedQuantity {
            name: format!("angular_momentum_({i},{j})"),
            initial_value: l0,
            max_drift,
            is_conserved: max_drift < self.tolerance,
        });
    }

    /// Check a generic quantity defined by a function.
    pub fn check_quantity<F: Fn(&AgentState<S, N>) -> S>(
        &mut self,
        name: &str,
        quantity_fn: F,
        trajectory: &[AgentState<S, N>],
    ) {
        if trajectory.is_empty() {
            return;
        }
        let q0 = quantity_fn(&trajectory[0]);
        let mut max_drift = S::zero();
        for state in &trajectory[1..] {
            let q = quantity_fn(state);
            let drift = (q - q0).abs();
            if drift > max_drift {
                max_drift = drift;
            }
        }
        self.quantities.push(ConservedQuantity {
            name: name.to_string(),
            initial_value: q0,
            max_drift,
            is_conserved: max_drift < self.tolerance,
        });
    }

    /// Number of quantities detected as conserved.
    pub fn num_conserved(&self) -> usize {
        self.quantities.iter().filter(|q| q.is_conserved).count()
    }

    /// Are ALL quantities conserved?
    pub fn all_conserved(&self) -> bool {
        self.quantities.iter().all(|q| q.is_conserved)
    }
}

/// Full conservation law verification for a given system.
///
/// Given a Lagrangian system, integrate and verify all applicable
/// conservation laws.
pub fn verify_all_conservation<S, V, const N: usize>(
    mass: S,
    potential: V,
    initial_state: &AgentState<S, N>,
    dt: S,
    steps: usize,
    tolerance: S,
) -> ConservationDetector<S, N>
where
    S: Scalar,
    V: Fn(&[S; N]) -> S + Clone,
{
    let lagrangian = MechanicalLagrangian {
        mass,
        potential_fn: potential.clone(),
    };

    let integrator = SymplecticIntegrator::new(dt).unwrap();
    let traj = integrator
        .integrate(mass, &potential, initial_state, steps)
        .unwrap();

    let mut detector = ConservationDetector::new(tolerance);

    // Always check energy
    detector.check_energy(&lagrangian, &traj);

    // Check momentum for each axis
    for axis in 0..N {
        detector.check_linear_momentum(mass, &traj, axis);
    }

    // Check angular momentum for each (i,j) pair
    for i in 0..N {
        for j in (i + 1)..N {
            detector.check_angular_momentum(mass, &traj, i, j);
        }
    }

    detector
}

/// Detect whether a trajectory lies approximately on a constant-energy surface.
///
/// Returns the standard deviation of energy values.
pub fn energy_spread<S: Scalar, const N: usize, L: Lagrangian<S, N>>(
    lagrangian: &L,
    trajectory: &[AgentState<S, N>],
) -> S {
    if trajectory.len() < 2 {
        return S::zero();
    }
    let n = S::from(trajectory.len()).unwrap_or_else(S::one);
    let mean: S = trajectory
        .iter()
        .map(|s| total_energy(lagrangian, s))
        .fold(S::zero(), |a, b| a + b)
        / n;
    let variance: S = trajectory
        .iter()
        .map(|s| {
            let e = total_energy(lagrangian, s);
            let d = e - mean;
            d * d
        })
        .fold(S::zero(), |a, b| a + b)
        / n;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::MechanicalLagrangian;

    #[test]
    fn test_energy_conservation_harmonic() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass,
            potential_fn: potential,
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0], [0.0]);
        let traj = integrator
            .integrate(mass, &potential, &initial, 5000)
            .unwrap();

        let mut detector = ConservationDetector::new(1e-4);
        detector.check_energy(&lagrangian, &traj);
        assert!(detector.quantities[0].is_conserved);
    }

    #[test]
    fn test_momentum_conservation_free_particle() {
        let mass = 2.0_f64;
        let potential = |_: &[f64; 2]| 0.0_f64;
        let dt = 0.01;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([0.0, 0.0], [1.0, 2.0]);
        let traj = integrator
            .integrate(mass, &potential, &initial, 100)
            .unwrap();

        let mut detector = ConservationDetector::new(1e-10);
        detector.check_linear_momentum(mass, &traj, 0);
        detector.check_linear_momentum(mass, &traj, 1);
        assert!(detector.all_conserved());
    }

    #[test]
    fn test_angular_momentum_central_potential() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 2]| {
            let r2 = q[0] * q[0] + q[1] * q[1];
            0.5 * r2
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0, 0.0], [0.0, 1.0]);
        let traj = integrator
            .integrate(mass, &potential, &initial, 5000)
            .unwrap();

        let mut detector = ConservationDetector::new(1e-4);
        detector.check_angular_momentum(mass, &traj, 0, 1);
        assert!(
            detector.quantities[0].is_conserved,
            "angular momentum should be conserved for central potential"
        );
    }

    #[test]
    fn test_momentum_not_conserved_harmonic() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let dt = 0.01;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0], [0.0]);
        let traj = integrator
            .integrate(mass, &potential, &initial, 100)
            .unwrap();

        let mut detector = ConservationDetector::new(1e-10);
        detector.check_linear_momentum(mass, &traj, 0);
        assert!(
            !detector.quantities[0].is_conserved,
            "momentum should NOT be conserved for harmonic potential"
        );
    }

    #[test]
    fn test_verify_all_harmonic() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let initial = AgentState::new([1.0], [0.0]);
        let detector = verify_all_conservation(mass, potential, &initial, 0.001, 5000, 1e-4);
        assert!(
            detector
                .quantities
                .iter()
                .find(|q| q.name == "energy")
                .unwrap()
                .is_conserved
        );
    }

    #[test]
    fn test_verify_all_free_particle() {
        let mass = 1.0_f64;
        let potential = |_: &[f64; 3]| 0.0_f64;
        let initial = AgentState::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        let detector = verify_all_conservation(mass, potential, &initial, 0.01, 100, 1e-8);
        assert!(
            detector.all_conserved(),
            "all quantities should be conserved for free particle"
        );
    }

    #[test]
    fn test_energy_spread_small() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass,
            potential_fn: potential,
        };
        let dt = 0.001;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([1.0], [0.0]);
        let traj = integrator
            .integrate(mass, &|q: &[f64; 1]| 0.5 * q[0] * q[0], &initial, 5000)
            .unwrap();

        let spread = energy_spread(&lagrangian, &traj);
        assert!(
            spread < 1e-4,
            "energy spread should be small for symplectic integrator"
        );
    }

    #[test]
    fn test_custom_quantity_kinetic_energy() {
        let mass = 1.0_f64;
        let potential = |_: &[f64; 1]| 0.0_f64;
        let dt = 0.01;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let initial = AgentState::new([0.0], [2.0]);
        let traj = integrator
            .integrate(mass, &potential, &initial, 100)
            .unwrap();

        let mut detector = ConservationDetector::new(1e-10);
        detector.check_quantity(
            "kinetic_energy",
            |s| 0.5 * mass * s.q_dot[0] * s.q_dot[0],
            &traj,
        );
        assert!(
            detector.quantities[0].is_conserved,
            "KE should be conserved for free particle"
        );
    }

    #[test]
    fn test_num_conserved() {
        let mass = 1.0_f64;
        let potential = |_: &[f64; 2]| 0.0_f64;
        let initial = AgentState::new([0.0, 0.0], [1.0, 2.0]);
        let detector = verify_all_conservation(mass, potential, &initial, 0.01, 100, 1e-8);
        // Energy + 2 momenta + 1 angular momentum = 4
        assert!(detector.num_conserved() >= 3);
    }

    #[test]
    fn test_conserved_quantity_verify() {
        let cq: ConservedQuantity<f64> = ConservedQuantity {
            name: "test".to_string(),
            initial_value: 1.0,
            max_drift: 0.001,
            is_conserved: true,
        };
        assert!(cq.verify(0.01));
        assert!(!cq.verify(0.0001));
    }

    #[test]
    fn test_empty_trajectory() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 1]| q[0];
        let lagrangian = MechanicalLagrangian {
            mass,
            potential_fn: potential,
        };
        let mut detector = ConservationDetector::new(1e-10);
        detector.check_energy(&lagrangian, &[]);
        assert!(detector.quantities.is_empty());
    }

    #[test]
    fn test_energy_spread_short_trajectories() {
        let mass = 1.0_f64;
        let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
        let lagrangian = MechanicalLagrangian {
            mass,
            potential_fn: potential,
        };
        let single = vec![AgentState::new([1.0], [0.0])];
        assert_eq!(energy_spread(&lagrangian, &single), 0.0);
        assert_eq!(energy_spread(&lagrangian, &[]), 0.0);
    }

    #[test]
    fn test_detector_all_conserved_empty() {
        let detector: ConservationDetector<f64, 1> = ConservationDetector::new(1e-6);
        assert!(detector.all_conserved());
        assert_eq!(detector.num_conserved(), 0);
    }

    #[test]
    fn test_verify_all_conservation_free_particle_1d() {
        let mass = 1.0_f64;
        let potential = |_q: &[f64; 1]| 0.0_f64;
        let initial = AgentState::new([0.0], [2.0]);
        let detector = verify_all_conservation(mass, potential, &initial, 0.01, 500, 1e-8);
        // Energy + 1 momentum = 2 conserved quantities.
        assert!(detector.all_conserved());
        assert_eq!(detector.num_conserved(), 2);
    }
}
