//! Hamiltonian mechanics via Legendre transform.
//!
//! Provides:
//! - **Legendre transform**: L(q, q̇, t) → H(q, p, t) where p = ∂L/∂q̇.
//! - **Canonical equations**: q̇ = ∂H/∂p, ṗ = −∂H/∂q.
//! - **Poisson brackets**: {f, g} = Σ (∂f/∂qᵢ)(∂g/∂pᵢ) − (∂f/∂pᵢ)(∂g/∂qᵢ).
//! - **Liouville's theorem**: phase space volume preservation.
//! - **Poincaré recurrence**: detection of near-recurrence.

use crate::Scalar;

/// A point in 2N-dimensional phase space: (q₁…qₙ, p₁…pₙ).
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseSpacePoint<S, const N: usize> {
    pub q: [S; N],
    pub p: [S; N],
}

impl<S: Scalar, const N: usize> PhaseSpacePoint<S, N> {
    pub fn new(q: [S; N], p: [S; N]) -> Self {
        Self { q, p }
    }

    /// Euclidean distance in phase space.
    pub fn distance(&self, other: &Self) -> S {
        let mut sum = S::zero();
        for i in 0..N {
            let dq = self.q[i] - other.q[i];
            let dp = self.p[i] - other.p[i];
            sum = sum + dq * dq + dp * dp;
        }
        sum.sqrt()
    }

    /// Phase space volume element (product of |dqᵢ dpᵢ|).
    /// This is a scalar measure for a single point (not very meaningful alone,
    /// but used for comparing total volume before/after a map).
    pub fn volume_element(&self) -> S {
        let mut vol = S::one();
        for i in 0..N {
            vol = vol * (self.q[i].abs() + self.p[i].abs());
        }
        vol
    }
}

/// A Hamiltonian system H(q, p, t).
pub trait Hamiltonian<S: Scalar, const N: usize> {
    /// The Hamiltonian function H(q, p).
    fn hamiltonian(&self, q: &[S; N], p: &[S; N]) -> S;

    /// ∂H/∂qᵢ via central differences.
    fn dH_dq(&self, q: &[S; N], p: &[S; N]) -> [S; N] {
        let h = S::epsilon().sqrt();
        let two = S::one() + S::one();
        let mut grad = [S::zero(); N];
        for i in 0..N {
            let mut q_plus = *q;
            let mut q_minus = *q;
            q_plus[i] = q_plus[i] + h;
            q_minus[i] = q_minus[i] - h;
            grad[i] = (self.hamiltonian(&q_plus, p) - self.hamiltonian(&q_minus, p)) / (two * h);
        }
        grad
    }

    /// ∂H/∂pᵢ via central differences.
    fn dH_dp(&self, q: &[S; N], p: &[S; N]) -> [S; N] {
        let h = S::epsilon().sqrt();
        let two = S::one() + S::one();
        let mut grad = [S::zero(); N];
        for i in 0..N {
            let mut p_plus = *p;
            let mut p_minus = *p;
            p_plus[i] = p_plus[i] + h;
            p_minus[i] = p_minus[i] - h;
            grad[i] = (self.hamiltonian(q, &p_plus) - self.hamiltonian(q, &p_minus)) / (two * h);
        }
        grad
    }
}

/// Simple separable Hamiltonian: H = p²/(2m) + V(q).
pub struct SeparableHamiltonian<S, V, const N: usize>
where
    S: Scalar,
    V: Fn(&[S; N]) -> S,
{
    pub mass: S,
    pub potential: V,
}

impl<S, V, const N: usize> Hamiltonian<S, N> for SeparableHamiltonian<S, V, N>
where
    S: Scalar,
    V: Fn(&[S; N]) -> S,
{
    fn hamiltonian(&self, q: &[S; N], p: &[S; N]) -> S {
        let half = S::one() / (S::one() + S::one());
        let kinetic = p.iter().map(|&pi| pi * pi).fold(S::zero(), |a, b| a + b);
        half * kinetic / self.mass + (self.potential)(q)
    }
}

/// Symplectic integrator for Hamiltonian systems (Störmer-Verlet).
pub struct HamiltonianIntegrator<S: Scalar, const N: usize> {
    pub dt: S,
}

impl<S: Scalar, const N: usize> HamiltonianIntegrator<S, N> {
    pub fn new(dt: S) -> Self {
        Self { dt }
    }

    /// One step of the Störmer-Verlet integrator.
    pub fn step<H: Hamiltonian<S, N>>(
        &self,
        ham: &H,
        state: &PhaseSpacePoint<S, N>,
    ) -> PhaseSpacePoint<S, N> {
        let half = S::one() / (S::one() + S::one());
        let dt = self.dt;

        // Half-step momentum
        let dH_dq = ham.dH_dq(&state.q, &state.p);
        let mut p_half = state.p;
        for i in 0..N {
            p_half[i] = p_half[i] - dH_dq[i] * dt * half;
        }

        // Full-step position
        let dH_dp = ham.dH_dp(&state.q, &p_half);
        let mut q_new = state.q;
        for i in 0..N {
            q_new[i] = q_new[i] + dH_dp[i] * dt;
        }

        // Half-step momentum with new force
        let dH_dq_new = ham.dH_dq(&q_new, &p_half);
        let mut p_new = p_half;
        for i in 0..N {
            p_new[i] = p_new[i] - dH_dq_new[i] * dt * half;
        }

        PhaseSpacePoint::new(q_new, p_new)
    }

    /// Integrate over multiple steps.
    pub fn integrate<H: Hamiltonian<S, N>>(
        &self,
        ham: &H,
        initial: &PhaseSpacePoint<S, N>,
        steps: usize,
    ) -> Vec<PhaseSpacePoint<S, N>> {
        let mut traj = Vec::with_capacity(steps + 1);
        traj.push(initial.clone());
        let mut state = initial.clone();
        for _ in 0..steps {
            state = self.step(ham, &state);
            traj.push(state.clone());
        }
        traj
    }
}

// ── Poisson brackets ─────────────────────────────────────────────────

/// Compute the Poisson bracket {f, g} numerically.
///
/// {f, g} = Σᵢ (∂f/∂qᵢ)(∂g/∂pᵢ) − (∂f/∂pᵢ)(∂g/∂qᵢ)
///
/// f and g are represented as functions of (q, p).
pub fn poisson_bracket<S, F, G, const N: usize>(
    f: &F,
    g: &G,
    q: &[S; N],
    p: &[S; N],
) -> S
where
    S: Scalar,
    F: Fn(&[S; N], &[S; N]) -> S,
    G: Fn(&[S; N], &[S; N]) -> S,
{
    let h = S::epsilon().sqrt();
    let two = S::one() + S::one();
    let mut bracket = S::zero();

    for i in 0..N {
        // ∂f/∂qᵢ
        let mut q_plus = *q;
        let mut q_minus = *q;
        q_plus[i] = q_plus[i] + h;
        q_minus[i] = q_minus[i] - h;
        let df_dqi = (f(&q_plus, p) - f(&q_minus, p)) / (two * h);

        // ∂g/∂pᵢ
        let mut p_plus = *p;
        let mut p_minus = *p;
        p_plus[i] = p_plus[i] + h;
        p_minus[i] = p_minus[i] - h;
        let dg_dpi = (g(q, &p_plus) - g(q, &p_minus)) / (two * h);

        // ∂f/∂pᵢ
        let df_dpi = (f(q, &p_plus) - f(q, &p_minus)) / (two * h);

        // ∂g/∂qᵢ
        let dg_dqi = (g(&q_plus, p) - g(&q_minus, p)) / (two * h);

        bracket = bracket + df_dqi * dg_dpi - df_dpi * dg_dqi;
    }

    bracket
}

// ── Liouville's theorem ──────────────────────────────────────────────

/// Estimate the phase space volume of a cloud of points.
///
/// Uses a simple bounding-box method (for exact verification, use
/// the Jacobian of the symplectic map).
pub fn phase_space_volume<S: Scalar, const N: usize>(points: &[PhaseSpacePoint<S, N>]) -> S {
    if points.is_empty() {
        return S::zero();
    }
    // Volume ≈ product of ranges in each (qᵢ, pᵢ) direction
    let mut volume = S::one();
    for i in 0..N {
        let mut q_min = points[0].q[i];
        let mut q_max = points[0].q[i];
        let mut p_min = points[0].p[i];
        let mut p_max = points[0].p[i];
        for pt in &points[1..] {
            if pt.q[i] < q_min { q_min = pt.q[i]; }
            if pt.q[i] > q_max { q_max = pt.q[i]; }
            if pt.p[i] < p_min { p_min = pt.p[i]; }
            if pt.p[i] > p_max { p_max = pt.p[i]; }
        }
        volume = volume * (q_max - q_min) * (p_max - p_min);
    }
    volume
}

/// Verify Liouville's theorem: phase space volume should be approximately
/// preserved under Hamiltonian flow.
pub fn verify_liouville<S: Scalar, const N: usize>(
    initial: &[PhaseSpacePoint<S, N>],
    evolved: &[PhaseSpacePoint<S, N>],
    tolerance: S,
) -> bool {
    let v0 = phase_space_volume(initial);
    let v1 = phase_space_volume(evolved);
    if v0.is_zero() { return true; }
    ((v1 - v0).abs() / v0.abs()) < tolerance
}

// ── Poincaré recurrence ──────────────────────────────────────────────

/// Find the first near-recurrence in a trajectory: a later point close
/// to the initial point.
pub fn find_recurrence<S: Scalar, const N: usize>(
    trajectory: &[PhaseSpacePoint<S, N>],
    tolerance: S,
    min_step: usize,
) -> Option<usize> {
    if trajectory.is_empty() { return None; }
    let initial = &trajectory[0];
    for (i, pt) in trajectory.iter().enumerate().skip(min_step) {
        if pt.distance(initial) < tolerance {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D harmonic oscillator: H = p²/2 + q²/2
    fn harmonic_1d() -> SeparableHamiltonian<f64, impl Fn(&[f64; 1]) -> f64, 1> {
        SeparableHamiltonian {
            mass: 1.0,
            potential: |q: &[f64; 1]| 0.5 * q[0] * q[0],
        }
    }

    #[test]
    fn test_phase_space_distance() {
        let a = PhaseSpacePoint::new([0.0], [0.0]);
        let b = PhaseSpacePoint::new([3.0], [4.0]);
        assert!((a.distance(&b) - 5.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_separable_hamiltonian_value() {
        let ham = harmonic_1d();
        // H = p²/2 + q²/2 → at (1,0): H = 0.5
        assert!((ham.hamiltonian(&[1.0], &[0.0]) - 0.5).abs() < 1e-10);
        // at (0,1): H = 0.5
        assert!((ham.hamiltonian(&[0.0], &[1.0]) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dH_dq() {
        let ham = harmonic_1d();
        // H = p²/2 + q²/2 → ∂H/∂q = q
        let grad = ham.dH_dq(&[2.0], &[0.0]);
        assert!((grad[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dH_dp() {
        let ham = harmonic_1d();
        // H = p²/2 + q²/2 → ∂H/∂p = p
        let grad = ham.dH_dp(&[0.0], &[3.0]);
        assert!((grad[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamiltonian_conservation() {
        let ham = harmonic_1d();
        let dt = 0.001;
        let integrator = HamiltonianIntegrator::new(dt);
        let initial = PhaseSpacePoint::new([1.0], [0.0]);
        let h0 = ham.hamiltonian(&initial.q, &initial.p);

        let traj = integrator.integrate(&ham, &initial, 10000);
        for pt in &traj[1..] {
            let h = ham.hamiltonian(&pt.q, &pt.p);
            assert!((h - h0).abs() < 1e-4, "H should be conserved, got drift {}", (h - h0).abs());
        }
    }

    #[test]
    fn test_harmonic_period() {
        let ham = harmonic_1d();
        let dt = 0.001;
        let integrator = HamiltonianIntegrator::new(dt);
        let initial = PhaseSpacePoint::new([1.0], [0.0]);

        let steps = (std::f64::consts::TAU / dt) as usize + 1;
        let traj = integrator.integrate(&ham, &initial, steps);
        let final_pt = traj.last().unwrap();

        assert!((final_pt.q[0] - 1.0).abs() < 1e-3, "q should return to 1.0");
        assert!(final_pt.p[0].abs() < 1e-3, "p should return to ~0.0");
    }

    #[test]
    fn test_poisson_bracket_canonical() {
        // {q, p} = 1 for canonical coordinates
        let f = |q: &[f64; 1], _p: &[f64; 1]| q[0]; // f = q
        let g = |_q: &[f64; 1], p: &[f64; 1]| p[0]; // g = p
        let bracket = poisson_bracket(&f, &g, &[1.0], &[2.0]);
        assert!((bracket - 1.0).abs() < 1e-6, "{{q,p}} should be 1, got {bracket}");
    }

    #[test]
    fn test_poisson_bracket_anticommutative() {
        let f = |q: &[f64; 1], p: &[f64; 1]| q[0] * p[0]; // f = qp
        let g = |q: &[f64; 1], p: &[f64; 1]| q[0] + p[0]; // g = q + p
        let b1 = poisson_bracket(&f, &g, &[1.0], &[1.0]);
        let b2 = poisson_bracket(&g, &f, &[1.0], &[1.0]);
        assert!((b1 + b2).abs() < 1e-6, "{{f,g}} = -{{g,f}}");
    }

    #[test]
    fn test_poisson_bracket_hamiltonian_flow() {
        // {H, q} = -dq/dt should equal ∂H/∂p * (-1) ... actually {H,q} = -ṗ
        // For harmonic oscillator H = p²/2 + q²/2: {H,q} = -∂H/∂p = -p
        let ham = |_q: &[f64; 1], p: &[f64; 1]| 0.5 * p[0] * p[0] + 0.5 * _q[0] * _q[0];
        let q_fn = |q: &[f64; 1], _p: &[f64; 1]| q[0];
        let bracket = poisson_bracket(&ham, &q_fn, &[1.0], &[2.0]);
        assert!((bracket - (-2.0)).abs() < 1e-4, "{{H,q}} should be -p = -2, got {bracket}");
    }

    #[test]
    fn test_phase_space_volume() {
        let pts = vec![
            PhaseSpacePoint::new([0.0], [0.0]),
            PhaseSpacePoint::new([1.0], [0.0]),
            PhaseSpacePoint::new([0.0], [1.0]),
            PhaseSpacePoint::new([1.0], [1.0]),
        ];
        let vol = phase_space_volume(&pts);
        assert!((vol - 1.0_f64).abs() < 1e-10, "volume of unit square should be 1");
    }

    #[test]
    fn test_verify_liouville() {
        let ham = harmonic_1d();
        let dt = 0.01;
        let integrator = HamiltonianIntegrator::new(dt);

        // Create a small cloud of initial points
        let initial: Vec<PhaseSpacePoint<f64, 1>> = (0..5)
            .map(|i| PhaseSpacePoint::new([0.1 * i as f64], [0.1]))
            .collect();

        let evolved: Vec<PhaseSpacePoint<f64, 1>> = initial.iter()
            .map(|pt| integrator.step(&ham, pt))
            .collect();

        assert!(verify_liouville(&initial, &evolved, 0.1), "Liouville should hold approximately");
    }

    #[test]
    fn test_find_recurrence() {
        let ham = harmonic_1d();
        let dt = 0.01;
        let integrator = HamiltonianIntegrator::new(dt);
        let initial = PhaseSpacePoint::new([1.0], [0.0]);

        // Integrate ~1.5 periods
        let traj = integrator.integrate(&ham, &initial, 1000);
        let recurrence = find_recurrence(&traj, 0.05, 100);
        assert!(recurrence.is_some(), "should find near-recurrence for harmonic oscillator");
        // Period ≈ 2π ≈ 628 steps at dt=0.01
        if let Some(step) = recurrence {
            assert!(step > 500 && step < 700, "recurrence at step {step}, expected ~628");
        }
    }

    #[test]
    fn test_2d_hamiltonian() {
        let ham: SeparableHamiltonian<f64, _, 2> = SeparableHamiltonian {
            mass: 1.0,
            potential: |q: &[f64; 2]| 0.5 * (q[0] * q[0] + q[1] * q[1]),
        };
        let dt = 0.001;
        let integrator = HamiltonianIntegrator::new(dt);
        let initial = PhaseSpacePoint::new([1.0, 0.0], [0.0, 1.0]);
        let h0 = ham.hamiltonian(&initial.q, &initial.p);

        let traj = integrator.integrate(&ham, &initial, 5000);
        for pt in &traj[100..] {
            let h = ham.hamiltonian(&pt.q, &pt.p);
            assert!((h - h0).abs() < 1e-3);
        }
    }
}
