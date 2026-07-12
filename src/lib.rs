//! Conservation laws for agent dynamics.
//!
//! This crate provides tools for modeling agent dynamics using Lagrangian
//! mechanics, verifying Noether's theorem for symmetry-derived conservation
//! laws, and checking conservation of quantities like energy and momentum.

pub mod conserved;
pub mod fleet_integration;
pub mod hamiltonian;
pub mod lagrangian;
pub mod noether;

use num_traits::Float;

/// Trait for differentiable scalar fields used in dynamics.
pub trait Scalar: Float + std::fmt::Debug + 'static {}
impl<T: Float + std::fmt::Debug + 'static> Scalar for T {}

/// Compute a central difference gradient of `f` at `x` with step `h`.
pub fn central_diff<F, S>(f: F, x: S, h: S) -> S
where
    F: Fn(S) -> S,
    S: Scalar,
{
    let two = S::one() + S::one();
    (f(x + h) - f(x - h)) / (two * h)
}

/// Compute a central difference time derivative of a trajectory `q(t)`.
pub fn time_derivative<S: Scalar>(q: &[S], dt: S) -> Vec<S> {
    let two = S::one() + S::one();
    let n = q.len();
    let mut dq = Vec::with_capacity(n);
    // Forward difference at start
    dq.push((q[1] - q[0]) / dt);
    // Central difference in interior
    for i in 1..n - 1 {
        dq.push((q[i + 1] - q[i - 1]) / (two * dt));
    }
    // Backward difference at end
    dq.push((q[n - 1] - q[n - 2]) / dt);
    dq
}
