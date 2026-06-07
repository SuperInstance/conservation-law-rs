//! Symplectic integration of a harmonic oscillator with energy tracking.
//!
//! Run with:
//! ```bash
//! cargo run --example harmonic_oscillator
//! ```

use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};

fn main() {
    // System parameters: mass m, spring constant k
    let m = 1.0_f64;
    let k = 4.0_f64; // ω = sqrt(k/m) = 2.0 rad/s

    // Potential energy V(q) = ½ k q²
    let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];

    // Build the Lagrangian L = T − V
    let lagrangian = MechanicalLagrangian {
        mass: m,
        potential_fn: potential,
    };

    // Initial state: displaced by 1.0, zero velocity
    let initial = AgentState::new([1.0], [0.0]);
    let e0 = total_energy(&lagrangian, &initial);
    println!("Initial energy: {:.6}", e0);

    // Symplectic integrator with dt = 0.001
    let dt = 0.001;
    let integrator = SymplecticIntegrator::new(dt).unwrap();

    // Integrate for 10 periods (T = π seconds each)
    let period = std::f64::consts::PI; // T = 2π/ω = π
    let steps_per_period = (period / dt) as usize;
    let total_steps = steps_per_period * 10;

    let traj = integrator
        .integrate(m, &potential, &initial, total_steps)
        .unwrap();

    // Print energy at the end of each period
    println!("\nPeriod | Position  | Velocity  | Energy    | Drift");
    println!("-------|-----------|-----------|-----------|-------------");
    for p in 0..=10 {
        let idx = p * steps_per_period;
        let state = &traj[idx.min(traj.len() - 1)];
        let e = total_energy(&lagrangian, state);
        let drift = (e - e0).abs();
        println!(
            "  {:2}   | {:+9.6} | {:+9.6} | {:9.6} | {:e}",
            p, state.q[0], state.q_dot[0], e, drift
        );
    }

    // Verify the symplectic property: energy should not drift
    let final_state = traj.last().unwrap();
    let e_final = total_energy(&lagrangian, final_state);
    let max_drift = (e_final - e0).abs();
    println!("\nMax energy drift over 10 periods: {:e}", max_drift);
    assert!(
        max_drift < 1e-8,
        "Symplectic integrator should conserve energy"
    );

    // Compare with explicit Euler (non-symplectic) to show the difference
    println!("\n--- Explicit Euler (for comparison) ---");
    let mut q = 1.0_f64;
    let mut v = 0.0_f64;
    for _ in 0..total_steps {
        let a = -k * q / m;
        q += v * dt;
        v += a * dt;
    }
    let e_euler = 0.5 * m * v * v + 0.5 * k * q * q;
    println!("Explicit Euler final energy: {:.6}", e_euler);
    println!("Explicit Euler energy drift: {:e}", (e_euler - e0).abs());
    println!("\nThe symplectic integrator drifts by ~{:e},", max_drift);
    println!("while explicit Euler drifts by ~{:e}.", (e_euler - e0).abs());
}
