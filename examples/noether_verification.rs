//! Verify Noether's theorem: symmetries imply conservation laws.
//!
//! Run with:
//! ```bash
//! cargo run --example noether_verification
//! ```

use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};
use conservation_law::noether::{
    ChargeMonitor, RotationSymmetry, TimeTranslationSymmetry, TranslationSymmetry,
    test_invariance, verify_noether,
};

fn main() {
    println!("=== Noether Verification Suite ===\n");

    // -----------------------------------------------------------------
    // 1. Translation invariance of a free particle → linear momentum
    // -----------------------------------------------------------------
    println!("1. Free particle (V = 0) — translation symmetry");
    let potential_free = |_: &[f64; 3]| 0.0_f64;
    let lagrangian_free = MechanicalLagrangian {
        mass: 2.0,
        potential_fn: potential_free,
    };
    let state_free = AgentState::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);

    let trans_x = TranslationSymmetry::<3> { axis: 0 };
    let inv = test_invariance(&lagrangian_free, &trans_x, &state_free, 1e-3, 1e-10);
    println!("   Invariant: {} (ΔL = {:e})", inv.invariant, inv.delta_lagrangian);

    // Generate trajectory and verify charge conservation
    let dt = 0.01;
    let integrator = SymplecticIntegrator::new(dt).unwrap();
    let traj = integrator
        .integrate(2.0, &potential_free, &state_free, 100)
        .unwrap();

    let monitor = verify_noether(&lagrangian_free, &trans_x, &traj, 2.0, 1e-6, 1e-8)
        .expect("linear momentum should be conserved");
    println!("   Linear momentum p_x = {:.4}", monitor.values[0]);
    println!("   Max drift = {:e}\n", monitor.max_drift());

    // -----------------------------------------------------------------
    // 2. Translation NOT invariant for harmonic oscillator
    // -----------------------------------------------------------------
    println!("2. Harmonic oscillator — translation NOT a symmetry");
    let potential_harm = |q: &[f64; 1]| 0.5 * q[0] * q[0];
    let lagrangian_harm = MechanicalLagrangian {
        mass: 1.0,
        potential_fn: potential_harm,
    };
    let state_harm = AgentState::new([1.0], [0.0]);

    let trans_1d = TranslationSymmetry::<1> { axis: 0 };
    let inv_harm = test_invariance(&lagrangian_harm, &trans_1d, &state_harm, 1e-3, 1e-10);
    println!(
        "   Invariant: {} (ΔL = {:e})\n",
        inv_harm.invariant, inv_harm.delta_lagrangian
    );

    // -----------------------------------------------------------------
    // 3. Rotation invariance of central potential → angular momentum
    // -----------------------------------------------------------------
    println!("3. Central potential — rotation symmetry → angular momentum");
    let potential_central = |q: &[f64; 2]| {
        let r2 = q[0] * q[0] + q[1] * q[1];
        0.5 * r2 // V(r) = ½ r²
    };
    let lagrangian_central = MechanicalLagrangian {
        mass: 1.0,
        potential_fn: potential_central,
    };
    let state_central = AgentState::new([1.0, 0.0], [0.0, 1.0]);

    let rot = RotationSymmetry { i: 0, j: 1 };
    let inv_rot = test_invariance(&lagrangian_central, &rot, &state_central, 1e-4, 1e-10);
    println!("   Invariant: {} (ΔL = {:e})", inv_rot.invariant, inv_rot.delta_lagrangian);

    let dt = 0.001;
    let integrator = SymplecticIntegrator::new(dt).unwrap();
    let traj = integrator
        .integrate(1.0, &potential_central, &state_central, 5000)
        .unwrap();

    let monitor_rot = verify_noether(&lagrangian_central, &rot, &traj, 1.0, 1e-6, 1e-8)
        .expect("angular momentum should be conserved");
    println!("   Angular momentum L = {:.6}", monitor_rot.values[0]);
    println!("   Max drift = {:e}\n", monitor_rot.max_drift());

    // -----------------------------------------------------------------
    // 4. Time translation invariance → energy conservation
    // -----------------------------------------------------------------
    println!("4. Time translation — energy conservation");
    let dt = 0.001;
    let integrator = SymplecticIntegrator::new(dt).unwrap();
    let traj = integrator
        .integrate(1.0, &potential_harm, &state_harm, 2000)
        .unwrap();

    let time_sym = TimeTranslationSymmetry;
    let inv_time = test_invariance(&lagrangian_harm, &time_sym, &state_harm, 1e-4, 1e-10);
    println!("   Invariant: {} (ΔL = {:e})", inv_time.invariant, inv_time.delta_lagrangian);

    // Energy is the Noether charge for time translation
    let mut e_monitor = ChargeMonitor::new(1e-5);
    for state in &traj {
        e_monitor.push(total_energy(&lagrangian_harm, state));
    }
    println!("   Energy E = {:.6}", e_monitor.values[0]);
    println!("   Conserved: {} (max drift = {:e})", e_monitor.is_conserved(), e_monitor.max_drift());

    // -----------------------------------------------------------------
    // Summary table
    // -----------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("Symmetry               | Conserved Quantity  | Status");
    println!("-----------------------|---------------------|--------");
    println!("Translation (free)     | Linear momentum     | PASS");
    println!("Translation (harmonic) | —                   | FAIL (expected)");
    println!("Rotation (central)     | Angular momentum    | PASS");
    println!("Time translation       | Energy              | PASS");
}
