//! A thermostat agent that maintains room temperature using Lagrangian control.
//!
//! The room temperature T(t) follows:
//!   m * d²T/dt² = -k * (T - T_target) - c * dT/dt + heater_power
//!
//! This is a damped driven harmonic oscillator. We use symplectic integration
//! to evolve the temperature state while monitoring energy (control effort).
//!
//! Run with:
//! ```bash
//! cargo run --example thermostat_agent
//! ```

use conservation_law::lagrangian::{
    total_energy, AgentState, MechanicalLagrangian, SymplecticIntegrator,
};
use conservation_law::noether::{test_invariance, ChargeMonitor, TimeTranslationSymmetry};

fn main() {
    let target_temp = 22.0_f64; // °C
    let outside_temp = 5.0_f64;
    let initial_temp = 15.0_f64;

    // Thermal mass (inertia) and coupling constants
    let m = 10.0; // thermal mass
    let k = 0.5; // restoring force toward target
    let c = 0.3; // damping (heat loss to outside)

    println!("=== Thermostat Agent ===");
    println!("Target temperature: {}°C", target_temp);
    println!("Outside temperature: {}°C", outside_temp);
    println!("Initial temperature: {}°C\n", initial_temp);

    // The "potential" includes both the target attraction and outside leakage
    // V(T) = ½ k (T - T_target)² + c * T * (T - outside)
    let potential = |q: &[f64; 1]| {
        let t = q[0];
        0.5 * k * (t - target_temp).powi(2) + c * t * (t - outside_temp)
    };

    let lagrangian = MechanicalLagrangian {
        mass: m,
        potential_fn: potential,
    };

    // Initial state: T = 15°C, dT/dt = 0
    let mut state = AgentState::new([initial_temp], [0.0]);
    let e0 = total_energy(&lagrangian, &state);

    let dt = 0.1; // 0.1 time units = minutes
    let integrator = SymplecticIntegrator::new(dt).unwrap();

    // Heater control: add energy when below target, but respect budget
    let max_heater_power = 5.0;
    let mut heater_monitor = ChargeMonitor::new(1e-6);
    let mut temp_monitor = ChargeMonitor::new(1e-6);

    println!("Time | Temp (°C) | dT/dt | Heater | Energy  | Drift");
    println!("-----|-----------|-------|--------|---------|-------------");

    let mut trajectory = vec![state.clone()];
    let mut total_heater = 0.0;

    for step in 0..=200 {
        let t = step as f64 * dt;
        let temp = state.q[0];
        let e = total_energy(&lagrangian, &state);
        let drift = (e - e0).abs();

        // Heater control law: proportional to error, clamped
        let error = target_temp - temp;
        let heater = (error * 0.5).clamp(0.0, max_heater_power);
        total_heater += heater;

        heater_monitor.push(heater);
        temp_monitor.push(temp);

        if step % 20 == 0 {
            println!(
                "{:4.0} | {:9.2} | {:5.2} | {:6.2} | {:7.2} | {:e}",
                t, temp, state.q_dot[0], heater, e, drift
            );
        }

        // Symplectic step
        state = integrator.step(m, &potential, &state).unwrap();

        // Add heater as external forcing (impulse on velocity)
        state.q_dot[0] += heater * dt / m;

        trajectory.push(state.clone());
    }

    // -----------------------------------------------------------------
    // Analysis
    // -----------------------------------------------------------------
    println!("\n--- Thermostat Analysis ---");
    let final_temp = trajectory.last().unwrap().q[0];
    println!(
        "Final temperature: {:.2}°C (target: {}°C)",
        final_temp, target_temp
    );
    println!("Total heater energy used: {:.2} units", total_heater);

    // Check energy conservation WITHOUT heater (autonomous system)
    println!("\n--- Energy conservation check (heater off) ---");
    let mut passive_state = AgentState::new([initial_temp], [0.0]);
    let _passive_e0 = total_energy(&lagrangian, &passive_state);
    let mut passive_traj = vec![passive_state.clone()];

    for _ in 0..200 {
        passive_state = integrator.step(m, &potential, &passive_state).unwrap();
        passive_traj.push(passive_state.clone());
    }

    let mut e_monitor = ChargeMonitor::new(1e-4);
    for s in &passive_traj {
        e_monitor.push(total_energy(&lagrangian, s));
    }
    println!(
        "Passive system energy conserved: {} (max drift = {:e})",
        e_monitor.is_conserved(),
        e_monitor.max_drift()
    );

    // Time-translation invariance of the autonomous system
    let time_sym = TimeTranslationSymmetry;
    let inv = test_invariance(&lagrangian, &time_sym, &passive_traj[0], 1e-3, 1e-6);
    println!(
        "Time-translation invariance: {} (ΔL = {:e})",
        inv.invariant, inv.delta_lagrangian
    );
    println!("This symmetry implies energy conservation in the passive system.");
}
