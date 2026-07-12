//! Property-based tests for conservation-law using proptest.

use conservation_law::lagrangian::{
    total_energy, AgentState, MechanicalLagrangian, SymplecticIntegrator,
};
use conservation_law::noether::{
    noether_charge, test_invariance, verify_noether, RotationSymmetry, TranslationSymmetry,
};
use proptest::prelude::*;

prop_compose! {
    fn arb_state()(q in -10.0f64..10.0, v in -10.0f64..10.0) -> AgentState<f64, 1> {
        AgentState::new([q], [v])
    }
}

prop_compose! {
    fn arb_state_2d()
        (q0 in -5.0f64..5.0, q1 in -5.0f64..5.0,
         v0 in -5.0f64..5.0, v1 in -5.0f64..5.0)
        -> AgentState<f64, 2>
    {
        AgentState::new([q0, q1], [v0, v1])
    }
}

proptest! {
    #[test]
    fn symplectic_energy_never_drifts_massively(
        state in arb_state_2d(),
        steps in 100usize..2000,
        dt in 1e-4f64..1e-2,
        k in 0.1f64..10.0,
        m in 0.5f64..5.0
    ) {
        let potential = move |q: &[f64; 2]| 0.5 * k * (q[0] * q[0] + q[1] * q[1]);
        let lagrangian = MechanicalLagrangian { mass: m, potential_fn: potential };
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let e0 = total_energy(&lagrangian, &state);

        let traj = integrator.integrate(m, &potential, &state, steps).unwrap();
        for s in &traj {
            let e = total_energy(&lagrangian, s);
            // Symplectic integrator guarantees bounded oscillation, not exact conservation
            let relative_error = ((e - e0) / e0).abs();
            prop_assert!(relative_error < 1e-2 || e0.abs() < 1e-10,
                "energy drift too large: e0={}, e={}, rel_err={}", e0, e, relative_error);
        }
    }

    #[test]
    fn free_particle_momentum_conserved(
        state in arb_state_2d(),
        steps in 10usize..500,
        dt in 1e-4f64..1e-2,
        m in 0.5f64..5.0
    ) {
        let potential = |_q: &[f64; 2]| 0.0_f64;
        let integrator = SymplecticIntegrator::new(dt).unwrap();
        let traj = integrator.integrate(m, &potential, &state, steps).unwrap();

        let sym_x = TranslationSymmetry::<2> { axis: 0 };
        let sym_y = TranslationSymmetry::<2> { axis: 1 };

        let monitor_x = verify_noether(
            &MechanicalLagrangian { mass: m, potential_fn: potential },
            &sym_x, &traj, m, 1e-6, 1e-8
        ).unwrap();
        let monitor_y = verify_noether(
            &MechanicalLagrangian { mass: m, potential_fn: potential },
            &sym_y, &traj, m, 1e-6, 1e-8
        ).unwrap();

        prop_assert!(monitor_x.is_conserved());
        prop_assert!(monitor_y.is_conserved());
    }

    #[test]
    fn harmonic_oscillator_is_not_translation_invariant(
        state in arb_state(),
        k in 0.1f64..10.0,
        m in 0.5f64..5.0
    ) {
        let potential = move |q: &[f64; 1]| 0.5 * k * q[0] * q[0];
        let lagrangian = MechanicalLagrangian { mass: m, potential_fn: potential };
        let sym = TranslationSymmetry::<1> { axis: 0 };
        let inv = test_invariance(&lagrangian, &sym, &state, 1e-3, 1e-10);
        prop_assert!(!inv.invariant);
    }

    #[test]
    fn noether_charge_linearity_in_velocity(
        q0 in -5.0f64..5.0,
        q1 in -5.0f64..5.0,
        v0 in -5.0f64..5.0,
        v1 in -5.0f64..5.0,
        m in 0.1f64..10.0,
        alpha in -2.0f64..2.0
    ) {
        let state = AgentState::new([q0, q1], [v0, v1]);
        let gen = [1.0_f64, 0.0];
        let q1 = noether_charge(m, &state, &gen);

        let state2 = AgentState::new([q0, q1], [alpha * v0, alpha * v1]);
        let q2 = noether_charge(m, &state2, &gen);

        // Charge is linear in velocity (p = m v)
        prop_assert!((q2 - alpha * q1).abs() < 1e-9);
    }

    #[test]
    fn rotation_invariance_central_potential_random_state(
        r in 0.1f64..10.0,
        theta in 0.0f64..std::f64::consts::TAU,
        vr in -5.0f64..5.0,
        vtheta in -5.0f64..5.0,
    ) {
        let state = AgentState::new(
            [r * theta.cos(), r * theta.sin()],
            [vr * theta.cos() - vtheta * theta.sin(),
             vr * theta.sin() + vtheta * theta.cos()]
        );
        let potential = |q: &[f64; 2]| (q[0]*q[0] + q[1]*q[1]).sqrt();
        let lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: potential };
        let sym = RotationSymmetry { i: 0, j: 1 };
        let inv = test_invariance(&lagrangian, &sym, &state, 1e-4, 1e-10);
        prop_assert!(inv.invariant);
    }

    #[test]
    fn total_energy_non_negative_for_bounded_potential(
        state in arb_state_2d(),
        k in 0.0f64..10.0,
        m in 0.1f64..10.0
    ) {
        let potential = move |q: &[f64; 2]| 0.5 * k * (q[0] * q[0] + q[1] * q[1]);
        let lagrangian = MechanicalLagrangian { mass: m, potential_fn: potential };
        let e = total_energy(&lagrangian, &state);
        prop_assert!(e >= 0.0, "energy should be non-negative for this system, got {}", e);
    }
}
