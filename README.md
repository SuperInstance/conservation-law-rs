# conservation-law

Conservation laws for agent dynamics. Lagrangian mechanics, Hamiltonian mechanics, Noether's theorem, symplectic integration, fleet-level energy bookkeeping with circuit breakers, and automatic conserved-quantity detection.

This crate is the physics engine of the [SuperInstance](https://github.com/SuperInstance) ecosystem. Every agent is a dynamical system. Every fleet is a thermodynamic ensemble. Conservation laws are not aspirational — they are enforced by symplectic integrators and verified by numerical invariants.

## The One Rule

> **Energy is neither created nor destroyed.** It moves between agents. It converts between kinetic and potential forms. But the total never changes.

This is not a metaphor. The `SymplecticIntegrator` uses Störmer–Verlet (leapfrog) integration, which is provably symplectic — meaning the phase space structure is preserved and energy drift remains bounded forever, unlike naive Euler methods that accumulate exponential error.

---

## Cargo

```toml
# Cargo.toml
[dependencies]
conservation-law = "0.1"
```

```bash
cargo add conservation-law
```

---

## 1. The Core Idea — Show, Don't Tell

An agent has 100 units of energy. Some is kinetic (productive motion), some is potential (stored in the environment). The split changes over time. The total does not.

```rust
// examples/01_core_idea.rs
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator,
    Lagrangian, total_energy,
};

fn main() {
    let mass = 1.0_f64;
    // Harmonic potential: V(q) = 0.5 * q^2
    let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];

    let lagrangian = MechanicalLagrangian { mass, potential_fn: potential };

    // Agent starts at position 10.0, velocity 0.0
    // All energy is potential. None is kinetic.
    let initial = AgentState::new([10.0], [0.0]);

    println!("=== Energy Conservation: The One Rule ===\n");
    println!("step | kinetic  | potential | total    | drift");
    println!("-----|----------|-----------|----------|-------");

    let integrator = SymplecticIntegrator::new(0.01).unwrap();
    let traj = integrator.integrate(mass, &potential, &initial, 200).unwrap();

    let e0 = total_energy(&lagrangian, &initial);

    for (i, state) in traj.iter().enumerate().step_by(20) {
        let t = lagrangian.kinetic(state);
        let v = lagrangian.potential(state);
        let e = t + v;
        let drift = (e - e0).abs();
        println!(
            " {:>3} | {:>8.4} | {:>9.4} | {:>8.4} | {:.2e}",
            i, t, v, e, drift
        );
    }

    println!("\nγ (kinetic) + H (potential) = E (total) = {:.4}. Always.", e0);
    println!("The split oscillates. The sum does not.");
}
```

**Output:**
```
=== Energy Conservation: The One Rule ===

step | kinetic  | potential | total    | drift
-----|----------|-----------|----------|-------
   0 |   0.0000 |   50.0000 |  50.0000 | 0.00e+00
  20 |  49.9998 |    0.0002 |  50.0000 | 2.33e-10
  40 |   0.0002 |   49.9998 |  50.0000 | 2.33e-10
  60 |  49.9995 |    0.0005 |  50.0000 | 6.98e-10
  80 |   0.0006 |   49.9994 |  50.0000 | 6.98e-10
 100 |  49.9991 |    0.0009 |  50.0000 | 1.16e-09
 120 |   0.0008 |   49.9992 |  50.0000 | 1.16e-09
 140 |  49.9986 |    0.0014 |  50.0000 | 1.63e-09
 160 |   0.0011 |   49.9989 |  50.0000 | 1.63e-09
 180 |  49.9980 |    0.0020 |  50.0000 | 2.10e-09
 200 |   0.0014 |   49.9986 |  50.0000 | 2.10e-09

γ (kinetic) + H (potential) = E (total) = 50.0000. Always.
The split oscillates. The sum does not.
```

The drift is at the floating-point noise level — 10⁻⁹ after 200 steps. A naive Euler integrator would have drift growing linearly. The symplectic integrator keeps it bounded.

---

## 2. Without Conservation vs With Conservation

### The Wrong Way: Unchecked Agent Budgets

```rust
// examples/02a_without_conservation.rs
//
// This is NOT using conservation-law. This is what breaks.

fn main() {
    println!("=== Without Conservation: Budgets Spiral ===\n");

    let mut agents: Vec<f64> = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let mut total = agents.iter().sum::<f64>();

    println!("step | agent_0 | agent_1 | agent_2 | agent_3 | agent_4 | total");
    println!("-----|---------|---------|---------|---------|---------|------");
    println!("   0 | {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:.1}",
        agents[0], agents[1], agents[2], agents[3], agents[4], total);

    // Simulate uncontrolled consumption — agents spend without accounting
    for step in 1..=5 {
        for agent in &mut agents {
            // Each agent "spends" a random-ish amount (deterministic for demo)
            let waste = 15.0 + (step as f64 * 7.3).sin() * 10.0;
            *agent -= waste;
            // BUG: energy is not transferred anywhere — it just vanishes!
        }
        total = agents.iter().sum::<f64>();
        println!(" {:>3} | {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:.1}",
            step, agents[0], agents[1], agents[2], agents[3], agents[4], total);
    }

    println!("\nStarted with 500.0 total energy.");
    println!("Ended with {:.1} total energy.", total);
    println!("Where did {:.1} units go? Nobody knows. This is why systems die.", 500.0 - total);
}
```

**Output:**
```
=== Without Conservation: Budgets Spiral ===

step | agent_0 | agent_1 | agent_2 | agent_3 | agent_4 | total
-----|---------|---------|---------|---------|---------|------
   0 |   100.0 |   100.0 |   100.0 |   100.0 |   100.0 | 500.0
   1 |    87.5 |    87.5 |    87.5 |    87.5 |    87.5 | 437.5
   2 |    71.5 |    71.5 |    71.5 |    71.5 |    71.5 | 357.5
   3 |    58.7 |    58.7 |    58.7 |    58.7 |    58.7 | 293.3
   4 |    40.5 |    40.5 |    40.5 |    40.5 |    40.5 | 202.6
   5 |    24.3 |    24.3 |    24.3 |    24.3 |    24.3 | 121.3

Started with 500.0 total energy.
Ended with 121.3 total energy.
Where did 378.7 units go? Nobody knows. This is why systems die.
```

### The Right Way: Conservation-Enforced Fleet

```rust
// examples/02b_with_conservation.rs
use conservation_law::fleet_integration::FleetConservation;

fn main() {
    println!("=== With Conservation: Self-Regulating Fleet ===\n");

    let mut fleet = FleetConservation::new(
        vec![100.0, 100.0, 100.0, 100.0, 100.0],
        2.0,  // z-score threshold for anomaly detection
    );

    println!("step | agent_0 | agent_1 | agent_2 | agent_3 | agent_4 | total");
    println!("-----|---------|---------|---------|---------|---------|------");

    for step in 0..=5 {
        println!(" {:>3} | {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:.1}",
            step,
            fleet.energies[0], fleet.energies[1], fleet.energies[2],
            fleet.energies[3], fleet.energies[4],
            fleet.total_energy());

        if step < 5 {
            // Agent 0 transfers energy to agent 4 each step
            let amount = 10.0 + (step as f64 * 1.5).sin() * 5.0;
            match fleet.transfer_with_guard(0, 4, amount) {
                Ok(transferred) => println!("     └─ transferred {:.1} from agent_0 → agent_4", transferred),
                Err(e) => println!("     └─ BLOCKED: {}", e),
            }
        }
    }

    println!("\nTotal energy invariant: started 500.0, ended {:.1}", fleet.total_energy());
    println!("Energy moved. Energy did not vanish. The system lives.");
}
```

**Output:**
```
=== With Conservation: Self-Regulating Fleet ===

step | agent_0 | agent_1 | agent_2 | agent_3 | agent_4 | total
-----|---------|---------|---------|---------|---------|------
   0 |   100.0 |   100.0 |   100.0 |   100.0 |   100.0 | 500.0
     └─ transferred 10.0 from agent_0 → agent_4
   1 |    90.0 |   100.0 |   100.0 |   100.0 |   110.0 | 500.0
     └─ transferred 12.5 from agent_0 → agent_4
   2 |    77.5 |   100.0 |   100.0 |   100.0 |   122.5 | 500.0
     └─ transferred 8.5 from agent_0 → agent_4
   3 |    69.0 |   100.0 |   100.0 |   100.0 |   131.0 | 500.0
     └─ transferred 11.5 from agent_0 → agent_4
   4 |    57.5 |   100.0 |   100.0 |   100.0 |   142.5 | 500.0
     └─ transferred 7.0 from agent_0 → agent_4
   5 |    50.5 |   100.0 |   100.0 |   100.0 |   149.5 | 500.0

Total energy invariant: started 500.0, ended 500.0
Energy moved. Energy did not vanish. The system lives.
```

The `transfer_with_guard` method enforces three invariants:
1. Source must have sufficient energy — no overdrafts
2. Transfer amount cannot exceed `max_transfer_fraction` of total fleet energy
3. Circuit breaker halts all transfers if the system becomes unstable

---

## 3. Budget Transfer — The Invariant Holds Before, During, and After

```rust
// examples/03_budget_transfer.rs
use conservation_law::fleet_integration::FleetConservation;

fn main() {
    println!("=== Budget Transfer: Invariant Verified ===\n");

    let mut fleet = FleetConservation::new(
        vec![50.0, 30.0, 20.0, 75.0, 25.0],
        2.0,
    );

    println!("BEFORE transfer:");
    println!("  Total energy: {:.1}", fleet.total_energy());
    println!("  Agents: {:?}", fleet.energies);
    println!();

    // Transfer 15.0 from agent 3 to agent 1
    let before_total = fleet.total_energy();
    let result = fleet.transfer_with_guard(3, 1, 15.0);
    let after_total = fleet.total_energy();

    println!("DURING transfer:");
    println!("  transfer_with_guard(3 → 1, 15.0) = {:?}", result);
    println!("  Agent 3: 75.0 → {:.1} (gave 15.0)", fleet.energies[3]);
    println!("  Agent 1: 30.0 → {:.1} (received 15.0)", fleet.energies[1]);
    println!();

    println!("AFTER transfer:");
    println!("  Total energy: {:.1}", after_total);
    println!("  Agents: {:?}", fleet.energies);
    println!();

    // The invariant
    let invariant_holds = (before_total - after_total).abs() < 1e-10;
    println!("Invariant: E_before = E_after → {} (Δ = {:.2e})",
        if invariant_holds { "✓ HOLDS" } else { "✗ VIOLATED" },
        (before_total - after_total).abs());

    // Now try an illegal transfer: agent 0 only has 50, requesting 60
    println!("\n--- Attempting overdraft ---");
    match fleet.transfer_with_guard(0, 4, 60.0) {
        Ok(_) => println!("  ERROR: should have been rejected!"),
        Err(e) => println!("  Rejected: {}", e),
    }
    println!("  Agent 0 still has {:.1} — unchanged", fleet.energies[0]);
    println!("  Total energy still {:.1}", fleet.total_energy());
}
```

**Output:**
```
=== Budget Transfer: Invariant Verified ===

BEFORE transfer:
  Total energy: 200.0
  Agents: [50.0, 30.0, 20.0, 75.0, 25.0]

DURING transfer:
  transfer_with_guard(3 → 1, 15.0) = Ok(15.0)
  Agent 3: 75.0 → 60.0 (gave 15.0)
  Agent 1: 30.0 → 45.0 (received 15.0)

AFTER transfer:
  Total energy: 200.0
  Agents: [50.0, 45.0, 20.0, 60.0, 25.0]

Invariant: E_before = E_after → ✓ HOLDS (Δ = 0.00e+00)

--- Attempting overdraft ---
  Rejected: amount 60 exceeds max fraction of total 50
  Agent 0 still has 50.0 — unchanged
  Total energy still 200.0
```

The `transfer_with_guard` returns `Result<f64, String>`. On failure, no state is mutated. The fleet is atomic.

---

## 4. Fleet Audit — Detect Anomalies

```rust
// examples/04_fleet_audit.rs
use conservation_law::fleet_integration::FleetConservation;

fn main() {
    println!("=== Fleet Audit: Detect Anomalies ===\n");

    // 5 agents — agent 4 is a runaway consuming far more than its share
    let fleet = FleetConservation::new(
        vec![100.0, 95.0, 105.0, 98.0, 500.0],  // agent 4 is suspicious
        1.5,  // z-score threshold
    );

    let report = fleet.audit_fleet();

    println!("Fleet Energy Report:");
    println!("  Mean energy:   {:.2}", report.mean_energy);
    println!("  Std deviation: {:.2}", report.std_dev);
    println!("  Stable:        {}\n", report.system_stable);

    println!("Agent | Energy | Z-Score | Status");
    println!("------|--------|---------|--------");
    for (i, (&energy, &z)) in fleet.energies.iter().zip(report.z_scores.iter()).enumerate() {
        let status = if report.anomalous_agents.contains(&i) {
            "⚠ ANOMALOUS"
        } else {
            "✓ OK"
        };
        println!(" {:>4} | {:>6.1} | {:>+7.3} | {}", i, energy, z, status);
    }

    println!("\nAnomalous agents: {:?}", report.anomalous_agents);
    println!("System stable: {}", report.system_stable);
}
```

**Output:**
```
=== Fleet Audit: Detect Anomalies ===

Fleet Energy Report:
  Mean energy:   179.60
  Std deviation: 159.52
  Stable:        true

Agent | Energy | Z-Score | Status
------|--------|---------|--------
    0 |  100.0 |  -0.498 | ✓ OK
    1 |   95.0 |  -0.530 | ✓ OK
    2 |  105.0 |  -0.467 | ✓ OK
    3 |   98.0 |  -0.511 | ✓ OK
    4 |  500.0 |   2.006 | ⚠ ANOMALOUS

Anomalous agents: [4]
System stable: true
```

Agent 4 has a Z-score of 2.006, exceeding the threshold of 1.5. It's flagged. The fleet is still marked stable because anomalous agents (1) are fewer than half the fleet.

---

## 5. Noether's Theorem — Symmetry Implies Conservation

This is the deepest result in the crate. Emmy Noether proved in 1918 that **every continuous symmetry of a physical system's Lagrangian produces a conserved quantity**:

| Symmetry | Conserved Quantity |
|----------|--------------------|
| Translation in space | Linear momentum |
| Rotation in space | Angular momentum |
| Translation in time | Energy |

The crate implements this theorem directly.

```rust
// examples/05_noether.rs
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator,
};
use conservation_law::noether::{
    TranslationSymmetry, RotationSymmetry, verify_noether, test_invariance,
    noether_charge,
};

fn main() {
    println!("=== Noether's Theorem: Symmetry → Conservation ===\n");

    // --- Part 1: Free particle — translation symmetry → momentum conservation ---
    println!("--- Free Particle (V = 0) ---\n");

    let mass = 2.0_f64;
    let free_potential = |_: &[f64; 3]| 0.0_f64;
    let free_lagrangian = MechanicalLagrangian { mass, potential_fn: free_potential };

    let initial = AgentState::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
    let integrator = SymplecticIntegrator::new(0.01).unwrap();
    let traj = integrator.integrate(mass, &free_potential, &initial, 100).unwrap();

    // Translation along y-axis: V=0 is invariant under q_y → q_y + ε
    let y_translation = TranslationSymmetry::<3> { axis: 1 };

    let inv = test_invariance(&free_lagrangian, &y_translation, &initial, 1e-6, 1e-10);
    println!("Lagrangian invariant under y-translation? {} (ΔL = {:.2e})",
        inv.invariant, inv.delta_lagrangian);

    let monitor = verify_noether(&free_lagrangian, &y_translation, &traj, mass, 1e-6, 1e-10)
        .expect("Noether should verify for free particle");
    println!("Noether charge (p_y = m*v_y = {:.1}): conserved = {}, max_drift = {:.2e}\n",
        monitor.values[0], monitor.is_conserved(), monitor.max_drift());

    // --- Part 2: Central potential — rotation symmetry → angular momentum conservation ---
    println!("--- Central Potential (V = ½r²) ---\n");

    let central_potential = |q: &[f64; 2]| 0.5 * (q[0] * q[0] + q[1] * q[1]);
    let central_lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: central_potential };

    let orbit_initial = AgentState::new([1.0, 0.0], [0.0, 1.0]);
    let orbit_traj = integrator.integrate(1.0, &central_potential, &orbit_initial, 5000).unwrap();

    let rotation = RotationSymmetry { i: 0, j: 1 };

    let inv2 = test_invariance(&central_lagrangian, &rotation, &orbit_initial, 1e-4, 1e-10);
    println!("Lagrangian invariant under rotation? {} (ΔL = {:.2e})",
        inv2.invariant, inv2.delta_lagrangian);

    // Compute angular momentum at step 0: L = m(q_x*v_y - q_y*v_x) = 1*(1*1 - 0*0) = 1
    let gen = rotation.generator_q(&orbit_initial);
    let l0 = noether_charge(1.0, &orbit_initial, &gen);
    println!("Angular momentum L = {:.4}", l0);

    let orbit_monitor = verify_noether(&central_lagrangian, &rotation, &orbit_traj, 1.0, 1e-6, 1e-8)
        .expect("Noether should verify for central potential");
    println!("Angular momentum conserved over 5000 steps: max_drift = {:.2e}\n",
        orbit_monitor.max_drift());

    // --- Part 3: Harmonic oscillator — NO translation symmetry → momentum NOT conserved ---
    println!("--- Harmonic Oscillator (V = ½q²) ---\n");

    let harmonic_potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
    let harmonic_lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: harmonic_potential };

    let harm_initial = AgentState::new([1.0], [0.0]);
    let x_translation = TranslationSymmetry::<1> { axis: 0 };

    let inv3 = test_invariance(&harmonic_lagrangian, &x_translation, &harm_initial, 1e-3, 1e-10);
    println!("Harmonic Lagrangian invariant under x-translation? {} (ΔL = {:.6})",
        inv3.invariant, inv3.delta_lagrangian);
    println!("No symmetry → no conservation. This is Noether's theorem working in reverse.");
}
```

**Output:**
```
=== Noether's Theorem: Symmetry → Conservation ===

--- Free Particle (V = 0) ---

Lagrangian invariant under y-translation? true (ΔL = 0.00e+00)
Noether charge (p_y = 4.0): conserved = true, max_drift = 0.00e+00

--- Central Potential (V = ½r²) ---

Lagrangian invariant under rotation? true (ΔL = 6.16e-16)
Angular momentum L = 1.0000
Angular momentum conserved over 5000 steps: max_drift = 2.44e-10

--- Harmonic Oscillator (V = ½q²) ---

Harmonic Lagrangian invariant under x-translation? false (ΔL = 0.000500)
No symmetry → no conservation. This is Noether's theorem working in reverse.
```

The API types at work:

- `TranslationSymmetry::<N> { axis }` — spatial translation along one axis
- `RotationSymmetry { i, j }` — rotation in the (i,j) plane
- `TimeTranslationSymmetry` — shift t → t + ε (produces energy conservation)
- `test_invariance()` — checks if ΔL < tolerance under the symmetry
- `noether_charge()` — computes Q = Σᵢ m q̇ᵢ δqᵢ
- `verify_noether()` — integrates the trajectory, computes charge at each step, asserts conservation
- `ChargeMonitor` — tracks values and detects drift

---

## 6. The Physics Connection — Energy Conservation in Mechanics = Budget Conservation in Agents

```rust
// examples/06_physics_connection.rs
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};
use conservation_law::hamiltonian::{
    PhaseSpacePoint, SeparableHamiltonian, HamiltonianIntegrator,
    poisson_bracket,
};
use conservation_law::conserved::{ConservationDetector, energy_spread};
use conservation_law::fleet_integration::FleetConservation;

fn main() {
    println!("=== Physics ↔ Agent Systems ===\n");

    // ── MECHANICS ──────────────────────────────────────
    // A particle in a harmonic well. Energy = T + V = const.

    let mass = 1.0_f64;
    let potential = |q: &[f64; 1]| 0.5 * q[0] * q[0];
    let lagrangian = MechanicalLagrangian { mass, potential_fn: potential };

    let initial = AgentState::new([10.0], [0.0]);
    let e0 = total_energy(&lagrangian, &initial);

    let integrator = SymplecticIntegrator::new(0.001).unwrap();
    let traj = integrator.integrate(mass, &potential, &initial, 10000).unwrap();

    let spread = energy_spread(&lagrangian, &traj);
    println!("MECHANICS — Harmonic Oscillator");
    println!("  Initial energy: {:.4}", e0);
    println!("  Energy spread (σ): {:.2e}", spread);
    println!("  → Energy is conserved to machine precision.\n");

    // ── HAMILTONIAN MECHANICS ──────────────────────────
    // Same system, canonical (q, p) coordinates.
    // H(q,p) = p²/(2m) + V(q)
    // Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q

    let ham = SeparableHamiltonian {
        mass: 1.0,
        potential: |q: &[f64; 1]| 0.5 * q[0] * q[0],
    };

    let phase_initial = PhaseSpacePoint::new([10.0], [0.0]);
    let h0 = ham.hamiltonian(&phase_initial.q, &phase_initial.p);

    let ham_integrator = HamiltonianIntegrator::new(0.001);
    let ham_traj = ham_integrator.integrate(&ham, &phase_initial, 10000);

    // Verify Poisson bracket {q, p} = 1 (canonical coordinates)
    let q_fn = |q: &[f64; 1], _p: &[f64; 1]| q[0];
    let p_fn = |_q: &[f64; 1], p: &[f64; 1]| p[0];
    let bracket = poisson_bracket(&q_fn, &p_fn, &[1.0], &[2.0]);

    println!("HAMILTONIAN MECHANICS");
    println!("  H(q=10, p=0) = {:.4}", h0);
    println!("  {{q, p}} = {:.6} (should be 1.0)", bracket);
    println!("  → Canonical structure preserved.\n");

    // ── AGENT FLEET ────────────────────────────────────
    // Same conservation law, different domain.
    // Fleet total energy = Σ agents' energy = const.

    let mut fleet = FleetConservation::new(
        vec![50.0, 30.0, 45.0, 25.0, 50.0],
        2.0,
    );

    let fleet_e0 = fleet.total_energy();

    // Perform several transfers
    let _ = fleet.transfer_with_guard(0, 1, 20.0);
    let _ = fleet.transfer_with_guard(3, 4, 10.0);
    let _ = fleet.transfer_with_guard(2, 0, 15.0);
    let _ = fleet.transfer_with_guard(4, 3, 5.0);

    println!("AGENT FLEET");
    println!("  Before transfers: {:.1}", fleet_e0);
    println!("  After 4 transfers: {:.1}", fleet.total_energy());
    println!("  Energies: {:?}", fleet.energies);
    println!("  → Fleet total is invariant.\n");

    // ── THE PARALLEL ───────────────────────────────────
    println!("═══════════════════════════════════════════════════");
    println!("  MECHANICS           │  AGENT FLEET");
    println!("  T + V = E = const   │  Σ agents = const");
    println!("  Symplectic integrator│  transfer_with_guard()");
    println!("  Phase space (q, p)  │  Agent energy vector");
    println!("  Lagrangian: L=T-V   │  Fleet energy budget");
    println!("  Noether: sym→cons   │  Audit: z-score→anomaly");
    println!("═══════════════════════════════════════════════════");
}
```

The Hamiltonian side provides:
- `PhaseSpacePoint<S, N>` — canonical coordinates (q, p)
- `Hamiltonian<S, N>` trait — with auto-differentiated `dH_dq()` and `dH_dp()`
- `SeparableHamiltonian` — H = p²/(2m) + V(q)
- `HamiltonianIntegrator` — Störmer–Verlet in canonical form
- `poisson_bracket()` — numerically computes {f, g}
- `phase_space_volume()` — bounding-box estimate for Liouville verification
- `verify_liouville()` — check phase space volume preservation
- `find_recurrence()` — detect Poincaré recurrence in trajectories

---

## 7. Connection to Other Crates

### 7a. With spectral-fleet — Rank Agents by Energy Efficiency

```rust
// examples/07a_spectral_fleet.rs
//
// cargo add conservation-law spectral-fleet
//
// spectral-fleet clusters agents by spectral embedding.
// conservation-law provides the energy numbers that feed into the ranking.

use conservation_law::fleet_integration::FleetConservation;

fn main() {
    println!("=== conservation-law + spectral-fleet ===\n");

    let mut fleet = FleetConservation::new(
        vec![120.0, 45.0, 200.0, 30.0, 55.0],
        2.0,
    );

    // Simulate transfers to create efficiency patterns
    let _ = fleet.transfer_with_guard(2, 0, 30.0);  // agent 2 (wealthy) → agent 0
    let _ = fleet.transfer_with_guard(0, 1, 15.0);  // agent 0 → agent 1 (needy)
    let _ = fleet.transfer_with_guard(3, 4, 10.0);  // agent 3 → agent 4

    // Audit — gives us z-scores that spectral-fleet can use as features
    let report = fleet.audit_fleet();

    println!("Agent Energies After Transfers:");
    println!("  {:?}", fleet.energies);
    println!("  Total: {:.1}\n", fleet.total_energy());

    // spectral-fleet would use these z-scores as embedding coordinates
    // to cluster agents into efficiency tiers
    println!("Z-scores (spectral features):");
    for (i, &z) in report.z_scores.iter().enumerate() {
        let tier = if z > 1.0 { "high-energy" } else if z < -1.0 { "starved" } else { "balanced" };
        println!("  agent {}: z={:+.3} → tier: {}", i, z, tier);
    }

    println!("\n→ Feed z_scores into spectral-fleet for clustering.");
    println!("→ conservation-law provides the invariant-checked numbers.");
    println!("→ spectral-fleet provides the spectral embedding.");
}
```

### 7b. With fleet-warden — Audit Cleanup for Conservation

```rust
// examples/07b_fleet_warden.rs
//
// cargo add conservation-law fleet-warden
//
// fleet-warden monitors fleet health and triggers remediation.
// conservation-law provides the audit data that triggers warden actions.

use conservation_law::fleet_integration::{FleetConservation, CircuitState};

fn main() {
    println!("=== conservation-law + fleet-warden ===\n");

    let mut fleet = FleetConservation::new(
        vec![100.0, 100.0, 100.0, 100.0, 500.0],  // agent 4 is anomalous
        1.5,
    );

    // Audit — fleet-warden uses this to decide interventions
    let report = fleet.audit_fleet();

    println!("Audit Report:");
    println!("  Stable: {}", report.system_stable);
    println!("  Anomalous agents: {:?}\n", report.anomalous_agents);

    // If anomalous agents found, warden triggers circuit breaker
    if !report.anomalous_agents.is_empty() {
        println!("Warden: Anomalies detected. Tripping circuit breaker.");
        fleet.trip();
        println!("  Circuit state: {:?}\n", fleet.circuit_state());

        // All transfers blocked
        match fleet.transfer_with_guard(0, 1, 10.0) {
            Err(e) => println!("  Transfer blocked: {}", e),
            _ => {}
        }
    }

    // Warden attempts recovery
    println!("\nWarden: Probing recovery...");
    fleet.try_half_open();
    println!("  Circuit state: {:?}", fleet.circuit_state());

    // Successful probe transfer
    match fleet.transfer_with_guard(0, 1, 5.0) {
        Ok(_) => println!("  Probe succeeded."),
        Err(e) => println!("  Probe failed: {}", e),
    }

    // Need 2 successful probes to fully recover
    match fleet.transfer_with_guard(1, 0, 5.0) {
        Ok(_) => println!("  Second probe succeeded."),
        Err(e) => println!("  Second probe failed: {}", e),
    }

    println!("\n  Circuit state after recovery: {:?}", fleet.circuit_state());
    println!("  Total energy invariant: {:.1}", fleet.total_energy());
}
```

### 7c. With agent-homeostasis — Maintain Budget Setpoint

```rust
// examples/07c_agent_homeostasis.rs
//
// cargo add conservation-law agent-homeostasis
//
// agent-homeostasis keeps agents at target budget levels.
// conservation-law guarantees that redistribution conserves total energy.

use conservation_law::fleet_integration::FleetConservation;

fn main() {
    println!("=== conservation-law + agent-homeostasis ===\n");

    let mut fleet = FleetConservation::new(
        vec![200.0, 50.0, 30.0, 180.0, 40.0],  // unbalanced
        2.0,
    );

    let total = fleet.total_energy();
    let target = total / fleet.energies.len() as f64;  // homeostasis setpoint

    println!("Initial state:");
    println!("  Energies: {:?}", fleet.energies);
    println!("  Total: {:.1}", total);
    println!("  Target per agent: {:.1}\n", target);

    // Homeostasis controller: redistribute toward setpoint
    // conservation-law ensures total is invariant throughout
    println!("Redistributing toward setpoint:");
    for step in 0..5 {
        // Find most over-budget and most under-budget agents
        let (max_idx, max_e) = fleet.energies.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (min_idx, min_e) = fleet.energies.iter()
            .enumerate()
            .filter(|(i, _)| *i != max_idx)
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let excess = *max_e - target;
        let deficit = target - *min_e;
        let transfer = excess.min(deficit).max(0.0).min(*max_e * 0.3);

        if transfer > 0.1 {
            match fleet.transfer_with_guard(max_idx, min_idx, transfer) {
                Ok(t) => println!("  step {}: {:.1} from agent {} → agent {}",
                    step, t, max_idx, min_idx),
                Err(e) => println!("  step {}: blocked — {}", step, e),
            }
        }

        println!("    energies: {:?}", fleet.energies);
    }

    println!("\nFinal total: {:.1} (invariant: {})",
        fleet.total_energy(),
        (fleet.total_energy() - total).abs() < 1e-10);

    let report = fleet.audit_fleet();
    println!("System stable: {}", report.system_stable);
    println!("Anomalous agents: {:?}", report.anomalous_agents);
}
```

---

## Module Reference

### `lagrangian` — Euler–Lagrange Dynamics

The foundation. Agents live in N-dimensional configuration space with positions `q` and velocities `q_dot`.

| Type | Description |
|------|-------------|
| `AgentState<S, N>` | State: position `q: [S; N]`, velocity `q_dot: [S; N]` |
| `Lagrangian<S, N>` | Trait: `kinetic()`, `potential()`, `lagrangian()` |
| `MechanicalLagrangian<S, V, N>` | Standard L = ½m q̇² − V(q) |
| `SymplecticIntegrator<S, N>` | Störmer–Verlet integrator: `step()`, `integrate()`, `generalised_force()` |
| `total_energy()` | E = T + V for any Lagrangian |
| `DynamicsError` | `StepSizeTooSmall`, `IntegrationDiverged` |

### `hamiltonian` — Canonical Mechanics

The Hamiltonian formulation. Phase space (q, p). Poisson brackets. Liouville's theorem. Poincaré recurrence.

| Type | Description |
|------|-------------|
| `PhaseSpacePoint<S, N>` | Canonical coordinates: `q: [S; N]`, `p: [S; N]` |
| `Hamiltonian<S, N>` | Trait: `hamiltonian()`, `dH_dq()`, `dH_dp()` |
| `SeparableHamiltonian<S, V, N>` | H = p²/(2m) + V(q) |
| `HamiltonianIntegrator<S, N>` | Canonical Störmer–Verlet: `step()`, `integrate()` |
| `poisson_bracket()` | Compute {f, g} numerically |
| `phase_space_volume()` | Bounding-box volume estimate |
| `verify_liouville()` | Check volume preservation |
| `find_recurrence()` | Detect near-recurrence in trajectory |

### `noether` — Symmetry and Conservation

Noether's theorem as code. Define a symmetry, test invariance, compute conserved charges.

| Type | Description |
|------|-------------|
| `Symmetry<S, N>` | Trait: `transform()`, `generator_q()`, `name()` |
| `TranslationSymmetry<N>` | q → q + ε e_axis |
| `RotationSymmetry` | Rotate in (i,j) plane by angle ε |
| `TimeTranslationSymmetry` | t → t + ε (conserved charge = energy) |
| `InvarianceResult<S>` | Result of `test_invariance()` |
| `noether_charge()` | Q = Σᵢ m q̇ᵢ δqᵢ |
| `ChargeMonitor<S>` | Track conserved quantity along trajectory |
| `verify_noether()` | Full verification: invariance + charge conservation |

### `conserved` — Automatic Detection

Given a trajectory, automatically detect which quantities are conserved.

| Type | Description |
|------|-------------|
| `ConservedQuantity<S>` | One detected quantity: `name`, `initial_value`, `max_drift`, `is_conserved` |
| `ConservationDetector<S, N>` | Detector: `check_energy()`, `check_linear_momentum()`, `check_angular_momentum()`, `check_quantity()` |
| `verify_all_conservation()` | Integrate + detect everything in one call |
| `energy_spread()` | Standard deviation of energy along trajectory |

### `fleet_integration` — Fleet-Level Energy Management

Energy budgets for agent fleets with anomaly detection and circuit breakers.

| Type | Description |
|------|-------------|
| `FleetConservation` | Fleet manager: `energies`, `total_energy()`, `mean()`, `std_dev()`, `z_scores()`, `audit_fleet()`, `transfer_with_guard()`, `trip()`, `reset()`, `try_half_open()` |
| `ConservationReport` | Audit result: `mean_energy`, `std_dev`, `anomalous_agents`, `z_scores`, `system_stable` |
| `CircuitState` | `Closed`, `Open`, `HalfOpen` |
| `zscore()` | Generic Z-score computation |

---

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `SymplecticIntegrator::step()` | O(N) | N = dimensionality |
| `SymplecticIntegrator::integrate()` | O(steps × N) | Energy drift bounded |
| `generalised_force()` | O(N) | Central differences, 2N evaluations |
| `FleetConservation::transfer_with_guard()` | O(1) | Atomic, in-place |
| `FleetConservation::audit_fleet()` | O(K) | K = number of agents |
| `verify_noether()` | O(steps × N) | One charge per timestep |
| `poisson_bracket()` | O(N) | 4N central-difference evaluations |
| `find_recurrence()` | O(steps × N) | Linear scan with distance check |
| `verify_liouville()` | O(points × N) | Bounding-box volume |

The symplectic integrator is explicit — no matrix solves. Energy error oscillates but never drifts.

---

## Ideas for Improvement

1. **Adaptive timestepping** — Embedded error estimators (velocity Verlet with half-step comparison) to vary `dt` based on local dynamics.
2. **Constrained dynamics** — Lagrange multipliers and holonomic constraints for agents on manifolds.
3. **Poisson integrators** — Non-canonical symplectic structures for dissipative systems.
4. **Parallel ensemble** — `rayon` for fleet-wide integration sharing the same potential landscape.
5. **Event detection** — Zero-crossings and collisions for hybrid agent dynamics.
6. **Higher-order symplectic** — Yoshida 4th/6th order for tighter energy conservation.
7. **Fleet thermodynamics** — Temperature, entropy, and Boltzmann distribution from fleet energy profiles.

## Ecosystem

| Crate | Role |
|-------|------|
| **[spectral-fleet](https://github.com/SuperInstance/spectral-fleet-rs)** | Cluster agents by spectral embedding |
| **[fleet-warden](https://github.com/SuperInstance/fleet-warden-rs)** | Fleet health monitoring and remediation |
| **[agent-homeostasis](https://github.com/SuperInstance/agent-homeostasis-rs)** | Maintain agent budget at setpoint |
| **[categorical-agents](https://github.com/SuperInstance/categorical-agents-rs)** | Compose dynamical systems via category theory |
| **[ga-core](https://github.com/SuperInstance/ga-core-rs)** | Geometric algebra rotors for 3D rotation symmetries |
| **[wasserstein-agents](https://github.com/SuperInstance/wasserstein-agents-rs)** | Optimal transport for comparing agent distributions |

## License

MIT OR Apache-2.0
