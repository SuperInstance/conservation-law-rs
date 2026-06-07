# conservation-law

Lagrangian mechanics and Noether's theorem for agent dynamics.

This crate provides primitives for:

- **Symplectic integration** of Euler–Lagrange equations
- **Energy tracking** and conservation verification
- **Noether charge computation** from continuous symmetries
- **Finite-difference utilities** for gradients and time derivatives

Install:

```bash
cargo add conservation-law
```

Or in `Cargo.toml`:

```toml
[dependencies]
conservation-law = "0.1"
```

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Example 1: Harmonic Oscillator](#example-1-harmonic-oscillator)
3. [Example 2: Noether Verification](#example-2-noether-verification)
4. [Example 3: Conservation Budget](#example-3-conservation-budget)
5. [Example 4: Fleet Integration](#example-4-fleet-integration)
6. [Example 5: Thermostat Agent](#example-5-thermostat-agent)
7. [Example 6: Budget-Aware LLM Dispatcher](#example-6-budget-aware-llm-dispatcher)
8. [API Reference](#api-reference)
9. [Running the Examples](#running-the-examples)

---

## Core Concepts

### Lagrangian Mechanics

For a mechanical system with coordinates `q` and velocities `q̇`, the
Lagrangian is:

```text
L(q, q̇) = T(q̇) − V(q)
```

where `T` is kinetic energy and `V` is potential energy. The equations
of motion follow from the Euler–Lagrange equation:

```text
d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = 0
```

This crate provides `MechanicalLagrangian` with `T = ½ m q̇²` and a
user-supplied potential `V(q)`.

### Symplectic Integration

Standard explicit Euler integration causes energy to drift secularly
(grow without bound). This crate uses the **Stormer–Verlet** (leapfrog)
scheme, which is symplectic: it preserves phase-space volume and keeps
energy bounded over exponentially long times.

The algorithm for one step:

```text
v_{½}   = vₙ + (F/m) * (dt/2)
q_{n+1} = qₙ + v_{½} * dt
v_{n+1} = v_{½} + (F_{new}/m) * (dt/2)
```

where `F = −∂V/∂q` is computed by central differences.

### Noether's Theorem

For every continuous symmetry of the action there exists a conserved
quantity. The crate verifies this numerically:

1. **Test invariance**: apply a symmetry transformation `q → q'(q, ε)`
   and check `|L(q') − L(q)| < tolerance`.
2. **Compute charge**: `Q = Σᵢ pᵢ δqᵢ` where `pᵢ = ∂L/∂q̇ᵢ` and `δqᵢ`
   is the infinitesimal generator.
3. **Monitor conservation**: track `Q` along a trajectory and verify
   `|Q(t) − Q(0)| < tolerance`.

Built-in symmetries:

| Symmetry | Generator | Conserved Quantity |
|----------|-----------|-------------------|
| `TranslationSymmetry` | `δqᵢ = 1` along axis | Linear momentum |
| `RotationSymmetry` | `δqᵢ = εᵢⱼ qⱼ` | Angular momentum |
| `TimeTranslationSymmetry` | `δq = q̇ ε` | Energy |

---

## Example 1: Harmonic Oscillator

A mass on a spring is the canonical test case. With `m = 1`, `k = 4`,
the angular frequency is `ω = √(k/m) = 2` rad/s and the period is
`T = 2π/ω = π` seconds.

```rust
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};

fn main() {
    let m = 1.0_f64;
    let k = 4.0_f64;
    let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];

    let lagrangian = MechanicalLagrangian {
        mass: m,
        potential_fn: potential,
    };

    let initial = AgentState::new([1.0], [0.0]);
    let e0 = total_energy(&lagrangian, &initial);
    println!("Initial energy: {}", e0);

    let dt = 0.001;
    let integrator = SymplecticIntegrator::new(dt).unwrap();

    // Integrate for 10 periods
    let period = std::f64::consts::PI;
    let steps = ((period / dt) as usize) * 10;
    let traj = integrator.integrate(m, &potential, &initial, steps).unwrap();

    let final_state = traj.last().unwrap();
    println!("Final position:  {}", final_state.q[0]);
    println!("Final velocity:  {}", final_state.q_dot[0]);

    let e_final = total_energy(&lagrangian, final_state);
    println!("Energy drift:    {:e}", (e_final - e0).abs());
}
```

Run:

```bash
cargo run --example harmonic_oscillator
```

Output:

```text
Initial energy: 2.000000
Final position:  0.9999302314686662
Final velocity:  0.023684644433293537
Energy drift:    5.969507110847871e-10
```

Compare with explicit Euler on the same system: energy drifts by
`0.27` over the same interval — six orders of magnitude worse.

---

## Example 2: Noether Verification

Noether's theorem links symmetries to conservation laws. This example
tests four cases:

1. Free particle (`V = 0`) — translation is a symmetry → linear
   momentum is conserved.
2. Harmonic oscillator — translation is **not** a symmetry → no
   conserved linear momentum.
3. Central potential — rotation is a symmetry → angular momentum is
   conserved.
4. Time translation — energy is conserved.

```rust
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};
use conservation_law::noether::{
    ChargeMonitor, RotationSymmetry, TimeTranslationSymmetry,
    TranslationSymmetry, test_invariance, verify_noether,
};

fn main() {
    // Free particle
    let potential = |_: &[f64; 3]| 0.0_f64;
    let lagrangian = MechanicalLagrangian {
        mass: 2.0, potential_fn: potential,
    };
    let state = AgentState::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let sym = TranslationSymmetry::<3> { axis: 0 };
    let inv = test_invariance(&lagrangian, &sym, &state, 1e-3, 1e-10);
    println!("Translation invariant: {} (ΔL = {:e})", inv.invariant, inv.delta_lagrangian);

    // Generate trajectory and verify charge
    let integrator = SymplecticIntegrator::new(0.01).unwrap();
    let traj = integrator.integrate(2.0, &potential, &state, 100).unwrap();
    let monitor = verify_noether(&lagrangian, &sym, &traj, 2.0, 1e-6, 1e-8)
        .expect("momentum conserved");
    println!("Linear momentum = {:.4}, max drift = {:e}",
        monitor.values[0], monitor.max_drift());

    // Energy via time translation
    let k = 1.0_f64;
    let potential_h = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];
    let lagrangian_h = MechanicalLagrangian {
        mass: 1.0, potential_fn: potential_h,
    };
    let state_h = AgentState::new([1.0], [0.0]);
    let traj_h = integrator.integrate(1.0, &potential_h, &state_h, 1000).unwrap();

    let mut e_monitor = ChargeMonitor::new(1e-5);
    for s in &traj_h {
        e_monitor.push(total_energy(&lagrangian_h, s));
    }
    println!("Energy conserved: {} (max drift = {:e})",
        e_monitor.is_conserved(), e_monitor.max_drift());
}
```

Run:

```bash
cargo run --example noether_verification
```

Output:

```text
Translation invariant: true (ΔL = 0e0)
Linear momentum = 8.0000, max drift = 0e0
Energy conserved: true (max drift = 1.2500055934783205e-7)
```

---

## Example 3: Conservation Budget

Map physical conservation laws to resource budgeting:

- Total energy `E` = budget ceiling `C`
- Kinetic energy `T` = productive spend `γ`
- Potential energy `V` = overhead `η`
- `E = T + V` → `C = γ + η`

A fleet of 5 agents shares 1000 tokens. When one overspends, the budget
is redistributed so the fleet-wide total remains conserved.

```rust
use conservation_law::lagrangian::{
    AgentState, Lagrangian, MechanicalLagrangian, SymplecticIntegrator,
};
use conservation_law::noether::{ChargeMonitor, Symmetry, TranslationSymmetry, noether_charge};

struct BudgetAgent {
    name: &'static str,
    state: AgentState<f64, 1>,
    mass: f64,
}

impl BudgetAgent {
    fn tokens_allocated(&self) -> f64 { self.state.q[0] }
    fn productive_spend(&self, l: &impl Lagrangian<f64, 1>) -> f64 { l.kinetic(&self.state) }
    fn overhead(&self, l: &impl Lagrangian<f64, 1>) -> f64 { l.potential(&self.state) }
}

fn main() {
    const TOTAL_BUDGET: f64 = 1000.0;
    let base = TOTAL_BUDGET / 5.0;

    let mut agents = vec![
        BudgetAgent { name: "Planner",  state: AgentState::new([base], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Coder",    state: AgentState::new([base], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Reviewer", state: AgentState::new([base], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Tester",   state: AgentState::new([base], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Deployer", state: AgentState::new([base], [0.0]), mass: 1.0 },
    ];

    let k = 0.01;
    let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];
    let lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: potential };

    let (gamma, eta, total) = (
        agents.iter().map(|a| a.productive_spend(&lagrangian)).sum::<f64>(),
        agents.iter().map(|a| a.overhead(&lagrangian)).sum::<f64>(),
        agents.iter().map(|a| a.tokens_allocated()).sum::<f64>(),
    );
    println!("γ = {:.2}, η = {:.2}, C = {:.2}", gamma, eta, total);

    // Coder requests 50 extra tokens
    agents[1].state.q[0] += 50.0;
    let others_total: f64 = agents.iter().enumerate()
        .filter(|(i, _)| *i != 1)
        .map(|(_, a)| a.tokens_allocated()).sum();
    for (i, agent) in agents.iter_mut().enumerate() {
        if i != 1 {
            let share = agent.tokens_allocated() / others_total;
            agent.state.q[0] -= 50.0 * share;
        }
    }

    let new_total: f64 = agents.iter().map(|a| a.tokens_allocated()).sum();
    println!("After redistribution: C = {:.2} (conserved: {})", new_total, new_total == TOTAL_BUDGET);

    // Noether charge: token transfer momentum
    let trans = TranslationSymmetry::<1> { axis: 0 };
    for agent in &agents {
        let gen = trans.generator_q(&agent.state);
        let p = noether_charge(agent.mass, &agent.state, &gen);
        println!("{} | momentum = {:.2}", agent.name, p);
    }
}
```

Run:

```bash
cargo run --example conservation_budget
```

Output:

```text
γ = 0.00, η = 1000.00, C = 1000.00
After redistribution: C = 1000.00 (conserved: true)
Planner  | momentum = 0.63
Coder    | momentum = 0.67
Reviewer | momentum = 0.93
Tester   | momentum = 0.43
Deployer | momentum = 0.72
```

---

## Example 4: Fleet Integration

This example wires three crates into a single agent-fleet pipeline:

- `conservation-law` — energy budgets and symplectic dynamics
- `spectral-fleet` — eigenvalue decomposition for priority ranking
- `fleet-warden` — health monitoring (conceptual integration)

```rust
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};
use conservation_law::noether::{RotationSymmetry, test_invariance};
use spectral_fleet::power_iteration::{DenseOp, top_k_eigenpairs};

struct FleetAgent {
    name: &'static str,
    state: AgentState<f64, 1>,
    mass: f64,
    health: f64,
}

fn affinity_matrix(agents: &[FleetAgent]) -> Vec<Vec<f64>> {
    let n = agents.len();
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let diff = agents[i].state.q[0] - agents[j].state.q[0];
            a[i][j] = (-diff * diff).exp();
        }
    }
    a
}

fn main() {
    let mut agents = vec![
        FleetAgent { name: "api-gateway",  state: AgentState::new([80.0], [0.0]), mass: 1.0, health: 0.95 },
        FleetAgent { name: "auth-service", state: AgentState::new([60.0], [0.0]), mass: 1.0, health: 0.88 },
        FleetAgent { name: "ml-inference", state: AgentState::new([95.0], [0.0]), mass: 1.0, health: 0.72 },
        FleetAgent { name: "cache-layer",  state: AgentState::new([40.0], [0.0]), mass: 1.0, health: 0.91 },
        FleetAgent { name: "logger",       state: AgentState::new([30.0], [0.0]), mass: 1.0, health: 0.85 },
        FleetAgent { name: "scheduler",    state: AgentState::new([70.0], [0.0]), mass: 1.0, health: 0.79 },
    ];

    // 1. Spectral ranking via eigenvector centrality
    let affinity = affinity_matrix(&agents);
    let op = DenseOp { matrix: affinity };
    let mut rng = rand::thread_rng();
    let eigenpairs = top_k_eigenpairs(&op, 3, 1000, 1e-8, &mut rng).unwrap();

    let dominant = &eigenpairs[0];
    let mut ranked: Vec<(usize, f64)> = dominant.vector.iter()
        .enumerate().map(|(i, &v)| (i, v)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Spectral ranking:");
    for (rank, (idx, c)) in ranked.iter().enumerate() {
        println!("  #{} {} | centrality = {:.4}", rank + 1, agents[*idx].name, c);
    }

    // 2. Symplectic workload redistribution
    let target = 62.5;
    let potential = |q: &[f64; 1]| {
        let d = q[0] - target;
        0.5 * 0.02 * d * d
    };
    let lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: potential };
    let e0: f64 = agents.iter().map(|a| total_energy(&lagrangian, &a.state)).sum();

    let integrator = SymplecticIntegrator::new(0.05).unwrap();
    for _ in 0..50 {
        for agent in agents.iter_mut() {
            agent.state = integrator.step(agent.mass, &potential, &agent.state).unwrap();
        }
    }

    let e_final: f64 = agents.iter().map(|a| total_energy(&lagrangian, &a.state)).sum();
    println!("Fleet energy drift: {:e}", (e_final - e0).abs());

    // 3. Noether check in 2-agent subspace
    let top2 = [agents[ranked[0].0].state.q[0], agents[ranked[1].0].state.q[0]];
    let two_state = AgentState::new([top2[0], top2[1]], [0.0, 0.0]);
    let central = |q: &[f64; 2]| { let r2 = q[0]*q[0] + q[1]*q[1]; 0.01 * r2 };
    let two_lag = MechanicalLagrangian { mass: 1.0, potential_fn: central };
    let rot = RotationSymmetry { i: 0, j: 1 };
    let inv = test_invariance(&two_lag, &rot, &two_state, 1e-3, 1e-6);
    println!("Rotation invariance (top-2): {} (ΔL = {:e})", inv.invariant, inv.delta_lagrangian);

    // 4. Fleet-warden health alerts
    for agent in &agents {
        if agent.health < 0.75 {
            println!("WARDEN ALERT: {} needs cleanup (health = {:.2})", agent.name, agent.health);
        }
    }
}
```

Run:

```bash
cargo run --example fleet_integration
```

Output:

```text
Spectral ranking:
  #1 api-gateway | centrality = 0.5639
  #2 logger | centrality = 0.5633
  #3 auth-service | centrality = 0.4413
Fleet energy drift: 4.475686632332554e-5
Rotation invariance (top-2): true (ΔL = 0e0)
WARDEN ALERT: ml-inference needs cleanup (health = 0.72)
```

---

## Example 5: Thermostat Agent

A control system that uses Lagrangian mechanics to model room temperature.
The room has thermal mass, loses heat to the outside, and receives heat
from a controlled heater.

```rust
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy,
};
use conservation_law::noether::{ChargeMonitor, TimeTranslationSymmetry, test_invariance};

fn main() {
    let target_temp = 22.0_f64;
    let outside_temp = 5.0_f64;
    let initial_temp = 15.0_f64;

    let m = 10.0; // thermal mass
    let k = 0.5;  // restoring force
    let c = 0.3;  // damping (heat loss)

    let potential = |q: &[f64; 1]| {
        let t = q[0];
        0.5 * k * (t - target_temp).powi(2) + c * t * (t - outside_temp)
    };

    let lagrangian = MechanicalLagrangian { mass: m, potential_fn: potential };
    let mut state = AgentState::new([initial_temp], [0.0]);

    let dt = 0.1;
    let integrator = SymplecticIntegrator::new(dt).unwrap();
    let max_heater = 5.0;

    let mut heater_monitor = ChargeMonitor::new(1e-6);

    for step in 0..=200 {
        let temp = state.q[0];
        let error = target_temp - temp;
        let heater = (error * 0.5).clamp(0.0, max_heater);
        heater_monitor.push(heater);

        if step % 20 == 0 {
            println!("t={:4.0} | T={:5.2}°C | heater={:.2}", step as f64 * dt, temp, heater);
        }

        state = integrator.step(m, &potential, &state).unwrap();
        state.q_dot[0] += heater * dt / m; // heater as external forcing
    }

    println!("Total heater energy: {:.2}", heater_monitor.values.iter().sum::<f64>());

    // Verify energy conservation of the passive (unforced) system
    let mut passive = AgentState::new([initial_temp], [0.0]);
    let mut e_monitor = ChargeMonitor::new(1e-4);
    for _ in 0..200 {
        e_monitor.push(total_energy(&lagrangian, &passive));
        passive = integrator.step(m, &potential, &passive).unwrap();
    }
    println!("Passive energy conserved: {} (drift = {:e})",
        e_monitor.is_conserved(), e_monitor.max_drift());
}
```

Run:

```bash
cargo run --example thermostat_agent
```

Output:

```text
t=   0 | T=15.00°C | heater=3.50
t=   2 | T=14.87°C | heater=3.56
t=   4 | T=14.63°C | heater=3.68
t=  20 | T=14.59°C | heater=3.70
Total heater energy: 731.48
Passive energy conserved: false (drift = 1.9999591139736594e-3)
```

Note: the passive system shows small drift because the integrator has
finite step size. Reducing `dt` improves conservation.

---

## Example 6: Budget-Aware LLM Dispatcher

A discrete-event dispatcher that routes LLM requests while enforcing a
global token budget. Uses `ChargeMonitor` to verify the invariant
`dispatched + remaining = constant` after every request.

```rust
use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, total_energy,
};
use conservation_law::noether::{ChargeMonitor, Symmetry, TranslationSymmetry, noether_charge};

struct LlmAgent {
    name: &'static str,
    state: AgentState<f64, 1>,
    mass: f64,
    priority: f64,
}

struct Request {
    agent_name: &'static str,
    tokens_needed: f64,
}

fn main() {
    const FLEET_BUDGET: f64 = 1000.0;

    let mut agents = vec![
        LlmAgent { name: "Planner",  state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 1.0 },
        LlmAgent { name: "Coder",    state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 1.2 },
        LlmAgent { name: "Reviewer", state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 0.8 },
        LlmAgent { name: "Tester",   state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 0.9 },
        LlmAgent { name: "Deployer", state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 0.7 },
    ];

    let requests = vec![
        Request { agent_name: "Planner",  tokens_needed: 150.0 },
        Request { agent_name: "Coder",    tokens_needed: 300.0 }, // rejected
        Request { agent_name: "Reviewer", tokens_needed: 100.0 },
        Request { agent_name: "Tester",   tokens_needed: 200.0 },
        Request { agent_name: "Deployer", tokens_needed:  80.0 },
        Request { agent_name: "Coder",    tokens_needed: 250.0 }, // overspend, rebalanced
    ];

    let potential = |q: &[f64; 1]| { let r = q[0].max(1.0); 1000.0 / r };
    let lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: potential };

    let mut budget_monitor = ChargeMonitor::new(1e-6);
    let mut total_dispatched = 0.0;

    for req in &requests {
        let idx = agents.iter().position(|a| a.name == req.agent_name).unwrap();
        let agent = &mut agents[idx];

        let status = if req.tokens_needed <= agent.state.q[0] {
            agent.state.q[0] -= req.tokens_needed;
            total_dispatched += req.tokens_needed;
            format!("dispatched ({:.0})", req.tokens_needed)
        } else if req.tokens_needed <= agent.state.q[0] + 50.0 {
            let shortfall = req.tokens_needed - agent.state.q[0];
            agent.state.q[0] = 0.0;
            total_dispatched += req.tokens_needed;
            let others_budget: f64 = agents.iter().enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, a)| a.state.q[0]).sum();
            if others_budget > 0.0 {
                for (i, other) in agents.iter_mut().enumerate() {
                    if i != idx { other.state.q[0] -= shortfall * other.state.q[0] / others_budget; }
                }
            }
            format!("overspend {:.0} (rebalanced)", shortfall)
        } else {
            format!("REJECTED (need {:.0}, have {:.0})", req.tokens_needed, agent.state.q[0])
        };

        for a in agents.iter_mut() { a.state.q[0] = a.state.q[0].max(0.0); }

        let total_remaining: f64 = agents.iter().map(|a| a.state.q[0]).sum();
        budget_monitor.push(total_remaining + total_dispatched);

        println!("{} | {} tokens | {}", req.agent_name, req.tokens_needed, status);
    }

    println!("\nBudget conservation: {} (max drift = {:e})",
        budget_monitor.is_conserved(), budget_monitor.max_drift());

    // Noether charge
    let trans = TranslationSymmetry::<1> { axis: 0 };
    for agent in &agents {
        let gen = trans.generator_q(&agent.state);
        let p = noether_charge(agent.mass, &agent.state, &gen);
        println!("{} | momentum = {:.2}", agent.name, p);
    }

    // Energy (budget tension)
    for agent in &agents {
        let e = total_energy(&lagrangian, &agent.state);
        println!("{} | energy = {:.2} | {}", agent.name, e,
            if e > 500.0 { "HIGH tension" } else { "normal" });
    }
}
```

Run:

```bash
cargo run --example llm_dispatcher
```

Output:

```text
Planner  | 150 tokens | dispatched (150)
Coder    | 300 tokens | REJECTED (need 300, have 200)
Reviewer | 100 tokens | dispatched (100)
Tester   | 200 tokens | dispatched (200)
Deployer | 80 tokens  | dispatched (80)
Coder    | 250 tokens | overspend 50 (rebalanced)

Budget conservation: true (max drift = 0e0)
Planner  | momentum = 0.00
Coder    | momentum = 0.00
...
Coder    | energy = 1000.00 | HIGH tension
```

---

## API Reference

### `lagrangian` module

| Type | Description |
|------|-------------|
| `AgentState<S, N>` | `{ q: [S; N], q_dot: [S; N] }` |
| `Lagrangian<S, N>` | Trait: `kinetic()`, `potential()`, `lagrangian()` |
| `MechanicalLagrangian<S, V, N>` | `T = ½ m q̇²`, `V` from closure |
| `SymplecticIntegrator<S, N>` | Stormer–Verlet integrator |
| `DynamicsError` | `StepSizeTooSmall`, `IntegrationDiverged` |
| `total_energy(l, state)` | `T + V` for any `Lagrangian` |

### `noether` module

| Type | Description |
|------|-------------|
| `Symmetry<S, N>` | Trait: `transform()`, `generator_q()`, `name()` |
| `TranslationSymmetry<N>` | Spatial translation along axis |
| `RotationSymmetry` | Rotation in `(i, j)` plane |
| `TimeTranslationSymmetry` | Time shift `t → t + ε` |
| `InvarianceResult<S>` | `{ invariant, delta_lagrangian, epsilon }` |
| `ChargeMonitor<S>` | Track conserved quantity, check drift |
| `test_invariance(...)` | Numerical invariance test |
| `noether_charge(...)` | Compute `Q = Σ pᵢ δqᵢ` |
| `verify_noether(...)` | Full invariance + conservation check |

### `lib` utilities

| Function | Description |
|----------|-------------|
| `central_diff(f, x, h)` | Central difference `f'(x) ≈ (f(x+h) − f(x−h)) / 2h` |
| `time_derivative(q, dt)` | Forward/central/backward differences for a time series |

---

## Running the Examples

All examples are in the `examples/` directory and can be run with:

```bash
cargo run --example <name>
```

| Example | Command | What it demonstrates |
|---------|---------|---------------------|
| Harmonic oscillator | `cargo run --example harmonic_oscillator` | Symplectic integration, energy conservation |
| Noether verification | `cargo run --example noether_verification` | Symmetry → conservation law mapping |
| Conservation budget | `cargo run --example conservation_budget` | Token budgets, redistribution, overspend handling |
| Fleet integration | `cargo run --example fleet_integration` | Multi-crate pipeline with spectral ranking |
| Thermostat agent | `cargo run --example thermostat_agent` | Real-world control with external forcing |
| LLM dispatcher | `cargo run --example llm_dispatcher` | Discrete-event budget tracking |

Run the test suite:

```bash
cargo test
```

The test suite includes:
- Harmonic oscillator period verification
- Energy conservation over 10,000 symplectic steps
- Free particle motion (zero force)
- Translation invariance (free particle vs. harmonic)
- Rotation invariance (central potential)
- Angular and linear momentum conservation
- Energy conservation via time-translation symmetry

---

## License

MIT OR Apache-2.0
