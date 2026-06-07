# conservation-law

> Lagrangian mechanics, symplectic integration, and Noether's theorem for agent dynamics.

## What This Does

This crate models agent trajectories as physical systems governed by Lagrangian mechanics. It provides a `SymplecticIntegrator` that evolves agents through configuration space while preserving energy, tools for defining custom potentials and symmetries, and numerical verification of Noether's theorem — ensuring that every continuous symmetry of your agent's dynamics produces a rigorously conserved quantity. If you want your agents to move with physical coherence rather than drifting numerically, this is the foundation.

## Why It Matters

AGI systems will not be static databases — they will be dynamic entities that act, move, and evolve over time. Conservation laws are the difference between systems that accumulate error and systems that maintain invariant structure. By grounding agent dynamics in Lagrangian mechanics, we ensure that energy, momentum, and angular momentum are not merely hoped for but mathematically guaranteed. A fleet of agents that respects conservation laws is a fleet that scales without chaos.

## Quick Start

```bash
cargo add conservation-law
```

```rust
use conservation_law::lagrangian::{AgentState, MechanicalLagrangian, SymplecticIntegrator, total_energy};
use conservation_law::noether::{RotationSymmetry, verify_noether};

fn main() {
    // Harmonic oscillator: V = ½ k r²
    let k = 1.0;
    let m = 1.0;
    let potential = |q: &[f64; 2]| 0.5 * k * (q[0] * q[0] + q[1] * q[1]);

    let lagrangian = MechanicalLagrangian {
        mass: m,
        potential_fn: potential,
    };

    let dt = 0.001;
    let integrator = SymplecticIntegrator::new(dt).unwrap();
    let initial = AgentState::new([1.0, 0.0], [0.0, 1.0]);

    // Integrate 5000 steps — energy stays bounded
    let traj = integrator.integrate(m, &potential, &initial, 5000).unwrap();

    // Verify Noether: central potential → angular momentum conserved
    let sym = RotationSymmetry { i: 0, j: 1 };
    let monitor = verify_noether(&lagrangian, &sym, &traj, m, 1e-6, 1e-8).unwrap();
    println!("Max angular momentum drift: {:?}", monitor.max_drift());
}
```

## Architecture

| Module | Purpose |
|--------|---------|
| `lagrangian` | Euler–Lagrange dynamics, symplectic integration, mechanical energy |
| `noether` | Symmetry transformations, invariance testing, conserved charge monitors |

## API Tour

### `AgentState<S, const N: usize>`

Encodes an agent's position `q` and velocity `q_dot` in N-dimensional space.

```rust
let state = AgentState::new([1.0, 0.0], [0.0, 1.0]); // 2D phase space
```

### `Lagrangian<S, N>` trait

Define kinetic and potential energy for any dynamical system.

```rust
pub trait Lagrangian<S: Scalar, const N: usize> {
    fn kinetic(&self, state: &AgentState<S, N>) -> S;
    fn potential(&self, state: &AgentState<S, N>) -> S;
}
```

### `MechanicalLagrangian<S, V, N>`

Built-in implementation: `T = ½ m q̇²` with user-supplied potential.

```rust
let lagrangian = MechanicalLagrangian {
    mass: 1.5,
    potential_fn: |q: &[f64; 2]| 0.5 * (q[0] * q[0] + q[1] * q[1]),
};
```

### `SymplecticIntegrator<S, N>`

Stormer–Verlet (leapfrog) integrator — preserves energy much better than explicit Euler.

```rust
let integrator = SymplecticIntegrator::new(0.001)?;
let trajectory = integrator.integrate(mass, &potential, &initial, 10_000)?;
```

### `Symmetry<S, N>` trait + `verify_noether`

Test invariance and compute conserved charges. Built-in symmetries: `TranslationSymmetry`, `RotationSymmetry`, `TimeTranslationSymmetry`.

```rust
let sym = TranslationSymmetry::<3> { axis: 1 };
let monitor = verify_noether(&lagrangian, &sym, &trajectory, mass, 1e-6, 1e-8)?;
assert!(monitor.is_conserved());
```

### `ChargeMonitor<S>`

Track a conserved quantity across a trajectory and detect drift.

```rust
let mut monitor = ChargeMonitor::new(1e-8);
for state in &trajectory {
    monitor.push(compute_charge(state));
}
assert!(monitor.is_conserved());
```

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Symplectic step | O(N) | N = configuration space dimension |
| Full trajectory | O(steps × N) | Energy drift bounded, no exponential accumulation |
| Generalised force | O(N) | Central differences with 2N evaluations |
| Noether verification | O(steps × N) | One charge evaluation per timestep |

The symplectic integrator is explicit and requires no matrix solves. For large-N systems, the cost is linear in the number of degrees of freedom.

## Ecosystem

- **[spectral-fleet](https://github.com/SuperInstance/spectral-fleet-rs)** — Cluster agents by spectral embedding before assigning potentials
- **[categorical-agents](https://github.com/SuperInstance/categorical-agents-rs)** — Compose dynamical systems via adjunctions and monads
- **[ga-core](https://github.com/SuperInstance/ga-core-rs)** — Geometric algebra rotors for rotation symmetries in 3D
- **[wasserstein-agents](https://github.com/SuperInstance/wasserstein-agents-rs)** — Optimal transport for comparing agent distribution trajectories

## Ideas for Improvement

1. **Adaptive timestepping** — Add embedded error estimators (e.g., velocity Verlet with half-step comparison) to vary `dt` based on local dynamics.
2. **Constrained dynamics** — Implement Lagrange multipliers and holonomic constraints for agents moving on manifolds.
3. **Poisson integrators** — Extend to non-canonical symplectic structures for non-Hamiltonian dissipative systems.
4. **Parallel ensemble integration** — Use `rayon` to integrate agent fleets in parallel while sharing the same potential landscape.
5. **Event detection** — Detect zero-crossings and collisions during integration for hybrid agent dynamics.

## License

MIT OR Apache-2.0
