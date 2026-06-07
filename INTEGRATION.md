# Integration Guide: conservation-law

## What This Crate Provides

- **`AgentState<S, N>`** — Agent state in N-dimensional configuration space (generalized coordinates + velocities)
- **`Lagrangian<S, N>`** trait — Define kinetic/potential energy for any agent system; computes L = T − V
- **`Symmetry<S, N>`** trait + `TranslationSymmetry`, `RotationSymmetry` — Define continuous symmetries and compute Noether charges
- **`PhaseSpacePoint<S, N>`** — Phase space point (q, p) with distance, volume element
- **`ConservationDetector<S, N>`** — Automatically detect conserved quantities from trajectories
- **`ConservationReport`** — Fleet-wide energy audit with anomaly detection (Z-score)
- **`CircuitState`** (Closed/Open/HalfOpen) — Circuit-breaker pattern for energy transfer protection
- **`central_diff`**, **`time_derivative`** — Numerical differentiation utilities

This crate provides the physics layer for the SuperInstance ecosystem: Lagrangian mechanics, Hamiltonian mechanics via Legendre transform, Noether's theorem for symmetry-derived conservation laws, and fleet-scale energy conservation with anomaly detection.

## How to Add This Crate

```bash
cargo add conservation-law
```

```rust
use conservation_lag::lagrangian::{AgentState, MechanicalLagrangian};
use conservation_lag::conserved::ConservationDetector;

// Define a 2D harmonic oscillator agent
let lagrangian = MechanicalLagrangian::harmonic_oscillator(1.0); // k=1.0
let state = AgentState::new([1.0, 0.0], [0.0, 1.0]);
println!("L = {}", lagrangian.lagrangian(&state));
```

## Integration Points

### fleet-warden

- **Why**: fleet-warden manages disk/resource budgets; conservation-law provides the mathematical framework for energy budgets. Fleet-warden's cleanup operations are conservation-audited: every resource deallocation must preserve the system's energy invariant (γ + H = C).
- **How**: After fleet-warden runs a cleanup sweep, feed the before/after resource metrics into `ConservationDetector` to verify no conservation law was violated.

```rust
use conservation_lag::fleet_integration::ConservationReport;
use conservation_lag::fleet_integration::CircuitState;

// fleet-warden reports per-agent energy before cleanup
let energies_before: Vec<f64> = vec![100.0, 95.0, 110.0];
let energies_after: Vec<f64> = vec![100.0, 95.0, 110.0];

// Audit: verify total energy is conserved post-cleanup
let report = ConservationReport::from_energies(&energies_before, &energies_after, 2.0);
if report.anomalous_agents.is_empty() {
    println!("Cleanup passed conservation audit");
}
```

- **Example**: Run `fleet-warden check`, parse the JSON scan report from stderr, construct a `ConservationReport`, and gate the actual cleanup on a clean audit.

### spectral-fleet

- **Why**: spectral-fleet computes eigenvalues of the fleet Laplacian; conservation-law uses these eigenvalues to verify the spectral gap invariant (γ + H = C). The eigenvalue spectrum IS the conservation audit.
- **How**: Feed spectral-fleet's eigenvalues into conservation-law's Noether charge computation to verify rotational/translational symmetry of the fleet.

```rust
use conservation_lag::noether::{NoetherCharge, TranslationSymmetry};
use conservation_lag::Scalar;

// Eigenvalues from spectral-fleet define the fleet's "energy landscape"
let eigenvalues: Vec<f64> = vec![0.0, 0.15, 0.42, 1.1, 2.3];
let spectral_gap = eigenvalues[1]; // Fiedler value = γ

// Conservation law: γ + H = C
let entropy_production = 0.3;
let capacity = spectral_gap + entropy_production;
println!("System capacity C = {}", capacity);
```

### t-minus

- **Why**: t-minus provides temporal scheduling; conservation-law ensures that scheduled operations don't violate energy conservation. Deadlines must respect the conservation budget.
- **How**: Before a t-minus deadline fires, verify the planned action conserves energy using `ConservationDetector`.

```rust
use conservation_lag::fleet_integration::CircuitState;

// t-minus fires a deadline; check circuit breaker first
let circuit_state = CircuitState::Closed; // from recent audit
if circuit_state == CircuitState::Closed {
    // Safe to proceed with scheduled action
} else {
    // System unstable — defer the action
}
```

## For AI Agents

- **Context needed**: Agent dimension N, scalar type (usually f64), kinetic/potential energy functions
- **Key imports**: `conservation_lag::lagrangian::*`, `conservation_lag::conserved::ConservationDetector`, `conservation_lag::fleet_integration::*`
- **Integration pattern**: Define `Lagrangian` → simulate trajectory → run `ConservationDetector::detect()` → verify conserved quantities
- **Error handling**: All conservation checks return `ConservedQuantity` with `max_drift` and `is_conserved` fields. Use `verify(tolerance)` to gate actions on conservation compliance.

## For Humans

- **Prerequisites**: Basic Lagrangian/Hamiltonian mechanics, understanding of Noether's theorem
- **Learning path**: Start with `lagrangian.rs` (simplest), then `noether.rs` (symmetries), then `conserved.rs` (automatic detection), then `fleet_integration.rs` (production use)
- **Common pitfalls**:
  - The scalar type must implement `num_traits::Float` — use f64 unless you have a reason not to
  - `central_diff` accuracy depends on step size h — too small amplifies floating point noise, too large loses resolution
  - Conservation reports assume energies are per-agent; mixing fleet totals with per-agent values causes false anomalies
