//! Fleet integration: energy conservation across a fleet of agents.
//!
//! Bridges conservation-law concepts with fleet management patterns:
//! - **Anomaly detection** via Z-score thresholds to trigger conservation audits
//!   when agent energy profiles deviate from fleet norms.
//! - **Circuit breaker** pattern (Closed / Open / HalfOpen) to halt energy
//!   transfers when the system is detected as unstable.
//!
//! This module treats a fleet of agents as a thermodynamic ensemble and applies
//! conservation-law bookkeeping to energy transfers between agents.

use crate::Scalar;

// ---------------------------------------------------------------------------
// Circuit-breaker states
// ---------------------------------------------------------------------------

/// Circuit-breaker state for energy transfer protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — transfers allowed.
    Closed,
    /// Transfer halted — system unstable.
    Open,
    /// Probing — allow one transfer to test recovery.
    HalfOpen,
}

// ---------------------------------------------------------------------------
// Conservation report
// ---------------------------------------------------------------------------

/// Result of a fleet-wide conservation audit.
#[derive(Debug, Clone)]
pub struct ConservationReport {
    /// Mean energy across the fleet.
    pub mean_energy: f64,
    /// Standard deviation of fleet energy.
    pub std_dev: f64,
    /// Indices of agents flagged as anomalous (Z-score > threshold).
    pub anomalous_agents: Vec<usize>,
    /// Z-scores for each agent.
    pub z_scores: Vec<f64>,
    /// Whether the overall system is considered stable.
    pub system_stable: bool,
}

// ---------------------------------------------------------------------------
// FleetConservation — main struct
// ---------------------------------------------------------------------------

/// Tracks energy budget across a fleet of agents and guards transfers using
/// anomaly detection (Z-score) and a circuit-breaker pattern.
#[derive(Debug, Clone)]
pub struct FleetConservation {
    /// Current energy held by each agent.
    pub energies: Vec<f64>,
    /// Z-score threshold above which an agent is flagged anomalous.
    pub z_threshold: f64,
    /// Maximum allowed fraction of a single transfer relative to total fleet energy.
    pub max_transfer_fraction: f64,
    /// Circuit-breaker state.
    pub circuit: CircuitState,
    /// Number of consecutive failures that trips the circuit to Open.
    pub failure_threshold: usize,
    /// Running count of consecutive transfer failures.
    consecutive_failures: usize,
    /// Number of successful probe transfers while in HalfOpen.
    probe_successes: usize,
    /// Probes needed to transition HalfOpen → Closed.
    probes_needed: usize,
}

impl FleetConservation {
    /// Create a new fleet with the given initial energies.
    pub fn new(energies: Vec<f64>, z_threshold: f64) -> Self {
        Self {
            energies,
            z_threshold,
            max_transfer_fraction: 0.25,
            circuit: CircuitState::Closed,
            failure_threshold: 3,
            consecutive_failures: 0,
            probe_successes: 0,
            probes_needed: 2,
        }
    }

    /// Total energy in the fleet.
    pub fn total_energy(&self) -> f64 {
        self.energies.iter().copied().fold(0.0, |a, b| a + b)
    }

    /// Mean energy across the fleet.
    pub fn mean(&self) -> f64 {
        if self.energies.is_empty() {
            return 0.0;
        }
        self.total_energy() / self.energies.len() as f64
    }

    /// Standard deviation of fleet energies.
    pub fn std_dev(&self) -> f64 {
        if self.energies.len() < 2 {
            return 0.0;
        }
        let mu = self.mean();
        let variance = self.energies.iter()
            .map(|&e| (e - mu) * (e - mu))
            .sum::<f64>()
            / self.energies.len() as f64;
        variance.sqrt()
    }

    /// Compute Z-scores for each agent's energy.
    pub fn z_scores(&self) -> Vec<f64> {
        let sigma = self.std_dev();
        if sigma < 1e-12 {
            return vec![0.0; self.energies.len()];
        }
        let mu = self.mean();
        self.energies.iter().map(|&e| (e - mu) / sigma).collect()
    }

    // ----- audit ----------------------------------------------------------

    /// Audit the fleet for anomalous energy profiles.
    ///
    /// Returns a [`ConservationReport`] with Z-scores, anomalous agent indices,
    /// and system-stability assessment.  If the number of anomalous agents
    /// exceeds half the fleet, the system is marked unstable and the circuit
    /// breaker trips to [`CircuitState::Open`].
    pub fn audit_fleet(&self) -> ConservationReport {
        let z_scores = self.z_scores();
        let anomalous: Vec<usize> = z_scores.iter()
            .enumerate()
            .filter(|(_, &z)| z.abs() > self.z_threshold)
            .map(|(i, _)| i)
            .collect();
        let stable = anomalous.len() * 2 <= self.energies.len();
        ConservationReport {
            mean_energy: self.mean(),
            std_dev: self.std_dev(),
            anomalous_agents: anomalous,
            z_scores,
            system_stable: stable,
        }
    }

    // ----- circuit breaker ------------------------------------------------

    /// Current circuit state.
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit
    }

    /// Trip the circuit breaker to Open.
    pub fn trip(&mut self) {
        self.circuit = CircuitState::Open;
        self.consecutive_failures = self.failure_threshold;
    }

    /// Reset the circuit breaker to Closed.
    pub fn reset(&mut self) {
        self.circuit = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.probe_successes = 0;
    }

    /// Attempt to transition from Open → HalfOpen (probing).
    pub fn try_half_open(&mut self) -> bool {
        if self.circuit == CircuitState::Open {
            self.circuit = CircuitState::HalfOpen;
            self.probe_successes = 0;
            return true;
        }
        false
    }

    // ----- transfer -------------------------------------------------------

    /// Transfer energy from agent `from_idx` to agent `to_idx`.
    ///
    /// Enforces:
    /// 1. Circuit breaker must not be Open (HalfOpen allows one probe).
    /// 2. Source agent must have sufficient energy.
    /// 3. Transfer amount must not exceed `max_transfer_fraction` of total energy.
    /// 4. No agent may go negative.
    ///
    /// On success, updates energies in place and returns the amount transferred.
    /// On failure, increments the consecutive failure counter and may trip the
    /// circuit breaker.
    pub fn transfer_with_guard(
        &mut self,
        from_idx: usize,
        to_idx: usize,
        amount: f64,
    ) -> Result<f64, String> {
        // Index bounds check
        if from_idx >= self.energies.len() || to_idx >= self.energies.len() {
            return Err("index out of bounds".into());
        }
        if from_idx == to_idx {
            return Err("source and destination must differ".into());
        }

        // Circuit-breaker guard
        match self.circuit {
            CircuitState::Open => {
                self.consecutive_failures += 1;
                return Err("circuit breaker is Open — transfers halted".into());
            }
            CircuitState::HalfOpen => {
                // Allow the transfer as a probe
            }
            CircuitState::Closed => {}
        }

        let total = self.total_energy();

        // Amount guard
        if amount <= 0.0 {
            return Err("amount must be positive".into());
        }
        if amount > self.max_transfer_fraction * total {
            self.record_failure();
            return Err(format!(
                "amount {} exceeds max fraction of total {}",
                amount,
                self.max_transfer_fraction * total
            ));
        }

        // Sufficient balance
        if self.energies[from_idx] < amount {
            self.record_failure();
            return Err(format!(
                "agent {} has insufficient energy ({})",
                from_idx, self.energies[from_idx]
            ));
        }

        // Execute transfer
        self.energies[from_idx] -= amount;
        self.energies[to_idx] += amount;

        // Record success (may close circuit)
        self.record_success();

        Ok(amount)
    }

    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.failure_threshold {
            self.circuit = CircuitState::Open;
        }
    }

    fn record_success(&mut self) {
        self.consecutive_failures = 0;
        match self.circuit {
            CircuitState::HalfOpen => {
                self.probe_successes += 1;
                if self.probe_successes >= self.probes_needed {
                    self.circuit = CircuitState::Closed;
                    self.probe_successes = 0;
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar helper: z-score computation for generic Scalar type
// ---------------------------------------------------------------------------

/// Compute Z-score for a value given mean and standard deviation.
pub fn zscore<S: Scalar>(value: S, mean: S, std_dev: S) -> S {
    if std_dev < S::epsilon() {
        return S::zero();
    }
    (value - mean) / std_dev
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_balanced_fleet() -> FleetConservation {
        FleetConservation::new(vec![100.0, 100.0, 100.0, 100.0], 2.0)
    }

    #[test]
    fn test_total_energy_conserved_on_transfer() {
        let mut fleet = make_balanced_fleet();
        let before = fleet.total_energy();
        fleet.transfer_with_guard(0, 1, 10.0).unwrap();
        let after = fleet.total_energy();
        assert!((before - after).abs() < 1e-10, "energy must be conserved");
    }

    #[test]
    fn test_transfer_updates_balances() {
        let mut fleet = make_balanced_fleet();
        fleet.transfer_with_guard(0, 1, 25.0).unwrap();
        assert_eq!(fleet.energies[0], 75.0);
        assert_eq!(fleet.energies[1], 125.0);
    }

    #[test]
    fn test_transfer_rejects_insufficient_energy() {
        let mut fleet = make_balanced_fleet();
        let result = fleet.transfer_with_guard(0, 1, 200.0);
        assert!(result.is_err());
        // Should not have changed balances
        assert_eq!(fleet.energies[0], 100.0);
    }

    #[test]
    fn test_transfer_rejects_out_of_bounds() {
        let mut fleet = make_balanced_fleet();
        let result = fleet.transfer_with_guard(0, 99, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_transfer_rejects_same_source_and_dest() {
        let mut fleet = make_balanced_fleet();
        let result = fleet.transfer_with_guard(2, 2, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_audit_detects_anomaly() {
        // One agent has wildly different energy
        let fleet = FleetConservation::new(vec![100.0, 100.0, 100.0, 10000.0], 1.5);
        let report = fleet.audit_fleet();
        assert!(!report.anomalous_agents.is_empty(), "should detect the outlier agent");
        assert_eq!(report.anomalous_agents[0], 3);
    }

    #[test]
    fn test_audit_balanced_fleet_no_anomalies() {
        let fleet = make_balanced_fleet();
        let report = fleet.audit_fleet();
        assert!(report.anomalous_agents.is_empty());
        assert!(report.system_stable);
    }

    #[test]
    fn test_circuit_breaker_trips_after_failures() {
        let mut fleet = FleetConservation::new(vec![100.0, 100.0], 2.0);
        fleet.failure_threshold = 2;
        // Trigger failures by trying over-large transfers
        let total = fleet.total_energy();
        let _ = fleet.transfer_with_guard(0, 1, total); // exceeds max fraction
        assert_ne!(fleet.circuit_state(), CircuitState::Open);
        let _ = fleet.transfer_with_guard(0, 1, total); // second failure
        assert_eq!(fleet.circuit_state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_halts_transfers_when_open() {
        let mut fleet = make_balanced_fleet();
        fleet.trip();
        let result = fleet.transfer_with_guard(0, 1, 1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Open"));
    }

    #[test]
    fn test_half_open_recovery() {
        let mut fleet = make_balanced_fleet();
        fleet.circuit = CircuitState::Open;
        assert!(fleet.try_half_open());
        assert_eq!(fleet.circuit_state(), CircuitState::HalfOpen);
        // Successful probe
        fleet.transfer_with_guard(0, 1, 5.0).unwrap();
        // Need probes_needed successes — default is 2
        fleet.transfer_with_guard(1, 0, 5.0).unwrap();
        assert_eq!(fleet.circuit_state(), CircuitState::Closed);
    }

    #[test]
    fn test_zscore_function() {
        let z = zscore(3.0_f64, 0.0, 1.0);
        assert!((z - 3.0).abs() < 1e-10);
        // Zero std_dev → return 0
        let z2 = zscore(5.0_f64, 5.0, 0.0);
        assert_eq!(z2, 0.0);
    }

    #[test]
    fn test_mean_and_std_dev() {
        let fleet = FleetConservation::new(vec![10.0, 20.0, 30.0], 2.0);
        assert!((fleet.mean() - 20.0).abs() < 1e-10);
        // Population std dev of [10,20,30] = sqrt((100+0+100)/3) = sqrt(200/3)
        let expected = (200.0_f64 / 3.0).sqrt();
        assert!((fleet.std_dev() - expected).abs() < 1e-10);
    }
}
