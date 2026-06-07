//! Token-budget conservation across 5 agents.
//!
//! Maps physical conservation laws to resource budgeting:
//! - Total energy  E  = budget ceiling C
//! - Kinetic energy T = productive spend γ (tokens doing useful work)
//! - Potential energy V = overhead/waste η (idle, wasted tokens)
//! - E = T + V  →  C = γ + η
//!
//! When an agent overspends, the budget is redistributed so that
//! the fleet-wide total remains conserved.
//!
//! Run with:
//! ```bash
//! cargo run --example conservation_budget
//! ```

use conservation_law::lagrangian::{
    AgentState, Lagrangian, MechanicalLagrangian, SymplecticIntegrator,
};
use conservation_law::noether::{ChargeMonitor, Symmetry, TranslationSymmetry, noether_charge};

/// A fleet agent with an energy budget.
struct BudgetAgent {
    name: &'static str,
    // State: q[0] = tokens allocated, q_dot[0] = spend rate
    state: AgentState<f64, 1>,
    mass: f64,
}

impl BudgetAgent {
    fn tokens_allocated(&self) -> f64 {
        self.state.q[0]
    }
    #[allow(dead_code)]
    fn spend_rate(&self) -> f64 {
        self.state.q_dot[0]
    }
    fn productive_spend(&self, lagrangian: &impl Lagrangian<f64, 1>) -> f64 {
        lagrangian.kinetic(&self.state)
    }
    fn overhead(&self, lagrangian: &impl Lagrangian<f64, 1>) -> f64 {
        lagrangian.potential(&self.state)
    }
}

fn main() {
    const TOTAL_BUDGET: f64 = 1000.0;
    const NUM_AGENTS: usize = 5;

    println!("=== Conservation Budget: {} tokens across {} agents ===\n", TOTAL_BUDGET, NUM_AGENTS);

    // Allocate evenly: each agent gets 200 tokens initially
    let base_alloc = TOTAL_BUDGET / NUM_AGENTS as f64;

    let mut agents: Vec<BudgetAgent> = vec![
        BudgetAgent { name: "Planner",    state: AgentState::new([base_alloc], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Coder",      state: AgentState::new([base_alloc], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Reviewer",   state: AgentState::new([base_alloc], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Tester",     state: AgentState::new([base_alloc], [0.0]), mass: 1.0 },
        BudgetAgent { name: "Deployer",   state: AgentState::new([base_alloc], [0.0]), mass: 1.0 },
    ];

    // Each agent has a "potential" that grows with allocation (diminishing returns)
    // V(q) = ½ k q²  →  larger allocation = higher potential = more overhead
    let k = 0.01;
    let potential = |q: &[f64; 1]| 0.5 * k * q[0] * q[0];

    let lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: potential };

    // Helper to compute fleet-wide totals
    let fleet_totals = |agents: &[BudgetAgent]| -> (f64, f64, f64) {
        let gamma: f64 = agents.iter().map(|a| a.productive_spend(&lagrangian)).sum();
        let eta: f64 = agents.iter().map(|a| a.overhead(&lagrangian)).sum();
        let total: f64 = agents.iter().map(|a| a.tokens_allocated()).sum();
        (gamma, eta, total)
    };

    let (gamma0, eta0, total0) = fleet_totals(&agents);
    println!("Initial state:");
    println!("  γ (productive) = {:.2}", gamma0);
    println!("  η (overhead)   = {:.2}", eta0);
    println!("  C (total)      = {:.2}", total0);
    println!("  γ + η          = {:.2}  (should equal C in physical analogy)", gamma0 + eta0);
    println!();

    // -----------------------------------------------------------------
    // Simulation: Coder overspends by 50 tokens
    // -----------------------------------------------------------------
    println!("--- Event: Coder requests 50 extra tokens ---");
    let overspend = 50.0;
    agents[1].state.q[0] += overspend;

    // To conserve the fleet budget, redistribute proportionally from others
    let excess = overspend;
    let others_total: f64 = agents.iter().enumerate()
        .filter(|(i, _)| *i != 1)
        .map(|(_, a)| a.tokens_allocated())
        .sum();

    for (i, agent) in agents.iter_mut().enumerate() {
        if i != 1 {
            let share = agent.tokens_allocated() / others_total;
            agent.state.q[0] -= excess * share;
        }
    }

    let (gamma1, eta1, total1) = fleet_totals(&agents);
    println!("After redistribution:");
    println!("  γ = {:.2}, η = {:.2}, C = {:.2}", gamma1, eta1, total1);
    println!("  Budget conserved: {} (was {}, now {})", total0 == total1, total0, total1);
    println!();

    // -----------------------------------------------------------------
    // Track budget conservation over time using symplectic integration
    // -----------------------------------------------------------------
    println!("--- Time evolution: budgets fluctuate, total is conserved ---");
    let dt = 0.1;
    let integrator = SymplecticIntegrator::new(dt).unwrap();

    // Give each agent a small oscillating spend rate
    let rates = [0.5, 1.2, 0.8, 0.3, 0.6];
    for (i, rate) in rates.iter().enumerate() {
        agents[i].state.q_dot[0] = *rate;
    }

    // A "force" that pulls each agent back toward its fair share
    let fair_share = TOTAL_BUDGET / NUM_AGENTS as f64;
    let restoring_potential = |q: &[f64; 1]| {
        let displacement = q[0] - fair_share;
        0.5 * 0.005 * displacement * displacement
    };

    println!("Step | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10}",
             agents[0].name, agents[1].name, agents[2].name, agents[3].name, agents[4].name, "Total");
    println!("-----|----------|----------|----------|----------|----------|------------");

    for step in 0..=20 {
        let total: f64 = agents.iter().map(|a| a.tokens_allocated()).sum();

        if step % 4 == 0 {
            println!(" {:3} | {:8.2} | {:8.2} | {:8.2} | {:8.2} | {:8.2} | {:10.2}",
                step,
                agents[0].tokens_allocated(),
                agents[1].tokens_allocated(),
                agents[2].tokens_allocated(),
                agents[3].tokens_allocated(),
                agents[4].tokens_allocated(),
                total
            );
        }

        // Evolve each agent one symplectic step
        // Note: symplectic integration conserves energy (T+V), not sum(q).
        // After each step we re-normalize to enforce the hard budget ceiling.
        for agent in agents.iter_mut() {
            let new_state = integrator
                .step(agent.mass, &restoring_potential, &agent.state)
                .unwrap();
            agent.state = new_state;
        }

        // Hard budget enforcement: renormalize allocations to sum = TOTAL_BUDGET
        let raw_total: f64 = agents.iter().map(|a| a.tokens_allocated()).sum();
        if raw_total > 0.0 {
            let scale = TOTAL_BUDGET / raw_total;
            for agent in agents.iter_mut() {
                agent.state.q[0] *= scale;
            }
        }
    }

    let final_total: f64 = agents.iter().map(|a| a.tokens_allocated()).sum();
    println!();
    println!("Budget conservation check:");
    println!("  Final total allocation = {:.2} (budget = {})", final_total, TOTAL_BUDGET);
    println!("  Drift = {:e} (renormalized to zero)\n", (final_total - TOTAL_BUDGET).abs());

    // -----------------------------------------------------------------
    // Linear momentum analogy: tokens transferred between agents
    // -----------------------------------------------------------------
    println!("--- Noether charge: token transfer momentum ---");
    let transfer_sym = TranslationSymmetry::<1> { axis: 0 };
    let mut momentum_monitor = ChargeMonitor::new(1e-6);
    for agent in &agents {
        let gen = transfer_sym.generator_q(&agent.state);
        let p = noether_charge(agent.mass, &agent.state, &gen);
        momentum_monitor.push(p);
    }
    println!("  Transfer momentum per agent: {:?}", momentum_monitor.values);
    println!("  This is the analog of 'budget flow' between agents.");
}
