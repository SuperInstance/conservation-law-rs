//! Budget-aware LLM call dispatcher.
//!
//! A fleet of LLM agents shares a token budget. Each request has a cost
//! (tokens). The dispatcher uses conservation-law primitives to:
//! 1. Track the "energy" (budget) of each agent.
//! 2. Verify that total tokens spent plus remaining equals the fleet budget.
//! 3. Use Noether charges to measure budget flow between agents.
//!
//! Run with:
//! ```bash
//! cargo run --example llm_dispatcher
//! ```

use conservation_law::lagrangian::{
    AgentState, MechanicalLagrangian, total_energy,
};
use conservation_law::noether::{ChargeMonitor, Symmetry, TranslationSymmetry, noether_charge};

/// An LLM agent with a token budget.
struct LlmAgent {
    name: &'static str,
    // q[0] = tokens remaining, q_dot[0] = request rate
    state: AgentState<f64, 1>,
    mass: f64,
    // Priority weight: higher = more important
    priority: f64,
}

/// A request to dispatch.
struct Request {
    agent_name: &'static str,
    tokens_needed: f64,
}

fn main() {
    const FLEET_BUDGET: f64 = 1000.0;

    println!("=== Budget-Aware LLM Dispatcher ===");
    println!("Fleet budget: {} tokens\n", FLEET_BUDGET);

    // Five agents with different priorities
    let mut agents = vec![
        LlmAgent { name: "Planner",    state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 1.0 },
        LlmAgent { name: "Coder",      state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 1.2 },
        LlmAgent { name: "Reviewer",   state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 0.8 },
        LlmAgent { name: "Tester",     state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 0.9 },
        LlmAgent { name: "Deployer",   state: AgentState::new([200.0], [0.0]), mass: 1.0, priority: 0.7 },
    ];

    // Queue of incoming requests
    let requests = vec![
        Request { agent_name: "Planner",    tokens_needed: 150.0 },
        Request { agent_name: "Coder",      tokens_needed: 300.0 },
        Request { agent_name: "Reviewer",   tokens_needed: 100.0 },
        Request { agent_name: "Tester",     tokens_needed: 200.0 },
        Request { agent_name: "Deployer",   tokens_needed:  80.0 },
        Request { agent_name: "Coder",      tokens_needed: 250.0 }, // overspend attempt
    ];

    // Budget potential: agents with low remaining tokens have high "potential"
    let budget_potential = |q: &[f64; 1]| {
        let remaining = q[0].max(1.0);
        1000.0 / remaining
    };

    let lagrangian = MechanicalLagrangian {
        mass: 1.0,
        potential_fn: budget_potential,
    };

    let mut budget_monitor = ChargeMonitor::new(1e-6);
    let mut total_dispatched = 0.0;

    println!("Initial state:");
    for agent in &agents {
        println!("  {:10} | {:6.0} tokens | priority = {:.1}",
            agent.name, agent.state.q[0], agent.priority);
    }
    println!();

    println!("--- Dispatching requests ---");
    println!("Request          | Needed | Status               | After");
    println!("-----------------|--------|----------------------|-------");

    for req in &requests {
        let agent_idx = agents.iter().position(|a| a.name == req.agent_name).unwrap();
        let agent = &mut agents[agent_idx];

        let status;

        // Check if agent has enough budget
        if req.tokens_needed <= agent.state.q[0] {
            // Direct dispatch
            agent.state.q[0] -= req.tokens_needed;
            total_dispatched += req.tokens_needed;
            status = format!("dispatched ({:.0})", req.tokens_needed);
        } else if req.tokens_needed <= agent.state.q[0] + 50.0 {
            // Small overspend: allow with warning, pull from others
            let shortfall = req.tokens_needed - agent.state.q[0];
            agent.state.q[0] = 0.0;
            total_dispatched += req.tokens_needed;

            // Redistribute from healthier agents proportionally
            let others_budget: f64 = agents.iter()
                .enumerate()
                .filter(|(i, _)| *i != agent_idx)
                .map(|(_, a)| a.state.q[0])
                .sum();

            if others_budget > 0.0 {
                for (i, other) in agents.iter_mut().enumerate() {
                    if i != agent_idx {
                        let share = other.state.q[0] / others_budget;
                        other.state.q[0] -= shortfall * share;
                    }
                }
            }
            status = format!("overspend {:.0} (rebalanced)", shortfall);
        } else {
            // Large overspend: reject
            status = format!("REJECTED (need {:.0}, have {:.0})", req.tokens_needed, agent.state.q[0]);
        }

        // Enforce non-negative budgets
        for a in agents.iter_mut() {
            a.state.q[0] = a.state.q[0].max(0.0);
        }

        let after = agents[agent_idx].state.q[0];
        println!("{:16} | {:6.0} | {:20} | {:5.0}",
            req.agent_name, req.tokens_needed, status, after);

        // Track budget invariant: dispatched + remaining = constant
        let total_remaining: f64 = agents.iter().map(|a| a.state.q[0]).sum();
        budget_monitor.push(total_remaining + total_dispatched);
    }

    // -----------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------
    println!();
    println!("--- Final state ---");
    for agent in &agents {
        println!("  {:10} | {:6.0} tokens remaining", agent.name, agent.state.q[0]);
    }

    let total_remaining: f64 = agents.iter().map(|a| a.state.q[0]).sum();
    println!("\nTotal dispatched:  {:.0} tokens", total_dispatched);
    println!("Total remaining:   {:.0} tokens", total_remaining);
    println!("Sum:               {:.0} tokens (budget = {})", total_dispatched + total_remaining, FLEET_BUDGET);

    // Budget conservation check
    println!("\nBudget conservation: {} (max drift = {:e})",
        budget_monitor.is_conserved(), budget_monitor.max_drift());

    // -----------------------------------------------------------------
    // Noether charge: "token momentum" — the rate of budget transfer
    // -----------------------------------------------------------------
    println!("\n--- Token transfer momentum (Noether charge) ---");
    let trans = TranslationSymmetry::<1> { axis: 0 };
    for agent in &agents {
        let gen = trans.generator_q(&agent.state);
        let momentum = noether_charge(agent.mass, &agent.state, &gen);
        println!("  {:10} | momentum = {:8.2}", agent.name, momentum);
    }
    println!("\nPositive momentum = agent has remaining capacity.");
    println!("Negative/zero momentum = agent is depleted.");

    // -----------------------------------------------------------------
    // Energy interpretation
    // -----------------------------------------------------------------
    println!("\n--- Energy (budget tension) per agent ---");
    for agent in &agents {
        let e = total_energy(&lagrangian, &agent.state);
        println!("  {:10} | energy = {:8.2} | tension = {}",
            agent.name, e,
            if e > 500.0 { "HIGH (low budget)" } else { "normal" });
    }
}
