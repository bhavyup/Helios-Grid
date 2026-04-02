---
description: "Use when implementing, refactoring, planning, reviewing, or proposing architecture for any Helios-Grid work. Enforces decentralized MARL goals, GNN coordination, P2P market design, simulation-first delivery, strict architecture-change gating, and phase-aware execution."
name: "Helios-Grid Global Context"
applyTo: "**"
---

# Helios-Grid Global Project Constitution

This file is an always-on constitution for all AI contributions in this repository.

The intent is simple:
- Preserve product truth.
- Prevent architectural drift.
- Keep implementation choices aligned with research goals.
- Keep this project decentralized by default.

If there is any conflict between quick implementation convenience and Helios-Grid principles, Helios-Grid principles win.

## 1) Project Identity

- Project name: Helios-Grid.
- Core identity: decentralized virtual power plant simulation platform.
- Core thesis: neighborhood-level peer-to-peer energy balancing through cooperative multi-agent reinforcement learning with topology-aware graph coordination.
- Baseline framing: centralized optimization exists only as a benchmark comparator, not as the default control model.

## 2) Why This Project Exists

Helios-Grid addresses a structural mismatch:
- Power generation is increasingly distributed (residential solar, batteries, micro-assets).
- Grid software coordination is still mostly centralized.

This mismatch causes:
- Renewable curtailment.
- Peak-load stress and peaker-plant reliance.
- Slower adaptation and weak local autonomy.
- Data centralization and privacy risks.

Primary objective:
- Demonstrate that decentralized RL agents with graph-aware coordination and market-driven P2P exchange can match or approach centralized optimization while improving scalability, resilience, and privacy posture.

## 3) Non-Negotiable Boundaries

These are hard constraints for all AI actions.

1. Preserve decentralized control as the default behavior.
2. Preserve graph-aware coordination as a first-class mechanism, not an optional afterthought.
3. Preserve P2P market logic as a first-class mechanism for local balancing.
4. Preserve benchmark comparators (rule-based, independent PPO, MAPPO/CTDE, centralized MILP).
5. Preserve simulation-first assumptions by default.
6. Do not require hardware integration unless explicitly requested.
7. Do not remove or replace the core stack without explicit user instruction.
8. Do not collapse service boundaries into a single centralized optimizer by default.
9. Do not optimize purely for short-term implementation speed if it undermines long-term research validity.
10. If a simplification is needed, label it clearly as temporary scaffolding and preserve upgrade paths.

## 4) Blockchain Policy (User Decision Enforced)

Blockchain is explicitly opt-in only.

- Default behavior: ignore blockchain design, settlement logic, and blockchain dependencies.
- Do not propose or implement Hardhat, Polygon Edge, Solidity contracts, token logic, or on-chain settlement unless the user explicitly requests blockchain work.
- If blockchain appears in earlier architecture text, treat it as optional context only, not an implementation requirement.

## 5) Strict Architecture-Change Gate (User Decision Enforced)

Major architecture changes require a migration rationale before implementation.

Definition of major architecture change includes:
- Replacing decentralized control with centralized dispatch.
- Replacing FastAPI microservice direction with another backend paradigm.
- Replacing RabbitMQ-style event-driven messaging with incompatible interaction patterns.
- Replacing PyTorch MARL/GNN direction with a non-equivalent paradigm.
- Replacing PostGIS/TimescaleDB/Redis data strategy with incompatible persistence assumptions.
- Collapsing research baselines so comparative evaluation becomes impossible.

Strict behavior required from AI:
- If asked to make a major architecture change and no migration rationale is supplied, refuse to execute the change.
- Ask for a migration rationale first.
- Only proceed after rationale is provided or explicit user override is given.

Minimum migration rationale must include:
1. Problem being solved by the change.
2. Why existing architecture is insufficient.
3. Expected measurable gains.
4. Tradeoffs and risks.
5. Migration strategy (incremental steps).
6. Rollback strategy.
7. Impact on baselines and evaluation comparability.
8. Impact on timeline and complexity.

## 6) Canonical Technical Direction

### 6.1 Frontend Target Stack

- Next.js 14
- React
- Three.js (and/or React Three Fiber)
- Recharts
- TailwindCSS

Primary frontend purpose:
- Real-time understanding of decentralized grid behavior.
- Not just static dashboards.

### 6.2 Backend Target Style

- Event-driven FastAPI microservices.

Planned service domains:
1. Grid Environment Service.
2. Agent Runtime Service.
3. GNN Coordination Service.
4. Market Engine Service.
5. Analytics and Evaluation Service.

### 6.3 Messaging and Data Layer

- RabbitMQ event bus with topic queues such as:
  - sim.tick
  - agent.actions
  - agent.observations
  - market.orders
  - market.settlements
  - grid.state
  - weather.updates
- PostGIS for spatial topology data.
- TimescaleDB for time-series operations data.
- Redis for cache and pub-sub patterns.

### 6.4 API Gateway Direction

- FastAPI API gateway is the orchestration boundary for API clients.
- Typical concerns:
  - authentication and authorization
  - simulation lifecycle management
  - rate limiting and safety controls
  - state query endpoints and real-time stream coordination

## 7) Architecture Blueprint (Reference)

The following architecture representation is canonical context for AI planning.

```text
+----------------------------------------------------------------------+
|                            FRONTEND LAYER                            |
|      Next.js 14 + React + Three.js + Recharts + TailwindCSS         |
|                                                                      |
|  - 3D neighborhood digital twin                                      |
|  - Energy flow dashboard                                             |
|  - Market and agent analytics panel                                  |
|  - Simulation controls                                                |
|                                                                      |
|  Stream channel: WebSocket (Socket.IO style) + REST                 |
+------------------------------+---------------------------------------+
                               |
+------------------------------v---------------------------------------+
|                     API GATEWAY (FastAPI)                            |
|               Auth, simulation lifecycle, rate controls              |
+---+-----------+-----------+-----------+-----------+------------------+
    |           |           |           |           |
    v           v           v           v           v
 Grid Env   Agent Runtime   GNN Coord   Market      Analytics
 Service    Service         Service     Engine      and Eval

 Shared eventing plane:
 - RabbitMQ topics: sim.tick, agent.actions, agent.observations,
   market.orders, market.settlements, grid.state, weather.updates

 Persistence:
 - PostGIS: topology + spatial metadata
 - TimescaleDB: time-series telemetry and market logs
 - Redis: cache and pub-sub

 Optional extension by explicit user request only:
 - Blockchain settlement layer
```

## 8) Current Repository Maturity and Build Strategy

Current maturity (as of now):
- Early stage scaffold.
- Backend exists with foundational FastAPI structure.
- Frontend not yet fully scaffolded.

Default implementation bias:
- Prioritize Phase 1 foundational work unless user asks to jump ahead.

Phase 1 focus:
1. Simulation primitives.
2. Data ingestion and synthetic generation.
3. Stable service interfaces.
4. Observability and testability.

Modularity rule:
- Design modules so they can split into independent services later with minimal rewrites.

## 9) Domain Model Expectations

AI-generated design and code should represent:
- Households/buildings as autonomous agents.
- Local energy state (generation, consumption, storage, import/export).
- Neighborhood topology as a graph.
- Market interactions with bids/asks and matching outcomes.
- Operational constraints (capacity, losses, SoC boundaries).

Default timestep assumption:
- 15-minute decision interval unless user specifies otherwise.

## 10) Core Algorithmic Guardrails

Preserve these concepts as first-class:
- PPO or related policy-gradient approach for per-agent control.
- Cooperative MARL pathway (CTDE / MAPPO variant).
- GNN message-passing coordination embeddings over topology graph.
- Market-based decentralized price discovery.

Acceptable simplification strategy:
- Use simple versions for initial velocity.
- Always preserve compatibility path to full target approach.
- Mark simplifications in code or docs as temporary.

## 11) Baseline and Comparator Requirements

Comparative evaluation must remain possible.

Keep support for:
1. Rule-based self-consumption baseline.
2. Independent PPO baseline.
3. MAPPO/CTDE cooperative baseline without topology-aware coordination.
4. Centralized MILP benchmark.

Do not remove baseline support unless explicitly requested.

## 12) Data and Simulation Assumptions

Simulation-first by default:
- No physical hardware dependency in default workflows.

Expected inputs and synthetic families:
- Weather and irradiance profiles (NASA POWER style).
- Household demand archetypes (Pecan Street style behavior).
- Battery dynamics (SoC constraints, efficiency, degradation approximations).
- Radial topology and transformer/line capacity constraints.
- Tariff and market pricing signals.

Stress-case readiness is part of the design, not an optional afterthought.

## 13) Physics and Market Constraints

AI should treat these as functional constraints:

Grid constraints:
- Transformer capacity limits.
- Line flow and congestion constraints.
- Approximate power-flow feasibility checks.

Battery constraints:
- Min and max SoC boundaries.
- Charge/discharge rate limits.
- Efficiency losses.

Market constraints:
- Bid/ask compatibility.
- Clearing and settlement logic.
- Topology-aware feasibility of transfers.

## 14) Research Gaps to Preserve in Design

AI plans should continuously reflect these gaps:
1. MARL non-stationarity in energy systems.
2. Missing integration of graph-aware communication with MARL in P2P balancing.
3. Under-characterized cooperative vs selfish behavior at realistic scale.

Design recommendation quality should improve these research questions, not dilute them.

## 15) Experimental Methodology Alignment

When generating experiments, roadmaps, or evaluation features, align with:

1. Baseline comparisons across five methods.
2. Ablation studies (with and without GNN, market, storage).
3. Scale tests (from small neighborhoods to large populations).
4. Heterogeneity tests (uniform vs mixed households).
5. Stress-test campaign under adverse events.
6. Fairness analysis (including distributional outcomes).
7. Emergent behavior analysis from learned communication.

## 16) Metrics That Must Stay First-Class

### Grid Efficiency

- Peak-load shaving efficiency.
- Renewable utilization rate.
- Curtailment rate.

### Environmental

- Carbon footprint reduction vs baseline.
- Grid import reduction.

### Economic and Fairness

- Average household cost reduction.
- Gini coefficient and other fairness indicators.
- Social welfare outcomes.

### Grid Stability and Resilience

- Voltage deviation proxies.
- Transformer overload event count.
- Recovery behavior after stress scenarios.

### Learning and Scalability

- Convergence behavior.
- Policy stability metrics.
- Per-timestep compute cost, especially at larger scale.

## 17) Phase Plan Guidance

### Phase 1: Simulation Foundation

Priorities:
- Environment and physics primitives.
- Data ingestion and synthetic generation setup.
- Event bus orchestration.
- Rule-based baseline viability.
- Initial dashboard visibility.

### Phase 2: Multi-Agent Intelligence

Priorities:
- PPO and MAPPO training pathways.
- GNN coordination service.
- Market engine maturity.
- Comparative baseline implementation.

### Phase 3: Evaluation and Analysis

Priorities:
- Full experiment matrix execution.
- Ablation and fairness studies.
- Scalability and resilience benchmarking.
- Report-ready outputs for publication-quality analysis.

## 18) Coding Standards and Delivery Expectations

### 18.1 Backend Python

- Strong typing by default.
- Deterministic behavior where practical.
- Testability over hidden side effects.
- Clear domain-oriented module boundaries.
- Keep external dependencies purposeful.

### 18.2 API and Service Contracts

- Prefer explicit request and response models.
- Keep endpoint semantics stable.
- Preserve forward compatibility for service decomposition.

### 18.3 Event Contracts

- Use stable topic naming.
- Keep payload schemas explicit and versionable.
- Avoid silent breaking changes to published event formats.

### 18.4 Frontend

- Prioritize interpretability of system behavior.
- Show neighborhood state, trade flow, and policy impact clearly.
- Favor responsive performance for real-time views.

### 18.5 Tests

- Add or update tests when behavior changes.
- Prefer deterministic tests with controlled random seeds for simulation and MARL scaffolding.
- Keep baselines testable as they are key comparators.

### 18.6 Documentation

- Record assumptions and simplifications.
- Mark temporary shortcuts explicitly.
- Document upgrade paths to full target behavior.

## 19) AI Planning Rules

When creating plans or implementation proposals:

1. Start with user request and current phase context.
2. State how proposal preserves decentralization, graph coordination, and P2P market assumptions.
3. State how proposal impacts baseline comparability.
4. State what is temporary vs production-target.
5. State validation strategy and metrics touched.

Do not produce generic architecture suggestions that ignore project identity.

## 20) AI Execution Rules

For implementation tasks:
- Prefer incremental, testable changes.
- Keep interfaces explicit.
- Avoid broad rewrites without necessity.
- Preserve current behavior unless change is intentional and documented.

For review tasks:
- Prioritize risks, regressions, and metric impacts.
- Highlight deviations from decentralized and research guardrails.

For debugging tasks:
- Preserve reproducibility.
- Avoid quick fixes that erase comparability or observability.

## 21) Mandatory Refusal and Escalation Cases

AI must refuse and request clarification/rationale when asked to:

1. Replace decentralized coordination with centralized dispatch as default.
2. Remove GNN/topology-aware coordination from target design.
3. Remove P2P market mechanics from target design.
4. Remove all baseline comparators.
5. Introduce blockchain implementation without explicit user request.
6. Perform major architecture shifts without migration rationale.

Refusal response behavior:
- Explain briefly why the request conflicts with repository constitution.
- Ask for explicit override and migration rationale when applicable.
- Offer a safer alternative aligned with Helios-Grid goals.

## 22) Practical Decision Rubric for Ambiguous Requests

When requirements are ambiguous, choose the option that best satisfies all of:

1. Preserves decentralization.
2. Preserves graph-aware coordination path.
3. Preserves P2P market path.
4. Preserves simulation-first speed of iteration.
5. Preserves baseline comparability.
6. Preserves testability and observability.

If no option satisfies all six, ask the user to prioritize tradeoffs.

## 23) Guidance for Temporary Scaffolding

Temporary scaffolding is allowed only if:
- It accelerates current phase execution.
- It does not block future decentralized/GNN/P2P architecture.
- It is clearly labeled as temporary.
- It has an explicit migration path.

Examples of acceptable temporary scaffolding:
- In-memory adapters before full persistence integration.
- Simplified synthetic generators before full dataset pipelines.
- Basic API stubs before asynchronous event integration.

Examples of unacceptable scaffolding:
- Hard-coding centralized control as default architecture.
- Deleting baseline pathways to reduce implementation complexity.
- Introducing blockchain dependencies without explicit request.

## 24) Frontend Experience Expectations

Frontend should communicate system intelligence and tradeoffs, not only render charts.

Expected capabilities over time:
- 3D neighborhood representation with meaningful state encodings.
- Real-time flow and market signals.
- Policy mode comparison controls.
- Stress-test triggers and playback controls.

Visualization should support debugging and scientific interpretation, not only presentation aesthetics.

## 25) Service-Specific Intent Summaries

### Grid Environment Service

Purpose:
- Simulate physics and operational context for all agents.

Must support:
- Solar, load, storage, topology constraints, and stress scenarios.

### Agent Runtime Service

Purpose:
- Manage decentralized decision-making logic for household agents.

Must support:
- Baseline and cooperative policy modes.

### GNN Coordination Service

Purpose:
- Provide topology-aware coordination embeddings.

Must support:
- Message passing over neighborhood graph each timestep.

### Market Engine Service

Purpose:
- Clear decentralized P2P trading.

Must support:
- Continuous double auction semantics and settlement outputs.

### Analytics and Evaluation Service

Purpose:
- Track performance, fairness, resilience, and learning quality.

Must support:
- Comparative evaluation across baselines and ablations.

## 26) Default Assumptions for AI if User Does Not Specify

If user request is underspecified, assume:
1. Decentralized default architecture remains.
2. Blockchain remains disabled/ignored.
3. Work should align with current phase maturity.
4. Baselines must remain evaluable.
5. Simulation-first implementation is acceptable.

## 27) Prohibited Drift Patterns

The following patterns are prohibited unless explicitly requested with rationale:

- Converting design into centralized command-and-control architecture.
- Treating GNN coordination as optional decoration with no model impact.
- Replacing P2P market with fixed static pricing without preserving market pathway.
- Designing features that cannot be compared to baseline methods.
- Choosing convenience abstractions that hide key metrics needed for research evaluation.

## 28) Quality Gates Before Declaring Work Complete

Before finalizing substantial code or design work, AI should check:

1. Does this preserve decentralized behavior?
2. Does this preserve graph-aware coordination path?
3. Does this preserve P2P market path?
4. Does this preserve baseline comparability?
5. Is blockchain avoided unless explicitly requested?
6. Is phase alignment respected?
7. Are tests or validation updates included where behavior changes?
8. Are temporary simplifications documented?

If any answer is no, AI should resolve it or explicitly call out the gap.

## 29) Research and Publication Orientation

Helios-Grid is not only an engineering project; it is also a research-grade platform.

Contributions should improve:
- Reproducibility.
- Comparative validity.
- Interpretability of outcomes.
- Scalability understanding.

AI should avoid choices that make experimental claims weaker or less defensible.

## 30) Final Behavioral Directive for AI in This Repository

Always optimize for alignment with Helios-Grid mission:

- Decentralized intelligence.
- Graph-neural coordination.
- P2P market balancing.
- Simulation-first experimentation.
- Comparative scientific rigor.

If uncertain, pause and ask clarifying questions before making irreversible decisions.

If asked to make a major architecture shift, require migration rationale first.

If blockchain is not explicitly requested, keep blockchain out of scope.

## 31) Detailed Algorithm Catalog

Use this catalog to keep implementation and planning aligned with intended methods.

| Component | Algorithm/Model | Intent |
| --- | --- | --- |
| Agent policy | PPO (or equivalent policy-gradient method) | Learn per-agent energy management policies |
| Multi-agent training | CTDE / MAPPO shared critic | Cooperative training with decentralized execution |
| Graph coordination | GraphSAGE or GAT | Encode topology-aware neighborhood context |
| Communication learning | CommNet/TarMAC-inspired messaging patterns | Learn what to communicate, not only what to act |
| Market mechanism | Continuous double auction | Decentralized price discovery and matching |
| Forecasting | LSTM or TFT (optional stage) | Short-horizon load and generation forecasting |
| Grid feasibility | DC power-flow approximation | Fast tractable operational feasibility checks |
| Centralized benchmark | MILP (PuLP/OR-Tools) | Global-optimal comparator under perfect information |
| Fairness attribution | Shapley approximation | Attribute cooperative gains across agents |
| Stability analysis | Lyapunov-style convergence proxies | Analyze training and policy stability |

Preserve this as directional intent, not rigid lock-in. Equivalent methods are acceptable only if they preserve comparability and research goals.

## 32) Data Simulation Blueprint (Detailed)

All default workflows remain software-simulated.

### 32.1 Solar Generation

- Weather source profile direction: NASA POWER style historical weather features.
- Core production approximation:

  P = A * eta * G * (1 - beta * (T - 25))

  where:
  - A is panel area
  - eta is panel efficiency
  - G is irradiance
  - beta is temperature coefficient
  - T is panel temperature in Celsius

- Household variation factors:
  - panel orientation offsets
  - shading profiles
  - age/degradation modifiers

### 32.2 Household Demand

- Behavioral template direction: Pecan Street style household archetypes.
- Include archetype diversity such as:
  - small apartment
  - family home
  - large home with EV
  - home office profile
- Preserve weekday/weekend and seasonal variation.
- Include stochastic appliance and EV events.

### 32.3 Battery Dynamics

- Capacity range and limits should stay configurable per household.
- Must enforce:
  - min SoC bound
  - max SoC bound
  - max charge/discharge rate
  - round-trip efficiency losses
  - optional simplified degradation model

### 32.4 Topology and Grid Constraints

- Generate radial or near-radial neighborhood topologies by default.
- Include line/transformer limits and congestion feasibility checks.
- Maintain topology persistence path in PostGIS-compatible structures.

### 32.5 Price Signals

- Include tariff and market signal pathways.
- Keep support for both utility tariff context and local P2P market price dynamics.

## 33) Stress Test Scenarios to Keep Available

Baseline stress scenarios should include:
1. Cloudy-day irradiance collapse (large sudden drop).
2. Heat-wave demand surge with efficiency impacts.
3. Grid disconnection/islanding window.
4. Synchronized EV charging surge.
5. Asymmetric generation distribution (solar unevenness across households).

When implementing simulation controls, keep stress test triggers first-class and reproducible.

## 34) Experimental Matrix Requirements

### 34.1 Baselines (Minimum)

1. Centralized MILP.
2. Rule-based self-consumption.
3. Independent PPO.
4. MAPPO/CTDE without topology-aware coordination.
5. Full Helios-Grid (MAPPO + GNN coordination + P2P market).

### 34.2 Ablations (Minimum)

1. No GNN coordination.
2. No P2P market.
3. No storage.
4. Full model.

### 34.3 Scale Tests

- Preserve explicit scale sweep from small to large neighborhoods.
- Keep at least one high-scale scenario where compute overhead is measured.

### 34.4 Heterogeneity Tests

- Compare uniform household sets against mixed archetype communities.

### 34.5 Fairness and Emergence

- Include fairness outcomes (cost distribution quality).
- Include emergent behavior analysis for learned communication and trading motifs.

## 35) Target Metrics and Acceptance Direction

When teams need practical targets, use these directional thresholds unless user overrides them:

| Category | Metric | Directional Target |
| --- | --- | --- |
| Grid efficiency | Peak-load shaving vs no-control | >= 20% |
| Grid efficiency | Renewable utilization | >= 85% |
| Grid efficiency | Curtailment rate | <= 5% |
| Environmental | Carbon footprint reduction vs rule-based | >= 25% |
| Environmental | Grid import reduction | >= 30% |
| Economic | Average household cost reduction | >= 15% |
| Fairness | Gini coefficient of household costs | <= 0.25 |
| Stability | Voltage deviation proxy | <= 3% |
| Stability | Transformer overload count/month | <= 2 |
| Optimality gap | Gap vs centralized MILP | <= 12% |
| MARL quality | Episodes to stable convergence | <= 100k |
| MARL quality | Reward variance near convergence | <= 5% |
| Scalability | Timestep compute at high-scale neighborhood | <= 2s |
| Resilience | Delivery maintained during grid disconnection | >= 70% |
| Resilience | Recovery after stress event | <= 30 simulated minutes |

Treat these values as research targets, not hard blockers for all implementation PRs.

## 36) Phase Deliverables and Timeline Intent

### 36.1 Phase 1 - Simulation Foundation

Expected deliverables:
- Functional multi-agent grid simulator with physics constraints.
- Rule-based baseline agents.
- Event-driven simulation tick propagation.
- Basic real-time visibility into energy state.

### 36.2 Phase 2 - Multi-Agent Intelligence

Expected deliverables:
- Independent PPO baseline.
- Cooperative MAPPO/CTDE pathway.
- GNN coordination embeddings integrated into policy observation path.
- P2P market engine with matching and settlement records.

### 36.3 Phase 3 - Evaluation and Analysis

Expected deliverables:
- Full evaluation matrix runs.
- Ablation, fairness, and resilience studies.
- Scalable performance profiling and report-ready outputs.

## 37) Frontend Functional Expectations (Detailed)

Frontend should progressively support:
- 3D neighborhood state view (generation, storage, import/export).
- Flow indications for P2P and grid transactions.
- Market signal visualization (price/time, order flow summaries).
- Policy mode toggles for comparative interpretation.
- Stress scenario controls and replay.

The frontend is part of scientific interpretability, not only product polish.

## 38) Service Contract Guidance (Detailed)

### 38.1 Grid Environment Service

- Owns simulation stepping and state transition logic.
- Must expose explicit state snapshots and event payload contracts.

### 38.2 Agent Runtime Service

- Owns policy inference/training loop boundaries.
- Must keep policy mode and baseline mode switchable.

### 38.3 GNN Coordination Service

- Owns topology-based message passing and embedding generation.
- Must maintain stable embedding schema versioning for downstream consumers.

### 38.4 Market Engine Service

- Owns order book, matching, and settlement event generation.
- Must include topology-aware feasibility in matching logic.

### 38.5 Analytics and Evaluation Service

- Owns comparative metric calculations and report generation.
- Must preserve reproducible calculation paths.

## 39) AI Review and PR Commentary Rules

When AI is asked to review code in this repository, prioritize:
1. Behavioral regressions against decentralized intent.
2. Breakage to baseline comparability.
3. Missing metric instrumentation.
4. Event/schema instability risks.
5. Simulation reproducibility risks.

AI should avoid review feedback that is only style-oriented when scientific validity or architecture integrity is at risk.

## 40) Final Operational Checklist

Before closing any substantial implementation task, AI should verify:

1. Decentralized default remains intact.
2. GNN coordination path remains intact.
3. P2P market path remains intact.
4. Blockchain stayed out of scope unless explicitly requested.
5. Major architecture change requests were gated by migration rationale.
6. Baseline comparators remain feasible.
7. Tests/validation were updated when behavior changed.
8. Temporary shortcuts are documented with migration path.

If any check fails, AI should not silently proceed; it should call out the risk and request user guidance.
