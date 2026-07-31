# Adaptive-Engine-Baseline-Model
# Ultra-Lightweight Adaptive Cycle Engine (ULACE)
**Morphing DSI Inlet Integration — Multi-Phase Propulsion Simulation**
**Author:** Maxon Ericsson | **Duration:** November 2025 – May 2026 | **Status:** Phase 4 Python complete — SolidWorks CAD/FEA in progress

---

## Project Overview
A Python-based simulation framework for an NGAD-class adaptive cycle engine coupled with a parametric morphing diverterless supersonic inlet (DSI). Demonstrates coupled inlet-engine co-design across a Mach 0.8–2.0 flight envelope across five phases.

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Literature Review — ACE and Morphing DSI Inlets | Complete |
| 2 | Baseline 0-D Turbofan Model and Validation | Complete |
| 3 | Adaptive Cycle Engine with Dynamic BPR Scheduling | Complete |
| 4 | Morphing DSI Inlet Integration and Optimization (Python) | Complete |
| 4 | Morphing DSI Inlet CAD and FEA (SolidWorks) | In Progress |
| 5 | Final Technical Report and Presentation | Upcoming |

---

## Repository Structure
---

## Phase 2 — Baseline 0-D Turbofan Model
Modular Python engine model implementing a real open Brayton cycle. Flow path: Inlet → Fan → Compressor → Combustor → Turbine → Nozzle.

| Parameter | Value |
|-----------|-------|
| Total Mass Flow | 50.0 kg/s |
| Bypass Ratio | 0.30 |
| Overall Pressure Ratio | 18.0 |
| Compressor Efficiency | 0.88 |
| Turbine Efficiency | 0.90 |

**Baseline Output — ISA Sea Level Static**

| Metric | Value |
|--------|-------|
| Total Thrust | 27.00 kN |
| Specific Impulse | 3578 s |
| Fuel Flow | 0.769 kg/s |
| Nozzle Exit Mach | 1.00 (choked) |

Validated against the GE CF6-80C2 and Rolls-Royce RB211-535. Component temperatures within 5% of published reference values.

---

## Phase 3 — Adaptive Cycle Engine
Dynamic bypass ratio scheduling from BPR 0.10 (combat) to BPR 1.296 (cruise) as a function of Mach number, altitude, and combat mode flag via `bpr_schedule.py`. A 500-point parametric sweep in `parametric_sweep.py` compared adaptive vs. fixed-cycle performance across the full flight envelope.

---

## Phase 4 — Morphing DSI Inlet Integration

### Inlet Model (`inlet_model.py`)
Physics-based DSI bump model using a two-oblique-shock plus terminal normal shock system. The theta-beta-Mach relation is solved via Newton iteration. Total-pressure recovery is computed as `P_recovery = PR1 × PR2 × PR_normal`. A scalar distortion penalty is applied to fan-face total pressure.

| Parameter | Symbol | Range |
|-----------|--------|-------|
| Bump height | h | 0.05 – 0.25 m |
| Leading-edge radius | r | 0.01 – 0.08 m |
| Contouring angle | θ | 5 – 25 deg |

### Engine Coupling (`engine_model.py`)
Fan-face total pressure set as `P_fan_face = P_ambient × p_recovery`. Variable fuel-air ratio combustor solves `f = Cp × (TIT − T2) / (η_comb × LHV)` targeting TIT = 1550 K, propagating inlet pressure loss into fuel burn and TSFC.

### Parametric Sweep (`coupled_sweep.py`)
500-point grid across 5 Mach × 4 altitudes × 5 bump heights × 5 contouring angles. Full results saved to `outputs/coupled_sweep_results.csv`.

### Optimizer (`inlet_optimizer.py`)
Multi-start L-BFGS-B optimizer minimizing mission-weighted TSFC across three Mach points with pressure recovery as tiebreaker:
---

## Known Model Constraints
- Distortion is a scalar approximation — a true DC60 requires 2-D fan-face pressure resolution
- TSFC optimization shows near-zero improvement in the 0-D model — documented as a model fidelity constraint
- Mach 1.4 with θ < 10° produces NaN in some altitude conditions — flagged `valid=0` in sweep output and excluded from optimizer

---

## Reference Engine
The ACE design point is the author's own simulation target. The P&W XA103 is cited as a class reference only — no proprietary data has been used or reverse-engineered.