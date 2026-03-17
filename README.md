# Adaptive-Engine-Baseline-Model
**Ultra-Lightweight Adaptive Cycle Engine — Phase 2 (Baseline 0-D Turbofan Model + Thermodynamic Validation)**  
Author: **Maxon Ericsson**

---

## Project Overview
This repository contains the **baseline thermodynamic model** and **Phase 2 validation outputs** for a next-generation **three-stream adaptive cycle engine** digitally integrated with a **morphing diverterless supersonic inlet (DSI)**.

The goal of this project is to develop, validate, and publish an open-source propulsion simulation framework suitable for early NGAD-class conceptual design. The project spans five phases — this repository covers **Phase 2** in full.

---

## Phase 2 — Completed Deliverables

### 1. Baseline 0-D Turbofan Engine Model
A fully modular Python engine model implementing a real open Brayton cycle:

**Flow path:** Inlet → Fan → Compressor → Combustor → Turbine → Nozzle

- Low-bypass turbofan architecture (BPR = 0.3, NGAD-class)
- Fan stage with bypass stream split and cold nozzle thrust
- Isentropic component models with real efficiency losses
- Work-balanced turbine (shaft power equals compressor demand)
- Choked converging nozzle with Mach exit diagnostics
- ISA atmosphere model with isentropic ram recovery for flight conditions

### 2. Performance Plots
Seven high-resolution output plots saved to `src/outputs/plots/`:

| Plot | Description |
|------|-------------|
| `thrust_vs_temperature.png` | Thrust sensitivity to ambient temperature |
| `isp_vs_temperature.png` | Isp sensitivity to ambient temperature |
| `thrust_vs_pr.png` | Thrust vs compressor pressure ratio |
| `isp_vs_pr.png` | Isp vs compressor pressure ratio |
| `thrust_flight_envelope.png` | Thrust heatmap across Mach × Altitude grid |
| `isp_flight_envelope.png` | Isp heatmap across Mach × Altitude grid |
| `ts_diagram.png` | Temperature-Entropy diagram (real Brayton cycle) |

### 3. Validation Report
Thermodynamic benchmarking against the **GE CF6-80C2** and **Rolls-Royce RB211-535** reference engines. Key findings:
- Component temperatures within 5% of published reference values
- Isentropic efficiencies (0.87–0.90) match published industry ranges
- Nozzle choke condition (M = 1.0) physically correct for operating pressure ratios
- Isp and thrust differences fully explained by intentional BPR design difference

### 4. Jupyter Validation Notebook
`Engine_Performance.ipynb` — interactive thermodynamic documentation including:
- Station outputs (T2–T5, P2–P5) with explicit units
- Nozzle choke diagnostics
- Sensitivity sweep plots inline
- Flight envelope heatmaps
- Markdown explanations at each stage

---

## Repository Structure
```
Adaptive-engine-baseline-model/
├── src/
│   ├── engine_model.py          # Main engine class (fan + core integration)
│   ├── Performance_Plots.py     # All plot generation scripts
│   ├── Engine_Performance.ipynb # Validation notebook
│   ├── components/
│   │   ├── fan.py               # Fan + bypass stream model
│   │   ├── compressor.py        # HP compressor model
│   │   ├── combustor.py         # Combustor model
│   │   ├── turbine.py           # Work-balanced turbine model
│   │   └── nozzle.py            # Converging nozzle with choke detection
│   └── outputs/
│       └── plots/               # Generated PNG output files
├── README.md
└── requirements.txt
```

---

## Baseline Design Parameters

| Parameter | Value |
|-----------|-------|
| Total Mass Flow | 50.0 kg/s |
| Bypass Ratio (BPR) | 0.30 |
| Fan Pressure Ratio (FPR) | 1.60 |
| Overall Pressure Ratio (OPR) | 18.0 |
| Fuel-Air Ratio | 0.020 |
| Compressor Efficiency | 0.88 |
| Turbine Efficiency | 0.90 |
| Fan Efficiency | 0.87 |

## Baseline Output (ISA Sea Level Static)

| Metric | Value |
|--------|-------|
| Core Thrust | 23.64 kN |
| Bypass Thrust | 3.36 kN |
| **Total Thrust** | **27.00 kN** |
| Specific Impulse | 3578 s |
| Fuel Flow | 0.769 kg/s |
| Nozzle Exit Mach | 1.00 (choked) |

---

## Setup & Usage
```bash
# Clone the repository
git clone https://github.com/ericssonmaxon/Adaptive-engine-baseline-model.git
cd Adaptive-engine-baseline-model/src

# Activate virtual environment
source ../.venv/bin/activate

# Run baseline engine
python engine_model.py

# Generate all plots
python Performance_Plots.py

# Open validation notebook
jupyter notebook Engine_Performance.ipynb
```

---

## Requirements
See `requirements.txt`. Key dependencies: `numpy`, `matplotlib`, `jupyter`.
