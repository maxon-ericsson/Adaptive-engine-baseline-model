"""
performance_plots.py
Generates all Phase 2 thermodynamic performance plots for the
Baseline 0-D Engine Model.

Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine 

This script produces:
    • Thrust vs Ambient Temperature
    • Isp vs Ambient Temperature
    • Thrust vs Compressor Pressure Ratio
    • Isp vs Compressor Pressure Ratio
    • Thrust across Flight Envelope (Mach vs Altitude heatmap)
    • Isp across Flight Envelope (Mach vs Altitude heatmap)

Outputs are saved as high-resolution PNG files in:
    outputs/plots/
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import engine model
from engine_model import EngineModel


# ============================================================
#  CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
#  ATMOSPHERE & RAM RECOVERY HELPERS
# ============================================================

def isa_atmosphere(altitude_m: float):
    """
    International Standard Atmosphere (ISA) troposphere model.
    Valid from 0 to 11,000 m (36,089 ft).

    Args:
        altitude_m: Geometric altitude [m]

    Returns:
        T_static: Static temperature [K]
        P_static: Static pressure [Pa]
    """
    T_static = 288.15 - 0.0065 * altitude_m
    P_static = 101325.0 * (T_static / 288.15) ** 5.2561
    return T_static, P_static


def inlet_total_conditions(T_static: float, P_static: float, mach: float):
    """
    Isentropic ram recovery — converts freestream static conditions
    to inlet total (stagnation) conditions using Mach number.

    At Mach 0 (static): T_total = T_static, P_total = P_static.
    At higher Mach: ram compression raises both T and P before
    the air even reaches the compressor face.

    Args:
        T_static: Freestream static temperature [K]
        P_static: Freestream static pressure [Pa]
        mach:     Freestream Mach number [-]

    Returns:
        T_total: Inlet total temperature [K]
        P_total: Inlet total pressure [Pa]
    """
    gamma = 1.4
    T_total = T_static * (1.0 + (gamma - 1.0) / 2.0 * mach ** 2)
    P_total = P_static * (T_total / T_static) ** (gamma / (gamma - 1.0))
    return T_total, P_total


# ============================================================
#  EXISTING PLOT FUNCTIONS (UNCHANGED)
# ============================================================

def plot_isp_vs_temperature(engine):
    temps = np.linspace(230, 310, 20)
    isps = [engine.run(T, 101325)["specific_impulse_s"] for T in temps]

    plt.figure(figsize=(8, 5))
    plt.plot(temps, isps, linewidth=2, color="purple")
    plt.xlabel("Ambient Temperature (K)")
    plt.ylabel("Specific Impulse (s)")
    plt.title("Isp vs Ambient Temperature")
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "isp_vs_temperature.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_thrust_vs_temperature(engine):
    temps = np.linspace(230, 310, 20)
    thrusts = [engine.run(T, 101325)["thrust_N"] for T in temps]

    plt.figure(figsize=(8, 5))
    plt.plot(temps, thrusts, linewidth=2, color="darkblue")
    plt.xlabel("Ambient Temperature (K)")
    plt.ylabel("Thrust (N)")
    plt.title("Thrust vs Ambient Temperature")
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "thrust_vs_temperature.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_thrust_vs_pr(engine):
    prs = np.linspace(10, 40, 20)
    thrusts = []

    for PR in prs:
        engine.compressor_PR = PR
        thrusts.append(engine.run(288.15, 101325)["thrust_N"])

    engine.compressor_PR = 18.0  # Reset PR to default

    plt.figure(figsize=(8, 5))
    plt.plot(prs, thrusts, linewidth=2, color="darkred")
    plt.xlabel("Compressor Pressure Ratio")
    plt.ylabel("Thrust (N)")
    plt.title("Thrust vs Compressor Pressure Ratio")
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "thrust_vs_pr.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_isp_vs_pr(engine):
    prs = np.linspace(10, 40, 20)
    isps = []

    for PR in prs:
        engine.compressor_PR = PR
        isps.append(engine.run(288.15, 101325)["specific_impulse_s"])

    engine.compressor_PR = 18.0  # Reset default

    plt.figure(figsize=(8, 5))
    plt.plot(prs, isps, linewidth=2, color="green")
    plt.xlabel("Compressor Pressure Ratio")
    plt.ylabel("Specific Impulse (s)")
    plt.title("Isp vs Compressor Pressure Ratio")
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "isp_vs_pr.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  FLIGHT ENVELOPE SWEEP FUNCTIONS (NEW)
# ============================================================

def plot_thrust_envelope(engine):
    """
    Heatmap of Thrust [N] across Mach number and altitude.

    For each (Mach, altitude) point:
        1. ISA atmosphere gives T_static, P_static
        2. Ram recovery gives T_total, P_total at inlet
        3. engine.run() computes thrust from those inlet conditions
    """
    machs     = np.linspace(0.0, 1.8, 25)
    altitudes = np.linspace(0, 12000, 25)     # metres (0 to ~40,000 ft)

    thrust_map = np.zeros((len(altitudes), len(machs)))

    for i, alt in enumerate(altitudes):
        T_s, P_s = isa_atmosphere(alt)
        for j, mach in enumerate(machs):
            T0, P0 = inlet_total_conditions(T_s, P_s, mach)
            thrust_map[i, j] = engine.run(T0, P0)["thrust_N"]

    # Convert altitude axis to feet for readability
    altitudes_ft = altitudes * 3.28084

    plt.figure(figsize=(10, 6))
    cp = plt.contourf(machs, altitudes_ft / 1000, thrust_map, levels=20, cmap="plasma")
    plt.colorbar(cp, label="Thrust (N)")
    plt.xlabel("Mach Number")
    plt.ylabel("Altitude (1000s of ft)")
    plt.title("Thrust Flight Envelope — Mach vs Altitude")
    plt.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(OUTPUT_DIR, "thrust_flight_envelope.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_isp_envelope(engine):
    """
    Heatmap of Specific Impulse [s] across Mach number and altitude.
    Same sweep logic as thrust envelope.
    """
    machs     = np.linspace(0.0, 1.8, 25)
    altitudes = np.linspace(0, 12000, 25)

    isp_map = np.zeros((len(altitudes), len(machs)))

    for i, alt in enumerate(altitudes):
        T_s, P_s = isa_atmosphere(alt)
        for j, mach in enumerate(machs):
            T0, P0 = inlet_total_conditions(T_s, P_s, mach)
            isp_map[i, j] = engine.run(T0, P0)["specific_impulse_s"]

    altitudes_ft = altitudes * 3.28084

    plt.figure(figsize=(10, 6))
    cp = plt.contourf(machs, altitudes_ft / 1000, isp_map, levels=20, cmap="viridis")
    plt.colorbar(cp, label="Specific Impulse (s)")
    plt.xlabel("Mach Number")
    plt.ylabel("Altitude (1000s of ft)")
    plt.title("Isp Flight Envelope — Mach vs Altitude")
    plt.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(OUTPUT_DIR, "isp_flight_envelope.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_ts_diagram(engine):
    """
    Temperature-Entropy diagram for the baseline Brayton cycle.
    Computes entropy change at each station relative to ambient.
    """
    import numpy as np

    # Run engine at sea level standard conditions
    T0 = 288.15    # K
    P0 = 101325.0  # Pa
    out = engine.run(T0, P0)

    # Cp and gamma for entropy calculations
    Cp    = 1005.0  # J/kg-K
    gamma = 1.4
    R     = 287.0   # J/kg-K

    # Station temperatures
    stations = {
        "0 (Ambient)":      (T0,           P0),
        "1 (Fan Exit)":     (out["T_fan"], out["P_fan"]),
        "2 (Compressor)":   (out["T2"],    out["P2"]),
        "3 (Combustor)":    (out["T3"],    out["P3"]),
        "4 (Turbine)":      (out["T4"],    out["P4"]),
        "5 (Nozzle Exit)":  (out["T5"],    out["P5"]),
    }

    labels = list(stations.keys())
    temps  = [v[0] for v in stations.values()]
    press  = [v[1] for v in stations.values()]

    # Compute entropy change relative to station 0
    # Δs = Cp * ln(T2/T1) - R * ln(P2/P1)
    entropies = [0.0]
    for i in range(1, len(temps)):
        ds = (Cp * np.log(temps[i] / temps[i-1])
              - R  * np.log(press[i] / press[i-1]))
        entropies.append(entropies[-1] + ds)

    # Convert to kJ/kg-K for readability
    entropies_kJ = [s / 1000 for s in entropies]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(entropies_kJ, temps, 'o-', linewidth=2,
             color="darkred", markersize=8, markerfacecolor="white",
             markeredgewidth=2)

    # Label each station
    offsets = [(0.002, 20), (0.002, 20), (0.002, 20),
               (0.002, 20), (-0.08, 20), (-0.08, 20)]
    for i, label in enumerate(labels):
        dx, dy = offsets[i]
        plt.annotate(label,
                     xy=(entropies_kJ[i], temps[i]),
                     xytext=(entropies_kJ[i] + dx, temps[i] + dy),
                     fontsize=8, color="darkred")

    plt.xlabel("Specific Entropy Change (kJ/kg·K)")
    plt.ylabel("Temperature (K)")
    plt.title("T-S Diagram — Baseline Turbofan Brayton Cycle")
    plt.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(OUTPUT_DIR, "ts_diagram.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  MAIN EXECUTION
# ============================================================

def main():
    print("Generating Phase 2 Performance Plots...")

    engine = EngineModel()

    # --- Existing plots ---
    plot_isp_vs_temperature(engine)
    plot_thrust_vs_temperature(engine)
    plot_thrust_vs_pr(engine)
    plot_isp_vs_pr(engine)
    plot_ts_diagram(engine)

    # --- Flight envelope sweeps ---
    print("Running flight envelope sweep (Mach x Altitude grid)...")
    plot_thrust_envelope(engine)
    plot_isp_envelope(engine)

    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()