"""
parametric_sweep.py
Phase 3 Parametric Performance Sweep — Adaptive vs Fixed Cycle
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine — Phase 3

This script:
    1. Sweeps Mach (0.0 to 1.8) x Altitude (0 to 12,000 m) grid
    2. Runs adaptive cycle and fixed cycle at every point
    3. Computes performance delta (adaptive minus fixed)
    4. Exports full dataset to CSV
    5. Generates comparison plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from engine_model import EngineModel

# ============================================================
#  OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR_PLOTS = os.path.join("outputs", "plots")
OUTPUT_DIR_DATA  = os.path.join("outputs", "data")
os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_DIR_DATA,  exist_ok=True)


# ============================================================
#  ISA ATMOSPHERE & RAM RECOVERY
# ============================================================

def isa_atmosphere(altitude_m: float):
    """ISA troposphere + lower-stratosphere model — returns T_static, P_static."""
    T_SL     = 288.15    # sea-level temperature [K]
    L        = 0.0065    # troposphere lapse rate [K/m]
    H_TROPO  = 11000.0   # tropopause altitude [m]
    T_STRAT  = 216.65    # isothermal stratosphere temperature [K]

    if altitude_m <= H_TROPO:
        T = T_SL - L * altitude_m
        P = 101325.0 * (T / T_SL) ** 5.2561
    else:
        # Isothermal stratosphere layer (11 000 – 20 000 m)
        T = T_STRAT
        P_tropo = 101325.0 * (T_STRAT / T_SL) ** 5.2561
        P = P_tropo * np.exp(-9.81 * (altitude_m - H_TROPO) / (287.05 * T_STRAT))
    return T, P


def inlet_total_conditions(T_static: float, P_static: float, mach: float):
    """Isentropic ram recovery — returns T_total, P_total."""
    gamma = 1.4
    T0 = T_static * (1.0 + (gamma - 1.0) / 2.0 * mach ** 2)
    P0 = P_static * (T0 / T_static) ** (gamma / (gamma - 1.0))
    return T0, P0


# ============================================================
#  PARAMETRIC SWEEP
# ============================================================

def run_parametric_sweep():
    """
    Run adaptive and fixed cycle engines across Mach x Altitude grid.
    Returns a pandas DataFrame with all results.
    """

    machs     = np.linspace(0.0, 1.8, 25)
    altitudes = np.linspace(0, 12000, 20)

    engine_adaptive = EngineModel()               # bypass_ratio=None → adaptive
    engine_fixed    = EngineModel(bypass_ratio=0.3)  # Phase 2 baseline

    records = []

    print("Running parametric sweep...")
    print(f"  Grid: {len(machs)} Mach points x {len(altitudes)} altitude points = {len(machs)*len(altitudes)} evaluations per engine")
    print(f"  Total engine runs: {2 * len(machs) * len(altitudes)}\n")

    for alt in altitudes:
        T_s, P_s = isa_atmosphere(alt)

        for mach in machs:
            T0, P0 = inlet_total_conditions(T_s, P_s, mach)

            # --- Adaptive cycle run ---
            a = engine_adaptive.run(T0, P0, mach=mach, altitude_m=alt)

            # --- Fixed cycle run ---
            f = engine_fixed.run(T0, P0, mach=mach, altitude_m=alt)

            records.append({
                # Flight condition
                "mach":            round(mach, 4),
                "altitude_m":      round(alt,  1),
                "altitude_ft":     round(alt * 3.28084, 0),

                # Adaptive cycle results
                "adp_bpr":         a["bypass_ratio"],
                "adp_thrust_kN":   round(a["thrust_N"] / 1e3, 4),
                "adp_core_kN":     round(a["core_thrust_N"] / 1e3, 4),
                "adp_bypass_kN":   round(a["bypass_thrust_N"] / 1e3, 4),
                "adp_isp_s":       round(a["specific_impulse_s"], 2),
                "adp_fuel_kgs":    round(a["fuel_flow_kg_s"], 5),
                "adp_m_core":      round(a["m_core_kg_s"], 4),
                "adp_m_bypass":    round(a["m_bypass_kg_s"], 4),

                # Fixed cycle results
                "fix_bpr":         f["bypass_ratio"],
                "fix_thrust_kN":   round(f["thrust_N"] / 1e3, 4),
                "fix_core_kN":     round(f["core_thrust_N"] / 1e3, 4),
                "fix_bypass_kN":   round(f["bypass_thrust_N"] / 1e3, 4),
                "fix_isp_s":       round(f["specific_impulse_s"], 2),
                "fix_fuel_kgs":    round(f["fuel_flow_kg_s"], 5),

                # Delta (adaptive minus fixed)
                "delta_thrust_kN": round((a["thrust_N"] - f["thrust_N"]) / 1e3, 4),
                "delta_isp_s":     round(a["specific_impulse_s"] - f["specific_impulse_s"], 2),
                "delta_fuel_kgs":  round(a["fuel_flow_kg_s"] - f["fuel_flow_kg_s"], 5),
                "delta_bpr":       round(a["bypass_ratio"] - f["bypass_ratio"], 4),
            })

    df = pd.DataFrame(records)
    print(f"Sweep complete. {len(df)} data points generated.\n")
    return df


# ============================================================
#  EXPORT TO CSV
# ============================================================

def export_csv(df: pd.DataFrame):
    path = os.path.join(OUTPUT_DIR_DATA, "phase3_parametric_data.csv")
    df.to_csv(path, index=False)
    print(f"Archival dataset saved to: {path}")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}\n")


# ============================================================
#  COMPARISON PLOTS
# ============================================================

def plot_thrust_comparison(df: pd.DataFrame):
    """Side-by-side thrust heatmaps: adaptive vs fixed."""
    machs     = np.sort(df["mach"].unique())
    altitudes = np.sort(df["altitude_ft"].unique())

    adp = df.pivot(index="altitude_ft", columns="mach", values="adp_thrust_kN").values
    fix = df.pivot(index="altitude_ft", columns="mach", values="fix_thrust_kN").values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cp1 = axes[0].contourf(machs, altitudes / 1000, adp, levels=20, cmap="plasma")
    fig.colorbar(cp1, ax=axes[0], label="Thrust (kN)")
    axes[0].set_title("Adaptive Cycle — Thrust (kN)")
    axes[0].set_xlabel("Mach Number")
    axes[0].set_ylabel("Altitude (1000s of ft)")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    cp2 = axes[1].contourf(machs, altitudes / 1000, fix, levels=20, cmap="plasma")
    fig.colorbar(cp2, ax=axes[1], label="Thrust (kN)")
    axes[1].set_title("Fixed Cycle (BPR=0.3) — Thrust (kN)")
    axes[1].set_xlabel("Mach Number")
    axes[1].set_ylabel("Altitude (1000s of ft)")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Thrust Comparison — Adaptive vs Fixed Cycle", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "thrust_adaptive_vs_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_isp_comparison(df: pd.DataFrame):
    """Side-by-side Isp heatmaps: adaptive vs fixed."""
    machs     = np.sort(df["mach"].unique())
    altitudes = np.sort(df["altitude_ft"].unique())

    adp = df.pivot(index="altitude_ft", columns="mach", values="adp_isp_s").values
    fix = df.pivot(index="altitude_ft", columns="mach", values="fix_isp_s").values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cp1 = axes[0].contourf(machs, altitudes / 1000, adp, levels=20, cmap="viridis")
    fig.colorbar(cp1, ax=axes[0], label="Isp (s)")
    axes[0].set_title("Adaptive Cycle — Isp (s)")
    axes[0].set_xlabel("Mach Number")
    axes[0].set_ylabel("Altitude (1000s of ft)")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    cp2 = axes[1].contourf(machs, altitudes / 1000, fix, levels=20, cmap="viridis")
    fig.colorbar(cp2, ax=axes[1], label="Isp (s)")
    axes[1].set_title("Fixed Cycle (BPR=0.3) — Isp (s)")
    axes[1].set_xlabel("Mach Number")
    axes[1].set_ylabel("Altitude (1000s of ft)")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Isp Comparison — Adaptive vs Fixed Cycle", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "isp_adaptive_vs_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_delta_heatmaps(df: pd.DataFrame):
    """Delta heatmaps showing where adaptive cycle gains and loses."""
    machs     = np.sort(df["mach"].unique())
    altitudes = np.sort(df["altitude_ft"].unique())

    d_thrust = df.pivot(index="altitude_ft", columns="mach", values="delta_thrust_kN").values
    d_isp    = df.pivot(index="altitude_ft", columns="mach", values="delta_isp_s").values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cp1 = axes[0].contourf(machs, altitudes / 1000, d_thrust, levels=20, cmap="RdBu_r")
    fig.colorbar(cp1, ax=axes[0], label="Delta Thrust (kN)")
    axes[0].set_title("Thrust Delta — Adaptive minus Fixed (kN)")
    axes[0].set_xlabel("Mach Number")
    axes[0].set_ylabel("Altitude (1000s of ft)")
    axes[0].axvline(x=0.4,  color="white", linestyle="--", alpha=0.6, linewidth=1)
    axes[0].axvline(x=0.9,  color="white", linestyle="--", alpha=0.6, linewidth=1)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    cp2 = axes[1].contourf(machs, altitudes / 1000, d_isp, levels=20, cmap="RdBu_r")
    fig.colorbar(cp2, ax=axes[1], label="Delta Isp (s)")
    axes[1].set_title("Isp Delta — Adaptive minus Fixed (s)")
    axes[1].set_xlabel("Mach Number")
    axes[1].set_ylabel("Altitude (1000s of ft)")
    axes[1].axvline(x=0.4,  color="white", linestyle="--", alpha=0.6, linewidth=1)
    axes[1].axvline(x=0.9,  color="white", linestyle="--", alpha=0.6, linewidth=1)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Performance Delta — Adaptive Cycle Gains and Losses vs Fixed Baseline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "delta_adaptive_vs_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_bpr_schedule_map(df: pd.DataFrame):
    """Heatmap of active BPR across the flight envelope."""
    machs     = np.sort(df["mach"].unique())
    altitudes = np.sort(df["altitude_ft"].unique())
    bpr_map   = df.pivot(index="altitude_ft", columns="mach", values="adp_bpr").values

    plt.figure(figsize=(10, 6))
    cp = plt.contourf(machs, altitudes / 1000, bpr_map, levels=20, cmap="coolwarm")
    plt.colorbar(cp, label="Bypass Ratio (BPR)")
    plt.xlabel("Mach Number")
    plt.ylabel("Altitude (1000s of ft)")
    plt.title("Adaptive BPR Schedule — Flight Envelope Map")
    plt.axvline(x=0.4, color="white", linestyle="--", alpha=0.7, linewidth=1.2, label="Mach 0.4 (takeoff limit)")
    plt.axvline(x=0.9, color="white", linestyle="--", alpha=0.7, linewidth=1.2, label="Mach 0.9 (cruise peak)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "bpr_schedule_map.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_mach_slice(df: pd.DataFrame):
    """
    Line plot of Thrust and Isp vs Mach at sea level —
    adaptive vs fixed direct comparison.
    """
    sl = df.query("altitude_m == 0").sort_values(by="mach")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(sl["mach"], sl["adp_thrust_kN"], linewidth=2.5, color="darkblue",  label="Adaptive Cycle")
    axes[0].plot(sl["mach"], sl["fix_thrust_kN"], linewidth=2.5, color="darkred", linestyle="--", label="Fixed Cycle (BPR=0.3)")
    axes[0].set_xlabel("Mach Number")
    axes[0].set_ylabel("Total Thrust (kN)")
    axes[0].set_title("Thrust vs Mach — Sea Level")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].axvline(x=0.4, color="grey", linestyle=":", alpha=0.7)
    axes[0].axvline(x=0.9, color="grey", linestyle=":", alpha=0.7)

    axes[1].plot(sl["mach"], sl["adp_isp_s"], linewidth=2.5, color="darkblue",  label="Adaptive Cycle")
    axes[1].plot(sl["mach"], sl["fix_isp_s"], linewidth=2.5, color="darkred", linestyle="--", label="Fixed Cycle (BPR=0.3)")
    axes[1].set_xlabel("Mach Number")
    axes[1].set_ylabel("Specific Impulse (s)")
    axes[1].set_title("Isp vs Mach — Sea Level")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].axvline(x=0.4, color="grey", linestyle=":", alpha=0.7)
    axes[1].axvline(x=0.9, color="grey", linestyle=":", alpha=0.7)

    plt.suptitle("Sea Level Performance — Adaptive vs Fixed Cycle", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "sea_level_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  SUMMARY STATISTICS
# ============================================================

def print_summary(df: pd.DataFrame):
    print("=" * 60)
    print("PHASE 3 PARAMETRIC SWEEP — SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\n{'Metric':<35s} {'Adaptive':>12s}  {'Fixed':>12s}  {'Delta':>12s}")
    print("-" * 75)

    metrics = [
        ("Mean Thrust (kN)",    "adp_thrust_kN",  "fix_thrust_kN"),
        ("Mean Isp (s)",        "adp_isp_s",      "fix_isp_s"),
        ("Mean Fuel Flow (kg/s)","adp_fuel_kgs",  "fix_fuel_kgs"),
        ("Max Thrust (kN)",     "adp_thrust_kN",  "fix_thrust_kN"),
        ("Min Isp (s)",         "adp_isp_s",      "fix_isp_s"),
    ]

    funcs = ["mean", "mean", "mean", "max", "min"]
    for (label, acol, fcol), func in zip(metrics, funcs):
        a_val = getattr(df[acol], func)()
        f_val = getattr(df[fcol], func)()
        delta = a_val - f_val
        print(f"{label:<35s} {a_val:>12.2f}  {f_val:>12.2f}  {delta:>+12.2f}")

    print(f"\n  BPR range (adaptive): {df['adp_bpr'].min():.3f} — {df['adp_bpr'].max():.3f}")
    print(f"  Data points:          {len(df)}")
    print(f"  Mach range:           {df['mach'].min():.2f} — {df['mach'].max():.2f}")
    print(f"  Altitude range:       {df['altitude_m'].min():.0f} — {df['altitude_m'].max():.0f} m")
    print("=" * 60)


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 3 — PARAMETRIC SWEEP")
    print("Adaptive Cycle vs Fixed Cycle Performance Comparison")
    print("=" * 60 + "\n")

    df = run_parametric_sweep()
    export_csv(df)

    print("Generating comparison plots...")
    plot_thrust_comparison(df)
    plot_isp_comparison(df)
    plot_delta_heatmaps(df)
    plot_bpr_schedule_map(df)
    plot_mach_slice(df)
    print(f"All plots saved to: {OUTPUT_DIR_PLOTS}\n")

    print_summary(df)


if __name__ == "__main__":
    main()