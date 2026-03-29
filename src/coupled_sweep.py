"""
coupled_sweep.py
----------------
Phase 4 – Coupled Inlet-Engine Parametric Sweep
Ultra-Lightweight Adaptive Cycle Engine Project
Author: Maxon Ericsson

Sweeps inlet geometry (h, theta) and flight conditions (Mach, altitude)
simultaneously, running the fully coupled inlet-engine model at each point.

Grid
----
  Mach     : 0.8, 1.0, 1.4, 1.6, 2.0          (5 values)
  Altitude  : 0, 5 000, 10 000, 15 000 m        (4 values)
  h         : 0.05, 0.10, 0.15, 0.20, 0.25 m   (5 values)
  theta     : 5, 10, 15, 20, 25 deg             (5 values)
  r         : 0.04 m (nominal, fixed)
  Total     : 5 × 4 × 5 × 5 = 500 points

Outputs
-------
  outputs/coupled_sweep_results.csv   — full 500-row dataset
  Console summary table               — best/worst TSFC by Mach

Atmosphere model
----------------
  ISA troposphere (h ≤ 11 000 m):
    T = 288.15 − 0.0065 × alt  [K]
    P = 101325 × (T / 288.15)^5.2561  [Pa]
"""

import numpy as np
import csv
import os
import sys
from itertools import product

from engine_model import EngineModel

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
R_FIXED        = 0.04       # leading-edge radius fixed across sweep [m]
MASS_FLOW      = 50.0       # core air mass flow [kg/s]
COMPRESSOR_PR  = 18.0       # compressor pressure ratio [-]
COMPRESSOR_EFF = 0.88       # compressor isentropic efficiency [-]
TURBINE_EFF    = 0.90       # turbine efficiency [-]
TIT = 1550.0                # target turbine inlet temperature [K]
DC60_LIMIT     = 0.40       # MIL-E-5007D distortion limit [-]

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "coupled_sweep_results.csv")

# ---------------------------------------------------------------------------
# SWEEP GRID
# ---------------------------------------------------------------------------
MACH_VALUES  = [0.80, 1.00, 1.40, 1.60, 2.00]
ALT_VALUES   = [0, 5_000, 10_000, 15_000]        # metres
H_VALUES     = [0.05, 0.10, 0.15, 0.20, 0.25]    # bump height [m]
THETA_VALUES = [5.0, 10.0, 15.0, 20.0, 25.0]     # contouring angle [deg]

TOTAL_POINTS = (len(MACH_VALUES) * len(ALT_VALUES) *
                len(H_VALUES)    * len(THETA_VALUES))


# ---------------------------------------------------------------------------
# ISA ATMOSPHERE
# ---------------------------------------------------------------------------

def isa_atmosphere(altitude_m: float):
    """
    International Standard Atmosphere, troposphere only (alt ≤ 11 000 m).

    Parameters
    ----------
    altitude_m : geometric altitude [m]

    Returns
    -------
    T_K  : static temperature [K]
    P_Pa : static pressure    [Pa]
    """
    T_sl   = 288.15          # sea-level temperature [K]
    P_sl   = 101_325.0       # sea-level pressure    [Pa]
    L      = 0.0065          # lapse rate            [K/m]
    exp    = 5.2561          # barometric exponent   [-]

    alt = min(altitude_m, 11_000.0)   # clamp to tropopause
    T   = T_sl - L * alt
    P   = P_sl * (T / T_sl) ** exp
    return T, P


# ---------------------------------------------------------------------------
# SINGLE SWEEP POINT
# ---------------------------------------------------------------------------

def run_point(mach: float, altitude: float,
              h: float, theta: float) -> dict:
    """
    Run one coupled inlet-engine evaluation.

    Returns a flat dict of all inputs and outputs for CSV logging.
    """
    T_amb, P_amb = isa_atmosphere(altitude)

    engine = EngineModel(
        mass_flow      = MASS_FLOW,
        compressor_PR  = COMPRESSOR_PR,
        compressor_eff = COMPRESSOR_EFF,
        turbine_eff    = TURBINE_EFF,
        tit            = TIT,
        mach           = mach,
        inlet_h        = h,
        inlet_r        = R_FIXED,
        inlet_theta    = theta,
    )

    out = engine.run(T_amb, P_amb)

    return {
        # --- inputs ---
        "mach"          : mach,
        "altitude_m"    : altitude,
        "h_m"           : h,
        "r_m"           : R_FIXED,
        "theta_deg"     : theta,
        "T_ambient_K"   : round(T_amb, 3),
        "P_ambient_Pa"  : round(P_amb, 1),
        # --- inlet ---
        "p_recovery"    : round(out["inlet_p_recovery"],   6),
    }


# ---------------------------------------------------------------------------
# FULL SWEEP
# ---------------------------------------------------------------------------

def run_sweep(verbose: bool = True) -> list:
    """
    Execute the full 500-point coupled sweep.

    Parameters
    ----------
    verbose : print progress every 50 points

    Returns
    -------
    list of result dicts, one per sweep point
    """
    results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  coupled_sweep.py — Phase 4 Coupled Sweep")
        print(f"  {TOTAL_POINTS} points: "
              f"{len(MACH_VALUES)}M × {len(ALT_VALUES)}alt × "
              f"{len(H_VALUES)}h × {len(THETA_VALUES)}θ")
        print("=" * 60)

    for i, (mach, alt, h, theta) in enumerate(
            product(MACH_VALUES, ALT_VALUES, H_VALUES, THETA_VALUES), 1):

        row = run_point(mach, alt, h, theta)
        results.append(row)

        if verbose and (i % 50 == 0 or i == TOTAL_POINTS):
            print(f"  [{i:>3}/{TOTAL_POINTS}]  "
                  f"M={mach:.2f}  alt={alt:>6}m  "
                  f"h={h:.2f}m  θ={theta:.0f}°  "
                  f"→  P_rec={row['p_recovery']:.4f}  "
                  f"TSFC={row['tsfc']:.6f}")

    return results


# ---------------------------------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------------------------------

def save_csv(results: list) -> None:
    """Write results list to CSV file."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved {len(results)} rows → {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(results: list) -> None:
    """Print best/worst TSFC by Mach number."""
    print("\n" + "=" * 60)
    print("  Summary — TSFC by Mach (best and worst geometry)")
    print("=" * 60)
    print(f"\n  {'Mach':>5}  {'Best TSFC':>10}  {'h':>5}  {'θ':>5}  "
          f"{'Worst TSFC':>11}  {'DC60':>6}  {'MIL':>4}")
    print("  " + "-" * 56)

    for mach in MACH_VALUES:
        subset = [r for r in results if r["mach"] == mach
                  and r["thrust_N"] > 0]
        if not subset:
            continue

        valid   = [r for r in subset if r["dc60"] <= DC60_LIMIT]
        best    = min(valid,  key=lambda r: r["tsfc"]) if valid  else min(subset, key=lambda r: r["tsfc"])
        worst   = max(subset, key=lambda r: r["tsfc"])

        print(f"  {mach:>5.2f}  {best['tsfc']:>10.6f}  "
              f"{best['h_m']:>5.2f}  {best['theta_deg']:>5.1f}  "
              f"{worst['tsfc']:>11.6f}  {worst['dc60']:>6.3f}  "
              f"{'OK' if best['meets_mil'] else 'FAIL':>4}")

    n_fail = sum(1 for r in results if not r["meets_mil"])
    n_dc60 = sum(1 for r in results if r["dc60"] > DC60_LIMIT)
    print(f"\n  MIL-E-5007D failures : {n_fail}/{len(results)}")
    print(f"  DC60 > 0.40 cases   : {n_dc60}/{len(results)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_sweep(verbose=True)
    save_csv(results)
    print_summary(results)