"""
bpr_schedule.py
Variable Bypass Ratio Schedule for Adaptive Cycle Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine — Phase 3

Physical Model:
- Bypass ratio is no longer fixed — it varies as a function of
  Mach number and altitude, mimicking the adaptive valve scheduling
  of an NGAD-class three-stream engine.
- At low speed: low BPR for maximum core thrust (takeoff/combat)
- At cruise speed: higher BPR for improved propulsive efficiency
- At altitude: slight BPR increase due to reduced core air density
"""

import numpy as np
from typing import Tuple


# ============================================================
# BPR SCHEDULE CONSTANTS
# ============================================================

# Mach breakpoints defining the schedule regions
MACH_TAKEOFF    = 0.4    # Below this: max thrust mode (low BPR)
MACH_TRANSONIC  = 0.9    # Below this: linear ramp from takeoff to cruise
MACH_SUPERSONIC = 1.8    # Above this: schedule is clamped

# BPR values at key operating points (sea level baseline)
BPR_MIN         = 0.15   # Minimum BPR — combat/takeoff mode
BPR_CRUISE      = 1.10   # Maximum BPR — efficient cruise mode
BPR_SUPERSONIC  = 0.35   # BPR at high supersonic — balance thrust/efficiency

# Altitude effect coefficient
# For every 1000 m of altitude gained, BPR increases slightly
# because thinner air allows the engine to push more bypass flow
ALTITUDE_BPR_COEFFICIENT = 0.008   # BPR increase per 1000 m altitude


# ============================================================
# CORE SCHEDULE FUNCTION
# ============================================================

def bpr_schedule(mach: float, altitude_m: float = 0.0) -> float:
    """
    Compute the adaptive bypass ratio as a function of Mach number
    and altitude.

    Physical Basis:
    - Low Mach (takeoff/combat): valve system closes third stream,
      directing maximum airflow to core for peak specific thrust.
    - Transonic acceleration: valves progressively open third stream,
      trading specific thrust for propulsive efficiency.
    - Supersonic cruise: BPR settles at an intermediate value —
      ram compression aids the core so bypass can remain partially open.
    - Altitude correction: as air density drops with altitude, the
      engine opens the bypass slightly to maintain efficient operation.

    Args:
        mach:       Freestream Mach number [-]
        altitude_m: Geometric altitude [m], default sea level

    Returns:
        float: Bypass ratio BPR = m_bypass / m_core [-]
    """

    # --------------------------------------------------------
    # STEP 1: Compute base BPR from Mach schedule
    # --------------------------------------------------------
    if mach <= MACH_TAKEOFF:
        # Low speed / takeoff / combat mode
        # Third stream fully diverted to core — minimum bypass
        bpr_base = BPR_MIN

    elif mach <= MACH_TRANSONIC:
        # Transonic acceleration — linear ramp
        # As Mach increases from 0.4 to 0.9, BPR ramps from
        # BPR_MIN up to BPR_CRUISE
        t = (mach - MACH_TAKEOFF) / (MACH_TRANSONIC - MACH_TAKEOFF)
        bpr_base = BPR_MIN + t * (BPR_CRUISE - BPR_MIN)

    elif mach <= MACH_SUPERSONIC:
        # Supersonic regime — linear ramp back down
        # BPR decreases from cruise peak back toward supersonic value
        # as ram compression reduces the need for bypass flow
        t = (mach - MACH_TRANSONIC) / (MACH_SUPERSONIC - MACH_TRANSONIC)
        bpr_base = BPR_CRUISE + t * (BPR_SUPERSONIC - BPR_CRUISE)

    else:
        # Beyond Mach 1.8 — clamp at supersonic value
        bpr_base = BPR_SUPERSONIC

    # --------------------------------------------------------
    # STEP 2: Apply altitude correction
    # --------------------------------------------------------
    # At altitude, air density drops — the engine can afford to
    # open the bypass slightly to maintain overall efficiency
    altitude_correction = ALTITUDE_BPR_COEFFICIENT * (altitude_m / 1000.0)
    bpr = bpr_base + altitude_correction

    # --------------------------------------------------------
    # STEP 3: Clamp to physically valid range
    # --------------------------------------------------------
    bpr = max(BPR_MIN, min(bpr, 1.5))

    return round(bpr, 4)


# ============================================================
# UTILITY — OPERATING MODE LABEL
# ============================================================

def get_operating_mode(mach: float, altitude_m: float = 0.0) -> str:
    """
    Return a human-readable label for the current engine operating mode.

    Args:
        mach:       Freestream Mach number [-]
        altitude_m: Geometric altitude [m]

    Returns:
        str: Operating mode label
    """
    bpr = bpr_schedule(mach, altitude_m)

    if mach <= MACH_TAKEOFF:
        return f"Takeoff / Combat  (M={mach:.2f}, BPR={bpr:.3f})"
    elif mach <= MACH_TRANSONIC:
        return f"Transonic Accel   (M={mach:.2f}, BPR={bpr:.3f})"
    elif mach <= MACH_SUPERSONIC:
        return f"Supersonic Cruise (M={mach:.2f}, BPR={bpr:.3f})"
    else:
        return f"Max Speed         (M={mach:.2f}, BPR={bpr:.3f})"


# ============================================================
# UTILITY — SWEEP ACROSS MACH RANGE
# ============================================================

def bpr_mach_sweep(
    mach_range: np.ndarray,
    altitude_m: float = 0.0
) -> np.ndarray:
    """
    Compute BPR across an array of Mach numbers at a fixed altitude.

    Args:
        mach_range: Array of Mach numbers [-]
        altitude_m: Fixed altitude for sweep [m]

    Returns:
        np.ndarray: Corresponding BPR values
    """
    return np.array([bpr_schedule(m, altitude_m) for m in mach_range])


# ============================================================
# VALIDATION & TESTING
# ============================================================

def validate_bpr_schedule() -> None:
    """
    Validate BPR schedule across representative operating points.
    """
    print("=" * 60)
    print("BPR SCHEDULE VALIDATION — PHASE 3 ADAPTIVE CYCLE")
    print("=" * 60)

    # Sea level test points
    test_points = [
        (0.0,   0,      "Static / Ground"),
        (0.3,   0,      "Low Subsonic"),
        (0.4,   0,      "Takeoff limit"),
        (0.65,  0,      "Mid Transonic"),
        (0.9,   0,      "Transonic peak"),
        (1.2,   5000,   "Low Supersonic"),
        (1.5,   10000,  "Mid Supersonic"),
        (1.8,   12000,  "Max Supersonic"),
    ]

    print(f"\n{'Condition':<25s} {'Mach':>6s} {'Alt (m)':>8s} {'BPR':>8s}  Mode")
    print("-" * 70)

    for mach, alt, label in test_points:
        bpr = bpr_schedule(mach, alt)
        mode = get_operating_mode(mach, alt)
        print(f"{label:<25s} {mach:>6.2f} {alt:>8.0f} {bpr:>8.4f}  {mode}")

    # Altitude effect test
    print(f"\n--- Altitude Effect at Mach 1.2 ---")
    for alt in [0, 3000, 6000, 9000, 12000]:
        bpr = bpr_schedule(1.2, alt)
        print(f"  Altitude = {alt:>6.0f} m   BPR = {bpr:.4f}")

    # Validation checks
    print(f"\n--- Validation Checks ---")
    bpr_sl   = bpr_schedule(0.0, 0)
    bpr_t    = bpr_schedule(0.9, 0)
    bpr_sup  = bpr_schedule(1.8, 0)
    bpr_alt  = bpr_schedule(1.2, 10000)
    bpr_low  = bpr_schedule(1.2, 0)

    checks = [
        (bpr_sl == BPR_MIN,              f"Sea level static BPR = {BPR_MIN} (min)"),
        (bpr_t  == BPR_CRUISE,           f"Mach 0.9 BPR = {BPR_CRUISE} (cruise peak)"),
        (bpr_sup == BPR_SUPERSONIC,      f"Mach 1.8 BPR = {BPR_SUPERSONIC} (supersonic)"),
        (bpr_alt > bpr_low,              f"BPR at altitude > BPR at sea level for same Mach"),
        (BPR_MIN <= bpr_sl <= 1.5,       f"BPR within valid bounds [0.20, 1.50]"),
    ]

    for passed, description in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {description}")

    print("=" * 60)


if __name__ == "__main__":
    validate_bpr_schedule()