"""
bpr_schedule.py
Variable Bypass Ratio Schedule for Adaptive Cycle Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine — Phase 3

Physical Model:
- Bypass ratio varies as a function of Mach number, altitude,
  and operating mode, mimicking the adaptive valve scheduling
  of an NGAD-class three-stream engine (XA103 architecture).

Operating Modes:
- Combat mode (combat_mode=True):
    Third stream fully closed. All airflow redirected to core
    and primary bypass for maximum specific thrust.
    Targets XA103 high-thrust mode (+20% thrust vs fixed cycle).

- Adaptive mode (combat_mode=False):
    Third stream opens progressively with Mach number.
    Targets XA103 high-efficiency mode (+25% fuel efficiency).

BPR Schedule (adaptive mode):
- Mach 0.0 to 0.4  : BPR = BPR_MIN  (takeoff / low-speed combat)
- Mach 0.4 to 0.9  : BPR ramps up to BPR_CRUISE (transonic accel)
- Mach 0.9 to 1.8  : BPR ramps down to BPR_SUPERSONIC (ram compression)
- Above Mach 1.8   : BPR clamped at BPR_SUPERSONIC
- Altitude effect  : +0.008 BPR per 1000 m gain
"""

import numpy as np


# ============================================================
# BPR SCHEDULE CONSTANTS
# ============================================================

# Mach breakpoints
MACH_TAKEOFF    = 0.4    # Below this: max thrust mode
MACH_TRANSONIC  = 0.9    # Cruise efficiency peak
MACH_SUPERSONIC = 1.8    # Schedule clamped above this

# BPR values at key operating points
BPR_MIN         = 0.10   # Combat floor — nearly all air to core
BPR_CRUISE      = 1.20   # Cruise peak — maximum bypass open
BPR_SUPERSONIC  = 0.30   # Supersonic — ram compression dominant

# Altitude correction
ALTITUDE_BPR_COEFFICIENT = 0.008   # BPR increase per 1000 m altitude


# ============================================================
# CORE SCHEDULE FUNCTION
# ============================================================

def bpr_schedule(
    mach: float,
    altitude_m: float = 0.0,
    combat_mode: bool = False
) -> float:
    """
    Compute adaptive bypass ratio as a function of Mach, altitude,
    and operating mode.

    Physical Basis:
    - Combat mode: third stream valve closes completely, all flow
      directed to core for maximum specific thrust. Mimics the
      XA103 high-thrust valve schedule.
    - Adaptive mode: third stream opens progressively from takeoff
      through transonic cruise, then closes again at supersonic
      speeds where ram compression handles inlet compression.
    - Altitude correction: thinner air at altitude allows a slightly
      higher bypass ratio while maintaining core operability.

    Args:
        mach:        Freestream Mach number [-]
        altitude_m:  Geometric altitude [m], default sea level
        combat_mode: If True, third stream closes for max thrust [-]

    Returns:
        float: Bypass ratio BPR = m_bypass / m_core [-]
    """

    # --------------------------------------------------------
    # COMBAT MODE — third stream fully closed
    # --------------------------------------------------------
    if combat_mode:
        return BPR_MIN

    # --------------------------------------------------------
    # STEP 1: Base BPR from Mach schedule
    # --------------------------------------------------------
    if mach <= MACH_TAKEOFF:
        bpr_base = BPR_MIN

    elif mach <= MACH_TRANSONIC:
        t = (mach - MACH_TAKEOFF) / (MACH_TRANSONIC - MACH_TAKEOFF)
        bpr_base = BPR_MIN + t * (BPR_CRUISE - BPR_MIN)

    elif mach <= MACH_SUPERSONIC:
        t = (mach - MACH_TRANSONIC) / (MACH_SUPERSONIC - MACH_TRANSONIC)
        bpr_base = BPR_CRUISE + t * (BPR_SUPERSONIC - BPR_CRUISE)

    else:
        bpr_base = BPR_SUPERSONIC

    # --------------------------------------------------------
    # STEP 2: Altitude correction
    # --------------------------------------------------------
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

def get_operating_mode(
    mach: float,
    altitude_m: float = 0.0,
    combat_mode: bool = False
) -> str:
    """
    Return a human-readable label for the current operating mode.

    Args:
        mach:        Freestream Mach number [-]
        altitude_m:  Geometric altitude [m]
        combat_mode: Whether engine is in combat mode [-]

    Returns:
        str: Operating mode label with BPR
    """
    bpr = bpr_schedule(mach, altitude_m, combat_mode)

    if combat_mode:
        return f"Combat / Max Thrust  (M={mach:.2f}, BPR={bpr:.3f})"
    elif mach <= MACH_TAKEOFF:
        return f"Takeoff              (M={mach:.2f}, BPR={bpr:.3f})"
    elif mach <= MACH_TRANSONIC:
        return f"Transonic Accel      (M={mach:.2f}, BPR={bpr:.3f})"
    elif mach <= MACH_SUPERSONIC:
        return f"Supersonic Cruise    (M={mach:.2f}, BPR={bpr:.3f})"
    else:
        return f"Max Speed            (M={mach:.2f}, BPR={bpr:.3f})"


# ============================================================
# UTILITY — MACH SWEEP
# ============================================================

def bpr_mach_sweep(
    mach_range: np.ndarray,
    altitude_m: float = 0.0,
    combat_mode: bool = False
) -> np.ndarray:
    """
    Compute BPR across an array of Mach numbers at fixed altitude.

    Args:
        mach_range:  Array of Mach numbers [-]
        altitude_m:  Fixed altitude [m]
        combat_mode: Whether to use combat mode [-]

    Returns:
        np.ndarray: BPR values matching mach_range
    """
    return np.array([bpr_schedule(m, altitude_m, combat_mode) for m in mach_range])


# ============================================================
# VALIDATION & TESTING
# ============================================================

def validate_bpr_schedule() -> None:
    """
    Validate BPR schedule across representative operating points
    for both adaptive and combat modes.
    """
    print("=" * 65)
    print("BPR SCHEDULE VALIDATION — PHASE 3 ADAPTIVE CYCLE")
    print("=" * 65)

    # Adaptive mode
    print(f"\n--- Adaptive Mode ---")
    print(f"{'Condition':<25s} {'Mach':>6s} {'Alt (m)':>8s} {'BPR':>8s}  Mode")
    print("-" * 65)

    adaptive_points = [
        (0.0,  0,     "Static / Ground"),
        (0.3,  0,     "Low Subsonic"),
        (0.4,  0,     "Takeoff limit"),
        (0.65, 0,     "Mid Transonic"),
        (0.9,  0,     "Transonic peak"),
        (1.2,  5000,  "Low Supersonic"),
        (1.5,  10000, "Mid Supersonic"),
        (1.8,  12000, "Max Supersonic"),
    ]

    for mach, alt, label in adaptive_points:
        bpr  = bpr_schedule(mach, alt, combat_mode=False)
        mode = get_operating_mode(mach, alt, combat_mode=False)
        print(f"{label:<25s} {mach:>6.2f} {alt:>8.0f} {bpr:>8.4f}  {mode}")

    # Combat mode
    print(f"\n--- Combat Mode (third stream closed) ---")
    print(f"{'Condition':<25s} {'Mach':>6s} {'Alt (m)':>8s} {'BPR':>8s}  Mode")
    print("-" * 65)

    combat_points = [
        (0.3,  0,    "Low Subsonic"),
        (0.9,  5000, "Transonic"),
        (1.5,  8000, "Supersonic"),
    ]

    for mach, alt, label in combat_points:
        bpr  = bpr_schedule(mach, alt, combat_mode=True)
        mode = get_operating_mode(mach, alt, combat_mode=True)
        print(f"{label:<25s} {mach:>6.2f} {alt:>8.0f} {bpr:>8.4f}  {mode}")

    # Altitude effect
    print(f"\n--- Altitude Effect at Mach 1.2 (Adaptive) ---")
    for alt in [0, 3000, 6000, 9000, 12000]:
        bpr = bpr_schedule(1.2, alt)
        print(f"  Altitude = {alt:>6.0f} m   BPR = {bpr:.4f}")

    # Validation checks
    print(f"\n--- Validation Checks ---")
    bpr_static  = bpr_schedule(0.0, 0)
    bpr_cruise  = bpr_schedule(0.9, 0)
    bpr_sup     = bpr_schedule(1.8, 0)
    bpr_alt     = bpr_schedule(1.2, 10000)
    bpr_low_alt = bpr_schedule(1.2, 0)
    bpr_combat  = bpr_schedule(0.9, 0, combat_mode=True)
    bpr_adapt   = bpr_schedule(0.9, 0, combat_mode=False)

    checks = [
        (bpr_static == BPR_MIN,
         f"Static BPR = {BPR_MIN} (combat floor)"),
        (bpr_cruise == BPR_CRUISE,
         f"Mach 0.9 BPR = {BPR_CRUISE} (cruise peak)"),
        (bpr_sup == BPR_SUPERSONIC,
         f"Mach 1.8 BPR = {BPR_SUPERSONIC} (supersonic)"),
        (bpr_alt > bpr_low_alt,
         f"BPR at altitude > BPR at sea level for same Mach"),
        (bpr_combat < bpr_adapt,
         f"Combat BPR ({bpr_combat}) < Adaptive BPR ({bpr_adapt}) at Mach 0.9"),
        (BPR_MIN <= bpr_static <= 1.5,
         f"BPR within valid bounds [{BPR_MIN}, 1.50]"),
    ]

    for passed, description in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {description}")

    print("=" * 65)


if __name__ == "__main__":
    validate_bpr_schedule()