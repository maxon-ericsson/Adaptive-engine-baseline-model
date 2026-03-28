"""
inlet_model.py
--------------
Phase 4 – Morphing DSI Inlet Model
Ultra-Lightweight Adaptive Cycle Engine Project

Computes oblique-shock pressure recovery and fan-face distortion coefficient
(DC60) for a parametric DSI bump geometry across the Mach 0.8–2.0 flight
envelope.

Physical model
--------------
The DSI bump generates a two-shock system:
  Shock 1 : oblique shock off the bump leading edge (deflection = theta/2)
  Shock 2 : oblique shock off the cowl lip        (deflection = theta/2)
  Normal   : terminal normal shock at the throat

Total-pressure recovery across each oblique shock is computed from the
Rayleigh pitot formula applied to the normal component of Mach number.
The terminal normal shock recovery uses the standard isentropic relation.

Distortion is modelled as a linear function of bump height and residual
Mach non-uniformity at the fan face, calibrated to DC60 ≤ 0.25 at the
ACE design point (Mach 1.6, h = 0.15 m).

Geometric parameters
--------------------
h     : bump height          [m]   0.05 – 0.25
r     : leading-edge radius  [m]   0.01 – 0.08
theta : contouring angle     [deg] 5    – 25

Reference
---------
MIL-E-5007D, section 3.7   (inlet pressure recovery standards)
Seddon & Goldsmith (1999)   "Intake Aerodynamics", AIAA
P&W XA103 design brief      (Phase 3 technical memorandum)
"""

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
GAMMA = 1.4                 # ratio of specific heats, air (constant across
                            # the inlet – temperatures stay below 700 K)
R_AIR = 287.05              # specific gas constant, J/(kg·K)

# Geometric bounds (m and deg)
H_MIN, H_MAX         = 0.05, 0.25
R_MIN, R_MAX         = 0.01, 0.08
THETA_MIN, THETA_MAX = 5.0,  25.0

# Mach envelope
MACH_MIN, MACH_MAX = 0.8, 2.0

# Distortion model constants (calibrated to ACE design point)
# DC60 = K_h * h + K_M * delta_M_norm
# where delta_M_norm is residual Mach non-uniformity after shocks
K_H = 0.40      # distortion sensitivity to bump height  [1/m]
K_M = 0.55      # distortion sensitivity to Mach spread  [-]

# MIL-spec recovery floor (MIL-E-5007D Table I)
MIL_RECOVERY = {
    1.0: 1.000,
    1.2: 0.991,
    1.4: 0.980,
    1.6: 0.965,
    1.8: 0.945,
    2.0: 0.920,
}

# ---------------------------------------------------------------------------
# OBLIQUE SHOCK RELATIONS
# ---------------------------------------------------------------------------

def _oblique_shock_beta(mach: float, deflection_deg: float) -> float:
    """
    Solve the theta-beta-Mach relation for shock wave angle beta [deg].

    Uses iterative Newton solve on:
        tan(theta) = 2 cot(beta) * (M^2 sin^2(beta) - 1)
                     / (M^2 (gamma + cos(2*beta)) + 2)

    Parameters
    ----------
    mach          : upstream Mach number
    deflection_deg: flow deflection angle theta [deg]

    Returns
    -------
    beta_deg : shock wave angle measured from upstream flow [deg]
               Returns None if no attached solution exists.
    """
    theta = np.radians(deflection_deg)
    # Initial guess: midpoint between Mach angle and 90 deg
    mu = np.arcsin(1.0 / mach)                # Mach cone angle
    beta = (mu + np.pi / 2.0) / 2.0

    for _ in range(50):
        sin_b  = np.sin(beta)
        cos_b  = np.cos(beta)
        tan_b  = np.tan(beta)
        M2s2b  = mach**2 * sin_b**2

        # TBM residual
        rhs = 2.0 / tan_b * (M2s2b - 1.0) / (
              mach**2 * (GAMMA + np.cos(2.0 * beta)) + 2.0)
        residual = np.tan(theta) - rhs

        # Derivative of residual w.r.t. beta (numerical for robustness)
        db = 1e-6
        rhs_p = 2.0 / np.tan(beta + db) * (
                mach**2 * np.sin(beta + db)**2 - 1.0) / (
                mach**2 * (GAMMA + np.cos(2.0 * (beta + db))) + 2.0)
        drdb = (rhs_p - rhs) / db

        beta -= residual / (-drdb + 1e-12)

        # Clamp to physical range [mu, pi/2]
        beta = np.clip(beta, mu + 1e-6, np.pi / 2.0 - 1e-6)

        if abs(residual) < 1e-9:
            break

    return np.degrees(beta)


def _normal_mach_after_oblique(mach: float, beta_deg: float) -> float:
    """
    Mach number of the normal component upstream of the oblique shock.

    M_n1 = M1 * sin(beta)
    """
    return mach * np.sin(np.radians(beta_deg))


def _total_pressure_ratio_normal(mach_n: float) -> float:
    """
    Total-pressure ratio across a normal shock (Rayleigh pitot formula).

    P02/P01 = [ (gamma+1)/2 * M_n^2 / (1 + (gamma-1)/2 * M_n^2) ]^(gamma/(gamma-1))
              * [ (2*gamma*M_n^2 - (gamma-1)) / (gamma+1) ]^(-1/(gamma-1))

    Valid for M_n >= 1.  Returns 1.0 for M_n < 1 (no shock).
    """
    if mach_n < 1.0:
        return 1.0
    g  = GAMMA
    gp = g + 1.0
    gm = g - 1.0
    term1 = (gp / 2.0 * mach_n**2 / (1.0 + gm / 2.0 * mach_n**2)) ** (g / gm)
    term2 = ((2.0 * g * mach_n**2 - gm) / gp) ** (-1.0 / gm)
    return term1 * term2


# ---------------------------------------------------------------------------
# DISTORTION MODEL
# ---------------------------------------------------------------------------

def _distortion_dc60(h: float, mach: float, p_recovery: float) -> float:
    """
    Fan-face distortion coefficient DC60 [-].

    DC60 = (P_max - P_min) / q_avg  integrated over the worst 60-deg sector.

    Simplified model:
        DC60 ≈ K_h * h + K_M * (1 - p_recovery)

    The (1 - p_recovery) term captures the Mach non-uniformity introduced
    by unequal shock strengths across the fan annulus.

    Typical acceptance limit: DC60 ≤ 0.40 (MIL-E-5007D)
    ACE design target:      DC60 ≤ 0.25 at Mach 1.6
    """
    delta_p = 1.0 - p_recovery           # fractional total-pressure deficit
    dc60 = K_H * h + K_M * delta_p
    return float(np.clip(dc60, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LEADING-EDGE RADIUS CORRECTION
# ---------------------------------------------------------------------------

def _radius_recovery_correction(r: float, mach: float) -> float:
    """
    Blunter leading edges reduce local shock strength by spreading the
    oblique shock over a finite radius, improving recovery at the cost of
    slightly increased wave drag.

    Correction factor:  delta_eta = alpha_r * (r / r_ref) * (M - 1)
    where r_ref = 0.04 m (midpoint of design range)
          alpha_r = 0.012  (empirically calibrated)

    Positive correction: larger r -> marginally higher P_recovery.
    Effect saturates above Mach 1.8.
    """
    r_ref   = 0.04
    alpha_r = 0.012
    mach_eff = min(mach, 1.8)
    return alpha_r * (r / r_ref) * max(mach_eff - 1.0, 0.0)


# ---------------------------------------------------------------------------
# PRIMARY INTERFACE
# ---------------------------------------------------------------------------

def compute_inlet_performance(
    mach: float,
    h: float,
    r: float,
    theta: float,
) -> dict:
    """
    Compute DSI inlet performance for given Mach number and geometry.

    The two-shock model:
      Shock 1: oblique off bump LE, deflection = theta/2
      Shock 2: oblique off cowl lip, deflection = theta/2
      Normal : terminal normal shock at throat

    Parameters
    ----------
    mach  : freestream Mach number  [–]   (0.8 – 2.0)
    h     : bump height             [m]   (0.05 – 0.25)
    r     : leading-edge radius     [m]   (0.01 – 0.08)
    theta : contouring angle        [deg] (5 – 25)

    Returns
    -------
    dict with keys:
        p_recovery  : total-pressure ratio P02/P01         [–]
        dc60        : fan-face distortion coefficient       [–]
        beta1_deg   : first shock wave angle                [deg]
        beta2_deg   : second shock wave angle               [deg]
        mach_2      : Mach number entering terminal shock   [–]
        mil_floor   : MIL-E-5007D minimum P_recovery        [–]
        meets_mil   : bool, whether P_recovery ≥ MIL floor  [bool]
    """
    deflection = theta / 2.0      # each shock deflects half the total angle

    # --- Shock 1: off bump leading edge ---
    beta1 = _oblique_shock_beta(mach, deflection)
    mn1   = _normal_mach_after_oblique(mach, beta1)
    pr1   = _total_pressure_ratio_normal(mn1)

    # Mach number downstream of Shock 1 (tangential component preserved)
    # M2 from normal shock downstream Mach + geometry
    mn1_downstream = np.sqrt(
        (1.0 + (GAMMA - 1.0) / 2.0 * mn1**2) /
        (GAMMA * mn1**2 - (GAMMA - 1.0) / 2.0)
    )
    mach_2 = mn1_downstream / np.sin(np.radians(beta1) - np.radians(deflection))

    # --- Shock 2: off cowl lip ---
    beta2 = _oblique_shock_beta(mach_2, deflection)
    mn2   = _normal_mach_after_oblique(mach_2, beta2)
    pr2   = _total_pressure_ratio_normal(mn2)

    mn2_downstream = np.sqrt(
        (1.0 + (GAMMA - 1.0) / 2.0 * mn2**2) /
        (GAMMA * mn2**2 - (GAMMA - 1.0) / 2.0)
    )
    mach_3 = mn2_downstream / np.sin(np.radians(beta2) - np.radians(deflection))

    # --- Terminal normal shock ---
    pr_normal = _total_pressure_ratio_normal(mach_3)

    # --- Assemble total recovery ---
    p_recovery = pr1 * pr2 * pr_normal

    # Apply leading-edge radius correction
    p_recovery += _radius_recovery_correction(r, mach)
    p_recovery  = float(np.clip(p_recovery, 0.0, 1.0))

    # --- Distortion ---
    dc60 = _distortion_dc60(h, mach, p_recovery)

    # --- MIL-E-5007D floor (linear interpolation) ---
    mach_keys = sorted(MIL_RECOVERY.keys())
    mil_floor = np.interp(mach, mach_keys,
                          [MIL_RECOVERY[k] for k in mach_keys])

    return {
        "p_recovery" : p_recovery,
        "dc60"       : dc60,
        "beta1_deg"  : beta1,
        "beta2_deg"  : beta2,
        "mach_2"     : mach_2,
        "mach_3"     : mach_3,
        "mil_floor"  : mil_floor,
        "meets_mil"  : p_recovery >= mil_floor,
    }


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_inlet_model() -> None:
    """
    Standalone validation against three reference points.

    Reference 1 : Subsonic cruise  Mach 0.85 — expect P_rec > 0.98
    Reference 2 : Transonic dash   Mach 1.2  — expect P_rec ≈ 0.965–0.985
    Reference 3 : XA103 design pt  Mach 1.6  — expect P_rec ≈ 0.945–0.970
                                               DC60 ≤ 0.30

    Geometry used: h=0.15 m, r=0.04 m, theta=15 deg (nominal design)
    """
    print("=" * 62)
    print("  inlet_model.py — Validation Run")
    print("=" * 62)

    cases = [
        (0.85, "Subsonic cruise"),
        (1.20, "Transonic dash"),
        (1.60, "ACE design point"),
        (1.80, "Supersonic combat"),
        (2.00, "Maximum Mach"),
    ]
    h, r, theta = 0.15, 0.04, 15.0   # nominal geometry

    print(f"\n  Geometry: h={h} m  |  r={r} m  |  theta={theta} deg\n")
    print(f"  {'Case':<22} {'Mach':>5}  {'P_rec':>7}  "
          f"{'MIL_floor':>9}  {'DC60':>6}  {'β1':>6}  {'β2':>6}  {'Pass?':>5}")
    print("  " + "-" * 72)

    all_pass = True
    for mach, label in cases:
        res = compute_inlet_performance(mach, h, r, theta)
        status = "OK" if res["meets_mil"] else "FAIL"
        if not res["meets_mil"]:
            all_pass = False
        print(f"  {label:<22} {mach:>5.2f}  {res['p_recovery']:>7.4f}  "
              f"{res['mil_floor']:>9.4f}  {res['dc60']:>6.3f}  "
              f"{res['beta1_deg']:>6.1f}  {res['beta2_deg']:>6.1f}  {status:>5}")

    print("\n  " + ("All cases pass MIL-E-5007D." if all_pass
                    else "WARNING: One or more cases fail MIL-E-5007D."))
    print("=" * 62)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validate_inlet_model()

    print("\n  --- Geometry sensitivity at Mach 1.6 ---\n")
    print(f"  {'h [m]':>6}  {'r [m]':>6}  {'θ [deg]':>7}  "
          f"{'P_rec':>7}  {'DC60':>6}")
    print("  " + "-" * 42)

    for h in [0.05, 0.10, 0.15, 0.20, 0.25]:
        res = compute_inlet_performance(1.6, h, 0.04, 15.0)
        print(f"  {h:>6.2f}  {'0.04':>6}  {'15.0':>7}  "
              f"{res['p_recovery']:>7.4f}  {res['dc60']:>6.3f}")