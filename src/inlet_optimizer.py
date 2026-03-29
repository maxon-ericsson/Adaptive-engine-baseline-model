"""
inlet_optimizer.py
------------------
Phase 4 – DSI Inlet Geometry Optimizer
Ultra-Lightweight Adaptive Cycle Engine Project
Author: Maxon Ericsson

Finds the bump geometry (h, r, theta) that minimizes a mission-weighted
TSFC across Mach 1.4 / 1.6 / 1.8 at the ACE design altitude, with
pressure recovery used as a tiebreaker when TSFC values are equal.

Objective
---------
  J = 0.25 × TSFC(M1.4) + 0.50 × TSFC(M1.6) + 0.25 × TSFC(M1.8)
      − α × P_recovery(M1.6)

  where α = 1e-6  (tiebreaker weight, never dominates TSFC term)

Optimizer
---------
  Algorithm : L-BFGS-B (scipy.optimize.minimize)
  Variables : h [m], r [m], theta [deg]
  Constraints:
      DC60     ≤ 0.40  at all three Mach points
      P_rec    ≥ MIL floor at all three Mach points
      thrust_N > 0     at all three Mach points

Design altitude : 10 000 m ISA
"""

import numpy as np
from scipy.optimize import minimize
import warnings

from engine_model import EngineModel
from inlet_model  import compute_inlet_performance

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DESIGN_ALT      = 10_000.0
T_DESIGN        = 288.15 - 0.0065 * DESIGN_ALT     # 223.15 K
P_DESIGN        = 101_325.0 * (T_DESIGN / 288.15) ** 5.2561

# Mission Mach points and weights
MACH_POINTS     = [1.40, 1.60, 1.80]
MACH_WEIGHTS    = [0.25, 0.50, 0.25]

# Tiebreaker weight on p_recovery at design Mach
ALPHA_PREC      = 1e-6

# Engine parameters
MASS_FLOW       = 50.0
COMPRESSOR_PR   = 18.0
COMPRESSOR_EFF  = 0.88
TURBINE_EFF     = 0.90
TIT             = 1550.0

# Geometry bounds
H_BOUNDS        = (0.05, 0.25)
R_BOUNDS        = (0.01, 0.08)
THETA_BOUNDS    = (5.0,  25.0)

# Constraint limits
DC60_LIMIT      = 0.40
PENALTY         = 1.0e6


# ---------------------------------------------------------------------------
# SINGLE-MACH EVALUATION
# ---------------------------------------------------------------------------

def evaluate_point(mach: float, h: float, r: float, theta: float) -> dict:
    """
    Run inlet + engine at one Mach number and return key metrics.
    Returns dict with tsfc, p_recovery, dc60, meets_mil, feasible.
    """
    inlet = compute_inlet_performance(mach=mach, h=h, r=r, theta=theta)

    p_recovery = inlet["p_recovery"]
    dc60       = inlet["dc60"]
    mil_floor  = inlet["mil_floor"]
    meets_mil  = inlet["meets_mil"]

    # Check feasibility before running engine
    if dc60 > DC60_LIMIT or p_recovery < mil_floor:
        return {
            "tsfc": float("inf"), "p_recovery": p_recovery,
            "dc60": dc60, "meets_mil": meets_mil, "feasible": False,
            "thrust_N": 0.0,
        }

    engine = EngineModel(
        mass_flow=MASS_FLOW, compressor_PR=COMPRESSOR_PR,
        compressor_eff=COMPRESSOR_EFF, turbine_eff=TURBINE_EFF,
        tit=TIT, mach=mach, inlet_h=h, inlet_r=r, inlet_theta=theta,
    )
    out = engine.run(T_DESIGN, P_DESIGN)

    if out["thrust_N"] <= 0:
        return {
            "tsfc": float("inf"), "p_recovery": p_recovery,
            "dc60": dc60, "meets_mil": meets_mil, "feasible": False,
            "thrust_N": out["thrust_N"],
        }

    return {
        "tsfc":       out["tsfc"],
        "p_recovery": p_recovery,
        "dc60":       dc60,
        "meets_mil":  meets_mil,
        "feasible":   True,
        "thrust_N":   out["thrust_N"],
    }


# ---------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ---------------------------------------------------------------------------

def objective(x: np.ndarray) -> float:
    """
    Mission-weighted TSFC with p_recovery tiebreaker.

    J = Σ w_i × TSFC(M_i) − α × P_recovery(M1.6)

    Infeasible points receive a penalty proportional to violation magnitude.
    """
    h, r, theta = float(x[0]), float(x[1]), float(x[2])

    weighted_tsfc = 0.0
    p_rec_design  = 0.0
    total_penalty = 0.0

    for mach, weight in zip(MACH_POINTS, MACH_WEIGHTS):
        res = evaluate_point(mach, h, r, theta)

        if not res["feasible"]:
            # Accumulate penalty from constraint violations
            inlet = compute_inlet_performance(mach, h, r, theta)
            dc60_viol  = max(0.0, res["dc60"]       - DC60_LIMIT)
            prec_viol  = max(0.0, inlet["mil_floor"] - res["p_recovery"])
            total_penalty += weight * PENALTY * (1.0 + 1e4 * (dc60_viol + prec_viol))
        else:
            weighted_tsfc += weight * res["tsfc"]

        if mach == 1.60:
            p_rec_design = res["p_recovery"]

    if total_penalty > 0:
        return total_penalty

    return weighted_tsfc - ALPHA_PREC * p_rec_design


# ---------------------------------------------------------------------------
# MULTI-START OPTIMIZER
# ---------------------------------------------------------------------------

def run_optimizer(n_starts: int = 12, verbose: bool = True) -> object:
    """
    Multi-start L-BFGS-B optimization across the geometry space.
    Returns the scipy OptimizeResult with the lowest objective value.
    """
    bounds = [H_BOUNDS, R_BOUNDS, THETA_BOUNDS]

    if verbose:
        print("=" * 66)
        print("  inlet_optimizer.py — Phase 4 Multi-Point DSI Optimizer")
        print(f"  Mach points : {MACH_POINTS}  |  Weights : {MACH_WEIGHTS}")
        print(f"  Altitude    : {DESIGN_ALT:.0f} m  "
              f"|  T = {T_DESIGN:.2f} K  |  P = {P_DESIGN:.1f} Pa")
        print(f"  {n_starts} random starts  |  L-BFGS-B")
        print("=" * 66)
        print(f"\n  {'Start':>5}  {'h [m]':>6}  {'r [m]':>6}  {'θ [°]':>6}  "
              f"{'J (weighted)':>14}  {'P_rec@1.6':>10}  {'DC60@1.6':>9}")
        print("  " + "-" * 64)

    rng         = np.random.default_rng(seed=42)
    best_val    = float("inf")
    best_result = None

    for i in range(n_starts):
        x0 = np.array([
            rng.uniform(*H_BOUNDS),
            rng.uniform(*R_BOUNDS),
            rng.uniform(*THETA_BOUNDS),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
            )

        if res.fun < best_val:
            best_val    = res.fun
            best_result = res

        if verbose:
            h, r, theta = res.x
            r16 = evaluate_point(1.60, h, r, theta)
            j_str = f"{res.fun:.10f}" if res.fun < PENALTY else "  INFEASIBLE"
            print(f"  {i+1:>5}  {h:>6.3f}  {r:>6.3f}  {theta:>6.1f}  "
                  f"{j_str:>14}  {r16['p_recovery']:>10.4f}  "
                  f"{r16['dc60']:>9.4f}")

    return best_result


# ---------------------------------------------------------------------------
# RESULT REPORTING
# ---------------------------------------------------------------------------

def report_optimum(result) -> dict:
    """
    Full performance breakdown at optimal geometry vs nominal baseline.
    """
    h, r, theta = float(result.x[0]), float(result.x[1]), float(result.x[2])

    print("\n" + "=" * 66)
    print("  OPTIMAL GEOMETRY — Full Performance Breakdown")
    print("=" * 66)
    print(f"\n  h     = {h:.4f} m")
    print(f"  r     = {r:.4f} m")
    print(f"  theta = {theta:.4f} deg\n")

    # Nominal baseline
    nom = {"h": 0.15, "r": 0.04, "theta": 15.0}

    print(f"  {'Mach':>5}  {'Metric':>12}  {'Optimal':>12}  "
          f"{'Nominal':>12}  {'Delta':>10}")
    print("  " + "-" * 56)

    output = {}
    for mach in MACH_POINTS:
        opt_res = evaluate_point(mach, h, r, theta)
        nom_res = evaluate_point(mach, nom["h"], nom["r"], nom["theta"])

        for metric in ["tsfc", "p_recovery", "dc60"]:
            ov = opt_res[metric]
            nv = nom_res[metric]
            if isinstance(ov, float) and ov < 1:
                delta = (ov - nv) / nv * 100 if nv != 0 else 0.0
                delta_str = f"{delta:+.4f}%"
                print(f"  {mach:>5.1f}  {metric:>12}  "
                      f"{ov:>12.6f}  {nv:>12.6f}  {delta_str:>10}")

        output[f"M{mach:.1f}_tsfc"]       = round(opt_res["tsfc"],       8)
        output[f"M{mach:.1f}_p_recovery"] = round(opt_res["p_recovery"], 6)
        output[f"M{mach:.1f}_dc60"]       = round(opt_res["dc60"],       6)
        output[f"M{mach:.1f}_meets_mil"]  = opt_res["meets_mil"]

    # Weighted TSFC summary
    w_opt = sum(w * evaluate_point(m, h, r, theta)["tsfc"]
                for m, w in zip(MACH_POINTS, MACH_WEIGHTS))
    w_nom = sum(w * evaluate_point(m, nom["h"], nom["r"], nom["theta"])["tsfc"]
                for m, w in zip(MACH_POINTS, MACH_WEIGHTS))
    improvement = (w_nom - w_opt) / w_nom * 100.0

    print(f"\n  Weighted TSFC (optimal) = {w_opt:.10f}")
    print(f"  Weighted TSFC (nominal) = {w_nom:.10f}")
    print(f"  Mission TSFC improvement = {improvement:+.4f}%")
    print("=" * 66)

    output.update({
        "h_opt": round(h, 4), "r_opt": round(r, 4),
        "theta_opt": round(theta, 4),
        "weighted_tsfc_opt": round(w_opt, 10),
        "weighted_tsfc_nom": round(w_nom, 10),
        "mission_tsfc_improvement_pct": round(improvement, 4),
    })
    return output


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result  = run_optimizer(n_starts=12, verbose=True)
    optimum = report_optimum(result)