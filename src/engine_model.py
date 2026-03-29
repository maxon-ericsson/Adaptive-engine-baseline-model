"""
engine_model.py
---------------
0-D Adaptive Cycle Engine Model (Phase 4 — Inlet-Coupled, Variable-f)
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine

Phase 4 changes:
  - inlet_model.py wired in: P_fan_face = P_ambient × p_recovery
  - Variable fuel-air ratio: f is solved to hit target turbine inlet
    temperature (TIT), so inlet pressure loss propagates into fuel burn
    and TSFC. Lower p_recovery → lower T2 → higher f → higher TSFC.
"""

# ============================================================
# IMPORTS
# ============================================================

import math
from components.compressor import compressor, compute_compressor_work
from components.combustor  import combustor
from components.turbine    import turbine
from components.nozzle     import nozzle, compute_thrust_simple, compute_specific_impulse
from inlet_model           import compute_inlet_performance

from typing import Dict

# ============================================================
# CONSTANTS
# ============================================================

g0        = 9.81        # gravitational acceleration [m/s²]
GAMMA_AIR = 1.4         # ratio of specific heats [-]
R_AIR     = 287.05      # specific gas constant [J/(kg·K)]
CP_AIR    = 1005.0      # specific heat at constant pressure [J/(kg·K)]
LHV_JET_A = 43.0e6     # lower heating value of Jet-A [J/kg]
ETA_COMB  = 0.99        # combustion efficiency [-]

# ============================================================
# ENGINE MODEL CLASS
# ============================================================

class EngineModel:
    """
    0-D coupled inlet-engine model with variable fuel-air ratio.

    Thermodynamic cycle:
        Inlet → Compressor → Combustor → Turbine → Nozzle

    Inlet coupling:
        P_fan_face = P_ambient × p_recovery(Mach, h, r, theta)

    Variable-f coupling:
        f = Cp × (TIT − T2) / (η_comb × LHV)
        Fuel burn adjusts to maintain constant turbine inlet temperature.
        Inlet pressure loss lowers T2, requiring more fuel, raising TSFC.
    """

    def __init__(
        self,
        mass_flow:      float = 50.0,     # kg/s core air flow
        compressor_PR:  float = 18.0,     # overall pressure ratio
        compressor_eff: float = 0.88,     # isentropic efficiency
        turbine_eff:    float = 0.90,     # turbine efficiency
        tit:            float = 1550.0,   # target turbine inlet temp [K]
        mach:           float = 0.0,      # freestream Mach number
        inlet_h:        float = 0.15,     # bump height         [m]
        inlet_r:        float = 0.04,     # leading-edge radius [m]
        inlet_theta:    float = 15.0,     # contouring angle    [deg]
    ) -> None:

        self.mass_flow      = mass_flow
        self.compressor_PR  = compressor_PR
        self.compressor_eff = compressor_eff
        self.turbine_eff    = turbine_eff
        self.tit            = tit
        self.mach           = mach
        self.inlet_h        = inlet_h
        self.inlet_r        = inlet_r
        self.inlet_theta    = inlet_theta

    # ============================================================
    # MAIN ENGINE RUN METHOD
    # ============================================================

    def run(self, T_ambient: float, P_ambient: float) -> Dict[str, float]:
        """
        Run the coupled inlet-engine cycle.

        Parameters
        ----------
        T_ambient : freestream static temperature [K]
        P_ambient : freestream static pressure    [Pa]

        Returns
        -------
        dict of station states, thrust, Isp, fuel flow, TSFC,
        and inlet performance metrics.
        """

        results: Dict[str, float] = {}

        # --------------------------------------------------------
        # 0. INLET
        # --------------------------------------------------------
        inlet = compute_inlet_performance(
            mach  = self.mach,
            h     = self.inlet_h,
            r     = self.inlet_r,
            theta = self.inlet_theta,
        )

        p_recovery = inlet["p_recovery"]
        dc60       = inlet["dc60"]
        P_fan_face = P_ambient * p_recovery

        # --------------------------------------------------------
        # 1. COMPRESSOR
        # --------------------------------------------------------
        T2, P2 = compressor(
            T_in           = T_ambient,
            P_in           = P_fan_face,
            pressure_ratio = self.compressor_PR,
            efficiency     = self.compressor_eff,
        )

        Wc = compute_compressor_work(
            T_in      = T_ambient,
            T_out     = T2,
            mass_flow = 1.0,
        )

        # --------------------------------------------------------
        # 2. COMBUSTOR — variable f to hit TIT
        # --------------------------------------------------------
        # Energy balance: η_comb × f × LHV = Cp × (TIT − T2)
        # Clamp f to physical range [0.01, 0.04]
        f = CP_AIR * (self.tit - T2) / (ETA_COMB * LHV_JET_A)
        f = max(0.010, min(f, 0.040))

        T3, P3 = combustor(
            T_in               = T2,
            P_in               = P2,
            fuel_air_ratio     = f,
            combustion_efficiency = ETA_COMB,
        )

        # --------------------------------------------------------
        # 3. TURBINE
        # --------------------------------------------------------
        T4, P4 = turbine(
            T_in          = T3,
            P_in          = P3,
            work_required = Wc,
            efficiency    = self.turbine_eff,
        )

        # --------------------------------------------------------
        # 4. NOZZLE → THRUST
        # --------------------------------------------------------
        T5, P5, V5, M5 = nozzle(
            T_in      = T4,
            P_in      = P4,
            P_ambient = P_ambient,
        )

        # Ram velocity and net thrust
        V0           = self.mach * math.sqrt(GAMMA_AIR * R_AIR * T_ambient)
        gross_thrust = compute_thrust_simple(self.mass_flow, V5)
        net_thrust   = self.mass_flow * (V5 - V0)

        mdot_fuel = self.mass_flow * f
        Isp       = compute_specific_impulse(net_thrust, mdot_fuel)
        tsfc      = mdot_fuel / net_thrust if net_thrust > 0 else float("inf")

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------
        results.update({
            # Inlet
            "inlet_p_recovery" : p_recovery,
            "inlet_dc60"       : dc60,
            "inlet_meets_mil"  : inlet["meets_mil"],
            "P_fan_face"       : P_fan_face,

            # Cycle stations
            "T2": T2, "P2": P2,
            "T3": T3, "P3": P3,
            "T4": T4, "P4": P4,
            "T5": T5, "P5": P5,

            # Fuel
            "fuel_air_ratio"     : f,
            "fuel_flow_kg_s"     : mdot_fuel,

            # Performance
            "V_exit"             : V5,
            "V0_ram"             : V0,
            "M_exit"             : M5,
            "gross_thrust_N"     : gross_thrust,
            "thrust_N"           : net_thrust,
            "specific_impulse_s" : Isp,
            "tsfc"               : tsfc,
        })

        return results


# ============================================================
# STANDALONE VALIDATION
# ============================================================

if __name__ == "__main__":

    T_sl = 288.15
    P_sl = 101325.0

    print("\n" + "=" * 66)
    print("  engine_model.py — Phase 4 Variable-f Inlet-Coupled Validation")
    print("=" * 66)

    cases = [
        ("Baseline (Mach 0)",    0.00, 0.15, 15.0),
        ("Mach 0.85 subsonic",   0.85, 0.15, 15.0),
        ("Mach 1.40 h=0.05 θ=5", 1.40, 0.05,  5.0),
        ("Mach 1.40 h=0.05 θ=25",1.40, 0.05, 25.0),
        ("Mach 1.60 θ=10",       1.60, 0.15, 10.0),
        ("Mach 1.60 θ=5",        1.60, 0.15,  5.0),
    ]

    print(f"\n  {'Case':<26} {'P_rec':>6}  {'f':>6}  "
          f"{'Thrust [N]':>10}  {'TSFC':>10}  {'DC60':>6}")
    print("  " + "-" * 72)

    for label, mach, h, theta in cases:
        eng = EngineModel(mach=mach, inlet_h=h, inlet_theta=theta)
        out = eng.run(T_sl, P_sl)
        thrust_str = f"{out['thrust_N']:>10.1f}" if out['thrust_N'] > 0 else "     ram>gross"
        tsfc_str   = f"{out['tsfc']:>10.6f}"     if out['tsfc'] < 1   else "           inf"
        print(f"  {label:<26} {out['inlet_p_recovery']:>6.4f}  "
              f"{out['fuel_air_ratio']:>6.4f}  "
              f"{thrust_str}  {tsfc_str}  {out['inlet_dc60']:>6.3f}")

    print(f"\n  Expected: f varies across cases. Mach 1.6 θ=5 (lower P_rec)")
    print(f"  should show higher f and TSFC than θ=10.")
    print("=" * 66)