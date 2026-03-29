"""
engine_model.py
---------------
0-D Adaptive Cycle Engine Model (Phase 4 — Inlet-Coupled)
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine

Phase 4 change: inlet_model.py is now wired in. The compressor receives
P_ambient * p_recovery at the fan face rather than raw P_ambient. This
couples inlet total-pressure loss directly into the engine cycle, so
geometry changes to the DSI bump propagate through to thrust and TSFC.
"""

# ============================================================
# IMPORTS
# ============================================================

from components.compressor import compressor, compute_compressor_work
from components.combustor  import combustor
from components.turbine    import turbine
from components.nozzle     import nozzle, compute_thrust_simple, compute_specific_impulse
from inlet_model           import compute_inlet_performance

from typing import Dict


# ============================================================
# CONSTANTS
# ============================================================

g0 = 9.81       # gravitational acceleration [m/s²]


# ============================================================
# ENGINE MODEL CLASS
# ============================================================

class EngineModel:
    """
    0-D coupled inlet-engine model.

    Thermodynamic cycle:
        Inlet → Compressor → Combustor → Turbine → Nozzle

    Inlet coupling:
        P_fan_face = P_ambient × p_recovery(Mach, h, r, theta)
        The compressor operates on this reduced inlet pressure,
        propagating inlet losses into thrust and TSFC.

    DC60 distortion is carried in the results dict for use by
    the optimizer as a constraint (DC60 ≤ 0.40).
    """

    def __init__(
        self,
        mass_flow:      float = 50.0,    # kg/s core air flow
        compressor_PR:  float = 18.0,    # overall pressure ratio
        compressor_eff: float = 0.88,    # isentropic efficiency
        turbine_eff:    float = 0.90,    # turbine efficiency
        f:              float = 0.020,   # fuel–air ratio
        mach:           float = 0.0,     # freestream Mach number
        inlet_h:        float = 0.15,    # bump height         [m]
        inlet_r:        float = 0.04,    # leading-edge radius [m]
        inlet_theta:    float = 15.0,    # contouring angle    [deg]
    ) -> None:

        self.mass_flow      = mass_flow
        self.compressor_PR  = compressor_PR
        self.compressor_eff = compressor_eff
        self.turbine_eff    = turbine_eff
        self.f              = f
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
        dict of station states, thrust, Isp, fuel flow,
        and inlet performance metrics.
        """

        results: Dict[str, float] = {}

        # --------------------------------------------------------
        # 0. INLET — compute fan-face conditions
        # --------------------------------------------------------
        inlet = compute_inlet_performance(
            mach  = self.mach,
            h     = self.inlet_h,
            r     = self.inlet_r,
            theta = self.inlet_theta,
        )

        p_recovery  = inlet["p_recovery"]
        dc60        = inlet["dc60"]

        # Fan-face total pressure after inlet losses
        P_fan_face  = P_ambient * p_recovery

        # --------------------------------------------------------
        # 1. COMPRESSOR — operates on reduced fan-face pressure
        # --------------------------------------------------------
        T2, P2 = compressor(
            T_in=T_ambient,
            P_in=P_fan_face,
            pressure_ratio=self.compressor_PR,
            efficiency=self.compressor_eff
        )

        Wc = compute_compressor_work(
            T_in=T_ambient,
            T_out=T2,
            mass_flow=1.0
        )

        # --------------------------------------------------------
        # 2. COMBUSTOR
        # --------------------------------------------------------
        T3, P3 = combustor(
            T_in=T2,
            P_in=P2,
            fuel_air_ratio=self.f
        )

        # --------------------------------------------------------
        # 3. TURBINE
        # --------------------------------------------------------
        T4, P4 = turbine(
            T_in=T3,
            P_in=P3,
            work_required=Wc,
            efficiency=self.turbine_eff
        )
        
        # --------------------------------------------------------
        # 4. NOZZLE → THRUST
        # --------------------------------------------------------
        T5, P5, V5, M5 = nozzle(
            T_in=T4,
            P_in=P4,
            P_ambient=P_ambient
        )

        # Freestream velocity (ram drag term)
        import math
        V0 = self.mach * math.sqrt(1.4 * 287.05 * T_ambient)

        # Net thrust = gross thrust - ram drag
        # Note: p_recovery affects absolute pressures P2-P4 but not
        # temperatures in a fixed-f model. Full thrust coupling requires
        # variable-f targeting constant TIT — implemented in coupled_sweep.py.
        gross_thrust = compute_thrust_simple(self.mass_flow, V5)
        net_thrust   = self.mass_flow * (V5 - V0)

        mdot_fuel = self.mass_flow * self.f
        Isp       = compute_specific_impulse(net_thrust, mdot_fuel)
        tsfc      = mdot_fuel / net_thrust if net_thrust > 0 else float("inf")

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------
        results.update({
            # Inlet
            "inlet_p_recovery"  : p_recovery,
            "inlet_dc60"        : dc60,
            "inlet_meets_mil"   : inlet["meets_mil"],
            "P_fan_face"        : P_fan_face,

            # Thermodynamic stations
            "T2": T2, "P2": P2,
            "T3": T3, "P3": P3,
            "T4": T4, "P4": P4,
            "T5": T5, "P5": P5,

            # Performance
            "V_exit"             : V5,
            "V0_ram"             : V0,
            "M_exit"             : M5,
            "gross_thrust_N"     : gross_thrust,
            "thrust_N"           : net_thrust,
            "specific_impulse_s" : Isp,
            "fuel_flow_kg_s"     : mdot_fuel,
            "tsfc"               : tsfc,
        })

        return results

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------
        results.update({
            # Inlet
            "inlet_p_recovery"  : p_recovery,
            "inlet_dc60"        : dc60,
            "inlet_meets_mil"   : inlet["meets_mil"],
            "P_fan_face"        : P_fan_face,

            # Thermodynamic stations
            "T2": T2, "P2": P2,
            "T3": T3, "P3": P3,
            "T4": T4, "P4": P4,
            "T5": T5, "P5": P5,

            # Performance
            "V_exit"             : V5,
            "M_exit"             : M5,
            "thrust_N"           : thrust,
            "specific_impulse_s" : Isp,
            "fuel_flow_kg_s"     : mdot_fuel,
            "tsfc"               : tsfc,
        })

        return results


# ============================================================
# STANDALONE VALIDATION
# ============================================================

if __name__ == "__main__":

    T_sl = 288.15    # sea-level ISA temperature [K]
    P_sl = 101325.0  # sea-level ISA pressure    [Pa]

    print("\n" + "=" * 58)
    print("  engine_model.py — Phase 4 Inlet-Coupled Validation")
    print("=" * 58)

    # --- Case 1: subsonic cruise, Mach 0.85 ---
    engine_sub = EngineModel(mach=0.85)
    out_sub    = engine_sub.run(T_sl, P_sl)

    # --- Case 2: ACE design point, Mach 1.6 ---
    engine_sup = EngineModel(mach=1.6)
    out_sup    = engine_sup.run(T_sl, P_sl)

    # --- Case 3: perfect inlet baseline (p_recovery = 1.0) ---
    engine_base = EngineModel(mach=0.0)
    out_base    = engine_base.run(T_sl, P_sl)

    cases = [
        ("Baseline (no inlet)", out_base),
        ("Mach 0.85 subsonic",  out_sub),
        ("Mach 1.60 supersonic", out_sup),
    ]

    print(f"\n  {'Case':<24} {'P_rec':>6}  {'DC60':>6}  "
          f"{'Thrust [N]':>10}  {'TSFC':>10}  {'Isp [s]':>8}")
    print("  " + "-" * 70)

    for label, out in cases:
        print(f"  {label:<24} {out['inlet_p_recovery']:>6.4f}  "
              f"{out['inlet_dc60']:>6.3f}  "
              f"{out['thrust_N']:>10.1f}  "
              f"{out['tsfc']:>10.6f}  "
              f"{out['specific_impulse_s']:>8.1f}")

    print("\n  Expected: Mach 1.6 thrust and Isp slightly lower than baseline")
    print("  due to inlet total-pressure loss.\n")
    print("=" * 58)