"""
engine_model.py
0-D Adaptive Cycle Turbofan Engine Model (Integrated Component Flow)
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine — Phase 3
"""

# ============================================================
# IMPORTS
# ============================================================

from components.compressor import compressor, compute_compressor_work
from components.fan import fan, split_mass_flow, compute_bypass_exit_velocity, compute_bypass_thrust
from components.combustor import combustor, compute_heat_release
from components.turbine import turbine
from components.nozzle import nozzle, compute_thrust_simple, compute_specific_impulse
from bpr_schedule import bpr_schedule

from typing import Dict, Optional


# ============================================================
# CONSTANTS
# ============================================================

g0 = 9.81       # gravitational acceleration [m/s²]


# ============================================================
# ENGINE MODEL CLASS
# ============================================================

class EngineModel:
    """
    0-D adaptive cycle turbofan engine model.

    Handles:
        - Fan + bypass stream (adaptive or fixed BPR)
        - Sequential thermodynamic processing
        - Fan -> Compressor -> Combustor -> Turbine -> Nozzle
        - Turbine work balance to power compressor
        - Fuel-air ratio calculations
        - Core thrust + bypass thrust -> total thrust
        - Isp

    Phase 3 Addition:
        - bypass_ratio=None activates the adaptive BPR schedule
          which varies BPR as a function of Mach and altitude.
        - bypass_ratio=<float> locks BPR to a fixed value (Phase 2 mode).
    """

    def __init__(
        self,
        mass_flow: float = 50.0,         # kg/s  total inlet air flow
        compressor_PR: float = 18.0,     # overall core pressure ratio
        compressor_eff: float = 0.88,    # compressor isentropic efficiency
        turbine_eff: float = 0.90,       # turbine isentropic efficiency
        f: float = 0.020,                # fuel-air ratio
        bypass_ratio: Optional[float] = None,      # None = adaptive schedule; float = fixed BPR
        fan_pressure_ratio: float = 1.6, # FPR across fan stage
        fan_eff: float = 0.87            # fan isentropic efficiency
    ) -> None:

        self.mass_flow          = mass_flow
        self.compressor_PR      = compressor_PR
        self.compressor_eff     = compressor_eff
        self.turbine_eff        = turbine_eff
        self.f                  = f
        self.bypass_ratio       = bypass_ratio   # None means use adaptive schedule
        self.fan_pressure_ratio = fan_pressure_ratio
        self.fan_eff            = fan_eff

    # ============================================================
    # MAIN ENGINE RUN METHOD
    # ============================================================

    def run(
        self,
        T_ambient: float,
        P_ambient: float,
        mach: float = 0.0,
        altitude_m: float = 0.0
    ) -> Dict[str, float]:
        """
        Run the 0-D adaptive turbofan and compute station states,
        thrust breakdown, Isp, and fuel flow.

        Args:
            T_ambient:  Inlet total temperature [K]
            P_ambient:  Inlet total pressure [Pa]
            mach:       Freestream Mach number (used for adaptive BPR) [-]
            altitude_m: Geometric altitude (used for adaptive BPR) [m]

        Returns:
            dict[str, float] of results
        """

        results: Dict[str, float] = {}

        # --------------------------------------------------------
        # ADAPTIVE BPR — resolve active bypass ratio
        # If bypass_ratio is None, read from the adaptive schedule.
        # If bypass_ratio is a fixed float, use that directly.
        # --------------------------------------------------------
        if self.bypass_ratio is None:
            active_bpr = bpr_schedule(mach, altitude_m)
        else:
            active_bpr = self.bypass_ratio

        # --------------------------------------------------------
        # 1. FAN — compresses entire inlet flow (core + bypass)
        # --------------------------------------------------------
        T_fan, P_fan = fan(
            T_in=T_ambient,
            P_in=P_ambient,
            fan_pressure_ratio=self.fan_pressure_ratio,
            efficiency=self.fan_eff
        )

        # Split mass flow into core and bypass streams using active BPR
        m_core, m_bypass = split_mass_flow(self.mass_flow, active_bpr)

        # Bypass stream exits through cold nozzle — compute bypass thrust
        V_bypass      = compute_bypass_exit_velocity(T_fan, P_fan, P_ambient)
        bypass_thrust = compute_bypass_thrust(m_bypass, V_bypass)

        # --------------------------------------------------------
        # 2. COMPRESSOR — core stream only, fed by fan exit
        # --------------------------------------------------------
        T2, P2 = compressor(
            T_in=T_fan,
            P_in=P_fan,
            pressure_ratio=self.compressor_PR,
            efficiency=self.compressor_eff
        )

        Wc = compute_compressor_work(
            T_in=T_fan,
            T_out=T2,
            mass_flow=1.0
        )

        # --------------------------------------------------------
        # 3. COMBUSTOR
        # --------------------------------------------------------
        T3, P3 = combustor(
            T_in=T2,
            P_in=P2,
            fuel_air_ratio=self.f
        )

        # --------------------------------------------------------
        # 4. TURBINE — provides compressor shaft work
        # --------------------------------------------------------
        T4, P4 = turbine(
            T_in=T3,
            P_in=P3,
            work_required=Wc,
            efficiency=self.turbine_eff
        )

        # --------------------------------------------------------
        # 5. NOZZLE -> CORE THRUST
        # --------------------------------------------------------
        T5, P5, V5, M5 = nozzle(
            T_in=T4,
            P_in=P4,
            P_ambient=P_ambient
        )

        core_thrust  = compute_thrust_simple(m_core, V5)
        total_thrust = core_thrust + bypass_thrust

        mdot_fuel = m_core * self.f
        Isp = compute_specific_impulse(total_thrust, mdot_fuel)

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------
        results.update({
            "T_fan": T_fan, "P_fan": P_fan,
            "T2": T2, "P2": P2,
            "T3": T3, "P3": P3,
            "T4": T4, "P4": P4,
            "T5": T5, "P5": P5,
            "V_exit": V5,
            "M_exit": M5,
            "m_core_kg_s":   m_core,
            "m_bypass_kg_s": m_bypass,
            "core_thrust_N":   core_thrust,
            "bypass_thrust_N": bypass_thrust,
            "thrust_N":        total_thrust,
            "specific_impulse_s": Isp,
            "fuel_flow_kg_s":     mdot_fuel,
            "bypass_ratio":       active_bpr,
            "adaptive_mode":      self.bypass_ratio is None,
        })

        return results


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    # --- Test 1: Adaptive mode ---
    print("\n=== ADAPTIVE MODE — Mach 0.9 at 8000 m ===")
    engine_adaptive = EngineModel()
    out = engine_adaptive.run(288.15, 101325.0, mach=0.9, altitude_m=8000)

    print(f"{'Active BPR':<25s} {out['bypass_ratio']:>12.4f}  -  (from schedule)")
    print(f"{'Adaptive Mode':<25s} {str(out['adaptive_mode']):>12s}")
    print(f"{'m_core':<25s} {out['m_core_kg_s']:>12.2f}  kg/s")
    print(f"{'m_bypass':<25s} {out['m_bypass_kg_s']:>12.2f}  kg/s")
    print(f"{'Core Thrust':<25s} {out['core_thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Bypass Thrust':<25s} {out['bypass_thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Total Thrust':<25s} {out['thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Isp':<25s} {out['specific_impulse_s']:>12.2f}  s")

    # --- Test 2: Fixed mode (Phase 2 baseline) ---
    print("\n=== FIXED MODE — BPR = 0.3 (Phase 2 baseline) ===")
    engine_fixed = EngineModel(bypass_ratio=0.3)
    out2 = engine_fixed.run(288.15, 101325.0)

    print(f"{'Active BPR':<25s} {out2['bypass_ratio']:>12.4f}  -  (fixed)")
    print(f"{'Adaptive Mode':<25s} {str(out2['adaptive_mode']):>12s}")
    print(f"{'Total Thrust':<25s} {out2['thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Isp':<25s} {out2['specific_impulse_s']:>12.2f}  s")

    print("\n--- Comparison ---")
    delta_thrust = out['thrust_N'] - out2['thrust_N']
    delta_isp    = out['specific_impulse_s'] - out2['specific_impulse_s']
    print(f"{'Thrust difference':<25s} {delta_thrust/1e3:>+12.2f}  kN")
    print(f"{'Isp difference':<25s} {delta_isp:>+12.2f}  s")