"""
engine_model.py
0-D Baseline Turbofan Engine Model (Integrated Component Flow)
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine — Phase 2
"""

# ============================================================
# IMPORTS
# ============================================================

from components.compressor import compressor, compute_compressor_work
from components.fan import fan, split_mass_flow, compute_bypass_exit_velocity, compute_bypass_thrust
from components.combustor import combustor, compute_heat_release
from components.turbine import turbine
from components.nozzle import nozzle, compute_thrust_simple, compute_specific_impulse

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
    Simple 0-D turbofan engine model wrapper.

    Handles:
        • Fan + bypass stream (BPR, FPR)
        • Sequential thermodynamic processing
        • Fan → Compressor → Combustor → Turbine → Nozzle
        • Turbine work balance to power compressor
        • Fuel–air ratio calculations
        • Core thrust + bypass thrust → total thrust
        • Isp
    """

    def __init__(
        self,
        mass_flow: float = 50.0,         # kg/s  total inlet air flow
        compressor_PR: float = 18.0,     # overall core pressure ratio
        compressor_eff: float = 0.88,    # compressor isentropic efficiency
        turbine_eff: float = 0.90,       # turbine isentropic efficiency
        f: float = 0.020,                # fuel–air ratio
        bypass_ratio: float = 0.3,       # BPR = m_bypass / m_core (NGAD-class)
        fan_pressure_ratio: float = 1.6, # FPR across fan stage
        fan_eff: float = 0.87            # fan isentropic efficiency
    ) -> None:

        self.mass_flow          = mass_flow
        self.compressor_PR      = compressor_PR
        self.compressor_eff     = compressor_eff
        self.turbine_eff        = turbine_eff
        self.f                  = f
        self.bypass_ratio       = bypass_ratio
        self.fan_pressure_ratio = fan_pressure_ratio
        self.fan_eff            = fan_eff

    # ============================================================
    # MAIN ENGINE RUN METHOD
    # ============================================================

    def run(self, T_ambient: float, P_ambient: float) -> Dict[str, float]:
        """
        Run the 0-D turbofan and compute:
            • Station thermodynamic states (T, P at each station)
            • Core thrust, bypass thrust, total thrust
            • Specific impulse (Isp)
            • Fuel flow

        Returns:
            dict[str, float] of results
        """

        results: Dict[str, float] = {}

        # --------------------------------------------------------
        # 1. FAN — compresses entire inlet flow (core + bypass)
        # --------------------------------------------------------
        T_fan, P_fan = fan(
            T_in=T_ambient,
            P_in=P_ambient,
            fan_pressure_ratio=self.fan_pressure_ratio,
            efficiency=self.fan_eff
        )

        # Split mass flow into core and bypass streams
        m_core, m_bypass = split_mass_flow(self.mass_flow, self.bypass_ratio)

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
        # 5. NOZZLE → CORE THRUST
        # --------------------------------------------------------
        T5, P5, V5, M5 = nozzle(
            T_in=T4,
            P_in=P4,
            P_ambient=P_ambient
        )

        core_thrust = compute_thrust_simple(m_core, V5)
        total_thrust = core_thrust + bypass_thrust

        mdot_fuel = m_core * self.f
        Isp = compute_specific_impulse(total_thrust, mdot_fuel)

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------
        results.update({
            # Fan exit
            "T_fan": T_fan, "P_fan": P_fan,

            # Core stations
            "T2": T2, "P2": P2,
            "T3": T3, "P3": P3,
            "T4": T4, "P4": P4,
            "T5": T5, "P5": P5,

            # Nozzle exit
            "V_exit": V5,
            "M_exit": M5,

            # Mass flow split
            "m_core_kg_s":   m_core,
            "m_bypass_kg_s": m_bypass,

            # Thrust breakdown
            "core_thrust_N":   core_thrust,
            "bypass_thrust_N": bypass_thrust,
            "thrust_N":        total_thrust,

            # Performance
            "specific_impulse_s": Isp,
            "fuel_flow_kg_s":     mdot_fuel,
            "bypass_ratio":       self.bypass_ratio,
        })

        return results


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    engine = EngineModel()
    out = engine.run(288.15, 101325.0)

    print("\n=== BASELINE TURBOFAN OUTPUT ===")
    print(f"{'Station':<25s} {'Value':>12s}  Unit")
    print("-" * 45)
    print(f"{'T_fan':<25s} {out['T_fan']:>12.2f}  K")
    print(f"{'P_fan':<25s} {out['P_fan']/1e3:>12.2f}  kPa")
    print(f"{'T2 (compressor exit)':<25s} {out['T2']:>12.2f}  K")
    print(f"{'P2':<25s} {out['P2']/1e6:>12.3f}  MPa")
    print(f"{'T3 (combustor exit)':<25s} {out['T3']:>12.2f}  K")
    print(f"{'T4 (turbine exit)':<25s} {out['T4']:>12.2f}  K")
    print(f"{'T5 (nozzle exit)':<25s} {out['T5']:>12.2f}  K")
    print(f"{'V_exit':<25s} {out['V_exit']:>12.2f}  m/s")
    print(f"{'M_exit':<25s} {out['M_exit']:>12.3f}  -")
    print("-" * 45)
    print(f"{'m_core':<25s} {out['m_core_kg_s']:>12.2f}  kg/s")
    print(f"{'m_bypass':<25s} {out['m_bypass_kg_s']:>12.2f}  kg/s")
    print(f"{'Core Thrust':<25s} {out['core_thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Bypass Thrust':<25s} {out['bypass_thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Total Thrust':<25s} {out['thrust_N']/1e3:>12.2f}  kN")
    print(f"{'Isp':<25s} {out['specific_impulse_s']:>12.2f}  s")
    print(f"{'Fuel Flow':<25s} {out['fuel_flow_kg_s']:>12.4f}  kg/s")