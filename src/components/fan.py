"""
fan.py
Fan Component Model for 0-D Baseline Turbofan Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine

Physical Model:
- Low-pressure fan compresses both core and bypass air
- Isentropic compression with efficiency losses
- Bypass stream exits directly to nozzle (no combustion)
- Core stream passes to high-pressure compressor
"""

import math
from typing import Tuple

# ===========================
#   CONSTANTS
# ===========================
GAMMA_AIR = 1.4
CP_AIR    = 1005.0
R_AIR     = 287.0


# ===========================
#   FAN MODEL
# ===========================
def fan(
    T_in: float,
    P_in: float,
    fan_pressure_ratio: float,
    efficiency: float
) -> Tuple[float, float]:
    """
    Low-pressure fan model with isentropic efficiency.

    Args:
        T_in: Inlet total temperature [K]
        P_in: Inlet total pressure [Pa]
        fan_pressure_ratio: Fan pressure ratio (P_out/P_in) [-]
            Typical values: 1.3-2.0 for military turbofans
        efficiency: Fan isentropic efficiency (0-1) [-]
            Typical values: 0.85-0.90

    Returns:
        Tuple[float, float]: (T_out, P_out)
    """
    P_out = P_in * fan_pressure_ratio

    exponent = (GAMMA_AIR - 1.0) / GAMMA_AIR
    T_out_isentropic = T_in * math.pow(fan_pressure_ratio, exponent)

    delta_T_isentropic = T_out_isentropic - T_in
    delta_T_actual     = delta_T_isentropic / efficiency
    T_out = T_in + delta_T_actual

    return T_out, P_out


# ===========================
#   UTILITY FUNCTIONS
# ===========================
def compute_fan_work(T_in: float, T_out: float, mass_flow: float = 1.0) -> float:
    """Shaft work required to drive the fan [J/kg or W]."""
    return mass_flow * CP_AIR * (T_out - T_in)


def split_mass_flow(total_mass_flow: float, bypass_ratio: float) -> Tuple[float, float]:
    """
    Split total inlet mass flow into core and bypass streams.

    BPR = m_bypass / m_core
    Returns: (m_dot_core, m_dot_bypass) [kg/s]
    """
    m_dot_core   = total_mass_flow / (1.0 + bypass_ratio)
    m_dot_bypass = total_mass_flow * bypass_ratio / (1.0 + bypass_ratio)
    return m_dot_core, m_dot_bypass


def compute_bypass_exit_velocity(T_fan_exit: float, P_fan_exit: float, P_ambient: float) -> float:
    """
    Isentropic expansion of bypass stream through cold nozzle to ambient.
    Returns bypass nozzle exit velocity [m/s].
    """
    pressure_ratio = P_fan_exit / P_ambient
    exponent = (GAMMA_AIR - 1.0) / GAMMA_AIR
    T_exit = T_fan_exit / math.pow(pressure_ratio, exponent)
    V_exit = math.sqrt(2.0 * CP_AIR * (T_fan_exit - T_exit))
    return V_exit


def compute_bypass_thrust(m_dot_bypass: float, V_bypass: float, V_freestream: float = 0.0) -> float:
    """Thrust contribution from the bypass stream [N]."""
    return m_dot_bypass * (V_bypass - V_freestream)


# ===========================
#   VALIDATION & TESTING
# ===========================
def validate_fan() -> None:
    print("=" * 60)
    print("FAN COMPONENT VALIDATION")
    print("=" * 60)

    T_in  = 288.15
    P_in  = 101325.0
    FPR   = 1.6
    eta   = 0.87
    BPR   = 0.3
    mdot  = 50.0

    T_out, P_out = fan(T_in, P_in, FPR, eta)

    print(f"\nInput Conditions:")
    print(f"  T_in  = {T_in:.2f} K")
    print(f"  P_in  = {P_in / 1e3:.2f} kPa")
    print(f"  FPR   = {FPR:.2f}")
    print(f"  eta   = {eta:.3f}")
    print(f"  BPR   = {BPR:.2f}")

    print(f"\nFan Exit Conditions:")
    print(f"  T_out = {T_out:.2f} K")
    print(f"  P_out = {P_out / 1e3:.2f} kPa")
    print(f"  dT    = {T_out - T_in:.2f} K")

    m_core, m_bypass = split_mass_flow(mdot, BPR)
    print(f"\nMass Flow Split (total={mdot} kg/s, BPR={BPR}):")
    print(f"  Core stream   = {m_core:.2f} kg/s")
    print(f"  Bypass stream = {m_bypass:.2f} kg/s")

    V_bypass = compute_bypass_exit_velocity(T_out, P_out, P_in)
    bypass_thrust = compute_bypass_thrust(m_bypass, V_bypass)
    print(f"\nBypass Stream:")
    print(f"  Exit velocity = {V_bypass:.1f} m/s")
    print(f"  Bypass thrust = {bypass_thrust:.1f} N")

    print(f"\nValidation Checks:")
    print(f"  {'✓' if 300 <= T_out <= 380 else '⚠'} Fan exit temp {'OK' if 300 <= T_out <= 380 else 'OUT OF RANGE'} (300-380 K)")
    print(f"  {'✓' if 1.3 <= FPR <= 2.0 else '⚠'} FPR {'OK' if 1.3 <= FPR <= 2.0 else 'OUT OF RANGE'} (1.3-2.0)")
    print("=" * 60)


if __name__ == "__main__":
    validate_fan()