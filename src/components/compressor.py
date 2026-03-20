"""
compressor.py
Compressor Component Model for 0-D Baseline Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine 

Physical Model:
- Isentropic compression with efficiency losses
- Based on ideal gas relations with real-gas corrections via efficiency
"""

import math
from typing import Tuple

# ===========================
#   CONSTANTS
# ===========================
GAMMA_AIR = 1.4
CP_AIR = 1005.0
R_AIR = 287.0


# ===========================
#   COMPRESSOR MODEL
# ===========================
def compressor(
    T_in: float,
    P_in: float,
    pressure_ratio: float,
    efficiency: float
) -> Tuple[float, float]:
    """
    Axial/centrifugal compressor model with isentropic efficiency.
    
    Physical Process:
    1. Air enters at ambient or inlet conditions
    2. Rotating blades/impellers add kinetic energy
    3. Diffusers convert kinetic energy to pressure rise
    4. Real process has losses (friction, separation, leakage)
    
    Args:
        T_in: Inlet temperature [K]
        P_in: Inlet pressure [Pa]
        pressure_ratio: Overall pressure ratio (P_out/P_in) [-]
        efficiency: Isentropic efficiency (0-1) [-]
            Typical values: 0.85-0.92 for modern compressors
    
    Returns:
        Tuple[float, float]: (T_out, P_out)
            T_out: Outlet temperature [K]
            P_out: Outlet pressure [Pa]
    
    Equations:
        Isentropic: T_out,ideal = T_in * (PR)^[(γ-1)/γ]
        Actual:     T_out = T_in + (T_out,ideal - T_in) / η_c
    """
    
    # ===========================
    # Outlet Pressure (Direct)
    # ===========================
    P_out = P_in * pressure_ratio
    
    # ===========================
    # Isentropic Temperature Rise
    # ===========================
    exponent = (GAMMA_AIR - 1.0) / GAMMA_AIR
    T_out_isentropic = T_in * math.pow(pressure_ratio, exponent)
    
    # ===========================
    # Actual Temperature (with losses)
    # ===========================
    # Isentropic efficiency: η_c = (T_s - T_in) / (T_actual - T_in)
    # Rearranging: T_actual = T_in + (T_s - T_in) / η_c
    
    delta_T_isentropic = T_out_isentropic - T_in
    delta_T_actual = delta_T_isentropic / efficiency
    T_out = T_in + delta_T_actual
    
    return T_out, P_out


# ===========================
#   UTILITY FUNCTIONS
# ===========================
def compute_compressor_work(T_in: float, T_out: float, mass_flow: float = 1.0) -> float:
    """
    Calculate compressor power requirement.
    
    Args:
        T_in: Inlet temperature [K]
        T_out: Outlet temperature [K]
        mass_flow: Air mass flow rate [kg/s]
    
    Returns:
        float: Compressor power [W] or specific work [J/kg] if mass_flow=1.0
    """
    return mass_flow * CP_AIR * (T_out - T_in)


def compute_stage_loading(T_in: float, T_out: float, num_stages: int) -> float:
    """
    Estimate temperature rise per compressor stage.
    
    Args:
        T_in: Inlet temperature [K]
        T_out: Outlet temperature [K]
        num_stages: Number of compressor stages
    
    Returns:
        float: Average temperature rise per stage [K]
    """
    return (T_out - T_in) / num_stages


# ===========================
#   VALIDATION & TESTING
# ===========================
def validate_compressor() -> None:
    """
    Test compressor model with representative values.
    """
    print("="*60)
    print("COMPRESSOR COMPONENT VALIDATION")
    print("="*60)
    
    # Test case: Sea-level to high-pressure compressor exit
    T_in = 288.15       # ISA sea level [K]
    P_in = 101325.0     # 1 atm [Pa]
    PR = 18.0           # Modern high-bypass turbofan
    eta = 0.88          # Typical efficiency
    
    T_out, P_out = compressor(T_in, P_in, PR, eta)
    
    print(f"\nInput Conditions:")
    print(f"  T_in  = {T_in:.2f} K ({T_in-273.15:.2f} °C)")
    print(f"  P_in  = {P_in/1e3:.2f} kPa ({P_in/1e5:.2f} bar)")
    print(f"  PR    = {PR:.1f}")
    print(f"  η_c   = {eta:.3f}")
    
    print(f"\nOutput Conditions:")
    print(f"  T_out = {T_out:.2f} K ({T_out-273.15:.2f} °C)")
    print(f"  P_out = {P_out/1e6:.3f} MPa ({P_out/1e5:.2f} bar)")
    
    # Calculate work
    specific_work = compute_compressor_work(T_in, T_out, mass_flow=1.0)
    
    print(f"\nPerformance Metrics:")
    print(f"  ΔT    = {T_out - T_in:.2f} K")
    print(f"  Work  = {specific_work/1e3:.2f} kJ/kg")
    
    # Typical checks
    if 450 <= T_out <= 650:
        print(f"  ✓ Exit temperature within typical range (450-650 K)")
    else:
        print(f"  ⚠ Exit temperature outside typical range")
    
    print("="*60)


if __name__ == "__main__":
    validate_compressor()