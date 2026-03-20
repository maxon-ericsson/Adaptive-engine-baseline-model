"""
turbine.py
Turbine Component Model for 0-D Baseline Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine 

Physical Model:
- Work-extraction turbine to drive compressor
- Isentropic expansion with efficiency losses
- Hot gas expansion (γ = 1.33 for combustion products)
- Energy balance ensures compressor work is supplied
"""

import math
from typing import Tuple

# ===========================
#   CONSTANTS
# ===========================
GAMMA_HOT = 1.33            # Ratio of specific heats for hot gas [-]
CP_HOT = 1005.0             # Specific heat (approximate) [J/kg-K]
T_MIN_PHYSICAL = 300.0      # Minimum physical temperature limit [K]


# ===========================
#   TURBINE MODEL
# ===========================
def turbine(
    T_in: float,
    P_in: float,
    work_required: float,
    efficiency: float
) -> Tuple[float, float]:
    """
    Axial turbine model that extracts work to drive compressor.
    
    Physical Process:
    1. Hot, high-pressure gas from combustor enters turbine
    2. Gas expands through turbine blades, spinning rotor
    3. Rotor mechanically coupled to compressor (same shaft)
    4. Must extract exactly enough work to drive compressor
    
    Args:
        T_in: Turbine inlet temperature [K]
        P_in: Turbine inlet pressure [Pa]
        work_required: Compressor work requirement [J/kg]
        efficiency: Turbine isentropic efficiency (0-1) [-]
            Typical values: 0.88-0.92 for modern turbines
    
    Returns:
        Tuple[float, float]: (T_out, P_out)
            T_out: Outlet temperature [K]
            P_out: Outlet pressure [Pa]
    
    Equations:
        Ideal:  ΔT_ideal = W_required / Cp
        Actual: ΔT_actual = ΔT_ideal / η_t
        P_out = P_in * (T_out/T_in)^[γ/(γ-1)]
    """
    
    # ===========================
    # Isentropic Temperature Drop
    # ===========================
    # For ideal turbine: W_ideal = Cp * ΔT_ideal
    # Therefore: ΔT_ideal = W_required / Cp
    delta_T_isentropic = work_required / CP_HOT
    
    # ===========================
    # Actual Temperature Drop
    # ===========================
    # Turbine efficiency: η_t = W_actual / W_ideal = ΔT_actual / ΔT_ideal
    # But we need W_actual = W_required, so:
    # η_t = ΔT_ideal / ΔT_actual
    # Rearranging: ΔT_actual = ΔT_ideal / η_t
    
    delta_T_actual = delta_T_isentropic / efficiency
    
    T_out = T_in - delta_T_actual
    
    # ===========================
    # Physical Temperature Limit
    # ===========================
    # Prevent unphysical low temperatures (would indicate model failure)
    if T_out < T_MIN_PHYSICAL:
        T_out = T_MIN_PHYSICAL
        # Note: In production code, this should raise a warning or exception
    
    # ===========================
    # Pressure Drop Calculation
    # ===========================
    # Isentropic relation for expansion: P/P_in = (T/T_in)^[γ/(γ-1)]
    temperature_ratio = T_out / T_in
    exponent = GAMMA_HOT / (GAMMA_HOT - 1.0)
    pressure_ratio = math.pow(temperature_ratio, exponent)
    
    P_out = P_in * pressure_ratio
    
    return T_out, P_out


# ===========================
#   UTILITY FUNCTIONS
# ===========================
def compute_turbine_power(T_in: float, T_out: float, mass_flow: float = 1.0) -> float:
    """
    Calculate turbine power output.
    
    Args:
        T_in: Inlet temperature [K]
        T_out: Outlet temperature [K]
        mass_flow: Gas mass flow rate [kg/s]
    
    Returns:
        float: Turbine power [W] or specific work [J/kg] if mass_flow=1.0
    """
    return mass_flow * CP_HOT * (T_in - T_out)


def compute_expansion_ratio(P_in: float, P_out: float) -> float:
    """
    Calculate turbine expansion ratio.
    
    Args:
        P_in: Inlet pressure [Pa]
        P_out: Outlet pressure [Pa]
    
    Returns:
        float: Expansion ratio P_in/P_out [-]
    """
    return P_in / P_out


def estimate_num_stages(expansion_ratio: float, 
                        stage_loading_max: float = 4.0) -> int:
    """
    Estimate number of turbine stages required.
    
    Args:
        expansion_ratio: Overall pressure ratio [-]
        stage_loading_max: Maximum expansion ratio per stage [-]
    
    Returns:
        int: Estimated number of stages
    """
    return math.ceil(math.log(expansion_ratio) / math.log(stage_loading_max))


# ===========================
#   VALIDATION & TESTING
# ===========================
def validate_turbine() -> None:
    """
    Test turbine model with representative values.
    """
    print("="*60)
    print("TURBINE COMPONENT VALIDATION")
    print("="*60)
    
    # Test case: High-pressure turbine
    T_in = 1500.0           # Typical TIT [K]
    P_in = 1.7e6            # ~17 bar after combustor [Pa]
    work_req = 215000.0     # Compressor work requirement [J/kg]
    eta = 0.90              # Turbine efficiency
    
    T_out, P_out = turbine(T_in, P_in, work_req, eta)
    
    print(f"\nInput Conditions:")
    print(f"  T_in       = {T_in:.1f} K ({T_in-273.15:.1f} °C)")
    print(f"  P_in       = {P_in/1e6:.3f} MPa ({P_in/1e5:.1f} bar)")
    print(f"  W_required = {work_req/1e3:.1f} kJ/kg")
    print(f"  η_t        = {eta:.3f}")
    
    print(f"\nOutput Conditions:")
    print(f"  T_out = {T_out:.1f} K ({T_out-273.15:.1f} °C)")
    print(f"  P_out = {P_out/1e6:.3f} MPa ({P_out/1e5:.1f} bar)")
    
    # Calculate metrics
    expansion_ratio = compute_expansion_ratio(P_in, P_out)
    power_output = compute_turbine_power(T_in, T_out, mass_flow=50.0)
    num_stages = estimate_num_stages(expansion_ratio)
    
    print(f"\nPerformance Metrics:")
    print(f"  ΔT             = {T_in - T_out:.1f} K")
    print(f"  Expansion Ratio = {expansion_ratio:.2f}")
    print(f"  Power (50 kg/s) = {power_output/1e6:.2f} MW")
    print(f"  Est. Stages     = {num_stages}")
    
    # Validation checks
    if 900 <= T_out <= 1200:
        print(f"  ✓ Exit temperature within typical range (900-1200 K)")
    else:
        print(f"  ⚠ Exit temperature outside typical range")
    
    if T_out < T_MIN_PHYSICAL:
        print(f"  ⚠ WARNING: Temperature below physical limit!")
    
    # Energy balance check
    power_extracted = compute_turbine_power(T_in, T_out, mass_flow=1.0)
    print(f"\n--- Energy Balance Check ---")
    print(f"  Work required:  {work_req/1e3:.2f} kJ/kg")
    print(f"  Work extracted: {power_extracted/1e3:.2f} kJ/kg")
    print(f"  Balance error:  {abs(power_extracted - work_req)/work_req * 100:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    validate_turbine()