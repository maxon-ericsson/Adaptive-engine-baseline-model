"""
combustor.py
Combustor Component Model for 0-D Baseline Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine

Physical Model:
- Constant-pressure heat addition with combustion efficiency
- Pressure loss due to friction and momentum effects
- Fuel energy release modeled via Lower Heating Value (LHV)
"""

from typing import Tuple

# ===========================
#   CONSTANTS
# ===========================
LHV_JET_A = 43.0e6          # Lower Heating Value for Jet-A fuel [J/kg]
CP_AIR = 1005.0             # Specific heat of air at constant pressure [J/kg-K]
DEFAULT_ETA_COMB = 0.99     # Default combustion efficiency [-]
DEFAULT_PRESSURE_LOSS = 0.03  # Default fractional pressure drop [-]


# ===========================
#   COMBUSTOR MODEL
# ===========================
def combustor(
    T_in: float,
    P_in: float,
    fuel_air_ratio: float,
    combustion_efficiency: float = DEFAULT_ETA_COMB,
    pressure_loss_factor: float = DEFAULT_PRESSURE_LOSS
) -> Tuple[float, float]:
    """
    Constant-pressure combustor model with heat addition and pressure loss.
    
    Physical Process:
    1. Fuel is injected and burns with incoming compressed air
    2. Chemical energy converts to thermal energy (temperature rise)
    3. Friction and mixing cause pressure drop
    
    Args:
        T_in: Inlet temperature [K]
        P_in: Inlet pressure [Pa]
        fuel_air_ratio: Fuel-to-air mass ratio [-]
        combustion_efficiency: Fraction of fuel energy released (0-1) [-]
        pressure_loss_factor: Fractional pressure drop (0-1) [-]
    
    Returns:
        Tuple[float, float]: (T_out, P_out)
            T_out: Outlet temperature [K]
            P_out: Outlet pressure [Pa]
    
    Equations:
        ΔT = η_comb * f * LHV / Cp
        P_out = P_in * (1 - ΔP/P)
    
    Example:
        >>> T_out, P_out = combustor(500.0, 1.8e6, 0.02)
        >>> print(f"Exit temp: {T_out:.1f} K, Exit pressure: {P_out/1e6:.2f} MPa")
    """
    
    # ===========================
    # Temperature Rise Calculation
    # ===========================
    # Energy balance: Q_released = ṁ_fuel * LHV * η_comb
    # Temperature rise: ΔT = Q / (ṁ_air * Cp)
    #                      = (ṁ_fuel/ṁ_air) * LHV * η_comb / Cp
    
    delta_T = combustion_efficiency * fuel_air_ratio * LHV_JET_A / CP_AIR
    T_out = T_in + delta_T
    
    # ===========================
    # Pressure Loss Calculation
    # ===========================
    # Total pressure drop due to:
    # - Friction with combustor walls
    # - Momentum effects from fuel injection
    # - Heat addition (Rayleigh flow effects)
    
    P_out = P_in * (1.0 - pressure_loss_factor)
    
    return T_out, P_out


# ===========================
#   UTILITY FUNCTIONS
# ===========================
def compute_heat_release(fuel_air_ratio: float, mass_flow: float) -> float:
    """
    Calculate total heat release rate in combustor.
    
    Args:
        fuel_air_ratio: Fuel-to-air mass ratio [-]
        mass_flow: Air mass flow rate [kg/s]
    
    Returns:
        float: Heat release rate [W]
    """
    mdot_fuel = mass_flow * fuel_air_ratio
    return mdot_fuel * LHV_JET_A


def compute_flame_temperature(T_in: float, fuel_air_ratio: float, 
                              efficiency: float = DEFAULT_ETA_COMB) -> float:
    """
    Calculate adiabatic flame temperature (simplified).
    
    Args:
        T_in: Inlet temperature [K]
        fuel_air_ratio: Fuel-to-air mass ratio [-]
        efficiency: Combustion efficiency [-]
    
    Returns:
        float: Approximate flame temperature [K]
    """
    T_out, _ = combustor(T_in, 101325.0, fuel_air_ratio, efficiency, 0.0)
    return T_out


# ===========================
#   VALIDATION & TESTING
# ===========================
def validate_combustor() -> None:
    """
    Test combustor model with representative values.
    Typical modern turbofan combustor exit temperatures: 1400-1700 K
    """
    print("="*60)
    print("COMBUSTOR COMPONENT VALIDATION")
    print("="*60)
    
    # Test case: Compressor exit to combustor exit
    T_in = 500.0        # Typical compressor exit temp [K]
    P_in = 1.8e6        # ~18 bar after compression [Pa]
    f = 0.02            # 2% fuel-air ratio
    
    T_out, P_out = combustor(T_in, P_in, f)
    
    print(f"\nInput Conditions:")
    print(f"  T_in  = {T_in:.1f} K ({T_in-273.15:.1f} °C)")
    print(f"  P_in  = {P_in/1e6:.2f} MPa ({P_in/1e5:.1f} bar)")
    print(f"  f     = {f:.4f} (fuel-air ratio)")
    
    print(f"\nOutput Conditions:")
    print(f"  T_out = {T_out:.1f} K ({T_out-273.15:.1f} °C)")
    print(f"  P_out = {P_out/1e6:.2f} MPa ({P_out/1e5:.1f} bar)")
    
    print(f"\nPerformance Metrics:")
    print(f"  ΔT = {T_out - T_in:.1f} K")
    print(f"  ΔP/P = {(P_in - P_out)/P_in * 100:.2f}%")
    
    # Heat release for 50 kg/s air flow
    heat_release = compute_heat_release(f, 50.0)
    print(f"  Heat Release (50 kg/s): {heat_release/1e6:.1f} MW")
    
    # Typical ranges check
    if 1400 <= T_out <= 1800:
        print(f"  ✓ Exit temperature within typical range (1400-1800 K)")
    else:
        print(f"  ⚠ Exit temperature outside typical range")
    
    # Pressure loss check
    if 0.02 <= (P_in - P_out)/P_in <= 0.05:
        print(f"  ✓ Pressure loss within typical range (2-5%)")
    else:
        print(f"  ⚠ Pressure loss outside typical range")
    
    print("="*60)


if __name__ == "__main__":
    validate_combustor()