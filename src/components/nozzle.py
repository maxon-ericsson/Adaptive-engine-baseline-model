"""
nozzle.py
Nozzle Component Model for 0-D Baseline Engine
Author: Maxon Ericsson
Project: Ultra-Lightweight Adaptive Cycle Engine 

Physical Model:
- Converging or converging-diverging nozzle
- Isentropic expansion to ambient pressure
- Choked flow detection and handling
- Calculates exit velocity and Mach number
"""

import math
from typing import Tuple

# ===========================
#   CONSTANTS
# ===========================
GAMMA_AIR = 1.4             # Ratio of specific heats for air [-]
R_AIR = 287.0               # Specific gas constant for air [J/kg-K]
CP_AIR = 1005.0             # Specific heat at constant pressure [J/kg-K]

# Critical pressure ratio for choked flow: [(γ+1)/2]^[γ/(γ-1)]
CRITICAL_PR = math.pow((GAMMA_AIR + 1.0) / 2.0, GAMMA_AIR / (GAMMA_AIR - 1.0))


# ===========================
#   NOZZLE MODEL
# ===========================
def nozzle(
    T_in: float,
    P_in: float,
    P_ambient: float = 101325.0
) -> Tuple[float, float, float, float]:
    """
    Converging nozzle with choked flow capability.
    
    Physical Process:
    1. High-pressure exhaust gas enters nozzle
    2. Area decreases → velocity increases, pressure decreases
    3. If pressure ratio exceeds critical value → sonic flow (choked)
    4. Otherwise → isentropic expansion to ambient pressure
    
    Args:
        T_in: Nozzle inlet temperature [K]
        P_in: Nozzle inlet pressure [Pa]
        P_ambient: Ambient back pressure [Pa]
    
    Returns:
        Tuple[float, float, float, float]: (T_exit, P_exit, V_exit, M_exit)
            T_exit: Exit temperature [K]
            P_exit: Exit pressure [Pa]
            V_exit: Exit velocity [m/s]
            M_exit: Exit Mach number [-]
    
    Flow Regimes:
        - Subsonic: P_in/P_ambient < 1.893 (for γ=1.4)
        - Choked:   P_in/P_ambient ≥ 1.893 (sonic at throat)
    
    Example:
        >>> T_e, P_e, V_e, M_e = nozzle(900.0, 300000, 101325)
        >>> print(f"Exit velocity: {V_e:.1f} m/s, Mach: {M_e:.2f}")
    """
    
    gamma = GAMMA_AIR
    R = R_AIR
    cp = CP_AIR
    
    # Calculate pressure ratio
    pressure_ratio = P_in / P_ambient
    
    # ===========================
    # CHECK FOR CHOKED FLOW
    # ===========================
    if pressure_ratio >= CRITICAL_PR:
        # ===========================
        # CHOKED FLOW (M = 1.0 at throat/exit)
        # ===========================
        # At sonic conditions: T* = T_0 * [2/(γ+1)]
        T_exit = T_in * (2.0 / (gamma + 1.0))
        
        # Pressure at sonic conditions
        P_exit = P_in * math.pow(2.0 / (gamma + 1.0), gamma / (gamma - 1.0))
        
        # Exit velocity equals speed of sound at throat
        # V* = √(γRT*) = a* (speed of sound at sonic conditions)
        a_star = math.sqrt(gamma * R * T_exit)
        V_exit = a_star
        
        # Mach number is exactly 1.0 at choked throat
        M_exit = 1.0
        
    else:
        # ===========================
        # ISENTROPIC EXPANSION (M < 1.0)
        # ===========================
        # Exit pressure equals ambient (fully expanded nozzle)
        P_exit = P_ambient
        P_ratio = P_exit / P_in
        
        # Isentropic relation: T_e/T_i = (P_e/P_i)^[(γ-1)/γ]
        exponent = (gamma - 1.0) / gamma
        T_exit = T_in * math.pow(P_ratio, exponent)
        
        # Energy equation: h_0 = h + V²/2
        # For ideal gas: Cp*T_0 = Cp*T + V²/2
        # Therefore: V = √[2*Cp*(T_in - T_exit)]
        delta_h = cp * (T_in - T_exit)
        V_exit = math.sqrt(max(2.0 * delta_h, 0.0))  # Ensure non-negative
        
        # Calculate Mach number: M = V / a
        # Speed of sound: a = √(γRT)
        a_exit = math.sqrt(gamma * R * T_exit)
        M_exit = V_exit / a_exit
    
    return T_exit, P_exit, V_exit, M_exit


# ===========================
#   UTILITY FUNCTIONS
# ===========================
def is_nozzle_choked(P_in: float, P_ambient: float) -> bool:
    """
    Determine if nozzle is operating in choked flow regime.
    
    Args:
        P_in: Inlet pressure [Pa]
        P_ambient: Ambient pressure [Pa]
    
    Returns:
        bool: True if choked (M=1 at throat), False otherwise
    """
    return (P_in / P_ambient) >= CRITICAL_PR


def compute_thrust_simple(mass_flow: float, V_exit: float) -> float:
    """
    Calculate thrust using simplified momentum equation.
    Assumes V_inlet = 0 and perfect expansion (P_exit = P_ambient).
    
    Args:
        mass_flow: Mass flow rate [kg/s]
        V_exit: Exit velocity [m/s]
    
    Returns:
        float: Thrust [N]
    """
    return mass_flow * V_exit


def compute_thrust_full(
    mass_flow: float,
    V_exit: float,
    V_inlet: float,
    P_exit: float,
    P_ambient: float,
    A_exit: float
) -> float:
    """
    Calculate thrust including momentum and pressure terms.
    
    Args:
        mass_flow: Mass flow rate [kg/s]
        V_exit: Exit velocity [m/s]
        V_inlet: Inlet velocity [m/s]
        P_exit: Exit pressure [Pa]
        P_ambient: Ambient pressure [Pa]
        A_exit: Exit area [m²]
    
    Returns:
        float: Thrust [N]
    
    Thrust Equation:
        F = ṁ(V_exit - V_inlet) + (P_exit - P_ambient) * A_exit
    """
    momentum_thrust = mass_flow * (V_exit - V_inlet)
    pressure_thrust = (P_exit - P_ambient) * A_exit
    
    return momentum_thrust + pressure_thrust


def compute_specific_impulse(thrust: float, mdot_fuel: float, g: float = 9.81) -> float:
    """
    Calculate specific impulse (Isp).
    
    Args:
        thrust: Thrust [N]
        mdot_fuel: Fuel mass flow rate [kg/s]
        g: Gravitational acceleration [m/s²]
    
    Returns:
        float: Specific impulse [s]
    """
    return thrust / (mdot_fuel * g)


# ===========================
#   VALIDATION & TESTING
# ===========================
def validate_nozzle() -> None:
    """
    Test nozzle model with subsonic and choked flow cases.
    """
    print("="*60)
    print("NOZZLE COMPONENT VALIDATION")
    print("="*60)
    
    # Test Case 1: Subsonic expansion
    print("\n--- TEST 1: Subsonic Expansion ---")
    T_in_1 = 900.0
    P_in_1 = 150000.0  # 1.5 bar
    P_amb = 101325.0
    
    T_exit_1, P_exit_1, V_exit_1, M_exit_1 = nozzle(T_in_1, P_in_1, P_amb)
    
    print(f"Inlet:  T={T_in_1:.1f} K, P={P_in_1/1e3:.1f} kPa")
    print(f"Ambient: P={P_amb/1e3:.1f} kPa")
    print(f"Exit:   T={T_exit_1:.1f} K, P={P_exit_1/1e3:.1f} kPa")
    print(f"        V={V_exit_1:.1f} m/s, M={M_exit_1:.3f}")
    print(f"Pressure Ratio: {P_in_1/P_amb:.3f}")
    print(f"Choked: {is_nozzle_choked(P_in_1, P_amb)}")
    
    # Test Case 2: Choked flow
    print("\n--- TEST 2: Choked Flow ---")
    T_in_2 = 1200.0
    P_in_2 = 400000.0  # 4 bar
    
    T_exit_2, P_exit_2, V_exit_2, M_exit_2 = nozzle(T_in_2, P_in_2, P_amb)
    
    print(f"Inlet:  T={T_in_2:.1f} K, P={P_in_2/1e3:.1f} kPa")
    print(f"Ambient: P={P_amb/1e3:.1f} kPa")
    print(f"Exit:   T={T_exit_2:.1f} K, P={P_exit_2/1e3:.1f} kPa")
    print(f"        V={V_exit_2:.1f} m/s, M={M_exit_2:.3f}")
    print(f"Pressure Ratio: {P_in_2/P_amb:.3f}")
    print(f"Choked: {is_nozzle_choked(P_in_2, P_amb)}")
    
    # Theoretical validation
    print("\n--- THEORETICAL VALIDATION ---")
    print(f"Critical pressure ratio (γ=1.4): {CRITICAL_PR:.3f}")
    print(f"Test 1 PR: {P_in_1/P_amb:.3f} ({'subsonic' if P_in_1/P_amb < CRITICAL_PR else 'choked'})")
    print(f"Test 2 PR: {P_in_2/P_amb:.3f} ({'subsonic' if P_in_2/P_amb < CRITICAL_PR else 'choked'})")
    
    # Thrust calculation example
    print("\n--- THRUST CALCULATION (50 kg/s) ---")
    thrust_1 = compute_thrust_simple(50.0, V_exit_1)
    thrust_2 = compute_thrust_simple(50.0, V_exit_2)
    print(f"Test 1 Thrust: {thrust_1:.0f} N ({thrust_1/1000:.1f} kN)")
    print(f"Test 2 Thrust: {thrust_2:.0f} N ({thrust_2/1000:.1f} kN)")
    
    # Validation checks
    print("\n--- VALIDATION CHECKS ---")
    if M_exit_1 < 1.0:
        print(f"✓ Test 1: Subsonic exit (M < 1) as expected")
    if M_exit_2 == 1.0:
        print(f"✓ Test 2: Sonic exit (M = 1) as expected for choked flow")
    
    if 200 <= V_exit_1 <= 600:
        print(f"✓ Test 1: Velocity within typical subsonic range")
    if 500 <= V_exit_2 <= 800:
        print(f"✓ Test 2: Velocity within typical sonic range")
    
    print("="*60)


if __name__ == "__main__":
    validate_nozzle()