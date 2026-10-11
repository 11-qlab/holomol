"""Species-specific constants for retention calculations.

Values are literature approximations.  Verify against:
  - NIST Diatomic Spectral Database
  - Bialyte et al., "Ultracold molecules" reviews
  - Ni et al. Science 322, 231 (2008) for KRb
  - Park et al. PRL 114, 205302 (2015) for NaK
"""
import numpy as np

KB  = 1.380649e-23       # J/K
H   = 6.62607015e-34     # J*s
AMU = 1.66053906660e-27  # kg

SPECIES = {
    "KRb": {
        "B_rot_GHz": 1.09,
        "dipole_Debye": 0.56,
        "mass_amu": 119.0,
        "polarizability_au": 300.0,
        "sigma_collision_m2": 5e-18,
    },
    "NaK": {
        "B_rot_GHz": 2.83,
        "dipole_Debye": 2.72,
        "mass_amu": 62.0,
        "polarizability_au": 250.0,
        "sigma_collision_m2": 3e-18,
    },
    "SrF": {
        "B_rot_GHz": 7.24,
        "dipole_Debye": 3.47,
        "mass_amu": 107.0,
        "polarizability_au": 200.0,
        "sigma_collision_m2": 2e-18,
    },
}


def debye_to_Cm(mu_D):
    return mu_D * 3.33564e-30


def rotational_temperature(B_GHz):
    return H * B_GHz * 1e9 / KB


def thermal_velocity(T, mass_amu):
    m = mass_amu * AMU
    return np.sqrt(8 * KB * T / (np.pi * m))


def summary():
    print(f"{'species':>6} {'B (GHz)':>10} {'T_rot (K)':>12} "
          f"{'mu (D)':>8} {'m (amu)':>8} {'sigma (m^2)':>12}")
    for name, s in SPECIES.items():
        print(f"{name:>6} {s['B_rot_GHz']:>10.3f} "
              f"{rotational_temperature(s['B_rot_GHz']):>12.4f} "
              f"{s['dipole_Debye']:>8.2f} {s['mass_amu']:>8.1f} "
              f"{s['sigma_collision_m2']:>12.2e}")


if __name__ == "__main__":
    summary()
