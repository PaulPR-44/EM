import numpy as np
from numpy import ndarray


def electric_field(charge: tuple[float, float, float], x: ndarray, y: ndarray, epsilon_0: float = 8.854187817e-12) -> tuple[float, float]:
    """
    Calculates the electric field vector (Ex, Ey) at point (x, y) due to a single point charge.
    :param charge: Tuple representing the point charge, where the tuple is (charge, x, y).
    :param x: x-coordinate of the point where the field is to be calculated.
    :param y: y-coordinate of the point where the field is to be calculated.
    :param epsilon_0: Permittivity of free space (default: 8.854187817e-12).
    :return: Tuple representing the electric field vector (Ex, Ey) at the given point.
    """

    r = np.sqrt((x - charge[1]) ** 2 + (y - charge[2]) ** 2)
    r = np.where(r == 0, np.inf, r)
    Ex = (charge[0] * (x - charge[1])) / (4 * np.pi * epsilon_0 * r ** 3)
    Ey = (charge[0] * (y - charge[2])) / (4 * np.pi * epsilon_0 * r ** 3)
    return Ex, Ey


def global_electric_field(charges: list[tuple[float, float, float]], x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
    """
    Calculates the total electric field at point (x, y) due to multiple point charges.
    :param charges: List of tuples representing point charges, where each tuple is (charge, x, y).
    :param x: x-coordinate of the point where the field is to be calculated.
    :param y: y-coordinate of the point where the field is to be calculated.
    :return: Tuple representing the electric field vector (Ex, Ey) at the given point.
    """

    Total_Ex = np.zeros_like(x)
    Total_Ey = np.zeros_like(y)

    # Accumulates total electric field components across all charges and grid points
    for charge in charges:
        Ex, Ey = electric_field(charge, x, y)
        Total_Ex += Ex
        Total_Ey += Ey

    return Total_Ex, Total_Ey

def electric_potential(charge: tuple[float, float, float], x: ndarray, y: ndarray, epsilon_0: float = 8.854187817e-12) -> ndarray:
    """
    Calculates the electric potential at point (x, y) due to a single point charge.
    :param charge: Tuple representing the point charge, where the tuple is (charge, x, y).
    :param x: x-coordinate of the point where the potential is to be calculated.
    :param y: y-coordinate of the point where the potential is to be calculated.
    :param epsilon_0: Permittivity of free space (default: 8.854187817e-12).
    :return: Electric potential at the given point.
    """

    r = np.sqrt((x - charge[1]) ** 2 + (y - charge[2]) ** 2)
    r = np.where(r == 0, np.inf, r)
    V = charge[0] / (4 * np.pi * epsilon_0 * r)
    return V

def global_electric_potential(charges: list[tuple[float, float, float]], x: ndarray, y: ndarray) -> ndarray:
    """
    Calculates the total electric potential at each point (x, y) due to multiple point charges.
    :param charges: List of tuples representing point charges, where each tuple is (charge, x, y).
    :param x: x-coordinates of the grid points.
    :param y: y-coordinates of the grid points.
    :return: 2D array representing the electric potential at each grid point.
    """
    V_total = np.zeros_like(x)
    for charge in charges:
        V_total += electric_potential(charge, x, y)
    return V_total