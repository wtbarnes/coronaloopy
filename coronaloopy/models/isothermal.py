"""
Implementations of various coronal loop scaling laws
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
import sunpy.sun.constants as sun_const
from scipy.integrate import cumtrapz

__all__ = ['Isothermal']


class Isothermal(object):
    """
    Hydrostatic loop solutions for an isothermal atmosphere

    Parameters
    ----------
    s : `~astropy.units.Quantity`
        Field-aligned loop coordinate
    r : `~astropy.units.Quantity`
        Radial distance as a function of `s`
    temperature : `~astropy.units.Quantity`
    pressure0 : `~astropy.units.Quantity`
        Pressure at :math:`r=R_{\odot}`
    """

    @u.quantity_input
    def __init__(self, s: u.cm, r: u.cm, temperature: u.K, pressure0: u.dyne/u.cm**2):
        self.s = s
        self.r = r
        self.temperature = temperature
        self.pressure0 = pressure0

    @property
    def _integral(self):
        # Add points to the front in the case that s[0] does not
        # correspond to R_sun as we do not know the initial value
        # at that point
        r = np.append(const.R_sun, self.r)
        s = np.append(-np.diff(self.s)[0], self.s)
        integrand = 1/r**2 * np.gradient(r) / np.gradient(s)
        # Integrate over the whole loop
        return cumtrapz(integrand.to('cm-2').value, s.to('cm').value) / u.cm

    @property
    @u.quantity_input
    def pressure(self) -> u.dyne / u.cm**2:
        return self.pressure0 * np.exp(-const.R_sun**2 / self.pressure_scale_height * self._integral)

    @property
    @u.quantity_input
    def pressure_scale_height(self) -> u.cm:
        return 2 * const.k_B * self.temperature / const.m_p / sun_const.equatorial_surface_gravity

    @property
    @u.quantity_input
    def density(self) -> u.cm**(-3):
        return self.pressure / (2*const.k_B*self.temperature)
