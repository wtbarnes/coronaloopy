"""
Implementations of various coronal loop scaling laws
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.special import beta, betaincinv

__all__ = ['MartensScalingLaws']

KAPPA_0 = 1e-6 * u.erg / u.cm / u.s * u.K**(-7/2)


class MartensScalingLaws(object):
    """
    Coronal loop scaling laws of [1]_

    Parameters
    ----------
    s : `~astropy.units.Quantity`
        Field-aligned loop coordinate for half of symmetric, semi-circular loop
    heating_constant : `astropy.units.Quantity`
        Constant of proportionality that relates the actual heating rate to the
        scaling with temperature and pressure. The actual units will depend on
        `alpha` and `beta`. See Eq. 2 of [1]_.
    alpha : `float`, optional
        Temperature dependence of the heating rate
    beta : `float`, optional
        Pressure depndence of the heating rate
    gamma : `float`, optional
        Temperature dependence of the radiative loss rate
    chi : `astropy.units.Quantity`, optional
        Constant of proportionality relating the actual radiative losses to the
        scaling with temperature. May need to adjust this based on the value of
        `gamma`.

    References
    ----------
    .. [1] Martens, P., 2010, ApJ, `714, 1290 <http://adsabs.harvard.edu/abs/2010ApJ...714.1290M>`_
    """

    @u.quantity_input
    def __init__(self, s: u.cm, heating_constant, alpha=0, beta=0, gamma=0.5,
                 chi=10**(-18.8) * u.erg * u.cm**3 / u.s * u.K**(0.5)):
        self.s = s
        self.heating_constant = heating_constant
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.chi = chi
        self.chi_0 = self.chi/(4.*(const.k_B**2))

    @property
    @u.quantity_input
    def loop_length(self,) -> u.cm:
        return np.diff(self.s).sum()

    @property
    def x(self,):
        return (self.s/self.loop_length).decompose()

    @property
    @u.quantity_input
    def max_temperature(self,) -> u.K:
        coeff_1 = np.sqrt(KAPPA_0 / self.chi_0 * (3 - 2*self.gamma))/(
            4 + 2*self.gamma + 2*self.alpha)
        coeff_2 = (7/2 + self.alpha)/(3/2 - self.gamma)
        beta_func = beta(self._lambda + 1, 0.5)
        index = self.alpha + 11/4*self.beta + self.gamma*self.beta/2 - 7/2
        return (self.chi_0 * coeff_2 / self.heating_constant
                * (coeff_1*beta_func/self.loop_length)**(2-self.beta))**(1/index)

    @property
    @u.quantity_input
    def temperature(self,) -> u.K:
        beta_term = betaincinv(self._lambda+1, 0.5, self.x.value)**(1./(2 + self.gamma + self.alpha))
        return self.max_temperature * beta_term

    @property
    @u.quantity_input
    def pressure(self,) -> u.dyne/(u.cm**2):
        coeff = np.sqrt(KAPPA_0 / self.chi_0 * (3 - 2*self.gamma))/(4 + 2*self.gamma + 2*self.alpha)
        beta_func = beta(self._lambda + 1, 0.5)
        p_0 = self.max_temperature**((11+2*self.gamma)/4) * coeff * beta_func / self.loop_length
        return np.ones(self.s.shape) * p_0

    @property
    @u.quantity_input
    def heating_rate(self,) -> u.erg/(u.cm**3)/u.s:
        return self.heating_constant * (self.pressure**self.beta) * (self.temperature**self.alpha)

    @property
    def _lambda(self):
        mu = -2*(2+self.gamma)/7
        nu = 2*self.alpha/7
        return (1.-2*nu + mu)/(2*(nu-mu))
