"""
Loop object for holding field-aligned coordinates and quantities
"""
import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst
import sunpy.sun.constants as sun_const

__all__ = ['Loop']


class Loop(object):
    """
    Container for geometric and thermodynamic properties of a coronal loop

    Parameters
    ----------
    coordinate : `astropy.coordinates.SkyCoord`
        Loop coordinates; should be able to transform to HGS
    field_strength : `astropy.units.Quantity`, optional
        Scalar magnetic field strength along the loop. If a scalar value,
        assumed to be uniform over the whole loop.
    cross_sectional_area : `astropy.units.Quantity`, optional
        Circular cross-sectional area along the loop. If a scalar value,
        assumed to be uniform over the whole loop.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> import coronaloopy
    >>> coordinate = SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm, frame='heliographic_stonyhurst', representation_type='cartesian')
    >>> loop = coronaloopy.Loop(coordinate)
    >>> loop
    coronaloopy Loop
    ----------------
    Loop full-length, L : 5.196 Mm
    Footpoints : (1 Mm,2 Mm,3 Mm),(4 Mm,5 Mm,6 Mm)
    Maximum field strength : 200.00 G
    """

    @u.quantity_input
    def __init__(self,
                 coordinate,
                 field_strength: (u.G, None)=None,
                 cross_sectional_area: (u.cm**2, None)=None):
        self.coordinate = coordinate
        self.field_strength = field_strength
        self.cross_sectional_area = cross_sectional_area

    def __repr__(self):
        return f'''coronaloopy Loop
----------------
Loop full-length, L : {self.length.to(u.Mm):.3f}
Footpoints :
    s=0: {self.coordinate[0]}
    s=L: {self.coordinate[-1]}
Maximum field strength : {np.max(self.field_strength):.2f}'''

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, x):
        self._coordinate = x.transform_to(HeliographicStonyhurst)
        self._coordinate.representation_type = 'cartesian'

    @property
    @u.quantity_input
    def cross_sectional_area(self) -> u.cm**2:
        """
        Cross-sectional area as a function of :math:`s`.
        """
        return np.atleast_1d(self._cross_sectional_area) * np.ones(self.field_aligned_coordinate.shape)

    @cross_sectional_area.setter
    def cross_sectional_area(self, x):
        self._cross_sectional_area = np.nan * u.cm**2 if x is None else x

    @property
    @u.quantity_input
    def field_strength(self) -> u.G:
        """
        Magnetic field strength as a function of :math:`s`.
        """
        return np.atleast_1d(self._field_strength) * np.ones(self.field_aligned_coordinate.shape)

    @field_strength.setter
    def field_strength(self, x):
        self._field_strength = np.nan * u.G if x is None else x

    @property
    @u.quantity_input
    def coordinate_direction(self) -> u.dimensionless_unscaled:
        """
        Unit vector indicating the direction of :math:`s` in HEEQ
        """
        grad_xyz = np.gradient(self.coordinate.cartesian.xyz.value, axis=1)
        return grad_xyz / np.linalg.norm(grad_xyz, axis=0)

    @property
    @u.quantity_input
    def field_aligned_coordinate(self) -> u.cm:
        """
        Field-aligned coordinate :math:`s` such that :math:`0<s<L`.
        """
        return np.append(0., np.linalg.norm(np.diff(self.coordinate.cartesian.xyz.value, axis=1),
                                            axis=0).cumsum()) * self.coordinate.cartesian.xyz.unit

    @property
    @u.quantity_input
    def field_aligned_coordinate_norm(self) -> u.dimensionless_unscaled:
        """
        Field-aligned coordinate normalized to the total loop length
        """
        return self.field_aligned_coordinate / self.length

    @property
    @u.quantity_input
    def length(self) -> u.cm:
        """
        Loop full-length :math:`L`, from footpoint to footpoint
        """
        return np.diff(self.field_aligned_coordinate).sum()

    @property
    @u.quantity_input
    def gravity(self) -> u.cm / (u.s**2):
        """
        Gravitational acceleration in the field-aligned direction.
        """
        r_hat = u.Quantity(np.stack([
            np.cos(self.coordinate.spherical.lat)*np.cos(self.coordinate.spherical.lon),
            np.cos(self.coordinate.spherical.lat)*np.sin(self.coordinate.spherical.lon),
            np.sin(self.coordinate.spherical.lat)
        ]))
        r_hat_dot_s_hat = (r_hat * self.coordinate_direction).sum(axis=0)
        return -sun_const.surface_gravity * (
            (sun_const.radius / self.coordinate.spherical.distance)**2) * r_hat_dot_s_hat
