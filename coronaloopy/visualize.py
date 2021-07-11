"""
Visualizaition functions related to 1D fieldlines
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy.map import GenericMap, make_fitswcs_header
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_earth
import sunpy.sun.constants as sun_const

__all__ = ['set_ax_lims', 'plot_fieldlines']


def is_visible(coords, observer):
    """
    Create mask of coordinates not blocked by the solar disk.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        Helioprojective oordinates of the object(s) of interest
    observer : `~astropy.coordinates.SkyCoord`
        Heliographic-Stonyhurst Location of the observer
    """
    theta_x = coords.Tx
    theta_y = coords.Ty
    distance = coords.distance
    rsun_obs = ((sun_const.radius / (observer.radius - sun_const.radius)).decompose()
                * u.radian).to(u.arcsec)
    off_disk = np.sqrt(theta_x**2 + theta_y**2) > rsun_obs
    in_front_of_disk = distance - observer.radius < 0.

    return np.any(np.stack([off_disk, in_front_of_disk], axis=1), axis=1)


def set_ax_lims(ax, xlim, ylim, smap):
    """
    Set limits on a `~sunpy.map.Map` plot
    """
    x_lims, y_lims = smap.world_to_pixel(
        SkyCoord(xlim, ylim, frame=smap.coordinate_frame))
    ax.set_xlim(x_lims.value)
    ax.set_ylim(y_lims.value)


def plot_fieldlines(*coords,
                    image_map=None,
                    observer=None,
                    check_visible=True,
                    draw_grid=True,
                    **kwargs):
    """
    Plot fieldlines on the surface of the Sun

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        List of fieldline coordinates
    observer : `~astropy.coordinates.SkyCoord`, optional
        Position of the observer. If None, defaults to position of Earth at the
        current time
    check_visible : `bool`, optional
        If True, mask coordinates that are obscured by the solar disk as determined
        by the observer location
    draw_grid : `bool`, optional
        If True, draw the HGS grid

    Other Parameters
    ----------------
    plot_kwargs : `dict`
        Additional parameters to pass to `~matplotlib.pyplot.plot` when
        drawing field lines.
    grid_kwargs : `dict`
        Additional parameters to pass to `~sunpy.map.Map.draw_grid`
    imshow_kwargs : `dict`
        Additional parameters to pass to `~sunpy.map.Map.plot`
    """
    plot_kwargs = kwargs.get('plot_kwargs', {})
    grid_kwargs = kwargs.get('grid_kwargs', {})
    imshow_kwargs = kwargs.get('imshow_kwargs', {})
    imshow_kwargs['alpha'] = 0 # make the dummy map transparent
    # Create a dummy map for some specified observer location
    data = np.ones((1000, 1000))  # make this big to give more fine-grained control in w2pix
    observer = get_earth(Time.now()) if observer is None else observer
    coord = SkyCoord(Tx=0*u.arcsec,
                     Ty=0*u.arcsec,
                     frame=Helioprojective(observer=observer, obstime=observer.obstime))
    meta = make_fitswcs_header(data, coord, scale=(1, 1)*u.arcsec/u.pixel,)
    image_map = GenericMap(data, meta)
    # Plot coordinates
    fig = kwargs.get('fig', plt.figure())
    ax = fig.add_subplot(111, projection=image_map)
    image_map.plot(axes=ax, **imshow_kwargs)
    for coord in coords:
        c = coord.transform_to(image_map.coordinate_frame)  # is this needed?
        if check_visible:
            c = c[is_visible(c, image_map.observer_coordinate)]
        if len(c) == 0:
            continue  # Matplotlib throws exception when no points are visible
        ax.plot_coord(c, **plot_kwargs)
    if draw_grid:
        image_map.draw_grid(axes=ax, **grid_kwargs)

    return fig, ax, image_map
