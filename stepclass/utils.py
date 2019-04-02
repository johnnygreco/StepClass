"""
Random utilities 
"""
from __future__ import division, print_function

import os
import numpy as np
import yaml
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from collections import namedtuple

project_dir = os.path.dirname(os.path.dirname(__file__))

pixscale = 0.168
zpt = 27.0

#Extinction correction factor for HSC
#A_lambda = Coeff * E(B-V)
ExtCoeff = namedtuple('ExtCoeff', 'g r i z y')
ext_coeff = ExtCoeff(g=3.233, r=2.291, i=1.635, z=1.261, y=1.076)


def read_config(fn):
    """
    Parse hugs pipeline configuration file.
    """
    with open(fn, 'r') as f:
        config_params = yaml.load(f)
    return config_params


def get_dust_map():
    """
    Use sdfmap to get E(B-V) values from the Schlegel, Finkbeiner & Davis 
    (1998) dust map.

    Notes 
    -----
    sfdmap can be downloaded here http://github.com/kbarbary/sfdmap.
    """
    import sfdmap
    dustmap = sfdmap.SFDMap()
    return dustmap


def check_random_state(seed):
    """
    Turn seed into a `numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : `None`, int, list of ints, or `numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.

    Returns
    -------
    random_state : `numpy.random.RandomState`
        RandomState object.

    Notes
    -----
    This routine is adapted from scikit-learn.  See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """
    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if type(seed)==list:
        if type(seed[0])==int:
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def check_kwargs_defaults(kwargs, defaults):
    """
    Build keyword argument by changing a default set of parameters.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments that are different for default values.
    defaults : dict
        The default parameter values.

    Returns
    -------
    kw : dict
        A new keyword argument.
    """
    kw = defaults.copy()
    for k, v in kwargs.items():
        kw[k] = v
    return kw


def measure_psf_fwhm(psf, fwhm_guess):
    """
    Fit a 2D Gaussian to observed psf to estimate the FWHM

    Parameters
    ----------
    psf : ndarray
        Observed psf
    fwhm_guess : float
        Guess for fwhm in pixels

    Returns
    -------
    mean_fwhm : float
        Mean x & y FWHM
    """
    sigma = fwhm_guess * gaussian_fwhm_to_sigma
    g_init = models.Gaussian2D(psf.max()*0.3,
                               psf.shape[1]/2,
                               psf.shape[0]/2,
                               sigma)
    fit_g = fitting.LevMarLSQFitter()
    xx, yy = np.meshgrid(np.arange(psf.shape[1]), np.arange(psf.shape[0]))
    best_fit = fit_g(g_init, xx, yy, psf)
    mean_fwhm = np.mean([best_fit.x_fwhm, best_fit.y_fwhm])
    return mean_fwhm
