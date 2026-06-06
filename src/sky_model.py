
import numpy as np
import pygdsm
import astropy_healpix

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time

from hera_sim import io
import hera_sim, pyuvsim, matvis
from hera_sim.visibilities import VisibilitySimulation, ModelData, MatVis

def noise_realisation(Tsys, freqs, dt):
    """
    Generate a random thermal noise realisation (mean zero) using 
    the radiometer equation, for a given system temperature.
    
    Parameters:
        Tsys (array_like):
            System temperature, in K.
        freqs (array_like):
            Frequency array, in MHz.
        dt (float):
            Integration time per time sample, in sec.
    
    Returns:
        noise (array_like):
            Random noise realisation, in K.
    """
    # Channel bandwidth
    dnu = (freqs[1] - freqs[0]) * 1e6 # Hz
    
    # Noise rms
    sigmaT = Tsys / np.sqrt(dnu * dt)
    
    # Generate noise
    noise = sigmaT * np.random.randn(*Tsys.shape)
    return noise


def gsm_sky_model(freqs, resolution="hi", nside=None):
    """
    Return a pyradiosky SkyModel object populated with a Global Sky Model datacube in 
    healpix format.
    Parameters
    ----------
    freqs : array_like
        Frequency array, in Hz.
    resolution : str, optional
        Whether to use the high or low resolution pygdsm maps. Options are 'hi' or 'low'.
    nside : int, optional
        Healpix nside to up- or down-sample the GSM sky model to. Default: `None` (use the 
        default from `pygdsm`, which is 1024).
    Returns
    -------
    sky_model : pyradiosky.SkyModel
        SkyModel object.
    """
    # Initialise GSM object
    gsm = pygdsm.GlobalSkyModel16(data_unit="TRJ", resolution=resolution, freq_unit="Hz")

    # Construct GSM datacube
    hpmap = gsm.generate(freqs=freqs) # FIXME: nside=1024, ring ordering, galactic coords
    hpmap_units = "K"

    # Set nside or resample
    nside_gsm = int(astropy_healpix.npix_to_nside(hpmap.shape[-1]))
    if nside is None:
        # Use default nside from pygdsm map
        nside = nside_gsm
    else:
        # Transform to a user-selected nside
        hpmap_new = np.zeros((hpmap.shape[0], astropy_healpix.nside_to_npix(nside)), 
                             dtype=hpmap.dtype)
        for i in range(hpmap.shape[0]):
            hpmap_new[i,:] = hp.ud_grade(hpmap[i,:], 
                                         nside_out=nside, 
                                         order_in="RING", 
                                         order_out="RING")
        hpmap = hpmap_new

    # Get datacube properties
    npix = astropy_healpix.nside_to_npix(nside)
    indices = np.arange(npix)
    history = "pygdsm.GlobalSkyModel2016, data_unit=TRJ, resolution=low, freq_unit=MHz"
    freq = u.Quantity(freqs, "hertz")

    # hmap is in K
    stokes = u.Quantity(np.zeros((4, len(freq), len(indices))), hpmap_units)
    stokes[0] = hpmap * u.Unit(hpmap_units)

    # Construct pyradiosky SkyModel
    sky_model = pyradiosky.SkyModel(
                                    nside=nside,
                                    hpx_inds=indices,
                                    stokes=stokes,
                                    spectral_type="full",
                                    freq_array=freq,
                                    history=history,
                                    frame="galactic",
                                    hpx_order="ring"
                                )

    sky_model.healpix_interp_transform(frame='icrs', full_sky=True, inplace=True) # do coord transform
    assert sky_model.component_type == "healpix"
    return sky_model


def empty_uvdata(ants=None, nfreq=20, ntimes=20, bandwidth=0.2e8, 
                 integration_time=40., 
                 start_time=2458902.33333, start_freq=1.e8, **kwargs):
    """
    Generate empty UVData object with the right shape.
    
    Parameters
    ----------
    ants (dict): None
        A dictionary mapping an integer to a three-tuple of ENU co-ordinates for
        each antenna. These antennas can be down-selected via keywords.
    ntimes : int, optional
        Number of time samples. Default: 20.
    
    bandwidth : float
        Total bandwidth, in Hz. Default: 0.2e8
    
    integration_time : float, optional
        Integration time per time sample. Default: 40. 
    
    start_time : float, optional
        Start date of observations, as Julian date. Default: 2458902.33333 
        (20:00 UTC on 2020-02-22)
    
    start_freq : float, optional
        Initial frequency channel, in Hz. Default: 1.e8.
    
    **kwargs : args
        Other arguments to be passed to `hera_sim.io.empty_uvdata`.
    
    Returns
    -------
    uvd : UVData
        Returns an empty UVData 
    """
    uvd = io.empty_uvdata(
        Nfreqs=nfreq,
        start_freq=start_freq,
        channel_width=bandwidth / nfreq,
        start_time=start_time,
        integration_time=integration_time,
        Ntimes=ntimes,
        array_layout=ants,
        **kwargs
    )
    
    # Add missing parameters
    #uvd._x_orientation.value = 'east'
    return uvd


def simulate_diffuse(freqs, beam_list, nside=64):
    
    # Construct a data model
    cfg_spec = dict( nfreq=freqs.size,
                     start_freq=freqs[0]*1e6,
                     bandwidth=(freqs[-1]-freqs[0])*1e6,
                     start_time=jd,
                     integration_time=dt,
                     ntimes=Ntimes)
    
    # Prepare empty data structure
    uvd = empty_uvdata(ants=ants, **cfg_spec)
    
    # FIXME: This can use a lot of memory in MPI mode, as there are Nprocs duplicates 
    # of the whole datacube!
    # Build SkyModel from GSM (pygdsm)
    gsm_sky = gsm_sky_model(np.unique(uvd.freq_array), resolution="lo", nside=nside)
    
    # Prepare model
    data_model = ModelData(uvdata=uvd, 
                           sky_model=gsm_sky,
                           beams=beam_list)
    
    # Initialise MatVis handler object
    matvis = MatVis(precision=2)

    # Create a VisibilitySimulation object
    simulator_diffuse = VisibilitySimulation(data_model=data_model, 
                                             simulator=matvis)
    
    # Run the simulation
    tstart = time.time()
    simulator_diffuse.simulate()
    print("\tSimulation (diffuse) took %2.1f sec" % (time.time() - tstart))
    
    # Get simulated data
    _d = uvd.get_data((0,0))
    
    # Convert from Jy to 
    assert uvd.vis_units == 'Jy'
    return _d
