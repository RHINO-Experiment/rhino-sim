import numpy as np

def get_beam_area(beam, freqs, naz=200, nza=300, beam_squared=False):
    
    # Regular grid in azimuth and zenith angle
    _az = np.linspace(0., 2.*np.pi, naz)
    _za = np.linspace(0., 0.5*np.pi, nza)
    az, za = np.meshgrid(_az, _za)
    
    # Get pixel sizes
    pix_az = _az[1] - _az[0]
    pix_za = _za[1] - _za[0]
    
    # Interpolate beam vs frequency
    # FIXME: Check whether normalisation is necessary
    #beam.peak_normalize()
    b = beam.efield_eval(az_array=az.flatten(), za_array=za.flatten(), freq_array=freqs*1e6)
    
    # Reshape to (Nfreqs, Naz, Nza)
    b = b[0,1,:,:].copy()
    b = b.reshape((freqs.size, az.shape[0], az.shape[1]))
    if beam_squared:
        b *= b
    
    # Multiply by sin(alt) and then sum to approximate integral
    b *= np.sin(0.5*np.pi - za)[np.newaxis,:,:]
    area = np.sum(b, axis=(1,2)) * (pix_az * pix_za)
    return area
