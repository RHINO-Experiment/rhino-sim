import numpy as np

def apply_noise_wave_params(Tsrc, Gamma_src, Gamma_rec, Tunc, Tcos, Tsin, Toffset):
    """
    Apply the parameters of the noise-wave formalism to an input source temperature.
    Units of the T parameters are K. All quantities are expected to have shape 
    (Ntimes, Nfreqs.)
    
    See Eq. 6 of https://arxiv.org/abs/2011.14052 for definitions.
    
    Parameters:
        Tsrc (array_like):
            True ("calibrated") temperature from the input source.
        Gamma_src (array_like):
            Complex reflection coefficient from the input source.
        Gamma_rec (array_like):
            Complex reflection coefficient to the receiver.
        Tunc (array_like):
            Temperature of the uncorrelated part of the reflection.
        Tcos (array_like):
            Cosine component of the correlated part of the reflection.
        Tsin (array_like):
            Sine component of the correlated part of the reflection.
        Toffset(array_like):
            Overall offset temperature in the signal chain.
    
    Returns:
        Prec (array_like):
            Observed power after passing through the signal chain.
    """
    # Complex kappa factor
    kappa = np.sqrt(1. - np.abs(Gamma_rec)**2.) / (1. - Gamma_src * Gamma_rec)
    
    # Calculate sum of noise wave effects
    Prec = Tsrc * (1. - np.abs(Gamma_src)**2.) * np.abs(kappa)**2. \
         + Tunc * np.abs(Gamma_src)**2. * np.abs(kappa)**2. \
         + Tcos * np.real(Gamma_src * kappa) \
         + Tsin * np.imag(Gamma_src * kappa) \
         + Toffset
    return Prec
