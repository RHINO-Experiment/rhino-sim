
import numpy as np

def Jy_to_TRJ(flux, freqs, omega):
    """
    Convert data in Jy to units of T_RJ (Kelvin).
    
    Parameters:
        flux (array_like):
            Array of shape (Ntimes, Nfreqs) in Jy units.
        freqs (array_like):
            Frequencies, in MHz.
        omega (array_like):
            Beam solid angle (sterad.) as a function of frequency.
    
    Returns:
        Trj (array_like):
            Input flux array rescale to T_RJ units.
    """
    assert len(flux.shape) == 2, "`flux` must have shape (Ntimes, Nfreqs)"
    assert flux.shape[1] == freqs.size, "`flux` must have shape (Ntimes, Nfreqs)"
    assert omega.size == freqs.size, "`omega' must have shape (Nfreqs,)"
    
    # Define constants
    kB = 1.380649e-23 # m^2 kg s^-2 K^-1
    C = 299792458. # m/s
    
    # Perform conversion
    lam = C / (freqs * 1e6) # wavelength, in m
    fac = lam**2. / (2. * kB * omega) * 1e-26
    Trj = flux * fac[np.newaxis,:]
    return Trj


def chebyshev_model(x, ncoeffs, amp):
    """
    This convenience function produces a Chebyshev series model in 
    frequency/time/etc. to produce a model for various components 
    and systematic effects.
    
    Parameters:
        x (array_like):
            Time/freq. variable.
        ncoeffs (int):
            No. of Chebyshev coefficients to use.
        amp (float):
            Overall amplitude of the Chebyshev polynomial.
    
    Returns:
        fn (array_like):
            Chebyshev function. 
    """
    # Chebyshev parameters
    coeffs = amp * np.random.randn(ncoeffs)

    # Construct time- and frequency-dependent random Chebyshev functions
    xx = np.linspace(-1., 1., x.size)
    fn = np.polynomial.chebyshev.Chebyshev(coeffs)(xx)
    
    # Return values
    return fn


def fit_gaussian_process(t, t_samp, y, corrscale1=0.005, corrscale2=0.1, verbose=True):
    """
    For a smooth Gaussian Process to a set of data points. Useful as 
    an interpolator.
    
    The GP kernel is taken to be the sum of a WhiteKernel, and two RBF 
    kernels with different correlation scales.
    
    Parameters:
        t (array_like):
            Set of times/sample points to interpolate onto.
        t_samp (array_like):
            Set of times/points that the input data were measured at.
        y (array_like):
            Values of the input data at the sample points.
        corrscale1, corrscale2 (float):
            Initial guesses for the correlation scales of the two RBF kernels.
        verbose (bool):
            Whether to print basic information about the GP fits.
    
    Returns:
        y_pred (array_like):
            Predicted values of the data at the sample points given by `t'.
        y_samp (array_like):
            Predicted values at the input sample points.
    """
    # Import sklearn functions
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
    
    # Construct kernel
    kernel = WhiteKernel() + RBF(corrscale1) + RBF(corrscale2)

    # Rescale time coords
    X = np.atleast_2d(t_samp[:]).T / np.max(t_samp)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0).fit(X, y)
    if verbose:
        print("GPR score (train): %5.3f" % gpr.score(X, y))

    # Predict for trailing-edge samples
    X2 = np.atleast_2d(times).T / np.max(t_samp)
    y2 = gpr.predict(X2, return_std=False)
    y1 = gpr.predict(X, return_std=False)
    if verbose:
        print(gpr.kernel_)
    return y2, y1
