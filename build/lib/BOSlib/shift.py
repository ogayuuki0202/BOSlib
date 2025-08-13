from skimage.metrics import structural_similarity as ssm
import numpy as np
from PIL import Image
from scipy import signal
import pandas as pd

import BOSlib.shift_utils as ib


def SSIM(ref_array : np.ndarray, exp_array : np.ndarray):
    """
    Compute the inverted Structural Similarity Index (SSIM) difference matrix between two grayscale images.

    Parameters
    ----------
    ref_array : np.ndarray
        The reference grayscale image array.
    exp_array : np.ndarray
        The experimental grayscale image array.

    Returns
    -------
    np.ndarray
        The inverted SSIM difference matrix, where higher values indicate greater dissimilarity between the two images.
    """
    # Compute the structural similarity matrix (SSM) on the grayscale images
    (score, diff) = ssm(ref_array, exp_array, full=True)
    diff_inv = -diff
    return diff_inv

def SP_BOS(ref_array : np.ndarray, exp_array : np.ndarray, binarization : str ="HPfilter", thresh : int = 128, freq : int = 500):
    """
    Calculate the displacement map of stripe patterns in experimental images using the Background Oriented Schlieren (BOS) method.
    
    This function computes the relative displacement between stripes in a reference and experimental image by compensating for background movement and noise. The displacement map is calculated by processing the images through several steps including image resizing, binarization, boundary detection, noise reduction, displacement calculation, and background compensation.

    Parameters
    ----------
    ref_array : np.ndarray
        The reference grayscale image array. This image represents the original, undisturbed pattern.
        
    exp_array : np.ndarray
        The experimental grayscale image array. This image represents the pattern after deformation due to external factors.
        
    binarization : str, optional, default="HPfilter"
        The method used for binarization of the images. Options are:
        - "thresh" : Use thresholding for binarization.
        - "HPfilter" : Use high-pass filtering for binarization.
        
    thresh : int, optional, default=128
        The threshold value used for binarization when `binarization="thresh"`. Pixels with values above the threshold are set to 1, and those below are set to 0.
        
    freq : int, optional, default=500
        The frequency parameter used for high-pass filtering when `binarization="HPfilter"`.

    Returns
    -------
    np.ndarray
        A 2D array representing the displacement map of the stripe patterns, with background movement compensated. Each value represents the relative displacement between the reference and experimental images, with noise and background displacements removed.

    Notes
    -----
    The method performs the following steps:
    1. Vertically stretches both the reference and experimental images by a factor of 10.
    2. Binarizes the images using either thresholding or high-pass filtering.
    3. Identifies the upper and lower boundaries of the stripes and calculates their centers for both images.
    4. Filters out noise by removing displacements larger than a certain threshold.
    5. Computes the displacement between the stripe centers.
    6. Compensates for background movement by normalizing the displacement map, subtracting the mean displacement over a specified region.
    """
 
    im_ref=Image.fromarray(ref_array)
    im_exp=Image.fromarray(exp_array)

    #streach the image vertivally *10
    im_ref=im_ref.resize((im_ref.size[0],im_ref.size[1]*10))
    im_exp=im_exp.resize((im_exp.size[0],im_exp.size[1]*10))

    ar_ref=np.array(im_ref)
    ar_exp=np.array(im_exp)

    if binarization =="thresh":
        # Binarization
        bin_ref = ib._biner_thresh(ar_ref, thresh)
        bin_exp = ib._biner_thresh(ar_exp, thresh)

        #print("Binarization",bin_ref.shape,bin_exp.shape)
    elif binarization =="HPfilter":
        bin_ref=ib._biner_HP(ar_ref, freq)
        bin_exp=ib._biner_HP(ar_exp, freq)
        #print("Binarization",bin_ref.shape,bin_exp.shape)
    else:
        raise ValueError("Binarization is thresh or HPfilter")
    
    # Detect the coordinates of the color boundaries in the binarized reference image
    ref_u, ref_d = ib._bin_indexer(bin_ref)
    ref_u = np.nan_to_num(ref_u)
    ref_d = np.nan_to_num(ref_d)
    #print("bin_indexer_ref",ref_u.shape,ref_d.shape)
    # Detect the coordinates of the color boundaries in the binarized experimental image
    # u represents the upper boundary of the white stripe, d represents the lower boundary
    exp_u, exp_d = ib._bin_indexer(bin_exp)
    exp_u = np.nan_to_num(exp_u)
    exp_d = np.nan_to_num(exp_d)
    #print("bin_indexer_exp",exp_u.shape,exp_d.shape)

    # Remove data with abnormally large displacements as noise
    ref_u, exp_u = ib._noize_reducer_2(ref_u, exp_u, 10)
    ref_d, exp_d = ib._noize_reducer_2(ref_d, exp_d, 10)
    #print("noize_reducer_2",exp_u.shape,exp_d.shape)
    #print("noize_reducer_2",ref_u.shape,ref_d.shape)
    
    # Combine the upper and lower boundary data to calculate the center of the stripe
    ref = ib._mixing(ref_u, ref_d)
    exp = ib._mixing(exp_u, exp_d)

    #print("mixing",ref.shape,exp.shape)
    
    # Calculate displacement (upward displacement is positive)
    diff = -(exp - ref)
    
    # Rearrange the displacement values into the correct positions and interpolate gaps
    diff_comp = ib._complementer(ref, diff)

    #print("complementer",diff_comp.shape)
    
    # Subtract the overall background movement by dividing by the mean displacement
    diff_comp = diff_comp - np.nanmean(diff_comp[0:1000, 10:100])

    return diff_comp

def S_BOS(ref_array: np.ndarray, exp_array: np.ndarray,freq_sample_row : int = 0):
    """
    Compute a 1D BOS displacement field by estimating phase differences
    between reference and experimental stripe signals.

    This function first identifies the dominant stripe frequency via FFT
    from a representative row (`freq_sample_area`) of `ref_array`. Then for
    each column signal it:
      1. Bandpass-filters around the base frequency.
      2. Normalizes amplitude and applies a sine-based phase correction.
      3. Calculates the local phase difference via lowpass filtered
         sine/cosine products.
      4. Converts phase shifts to physical displacement values.

    Parameters
    ----------
    ref_array : np.ndarray, shape (M, N)
        Reference image signals, with M samples (rows) and N separate
        stripe‐signal columns.
    exp_array : np.ndarray, shape (M, N)
        Experimental image signals matching the dimensions of `ref_array`.
    freq_sample_row : int
        Row index in `ref_array` used to detect the dominant stripe frequency
        for filtering and phase calculation.

    Returns
    -------
    delta_h : np.ndarray, shape (M, N)
        Displacement field (Δh) computed from the phase differences between
        each column of `ref_array` and `exp_array`. Units are cycles/(2π·f),
        where f is the dominant stripe frequency.
    """
    def freq_finder(sig):
        """
        Identify the dominant frequency in the signal using the FFT.
        
        Parameters:
        -----------
        sig : np.ndarray
            1D numpy array representing a signal.
            
        Returns:
        --------
        float
            The dominant frequency (above 0.01 Hz) based on amplitude.
        """
        # Compute FFT frequencies
        freq = np.fft.fftfreq(sig.shape[0])
        # Compute FFT of the signal and normalize the amplitude
        fk = np.fft.fft(sig)
        fk = abs(fk / (sig.shape[0] / 2))
        # Combine frequencies and amplitudes into a DataFrame
        fk_df = pd.DataFrame(np.vstack([freq, fk]).T, columns=["freq", "amp"])
        # Sort DataFrame by frequency and keep only non-negative frequencies
        fk_df = fk_df.sort_values('freq')
        fk_df = fk_df[fk_df["freq"] >= 0]
        # Select frequencies above 0.01 Hz and sort by amplitude in descending order
        freq_search = fk_df[fk_df["freq"] >= 0.01].sort_values('amp', ascending=False)
        # Return the frequency corresponding to the highest amplitude
        return freq_search.iloc[0, 0]

    def bandpass(x, fa, fb):
        """
        Apply a bandpass Butterworth filter to the signal.
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal.
        fa : float
            Lower cutoff frequency multiplier.
        fb : float
            Upper cutoff frequency multiplier.
            
        Returns:
        --------
        np.ndarray
            The bandpass-filtered signal.
        """
        gpass, gstop = 2, 60  # Passband and stopband gains (dB)
        fp, fs = np.array([fa, fb]), np.array([fa / 2, fb * 2])
        fn = 1 / 2  # Nyquist frequency (assuming a normalized sample rate)
        wp, ws = fp / fn, fs / fn
        # Determine the minimum filter order that meets the specifications
        N, Wn = signal.buttord(wp, ws, gpass, gstop)
        # Get the filter coefficients for a Butterworth filter
        b, a = signal.butter(N, Wn, "band")
        # Apply the filter forward and backward to avoid phase distortion
        return signal.filtfilt(b, a, x)

    def lowpass(x, lowcut):
        """
        Apply a lowpass Butterworth filter to the signal.
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal.
        lowcut : float
            The low cutoff frequency.
            
        Returns:
        --------
        np.ndarray
            The lowpass-filtered signal.
        """
        order, nyq = 8, 0.5 * 1  # Order and Nyquist frequency (assuming sample rate = 1)
        low = lowcut / nyq
        # Get the filter coefficients for a lowpass Butterworth filter
        b, a = signal.butter(order, low, btype='low')
        # Apply the filter with zero-phase distortion
        return signal.filtfilt(b, a, x)

    def signal_separate(sig, f1):
        """
        Separate the signal into a constant (mean) component and a bandpass-filtered component.
        
        Parameters:
        -----------
        sig : np.ndarray
            Input signal.
        f1 : float
            Base frequency used to define the bandpass range.
            
        Returns:
        --------
        np.ndarray
            2D array where the first column is the signal mean and the second column is the bandpass-filtered signal.
        """
        sig_f = np.zeros([sig.shape[0], 2])
        # First column: constant value equal to the mean of the signal
        sig_f[:, 0] = sig.mean()
        # Second column: bandpass filtered signal using a frequency window around f1
        sig_f[:, 1] = bandpass(sig, f1 * 0.7, f1 * 1.5)
        return sig_f

    def signal_scale_normalize(sig, f):
        """
        Normalize the signal based on a rolling maximum amplitude and add a sine correction.
        
        Parameters:
        -----------
        sig : np.ndarray
            Input signal.
        f : float
            Frequency used to calculate the sine correction.
            
        Returns:
        --------
        np.ndarray
            The normalized signal.
        """
        # Calculate the rolling maximum absolute value over a window of 0.5/f samples
        sig_abs = np.array(pd.Series(abs(sig)).rolling(int(0.5 / f), center=True).max())
        # Suppress parts of the signal where the amplitude is significantly below the mean
        sig[sig_abs < np.nanmean(sig_abs) * 0.5] = 0
        y = np.arange(0, sig.shape[0], 1)
        # Generate a sine wave for phase correction
        S = np.sin(2 * np.pi * f * y)
        # Create a correction term based on the amplitude threshold
        S1 = (1 - (sig_abs > np.nanmean(sig_abs * 0.5))) * S
        # Add the correction term to the signal
        sig = sig + S1
        # Avoid division by very small numbers
        sig_abs[sig_abs < np.nanmean(sig_abs * 0.5)] = 1
        # Normalize the signal
        sig_norm = sig / sig_abs
        sig_norm[np.isnan(sig_norm)] = 0
        return sig_norm

    def phase_calculate(ref, exp, f1):
        """
        Calculate the phase difference between the reference and experimental signals.
        
        Parameters:
        -----------
        ref : np.ndarray
            Normalized reference signal.
        exp : np.ndarray
            Normalized experimental signal.
        f1 : float
            Base frequency.
            
        Returns:
        --------
        np.ndarray
            The phase difference calculated using lowpass filtered sine and cosine components.
        """
        # Compute sine and its gradient (approximation for cosine)
        sin_ref = ref
        cos_ref = np.gradient(ref) / (f1 * 2 * np.pi)
        # Compute lowpass filtered products to obtain sine and cosine components of the phase difference
        cos_phi = lowpass(sin_ref * exp, f1)
        sin_phi = lowpass(cos_ref * exp, f1)
        # Calculate the phase difference using arctan2 for correct quadrant determination
        return np.arctan2(sin_phi, cos_phi)

    def phase_1DBOS_process(sig_ref, sig_exp, f1):
        """
        Process a pair of reference and experimental signals to compute the phase difference.
        
        Parameters:
        -----------
        sig_ref : np.ndarray
            Single column from the reference signal array.
        sig_exp : np.ndarray
            Corresponding column from the experimental signal array.
        f1 : float
            Base frequency obtained from the reference array.
            
        Returns:
        --------
        np.ndarray
            The phase difference between the processed reference and experimental signals.
        """
        # Separate the signal into mean and bandpass-filtered components and normalize them
        separate_sig_ref = signal_scale_normalize(signal_separate(sig_ref, f1)[:, 1], f1)
        separate_sig_exp = signal_scale_normalize(signal_separate(sig_exp, f1)[:, 1], f1)
        # Calculate the phase difference between the normalized signals
        return phase_calculate(separate_sig_ref, separate_sig_exp, f1)

    # Determine the dominant frequency from a representative column (column 100) of the reference array
    f1 = freq_finder(ref_array[:,freq_sample_row])
    # Initialize a 2D array to store phase differences for each column
    phi_2D = np.zeros([ref_array.shape[0], ref_array.shape[1]]).astype("float64")
    
    # Process each column of the reference and experimental arrays
    for x in range(ref_array.shape[1]):
        phi_2D[:, x] = phase_1DBOS_process(ref_array[:, x], exp_array[:, x], f1)
    
    # Convert phase differences into displacement by dividing by (2*pi*f1)
    delta_h = phi_2D / (2 * np.pi * f1)
    return delta_h
