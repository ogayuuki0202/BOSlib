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

def S_BOS(ref_array : np.ndarray, exp_array : np.ndarray):
    def freq_finder(sig):
        freq = np.fft.fftfreq(sig.shape[0])
        fk = np.fft.fft(sig)
        fk = abs(fk / (sig.shape[0] / 2))
        fk_df = pd.DataFrame(np.vstack([freq, fk]).T, columns=["freq", "amp"])
        fk_df = fk_df.sort_values('freq')
        fk_df = fk_df[fk_df["freq"] >= 0]
        freq_search = fk_df[fk_df["freq"] >= 0.01].sort_values('amp', ascending=False)
        return freq_search.iloc[0, 0]

    def bandpass(x, fa, fb):
        gpass, gstop = 2, 60
        fp, fs = np.array([fa, fb]), np.array([fa / 2, fb * 2])
        fn = 1 / 2
        wp, ws = fp / fn, fs / fn
        N, Wn = signal.buttord(wp, ws, gpass, gstop)
        b, a = signal.butter(N, Wn, "band")
        return signal.filtfilt(b, a, x)

    def lowpass(x, lowcut):
        order, nyq = 8, 0.5 * 1
        low = lowcut / nyq
        b, a = signal.butter(order, low, btype='low')
        return signal.filtfilt(b, a, x)

    def signal_separate(sig, f1):
        sig_f = np.zeros([sig.shape[0], 2])
        sig_f[:, 0] = sig.mean()
        sig_f[:, 1] = bandpass(sig, f1 * 0.7, f1 * 1.5)
        return sig_f

    def signal_scale_normalize(sig, f):
        sig_abs = np.array(pd.Series(abs(sig)).rolling(int(0.5 / f), center=True).max())
        sig[sig_abs < np.nanmean(sig_abs) * 0.5] = 0
        y = np.arange(0, sig.shape[0], 1)
        S = np.sin(2 * np.pi * f * y)
        S1 = (1 - (sig_abs > np.nanmean(sig_abs * 0.5))) * S
        sig = sig + S1
        sig_abs[sig_abs < np.nanmean(sig_abs * 0.5)] = 1
        sig_norm = sig / sig_abs
        sig_norm[np.isnan(sig_norm)] = 0
        return sig_norm

    def phase_calculate(ref, exp, f1):
        sin_ref, cos_ref = ref, np.gradient(ref) / (f1 * 2 * np.pi)
        cos_phi, sin_phi = lowpass(sin_ref * exp, f1), lowpass(cos_ref * exp, f1)
        return np.arctan2(sin_phi, cos_phi)

    def phase_1DBOS_process(sig_ref, sig_exp, f1):
        separate_sig_ref = signal_scale_normalize(signal_separate(sig_ref, f1)[:, 1], f1)
        separate_sig_exp = signal_scale_normalize(signal_separate(sig_exp, f1)[:, 1], f1)
        return phase_calculate(separate_sig_ref, separate_sig_exp, f1)

    f1 = freq_finder(ref_array[:, 100])
    phi_2D = np.zeros([ref_array.shape[0], ref_array.shape[1]]).astype("float64")
    
    for x in range(ref_array.shape[1]):
        phi_2D[:, x] = phase_1DBOS_process(ref_array[:, x], exp_array[:, x], f1)
    
    delta_h = phi_2D / (2 * np.pi * f1)
    return delta_h
