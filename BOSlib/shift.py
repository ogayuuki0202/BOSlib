from skimage.metrics import structural_similarity as ssm
import numpy as np
from PIL import Image
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

        print("Binarization",bin_ref.shape,bin_exp.shape)
    elif binarization =="HPfilter":
        bin_ref=ib._biner_HP(ar_ref, freq)
        bin_exp=ib._biner_HP(ar_exp, freq)
        print("Binarization",bin_ref.shape,bin_exp.shape)
    else:
        raise ValueError("Binarization is thresh or HPfilter")
    
    # Detect the coordinates of the color boundaries in the binarized reference image
    ref_u, ref_d = ib._bin_indexer(bin_ref)
    ref_u = np.nan_to_num(ref_u)
    ref_d = np.nan_to_num(ref_d)
    print("bin_indexer_ref",ref_u.shape,ref_d.shape)
    # Detect the coordinates of the color boundaries in the binarized experimental image
    # u represents the upper boundary of the white stripe, d represents the lower boundary
    exp_u, exp_d = ib._bin_indexer(bin_exp)
    exp_u = np.nan_to_num(exp_u)
    exp_d = np.nan_to_num(exp_d)
    print("bin_indexer_exp",exp_u.shape,exp_d.shape)

    # Remove data with abnormally large displacements as noise
    ref_u, exp_u = ib._noize_reducer_2(ref_u, exp_u, 10)
    ref_d, exp_d = ib._noize_reducer_2(ref_d, exp_d, 10)
    print("noize_reducer_2",exp_u.shape,exp_d.shape)
    print("noize_reducer_2",ref_u.shape,ref_d.shape)
    
    # Combine the upper and lower boundary data to calculate the center of the stripe
    ref = ib._mixing(ref_u, ref_d)
    exp = ib._mixing(exp_u, exp_d)

    print("mixing",ref.shape,exp.shape)
    
    # Calculate displacement (upward displacement is positive)
    diff = -(exp - ref)
    
    # Rearrange the displacement values into the correct positions and interpolate gaps
    diff_comp = ib._complementer(ref, diff)

    print("complementer",diff_comp.shape)
    
    # Subtract the overall background movement by dividing by the mean displacement
    diff_comp = diff_comp - np.nanmean(diff_comp[0:1000, 10:100])

    return diff_comp
