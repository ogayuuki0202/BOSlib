import numpy as np
from PIL import Image
import pandas as pd
from scipy import signal

def _biner_thresh(ar_in: np.ndarray, thresh: int) -> np.ndarray:
    """
    Binarize an array based on a threshold value.

    Parameters
    ----------
    ar_in : np.ndarray
        Input array to be binarized.
    thresh : int
        Threshold value for binarization.

    Returns
    -------
    np.ndarray
        Binarized array where values above the threshold are `True` and values below are `False`.
    """
    ar_bin = ar_in > thresh
    return ar_bin

def _bin_indexer(ar_in: np.ndarray) -> tuple:
    """
    Detect color boundary coordinates in a binarized image by finding gradient edges.

    Parameters
    ----------
    ar_in : np.ndarray
        Input binarized image array.

    Returns
    -------
    tuple
        Arrays containing the y-coordinates for the detected upper and lower boundaries of stripes.
    """
    # Convert to int for differentiation
    ar_in = ar_in.astype(np.int8)
    ar2 = np.delete(ar_in, 0, 0)
    ar3 = np.delete(ar_in, ar_in.shape[0] - 1, 0)
    ar4 = ar2 - ar3
    
    # Detect positive gradient (upper boundary)
    u_tuple = np.where(ar4 > 0)
    u_index = np.stack([u_tuple[1], u_tuple[0]]).T
    
    # Convert to DataFrame for processing
    df = pd.DataFrame(u_index, columns=["X", "Y"])
    df = df.pivot(index="Y", columns="X", values="Y")
    
    # Group by index for averaging boundaries
    df["index"] = df.index / 10
    df["index"] = df[["index"]].astype(int)
    df = df.groupby("index").mean() / 10
    df.columns = range(df.shape[1])
    u_index = np.array(df)
    
    # Initialize array to store boundary positions
    u_index_2 = np.zeros([1000, ar_in.shape[1]])
    
    # Process each column to handle NaN values
    for x in range(ar_in.shape[1]):

        ar_loop=u_index[:,x][np.where(~np.isnan(u_index[:,x]))[0]]
        u_index_2[:,x]=np.concatenate([ar_loop,np.full(1000-ar_loop.shape[0],np.nan)])
    
    # Detect negative gradient (lower boundary)
    d_tuple = np.where(ar4 < 0)
    d_index = np.stack([d_tuple[1], d_tuple[0]]).T
    
    # Process for lower boundary similar to upper
    df = pd.DataFrame(d_index, columns=["X", "Y"])
    df = df.pivot(index="Y", columns="X", values="Y")
    df["index"] = df.index / 10
    df["index"] = df[["index"]].astype(int)
    df = df.groupby("index").mean() / 10
    df.columns = range(df.shape[1])
    d_index = np.array(df)
    
    # Initialize array for lower boundary positions
    d_index_2 = np.zeros([1000, ar_in.shape[1]])
    
    # Process each column to fill NaNs
    for x in range(ar_in.shape[1]):
        ar_loop = d_index[:, x][~np.isnan(d_index[:, x])]
        d_index_2[:, x] = np.concatenate([ar_loop, np.full(1000 - ar_loop.shape[0], np.nan)])
    
    return u_index_2, d_index_2

def _noize_reducer_2(ar_ref: np.ndarray, ar_exp: np.ndarray, diff_thresh: int) -> tuple:
    """
    Remove noise by aligning arrays based on a displacement threshold.

    Parameters
    ----------
    ar_ref : np.ndarray
        Reference array.
    ar_exp : np.ndarray
        Experimental array.
    diff_thresh : int
        Threshold for detecting displacement.

    Returns
    -------
    tuple
        Arrays with noise filtered based on the displacement threshold.
    """
    for x in range(ar_ref.shape[1]):
        ref = ar_ref[:, x]
        exp = ar_exp[:, x]
        
        while np.any(abs(exp - ref) > diff_thresh):
            if np.any(exp - ref > diff_thresh):
                y = np.where(exp - ref > diff_thresh)[0].min()
                ref = np.delete(ref, y)
                ref = np.insert(ref, ref.shape[0], np.nan)
            
            if np.any(exp - ref < -diff_thresh):
                y = np.where(exp - ref < -diff_thresh)[0].min()
                exp = np.delete(exp, y)
                exp = np.insert(exp, exp.shape[0], np.nan)
        
        ar_ref[:, x] = ref
        ar_exp[:, x] = exp
    
    return ar_ref, ar_exp

def _mixing(u_ar: np.ndarray, d_ar: np.ndarray) -> np.ndarray:
    """
    Calculate the center positions between upper and lower boundaries.

    Parameters
    ----------
    u_ar : np.ndarray
        Array of upper boundary coordinates.
    d_ar : np.ndarray
        Array of lower boundary coordinates.

    Returns
    -------
    np.ndarray
        Array with center positions between boundaries.
    """
    ar = np.full([u_ar.shape[0] * 2, u_ar.shape[1]], np.nan)
    ar[::2] = u_ar
    ar[1::2] = d_ar
    ar2 = np.delete(ar, 0, 0)
    ar3 = np.delete(ar, ar.shape[0] - 1, 0)
    ar = (ar2 + ar3) / 2
    
    return ar

def _complementer(ref_ar: np.ndarray, diff_ar: np.ndarray) -> np.ndarray:
    """
    Rearrange displacement data to correct positions and interpolate gaps.

    Parameters
    ----------
    ref_ar : np.ndarray
        Reference array containing stripe positions.
    diff_ar : np.ndarray
        Array of displacement values.

    Returns
    -------
    np.ndarray
        Compensated displacement array with interpolated gaps.
    """
    max_ar = int(np.nanmax(ref_ar))
    diff_2 = np.vstack([np.full(max_ar, -1), range(max_ar), np.zeros(max_ar)]).T
    
    for x in range(ref_ar.shape[1]):
        ar_loop = np.vstack([np.full_like(ref_ar[:, x], x), ref_ar[:, x], diff_ar[:, x]]).T
        ar_loop = ar_loop[~np.isnan(ar_loop).any(axis=1)]
        diff_2 = np.concatenate([diff_2, ar_loop])
    
    diff_2[:, 1] = diff_2[:, 1].astype(int)
    diff_df = pd.DataFrame(diff_2)
    diff_df = diff_df.pivot_table(columns=0, index=1, values=2)
    diff_df = diff_df.interpolate(limit=50)
    
    diff_comp = diff_df.values
    
    return diff_comp

def _stretch_image_vertically(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Stretch a grayscale image vertically by a given scale factor.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image as a 2D numpy array.
    scale_factor : int
        The factor by which to stretch the image vertically.

    Returns
    -------
    np.ndarray
        Vertically stretched image.
    """
    # Verify that the input image is 2D
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    # Vertically stretch the image by repeating each row 'scale_factor' times
    stretched_image = np.repeat(image, scale_factor, axis=0)

    return stretched_image

def _cycle(ref_array: np.ndarray):
    """
    Calculate the cycle length of stripes in a reference image.

    Parameters
    ----------
    ref_array : np.ndarray
        The reference image to be analyzed, provided as a 2D numpy array.

    Returns
    -------
    float
        The calculated cycle length based on the detected boundaries in the reference image.
    """
    
    # Vertically stretch the reference image by a factor of 10
    im_ref=Image.fromarray(ref_array)
    im_ref=im_ref.resize((im_ref.size[0],im_ref.size[1]*10))
    
    # Convert the stretched image to a numpy array
    ar_ref = np.array(im_ref)
    
    # Binarize the stretched image using a threshold of 128
    bin_ref = _biner_thresh(ar_ref, 128)
    
    # Detect upper and lower boundaries in the binarized image
    ref_u, ref_d = _bin_indexer(bin_ref)
    
    # Mix the upper and lower boundary coordinates to find the midpoints
    ref = _mixing(ref_u, ref_d)
    
    # Calculate the intervals between midpoints by finding differences
    ref_interbal = np.delete(ref, 0, 0) - np.delete(ref, ref.shape[0] - 1, 0)
    
    # Count the number of valid intervals (non-NaN values)
    count = np.count_nonzero(~np.isnan(ref_interbal[:, 0]))
    
    # Trim the intervals array to exclude unnecessary values
    ref_interbal = ref_interbal[0:count - 2]
    
    # Calculate the cycle length as twice the average interval length
    # This accounts for both peaks and valleys in the detected boundaries
    cycle = np.nanmean(ref_interbal) * 2
    
    return cycle

def _biner_HP(ar_in,cycle):
    # パラメータ設定
    sample_freq = 1#サンプリング周波数[1/px]
    cutoff_freq = 1/cycle#カットオフ周波数[1/px]
    filter_order = 8#次数
    
    # フィルタの作成
    sos_high = signal.butter(filter_order, cutoff_freq, 'highpass', output='sos', fs=sample_freq)
    #sos_low = signal.butter(filter_order, cutoff_freq, 'lowpass', output='sos', fs=sample_freq)
    
    #答え格納用ndarrayの作成(符号付16bitを指定しないと勝手に符号なしにされるZO☆)
    ar_h=np.zeros_like(ar_in).astype(np.int16)
    
    #1行ずつ処理
    for x in range(ar_in.shape[1]):
        ar_x=ar_in[:,x]
        
        #フィルタの適用
        ar_h[:,x] = signal.sosfiltfilt(sos_high, ar_x)
        #yf_l = signal.sosfiltfilt(sos_low, y)
    
    #plt.plot(yf_h[100:1500])
    ar_bin=ar_h>0
    #y_bin=pd.DataFrame(y_bin*1)
    
    
    
    #f ar_bin[0,:].sum()==0 or ar_bin[0,:].sum()==ar_bin.shape[1]:
    first_state=ar_bin[0,0]
    for x in range(ar_bin.shape[1]):
        y=0
        state=ar_bin[0,x]!=first_state
        
        while(state==False):
            ar_bin[y,x]=1-ar_bin[y,x]
            y=y+1
            state=ar_bin[y,x]!=first_state
            
        
    #上端の汚い部分をごまかす
    x=0
    count=np.nan
    while(count!=0):
        count=ar_bin[x-1,:].sum()
        ar_bin[x,:]=0
        x=x+1
    
    return ar_bin
