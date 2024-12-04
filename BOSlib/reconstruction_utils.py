from tqdm import tqdm
import numpy as np    

def sinogram_maker_axialsymmetry(angle):
    """
    Generates a sinogram with axial symmetry from a single 2D refractive angle image.

    Parameters
    ----------
    angle : np.ndarray
        A 2D numpy array representing the refractive angle image. Each row in this array 
        is broadcast across the height dimension to simulate an axially symmetric sinogram.

    Returns
    -------
    np.ndarray
        A 3D numpy array representing the sinogram with axial symmetry. Each slice along 
        the first dimension corresponds to a projection at a different angle, where each 
        projection is a symmetric repetition of the refractive angle row values across 
        the height and width dimensions.

    Notes
    -----
    This function assumes axial symmetry for the generated sinogram by replicating each row 
    of the input angle image across both dimensions (height and width) for each slice in 
    the 3D sinogram. The input image is first rotated by 90 degrees for alignment.

    Examples
    --------
    >>> angle_image = np.random.rand(100, 200)  # A 100x200 refractive angle image
    >>> sinogram = sinogram_maker_axialsymmetry(angle_image)
    >>> print(sinogram.shape)
    (200, 200, 200)
    """
    # Rotate the angle image by 90 degrees
    angle = np.rot90(angle)
    height = angle.shape[1]
    
    # Initialize an empty 3D array for the sinogram
    sinogram = np.empty((angle.shape[0], height, height), dtype=angle.dtype)

    # Loop through each row in the rotated angle image
    for i, d_angle in enumerate(tqdm(angle)):
        # Broadcast each row across the height to create a symmetric 2D projection
        sinogram[i] = np.broadcast_to(d_angle[:, np.newaxis], (height, height))
        
    return sinogram
