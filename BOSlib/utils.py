import numpy as np
import BOSlib.shift_utils as ib
from tqdm import tqdm,trange
import matplotlib.pyplot as plt

def shift2angle(shift: np.ndarray, ref_array: np.ndarray, sensor_pitch: float, resolution_of_background: float, Lb: float, Lci: float, binarization : str ="HPfilter", thresh : int = 128, freq : int = 500):
    """
    Convert the background image displacement to the angle of light refraction.

    Parameters
    ----------
    shift : np.ndarray
        Displacement values from the background image.
    ref_array : np.ndarray
        Reference image array used for calculations.
    sensor_pitch : float
        The pitch of the image sensor in mm.
    resolution_of_background : float
        It represents the number of lines per mm.
    Lb : float
        Distance from the background to the object being captured(mm).
    Lci : float
        Distance from the image sensor to the object being captured(mm).
    
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
    tuple
        - angle : np.ndarray
            The calculated angles of light refraction.
        - Lc : float
            The distance from the object to the lens.
        - Li : float
            The distance from the lens to the image sensor.
        - projection_ratio : float
            The ratio of projection based on the dimensions.
    """
    Lb=Lb*10**-3
    Lci=Lci*10**-3
    
    # Size of one LP (in pixels)
    dpLP = ib._cycle(ref_array)

    sensor_pitch = sensor_pitch * 10**-3  # Convert sensor pitch from mm to m
    BGmpLP = 1 / resolution_of_background * 10**-3  # Convert pattern resolution from mm to m

    # Size of one LP on the projection plane (m/LP)
    mpLP = dpLP * sensor_pitch

    # Magnification of the imaging
    projection_ratio = mpLP / BGmpLP

    # Total length
    Lbi = Lci + Lb

    Lc = Lbi / (projection_ratio + 1) - Lb  # Distance from the object to the lens
    Li = Lci - Lc  # Distance from the lens to the image sensor

    # Calculate the angle based on shift and projection properties
    angle = shift * (sensor_pitch) / (projection_ratio * Lb)
    np.nan_to_num(angle, copy=False)  # Replace NaN values with zero in the angle array

    return angle, Lc, Li, projection_ratio

def get_gladstone_dale_constant(temperature, pressure, humidity):
    """
    Calculate the Gladstone-Dale constant based on temperature, pressure, and humidity without using metpy.

    Parameters
    ----------
    temperature : float
        Temperature in degrees Celsius (°C).
    pressure : float
        Pressure in hectopascals (hPa).
    humidity : float
        Humidity as a percentage (%).

    Returns
    -------
    tuple
        - G : float
            The calculated Gladstone-Dale constant.
        - density : float
            The density of the atmosphere.
    """
    
    # Constants
    R_dry = 287.058  # Specific gas constant for dry air, J/(kg·K)
    R_water_vapor = 461.495  # Specific gas constant for water vapor, J/(kg·K)
    
    # Convert input values
    T_kelvin = temperature + 273.15  # Convert temperature to Kelvin
    p_pa = pressure * 100  # Convert pressure to Pascals
    e_saturation = 6.1078 * 10 ** ((7.5 * temperature) / (237.3 + temperature))  # Saturation vapor pressure in hPa
    e_actual = e_saturation * (humidity / 100)  # Actual vapor pressure in hPa
    p_dry = p_pa - e_actual * 100  # Partial pressure of dry air in Pa
    
    # Calculate densities
    density_dry = p_dry / (R_dry * T_kelvin)  # Density of dry air
    density_vapor = (e_actual * 100) / (R_water_vapor * T_kelvin)  # Density of water vapor
    
    # Total density of humid air
    density_air = density_dry + density_vapor
    
    # Gladstone-Dale constant calculation
    n_air = 1.0003  # Refractive index of air
    G = (n_air - 1) / density_air

    return G, density_air

def _compute_laplacian_chunk(array_chunk):
    """
    Compute the Laplacian for a chunk of an array.

    Parameters
    ----------
    array_chunk : ndarray
        A chunk of the original array, assumed to be 3D.

    Returns
    -------
    laplacian_chunk : ndarray
        The Laplacian of the input array chunk.
    """
    grad_yy = np.gradient(array_chunk, axis=1)
    grad_zz = np.gradient(array_chunk, axis=2)
    laplacian_chunk = grad_yy + grad_zz
    return laplacian_chunk

def compute_laplacian_in_chunks(array, chunk_size):
    """
    Compute the Laplacian of a 3D array in chunks to reduce memory usage.

    Parameters
    ----------
    array : ndarray
        The 3D input array for which the Laplacian is calculated.
    chunk_size : int
        The size of each chunk along each dimension.

    Returns
    -------
    laplacian : ndarray
        The computed Laplacian of the input array.
    """
    # Get the shape of the input array
    shape = array.shape
    
    # Create an array to store the result
    laplacian = np.zeros_like(array)
    
    # Process each chunk
    for i in trange(0, shape[0], chunk_size):
        for j in range(0, shape[1], chunk_size):
            for k in range(0, shape[2], chunk_size):
                # Extract the current chunk
                chunk = array[i:i+chunk_size, j:j+chunk_size, k:k+chunk_size]
                
                # Compute the Laplacian for the chunk
                laplacian_chunk = _compute_laplacian_chunk(chunk)
                
                # Store the result in the corresponding position in the original array
                laplacian[i:i+chunk_size, j:j+chunk_size, k:k+chunk_size] = laplacian_chunk
    
    return laplacian


def stripe_generator(width:int,height:int,stripe_width:int):

    """
    Generate a horizontal stripe pattern image and save it as a PNG file.

    Args:
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.
        stripe_width (int): The width of each stripe in pixels.

    Returns:
        None
    """

    # 横縞パターンの生成
    image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        if (i // stripe_width) % 2 == 0:
            image[i, :] = 255  # 白 (モノクロ: 255)

    # 画像を表示
    plt.imshow(image, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()

    # 画像を保存
    plt.imsave(f'horizontal_stripes{stripe_width}px.png', image, cmap='binary')