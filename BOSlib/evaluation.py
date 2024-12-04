import open3d as o3d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import statistics
import mpl_toolkits.axes_grid1

class CalculateDiff:
    """
    A class to calculate the difference between two point clouds in terms of neighbor densities.

    Attributes
    ----------
    output_path : str
        Path where the resulting point cloud with differences will be saved.
    r : float
        Radius for neighbor point sampling in KDTree.

    Methods
    -------
    diff(pcl1, pcl2)
        Calculates and visualizes the relative density differences between two point clouds.
    ectraction_neighborpoints(pointcloud, target_positions)
        Extracts the density of neighboring points around specified target positions.
    """
    def __init__(self, output_path: str, r: float) -> None:
        """
        Initializes the CalculateDiff class.

        Parameters
        ----------
        output_path : str
            Path to save the output point cloud.
        r : float
            Radius for neighbor point sampling in KDTree.
        """
        self.outputpath = output_path
        self.r = r

    def diff(self, pcl1, pcl2):
        """
        Calculates the relative density difference between two point clouds.

        The method computes the difference in neighbor densities for points in two point clouds.
        It normalizes the differences and clips them for visualization, then creates a new point
        cloud to save the results.

        Parameters
        ----------
        pcl1 : open3d.geometry.PointCloud
            The first point cloud.
        pcl2 : open3d.geometry.PointCloud
            The second point cloud.

        Returns
        -------
        open3d.geometry.PointCloud
            The point cloud representing the relative density differences.
        """
        # Initialize the output point cloud
        diff_pointcloud = o3d.geometry.PointCloud()
        
        # Extract point positions from the input point clouds
        positions_pcl1 = np.array(pcl1.points)
        positions_pcl2 = np.array(pcl2.points)

        # Use the sparser point cloud for density calculation
        if positions_pcl1.shape[0] < positions_pcl2.shape[0]:
            ground_position = positions_pcl1
        else:
            ground_position = positions_pcl2

        # Compute neighbor densities for each point cloud
        density_pcl1 = self.ectraction_neighborpoints(pcl1, ground_position)
        density_pcl2 = self.ectraction_neighborpoints(pcl2, ground_position)
        density_diff = density_pcl1 - density_pcl2

        # Convert to relative error
        density_diff_relative = 100 * np.divide(
            np.abs(density_diff),
            np.array(density_pcl1)
        )
        
        # Clip relative differences to a maximum of 100
        density_diff_relative = np.clip(density_diff_relative, 0, 100)

        # Apply the differences to the output point cloud
        diff_pointcloud.normals = o3d.utility.Vector3dVector(density_diff_relative)
        diff_pointcloud.points = o3d.utility.Vector3dVector(ground_position)

        # Normalize density differences and map them to RGB values
        RGB, minval, maxval = _normalize(density_diff)
        diff_pointcloud.colors = o3d.utility.Vector3dVector(np.array(RGB))

        # Save the resulting point cloud
        o3d.io.write_point_cloud(self.outputpath, diff_pointcloud, format='pts', compressed=True)

        return diff_pointcloud

    def ectraction_neighborpoints(self, pointcloud, target_positions):
        """
        Extracts the density of neighbor points for given target positions in a point cloud.

        This function uses KDTree for efficient neighbor search.

        Parameters
        ----------
        pointcloud : open3d.geometry.PointCloud
            The input point cloud.
        target_positions : numpy.ndarray
            Array of positions to search for neighbors.

        Returns
        -------
        numpy.ndarray
            Array of densities (number of neighbor points) for each target position.
        """
        # Create a KDTree for neighbor point search
        kdtree = o3d.geometry.KDTreeFlann(pointcloud)
        radius = self.r  # Radius for neighbor search
        
        all_indices = []  # List to store indices of neighbors
        for pos in tqdm(target_positions, desc="Extracting neighbor points"):
            [k, idx, _] = kdtree.search_radius_vector_3d(pos, radius)
            if np.asarray(idx).shape[0] == 0:
                index = [0]
            elif np.asarray(idx).shape[0] == 1:
                index = idx
            else:
                index = [np.asarray(idx)[0]]
            all_indices.extend([index])

        # Extract neighbor densities
        neighbor_density = np.asarray(pointcloud.normals)[all_indices, :][:, 0]
        neighbor_density_array = np.asarray(neighbor_density)
        return neighbor_density_array
  
def _normalize(self,data):
    """
    Min-Maxスケーリングを使用してデータを正規化します。
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data, min_val, max_val
    
def viewer(
    pointcloud_path: str, vmax: float, vcentre: float, vmin: float, 
    color: str, unit_colorbar: str, unit_xy: str, rho: float
) -> None:
    """
    Visualizes a point cloud with color-coded density values as a scatter plot.

    Parameters
    ----------
    pointcloud_path : str
        Path to the point cloud file to be loaded.
    vmax : float
        Maximum value for the color scale. If None, it is set to the maximum of the normalized density.
    vcentre : float
        Center value for the color scale.
    vmin : float
        Minimum value for the color scale.
    color : str
        Colormap to use for visualization.
    unit_colorbar : str
        Label for the colorbar indicating the unit of the density values.
    unit_xy : str
        Label for the x and y axes indicating the unit of the coordinates.
    rho : float
        Normalization factor for density values.

    Returns
    -------
    None
        Displays a scatter plot of the point cloud with density visualized as colors.

    Notes
    -----
    The density values are normalized by `rho`, and their statistics (max, min, mean, median) 
    are printed to the console. The point cloud's x and y coordinates are used for the scatter plot.
    """
    # Load the point cloud
    pointcloud = o3d.io.read_point_cloud(pointcloud_path)

    # Extract coordinates and density values
    x = np.asarray(pointcloud.points)[:, 0]
    y = np.asarray(pointcloud.points)[:, 1]
    density = np.asarray(pointcloud.normals)[:, 0]
    density_nondim = density / rho  # Normalize density by rho

    # Configure color normalization for the scatter plot
    if vmax is None:
        norm = Normalize(vmin=density_nondim.min(), vmax=density_nondim.max())
    else:
        # Use a TwoSlopeNorm for customized vmin, vcenter, and vmax
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcentre, vmax=vmax)

    # Create figure and axes for the scatter plot
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    # Plot the scatter plot
    sc = ax.scatter(x, y, c=density_nondim, cmap=color, s=1, norm=norm)
    ax.set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Add a colorbar to the plot
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad=0.1)
    cbar = plt.colorbar(sc, ax=ax, cax=cax, orientation='vertical')
    cbar.set_label(unit_colorbar)  # Set colorbar label

    # Set axis labels and titles
    ax.set_xlabel(unit_xy)
    ax.set_ylabel(unit_xy)

    # Display the plot
    plt.show()

class array2pointcloud:
    """
    Converts a 2D array into a 3D point cloud with associated density and color information.

    Parameters
    ----------
    px2mm_y : float
        Conversion factor from pixels to millimeters along the y-axis.
    px2mm_x : float
        Conversion factor from pixels to millimeters along the x-axis.
    array : np.array
        Input 2D array representing pixel data.
    ox : int
        Origin offset in pixels along the x-axis.
    oy : int
        Origin offset in pixels along the y-axis.
    outpath : str
        Path where the resulting point cloud file will be saved.
    Flip : bool
        Whether to flip the array horizontally.

    Attributes
    ----------
    data_px : np.array
        The original or flipped array data in pixel units.
    data_mm : np.array
        Transformed data with coordinates in millimeter units and density as the fourth column.
    points : np.array
        3D points representing the x, y, and z coordinates.
    density : np.array
        Density values expanded for storing as normals.
    RGB : np.array
        RGB color values derived from the density data.

    Methods
    -------
    __call__()
        Executes the conversion process and saves the resulting point cloud.
    px2mm_method()
        Converts the pixel coordinates and density values to millimeter units.
    reshaper()
        Extracts and reshapes the 3D points, density, and RGB values.
    set_array()
        Assembles the data into an Open3D PointCloud object.
    """
    def __init__(self, px2mm_y: float, px2mm_x: float, array: np.array, 
                 ox: int, oy: int, outpath: str, Flip: bool) -> None:
        self.px2mm_x = px2mm_x
        self.px2mm_y = px2mm_y
        self.data_px = array
        self.ox = ox
        self.oy = oy
        self.output_path = outpath
        self.flip = Flip

        # Initialize placeholders for processed data
        self.data_mm = None
        self.points = None
        self.density = None
        self.RGB = None

    def __call__(self):
        """
        Executes the full conversion pipeline and saves the result as a point cloud.
        """
        self.px2mm_method()
        self.reshaper()
        pcd = self.set_array()
        o3d.io.write_point_cloud(self.output_path, pcd, format='pts', compressed=True)

    def px2mm_method(self):
        """
        Converts pixel-based coordinates to millimeter-based coordinates.

        Notes
        -----
        If `self.flip` is True, the input array is flipped horizontally. The resulting
        millimeter-based coordinates and density values are stored in `self.data_mm`.
        """
        if self.flip:
            self.data_px = np.fliplr(self.data_px)

        data_list = []
        for i in range(self.data_px.shape[0]):
            for j in range(self.data_px.shape[1]):
                # Calculate millimeter coordinates and append density value
                point = [float(self.px2mm_x * (j - self.ox)),
                         float(self.px2mm_y * (i - self.oy)),
                         0.0,  # z-coordinate is 0
                         float(self.data_px[i, j])]
                data_list.append(point)

        self.data_mm = np.array(data_list)

    def reshaper(self):
        """
        Reshapes the transformed data into points, density, and RGB values.

        Notes
        -----
        Density values are tiled to create normals for the point cloud.
        The `nm` function is used to normalize density values into RGB colors.
        """
        self.points = self.data_mm[:, :3]  # Extract 3D coordinates
        self.density = np.tile(self.data_mm[:, 3], (3, 1)).T  # Expand density values
        colors, _, _ = _normalize(self.density)  # Normalize density to RGB
        self.RGB = np.array(colors)

    def set_array(self):
        """
        Creates an Open3D PointCloud object from the processed data.

        Returns
        -------
        pcd : o3d.geometry.PointCloud
            The resulting point cloud object with points, colors, and normals.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.RGB)
        pcd.normals = o3d.utility.Vector3dVector(self.density)

        return pcd
