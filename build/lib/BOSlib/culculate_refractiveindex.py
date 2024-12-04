from tqdm.contrib import tenumerate
from tqdm import tqdm
import numpy as np

def SOR_2D(array_laplacian: np.ndarray,omega_SOR: float,e: float,tolerance:float =1e-24,max_stable_iters:int=1000000):
    """
    Performs the Successive Over-Relaxation (SOR) method on a 2D Laplacian array to solve for steady-state solutions 
    within each slice of the array.

    Parameters
    ----------
    array_laplacian : np.ndarray
        A 3D numpy array where each slice represents a 2D Laplacian grid to be solved.
    omega_SOR : float
        The relaxation factor for the SOR method. Values between 1 and 2 can speed up convergence.
    e : float
        The convergence tolerance threshold for each slice. Iterations stop once the change (`delta`) 
        falls below this threshold.
    tolerance : float, optional
        The tolerance level to determine stability in convergence. Defaults to 1e-24.
    max_stable_iters : int, optional
        The maximum number of stable iterations allowed per slice before termination, regardless of convergence. 
        Defaults to 1000000.

    Returns
    -------
    np.ndarray
        A 3D numpy array containing the steady-state solution `u` for each 2D slice in `array_laplacian`.

    Notes
    -----
    - The SOR method updates each element in the `u` array by considering its neighbors and applying the 
      relaxation factor `omega_SOR`.
    - Boundaries are fixed to zero for each slice, enforcing Dirichlet boundary conditions.
    - Convergence for each slice stops either when `delta` is less than `e` or after a stable count of 
      iterations (determined by `tolerance` and `max_stable_iters`) has been reached.

    Examples
    --------
    >>> laplacian = np.random.rand(10, 100, 100)  # 10 slices of 100x100 grids
    >>> solution = SOR_2D(laplacian, omega_SOR=1.5, e=1e-6)
    >>> print(solution.shape)
    (10, 100, 100)
    """
    Lx=array_laplacian.shape[0]
    Ly=array_laplacian.shape[1]
    Lz=array_laplacian.shape[2]
    delta = 1.0
    n_iter=0
    u_list=[]
    stable_count = 0  # Reset stable count for each batch
    prev_delta = float('inf')  # Initialize previous delta
    for slice_laplacian in tqdm(array_laplacian,desc="slice",leave=True):
        u=np.zeros([Ly,Lz])
        delta = 1.0
        n_iter=0
        while delta > e and stable_count < max_stable_iters:
            u_in=u.copy()
            delta = np.max(abs(u-u_in))
            n_iter+=1
            # Perform SOR update on the inner region
            u[1:-1, 1:-1] = u[1:-1, 1:-1] + omega_SOR * (
                (u_in[2:, 1:-1] + u_in[ :-2, 1:-1] + u_in[1:-1, 2:] + u_in[1:-1, :-2] 
                 + slice_laplacian[1:-1, 1:-1]) / 4 - u[1:-1, 1:-1]
            )

            u[0][:]=0
            u[Ly-1][:]=0
            u[:][0]=0
            u[:][Lz-1]=0

            delta=np.max(abs(u-u_in))            
            # Check if residual change is within tolerance to count as stable
            if abs(delta - prev_delta) < tolerance:
                stable_count += 1
            else:
                stable_count = 0

            prev_delta = delta  # Update previous delta for next iteration

            # Print iteration information
            print("\r", f'Iteration: {n_iter}, Residual: {delta} Stable Count: {stable_count}', end="")

            # Update iteration count
            n_iter += 1

        u_list.append(u)

    return np.array(u_list)