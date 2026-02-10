"""
A script that reads in gravitational wave psi 4 data and black hole positional data,
applies spin-weighted spherical harmonics to the data, and creates a Mayavi animation
of the black holes and their gravitational waves. At each state, the black holes are 
moved to their respective positions and the render is saved as a .png file.
"""

# --- ADJUSTABLE PARAMETERS FOR VISUALIZATION ---
# Feel free to modify to change the appearance of the movie/rendering process!

USE_SYS_ARGS = True # Change to turn on/off default parameters. Leave on if you want to input your own data.
STATUS_MESSAGES = True # Change to turn on/off status reports during rendering
TRAJECTORY_LINES = False # Change to turn on/off lines tracking trajectories of black holes
PIP_VIEW = False # Change to turn on/off picture in picture view at corner of movie showing close up of black holes
FREQ_SOUND = True # Change to turn on/off background sound based on strain frequency
APPARENT_HORIZONS = False # Change to turn on/off accurate horizon rendering. Use only if you have data.

import os
import sys
import time
import psutil
from math import erf
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.special import erf  # Vectorized error function for arrays
from scipy.integrate import quad
from scipy.spatial import ConvexHull
from scipy.signal import hilbert
from scipy.signal import resample
import quaternionic
import spherical
import imageio.v2 as imageio
import vtk  # Unused, but Required by TVTK.
from tvtk.api import tvtk
from mayavi import mlab
from mayavi.api import Engine
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.sources.parametric_surface import ParametricSurface
from mayavi.modules.surface import Surface
from mayavi.modules.scalar_cut_plane import ScalarCutPlane
import psi4_FFI_to_strain as psi4strain
import traceback
from scipy.io.wavfile import write as write_wav
from moviepy import VideoFileClip, AudioFileClip

# Default parameters used when USE_SYS_ARGS is False
BH_DIR = "../data/GW150914_data/r100" # changeable with sys arguments
MOVIE_DIR = "../data/GW150914_data/movies" # changeable with sys arguments
S_MODE = -2
EXT_RAD = 100 # changeable with sys arguments

def swsh_summation_angles(
    colat: float,
    azi: NDArray[np.float64],
    mode_data: NDArray[np.complex128],
    ell_min: int,
    ell_max: int,
    status_messages=True
) -> NDArray[np.complex128]:
    """
    Sum all the strain modes after factoring in the
    corresponding spin-weighted spherical harmonic
    for the specified angles in the mesh. Stored as an array corresponding to [angle, time] indices.

    This version is optimized to avoid creating the large intermediate
    (n_modes, n_pts, n_times) array by looping over the modes
    and summing the contributions directly.

    :param colat: Colatitude angle for the SWSH factor.
    :param azi: Azimuthal angles for the SWSH factor.
    :param mode_data: Numpy array containing strain data for all the modes, shape (n_modes, n_times).
    :param ell_min: Minimum l mode to use in SWSH
    :param ell_max: Maximum l mode to use in SWSH
    :return: A complex valued numpy array of the superimposed wave, shape (n_azi_pts, n_times).
    """

    quat_arr = quaternionic.array.from_spherical_coordinates(colat, azi)
    winger = spherical.Wigner(ell_max, ell_min)
    # Create an swsh array shaped like (n_modes, n_quaternions)
    swsh_arr = winger.sYlm(S_MODE, quat_arr).T

    # Get shapes of inputs
    n_modes, n_times = mode_data.shape
    n_pts = azi.shape[0]

    # Pre-allocate the *final* result array, which fits in memory
    result = np.zeros((n_pts, n_times), dtype=np.complex128)

    # Loop over the modes (the axis we are summing over) with a progress bar
    for i in range(n_modes):
        # Get the contribution for this single mode
        # mode_data[i, :] has shape (n_times,)
        # swsh_arr[i, :, np.newaxis] has shape (n_pts, 1)
        # Their product broadcasts to (n_pts, n_times)
        contribution = swsh_arr[i, :, np.newaxis] * mode_data[i, :]

        # Add this mode's contribution to the total sum
        result += contribution

        # Update status
        if status_messages:
            # Calculate progress based on the outer loop index 'azi_idx'
            progress = (i + 1) / (n_modes) * 100
            # Use f-string formatting
            print(f"\rProgress: {progress:.1f}% completed", end="", flush=True)

    if status_messages:
        print() # Print newline after status messages

    return result

def interpolate_coords_by_time(
    old_times: NDArray[np.float64],
    e1: NDArray[np.float64],
    e2: NDArray[np.float64],
    e3: NDArray[np.float64],
    new_times: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Interpolate the 3D coordinates to the given time states.

    :param old_times: 1D array of original time values.
    :param e1: 1D array of the first coordinate values corresponding to old_times.
    :param e2: 1D array of the second coordinate values corresponding to old_times.
    :param e3: 1D array of the third coordinate values corresponding to old_times.
    :param new_times: 1D array of new time values to interpolate to.
    :return: A tuple of three 1D arrays representing the interpolated coordinates (e1, e2, e3) at new_times.

    DocTests:
    >>> old_times = np.array([0., 1., 2.])
    >>> e1 = np.array([0., 1., 4.])
    >>> e2 = np.array([0., 2., 5.])
    >>> e3 = np.array([0., 3., 6.])
    >>> new_times = np.array([0.5, 1.5])
    >>> interpolate_coords_by_time(old_times, e1, e2, e3, new_times)
    (array([0.5, 2.5]), array([1. , 3.5]), array([1.5, 4.5]))
    """

    # Create a single interpolator for all coordinates
    interpolator = interp1d(old_times, np.vstack((e1, e2, e3)), fill_value="extrapolate")

    # Interpolate all coordinates at once
    new_e1, new_e2, new_e3 = interpolator(new_times)

    return new_e1, new_e2, new_e3

def initialize_tvtk_grid(num_azi: int, num_radius: int) -> Tuple:
    """
    Set initial parameters for the mesh generation module and return
    a circular, polar mesh with manipulation objects to write and save data.

    :param num_azi: Number of azimuthal points on the mesh.
    :param num_radius: Number of radial points on the mesh.
    :returns: tvtk.FloatArray for strain data,
              tvtk.UnstructuredGrid representing the mesh topology,
              tvtk.Points holding the mesh coordinates.

    DocTests:
    >>> strain_array, grid, points = initialize_tvtk_grid(3, 4)
    >>> isinstance(strain_array, tvtk.FloatArray)
    True
    >>> isinstance(grid, tvtk.UnstructuredGrid)
    True
    >>> isinstance(points, tvtk.Points)
    True
    """

    # Create tvtk objects
    points = tvtk.Points()
    grid = tvtk.UnstructuredGrid()
    strain_array = tvtk.FloatArray(
        name="Strain", number_of_components=1, number_of_tuples=num_azi * num_radius
    )

    # Precompute next_i for all azimuthal points
    next_i_list = [(i + 1) % num_azi for i in range(num_azi)]

    # Precompute cell connectivity data
    cells_data = []
    for j in range(num_radius - 1):
        this_j = j * num_azi
        next_j = (j + 1) * num_azi
        for i in range(num_azi):
            ni = next_i_list[i]
            cells_data.extend([4, i + this_j, ni + this_j, ni + next_j, i + next_j])

    # Convert to VTK cell array
    cell_data = np.array(cells_data, dtype=np.int64)
    id_array = tvtk.IdTypeArray()
    id_array.from_array(cell_data)

    cell_array = tvtk.CellArray()
    cell_array.set_cells((num_radius - 1) * num_azi, id_array)

    # Configure grid
    quad_cell_type = tvtk.Quad().cell_type
    grid.set_cells(quad_cell_type, cell_array)

    return strain_array, grid, points

def create_gw(
    engine: Any, # Mayavi Engine instance expected
    grid: tvtk.UnstructuredGrid,
    color: Tuple[float, float, float],
    display_radius: int,
    wireframe: bool = False,
) -> None:
    """
    Create and display a gravitational wave strain from a given grid.

    :param engine: Mayavi engine for visualization.
    :param grid: tvtk.UnstructuredGrid representing the strain data.
    :param color: Color of the strain as an RGB tuple (0, 0, 0) to (1, 1, 1).
    :param display_radius: Controls the visible radius for wireframe contours.
    :param wireframe: Whether to display the strain as a wireframe with contours.

    DocTests: Requires a running Mayavi engine, difficult to test standalone. Will be skipped.
    """

    # Get the current scene efficiently
    scene = getattr(engine, "current_scene", engine.scenes[0])

    # Create and configure data source
    gw = VTKDataSource(data=grid)
    engine.add_source(gw, scene)

    # Configure surface visualization
    surface = Surface()
    engine.add_filter(surface, gw)
    surface.actor.mapper.scalar_visibility = False
    surface.actor.property.color = color

    def gen_contour(coord: NDArray, normal: NDArray) -> None:
        """Helper function to generate a contour plane."""
        contour = ScalarCutPlane()
        engine.add_filter(contour, gw)
        contour.implicit_plane.widget.enabled = False
        contour.implicit_plane.plane.origin = coord
        contour.implicit_plane.plane.normal = normal
        contour.actor.property.line_width = 5
        contour.actor.property.opacity = 0.5

    if wireframe:
        # Generate all contour parameters in vectorized operations
        wire_intervals = np.linspace(-display_radius, display_radius, 14)

        # Precompute all coordinates and normals
        x_coords = np.column_stack((wire_intervals, np.zeros_like(wire_intervals), np.zeros_like(wire_intervals)))
        y_coords = np.column_stack((np.zeros_like(wire_intervals), wire_intervals, np.zeros_like(wire_intervals)))
        all_coords = np.vstack((x_coords, y_coords))

        # Create normals using broadcasting
        all_normals = np.repeat([[1, 0, 0], [0, 1, 0]], len(wire_intervals), axis=0)

        # Process all contours in a single loop
        for coord, normal in zip(all_coords, all_normals):
            gen_contour(coord, normal)

def create_sphere(
    engine: Engine,
    radius: float = 1,
    color: tuple[float, float, float] = (1, 0, 0)
) -> Surface:
    """
    Create and display a spherical surface with the given parameters.

    :param engine: Mayavi engine for visualization.
    :param radius: Radius of the sphere.
    :param color: Color of the sphere as an RGB tuple (0, 0, 0) to (1, 1, 1).
    :return: The Surface object representing the sphere.

    DocTests: Requires a running Mayavi engine, difficult to test standalone. Will be skipped.
    """

    # Use the current scene
    scene = engine.current_scene if hasattr(engine, "current_scene") else engine.scenes[0]

    # Create and configure the parametric surface
    ps = ParametricSurface(function="ellipsoid")
    ps.parametric_function.x_radius = radius
    ps.parametric_function.y_radius = radius
    ps.parametric_function.z_radius = radius

    # Add the parametric surface to the engine
    engine.add_source(ps, scene)

    # Apply surface visualization and configure properties
    s = Surface()
    engine.add_filter(s, ps)
    s.actor.mapper.scalar_visibility = False
    s.actor.property.color = color

    return s

def dhms_time(seconds: float) -> str:
    """
    Convert a given number of seconds into a string indicating the remaining time.

    :param seconds: Number of seconds.
    :return: A string indicating the remaining time (days, hours, minutes).

    DocTests:
    >>> dhms_time(90061)
    '1 day 1 hour 1 minute'
    >>> dhms_time(7200)
    '2 hours'
    >>> dhms_time(59)
    ''
    >>> dhms_time(3665)
    '1 hour 1 minute'
    """

    divisors = (
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
    )
    parts = []
    remaining = seconds
    for divisor, label in divisors:
        value = int(remaining // divisor)
        remaining = remaining % divisor
        if value > 0:
            parts.append(f"{value} {label}{'s' if value != 1 else ''}")
    return " ".join(parts)

def compute_strain_to_mesh(
    strain_azi: NDArray[np.float32],
    equal_times: NDArray[np.float32], # Not used directly, but lerp_times depends on it
    radius_values: NDArray[np.float32], # Not used directly, but lerp_times depends on it
    lerp_times: NDArray[np.float32],  # Precomputed 2D array of interpolation points
    time_array: NDArray[np.float32],
    dropoff_2D_flat: NDArray[np.float32],
    use_symlog: bool,
    mmap_filename: str,
    status_messages=STATUS_MESSAGES
) -> np.memmap:
    """
    Interpolates strain data onto a polar mesh grid over specific time points.

    This function takes precomputed strain data, originally defined over a set
    of time points (`time_array`) for various azimuths (`strain_azi`), and
    interpolates it onto a target spatio-temporal grid. The spatial grid
    is implicitly polar (radius and azimuth), and the target time points
    vary with radius, as defined by `lerp_times`.

    The interpolation applies a radial dropoff factor (`dropoff_2D_flat`) and
    optionally uses a symmetric logarithmic scaling (`symlog`) on the strain
    values before interpolation.

    To manage memory, especially for a large number of azimuth points, the
    interpolation is performed in chunks along the azimuth dimension.


    :param strain_azi: Precomputed strain values, summed over modes. Shape: (n_azi_pts, n_original_times).
    :param equal_times: Array of equally spaced times for the final desired mesh state. Shape: (n_times,). Note:
                        Not used directly in calculations, but typically used to generate `lerp_times` and defines the
                        size of the time dimension in the output.
    :param radius_values: Array of radius values for the mesh grid. Shape: (n_rad_pts,). Note: Not used directly in
                          calculations, but typically used to generate `lerp_times` and defines the size of the radius
                          dimension in the output.
    :param lerp_times: Precomputed 2D array of interpolation time points. These are the target time coordinates (`x'`
                       values for `np.interp`) at which to evaluate the interpolated strain. Shape: (n_rad_pts,
                       n_times).
    :param time_array: Original time points corresponding to the `strain_azi` data (the `x` coordinates for `np.interp`).
                       Shape: (n_original_times,).
    :param dropoff_2D_flat: Array of scaling factors applied to the strain data, typically dependent on radius.
                            Applied before interpolation. Shape: (n_rad_pts,).
    :param mmap_filename: The file path to create the memory-mapped file for the output.
    :param use_symlog: If True, apply a symmetric logarithmic transformation to the strain data before interpolation
    :param status_messages: If True, print progress messages and estimated chunk size.
    :return: A numpy memmap containing the interpolated strain values on the spatio-temporal mesh.
             Shape: (n_rad_pts, n_azi_pts, n_times).
    :throws RuntimeError: If the available memory reported by psutil is 0, the code cannot proceed.
    """

    # Get the size of each array
    n_rad_pts = len(radius_values)
    n_azi_pts = strain_azi.shape[0]
    n_times = len(equal_times)

    # Estimate chunk size based on available memory (heuristic)
    chunk_size = min(int(psutil.virtual_memory().available / 70000000), n_azi_pts)
    if chunk_size == 0:
        raise RuntimeError("Not enough memory to begin mesh calculations. Reopen your terminal and try again.")

    if status_messages:
         print(f"Using chunk size: {chunk_size} for azimuth interpolation.")

    strain_to_mesh = np.memmap(mmap_filename, dtype=np.float32, mode='w+', shape=(n_rad_pts, n_azi_pts, n_times))

    for start_idx in range(0, n_azi_pts, chunk_size):
        end_idx = min(start_idx + chunk_size, n_azi_pts)

        # Interpolate each azimuth in the current chunk
        for azi_idx in range(start_idx, end_idx):
            # Use np.interp for each radial profile corresponding to this azimuth
            # The values to interpolate *from* are strain_azi[azi_idx, :] at times time_array
            # The points to interpolate *to* are the times given by lerp_times[:, time_idx] for each time_idx
            # Result shape for one azi_idx should be (n_rad_pts, n_times)
            for rad_idx in range(n_rad_pts):
                strain_to_mesh[rad_idx, azi_idx, :] = np.interp(
                    lerp_times[rad_idx, :], # Time coordinates to interpolate to
                    time_array, # Original time coordinates (shape n_original_times)
                    # Original strain values (shape n_original_times) scaled with symlog (if enabled) and dropoff factors
                    dropoff_2D_flat[rad_idx] * (np.sign(strain_azi[azi_idx, :]) * np.log1p(np.abs(strain_azi[azi_idx, :])) if use_symlog else strain_azi[azi_idx, :])
                )

            # Update status
            if status_messages:
                # Calculate progress based on the outer loop index 'azi_idx'
                progress = (azi_idx + 1) / (n_azi_pts) * 100
                # Use f-string formatting
                print(f"\rProgress: {progress:.1f}% completed", end="", flush=True)

    # Flush mmap to disk
    strain_to_mesh.flush()

    # Create a new line after status messages complete
    if status_messages:
        print() # Move to the next line

    return strain_to_mesh

def get_amplitude_scale_factor(swsh_array: NDArray[np.float64],
                               azimuth_values: NDArray[np.float64],
                               r_omitted: float,
                               camera_dist_max: float,
                               camera_elevation_max: float,
                               camera_azi: float
) -> float:
    """
    Calculates the factor by which the strain data should be scaled, based on limiting camera parameters and the
    size of the hole generated in the mesh for the black holes. The method will calculate the amplitude scale factor
    such that the peak of the gravitational waves, from the perspective of the camera, touch the edge of the hole in
    the center of the mesh. This is to ensure that the maximum scaling for the strain is used for visual effect without
    impeding the view of the black holes in the center. For more explanation on the algorithm used, visit the
    GitHub documentation.

    :param swsh_array: Array of strain over time at each azimuth angle. Shape: (n_azi, n_times)
    :param azimuth_values: Array of azimuth values from 0 to 360. Shape: (n_azi)
    :param r_omitted: Radius of the central hole in the mesh, calculated from black hole positions at start
    :param camera_dist_max: Maximum distance the camera zooms out during the simulation
    :param camera_elevation_max: Maximum elevation the camera achieves (minimum angle between it and the horizon)
    :param camera_azi: Azimuth angle of the camera (fixed)l
    :return: A float by which to scale the strain data for optimal waveform size

    DocTests:
    >>> swsh_array = np.array([[0.841470984808,0.917921410456,1.33501520964,-5.39406678297,9.23284682067],
    ...                        [0.841470984808,1.00285846158,1.59350892505,-7.03426538413,13.1544351329],
    ...                        [0.841470984808,0.959638368983,1.45911802242,-6.16343231814,11.0292008758]])
    >>> azimuth_values = np.array([0, 0.1, 6.2])
    >>> get_amplitude_scale_factor(swsh_array, azimuth_values, 5, 20, np.pi / 4, 0)
    np.float64(0.5435312594361945)
    """

    dropoff_radius = 1.875 * r_omitted # Point at which strain is cut off using erf, plus a little more due to curvature

    # Maximum distance from the camera to the center in the xy plane
    xy_camera_dist = camera_dist_max * np.sin(camera_elevation_max)

    # Coordinates (x, y) at which the camera viewline intersects a circle with the dropoff radius
    x_viewline_intersects_dropoff = ((xy_camera_dist**2 / r_omitted) - np.sqrt(dropoff_radius**2 + (dropoff_radius * \
                                    xy_camera_dist / r_omitted)**2 - xy_camera_dist**2)) / ((xy_camera_dist / \
                                    r_omitted)**2 + 1)
    y_viewline_intersects_dropoff = xy_camera_dist * (1 - (x_viewline_intersects_dropoff / r_omitted))

    # Azimuth values that should be considered in amplitue scaling, consider only the ones where strain could impede view
    azi_scan_bound = np.pi / 2 - np.arctan(y_viewline_intersects_dropoff / x_viewline_intersects_dropoff)

    # Camera view of the center hole is elliptic, calculate semimajor and semiminor axes lengths of this view
    view_semiminor_axis = (camera_dist_max * y_viewline_intersects_dropoff * np.cos(camera_elevation_max)) / \
                          xy_camera_dist - (camera_dist_max * np.cos(camera_elevation_max) * (dropoff_radius - \
                          r_omitted)) / (xy_camera_dist - r_omitted)
    view_semimajor_axis = r_omitted * (1 - y_viewline_intersects_dropoff / xy_camera_dist)

    # Apply a mask for azimuth values, ensuring within the scan bound (special case for camera_azi = 0 or 2*pi
    view_condit = (((azimuth_values > 2 * np.pi - azi_scan_bound) & (azimuth_values < 2 * np.pi)) | \
                  ((azimuth_values >= 0) & (azimuth_values < azi_scan_bound))) if (camera_azi == 0 or camera_azi == 2 * \
                  np.pi) else ((camera_azi - azi_scan_bound <= azimuth_values) & (azimuth_values < camera_azi + \
                  azi_scan_bound))
    valid_azi_idx = np.where(view_condit)

    # A loop to calculate the minimum scale factor allowable across all azis (since the strain max is limiting)
    min_scale_factor = float('inf')

    for idx in valid_azi_idx[0]:
        # Using the semimajor and semiminor axes, calculate a bound line that the waves cannot intersect and find its
        # value at a certain azi
        z_max_azi = camera_dist_max * y_viewline_intersects_dropoff * np.cos(camera_elevation_max) / xy_camera_dist - \
                    view_semiminor_axis * np.sqrt(1 - (dropoff_radius * np.sin(azimuth_values[idx] - camera_azi) / \
                    view_semimajor_axis)**2)

        # Calculate a trial amplitude scale factor, based on the maximum strain over time along this azimuth
        factor = z_max_azi / np.max(np.abs(swsh_array[idx, :]))

        # If the factor is less than the minimum, it becomes the minimum
        if factor < min_scale_factor:
            min_scale_factor = factor

    return min_scale_factor

def find_idx(array: NDArray[np.float64], value: float) -> NDArray[np.int64]:
    """
    Finds all indexes where a value could be inserted into an array without causing a break in trends (increasing,
    decreasing, constant).

    :param array: The numpy array to search for the value's proper position(s) should it occur within trends
    :value: The float value for whose place to search for
    :return: A numpy array containing any indexes where value could be inserted into array without changing trends

    DocTests:
    >>> find_idx(np.array([5, 7, 5, 3, 2, 3]), 4)
    array([0, 3, 5])
    """

    # Requires array to have a minimum size
    if array.size <= 1:
        raise ValueError("Input array must have more than 2 elements.")

    idxs = []

    # Tries to identify a trend at the start and, if the value comes before the start, would it match that trend
    if value < array[0] and array[0] < array[1] or value > array[0] and array[0] > array[1] or value == array[0]:
        idxs.append(0)

    # Iterate through every element except the last
    for i in range(len(array) - 1):
        this = array[i]
        next = array[i + 1]
        if this == value:
            idxs.append(i) # Append an index if the value perfectly matches

        # If the value is between two other adjacent values, append that as well
        elif this > value and next < value or this < value and next > value:
            left_neighbor_diff = np.abs(value - array[i])
            right_neighbor_diff = np.abs(value - array[i + 1])
            idxs.append(i if left_neighbor_diff < right_neighbor_diff else i + 1)

    # Tries to identify a trend at the end and, if the value comes after the end, would it match that trend
    if value < array[-1] and array[-1] < array[-2] or value > array[-1] and array[-1] > array[-2] or value == array[-1]:
        idxs.append(len(array) - 1)

    return np.array(idxs)

def load_data_and_hull(data_file_path: str, bh_scaling_factor):
    """
    Loads 3D points from a file and computes their convex hull.

    Args:
        data_file_path (str): The path to the (x, y, z) data file.

    Returns:
        A tuple of (x, y, z, triangles, N_points)
        Returns (None, None, None, None, 0) on failure.
    """
    x_data, y_data, z_data = [], [], []

    try:
        with open(data_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        x_data.append(float(parts[0]))
                        y_data.append(float(parts[1]))
                        z_data.append(float(parts[2]))
                except ValueError:
                    print(f"Skipping malformed line: {line}")

        x = np.array(x_data)
        y = np.array(y_data)
        z = np.array(z_data)

        if x.size == 0:
            print(f"Error: No valid data loaded from {data_file_path}.")
            return None, None, None, None, 0

        # --- Surface Reconstruction (Convex Hull) ---
        points_for_hull = np.column_stack((x, y, z))
        hull = ConvexHull(points_for_hull)

        return x, y, z, hull.simplices, x.size

    except FileNotFoundError:
        print(f"Error: Data file '{data_file_path}' not found.")
        return None, None, None, None, 0
    except Exception as e:
        print(f"An error occurred during file reading or hull calculation: {e}")
        return None, None, None, None, 0

def update_mesh_data(source_object, data_file_path: str, bh_scaling_factor: float):
    """
    Updates an existing Mayavi VTKDataSource object with new data.

    Args:
        source_object: The VTKDataSource object returned by create_mesh_source().
        data_file_path (str): Path to the new data file to load.
    """
    # Load the new data and calculate the new hull
    x, y, z, triangles, n_points = load_data_and_hull(data_file_path, bh_scaling_factor)

    if n_points == 0:
        print(f"Failed to update mesh, no data loaded from {data_file_path}.")
        return

    # Combine x, y, z into an (N_points, 3) array for VTK
    new_points_array = np.column_stack((x, y, z))

    # Get the underlying dataset (a vtkPolyData object) from the source
    # This is the key to efficient updates!
    dataset = source_object.data

    # Update the points (the vertices of the triangles)
    dataset.points.from_array(new_points_array)

    # Update the triangles (the "faces" connecting the points)
    # .from_array() expects a 1D list of [n_verts_per_face, idx1, idx2, idx3, ...]
    # So we have to insert a '3' before each triangle's indices
    n_triangles = triangles.shape[0]
    # Create an array of '3's, one for each triangle
    triangle_format = np.full((n_triangles, 1), 3, dtype=np.int64)
    # Stick the '3's in front of the triangle indices
    polys_array = np.hstack((triangle_format, triangles))

    dataset.polys.from_array(polys_array)

    # Force the dataset to update its internal state
    dataset.modified()

# --- 3. New Function: Create Mesh Source ---
def plot_initial_mesh(engine: Engine, data_file_path: str, bh_scaling_factor: float):
    """
    Loads data from a file and creates a Mayavi VTKDataSource object.
    This object can then be manually added to an engine.

    Args:
        data_file_path (str): The path to the data file.

    Returns:
        The VTKDataSource object if successful, else None.
    """

    # Use the current scene
    scene = engine.current_scene if hasattr(engine, "current_scene") else engine.scenes[0]

    x, y, z, triangles, n_points = load_data_and_hull(data_file_path, bh_scaling_factor)

    if n_points == 0:
        print(f"Failed to create mesh source, no data in {data_file_path}.")
        return None

    # 1. Create the (N_points, 3) array of point coordinates
    points_array = np.column_stack((x, y, z))

    # 2. Create the 1D array of triangle definitions
    n_triangles = triangles.shape[0]
    triangle_format = np.full((n_triangles, 1), 3, dtype=np.int64)
    polys_array = np.hstack((triangle_format, triangles))

    # 3. Create the vtkPolyData object that holds the geometry
    polydata = tvtk.PolyData(points=points_array, polys=polys_array)

    # 5. Create the Mayavi data source
    source = VTKDataSource(data=polydata)

    engine.add_source(source, scene)

    # 3. Make the source visible by adding a Surface module
    # We save the 'surface' object to pass to update_mesh_data
    surface = mlab.pipeline.surface(source)

    # Set the actor's color. (0,0,0) is black. (1,1,1) is white.
    surface.actor.property.color = (0, 0, 0)

    return source, surface

def main() -> None:
    """
    Execute the main workflow of the gravitational wave animation script.

    Reads strain data, calculates and factors in spin-weighted spherical harmonics,
    linearly interpolates the strain to fit mesh points, and creates .tvtk mesh
    files for each time state. The meshes represent the full superimposed waveform at the polar angle pi/2
    (the plane of the binary black hole merger). At each state, moves the black holes
    to their respective positions and saves the mesh as a .png file. Finally,
    compiles the PNG frames into an MP4 movie.
    """

    # Convert psi4 data to strain using imported script
    # This should ideally be called explicitly if needed, or integrated better.
    # For now, assuming psi4strain.psi4_ffi_to_strain handles it or data is pre-converted.
    # psi4_to_strain.main() # Example: If psi4_to_strain module existed

    # Check initial parameters
    time0 = time.time() # Get the initial simulation time
    global BH_DIR, MOVIE_DIR, EXT_RAD # Allow modification of globals based on args
    bh1_rel_mass: float = 1.0 # Default mass
    bh2_rel_mass: float = 1.24 # Default mass ratio for GW150914
    use_symlog: bool = False # Default scale

    if USE_SYS_ARGS:
        argc = len(sys.argv)
        if argc not in (2, 3):
            # Use raise RuntimeError for error exit
            raise RuntimeError(
                f"Usage: python3 {sys.argv[0]} <path_to_data_folder> [use_symlog: True/False]\n\n"
                f"Example: python {sys.argv[0]} ../data/GW150914_data/r100 true\n\n"
                "Arguments:\n"
                "\t<path_to_data_folder>: Path to the directory containing merger data and converted strain.\n"
                "\t                       Use LIST for a list of available data directories.\n"
                "\t[use_symlog]: Optional. Use symmetric log scale for strain (True/False, default: False)."
            ) # Use f-string for cleaner formatting
        else:
            # Change directories and extraction radius based on inputs
            simulation_name = sys.argv[1]
            bh_dir = os.path.join("../data", simulation_name)

            # Set psi4_output_dir relative to bh_dir
            psi4_output_dir = os.path.join(bh_dir, "strain")
            movie_dir = os.path.join(bh_dir, "movies")  # Optimized path construction

            # Handle optional symlog argument
            if argc == 3:
                symlog_arg = sys.argv[2].lower()
                if symlog_arg == 'true':
                    use_symlog = True
                elif symlog_arg == 'false':
                    use_symlog = False
                else:
                    raise ValueError("Argument 'use_symlog' must be 'true' or 'false'.")

    else: # Use default parameters defined at the top
        bh_dir = BH_DIR
        movie_dir = MOVIE_DIR
        psi4_output_dir = os.path.join(bh_dir, "converted_strain")
        # Default mass ratio for GW150914 already set
        # Default use_symlog is False

    # --- Ensure directories exist ---

    # List of available directories
    data_path = "../data"
    available_dirs = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]

    # Generate a string to list all available directories in the data path
    dir_str = f"Available directories in {data_path}:\n\n"
    for name in available_dirs:
        dir_str += name + "\n"

    # Handle the case where the user has asked for the directory list
    if simulation_name == "LIST":
        print(f"\n{dir_str}")
        sys.exit(0)
    # Handle the case where the user entered an invalid simulation name. Also gives the directory list
    elif simulation_name not in available_dirs:
        raise FileNotFoundError(f"Data directory not found: {bh_dir}\n\n{dir_str}")

    # --- Movie File Path Handling ---
    bh_file_name = "puncture_posns_vels_regridxyzU.txt" # Name for the black hole position file
    bh_file_path = os.path.join(bh_dir, "puncture", bh_file_name) # Black hole position file path
    # Puncture file MUST EXIST for code to run
    if not os.path.isfile(bh_file_path):
         raise FileNotFoundError(f"Black hole position file not found: {bh_file_path}")

    bh_scaling_factor = 2.0 # Visual scaling of black holes

    # A while loop to figure out where to save the simulation
    movie_number = 1 # Movies are saved as real_movie1, real_movie2, etc
    while True:
        movie_dir_name = f"real_movie{movie_number}"
        movie_file_path = os.path.join(movie_dir, movie_dir_name)

        if os.path.exists(movie_file_path):
            # Ask the user for permission to override existing file with same name
            response = input(f"{movie_file_path} already exists. Would you like to overwrite it? Y/N: ")
            if response.lower() != 'y':
                movie_number += 1
                continue # Continue if no clear permission was given

            # User confirmed overwrite
            if STATUS_MESSAGES:
                print(f"Overwriting existing files in {movie_file_path}...")
            try:
                # Clear existing files if user confirms overwrite
                for file in os.listdir(movie_file_path):
                    try:
                        os.remove(os.path.join(movie_file_path, file))
                    except OSError as e:
                        print(f"Warning: Could not remove file {file}: {e}")
                break # Exit loop after clearing or attempting to clear
            except FileNotFoundError:
                # The directory might have been deleted between check and listdir
                print(f"Warning: Directory {movie_file_path} disappeared.")
                # Allow code to continue, creating the movie file path from scratch this time
            except Exception as e:
                raise RuntimeError(f"Error clearing directory {movie_file_path}: {e}")

        # If no directory of the same name is present, create one (and parent movie_dir if needed)
        try:
            os.makedirs(movie_file_path, mode=0o755, exist_ok=True)
            print(f"Output will be saved in: {movie_file_path}")
            break # Exit loop after successful creation or confirmation
        except OSError as e:
            raise RuntimeError(f"Could not create output directory {movie_file_path}: {e}")

    movie_path = os.path.join(movie_file_path, movie_dir_name) # Full path + name of the movie
    movie_path_name = movie_path + ".mp4"

    # --- Extraction Radius Calculations ---
    bh_file_list = os.listdir(bh_dir) # Extract the files in the black hole directory
    psi4_dir = os.path.join(bh_dir, "psi4")
    strain_dir = os.path.join(bh_dir, "strain")
    psi4_exists = os.path.isdir(psi4_dir)
    strain_exists = os.path.isdir(strain_dir)

    strain_files = []
    psi4_files = []
    if strain_exists: # If strain data is provided, use file names from that one
        strain_file_list = os.listdir(strain_dir)
        strain_files += [f for f in strain_file_list if os.path.isfile(os.path.join(strain_dir, f))] # List only files
    if psi4_exists: # If psi 4 data is provided, use file names from that one
        psi4_file_list = os.listdir(psi4_dir)
        psi4_files += [f for f in psi4_file_list if os.path.isfile(os.path.join(psi4_dir, f))] # List only files
    if not (strain_exists or psi4_exists):
        # Throw an error if no data is provided
        raise FileNotFoundError(f"No psi4 or strain data found in the directory {bh_dir}")

    bh_files = strain_files + psi4_files # Concatenate psi4 and strain files into general files
    extraction_radii = np.empty(0)
    for b in bh_files:
        # Attempt to convert the part of the file name that is supposed to be the extraction radius into a float
        try:
            radius_extraction = float(b[-10:-4])
        except (ValueError, IndexError):
            try:
                radius_extraction = float(b[-7:-4]) # Handle the case where it might be infinity
            except (ValueError, IndexError):
                continue

        extraction_radii = np.unique(np.append(extraction_radii, radius_extraction)) # Save the extraction radius if unique

    size = len(extraction_radii)
    if size == 1:
        radius_extraction = extraction_radii[0] # If only one extraction radius is found, use that one
    elif size == 0:
        # If no extraction radii are found, the files are probably incorrectly named
        raise RuntimeError("No extraction radii found. Ensure files are formatted as such: {filename}_l#-r{####.# or inf}")
    elif size > 1:
        # Handle the case where multiple extraction radii are found
        print("Warning: Multiple extraction radii found in the directory.")
        while True:
            response = input("Please enter the extraction radius you would like to use: ")
            try:
                # Attempt to parse user input into extraction radius float
                radius_extraction = float(response)
                # Print availabe extraction radii if user inputs one that is unavailable
                if radius_extraction == float('inf'):
                    break
                elif radius_extraction not in extraction_radii:
                    print("Available extraction radii:")
                    for r in extraction_radii:
                        print(r)
                else:
                    break # End the loop if a valid extraction radius has been entered
            except ValueError:
                print("Please enter 'inf' or a float from 0.0 to 9999.0.") # Handle the case where something else was entered

    ext_rad = radius_extraction

    ext_rad_num = ext_rad if ext_rad != float('inf') else 0 # A number to use to manipulate data, based on extraction radius

    if STATUS_MESSAGES:
        print(f"Using extraction radius {ext_rad if ext_rad >= 0 else radius_extraction} for {'strain' if strain_exists else 'psi_4'} data")

    # Check to see which directory houses the appropriate ext_rad
    r_ext_in_strain = False
    for file in strain_files:
        if str(radius_extraction) in file:
            r_ext_in_strain = True # If the extraction radius is in strain file name, set to true. Else, default to psi4

    # --- Minimum and Maximum Ell Mode Calculations ---
    ells = np.empty(0)
    # Convert extraction radius used into a properly formatted string ####.# to determine which files to search
    if ext_rad != float('inf'):
        str_ext_rad = ("0" if ext_rad < 1000 else "") + str(ext_rad)
    else:
        str_ext_rad = str(ext_rad) # If inf, use that

    appropriate_bounds = (-10, -4) if ext_rad != float('inf') else (-7, -4) # Appropriate bounds for which to look for extraction radius
    appropriate_ell_idx = -13 if ext_rad != float('inf') else -10

    for b in (strain_files if r_ext_in_strain else psi4_files):
        # Only search files with the appropriate extraction radius
        if b[appropriate_bounds[0]:appropriate_bounds[1]] == str_ext_rad:
            # Attempt to convert the part of the file name that is supposed to be the mode into an integer
            try:
                ell = float(b[appropriate_ell_idx])
            except ValueError:
                continue # Skip over files that fail or don't have a mode
            if ell not in ells:
                ells = np.append(ells, ell) # Save the ell mode if unique

    ells = np.sort(ells.astype(int)) # Put the ell modes in order
    if len(ells) == 0:
        # If no modes are found, the files are probably incorrectly named
        raise RuntimeError("No l modes found. Ensure files are formatted as such: {filename}_l#-r{####.# or inf}")
    # Extract the min and max ells
    ell_min = ells[0]
    ell_max = ells[-1]

    # Calculate the difference between consecutive ells to detect gaps
    diff_ells = np.diff(ells)
    gaps = np.where(diff_ells > 1)
    # If there are any gaps, raise an error
    if gaps[0].size > 0:
        raise RuntimeError(f"A gap was detected in the l modes: Minimum l is {ell_min}, maximum l is {ell_max}, but no l={gaps[0][0] + ell_min + 1} file was found")

    if STATUS_MESSAGES:
        print(f"Using minimum mode l={ell_min} and maximum mode l={ell_max} for {'strain' if strain_exists else 'psi_4'} data")

    # --- Simulation & Visualization Parameters ---
    n_rad_pts = 450       # number of points along the radius
    n_azi_pts = 180       # number of points along the azimuth
    colat = np.pi / 2     # colatitude angle (pi/2 for equatorial plane)

    # Cosmetic & camera parameters
    wireframe = True
    frames_per_second = 24
    save_rate = 10  # Saves every Nth simulation time step
    resolution = (1920, 1090) # Width, Height
    gw_color = (0.28, 0.46, 1.0) # Blueish
    bh_color = (0.1, 0.1, 0.1)   # Dark grey/black
    zoomout_distance = 350 # Max camera distance after zoomout
    initial_elevation_angle = 50 # Initial camera elevation 
    final_elevation_angle = 34   # Final camera elevation
    azi_angle = 45 # Default pi/4 azimuth camera angle
    FOV_angle = 30 # Default field of view angle
    pip_xfraction = 0.8 # Fraction of corner of screen taken up by picture in picture view

    time1 = time.time() # End of initial setup

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"{'*' * 70}\nConverting psi4 data to strain...")

    # See if strain files exist, and if so, attempt to load them
    if r_ext_in_strain:
        # One time iteration parameters
        time_array_set = False
        mode_data_set = False

        if STATUS_MESSAGES:
            print(f"Using existing strain data files in {strain_dir}")

        # Iterate through each l mode and extract the appropriate strain data 
        files_processed = 0
        for l in range(ell_min, ell_max + 1):

            # Construct filename and file path safely
            filename = psi4strain.STRAIN_FILE_FMT + f"_l{l}-r{str_ext_rad}.txt"
            file_path = os.path.join(psi4_output_dir, filename)

            # Determine expected number of columns based on l
            # Time + each m from -l to l = 1 + (2l+1) = 2l+2 columns
            # usecols goes up to max_col index, so max_col should be 2l+1
            max_col_idx = 2 * l + 1
            cols_to_use = range(0, max_col_idx + 1) # Range includes 0, stops before max_col_idx + 1

            # Try to load the text files
            try:
                # Load data, skipping header lines dynamically
                # Header lines = 1 (time) + (number of modes = 2l+1)
                num_skip_rows = 2*l + 2
                data_all = np.loadtxt(file_path, dtype=np.complex128, skiprows=num_skip_rows, usecols=cols_to_use)
                files_processed += 1
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}, skipping l={l}.")
                continue
            except ValueError as e:
                # Catch errors if columns don't exist or data is malformed
                print(f"Warning: Error loading {file_path}: {e}. Skipping l={l}.")
                continue
            except IndexError:
                # This might occur if usecols exceeds actual columns due to bad file or wrong max_col calculation
                print(f"Warning: Index error loading columns from {file_path}. Skipping l={l}.")
                continue

            # For the first l mode, set the time array to the first column (the time column)
            if not time_array_set:
                time_array = data_all[:, 0].real
                time_array_set = True
            # For the first l mode, initialize the shape of the mode_data
            if not mode_data_set:
                mode_data = np.empty((0, len(time_array)))
                mode_data_set = True

            # Iteratively add to 2D mode_data array: shape (all l and m, n_times)
            mode_data = np.vstack((mode_data, data_all[:, 1:].T))

        # If all the files are skipped, end the program
        if files_processed == 0:
            raise FileNotFoundError(f"No valid strain files found in directory {psi4_output_dir} for l={ell_min} to {ell_max}.")

    # If there is no strain, use psi4_FFI_to_strain to convert psi 4 into strain
    else:
        # Convert psi4 to strain and load strain data
        try:
            # Pass the directory where the strain files are expected
            time_array, mode_data = psi4strain.psi4_ffi_to_strain(psi4_dir, psi4_output_dir, ell_max, ext_rad)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading strain data: {e}. Ensure converted files exist in {psi4_output_dir} or check psi4strain function.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during psi4 data conversion and strain loading: {e}")

    # Extract number of times and throw an error if there are no times
    n_times = len(time_array)
    if n_times == 0:
        raise ValueError(f"Loaded time array is empty. Cannot proceed.")
    n_frames = int(n_times / save_rate)

    if STATUS_MESSAGES:
        print(f"Loaded {mode_data.shape[0]} modes over {n_times} time steps.")

    time2=time.time() # End of strain conversion

    if STATUS_MESSAGES:
        print(f"{'*' * 70}\nCalculating black hole trajectories...")

    # Import black hole data using more efficient loading
    try:
        # Skiprows assumes a fixed header length. Verify this matches the file spec.
        bh_data = np.loadtxt(bh_file_path, skiprows=14, dtype=np.float64)
    except FileNotFoundError:
         raise FileNotFoundError(f"Black hole position file not found: {bh_file_path}")
    except Exception as e:
         raise RuntimeError(f"Error loading black hole data from {bh_file_path}: {e}")

    if bh_data.shape[1] < 5 or bh_data.shape[0] < 2 or bh_data.ndim != 2:
         raise ValueError(f"""Black hole data in {bh_file_path} has unexpected shape or too few rows. Ensure the file meets the following criteria:
    - 15 rows of header lines
    - At least 5 columns of data formatted as follows:
        column 0: retarded time
        column 1: areal mass of black hole 1
        column 2: areal mass of black hole 2
        column 3: x position of black hole 1
        column 4: y position of black hole 1""")

    merge_idx_bh = np.argmax(np.diff(bh_data[:, 1])) + 1 # Find index where BH mass jumps as index of merge point
    merge_time = bh_data[merge_idx_bh, 0]

    # Extract BH time array
    bh_time = bh_data[:, 0]

    # Mass ratio calculation using slice object up to merger
    pre_merge = slice(None, merge_idx_bh)

    # Calculating the average areal mass of each black hole before the merge point
    bh1_avg_mass = np.mean(bh_data[pre_merge, 1])
    bh2_avg_mass = np.mean(bh_data[pre_merge, 14])

    bh_total_mass = bh1_avg_mass + bh2_avg_mass
    bh1_rel_mass = bh1_avg_mass / bh_total_mass
    bh2_rel_mass = bh2_avg_mass / bh_total_mass

    mass_ratio = bh1_rel_mass / bh2_rel_mass

    # Extract BH coordinates (Check columns: 3=x, 4=y, assuming z=0 initially)
    bh1_x0, bh1_y0, bh1_z0 = bh_data[:, 3], bh_data[:, 4], bh_data[:, 5]  # Assume motion is in xy-plane

    # Interpolate BH positions to the *strain* time array (equal_times)
    # Time array optimization: Use the actual strain time array for interpolation basis
    equal_times = np.linspace(time_array[0], time_array[-1], num=n_times) # Create equally spaced times for output frames
    merge_idx_equal = find_idx(equal_times, merge_time)[0] # Find merge index in the output time array

    # Maintain original interpolation call
    bh1_x, bh1_y, bh1_z = interpolate_coords_by_time(
        bh_time, bh1_x0, bh1_y0, bh1_z0, equal_times
    )

    # Vectorized coordinate calculation using array stacking
    pre_slice = slice(None, merge_idx_equal)
    post_slice = slice(merge_idx_equal, None)

    # Single array operation for all coordinates
    bh2_coords = np.concatenate([
        -1 * mass_ratio * np.column_stack((bh1_x[pre_slice], bh1_y[pre_slice], bh1_z[pre_slice])),
        np.column_stack((bh1_x[post_slice], bh1_y[post_slice], bh1_z[post_slice])) # bh2 initially moves opposite bh1, but
                                                                                   # changes to following bh1 after merge
    ])

    bh2_x, bh2_y, bh2_z = bh2_coords.T  # Transpose and unpack

    less_massive_x = bh1_x if bh1_rel_mass < bh2_rel_mass else bh2_x # Find x and y coordinates of less massive bh
    less_massive_y = bh1_y if bh1_rel_mass < bh2_rel_mass else bh2_y # These will be the most "sweeping"

    if STATUS_MESSAGES:
        disp_mass_ratio = mass_ratio if mass_ratio >= 1 else 1 / mass_ratio # First bh should always be >=1 in the ratio
        print(f"Black hole mass ratio: {disp_mass_ratio:.3f}:{1}")

    time3=time.time() # End of black hole position calculations

    if APPARENT_HORIZONS:
        if STATUS_MESSAGES:
            print(f"{'*' * 70}\nComputing horizon data...")

        horizon_dir = os.path.join(bh_dir, "horizons")

        try:
            horizon_files = os.listdir(horizon_dir)
        except FileNotFoundError:
            raise FileNotFoundError(f"Black hole position file not found: {horizon_dir}")
        except Exception as e:
            raise RuntimeError(f"Error loading black hole data from {horizon_dir}: {e}")

        horizon_files.sort()

        horizon_times = np.empty(0)
        horizon_names = {}
        horizon_merge_time = -1

        for file in horizon_files:
            try:
                horizon_time = int(file[-14:-7])
                time_string = str(float(horizon_time))
                horizon_filepath = os.path.join(horizon_dir, file)
                if file[-6:-3] == "ah3" and horizon_merge_time < 0:
                    horizon_merge_time = horizon_time # Note the time the black holes form a common horizon
                horizon_times = np.unique(np.append(horizon_times, horizon_time)) # Try converting last part of filename to timestep number
                if time_string in horizon_names:
                    horizon_names[time_string].append(horizon_filepath) # If it works, append the file name to the horizon_names
                else:
                    horizon_names[time_string] = [horizon_filepath]
            except (ValueError, IndexError): # If it's out of bounds or doesn't convert to a number, it's not a file we want
                continue

        position_to_horizon = horizon_merge_time / (merge_time + ext_rad_num)

    timeh = time.time()

    if STATUS_MESSAGES:
        print(f"{'*' * 70}\nComputing camera parameters...")

    try:
        aspect_ratio = resolution[0] / resolution[1] # Calculate the aspect ratio. This is used to calculate horizontal FOV
    except ZeroDivisionError as e:
        # Throw an error if height is somehow zero
        raise ValueError("Height of the scene is zero. Cannot calculate aspect ratio.")

    vert_FOV = np.radians(FOV_angle) # Get the vertical field of vision angle in radians
    horz_FOV = 2 * np.arctan(np.tan(vert_FOV / 2) * aspect_ratio) # Get the horizontal field of vision angle
    # Find out which field of view is larger and smaller. The smaller FOV will be used to calculate how far the
    # camera should zoom out (since both black holes should be visible even if lined up along the smaller FOV 
    # at the start). The larger FOV will be used to calculate the display radius (since the mesh needs to take up
    # the entire screen from the start)
    larger_FOV, smaller_FOV = np.maximum(horz_FOV, vert_FOV), np.minimum(horz_FOV, vert_FOV)

    # Calculate appropriate display radius
    magnitudes = np.sqrt(less_massive_x**2 + less_massive_y**2) # Get the distance between bh1 and the center (bh1 is less massive)
    bh_azis = np.unwrap(np.arctan2(less_massive_y, less_massive_x))

    init_elevation_rads = np.radians(initial_elevation_angle)
    fin_elevation_rads = np.radians(final_elevation_angle)
    # Calculate the appropriate starting zoom based on limiting camera parameters and black hole initial positions
    zoom_start = np.maximum(80, magnitudes[0] * np.sin((np.pi - smaller_FOV) / 2 + fin_elevation_rads) \
                 / np.sin(smaller_FOV / 2))

    if STATUS_MESSAGES:
        print(f"Starting zoom: {zoom_start} (Default is 80)")

    time4 = time.time() # End of camera parameter calculations

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"{'*' * 70}\nInitializing grid points..."
        )

    # Calculate the display radius based on limiting final camera parameters
    start_view_radius = np.hypot(zoom_start * np.cos(larger_FOV) * np.sin(larger_FOV / 2) \
                        / (np.cos(init_elevation_rads + larger_FOV / 2) * np.cos(init_elevation_rads)), \
                        zoom_start * np.cos(init_elevation_rads) * np.tan(larger_FOV / 2))
    display_radius = np.maximum(300, start_view_radius) # Ensures the display radius is, at a minimum, 300

    if STATUS_MESSAGES:
        print(f"Display radius: {display_radius} (Default is 300)")

    # Defining a shifted error function, rising sharply after the default display radius. This will be used to determine
    # how much to add to the distance between each radius point on the mesh grid. Points outside the default display
    # radius will be rendered at lower resolutions, depending on available memory
    def shifted_erf(x):
        return 1 + erf(x - 300)

    float64size = np.dtype(np.float64).itemsize # Get the size of a float64 object
    available_memory = psutil.virtual_memory().available - 1000000000 # Get the available memory, with a significant buffer
    # Calculate the number of radius points. Default is 350, then as many additional points are added as memory will allow.
    max_rads = 350 + int((available_memory) / (n_times * n_azi_pts * float64size))
    if STATUS_MESSAGES:
        print(f"Maximum radius points: {max_rads} (Allocated from memory, default is 350)")

    # Calculate by what factor the shifted error function should be scaled vertically
    resolution_dropoff_factor = (display_radius - 300) / (2 * (max_rads - 450)) - 1 / 3
    gen_rad = 0
    rad_vals = []

    # Loop through radius values, generating each radius point until the display radius is reached
    while gen_rad < display_radius:
        rad_vals.append(gen_rad)
        # Calculate the next distance between radius points. Within the default display radius, rad_delta is typically
        # Close to 2/3, which is default
        rad_delta = resolution_dropoff_factor * shifted_erf(gen_rad) + 2 / 3
        gen_rad += rad_delta

    # Ensure the display radius point is also added to rad_vals, for consistent edges
    if display_radius not in rad_vals:
        rad_vals.append(display_radius)

    # Cast the rad_vals as a numpy array and get the length
    radius_values = np.array(rad_vals)
    n_rad_pts = len(radius_values)

    # Initialize a grid with the calculated azimuth and radius points
    strain_array, grid, points = initialize_tvtk_grid(n_azi_pts, n_rad_pts)

    # theta and radius values for the mesh
    azimuth_values = np.linspace(0, 2 * np.pi, n_azi_pts, endpoint=False, dtype=np.float32) # Use float32

    # Create meshgrid (ij indexing gives radius changing fastest)
    rv, az = np.meshgrid(radius_values, azimuth_values, indexing="ij")
    # Calculate Cartesian coordinates for the flat mesh
    x_values = rv * np.cos(az)
    y_values = rv * np.sin(az)

    if STATUS_MESSAGES:
         print("Calculating spin-weighted spherical harmonics...")

    # Apply spin-weighted spherical harmonics, superimpose modes, and interpolate to mesh points
    strain_azi = swsh_summation_angles(colat, azimuth_values, mode_data, ell_min, ell_max, STATUS_MESSAGES).real

    # Broadcasts equal_times and radius_values together to create a 2D array (n_radii, n_times) that shows the retarded
    # time at each radius, plus the extraction radius
    lerp_times = equal_times[np.newaxis, :] - radius_values[:, np.newaxis] + ext_rad_num

    time5=time.time() # End of grid setup

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"{'*' * 70}\nCalculating cosmetic data..."
        )

    orbit_start = find_idx(np.array(bh_azis), bh_azis[0] + (np.pi / 2 if bh_azis[0] < bh_azis[1] else np.pi / -2))
    # If the array is 0 size, or if it only contains edge cases, set equal to last azi
    if orbit_start[0] == bh_azis.size - 1:
        orbit_start_idx = len(bh_azis) - 2
    else:
        orbit_start_idx = orbit_start[0]

    orbit_end = find_idx(np.array(bh_azis), bh_azis[0] + (5 * np.pi / 2 if bh_azis[0] < bh_azis[1] else 5 * np.pi / -2))
    if orbit_end[0] == bh_azis.size - 1:
        orbit_end_idx = len(bh_azis) - 1
    else:
        orbit_end_idx = orbit_end[0]

    # Find radius of the center hole in the mesh (based on uninterpolated max separation + BH size)
    # Hole radius = factor * (max_separation + scaled radius of larger BH)
    omitted_radius_length = np.max(magnitudes[orbit_start_idx:orbit_end_idx]) + bh_scaling_factor * max(bh1_rel_mass, bh2_rel_mass) + 1
    if omitted_radius_length > 100 and not TRAJECTORY_LINES:
        omitted_radius_length = 5 # Ensure that simulations where black holes are really far apart don't generate massive
                                  # holes, so long as trajectories aren't being tracked
        if STATUS_MESSAGES:
            print("WARNING: Black holes are spaced too far apart to generate proportional hole in mesh. Default hole will be used.")

    # Find point at which to taper off gravitational waves
    width = 0.5 * omitted_radius_length # Width of the transition region
    dropoff_radius = width + omitted_radius_length # Radius at which to start tapering beyond the hole

    # Find max amplitude scale factor allowable without impeding view of center hole (based on dropoff radius)
    # Apply amplitude scale factor calculation based on spin-weighted spherical harmonics max strain
    amplitude_scale_factor = get_amplitude_scale_factor(np.sign(strain_azi) * np.log1p(np.abs(strain_azi)) if use_symlog else strain_azi, azimuth_values, omitted_radius_length, zoomout_distance, fin_elevation_rads, np.radians(azi_angle)) 
    # Dropoff factor (smooth transition to zero amplitude near omitted hole), apply amplitude scale factor
    dropoff_2D_flat = (0.5 + 0.5 * erf((radius_values - dropoff_radius)/width)).ravel() * amplitude_scale_factor

    # Report calculated values
    if STATUS_MESSAGES:
        print(f"Amplitude scale factor: {amplitude_scale_factor:.3f}")

    # Zoom out a quarter of the way through the data
    zoomout_time = (equal_times[-1] - equal_times[0]) / 4
    zoomout_idx = find_idx(equal_times, zoomout_time)[0]

    time6 = time.time() # End of cosmetic data calculations

    if STATUS_MESSAGES:
        print(f"{'*' * 70}\nConstructing mesh points in 3D...")

    mmap_name = os.path.join(movie_file_path, "datamap")
    # Interpolate the strain to the appropriate points on the mesh grid
    strain_to_mesh = compute_strain_to_mesh(
        strain_azi,
        equal_times.astype(np.float32),
        radius_values,
        lerp_times,
        time_array.astype(np.float32),
        dropoff_2D_flat,
        use_symlog,
        mmap_name,
        STATUS_MESSAGES
    )

    time7=time.time()

    if STATUS_MESSAGES:
        print(f"{'*' * 70}\nInitializing animation...")

    # --- Precompute values for animation loop ---
    bh1_scaled_radius = bh1_rel_mass * bh_scaling_factor
    bh2_scaled_radius = bh2_rel_mass * bh_scaling_factor
    bh_merged_radius = (bh1_scaled_radius + bh2_scaled_radius) / bh2_scaled_radius

    # Indices of the simulation time steps to actually render
    valid_indices = np.arange(0, n_times, save_rate)
    n_valid = len(valid_indices) # Number of frames to actually generate

    # Precompute geometric data & masks with vectorization
    # Mask for points outside the central hole
    valid_mask = (rv > omitted_radius_length).ravel()

    # Flattened xy coordinates of the mesh
    x_flat, y_flat = x_values.ravel(), y_values.ravel()

    # Initialize VTK data structures once
    points = tvtk.Points() # This holds the 3D coordinates
    vtk_array = tvtk.FloatArray() # VTK array to store coordinate data
    vtk_array.number_of_components = 3
    vtk_array.number_of_tuples = len(x_flat)
    points.data = vtk_array

    # Get a NumPy view for efficient modification
    np_points = vtk_array.to_array().reshape(-1, 3)
    # Set XY coordinates (these don't change)
    np_points[:, 0] = x_flat
    np_points[:, 1] = y_flat
    # Z coordinate will be updated in the loop

    # Precompute camera parameters for each output frame
    time_indices = np.arange(n_times) # Indices 0 to n_frames-1
    # Smoothly decrease elevation angle over time until it hits the target (this means the camera rises)
    elevations = np.maximum(initial_elevation_angle - time_indices * 0.016, final_elevation_angle)
    # Smoothly increase distance during zoomout phase
    if zoom_start < zoomout_distance:
        distances = np.minimum(np.where(
            time_indices < zoomout_idx,
            zoom_start,
            zoom_start + (time_indices - zoomout_idx) * 0.175
        ), zoomout_distance)
    elif zoom_start > zoomout_distance:
        distances = np.maximum(np.where(
            time_indices < zoomout_idx,
            zoom_start,
            zoom_start - (time_indices - zoomout_idx) * 0.175
        ), zoomout_distance)

    # Precompute percentage thresholds for progress report (based on simulation time)
    percentage_thresholds = np.round(np.linspace(0, n_times, 101)).astype(int)

    # Precompute frame filenames using f-strings and os.path.join
    frame_filenames = [os.path.join(movie_file_path, f"z_frame_{i:05d}.png") for i in range(n_valid)]

    # Configure engine and rendering upfront
    engine = Engine()
    engine.start()
    fig = mlab.figure(engine=engine, size=resolution) # Default background color
    fig.scene.interactor.disable() # Make it so camera view can't be changed accidentally

    # Initialize visualization objects once
    # GW Surface
    create_gw(engine, grid, gw_color, display_radius, wireframe)

    # Black Holes
    if APPARENT_HORIZONS:
        # Create source and mesh for black hole horizons
        bh1_source, bh1 = plot_initial_mesh(engine, horizon_names["0.0"][0], bh_scaling_factor)
        bh2_source, bh2 = plot_initial_mesh(engine, horizon_names["0.0"][1], bh_scaling_factor)
    else:
        bh1 = create_sphere(engine, bh1_scaled_radius, bh_color)
        bh2 = create_sphere(engine, bh2_scaled_radius, bh_color)

        # Precompute merge rescale index
        merge_condition = (valid_indices > merge_idx_equal) & (valid_indices < merge_idx_equal + save_rate)
        merge_rescale_indices = np.where(merge_condition)[0]
        merge_rescale_idx = merge_rescale_indices[0] if merge_rescale_indices.size > 0 else -1

    # Create trajectory tracking lines if needed
    if TRAJECTORY_LINES:
        bh1_trajectory = mlab.plot3d(bh1_x[0], bh1_y[0], bh1_z[0], figure=fig, tube_radius=0.1)
        bh2_trajectory = mlab.plot3d(bh2_x[0], bh2_y[0], bh2_z[0], figure=fig, tube_radius=0.1)
        bh1_trajectory.actor.property.opacity = 0.5 # Make them semi-transparent so they don't cover up black holes
        bh2_trajectory.actor.property.opacity = 0.5

    if PIP_VIEW: # For adding picture-in-picture view to MayaVi window
        render_window = fig.scene.render_window # Get the main render window from the scene

        inset_renderer = tvtk.Renderer() # Create a new renderer

        # Set horizontal position and calculate vertical position to ensure view is square
        inset_renderer.viewport = (pip_xfraction, 1 - aspect_ratio * (1 - pip_xfraction), 1.0, 1.0)
        inset_renderer.background = (0.5, 0.5, 0.5) # Set a different bg color

        render_window.add_renderer(inset_renderer) # Add render window to screen

        inset_camera = inset_renderer.active_camera # Get the camera of the view
        inset_camera.parallel_projection = True # Use an orthographic projection
        inset_camera.parallel_scale = omitted_radius_length # Set the camera to view entire omitted radius
        inset_camera.focal_point = (0, 0, 0) # Look at the center
        inset_camera.position = (0, 0, 15) # Position the camera along z-axis

        # Add bh horizons to this new view
        inset_renderer.add_actor(bh1.actor.actor)
        inset_renderer.add_actor(bh2.actor.actor)

        if TRAJECTORY_LINES:
            # If trajectory lines are enabled, add those too
            inset_renderer.add_actor(bh1_trajectory.actor.actor)
            inset_renderer.add_actor(bh2_trajectory.actor.actor)

    # Initialize timing and progress tracking
    start_time = time.time()

    # Report setup times - Use f-strings
    if STATUS_MESSAGES:
        print(f"Timing Report (seconds):")
        print(f"  Parameter & Movie Setup: {time1 - time0:.3f}")
        print(f"  Psi 4 Conversion/Strain Load: {time2 - time1:.3f}")
        print(f"  BH Trajectories: {time3 - time2:.3f}")
        if APPARENT_HORIZONS:
            print(f"Apparent Horizon Load: {timeh - time3:.3f}")
        print(f"  Camera Parameters: {time4 - (timeh if APPARENT_HORIZONS else time3):.3f}")
        print(f"  Grid Init: {time5 - time4:.3f}")
        print(f"  Cosmetic Calcs: {time6 - time5:.3f}")
        print(f"  Mesh Construction: {time7 - time6:.3f}")
        print(f"  Animation Setup: {start_time - time7:.3f}")
        print(f"  Total Setup Time: {start_time - time0:.3f}")

    # --- Animation Loop ---
    # Use @mlab.animate decorator for potential interactive use,
    # but run it directly for offscreen rendering.
    #@mlab.animate(delay=10, ui=True)

    def anim():
        """
        Generator function to drive the animation frame by frame,
        generate corresponding audio, and merge them.
        """
        current_percent = 0

        if STATUS_MESSAGES:
            print(f"Starting video frame generation for: {movie_path_name}")

        with imageio.get_writer(movie_path_name, fps=frames_per_second, codec="libx264", quality=8) as writer:
            for idx, time_idx in enumerate(valid_indices):
                # --- Status Update & ETA ---
                if idx == 10: # Estimate after 10 frames
                    end_time = time.time()
                    eta = (end_time - start_time) * n_frames / 10
                    print(
                        f"""\nCreating {n_frames} frames and saving them to:
{movie_path_name}\nEstimated time: {dhms_time(eta)}"""  # <-- MODIFIED: Show silent path
                    )

                # Update progress percent
                if STATUS_MESSAGES and time_idx !=0 and current_percent < len(percentage_thresholds) and time_idx > percentage_thresholds[current_percent]:
                    eta = ((time.time() - start_time) / time_idx) * (n_times - time_idx)
                    print(f"\r{int(time_idx * 100 / n_times)}% done, {dhms_time(eta)} remaining", end="", flush=True)
                    current_percent +=1

                # --- Update Scene Objects ---
                if APPARENT_HORIZONS:
                    horizon_idx = find_idx(horizon_times, position_to_horizon * (equal_times[time_idx] + ext_rad_num))[0]
                    update_mesh_data(bh1_source, horizon_names[str(horizon_times[horizon_idx])][0], bh_scaling_factor)
                    try:
                        update_mesh_data(bh2_source, horizon_names[str(horizon_times[horizon_idx])][1], bh_scaling_factor)
                    except IndexError:
                        try:
                            bh2_mesh.visible = False
                        except ValueError:
                            pass
                else:
                    bh1.actor.actor.position = bh1_x[time_idx], bh1_y[time_idx], bh1_z[time_idx] # Update first bh position
                    bh2.actor.actor.position = bh2_x[time_idx], bh2_y[time_idx], bh2_z[time_idx] # Update second bh position
                    # Rescale bh2 if black holes have merged to represent combined object (at the specific frame index)
                    if idx == merge_rescale_idx:
                        # For a sphere, equally scale in all directions
                        bh2.actor.actor.scale = bh_merged_radius, bh_merged_radius, bh_merged_radius

                if TRAJECTORY_LINES and time_idx > 0:
                    bh1_trajectory.mlab_source.reset(x=bh1_x[:time_idx], y=bh1_y[:time_idx], z=bh1_z[:time_idx])
                    bh2_trajectory.mlab_source.reset(x=bh2_x[:time_idx], y=bh2_y[:time_idx], z=bh2_z[:time_idx])

                strain_slice = strain_to_mesh[..., time_idx].ravel() 
                np_points[:, 2] = np.where(valid_mask, strain_slice, np.nan)
                vtk_array.modified()
                strain_array.from_array(strain_slice[valid_mask])
                grid._set_points(points)
                grid.modified()

                # --- Update Camera ---
                mlab.view(
                    azimuth=azi_angle,
                    elevation=elevations[time_idx], 
                    distance=distances[time_idx], 
                    focalpoint=(0, 0, 0) 
                )

                # --- Save Frame --
                frame = mlab.screenshot(antialiased=True) 
                imageio.imwrite(frame_filenames[idx], frame)

                # --- Append Frame to Movie ---
                try:
                    writer.append_data(frame) # Write scene to movie
                except Exception as e:
                    print(f"\nError saving frame {frame_filenames[idx]} to movie: {e}")

        # --- End of frame loop and 'with' block ---
        print("\nVideo frame generation complete.")

        # --- End of Loop ---
        mlab.close(all=True) # Close the Mayavi figure/engine
        print("\nDone", flush=True) # Newline after progress bar

    # Run the animation script
    _ = anim()
    mlab.show()

    # --- Generate audio file (if requested) ---

    if FREQ_SOUND:

        # Calculate sample rate with inverse of timestep
        movie_length = n_valid / frames_per_second
        dt = 1 / frames_per_second
        target_rate = 48000

        # Normalize Strain Data (Safety check) - Remove DC offset and normalize to avoid numerical issues during Hilbert
        sound_strain = strain_to_mesh[0, 0, :][valid_indices]
        sound_strain = sound_strain - np.mean(sound_strain)

        # Normalize strain data
        if np.max(np.abs(sound_strain)) > 0:
             sound_strain = sound_strain / np.max(np.abs(sound_strain))

        # We pad with a reflection of the data to maintain continuity.
        # 10% padding on each side is usually sufficient.
        pad_length = int(n_valid * 0.1)
        sound_strain_padded = np.pad(sound_strain, (pad_length, pad_length), mode='reflect')
        pad_movie_length = 1.2 * movie_length

        # Extract Instantaneous Properties using the Hilbert Transform
        # Separate envelope and phase can be extracted from analytic signal
        analytic_signal = hilbert(sound_strain_padded)

        # Extract Amplitude Envelope
        amplitude_envelope = np.abs(analytic_signal)

        # Instantaneous phase of signal, unwrapped to remove 2 * pi jumps
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Derivative of phase is angular frequency, divide by 2 * pi
        # to get angular frequency in cycles per sample (time_idx)
        instantaneous_freq = np.gradient(instantaneous_phase) * frames_per_second / (2 * np.pi)

        # Scale freq based on the valid region
        valid_freqs = instantaneous_freq[pad_length:-pad_length]
        freq_scaling = 8000 / np.max(valid_freqs) # Scales up to 10000 Hz, a relatively high note

        # Ensure only frequencies for saved frames are considered
        audio_freq = instantaneous_freq * freq_scaling

        # Clamp negative frequencies (can happen due to noise/numerical artifacts)
        audio_freq = np.maximum(audio_freq, 0)

        # Calculate target samples including the padding
        num_audio_samples_padded = int(pad_movie_length * target_rate)

        # Resample to fit the required audio bitrate & movie length
        audio_freq_upsampled = resample(audio_freq, num_audio_samples_padded)
        amplitude_upsampled = resample(amplitude_envelope, num_audio_samples_padded)

        # Synthesize the Sound Wave
        # Must integrate frequency to get the new phase, Phase_new = Cumulative Sum of (Frequency * Time_Step * 2pi)
        dt_audio = 1 / target_rate
        new_phase = np.cumsum(audio_freq_upsampled * dt_audio * 2.0 * np.pi)

        # Generate the new audio signal: Amplitude * sin(Phase), and slice only for valid indices
        raw_audio_padded = amplitude_upsampled * np.sin(new_phase)

        # We need to map the original pad_length (in input samples) to output samples
        pad_length_audio = int(1 / 12 * num_audio_samples_padded)

        # Slice the center, removing the edge artifacts
        audio_resampled = raw_audio_padded[pad_length_audio : -pad_length_audio]

        # Even with padding, a hard start/stop can cause a "click".
        # Apply a 50ms fade in/out.
        fade_len = int(0.05 * target_rate) # 50ms
        if len(audio_resampled) > 2 * fade_len:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            audio_resampled[:fade_len] *= fade_in
            audio_resampled[-fade_len:] *= fade_out

        # Final Audio Formatting
        # Normalize to 16-bit integer range for WAV format, leave a little headroom (0.95) to prevent clipping
        max_signal = np.max(np.abs(audio_resampled))

        if max_signal > 0:
            audio_resampled = audio_resampled / max_signal

        audio_int16 = (audio_resampled * 32767 * 0.95).astype(np.int16)

        # Setup file names and paths
        audio_path = f"{movie_path}_audio.wav"
        movie_with_audio = f"{movie_path}_sound.mp4"

        # Write to file
        try:
            write_wav(audio_path, target_rate, audio_int16)
            print("Audio generation complete.")
        except Exception as e:
            print(f"\nAn error occurred during audio generation: {e}")
            print("Skipping audio merging.")
            return # Exit if audio failed

        # --- Merge Video and Audio ---
        if STATUS_MESSAGES:
            print(f"Merging video and audio...")
        try:
            # Setup video clip and audio clip
            video_clip = VideoFileClip(movie_path_name)
            audio_clip = AudioFileClip(audio_path)

            # Set the audio of the video clip, with durations equal
            final_clip = video_clip.with_audio(audio_clip.with_duration(video_clip.duration))

            # Write the final file
            final_clip.write_videofile(
                movie_with_audio,
                fps=frames_per_second,
                codec="libx264",
                audio_codec="aac",  # Common audio codec for mp4
                logger='bar'        # Show a progress bar
            )

            final_clip.close()
            video_clip.close()
            audio_clip.close()

            # Clean up temporary files
            os.remove(audio_path)
            os.remove(movie_path_name)

        except Exception as e:
             print(f"\nAn error occurred during merging: {e}")
             print("Please check your 'moviepy' and 'ffmpeg' installation.")
             print(f"Your silent video is at: {movie_path_name}")
             print(f"Your audio file is at: {audio_path}")

    # Use f-strings for final messages
    total_time = time.time() - start_time
    print(
        f"\nSaved {n_frames} frames to {movie_file_path} in {dhms_time(total_time)}.")
    print(f"Movie saved to {movie_with_audio if FREQ_SOUND else movie_path_name}")
    sys.exit(0)

# --- Doctest and Main Execution Guard ---
if __name__ == "__main__":
    # run doctests first
    import doctest

    # Test functions in this module
    results = doctest.testmod(verbose=False) # Set verbose=True for detailed output

    # Test functions in the imported psi4strain module
    try:
        p4s_results = doctest.testmod(psi4strain, verbose=False)
    except Exception as e:
         raise RuntimeError(f"Could not run doctests for psi4strain module: {e}")
         sys.exit(1)

    if p4s_results.failed > 0:
        # Use f-string for error message
        print(
            f"""Doctest in {psi4strain} failed:
{p4s_results.failed} of {p4s_results.attempted} test(s) passed"""
        )
        sys.exit(1)
    else:
        # Use f-string for success message
        print(
            f"""Doctest in {psi4strain} passed:
All {p4s_results.attempted} test(s) passed"""
        )

    if results.failed > 0:
        # Use f-string for error message
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        exit(1)
    else:
        # Use f-string for success message
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    # Run main function only if all doctests passed
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError, IndexError) as e:
        # Catch expected errors from main() and print cleanly
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         # Catch unexpected errors
         print(f"\nAn unexpected error occurred during main execution: {e}", file=sys.stderr)
         # Optionally print traceback for debugging unexpected errors
         traceback.print_exc()
         sys.exit(1)
