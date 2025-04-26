"""
A script that reads in gravitational wave psi 4 data and black hole positional data,
applies spin-weighted spherical harmonics to the data, and creates a Mayavi animation
of the black holes and their gravitational waves. At each state, the black holes are 
moved to their respective positions and the render is saved as a .png file.
"""

import os
import sys
import time
import psutil
from math import erf
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.special import erf  # Vectorized error function for arrays
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

# Default parameters used when USE_SYS_ARGS is False
BH_DIR = "../data/GW150914_data/r100" # changeable with sys arguments
MOVIE_DIR = "../data/GW150914_data/movies" # changeable with sys arguments
S_MODE = -2
EXT_RAD = 100 # changeable with sys arguments
USE_SYS_ARGS = True # Change to turn on/off default parameters
STATUS_MESSAGES = True # Change to turn on/off status reports during rendering

def swsh_summation_angles(colat: float, azi: NDArray[np.float64], mode_data: NDArray[np.complex128], ell_min: int, ell_max: int) -> NDArray[np.complex128]:
    """
    Sum all the strain modes after factoring in the
    corresponding spin-weighted spherical harmonic
    for the specified angles in the mesh. Stored as an array corresponding to [angle, time] indices.

    :param colat: Colatitude angle for the SWSH factor.
    :param azi: Azimuthal angles for the SWSH factor.
    :param mode_data: Numpy array containing strain data for all the modes, shape (n_modes, n_times).
    :param ell_min: Minimum l mode to use in SWSH
    :param ell_max: Maximum l mode to use in SWSH
    :return: A complex valued numpy array of the superimposed wave, shape (n_azi_pts, n_times).

    DocTests:
    >>> mode_data = np.zeros((77, 3), dtype=complex)
    >>> mode_idx = 0
    >>> for l in range(2, 9):
    ...     for m in range(-l, l+1):
    ...         mode_data[mode_idx] = np.array([1+1j, 2+3j, 4+5j])
    ...         mode_idx += 1
    >>> np.round(swsh_summation_angles(np.pi/2, np.array([0]), mode_data, 2, 8), 5)
    array([[ 4.69306 +4.69306j,  9.38612+14.07918j, 18.77224+23.4653j ]])
    """

    quat_arr = quaternionic.array.from_spherical_coordinates(colat, azi)
    winger = spherical.Wigner(ell_max, ell_min)
    # Create an swsh array shaped like (n_modes, n_quaternions)
    swsh_arr = winger.sYlm(S_MODE, quat_arr).T
    # mode_data has shape (n_modes, n_times), swsh_arr has shape (n_modes, n_pts).
    # Pairwise multiply and sum over modes: the result has shape (n_pts, n_times).
    pairwise_product = mode_data[:, np.newaxis, :] * swsh_arr[:, :, np.newaxis]
    return np.sum(pairwise_product, axis=0)

def generate_interpolation_points(
    time_array: NDArray[np.float64],
    radius_values: NDArray[np.float64],
    r_ext: float,
) -> NDArray[np.float64]:
    """
    Fill out a 2D array of adjusted time values for the wave strain to be
    linearly interpolated to. First index of the result represents the radial
    distance index, and the second index represents the simulation time index (aka which mesh).

    :param time_array: Numpy array of strain time indices.
    :param radius_values: Numpy array of the radial points on the mesh.
    :param r_ext: Extraction radius of the original data.
    :return: A 2D numpy array (n_radius, n_times) of time values.

    DocTests:
    >>> time_array = np.array([0.0, 1.0, 2.0, 3.0])
    >>> radius_values = np.array([10.0, 20.0])
    >>> r_ext = 5.0
    >>> generate_interpolation_points(time_array, radius_values, r_ext)
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    """

    # Precompute min and max of time_array
    time_min, time_max = time_array.min(), time_array.max()

    # Use broadcasting directly in the computation
    target_times = time_array[np.newaxis, :] - radius_values[:, np.newaxis] + r_ext

    # Clip the values in-place
    np.clip(target_times, time_min, time_max, out=target_times)

    return target_times

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

def change_object_position(obj: Surface, position: tuple[float, float, float]) -> None:
    """
    Change the Cartesian position of a Mayavi surface to the given coordinates.

    :param obj: Mayavi Surface object to reposition
    :param position: New (x, y, z) position as a tuple of floats

    DocTests: Requires a Mayavi Surface object, difficult to test standalone. Will be skipped.
    """

    obj.actor.actor.position = position  # Direct tuple assignment (no numpy conversion needed)

def rescale_object(obj: Surface, size: float) -> None:
    """
    Rescale a Mayavi surface equally in all directions by a given factor.

    :param obj: Mayavi Surface object to reposition
    :param size: Scale factor to multiply dimensions by

    DocTests: Requires a Mayavi Surface object, difficult to test standalone. Will be skipped.
    """

    obj.actor.actor.scale = (size, size, size)  # Direct tuple assignment (no numpy conversion needed)

def dhms_time(seconds: float) -> str:
    """
    Convert a given number of seconds into a string indicating the remaining time.

    :param seconds: Number of seconds.
    :return: A string indicating the remaining time (days, hours, minutes).

    DocTests:
    >>> dhms_time(90061)
    '1 days 1 hours 1 minutes'
    >>> dhms_time(3600)
    '1 hours'
    >>> dhms_time(59)
    ''
    >>> dhms_time(3665)
    '1 hours 1 minutes'
    """

    divisors = (
        (86400, "days"),
        (3600, "hours"),
        (60, "minutes"),
    )
    parts = []
    remaining = seconds
    for divisor, label in divisors:
        value = int(remaining // divisor)
        remaining = remaining % divisor
        if value > 0:
            parts.append(f"{value} {label}")
    return " ".join(parts)

def convert_to_movie(input_path: str, movie_name: str, fps: int = 24, status_messages: bool = True) -> None:
    """
    Convert a series of .png files into a movie using imageio.

    :param input_path: Path to the directory containing the .png files.
    :param movie_name: Name of the movie file (without extension).
    :param fps: Frames per second (default is 24).
    :param status_messages: Whether to print progress messages (default is True).

    Doctests: Requires an Imageio writer object, difficult to test standalone. Will be skipped.
    """

    # Efficient directory scanning with immediate sorting and filtering
    try:
        with os.scandir(input_path) as it:
            # Filter for PNG files specifically
            image_files = sorted([entry.name for entry in it if entry.is_file() and entry.name.lower().endswith('.png')])
    except FileNotFoundError:
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    if not image_files:
        print(f"No PNG files found in {input_path}. Movie not created.")
        return

    output_path = os.path.join(input_path, f"{movie_name}.mp4")
    total_frames = len(image_files)

    # Use context manager for automatic resource cleanup
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        quality=8,
        macro_block_size=None
    ) as writer:

        prev_progress = -1
        input_path_obj = os.fspath(input_path)  # Cache path for faster joins

        for i, img_file in enumerate(image_files, 1):
            # Directly construct path string for maximum speed
            file_path = f"{input_path_obj}/{img_file}"
            try:
                writer.append_data(imageio.imread(file_path))
            except FileNotFoundError:
                print(f"Warning: Image file not found: {file_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: Error reading image file {file_path}: {e}. Skipping.")
                continue

            if status_messages:
                # Integer-based progress tracking for minimal computation
                current_progress = (i * 1000) // total_frames
                if current_progress != prev_progress:
                    # Use f-string for formatting
                    print(f"\rProgress: {current_progress/10:.1f}% completed", end="", flush=True)
                    prev_progress = current_progress

        if status_messages:
            print("\rProgress: 100.0% completed", flush=True) # Ensure 100% is shown

def max_strain_values(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Identify peak strain values where increasing/decreasing trends change.

    :param data: Numpy 2D array of strain values with column 0 as time and column 1 as strain magnitude.
    :return: Numpy 2D array of local maxima in data with column 0 as the value's index in the original data,
             column 1 as the time, and column 2 as the absolute max strain value.

    DocTests:
    >>> data = np.array([[0., 0.], [1., 5.], [2., 3.], [3., 4.], [4., 2.], [5., 6.]])
    >>> max_strain_values(data)
    array([[1., 1., 5.],
           [3., 3., 4.],
           [5., 5., 6.]])
    >>> data_flat = np.array([[0., 2.], [1., 2.], [2., 2.]])
    >>> max_strain_values(data_flat)
    array([[2., 2., 2.]])
    >>> data_decreasing = np.array([[0., 5.], [1., 4.], [2., 3.]])
    >>> max_strain_values(data_decreasing)
    array([[0., 0., 5.]])
    """
    if data.shape[0] < 2:
        if data.shape[0] == 1:
             # Return the single point if only one exists
             return np.array([[0., data[0, 0], np.abs(data[0, 1])]])
        else:
             return np.empty((0, 3)) # Return empty if no data

    strain_vals = data[:, 1]
    times = data[:, 0]
    peaks = []

    prev_val = strain_vals[0]
    # Start by assuming it's increasing from negative infinity or flat
    increasing = True if strain_vals.shape[0] <= 1 or strain_vals[1] >= strain_vals[0] else False

    # Add the first point if it's a peak relative to the start
    if not increasing:
         peaks.append((0, times[0], abs(prev_val)))

    for t in range(1, len(strain_vals)):
        current_val = strain_vals[t]

        if increasing:
            if current_val < prev_val:  # Peak detected (transition from increasing to decreasing)
                peaks.append((t-1, times[t-1], abs(prev_val)))
                increasing = False
        else: # Was decreasing
            if current_val > prev_val: # Valley detected (transition from decreasing to increasing)
                increasing = True
            elif current_val == prev_val: # Plateau after decrease - consider previous point the peak
                 pass # Don't add the current point as a peak yet

        prev_val = current_val

    # Add the last point if the trend was increasing towards the end
    if increasing:
        peaks.append((len(strain_vals)-1, times[-1], abs(strain_vals[-1])))

    return np.array(peaks)

def get_local_maxima(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Identify local maxima in already peak-detected strain data.

    Assumes input data contains points representing peaks from `max_strain_values`.

    :param data: Numpy 2D array of local maxima strain values with column 0 as the original index,
             column 1 as the time, and column 2 as the absolute strain value.
    :return: Numpy 2D array of local maxima among the input peaks, with the same column structure.
             Returns empty if input has < 3 points.

    DocTests:
    >>> peaks = np.array([[1., 1., 5.], [3., 3., 4.], [5., 5., 6.]])
    >>> get_local_maxima(peaks)
    array([[1., 1., 5.],
           [5., 5., 6.]])
    >>> peaks_single = np.array([[1., 1., 5.]])
    >>> get_local_maxima(peaks_single)
    array([], shape=(0, 3), dtype=float64)
    >>> peaks_two = np.array([[1., 1., 5.], [3., 3., 4.]])
    >>> get_local_maxima(peaks_two)
    array([], shape=(0, 3), dtype=float64)
    >>> peaks_three = np.array([[1., 1., 5.], [3., 3., 4.], [5., 5., 3.]])
    >>> get_local_maxima(peaks_three)
    array([[1., 1., 5.]])
    """
    if data.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64) # Need at least 3 points for a local maximum

    orig_indices = data[:, 0]
    time_col = data[:, 1]
    values = data[:, 2]
    local_max = []

    # Check the first point
    if values[0] >= values[1]:
        local_max.append([orig_indices[0], time_col[0], values[0]])

    # Check intermediate points
    for t in range(1, len(values) - 1):
        if values[t] >= values[t-1] and values[t] >= values[t+1]:
            # Add only if it's strictly greater than one neighbor or equal to both
            if values[t] > values[t-1] or values[t] > values[t+1] or (values[t] == values[t-1] and values[t] == values[t+1]):
                 local_max.append([orig_indices[t], time_col[t], values[t]])

    # Check the last point
    if values[-1] >= values[-2]:
        local_max.append([orig_indices[-1], time_col[-1], values[-1]])

    # Remove duplicates if plateaus caused multiple entries for the same peak time/value
    if local_max:
        unique_max = np.unique(np.array(local_max, dtype=np.float64), axis=0)
        return unique_max
    else:
        return np.empty((0, 3), dtype=np.float64)

def local_max_iterative(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Iteratively find local maxima of data until doing so further would reduce
    the size of the dataset by more than half.

    :param data: Numpy 2D array of local maxima strain values (output from `max_strain_values` or previous iteration)
                 with column 0 as the original index, column 1 as the time, and column 2 as the strain value.
    :return: Numpy 2D array of strain values found by iterating searches for local maxima, with the same column structure.
    Maximizing the data beyond this
    point might cause significant reductions in the information.

    DocTests:
    >>> data = np.array([
    ... [1, 1, 5], [3, 3, 4], [5, 5, 6], [7, 7, 5], [9, 9, 7],
    ... [11, 11, 6], [13, 13, 8], [15, 15, 7]
    ... ])
    >>> local_max_iterative(data) # First iter: [1,5,9,13]; Second iter: [1,9,13]. 2*3 > 4 is True. Return iter1.
    array([[ 1.,  1.,  5.],
           [ 5.,  5.,  6.],
           [ 9.,  9.,  7.],
           [13., 13.,  8.]])
    """

    current = data
    if current.shape[0] < 3: # Cannot find local maxima with less than 3 points
        return current

    while True:
        max_data = get_local_maxima(current)
        if max_data.shape[0] < 3: # Cannot iterate further
            return current # Return the previous iteration's result

        max_max_data = get_local_maxima(max_data)

        # Stop if the next iteration significantly reduces data points
        if 2 * max_max_data.shape[0] > max_data.shape[0] or max_max_data.shape[0] < 3:
            # If max_data has few points, max_max_data might be empty or too small,
            # leading to 0 > N (false) or small_num > N (false).
            # The condition max_max_data.shape[0] < 3 handles this edge case.
            # We return max_data because applying get_local_maxima again (to get max_max_data)
            # reduced the size too much according to the 2* criterion OR resulted in too few points.
             return max_data
        else:
             current = max_data # Continue iteration

def get_max_max(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Iteratively find local maxima of data until doing so
    further would reduce the size of the dataset to 4 or fewer points.

    :param data: Numpy 2D array of local maxima strain values (output from `max_strain_values` or previous iteration)
                 with column 0 as the original index, column 1 as the time, and column 2 as the strain value.
    :return: Numpy 2D array of strain values found by iterating searches for local maxima, with the same column structure.
    Maximizing the data beyond this
    point might cause a loss of information rendering the data less reliable for certain analyses.

    DocTests:
    >>> data = np.array([
    ... [1, 1, 5], [3, 3, 4], [5, 5, 6], [7, 7, 5], [9, 9, 7],
    ... [11, 11, 6], [13, 13, 8], [15, 15, 7], [17, 17, 9], [19, 19, 8]
    ... ]) # Iter1: [1,5,9,13,17]; Iter2: [1,9,17]. Shape is 3 <= 4. Stop.
    >>> get_max_max(data)
    array([[ 1.,  1.,  5.],
           [ 5.,  5.,  6.],
           [ 9.,  9.,  7.],
           [13., 13.,  8.],
           [17., 17.,  9.]])
    """

    current = data
    if current.shape[0] <= 4: # Return immediately if already small
         return current

    while True:
        next_max = get_local_maxima(current)
        if next_max.shape[0] <= 4 or next_max.shape[0] == current.shape[0]:
            # Stop if size is small enough or if no reduction occurred
            # (could happen with flat peaks)
            break
        current = next_max
    return current

def get_strain_after_burst(data: NDArray[np.float64], end_of_burst_value: float) -> float:
    """
    Get the strain value in the peak data immediately after the radiation burst has ended.

    Finds the first peak *after* the peak identified as the end of the burst.

    :param data: Numpy 2D array of local maxima strain values (output of `max_strain_values`)
                 with column 0 as original index, column 1 as time, and column 2 as absolute strain value.
    :param end_of_burst_value: The absolute value of the strain at the designated end of the radiation burst
                               (must exist in data[:, 2]).
    :return: The absolute strain value of the first peak immediately following the end_of_burst_value peak.
             Returns 0 if no such peak exists or data is too small.

    DocTests:
    >>> data = np.array([[1., 1., 5.], [3., 3., 4.], [5., 5., 6.], [7., 7., 3.], [9., 9., 2.]])
    >>> get_strain_after_burst(data, 6.0) # Burst ends at peak value 6.0, next peak is 3.0
    3.0
    >>> get_strain_after_burst(data, 3.0) # Burst ends at peak value 3.0, next peak is 2.0
    2.0
    >>> get_strain_after_burst(data, 2.0) # Burst ends at last peak, no peak after
    0.0
    >>> data_short = np.array([[1., 1., 5.], [3., 3., 4.]])
    >>> get_strain_after_burst(data_short, 5.0) # Returns the next value even if only two points
    4.0
    >>> get_strain_after_burst(data_short, 4.0)
    0.0
    """

    if data.shape[0] < 2: # Need at least two points to find one after another
        return 0.0
    try:
        # Find the index of the row where the strain value matches end_of_burst_value
        end_of_burst_indices = np.where(data[:, 2] == end_of_burst_value)[0]
        if end_of_burst_indices.size == 0:
             print(f"Warning: end_of_burst_value {end_of_burst_value} not found in data.")
             return 0.0
        end_of_burst_index = end_of_burst_indices[0] # Use the first occurrence if multiple

        # Check if this is the last point in the array
        if end_of_burst_index >= data.shape[0] - 1:
            return 0.0 # No points after the burst end

        # Return the strain value of the next point
        return data[end_of_burst_index + 1, 2]

    except IndexError:
         # Should not happen with the checks above, but for safety
         print("Warning: Index error occurred in get_strain_after_burst.")
         return 0.0

def calculate_zoomout_time(data: NDArray[np.float64]) -> float:
    """
    Calculate the time the visualization should start zooming out based on major peaks.

    Finds the two largest peaks in the iteratively-maximized data and calculates
    a time point shortly after the second largest peak, weighted towards the largest.

    :param data: Numpy 2D array of iteratively maximized local maxima strain values
                 (output from `get_max_max` or `local_max_iterative`) with column 0 as original index,
                 column 1 as time, and column 2 as the strain value.
    :return: The time at which to begin zooming out in the visualization. Returns infinity if fewer than 2 peaks are found.

    DocTests:
    >>> data = np.array([[1, 1, 5.], [9, 9, 7.], [17, 17, 9.]]) # Peaks at t=1,9,17; vals=5,7,9
    >>> calculate_zoomout_time(data) # max1=(17,9), max2=(9,7). time = 9 + (17+9)/4 = 9 + 6.5 = 15.5
    15.5
    >>> data_two_peaks = np.array([[1, 1, 5.], [9, 9, 7.]])
    >>> calculate_zoomout_time(data_two_peaks) # max1=(9,7), max2=(1,5). time = 1 + (9+1)/4 = 1 + 2.5 = 3.5
    3.5
    >>> data_one_peak = np.array([[1, 1, 5.]])
    >>> calculate_zoomout_time(data_one_peak)
    inf
    >>> data_empty = np.empty((0,3))
    >>> calculate_zoomout_time(data_empty)
    inf
    """

    values = data[:, 2]
    times = data[:, 1] # Get the time column

    if values.size < 2: # Need at least two peaks
        return float('inf')

    # Find the index and time of the largest peak (max1)
    max1_idx = np.argmax(values)
    max1_time = times[max1_idx]
    max1_val = values[max1_idx]

    # Find the index and time of the second largest peak (max2)
    # Temporarily set max1 value to negative infinity to find the next max
    temp_values = np.copy(values)
    temp_values[max1_idx] = -np.inf
    max2_idx = np.argmax(temp_values)
    max2_time = times[max2_idx]
    # max2_val = values[max2_idx] # We don't actually need the value

    # Calculate zoomout time: Start at the time of the second peak,
    # then add a quarter of the sum of the peak times.
    # This biases the zoomout start towards the larger peak but ensures it happens after the second peak.
    zoom_time = max2_time + (max1_time + max2_time) / 4

    return zoom_time

def extract_max_strain_and_zoomout_time(
    dir_path: str,
    r_ext: int,
    ell_min: int,
    ell_max: int
) -> Tuple[float, float, float]:
    """
    Calculate the overall maximum strain, the strain immediately after the burst,
    and the earliest zoomout time across all l,m modes in the input directory.

    :param dir_path: The directory housing the converted strain data files (e.g., Rpsi4_r..._l..._conv_to_strain.txt).
    :param r_ext: The radius of extraction of the data (used to construct filenames).
    :param ell_min: The minimum l to be used in the calculations
    :param ell_max: The maximum l to be used in the calculations
    :return: A tuple containing:
             - The maximum strain value found immediately after the burst across all processed modes.
             - The overall maximum strain value found in the iteratively maximized data across all processed modes.
             - The minimum (earliest) zoomout time calculated across all processed modes.
    :raises FileNotFoundError: If no valid strain files are found for the given l range.
    """
    # This function requires reading specific files. Cannot be tested with doctest without mocking/setup.
    overall_strain_after_burst = float('-inf')
    min_zoomout_time = float('inf')
    overall_max_strain = float('-inf')
    files_processed = 0

    for l in range(ell_min, ell_max + 1):
         # Construct filename safely
        filename = f"Rpsi4_r{0 if r_ext < 1000 else ''}{r_ext}_l{l}_conv_to_strain.txt"
        file_path = os.path.join(dir_path, filename)

        # Determine expected number of columns based on l
        # Time + (Real + Imaginary for each m from -l to l) = 1 + 2*(2l+1) = 4l+3 columns
        # usecols goes up to max_col index, so max_col should be 4l+2
        max_col_idx = 4 * l + 2
        cols_to_use = range(0, max_col_idx + 1) # Range includes 0, stops before max_col_idx + 1

        try:
            # Load data, skipping header lines dynamically
            # Header lines = 1 (comment) + (number of modes = 2l+1) * 2 (Re/Im lines)
            num_skip_rows = 4*l+3
            data_all = np.loadtxt(file_path, skiprows=num_skip_rows, usecols=cols_to_use)
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

        time_col = data_all[:, 0]

        # Iterate through valid m values for this l
        for m in range(-l, l + 1):
            # Calculate column indices for real and imaginary parts
            # Real part: col index 1 (time) + (m-(-l)) * 2 = 1 + 2*(m+l)
            # Imaginary part: col index + 1
            col_real = 2*l + 2*m + 1
            col_imag = col_real + 1

            # Ensure calculated indices are within the loaded data bounds
            if col_imag >= data_all.shape[1]:
                print(f"Warning: Calculated column index {col_imag} out of bounds for l={l}, m={m} in {file_path}. Skipping.")
                continue

            real = data_all[:, col_real]
            imag = data_all[:, col_imag]
            magnitude = np.hypot(real, imag) # Efficient calculation of sqrt(real^2 + imag^2)
            data_lm = np.column_stack((time_col, magnitude)) # Time and magnitude for this mode

            # --- Process this l,m mode ---
            max_strain_peaks = max_strain_values(data_lm)
            if max_strain_peaks.size == 0:
                continue # Skip if no peaks found

            data_local_max = local_max_iterative(max_strain_peaks)
            if data_local_max.size == 0:
                 continue # Skip if iterative max is empty

            data_max_max = get_max_max(data_local_max)
            if data_max_max.size == 0:
                continue # Skip if final max is empty

            # Calculate zoomout time for this mode
            zt = calculate_zoomout_time(data_max_max)

            # Calculate strain after burst for this mode
            # Burst ends at the first peak found by local_max_iterative
            end_burst_val = data_local_max[0, 2]
            sab = get_strain_after_burst(max_strain_peaks, end_burst_val)

            # Find the max strain value from the most reduced set for this mode
            mode_max_strain = np.max(data_max_max[:, 2]) if data_max_max.size > 0 else float('-inf')

            # Update overall values
            overall_strain_after_burst = max(overall_strain_after_burst, sab)
            min_zoomout_time = min(min_zoomout_time, zt)
            overall_max_strain = max(overall_max_strain, mode_max_strain)

    if files_processed == 0:
         raise FileNotFoundError(f"No valid strain files found in directory {dir_path} for l={ell_min} to {ell_max}.")

    # Handle cases where no valid peaks/times were found
    if overall_strain_after_burst == float('-inf'): overall_strain_after_burst = 0.0
    if overall_max_strain == float('-inf'): overall_max_strain = 0.0
    # min_zoomout_time remains inf if never updated

    return overall_strain_after_burst, overall_max_strain, min_zoomout_time


def compute_strain_to_mesh(
    strain_azi: NDArray[np.float32],
    equal_times: NDArray[np.float32], # Not used directly, but lerp_times depends on it
    radius_values: NDArray[np.float32], # Not used directly, but lerp_times depends on it
    lerp_times: NDArray[np.float32],  # Precomputed 2D array of interpolation points
    time_array: NDArray[np.float32],
    dropoff_2D_flat: NDArray[np.float32],
    use_symlog: bool,
    status_messages=True
) -> NDArray[np.float32]:
    """Interpolates strain data onto a polar mesh grid over specific time points.

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
    :param equal_times: Array of equally spaced times for the final desired mesh state. Shape: (n_equal_times,). Note:
                        Not used directly in calculations, but typically used to generate `lerp_times` and defines the
                        size of the time dimension in the output.
    :param radius_values: Array of radius values for the mesh grid. Shape: (n_rad_pts,). Note: Not used directly in
                          calculations, but typically used to generate `lerp_times` and defines the size of the radius
                          dimension in the output.
    :param lerp_times: Precomputed 2D array of interpolation time points. These are the target time coordinates (`x'`
                       values for `np.interp`) at which to evaluate the interpolated strain. Shape: (n_rad_pts,
                       n_equal_times).
    :param time_array: Original time points corresponding to the `strain_azi` data (the `x` coordinates for `np.interp`).
                       Shape: (n_original_times,).
    :param dropoff_2D_flat: Array of scaling factors applied to the strain data, typically dependent on radius.
                            Applied before interpolation. Shape: (n_rad_pts,).
    :param use_symlog: If True, apply a symmetric logarithmic transformation to the strain data before interpolation
    :param status_messages: If True, print progress messages and estimated chunk size.
    :return: A 3D array containing the interpolated strain values on the spatio-temporal mesh.
             Shape: (n_rad_pts, n_azi_pts, n_equal_times).

    Doctests:
    >>> # Setup parameters
    >>> n_rad_pts = 2
    >>> n_azi_pts = 3
    >>> n_original_times = 4
    >>> n_equal_times = 5
    >>>
    >>> # Setup dummy data
    >>> strain_azi = np.arange(n_azi_pts * n_original_times, dtype=np.float32).reshape((n_azi_pts, n_original_times))
    >>> # strain_azi = [[ 0.  1.  2.  3.], [ 4.  5.  6.  7.], [ 8.  9. 10. 11.]]
    >>> equal_times = np.linspace(0.5, 2.5, n_equal_times, dtype=np.float32) # 0.5, 1., 1.5, 2., 2.5
    >>> radius_values = np.array([10.0, 20.0], dtype=np.float32)
    >>> lerp_times = np.vstack((np.linspace(0.5, 2.5, n_equal_times), # 0.5, 1.0, 1.5, 2.0, 2.5
    ...                         np.linspace(0.2, 2.8, n_equal_times) # 0.2, 0.85, 1.5, 2.15, 2.8
    ... ))
    >>> time_array = np.arange(n_original_times, dtype=np.float32) # 0. 1. 2. 3.
    >>> dropoff_2D_flat = np.array([1.0, 0.5], dtype=np.float32)
    >>> # Run the function
    >>> result = compute_strain_to_mesh(
    ...     strain_azi, equal_times, radius_values, lerp_times,
    ...     time_array, dropoff_2D_flat, False, False
    ... )
    >>>
    >>> # Check shape
    >>> result.shape
    (2, 3, 5)
    >>>
    >>> # Check some values (calculated via np.interp manually)
    >>> print(result[0, 0, :])
    [0.5 1.  1.5 2.  2.5]
    >>> print(result[1, 1, :])
    [2.1   2.425 2.75  3.075 3.4  ]
    >>> print(result[0, 2, :])
    [ 8.5  9.   9.5 10.  10.5]
    """

    # Get the size of each array
    n_rad_pts = len(radius_values)
    n_azi_pts = strain_azi.shape[0]
    n_equal_times = len(equal_times)

    # Estimate chunk size based on available memory (heuristic)
    chunk_size = min(int(psutil.virtual_memory().available / 70000000), n_azi_pts)

    if status_messages:
         print(f"Using chunk size: {chunk_size} for azimuth interpolation.")

    strain_to_mesh = np.zeros((n_rad_pts, n_azi_pts, n_equal_times), dtype=np.float32)

    for start_idx in range(0, n_azi_pts, chunk_size):
        end_idx = min(start_idx + chunk_size, n_azi_pts)

        # Interpolate each azimuth in the current chunk
        for i, azi_idx in enumerate(range(start_idx, end_idx)):
            # Use np.interp for each radial profile corresponding to this azimuth
            # The values to interpolate *from* are strain_azi[azi_idx, :] at times time_array
            # The points to interpolate *to* are the times given by lerp_times[:, time_idx] for each time_idx
            # Result shape for one azi_idx should be (n_rad_pts, n_equal_times)
            for rad_idx in range(n_rad_pts):
                strain_to_mesh[rad_idx, azi_idx, :] = np.interp(
                lerp_times[rad_idx, :],  # x-coordinates to interpolate to (shape n_rad, n_equal_times) -> flattens automatically
                time_array, # Original x-coordinates (shape n_original_times)
                # Original y-values (shape n_original_times) scaled with symlog (if enabled) and dropoff factors
                dropoff_2D_flat[rad_idx] * (np.sign(strain_azi[azi_idx, :]) * np.log1p(np.abs(strain_azi[azi_idx, :])) if use_symlog else strain_azi[azi_idx, :]), 
                left=np.nan, # Value for x < time_array[0]
                right=np.nan # Value for x > time_array[-1]
                )
            # Update status
            if status_messages:
                # Calculate progress based on the outer loop index 'azi_idx'
                progress = (azi_idx + 1) / (n_azi_pts) * 100
                # Use f-string formatting
                print(f"\rProgress: {int(progress)}% completed", end="", flush=True)

    # Create a new line after status messages complete
    if status_messages:
        print() # Move to the next line

    return strain_to_mesh

def idx_time(array: NDArray[np.float64], time: float) -> int:
    """
    Calculate the index at which the value of a sorted array is closest to the value `time`.

    :param array: Numpy array of time values (assumed to be sorted).
    :param time: Target time to search for in the array.
    :return: The index of the element in `array` closest to `time`.

    DocTests:
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> idx_time(times, 2.1)
    2
    >>> idx_time(times, 1.9)
    2
    >>> idx_time(times, 3.0)
    3
    >>> idx_time(times, -1.0)
    0
    >>> idx_time(times, 5.0)
    4
    """

    if array.size == 0:
        raise ValueError("Input array cannot be empty.")

    # Find the index where `time` would be inserted to maintain order
    insert_idx = np.searchsorted(array, time, side='left')

    # Handle edge cases: time is before the first element or after the last
    if insert_idx == 0:
        return 0
    if insert_idx == len(array):
        return len(array) - 1

    # Check which neighbor is closer: the one before or the one at insert_idx
    left_neighbor_diff = time - array[insert_idx - 1]
    right_neighbor_diff = array[insert_idx] - time

    if left_neighbor_diff <= right_neighbor_diff:
        return insert_idx - 1
    else:
        return insert_idx

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
    time0 = time.time()
    global BH_DIR, MOVIE_DIR, EXT_RAD # Allow modification of globals based on args
    bh1_mass: float = 1.0 # Default mass
    bh2_mass: float = 1.24 # Default mass ratio for GW150914
    use_symlog: bool = False # Default scale
 
    if USE_SYS_ARGS:
        argc = len(sys.argv)
        if argc not in (2, 3):
            # Use raise RuntimeError for error exit
            raise RuntimeError(
                f"""Usage: python {sys.argv[0]} <path_to_data_folder> [use_symlog: True/False]
Example: python {sys.argv[0]} ../data/GW150914_data/r100 100 true

                Arguments:
                  <path_to_data_folder>: Path to the directory containing merger data and converted strain.
                  [use_symlog]: Optional. Use symmetric log scale for strain (True/False, default: False)."""
            ) # Use f-string for cleaner formatting
        else:
            # Change directories and extraction radius based on inputs
            bh_dir = str(sys.argv[1])

            # Set psi4_output_dir relative to bh_dir
            psi4_output_dir = os.path.join(bh_dir, "converted_strain")
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
    if not os.path.isdir(bh_dir):
        raise FileNotFoundError(f"Data directory not found: {bh_dir}")
    if not os.path.isdir(psi4_output_dir):
         print(f"Warning: Converted strain directory not found: {psi4_output_dir}. Attempting to load strain files from data directory.")
         # Allow fallback to bh_dir if converted_strain doesn't exist,
         # but psi4strain function needs to handle this path correctly.
         # For extract_max_strain_and_zoomout_time, we pass the expected dir.
         # If psi4strain.psi4_ffi_to_strain needs the base dir, adjust its call.

    # --- Movie File Path Handling ---
    bh_file_name = "puncture_posns_vels_regridxyzU.txt"
    bh_file_path = os.path.join(bh_dir, bh_file_name)
    if not os.path.isfile(bh_file_path):
         raise FileNotFoundError(f"Black hole position file not found: {bh_file_path}")

    bh_scaling_factor = 1.0 # Scaling for BH visualization size

    movie_number = 1
    while True:
        movie_dir_name = f"real_movie{movie_number}"
        movie_file_path = os.path.join(movie_dir, movie_dir_name)

        if os.path.exists(movie_file_path):
            # Ask the user for permission to override existing file with same name
            response = input(f"{movie_file_path} already exists. Would you like to overwrite it? Y/N: ")
            if response.lower() != 'y':
                movie_number += 1
                continue

            # User confirmed overwrite
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
                # Proceed to create it again.
                continue
            except Exception as e:
                raise RuntimeError(f"Error clearing directory {movie_file_path}: {e}")

        # If no directory of the same name is present, create one (and parent movie_dir if needed)
        try:
            os.makedirs(movie_file_path, mode=0o755, exist_ok=True) # exist_ok handles race condition if created between ch>            print(f"Output will be saved in: {movie_file_path}")
            break # Exit loop after successful creation or confirmation
        except OSError as e:
            raise RuntimeError(f"Could not create output directory {movie_file_path}: {e}")

    # --- Extraction Radius Calculations ---
    bh_file_list = os.listdir(bh_dir) # Extract the files in the black hole directory
    bh_files = [f for f in bh_file_list if os.path.isfile(os.path.join(bh_dir, f))] # List the names of these files

    extraction_radii = np.empty(0)
    for b in bh_files:
        # Attempt to convert the part of the file name that is supposed to be the extraction radius into a float
        try:
            radius_extraction = float(b[-10:-4])
        except ValueError:
            continue # Skip over files that fail or don't have an extraction radius
        if radius_extraction not in extraction_radii:
            extraction_radii = np.append(extraction_radii, radius_extraction) # Save the extraction radius if unique

    size = len(extraction_radii)
    if size == 1:
        ext_rad = extraction_radii[0] # If only one extraction radius is found, use that one
    elif size == 0:
        # If no extraction radii are found, the files are probably incorrectly named
        raise RuntimeError("No psi 4 files found in the directory. Ensure psi 4 files are formatted as such: Rpsi4_l#-r####.#")
    elif size > 1:
        # Handle the case where multiple extraction radii are found
        print("Warning: Multiple extraction radii found in the directory.")
        while True:
            response = input("Please enter the extraction radius you would like to use: ")
            try:
                # Attempt to parse user input into extraction radius float
                radius_extraction = float(response)
                # Print availabe extraction radii if user inputs one that is unavailable
                if radius_extraction not in extraction_radii:
                    print("Available extraction radii:")
                    for r in extraction_radii:
                        print(r)
                else:
                    break # End the loop if a valid extraction radius has been entered
            except ValueError:
                print("Please enter a float from 0.0 to 9999.0.") # Handle the case where something else was entered
        ext_rad = radius_extraction
    print(f"Using extraction radius {ext_rad} for psi 4 data")

    # --- Minimum and Maximum Ell Mode Calculations ---
    ells = np.empty(0)
    # Convert extraction radius used into a properly formatted string ####.# to determine which files to search
    str_ext_rad = ("0" if ext_rad < 1000 else "") + str(ext_rad)
    for b in bh_files:
        # Only search files with the appropriate extraction radius
        if b[-10:-4] == str_ext_rad:
            # Attempt to convert the part of the file name that is supposed to be the mode into an integer
            try:
                ell = float(b[7])
            except ValueError:
                continue # Skip over files that fail or don't have a mode
            if ell not in ells:
                ells = np.append(ells, ell) # Save the ell mode if unique

    ells = np.sort(ells.astype(int)) # Put the ell modes in order
    if len(ells) == 0:
        # If no modes are found, the files are probably incorrectly named
        raise RuntimeError("No psi 4 files found in the directory. Ensure psi 4 files are formatted as such: Rpsi4_l#-r####.#")
    # Extract the min and max ells
    ell_min = ells[0]
    ell_max = ells[-1]

    # Calculate the difference between consecutive ells to detect gaps
    diff_ells = np.diff(ells)
    gaps = np.where(diff_ells > 1)
    # If there are any gaps, raise an error
    if gaps[0].size > 0:
        raise RuntimeError(f"A gap was detected in the l modes: Minimum l is {ell_min}, maximum l is {ell_max}, but no l={gaps[0][0] + ell_min + 1} file was found")

    print(f"Using minimum mode l={ell_min} and maximum mode l={ell_max} for psi 4 data")

    # --- Simulation & Visualization Parameters ---
    display_radius = 300  # radius for the mesh visualization
    n_rad_pts = 450       # number of points along the radius
    n_azi_pts = 180       # number of points along the azimuth
    colat = np.pi / 2     # colatitude angle (pi/2 for equatorial plane)

    # Cosmetic parameters
    wireframe = True
    frames_per_second = 24
    save_rate = 10  # Saves every Nth simulation time step
    resolution = (1920, 1080) # Width, Height
    gw_color = (0.28, 0.46, 1.0) # Blueish
    bh_color = (0.1, 0.1, 0.1)   # Dark grey/black
    zoomout_distance = 350 # Max camera distance after zoomout
    elevation_angle = 34   # Initial camera elevation

    time1 = time.time()

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"""{'*' * 70}
    Initializing grid points..."""
        )
    strain_array, grid, points = initialize_tvtk_grid(n_azi_pts, n_rad_pts)

    time2=time.time()

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"""{'*' * 70}
    Converting psi4 data to strain..."""
        )

    # Convert psi4 to strain and load strain data
    try:
        # Pass the directory where the strain files are expected
        time_array, mode_data = psi4strain.psi4_ffi_to_strain(bh_dir, psi4_output_dir, ell_max, ext_rad)
    except FileNotFoundError as e:
         raise FileNotFoundError(f"Error loading strain data: {e}. Ensure converted files exist in {psi4_output_dir} or check psi4strain function.")
    except Exception as e:
         raise RuntimeError(f"An unexpected error occurred during psi4 data conversion and strain loading: {e}")

    n_times = len(time_array)
    if n_times == 0:
        raise ValueError("Loaded time array is empty. Cannot proceed.")

    n_frames = int(n_times / save_rate)
    print(f"Loaded {mode_data.shape[0]} modes over {n_times} time steps.")

    time3=time.time()

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"""{'*' * 70}
    Calculating black hole trajectories..."""
        )

    # Import black hole data using more efficient loading
    try:
        # Skiprows assumes a fixed header length. Verify this matches the file spec.
        bh_data = np.loadtxt(bh_file_path, skiprows=14, dtype=np.float64)
    except FileNotFoundError:
         raise FileNotFoundError(f"Black hole position file not found: {bh_file_path}")
    except Exception as e:
         raise RuntimeError(f"Error loading black hole data from {bh_file_path}: {e}")

    if bh_data.shape[1] < 5 or bh_data.shape[0] < 2 or bh_data.ndim != 2:
         raise ValueError(f"""Black hole data in {bh_file_path} has unexpected shape or too few rows. Ensure the file meets>    - 15 rows of header lines
    - At least 5 columns of data formatted as follows:
        column 0: retarded time
        column 1: areal mass of black hole 1
        column 2: areal mass of black hole 2
        column 3: x position of black hole 1
        column 4: y position of black hole 1
    - At least 2 rows after the header, the first of which must be a dummy row with the merge time of the black holes in co>

    NOTE: For SXS simulations, run the included h5_to_ascii.py to convert the .h5 data into this format""")

    merge_idx_bh = np.argmax(np.diff(bh_data[:, 1])) + 1 # Find index where BH mass jumps as index of merge point
    merge_time = bh_data[merge_idx_bh, 0]

    # Extract BH time array
    bh_time = bh_data[:, 0]

    # Mass ratio calculation using slice object up to merger
    pre_merge = slice(None, merge_idx_bh)

    # Calculating the average areal mass of each black hole before the merge point
    bh1_avg_mass = np.mean(bh_data[pre_merge, 1])
    bh2_avg_mass = np.mean(bh_data[pre_merge, 14])

    if bh1_avg_mass == 0 or bh2_avg_mass == 0:
        print("Warning: Mean mass is zero pre-merger, using default masses from GW150914.")
        # Keep default bh1_mass, bh2_mass

    else:
        # Sets the second black hole as the more massive one if not already
        if bh1_avg_mass > bh2_avg_mass:
            bh1_avg_mass, bh2_avg_mass = bh2_avg_mass, bh1_avg_mass

        # For the purpose of scaling, the larger black hole's mass will be set to one, and the smaller black hole's mass
        # is adjusted accordingly by computing a ratio of the masses
        bh2_mass = 1
        bh1_mass = bh1_avg_mass / bh2_avg_mass # This will later be used as the mass ratio since bh1_mass / bh2_mass is now
                                               # bh1_mass

    # Extract BH coordinates (Check columns: 3=x, 4=y, assuming z=0 initially)
    bh1_x0, bh1_y0 = bh_data[:, 3], bh_data[:, 4]
    bh1_z0 = np.zeros_like(bh1_x0)  # Assume motion is in xy-plane

    # Interpolate BH positions to the *strain* time array (equal_times)
    # Time array optimization: Use the actual strain time array for interpolation basis
    equal_times = np.linspace(time_array[0], time_array[-1], num=n_times) # Create equally spaced times for output frames
    merge_idx_equal = idx_time(equal_times, merge_time) # Find merge index in the output time array

    # Maintain original interpolation call
    bh1_x, bh1_y, bh1_z = interpolate_coords_by_time(
        bh_time, bh1_x0, bh1_y0, bh1_z0, equal_times
    )

    # Vectorized coordinate calculation using array stacking
    pre_slice = slice(None, merge_idx_equal)
    post_slice = slice(merge_idx_equal, None)

    # Single array operation for all coordinates
    bh2_coords = np.concatenate([
        -np.column_stack((bh1_x[pre_slice], bh1_y[pre_slice], bh1_z[pre_slice])) * bh1_mass, # bh1_mass works as mass ratio
        np.column_stack((bh1_x[post_slice], bh1_y[post_slice], bh1_z[post_slice])) # bh2 initially moves opposite bh1, but
                                                                                   # changes to following bh1 after merge
    ])
    bh2_x, bh2_y, bh2_z = bh2_coords.T  # Transpose and unpack

    time4=time.time()
    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
            f"""{'*' * 70}
    Calculating cosmetic data..."""
        )
    # Find radius of the center hole in the mesh (based on uninterpolated max separation + BH size)
    # Hole radius = factor * (max_sep + scaled radius of larger BH)
    omitted_radius_length = 1.45*(np.sqrt(np.max(bh1_x0**2 + bh1_y0**2)) + bh_scaling_factor * max(bh1_mass, bh2_mass))

    # Find point at which to taper off gravitational waves
    width = 0.5 * omitted_radius_length # Width of the transition region
    dropoff_radius = width + omitted_radius_length # Radius at which to start tapering beyond the hole

    # --- Extract critical strain values and zoom time ---
    try:
         # Pass the directory where converted files are expected
         strain_after_burst, max_strain, zoomout_time = extract_max_strain_and_zoomout_time(psi4_output_dir, ext_rad, ell_min, ell_max)
    except FileNotFoundError as e:
         raise FileNotFoundError(f"Cannot calculate max strain/zoom time: {e}. Ensure strain files exist.")
    except Exception as e:
         raise RuntimeError(f"Error during max strain/zoom time calculation: {e}")

    # Find max z allowable without impeding view of center hole (based on dropoff radius)
    try:
         tangent = np.tan(np.radians(elevation_angle))
         if tangent == np.nan:
             raise ValueError(f"Cannot calculate max allowable z coordinate. Ensure camera elevation angle is not 90 degrees.")
         z_max = dropoff_radius / tangent
    except ZeroDivisionError as e:
         raise ZeroDivisionError(f"Cannot calculate max allowable z coordinate: {e}. Ensure camera elevation angle is not zero.")

    # Calculate factor by which to scale strain data
    if max_strain == 0:
        amplitude_scale_factor = 1.0 # Default scaling if no strain detected
        print("Warning: Max strain is zero, using default amplitude scaling.")
    else:
        # Calculate the amplitude scale factor using symlog of max strain (if enabled)
        amplitude_scale_factor = z_max / (np.sign(max_strain) * np.log1p(np.abs(max_strain)) if use_symlog else max_strain)

    # Report calculated values
    # Use f-strings for print statements
    print(f"Max overall strain (iterative peaks): {max_strain}")
    print(f"Strain value after burst: {strain_after_burst}")
    print(f"Amplitude scale factor: {amplitude_scale_factor}")

    if zoomout_time == float('inf'):
         print("Zoomout time: Not determined (likely too few peaks). Zoomout disabled.")
         zoomout_idx = n_times # Set index beyond array to disable zoom
    else:
         print(f"Calculated zoomout time: {zoomout_time}")
         # Find closest index in the *output* time array (equal_times)
         zoomout_idx = idx_time(equal_times, zoomout_time)

    # theta and radius values for the mesh
    radius_values = np.linspace(0, display_radius, n_rad_pts, dtype=np.float32) # Use float32 for memory
    azimuth_values = np.linspace(0, 2 * np.pi, n_azi_pts, endpoint=False, dtype=np.float32) # Use float32

    # Create meshgrid (ij indexing gives radius changing fastest)
    rv, az = np.meshgrid(radius_values, azimuth_values, indexing="ij")
    # Calculate Cartesian coordinates for the flat mesh
    x_values = rv * np.cos(az)
    y_values = rv * np.sin(az)

    # Dropoff factor (smooth transition to zero amplitude near omitted hole), apply amplitude scale factor
    dropoff_2D_flat = (0.5 + 0.5 * erf((radius_values - dropoff_radius)/width)).ravel() * amplitude_scale_factor

    time5 = time.time()

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
        f"""{'*' * 70}
    Constructing mesh points in 3D..."""
        )

    # Apply spin-weighted spherical harmonics, superimpose modes, and interpolate to mesh points
    strain_azi = swsh_summation_angles(colat, azimuth_values, mode_data, ell_min, ell_max).real
    lerp_times = generate_interpolation_points(equal_times, radius_values, ext_rad)

    strain_to_mesh = compute_strain_to_mesh(
        strain_azi, 
        equal_times.astype(np.float32), # Pass float32 version 
        radius_values, # Already float32
        lerp_times, # Already float32
        time_array.astype(np.float32), # Pass float32 version
        dropoff_2D_flat,
        use_symlog,
        STATUS_MESSAGES
    )
    # Result shape (n_rad_pts, n_azi_pts, n_times)

    time6=time.time()

    if STATUS_MESSAGES:
        print( # Horizontal line of asterisks
        f"""{'*' * 70}
    Initializing animation..."""
        )

    # --- Precompute values for animation loop ---
    bh1_scaled_radius = bh1_mass * bh_scaling_factor
    bh2_scaled_radius = bh2_mass * bh_scaling_factor

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
    elevations = np.maximum(50 - time_indices * 0.016, elevation_angle)
    # Smoothly increase distance during zoomout phase
    distances = np.minimum(np.where(
        time_indices < zoomout_idx,
        80,
        80 + (time_indices - zoomout_idx) * 0.175
    ), zoomout_distance)

    # Precompute merge rescale index
    merge_condition = (valid_indices > merge_idx_equal) & (valid_indices < merge_idx_equal + save_rate)
    merge_rescale_indices = np.where(merge_condition)[0]
    merge_rescale_idx = merge_rescale_indices[0] if merge_rescale_indices.size > 0 else -1

    # Precompute percentage thresholds for progress report (based on simulation time)
    percentage_thresholds = np.round(np.linspace(0, n_times, 101)).astype(int)

    # Precompute frame filenames using f-strings and os.path.join
    frame_filenames = [os.path.join(movie_file_path, f"z_frame_{i:05d}.png") for i in range(n_valid)]

    # Configure engine and rendering upfront
    mlab.options.offscreen = False # Set True for non-interactive rendering to files
    engine = Engine()
    engine.start()

    # Create figure AFTER engine starts if offscreen
    fig = mlab.figure(engine=engine, size=resolution) # Default background color

    # Initialize visualization objects once
    # GW Surface
    create_gw(engine, grid, gw_color, display_radius, wireframe)

    # Black Holes
    bh1 = create_sphere(engine, bh1_scaled_radius, bh_color)
    bh2 = create_sphere(engine, bh2_scaled_radius, bh_color)

    # Initialize timing and progress tracking
    start_time = time.time()

    # Report setup times - Use f-strings
    print(f"Timing Report (seconds):")
    print(f"  Parameter & Movie Setup: {time1 - time0:.3f}")
    print(f"  Grid Init: {time2 - time1:.3f}")
    print(f"  Psi 4 Conversion & Strain Load: {time3 - time2:.3f}")
    print(f"  BH Trajectories: {time4 - time3:.3f}")
    print(f"  Cosmetic Calcs: {time5 - time4:.3f}")
    print(f"  Mesh Construction: {time6 - time5:.3f}")
    print(f"  Animation Setup: {start_time - time6:.3f}")
    print(f"  Total Setup Time: {start_time - time0:.3f}")

    # --- Animation Loop ---
    # Use @mlab.animate decorator for potential interactive use,
    # but run it directly for offscreen rendering.
    #@mlab.animate(delay=10, ui=True)
    def anim():
        """Generator function to drive the animation frame by frame."""
        current_percent = 0
        for idx, time_idx in enumerate(valid_indices):
            # --- Status Update & ETA ---
            if idx == 10: # Estimate after 10 frames
                end_time = time.time()
                eta = (end_time - start_time) * n_frames / 10
                print(
                    f"""Creating {n_frames} frames and saving them to:
{movie_file_path}\nEstimated time: {dhms_time(eta)}"""
                )

                # Update progress percent
            if STATUS_MESSAGES and time_idx !=0 and current_percent < len(percentage_thresholds) and time_idx > percentage_thresholds[current_percent]:
                eta = ((time.time() - start_time) / time_idx) * (n_times - time_idx)
                print(f"{int(time_idx  * 100 / n_times)}% done, ", f"{dhms_time(eta)} remaining", end="\r", flush=True)
                current_percent +=1

            # --- Update Scene Objects ---
            # Rescale bh2 if black holes have merged to represent combined object (at the specific frame index)
            if idx == merge_rescale_idx:
                rescale_object(bh2, (bh1_scaled_radius + bh2_scaled_radius) / bh2_scaled_radius)

            # Update black hole positions using interpolated data for the current *simulation time index*
            change_object_position(bh1, (bh1_x[time_idx], bh1_y[time_idx], bh1_z[time_idx]))
            change_object_position(bh2, (bh2_x[time_idx], bh2_y[time_idx], bh2_z[time_idx]))

            # Update Mesh Z-coordinates (Strain Visualization)
            # Get the strain slice for the current *simulation time index*
            strain_slice = strain_to_mesh[..., time_idx].ravel() # Flatten (n_rad, n_azi) -> (n_points)

            # Apply masking: set Z to NaN for points inside the hole
            np_points[:, 2] = np.where(valid_mask, strain_slice, np.nan)

            # Notify VTK that the points data has changed
            vtk_array.modified()

            # Update grid source
            strain_array.from_array(strain_slice[valid_mask])
            grid._set_points(points)
            grid.modified()

            # --- Update Camera ---
            mlab.view(
                elevation=elevations[time_idx], # Use precomputed elevation for this frame
                distance=distances[time_idx], # Use precomputed distance for this frame
                focalpoint=(0, 0, 0) # Keep focused on the origin
            )

            # --- Save Frame --
            try:
                mlab.savefig(frame_filenames[idx])
            except Exception as e:
                 print(f"\nError saving frame {frame_filenames[idx]}: {e}")

            # yield # Yield control for the @mlab.animate decorator (if used interactively)

        # --- End of Loop ---
        mlab.close(all=True) # Close the Mayavi figure/engine
        total_time = time.time() - start_time
        print("\nDone", flush=True) # Newline after progress bar
        # Use f-strings for final messages
        print(
            f"\nSaved {n_frames} frames to {movie_file_path} ",
            f"in {dhms_time(total_time)}.",
        )
        print("Creating movie...")
        try:
            convert_to_movie(movie_file_path, movie_dir_name, frames_per_second)
            print(f"Movie saved to {movie_file_path}/{movie_dir_name}.mp4")
        except Exception as e:
             print(f"\nError creating movie from frames: {e}")
             print("You may need to run ffmpeg manually:")
             print(f"  cd {movie_file_path}")
             print(f"  ffmpeg -framerate {frames_per_second} -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p {movie_dir_name}.mp4")

    # Run the animation script
    _ = anim()
    mlab.show()

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
         import traceback
         traceback.print_exc()
         sys.exit(1)
