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
ELL_MAX = 8
ELL_MIN = 2
S_MODE = -2
EXT_RAD = 100 # changeable with sys arguments
USE_SYS_ARGS = True # Change to turn on/off default parameters
STATUS_MESSAGES = True # Change to turn on/off status reports during rendering

def swsh_summation_angles(colat: float, azi: NDArray[np.float64], mode_data):
    """
    Adds up all the strain modes after factoring in corresponding spin-weighted spherical harmonic
    to specified angle in the mesh. Stored as an array corresponding to [angle, time] time_idxs.

    :param colat: colatitude angle for the SWSH factor
    :param azi: azimuthal angle for the SWSH factor
    :param mode_data: numpy array containing strain data for all the modes
    :return: a complex valued numpy( m1 d1 + m2 ( d1 + d2 ) + m3 ( d1 + d2 + d3 ) )/( m1 + m2 + m3 ) array of the superimposed wave

    >>> mode_data = np.zeros((77, 3), dtype=complex)
    >>> mode_idx = 0
    >>> for l in range(2, 9):
    ...     for m in range(-l, l+1):
    ...         mode_data[mode_idx] = np.array([1+1j, 2+3j, 4+5j])
    ...         mode_idx += 1
    >>> np.round(swsh_summation_angles(np.pi/2, np.array([0]), mode_data), 5)
    array([[ 4.69306 +4.69306j,  9.38612+14.07918j, 18.77224+23.4653j ]])
    """
    quat_arr = quaternionic.array.from_spherical_coordinates(colat, azi)
    winger = spherical.Wigner(ELL_MAX, ELL_MIN)
    # Create an swsh array shaped like (n_modes, n_quaternions)
    swsh_arr = winger.sYlm(S_MODE, quat_arr).T
    # mode_data has shape (n_modes, n_times), swsh_arr has shape (n_mode, n_pts).
    # Pairwise multiply and sum over modes: the result has shape (n_pts, n_times).
    pairwise_product = mode_data[:, np.newaxis, :] * swsh_arr[:, :, np.newaxis]
    return np.sum(pairwise_product, axis=0)

def generate_interpolation_points(
    time_array: NDArray[np.float64],
    radius_values: NDArray[np.float64],
    r_ext: float,
) -> NDArray[np.float64]:
    """
    Fills out a 2D array of adjusted time values for the wave strain to be
    linearly interpolated to. First index of the result represents the simulation
    time time_idx (aka which mesh), and the second index represents radial distance to
    interpolate to.

    :param time_array: numpy array of of strain time time_idxs.
    :param radius_values: numpy array of the radial points on the mesh.
    :param r_ext: extraction radius of the original data.
    :return: a 2D numpy array (n_radius, n_times) of time values.
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
    Interpolates the 3D coordinates to the given time state.

    :param old_times: 1D array of original time values.
    :param e1: 1D array of the first coordinate values.
    :param e2: 1D array of the second coordinate values.
    :param e3: 1D array of the third coordinate values.
    :param new_times: 1D array of new time values to interpolate to.
    :return: A tuple of three 1D arrays representing the interpolated coordinates.
    """

    # Create a single interpolator for all coordinates
    interpolator = interp1d(old_times, np.vstack((e1, e2, e3)), fill_value="extrapolate")

    # Interpolate all coordinates at once
    new_e1, new_e2, new_e3 = interpolator(new_times)

    return new_e1, new_e2, new_e3

def initialize_tvtk_grid(num_azi: int, num_radius: int) -> Tuple:
    """
    Sets initial parameters for the mesh generation module and returns
    a circular, polar mesh with manipulation objects to write and save data.

    :param num_azi: number of azimuthal points on the mesh
    :param num_radius: number of radial points on the mesh
    :returns: tvtk.FloatArray,
              tvtk.UnstructuredGrid,
              tvtk.Points
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
    engine: Any,
    grid: Any,
    color: Tuple[float, float, float],
    display_radius: int,
    wireframe: bool = False,
) -> None:
    """
    Creates and displays a gravitational wave strain from a given grid.

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
    engine: Engine, radius: float = 1, color: tuple[float, float, float] = (1, 0, 0)
) -> Surface:
    """
    Creates and displays a spherical surface with the given parameters.

    :param engine: Mayavi engine for visualization.
    :param radius: Radius of the sphere.
    :param color: Color of the sphere as an RGB tuple (0, 0, 0) to (1, 1, 1).
    :return: The Surface object representing the sphere.
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
    Changes the Cartesian position of a Mayavi surface to the given coordinates.

    :param obj: Mayavi Surface object to reposition
    :param position: New (x, y, z) position as a tuple of floats
    """

    obj.actor.actor.position = position  # Direct tuple assignment (no numpy conversion needed)

def rescale_object(obj: Surface, size: float) -> None:
    """
    Rescales a Mayavi surface equally in all directions by a given factor.

    :param obj: Mayavi Surface object to reposition
    :param size: Scale factor to multiply dimensions by
    """

    obj.actor.actor.scale = (size, size, size)  # Direct tuple assignment (no numpy conversion needed)

def dhms_time(seconds: float) -> str:
    """
    Converts a given number of seconds into a string indicating the remaining time.

    :param seconds: Number of seconds
    :return: A string indicating the remaining time (days, hours, minutes)
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
    Converts a series of .png files into a movie using imageio.

    :param input_path: path to the directory containing the .png files
    :param movie_name: name of the movie file
    :param fps: frames per second (24 by default)
    """

    # Efficient directory scanning with immediate sorting and filtering
    with os.scandir(input_path) as it:
        image_files = sorted([entry.name for entry in it])

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
            writer.append_data(imageio.imread(file_path))

            if status_messages:
                # Integer-based progress tracking for minimal computation
                current_progress = (i * 1000) // total_frames
                if current_progress != prev_progress:
                    print(f"\rProgress: {current_progress/10:.1f}% completed", end="", flush=True)
                    prev_progress = current_progress

        if status_messages:
            print("\rProgress: 100.0% completed", flush=True)

def ask_user(message: str) -> bool:
    """
    Allows user input in the command terminal to a Yes/No response.
    Returns boolean based on input.

    :param message: message to ask the user (indicate Y/N input).
    """

    response = input(message)
    if response.lower() != "y":
        return False
    else:
        return True

def symlog(x: float) -> float:
    """
    Computes the signed logarithm of float input x

    :param x: Number for which to compute the signed logarithm
    :return: The signed logarithm of the input
    """

    return np.sign(x) * np.log1p(np.abs(x))  # log1p(x) = log(1 + x), avoids log(0) issues

def max_strain_values(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Identifies peak strain values where increasing/decreasing trends change.

    :param data: Numpy 2D array of strain values with column 0 as time and column 1 as strain
    :return: numpy 2D array of local maxima in data with column 0 as the value's index in the original data,
    column 1 as the time, and column 2 as the max strain values 
    """
    strain_vals = data[:, 1]
    times = data[:, 0]
    peaks = []

    prev_val = strain_vals[0]
    increasing = True

    for t in range(1, len(strain_vals)):
        current_val = strain_vals[t]

        if increasing:
            if current_val <= prev_val:  # Peak detected
                peaks.append((t-1, times[t-1], abs(prev_val)))
                increasing = False
        else:
            if current_val > prev_val:
                increasing = True

        prev_val = current_val

    if increasing:
        peaks.append((len(strain_vals)-1, times[-1], abs(strain_vals[-1])))
    return np.array(peaks)

def get_local_maxima(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Identifies local maxima in already maximized strain data.

    :param data: Numpy 2D array of local maxima strain values with column 0 as the value's index in the original data, 
    column 1 as the time, and column 2 as the strain values
    :return: numpy 2D array of local maxima in local maxima data with column 0 as the value's index in the original data,
    column 1 as the time, and column 2 as the max strain values
    """

    time_col = data[:, 0]
    y_col = data[:, 1]
    values = data[:, 2]
    local_max = []
    decreasing = False
    prev_value = values[0]

    for t in range(1, len(values)):
        current_value = values[t]
        if decreasing:
            if current_value > prev_value:
                decreasing = False
        else:
            if current_value < prev_value:
                local_max.append([time_col[t-1], y_col[t-1], prev_value])
                decreasing = True
        prev_value = current_value

    if not decreasing:
        local_max.append([time_col[-1], y_col[-1], values[-1]])
    return np.array(local_max, dtype=np.float64) if local_max else np.empty((0, 3))

def local_max_iterative(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Iteratively finds local max of data until doing so further would reduce the size of the dataset by a certain factor

    :param data: Numpy 2D array of local maxima strain values with column 0 as the value's index in the original data,
    column 1 as the time, and column 2 as the strain values
    :return: Numpy 2D array of strain values found by iterating searches for local maxima, with column 0 as the value's 
    index in the original data, column 1 as the time, and column 2 as the strain values. Maximizing the data beyond this
    point causes significant reductions in the information in the data.
    """

    current = data
    while True:
        max_data = get_local_maxima(current)
        max_max_data = get_local_maxima(max_data)
        if 2 * max_max_data.shape[0] > max_data.shape[0]:
            current = max_data
        else:
            return max_data

def get_max_max(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Iteratively finds local max of data until doing so further would reduce the size of the dataset to below a certain
    size.

    :param data: Numpy 2D array of local maxima strain values with column 0 as the value's index in the original data,
    column 1 as the time, and column 2 as the strain values
    :return: Numpy 2D array of strain values found by iterating searches for local maxima, with column 0 as the value's
    index in the original data, column 1 as the time, and column 2 as the strain values. Maximizing the data beyond this
    point causes a loss of information rendering the data unreliable.
    """

    current = data
    while True:
        current = get_local_maxima(current)
        if current.shape[0] <= 4:
            break
    return current

def get_strain_after_burst(data: NDArray[np.float64], end_of_burst_value: float) -> float:
    """
    Gets the strain value in the data immediately after the radiation burst has ended.

    :param data: Numpy 2D array of local maxima strain values with column 0 as the value's index in the original data,
    column 1 as the time, and column 2 as the strain values
    :param end_of_burst_value: The value of the strain at the end of the radiation burst
    :return: The strain value immediately after the burst has ended
    """

    if data.shape[0] < 3:
        return 0
    end_of_burst_index = np.where(data[:, 2] == end_of_burst_value)[0][0]
    data_subset = data[end_of_burst_index:, 2]
    if len(data_subset) < 2:
        return data_subset[-1]
    diff = data_subset[1:] > data_subset[:-1]
    indices = np.where(diff)[0]
    return data_subset[indices[0]] if indices.size > 0 else data_subset[-1]

def calculate_zoomout_time(data: NDArray[np.float64]) -> float:
    """
    Calculates the time the visualization should start zooming out

    :param data: Numpy 2D array of local maxima strain values with column 0 as the value's index in the original data,
    column 1 as the time, and column 2 as the strain values
    :return: The time at which to begin zooming out in the visualization
    """

    values = data[:, 2]
    if len(values) < 1:
        return float('inf')
    max1_val = values.max()
    max1_indices = np.where(values == max1_val)[0]
    max1_idx = max1_indices[-1]
    max1_time = data[max1_idx, 1]

    if len(values) < 2:
        return float('inf')

    mask = np.arange(len(values)) != max1_idx
    remaining_values = values[mask]
    if remaining_values.size == 0:
        return float('inf')
    max2_val = remaining_values.max()
    max2_candidate_indices = np.where(values == max2_val)[0]
    max2_candidate_indices = [idx for idx in max2_candidate_indices if idx != max1_idx]
    if not max2_candidate_indices:
        return float('inf')
    max2_idx = max2_candidate_indices[-1]
    max2_time = data[max2_idx, 1]

    return max2_time + (max1_time + max2_time) / 4

def extract_max_strain_and_zoomout_time(dir: str, r_ext: int) -> Tuple:
    """
    Calculates the strain immediately after the radiation burst, the max strain in the data in the input directory,
    and the time the visualization should start zooming out

    :param dir: The directory housing the strain data for each l
    :param r_ext: The radius of extraction of the data
    :return: The max strain immediately after the radiation burst, across all l and m
             The max strain in all the data, across all l and m
             The minimum time at which the visualization should start zooming out, across all l and m
    """

    strain_after_burst = float('-inf')
    min_zoomout_time = float('inf')
    max_strain = float('-inf')

    for l in range(ELL_MIN, ELL_MAX + 1):
        file_path = os.path.join(dir, f"Rpsi4_r{0 if r_ext < 1000 else ''}{r_ext}_l{l}_conv_to_strain.txt")
        max_col = 4 * l + 2
        try:
            data_all = np.loadtxt(file_path, skiprows=4*l+3, usecols=range(0, max_col+1))
        except (FileNotFoundError, IndexError):
            continue

        valid_columns = []
        for m in range(-l, l+1):
            col1 = 2*l + 2*m + 1
            col2 = col1 + 1
            if col2 < data_all.shape[1]:
                valid_columns.append((m, col1, col2))

        for m, col1, col2 in valid_columns:
            real = data_all[:, col1]
            imag = data_all[:, col2]
            magnitude = np.hypot(real, imag)
            time_col = data_all[:, 0]
            data = np.column_stack((time_col, magnitude))

            max_strain_data = max_strain_values(data)
            if max_strain_data.size == 0:
                continue

            data_max_local_max = local_max_iterative(max_strain_data)
            data_max_max = get_max_max(data_max_local_max)
            if data_max_max.size == 0:
                continue

            zt = calculate_zoomout_time(data_max_max)
            sab = get_strain_after_burst(max_strain_data, data_max_local_max[0, 2])
            mv = np.max(data_max_max[:, 2])

            strain_after_burst = max(strain_after_burst, sab)
            min_zoomout_time = min(min_zoomout_time, zt)
            max_strain = max(max_strain, mv)

    return strain_after_burst, max_strain, min_zoomout_time

def compute_strain_to_mesh(
    strain_azi: NDArray[np.float32],
    equal_times: NDArray[np.float32],
    radius_values: NDArray[np.float32],
    lerp_times: NDArray[np.float32],  # Precomputed 2D array of interpolation points
    time_array: NDArray[np.float32],
    status_messages=True
) -> NDArray[np.float32]:
    """
    Computing mesh points in 3D using precomputed strain values with reduced memory footprint.

    :param strain_azi: Precomputed strain values as a 2D array of shape (n_azi_pts, n_time_points).
    :param equal_times: Array of times at which to interpolate the strain.
    :param radius_values: Array of radius values for which to compute the strain on the mesh.
    :param lerp_times: Precomputed 2D array of interpolation points of shape (n_rad_pts, n_times).
    :param time_array: Original time points corresponding to the strain_azi data.
    :param status_messages: Optional turning on progress bars
    :return: 3D array of shape (n_rad_pts, n_azi_pts, n_times) containing the interpolated strain values.
    """

    n_rad_pts = len(radius_values)
    n_azi_pts = strain_azi.shape[0]
    n_times = len(equal_times)

    # Compute strain_azi in chunks to reduce peak memory usage
    chunk_size = min(int(psutil.virtual_memory().available / 7000000), n_azi_pts)  # Adjust based on available memory
    strain_to_mesh = np.zeros((n_rad_pts, n_azi_pts, n_times), dtype=np.float32)

    for start_idx in range(0, n_azi_pts, chunk_size):
        end_idx = min(start_idx + chunk_size, n_azi_pts)
        strain_azi_chunk = strain_azi[start_idx:end_idx, :]

        # Interpolate each azimuth in the current chunk
        for i, azi_idx in enumerate(range(start_idx, end_idx)):
            strain_to_mesh[:, azi_idx, :] = np.interp(
                lerp_times,
                time_array,
                strain_azi_chunk[i, :],
                left=np.nan,
                right=np.nan
            )
            # Update status
            if status_messages:
                progress = (start_idx + i) / (n_azi_pts - 1) * 100
                print(f"\rProgress: {int(progress)}% completed", end="", flush=True)

    # Create a new line after status messages
    if status_messages:
        print()

    return strain_to_mesh

def idx_time(array: NDArray[np.float64], time: float) -> int:
    """
    Calculates the index at which the value of an array is closest to the value of time

    :param array: Numpy array of time values
    :param time: Target time to search for in an array
    :return: The index at which the time is closest to the target time
    """

    diff = np.abs(array - time)
    zero_match = np.where(diff == 0)[0]
    return zero_match[0] if zero_match.size > 0 else np.argmin(diff)

def main() -> None:
    """
    Main function that reads the strain data,
    calculates and factors in spin-weighted spherical harmonics,
    linearly interpolates the strain to fit the mesh points,
    and creates .tvtk mesh file for each time state of the simulation.
    The meshes represent the full superimposed waveform at the polar angle pi/2,
    aka the same plane as the binary black hole merger. At each state, the black holes
    are moved to their respective positions and the mesh is saved as a .png file.
    """

    # Convert psi4 data to strain using imported script
    # psi4_to_strain.main()

    # Check initial parameters
    time0 = time.time()
    if USE_SYS_ARGS:
        argc = len(sys.argv)
        if argc not in (3, 4):
            raise RuntimeError(
                """Please include path to merger data as well as the psi4 extraction radius of that data.
                Usage (spaces between arguments): python3
                                                  scripts/animation_main.py
                                                  <path to data folder>
                                                  <extraction radius (r/M) (4 digits, e.g. 0100)>
                                                  <optional: use symlog scale?>"""
            )
        else:
            # change directories and extraction radius based on inputs
            bh_dir = str(sys.argv[1])
            psi4_output_dir = os.path.join(bh_dir, "converted_strain")
            ext_rad = float(sys.argv[2])
            movie_dir = os.path.join(bh_dir, "movies")  # Optimized path construction
            use_symlog = sys.argv[3] if argc == 4 else False

    else:
        bh_dir = BH_DIR
        movie_dir = MOVIE_DIR
        ext_rad = EXT_RAD
        psi4_output_dir = os.path.join(bh_dir, "converted_strain")
        # mass ratio for default system GW150914
        bh1_mass = 1
        bh2_mass = 1.24
        use_symlog = False

    # File names
    bh_file_name = "puncture_posns_vels_regridxyzU.txt"
    bh_file_path = os.path.join(bh_dir, bh_file_name)
    bh_scaling_factor = 1

    movie_number = 1
    while True:
        movie_dir_name = f"real_movie{movie_number}"
        movie_file_path = os.path.join(movie_dir, movie_dir_name)

        if os.path.exists(movie_file_path):
            if not ask_user(
                f"{movie_file_path} already exists. Would you like to overwrite it? Y/N: "
            ):
                movie_number += 1
                continue
            # Clear existing files if user confirms overwrite
            for file in os.listdir(movie_file_path):
                os.remove(os.path.join(movie_file_path, file))
            break

        os.makedirs(movie_file_path, mode=0o755, exist_ok=True)
        break

    time1 = time.time()
    # Mathematical parameters
    display_radius = 300  # radius for the mesh
    n_rad_pts = 450
    n_azi_pts = 180
    colat = np.pi / 2  # colatitude angle representative of the plane of merger

    # Cosmetic parameters
    wireframe = True
    frames_per_second = 24
    save_rate = 10  # Saves every Nth frame
    resolution = (1920, 1080)
    gw_color = (0.28, 0.46, 1.0)
    bh_color = (0.1, 0.1, 0.1)
    zoomout_distance = 350
    elevation_angle = 34
    time2 = time.time()

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Initializing grid points..."""
        )
    strain_array, grid, points = initialize_tvtk_grid(n_azi_pts, n_rad_pts)
    time3=time.time()

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Converting psi4 data to strain..."""
        )

    # Import strain data
    time_array, mode_data = psi4strain.psi4_ffi_to_strain(bh_dir, psi4_output_dir, ELL_MAX, ext_rad)

    n_times = len(time_array)
    n_frames = int(n_times / save_rate)

    time4=time.time()

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Calculating black hole trajectories..."""
        )

        # Import black hole data using more efficient loading
    bh_data = np.loadtxt(bh_file_path, skiprows=15, dtype=np.float64)
    merge_time = bh_data[0, 0]
    bh_data = bh_data[1:]  # More efficient than np.delete for large datasets

    # Mass ratio calculation using slice object
    merge_idx = idx_time(bh_data[:, 0], merge_time)
    pre_merge = slice(None, merge_idx)
    bh2_mass = np.mean(bh_data[pre_merge, 14]) / np.mean(bh_data[pre_merge, 1])

    # Efficient mass comparison and swap
    bh1_mass, bh2_mass = (1, bh2_mass) if 1 < bh2_mass else (bh2_mass, 1)

    # Vectorized coordinate initialization
    bh_time = bh_data[:, 0]
    bh1_x0, bh1_y0 = bh_data[:, 3], bh_data[:, 4]
    bh1_z0 = np.zeros_like(bh1_x0)  # More memory-efficient

    # Time array optimization
    equal_times = np.linspace(time_array[0], time_array[-1], num=n_times)
    merge_idx2 = idx_time(equal_times, merge_time)

    # Maintain original interpolation call
    bh1_x, bh1_y, bh1_z = interpolate_coords_by_time(
        bh_time, bh1_x0, bh1_y0, bh1_z0, equal_times
    )

    # Vectorized coordinate calculation using array stacking
    bh_mass_ratio = bh1_mass / bh2_mass
    pre_slice = slice(None, merge_idx2)
    post_slice = slice(merge_idx2, None)

    # Single array operation for all coordinates
    bh2_coords = np.concatenate([
        -np.column_stack((bh1_x[pre_slice], bh1_y[pre_slice], bh1_z[pre_slice])) * bh_mass_ratio,
        np.column_stack((bh1_x[post_slice], bh1_y[post_slice], bh1_z[post_slice]))
    ])
    bh2_x, bh2_y, bh2_z = bh2_coords.T  # Transpose and unpack

    time5=time.time()
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Calculating cosmetic data..."""
        )
    # Find radius of the center hole in the mesh
    omitted_radius_length = 1.45*(np.sqrt(np.max(bh1_x0*bh1_x0 + bh1_y0*bh1_y0)) + bh_scaling_factor * max(bh1_mass, bh2_mass))

    # Find point at which to taper off gravitational waves
    width = 0.5 * omitted_radius_length
    dropoff_radius = width + omitted_radius_length

    # Find the strain immediately after the burst, the max strain of the data, and the time the simulation should start zooming out
    strain_after_burst, max_strain, zoomout_time = extract_max_strain_and_zoomout_time(os.path.join(bh_dir, 'converted_strain'), ext_rad)

    # Find max z allowable without impeding view of center hole
    z_max = dropoff_radius / np.tan(np.radians(elevation_angle))

    amplitude_scale_factor = z_max / (symlog(max_strain) if use_symlog else max_strain)

    print(f"Max strain: {max_strain}")
    print(f"Amplitude scale factor: {amplitude_scale_factor}")
    print(f"Zoomout time: {zoomout_time}")

    # Find closest index using vectorized operations
    zoomout_idx = idx_time(equal_times, zoomout_time)

    # radius for the mesh
    display_radius = 300
    n_rad_pts = 450
    n_azi_pts = 180

    # theta and radius values for the mesh
    radius_values = np.linspace(0, display_radius, n_rad_pts)
    azimuth_values = np.linspace(0, 2 * np.pi, n_azi_pts, endpoint=False)

    rv, az = np.meshgrid(radius_values, azimuth_values, indexing="ij")
    x_values = rv * np.cos(az)
    y_values = rv * np.sin(az)

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Constructing mesh points in 3D..."""
        )

    # Apply spin-weighted spherical harmonics, superimpose modes, and interpolate to mesh points
    strain_azi = swsh_summation_angles(colat, azimuth_values, mode_data).real
    lerp_times = generate_interpolation_points(equal_times, radius_values, ext_rad)

    strain_to_mesh = compute_strain_to_mesh(
        strain_azi, 
        equal_times, 
        radius_values, 
        lerp_times,
        time_array
    )

    time6=time.time()

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Initializing animation..."""
        )

    # Precompute all constant values once
    bh1_scaled = bh1_mass * bh_scaling_factor
    bh2_scaled = bh2_mass * bh_scaling_factor
    valid_indices = np.arange(0, n_times, save_rate)
    n_valid = len(valid_indices)

    # Precompute geometric data with vectorization
    valid_mask = (rv > omitted_radius_length).ravel()
    dropoff_2D_flat = (0.5 + 0.5 * erf((rv - dropoff_radius)/width)).ravel() * amplitude_scale_factor
    x_flat, y_flat = x_values.ravel(), y_values.ravel()

    # Define strain processing based on symlog setting (avoids per-frame condition check)
    if use_symlog:
        def process_strain(strain_slice):
            return symlog(strain_slice) * dropoff_2D_flat
    else:
        def process_strain(strain_slice):
            return strain_slice * dropoff_2D_flat

    # Initialize VTK data structures once
    points = tvtk.Points()
    vtk_array = tvtk.FloatArray()
    vtk_array.number_of_components = 3
    vtk_array.number_of_tuples = len(x_flat)
    points.data = vtk_array
    np_points = vtk_array.to_array().reshape(-1, 3)
    np_points[:, 0] = x_flat
    np_points[:, 1] = y_flat

    # Precompute camera parameters
    time_indices = np.arange(n_times)
    elevations = np.maximum(50 - time_indices * 0.016, elevation_angle)
    distances = np.minimum(np.where(
        time_indices < zoomout_idx,
        80,
        80 + (time_indices - zoomout_idx) * 0.175
    ), zoomout_distance)

    # Precompute merge rescale index
    merge_condition = (valid_indices > merge_idx2) & (valid_indices < merge_idx2 + save_rate)
    merge_rescale_indices = np.where(merge_condition)[0]
    merge_rescale_idx = merge_rescale_indices[0] if merge_rescale_indices.size > 0 else -1

    # Precompute percentage thresholds
    percentage_thresholds = np.round(np.linspace(0, n_times, 100)).astype(int)

    # Precompute frame filenames
    frame_filenames = [os.path.join(movie_file_path, f"z_frame_{i:05d}.png") for i in range(n_valid)]

    # Configure engine and rendering upfront
    mlab.options.offscreen = False  # Keep False for interactive, set True for rendering
    engine = Engine()
    engine.start()
    fig = mlab.figure(engine=engine, size=resolution)

    # Initialize visualization objects once
    create_gw(engine, grid, gw_color, display_radius, wireframe)
    bh1 = create_sphere(engine, bh1_scaled, bh_color)
    bh2 = create_sphere(engine, bh2_scaled, bh_color)

    # Initialize timing and progress tracking
    start_time = time.time()
    time6=time.time()

    print(f"0:{time1-time0}\n1:{time2-time1}\n2:{time3-time2}\n3:{time4-time3}\n4:{time5-time4}\n5:{time6-time5}\na:{time6-time0}")
    @mlab.animate(delay=10, ui=False)
    def anim():
        current_percent = 0
        for idx, time_idx in enumerate(valid_indices):
             # Print status messages
            if time_idx == 10 * save_rate:
                end_time = time.time()
                eta = (end_time - start_time) * n_frames / 10
                print(
                    f"""Creating {n_frames} frames and saving them to:
{movie_file_path}\nEstimated time: {dhms_time(eta)}"""
                )
            if STATUS_MESSAGES and time_idx !=0 and current_percent < len(percentage_thresholds) and time_idx > percentage_thresholds[current_percent]:
                eta = ((time.time() - start_time) / time_idx) * (n_times - time_idx)
                print(
                    f"{int(time_idx  * 100 / n_times)}% done, ",
                    f"{dhms_time(eta)} remaining",
                    end="\r",
                )
                current_percent +=1

            # Resize bh2 if black holes have merged
            if idx == merge_rescale_idx:
                rescale_object(bh2, (bh1_scaled + bh2_scaled) / bh2_scaled)

            # Update black hole positions
            change_object_position(bh1, (bh1_x[time_idx], bh1_y[time_idx], bh1_z[time_idx]))
            change_object_position(bh2, (bh2_x[time_idx], bh2_y[time_idx], bh2_z[time_idx]))

            # Memory-efficient strain processing
            strain_slice = strain_to_mesh[..., time_idx].ravel()
            processed_strain = process_strain(strain_slice)
            np_points[:, 2] = np.where(valid_mask, processed_strain, np.nan)
            vtk_array.modified()

            # Update grid
            strain_array.from_array(strain_slice[valid_mask])
            grid._set_points(points)
            grid.modified()

            # Update camera
            mlab.view(
                elevation=elevations[time_idx],
                distance=distances[time_idx],
                focalpoint=(0, 0, 0)
            )

            # Save frame
            mlab.savefig(frame_filenames[idx])

        mlab.close()
        total_time = time.time() - start_time
        print("Done", end="\r")
        print(
            f"\nSaved {n_frames} frames to {movie_file_path} ",
            f"in {dhms_time(total_time)}.",
        )
        print("Creating movie...")
        convert_to_movie(movie_file_path, movie_dir_name, frames_per_second)
        print(f"Movie saved to {movie_file_path}/{movie_dir_name}.mp4")
        sys.exit(0)

    _ = anim()
    mlab.show()

# This should automatically create the movie file...
# if it doesn't work, run the following in the movie directory:
# $ffmpeg -framerate 24 -i frame_%05d.png <movie_name>.mp4

if __name__ == "__main__":
    # run doctests first
    import doctest

    results = doctest.testmod()
    p4s_results = doctest.testmod(psi4strain)

    if p4s_results.failed > 0:
        print(
            f"""Doctest in {psi4strain} failed:
{p4s_results.failed} of {p4s_results.attempted} test(s) passed"""
        )
        sys.exit(1)
    else:
        print(
            f"""Doctest in {psi4strain} passed:
All {p4s_results.attempted} test(s) passed"""
        )

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
    # run main() after tests
    main()
