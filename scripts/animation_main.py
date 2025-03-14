"""
A script that reads in gravitational wave psi 4 data and black hole positional data,
applies spin-weighted spherical harmonics to the data, and creates a Mayavi animation
of the black holes and their gravitational waves. At each state, the black holes are 
moved to their respective positions and the render is saved as a .png file.
"""

import os
import sys
import csv
import time
from math import erf
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.special import erf  # Vectorized error function for arrays
import quaternionic
import spherical
import cv2 # pip install opencv-python
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
    :return: a complex valued numpy array of the superimposed wave

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
    Generates a 2D array of adjusted time values for wave strain interpolation.

    :param time_array: 1D NumPy array of strain time indices.
    :param radius_values: 1D NumPy array of radial points on the mesh.
    :param r_ext: Extraction radius of the original data.
    :return: A 2D NumPy array (n_radius, n_times) of adjusted time values.
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

    :param num_azi: Number of azimuthal points on the mesh.
    :param num_radius: Number of radial points on the mesh.
    :returns: A tuple containing:
              - tvtk.FloatArray: Array to store strain data.
              - tvtk.UnstructuredGrid: The unstructured grid representing the mesh.
              - tvtk.Points: Points object for the mesh.
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

    # Create cells
    cell_array = tvtk.CellArray()
    quad = tvtk.Quad()  # Reuse the same quad object

    for j in range(num_radius - 1):
        next_j = (j + 1) * num_azi
        this_j = j * num_azi
        for i in range(num_azi):
            next_i = (i + 1) % num_azi  # Precompute (i + 1) % num_azi
            point_ids = [
                i + this_j,
                next_i + this_j,
                next_i + next_j,
                i + next_j,
            ]
            for idx, pid in enumerate(point_ids):
                quad.point_ids.set_id(idx, pid)
            cell_array.insert_next_cell(quad)

    # Set grid properties
    grid.set_cells(quad.cell_type, cell_array)

    return strain_array, grid, points

def create_gw(
    engine: Engine,
    grid: Any,
    color: Tuple[float, float, float],
    display_radius: int,
    wireframe: bool = False,
) -> None:
    """
    Creates and displays a gravitational wave strain from a given grid.
    :param engine: Mayavi engine
    :param grid: tvtk.UnstructuredGrid
    :param color: color of the strain in a tuple ranging from (0, 0, 0) to (1, 1, 1)
    :param wireframe: whether to display the strain as a wireframe or a surface
    """
    scene = engine.scenes[0]
    gw = VTKDataSource(data=grid)
    engine.add_source(gw, scene)
    s = Surface()
    engine.add_filter(s, gw)
    s.actor.mapper.scalar_visibility = False
    s.actor.property.color = color

    def gen_contour(coord: NDArray, normal: NDArray):
        contour = ScalarCutPlane()
        engine.add_filter(contour, gw)
        contour.implicit_plane.widget.enabled = False
        contour.implicit_plane.plane.origin = coord
        contour.implicit_plane.plane.normal = normal
        contour.actor.property.line_width = 5
        contour.actor.property.opacity = 0.5

    if wireframe:
        wire_intervals = np.linspace(-display_radius, display_radius, 14)

        for c in wire_intervals:
            gen_contour(np.array([c, 0, 0]), np.array([1, 0, 0]))
            gen_contour(np.array([0, c, 0]), np.array([0, 1, 0]))
        '''
        s.actor.property.representation = "wireframe"
        s.actor.property.color = (0, 0, 0)
        s.actor.property.line_width = 0.005
        s.actor.property.opacity = 0.5
        '''
def create_gw(
    engine: Engine,
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
    # Use the current scene
    scene = engine.current_scene if hasattr(engine, "current_scene") else engine.scenes[0]

    # Create and add VTK data source
    gw = VTKDataSource(data=grid)
    engine.add_source(gw, scene)

    # Apply surface visualization and configure properties
    s = Surface()
    engine.add_filter(s, gw)
    s.actor.mapper.scalar_visibility = False
    s.actor.property.color = color

    def gen_contour(coord: NDArray, normal: NDArray):
        """Helper function to generate a contour plane."""
        contour = ScalarCutPlane()
        engine.add_filter(contour, gw)
        contour.implicit_plane.widget.enabled = False
        contour.implicit_plane.plane.origin = coord
        contour.implicit_plane.plane.normal = normal
        contour.actor.property.line_width = 5
        contour.actor.property.opacity = 0.5

    if wireframe:
        wire_intervals = np.linspace(-display_radius, display_radius, 14)

        # Generate contours along x and y axes
        for c in wire_intervals:
            gen_contour(np.array([c, 0, 0]), np.array([1, 0, 0]))  # X-axis contours
            gen_contour(np.array([0, c, 0]), np.array([0, 1, 0]))  # Y-axis contours

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

def convert_to_movie(input_path: str, movie_name: str, fps: int = 24) -> None:
    """
    Converts a series of .png files into a movie using OpenCV.
    :param input_path: Path to the directory containing the .png files
    :param movie_name: Name of the movie file
    :param fps: Frames per second (24 by default)
    """
    # Get sorted list of PNG files and their full paths
    frames = sorted(f for f in os.listdir(input_path) if f.endswith(".png"))
    full_paths = (os.path.join(input_path, f) for f in frames)  # Generator for paths

    # Read first frame to get dimensions
    first_frame = cv2.imread(next(full_paths))
    height, width, _ = first_frame.shape

    # Initialize video writer with first frame
    video = cv2.VideoWriter(
        os.path.join(input_path, f"{movie_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    video.write(first_frame)

    # Write remaining frames
    for path in full_paths:
        video.write(cv2.imread(path))

    video.release()

def ask_user(message: str):
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

def max_bh_dist(file_path, x_col, y_col):
    # Load only the required columns and compute squares simultaneously
    x, y = np.loadtxt(file_path, usecols=(x_col, y_col), unpack=True)

    # Find maximum of squared values first (avoids sqrt on all elements)
    max_squared = np.max(x*x + y*y)

    # Single sqrt operation at the end
    return np.sqrt(max_squared)

def max_strain_values(data):
    """Identifies peak strain values where increasing/decreasing trends change."""
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
            if current_val >= prev_val:  # Valley detected
                peaks.append((t-1, times[t-1], abs(prev_val)))
                increasing = True
            elif current_val < prev_val:
                increasing = False

        prev_val = current_val

    # Add final point
    peaks.append((len(strain_vals)-1, times[-1], abs(strain_vals[-1])))

    return np.array(peaks)

def max_value(data):
    """Finds the last occurrence of the maximum value in the third column."""
    third_col = data[:, 2]
    max_val = third_col.max()
    last_max_idx = len(third_col) - 1 - np.argmax(third_col[::-1] == max_val)
    return last_max_idx, data[last_max_idx, 0], max_val

def get_local_maxima(data):
    """Identifies local maxima in the third column of the dataset."""
    # Precompute columns for faster access
    time_col = data[:, 0]
    y_col = data[:, 1]
    values = data[:, 2]

    local_max = []  # Use list for O(1) appends
    decreasing = False
    prev_value = values[0]

    for t in range(1, len(values)):
        current_value = values[t]

        if decreasing:
            if current_value > prev_value:
                decreasing = False
        else:
            if current_value < prev_value:
                # Record peak at previous timestamp
                local_max.append([time_col[t-1], y_col[t-1], prev_value])
                decreasing = True
            elif current_value < prev_value:
                decreasing = True

        prev_value = current_value

    # Add final point if still increasing
    if not decreasing:
        local_max.append([time_col[-1], y_col[-1], values[-1]])

    return np.array(local_max, dtype=np.float64) if local_max else np.empty((0, 3))

def local_max_iterative(data):
    max_data = get_local_maxima(data)
    if max_data.shape[0] > 4:
        max_data = local_max_iterative(max_data)
    return max_data

def calculate_zoomout_time(local_max_data):
    """Calculate zoomout time using the two largest values in the third column."""
    # Extract relevant column once
    values = local_max_data[:, 2]

    # Find first maximum (last occurrence)
    max1_idx = len(values) - 1 - np.argmax(values[::-1] == np.max(values))
    max1_val = values[max1_idx]
    max1_time = local_max_data[max1_idx, 0]

    if len(values) < 2:
        return float('inf'), max1_val

    # Find second maximum (last occurrence excluding first max)
    mask = np.arange(len(values)) != max1_idx
    remaining_values = values[mask]
    max2_val = np.max(remaining_values)
    max2_idx = np.where((values == max2_val) & mask)[0][-1]
    max2_time = local_max_data[max2_idx, 0]

    return int((max1_time + max2_time) // 2), max1_val

def extract_max_strain_and_zoomout_time(dir):
    """Optimized version using batched file I/O and vectorized operations."""
    max_strain = float('-inf')
    min_zoomout_time = float('inf')

    for l in range(ELL_MIN, ELL_MAX + 1):
        file_path = os.path.join(dir, f"Rpsi4_r0100.0_l{l}_conv_to_strain.txt")

        # Precompute all possible columns for this l
        max_col = 4 * l + 2
        try:
            # Load all relevant columns in one read
            data_all = np.loadtxt(file_path, skiprows=4*l+3, usecols=range(0, max_col+1))
        except (FileNotFoundError, IndexError):
            continue

        # Precompute magnitudes for all possible m values
        mag_data = np.empty((data_all.shape[0], 0))
        m_values = []
        valid_columns = []

        for m in range(-l, l+1):
            col1 = 2*l + 2*m + 1
            col2 = col1 + 1
            if col2 >= data_all.shape[1]:
                continue
            valid_columns.append((m, col1, col2))

        # Batch process valid m values
        for m, col1, col2 in valid_columns:
            real = data_all[:, col1]
            imag = data_all[:, col2]
            magnitude = np.hypot(real, imag)  # Faster than custom cmplx_magnitude
            time_col = data_all[:, 0]
            data = np.column_stack((time_col, magnitude))

            # Find peaks and zoomout time
            max_strain_data = max_strain_values(data)
            if max_strain_data.size == 0:
                continue

            data_max_local_max = local_max_iterative(max_strain_data)
            zi, mv = calculate_zoomout_time(data_max_local_max)

            # Update tracking values
            if mv > max_strain:
                max_strain = mv
            if zi != float('inf'):
                zt = time_col[int(zi)]
                if zt < min_zoomout_time:
                    min_zoomout_time = zt

    return max_strain, min_zoomout_time

def main() -> None:
    """
    Main function that reads the strain data,
    calculates and factors in spin-weighted spherical harmonics,
    linearly interpolates the strain to fit the mesh points,
    and creates .tvtk mesh file for each time state of the simulation.
    The meshes represen1, 0] > mt:t the full superimposed waveform at the polar angle pi/2,
    aka the same plane as the binary black hole merger. At each state, the black holes
    are moved to their respective positions and the mesh is saved as a .png file.
    """

    # Convert psi4 data to strain using imported script
    # psi4_to_strain.main()

    # Check initial parameters
    time0 = time.time()
    if USE_SYS_ARGS:
        if len(sys.argv) != 5:
            raise RuntimeError(
                """Please include path to merger data as well as the psi4 extraction radius of that data.
                Usage (spaces between arguments): python3 
                                                  scripts/animation_main.py 
                                                  <path to data folder> 
                                                  <extraction radius (r/M) (4 digits, e.g. 0100)>
                                                  <mass of one black hole>
                                                  <mass of other black hole>"""
            )
        else:
            # change directories and extraction radius based on inputs
            bh_dir = str(sys.argv[1])
            psi4_output_dir = os.path.join(bh_dir, "converted_strain")
            ext_rad = float(sys.argv[2])
            bh1_mass = float(sys.argv[3])
            bh2_mass = float(sys.argv[4])
            movie_dir = os.path.join(str(sys.argv[1]), "movies")
        #if ask_user(
        #    f"Save converted strain to {bh_dir} ? (Y/N): "
        #):
        #    psi4strain.WRITE_FILES = True
    else:
        bh_dir = BH_DIR
        movie_dir = MOVIE_DIR
        ext_rad = EXT_RAD
        psi4_output_dir = os.path.join(bh_dir, "converted_strain")
        # mass ratio for default system GW150914
        bh1_mass = 1
        bh2_mass = 1.24

    # File names
    bh_file_name = "puncture_posns_vels_regridxyzU.txt"
    bh_file_path = os.path.join(bh_dir, bh_file_name)
    bh_scaling_factor = 1

    movie_dir_name = "real_movie2"
    movie_file_path = os.path.join(movie_dir, movie_dir_name)

    if os.path.exists(movie_file_path):
        if ask_user(
            f"""{movie_file_path} already exists. Would you like to overwrite it? Y/N: """
        ) == False:
            print("Please choose a different directory.")
            exit()
        for file in os.listdir(movie_file_path):
            os.remove(os.path.join(movie_file_path, file))
    else:
        os.makedirs(movie_file_path)
    time1 = time.time()
    # Mathematical parameters
    n_rad_pts = 450
    n_azi_pts = 180
    display_radius = 300
    omitted_radius_length = 1.45*(max_bh_dist(bh_file_path, 3, 4) + bh_scaling_factor * max(bh1_mass, bh2_mass))
    colat = np.pi / 2  # colatitude angle representative of the plane of merger

    # Cosmetic parameters
    wireframe = True
    frames_per_second = 24
    save_rate = 10  # Saves every Nth frame
    resolution = (1920, 1080)
    gw_color = (0.28, 0.46, 1.0)
    bh_color = (0.1, 0.1, 0.1)
    time2 = time.time()

    # ---Preliminary Calculations---
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Initializing grid points..."""
        )
    strain_array, grid, points = initialize_tvtk_grid(n_azi_pts, n_rad_pts)
    width = 0.5 * omitted_radius_length
    dropoff_radius = width + omitted_radius_length
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

    # theta and radius values for the mesh
    radius_values = np.linspace(0, display_radius, n_rad_pts)
    azimuth_values = np.linspace(0, 2 * np.pi, n_azi_pts, endpoint=False)

    rv, az = np.meshgrid(radius_values, azimuth_values, indexing="ij")
    x_values = rv * np.cos(az)
    y_values = rv * np.sin(az)
    time4=time.time()

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Calculating cosmetic data..."""
        )

    max_strain, zoomout_time = extract_max_strain_and_zoomout_time(os.path.join(bh_dir, 'converted_strain'))
    amplitude_scale_factor = 80 / max_strain

    print(f"Max strain: {max_strain}")
    print(f"Amplitude scale factor: {amplitude_scale_factor}")
    print(f"Zoomout time: {zoomout_time}")

    # Vectorized time array generation
    initial_time, final_time = time_array[0], time_array[-1]
    equal_times = np.linspace(initial_time, final_time, num=n_times)

    # Find closest index using vectorized operations
    zoomout_diff = np.abs(equal_times - zoomout_time)
    zero_match = np.where(zoomout_diff == 0)[0]
    zoomout_idx = zero_match[0] if zero_match.size > 0 else np.argmin(zoomout_diff)

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Constructing mesh points in 3D..."""
        )

    # Apply spin-weighted spherical harmonics, superimpose modes, and interpolate to mesh points
    strain_azi = swsh_summation_angles(colat, azimuth_values, mode_data).real
    lerp_times = generate_interpolation_points(equal_times, radius_values, ext_rad)
    strain_to_mesh = np.zeros((n_rad_pts, n_azi_pts, n_times))

    for i in range(n_azi_pts):
        # strain_azi, a function of time_array, is evaluated at t = lerp_times.
        strain_to_mesh[:, i, :] = np.interp(lerp_times, time_array, strain_azi[i, :])

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Calculating black hole trajectories..."""
        )
    time5=time.time()
    # Import black hole data
    if bh1_mass > bh2_mass:  # then swap
        bh1_mass, bh2_mass = bh2_mass, bh1_mass
    # _ = next(reader)  # uncomment to skip the header row
    bh_data = np.loadtxt(bh_file_path, skiprows=14)
    bh_time = bh_data[:, 0]
    # x is flipped because the data is in a different coordinate system
    bh1_x0 = -bh_data[:, 3]
    # z axis in the data is interpreted as y axis in the visualization
    bh1_y0 = bh_data[:, 4]
    bh1_z0 = np.zeros(len(bh1_x0))

    bh1_x, bh1_y, bh1_z = interpolate_coords_by_time(
        bh_time, bh1_x0, bh1_y0, bh1_z0, equal_times
    )

    bh_mass_ratio = bh1_mass / bh2_mass
    bh2_x = -bh1_x * bh_mass_ratio
    bh2_y = -bh1_y * bh_mass_ratio
    bh2_z = -bh1_z * bh_mass_ratio
    time6=time.time()
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Initializing animation..."""
        )

    # Configure engine and rendering upfront
    mlab.options.offscreen = False  # Keep False for interactive, set True for rendering
    engine = Engine()
    engine.start()
    fig = mlab.figure(engine=engine, size=resolution)

    # Precompute all constant values once
    bh1_scaled = bh1_mass * bh_scaling_factor
    bh2_scaled = bh2_mass * bh_scaling_factor
    valid_indices = np.arange(0, n_times, save_rate)
    n_valid = len(valid_indices)

    # Initialize visualization objects once
    create_gw(engine, grid, gw_color, display_radius, wireframe)
    bh1 = create_sphere(engine, bh1_scaled, bh_color)
    bh2 = create_sphere(engine, bh2_scaled, bh_color)

    # Precompute geometric data with vectorization
    R, A = np.meshgrid(radius_values, azimuth_values, indexing='ij')
    valid_mask = (R > omitted_radius_length).ravel()
    dropoff_2D = (0.5 + 0.5 * erf((R - dropoff_radius)/width)).ravel() * amplitude_scale_factor
    x_flat, y_flat = x_values.ravel(), y_values.ravel()

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
    elevations = np.maximum(50 - time_indices * 0.016, 34)
    distances = np.minimum(np.where(
        time_indices < zoomout_idx,
        80,
        80 + (time_indices - zoomout_idx) * 0.175
    ), 370)

    # Initialize timing and progress tracking
    start_time = time.time()
    status_interval = max(n_valid // 10, 1)


    start_time = time.time()
    percentage = list(np.round(np.linspace(0, n_times, 100)).astype(int))
    time6=time.time()
    print(f"0:{time1-time0}\n1:{time2-time1}\n2:{time3-time2}\n3:{time4-time3}\n4:{time5-time4}\n5:{time6-time5}\na:{time6-time0}\n")
    @mlab.animate(delay=10, ui=False)
    def anim():
        for idx, time_idx in enumerate(valid_indices):
             # Print status messages
            if time_idx == 10 * save_rate:
                end_time = time.time()
                eta = (end_time - start_time) * n_frames / 10
                print(
                    f"""Creating {n_frames} frames and saving them to:
{movie_file_path}\nEstimated time: {dhms_time(eta)}"""
                )
            if STATUS_MESSAGES and time_idx != 0 and time_idx > percentage[0]:
                eta = ((time.time() - start_time) / time_idx) * (n_times - time_idx)
                print(
                    f"{int(time_idx  * 100 / n_times)}% done, ",
                    f"{dhms_time(eta)} remaining",
                    end="\r",
                )
                percentage.pop(0)

            # Update positions
            change_object_position(bh1, (bh1_x[time_idx], bh1_y[time_idx], bh1_z[time_idx]))
            change_object_position(bh2, (bh2_x[time_idx], bh2_y[time_idx], bh2_z[time_idx]))

            # Update strain visualization
            strain_flat = strain_to_mesh[..., time_idx].ravel()
            np_points[:, 2] = np.where(valid_mask, strain_flat * dropoff_2D, np.nan)
            vtk_array.modified()

            # Update grid
            strain_array.from_array(strain_flat[valid_mask])
            grid._set_points(points)
            grid.modified()

            # Update camera
            mlab.view(
                elevation=elevations[time_idx],
                distance=distances[time_idx],
                focalpoint=(0, 0, 0)
            )

            # Save frame
            mlab.savefig(os.path.join(movie_file_path, f"z_frame_{idx:05d}.png"))

            # Early exit check
            if idx == n_valid - 1:
                total_time = time.time() - start_time
                print("Done", end="\r")
                print(
                    f"\nSaved {n_frames} frames to {movie_file_path} ",
                    f"in {dhms_time(total_time)}.",
                )
                print("Creating movie...")
                convert_to_movie(movie_file_path, movie_dir_name, frames_per_second)
                print(f"Movie saved to {movie_file_path}/{movie_dir_name}.mp4")
                mlab.close()
                sys.exit(0)

            yield

    _ = anim()
    mlab.show()

    """# Create Mayavi objects
    # Configure engine and rendering upfront
    mlab.options.offscreen = False  # Uncomment for 30-50% faster rendering
    engine = Engine()
    engine.start()
    mlab.figure(engine=engine, size=resolution)

    # Precompute scaled values once
    bh1_scaled = bh1_mass * bh_scaling_factor
    bh2_scaled = bh2_mass * bh_scaling_factor

    # Create visualization objects
    create_gw(engine, grid, gw_color, display_radius, wireframe)
    bh1 = create_sphere(engine, bh1_scaled, bh_color)
    bh2 = create_sphere(engine, bh2_scaled, bh_color)

    # Set initial view parameters in one call
    mlab.view(
        azimuth=60, 
        elevation=50, 
        distance=80, 
        focalpoint=(0, 0, 0),
        reset_roll=True
    )

    # Initialize timing and progress tracking
    start_time = time.time()
    percentage = np.round(np.linspace(0, n_times, 100)).astype(int).tolist()

    def anim():

        # Precompute all valid time indices
        valid_indices = [i for i in range(n_times) if i % save_rate == 0]
        n_valid = len(valid_indices)

        # Precompute geometric data with proper broadcasting
        dropoff_factors = 0.5 + 0.5 * erf((radius_values - dropoff_radius) / width)
        valid_radii = radius_values > omitted_radius_length
        x_flat = x_values.ravel()
        y_flat = y_values.ravel()
        n_points = len(x_flat)

        # Create meshgrid for broadcasting
        R, A = np.meshgrid(radius_values, azimuth_values, indexing='ij')
        valid_mask = (R > omitted_radius_length).ravel()
        dropoff_2D = (dropoff_factors[:, np.newaxis] * np.ones_like(A)).ravel()

        # Precompute camera parameters
        elevations = np.maximum(50 - np.arange(n_times) * 0.016, 34)
        distances = np.where(
            np.arange(n_times) < zoomout_idx,
            80,
            np.minimum(80 + (np.arange(n_times) - zoomout_idx) * 0.175, 370)
        )

        # Initialize VTK points array properly
        points = tvtk.Points()
        vtk_array = tvtk.FloatArray()
        vtk_array.number_of_components = 3
        vtk_array.number_of_tuples = n_points
        points.data = vtk_array

        # Get numpy array view with proper shape (N, 3)
        np_points = vtk_array.to_array().reshape(-1, 3)
        np_points[:, 0] = x_flat  # Set X coordinates
        np_points[:, 1] = y_flat  # Set Y coordinates

        for idx, time_idx in enumerate(valid_indices):
            # Status messages
            if idx == 10:
                eta = (time.time() - start_time) * n_valid / 10
                print(f"Creating {n_valid} frames\nETA: {dhms_time(eta)}")

            # Update black hole positions
            change_object_position(bh1, (bh1_x[time_idx], bh1_y[time_idx], bh1_z[time_idx]))
            change_object_position(bh2, (bh2_x[time_idx], bh2_y[time_idx], bh2_z[time_idx]))

            # Calculate strain values
            strain_flat = strain_to_mesh[..., time_idx].ravel()

            # Update Z-values using numpy interface
            z_values = np.where(
                valid_mask,
                strain_flat * amplitude_scale_factor * dropoff_2D,
                np.nan
            )

            # Update Z coordinates directly in preallocated array
            np_points[:, 2] = z_values
            vtk_array.modified()
            points.modified()

            # Update strain array
            strain_array.from_array(strain_flat[valid_mask])
            grid._set_points(points)
            grid.modified()

            # Update camera view
            mlab.view(
                elevation=elevations[time_idx],
                distance=distances[time_idx],
                focalpoint=(0, 0, 0)
            )

            # Save frame
            mlab.savefig(os.path.join(movie_file_path, f"z_frame_{idx:05d}.png"))

            # Exit condition
            if idx == n_valid - 1:
                total_time = time.time() - start_time
                print(f"\nSaved {n_valid} frames in {dhms_time(total_time)}")
                convert_to_movie(movie_file_path, movie_dir_name, frames_per_second)
                mlab.close()
                exit()

    _ = anim()
    mlab.show()"""


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
