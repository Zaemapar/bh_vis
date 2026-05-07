"""
  A script that reads in gravitational wave psi 4 data and black hole positional data,
  applies spin-weighted spherical harmonics to the data, and creates a Mayavi animation
  of the black holes and their gravitational waves. At each state, the black holes are
  moved to their respective positions and the render is saved as a .png file.
  """

  # --- DEFAULT PARAMETERS FOR VISUALIZATION ---
  # Defaults for the appearance of the movie/rendering process

  USE_SYS_ARGS = True # Change to turn on/off default parameters. Leave on if you want to input your own data.
  STATUS_MESSAGES = True # Change to turn on/off status reports during rendering
  TRAJECTORY_LINES = False # Change to turn on/off lines tracking trajectories of black holes
  PIP_VIEW = False # Change to turn on/off picture in picture view at corner of movie showing close up of black holes
  FREQ_SOUND = True # Change to turn on/off background sound based on strain frequency
  APPARENT_HORIZONS = False # Change to turn on/off accurate horizon rendering. Use only if you have data.
  SPIN_VECTORS = False # Change to turn on/off spin vectors for black holes
  COLORMAP = False # Change to turn on/off cross-polarized colormap
  MASS = 1 # Total mass of the system, in solar masses
  MOVIE_LENGTH = 10 # Length of aesthetic movie in seconds
  THREE_DIMENSIONAL_VIS = False # Change to turn on/off 3D visualization of strain.
  USE_SYMLOG = False # Change to turn on/off use of symmetric log scale for strain. Can improve visibility of early inspiral, but
may cause visual artifacts.

  import os
  import sys
  import time
  import psutil
  import math
  from typing import Tuple, Any, List, Union
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
  from mayavi.filters.set_active_attribute import SetActiveAttribute
  from mayavi.modules.iso_surface import IsoSurface
  import psi4_FFI_to_strain as psi4strain
  import traceback
  from scipy.io.wavfile import write as write_wav
  from moviepy import VideoFileClip, AudioFileClip
  import matplotlib.pyplot as plt
  import ffmpeg

  # Default parameters used when USE_SYS_ARGS is True
  BH_DIR = "../data/GW150914_data/r100"
  MOVIE_DIR = "../data/GW150914_data/movies"
  S_MODE = -2
  ALPHA = 0
  DELTA = 0
  PSI = 0
  EXT_RAD = 100

  # Other misc constants
  GC3 = 4.92549e-6 # Gravitational constant times speed of light cubed, in seconds per solar mass. Used for unit conversions.

  def compute_detector_strain(complex_strain: NDArray[np.complex128], alpha: float, delta: float, psi: float) ->
NDArray[np.float64]:
      # F_+ = 1/2 (1 + cos^2(delta)) cos(2 alpha) cos(2 psi) - cos(delta) sin(2 alpha) sin(2 psi)
      f_plus = 0.5 * (1 + np.cos(delta)**2) * np.cos(2 * alpha) * np.cos(2 * psi) - np.cos(delta) * np.sin(2 * alpha) * np.sin(2 *
psi)
      # F_x = 1/2 (1 + cos^2(delta)) cos(2 alpha) sin(2 psi) + cos(delta) sin(2 alpha) cos(2 psi)
      f_cross = 0.5 * (1 + np.cos(delta)**2) * np.cos(2 * alpha) * np.sin(2 * psi) + np.cos(delta) * np.sin(2 * alpha) * np.cos(2 *
psi)

      # Combined strain
      return f_plus * np.real(complex_strain) + f_cross * np.imag(complex_strain)

  def swsh_summation_angles(
      colat: NDArray[np.float64],
      azi: NDArray[np.float64],
      mode_data: NDArray[np.complex128],
      ell_min: int,
      ell_max: int,
      status_messages=True,
  ) -> NDArray[np.complex128]:
      """
      Optimized SWSH summation using a single BLAS-backed matrix multiplication.

      Computes the spin-weighted spherical harmonic matrix once and performs
      a matrix multiply to combine it with `mode_data`:

          result = swsh_arr.T @ mode_data

      This replaces the previous Python loop over modes with a single
      high-performance BLAS call.
      """

      n_times = mode_data.shape[1]
      n_colat = len(colat)
      n_azi = len(azi)

      if status_messages:
          print("Computing quaternionic coordinates and Wigner SWSH matrix...")

      quat_arr = quaternionic.array.from_spherical_coordinates(colat, azi)
      winger = spherical.Wigner(ell_max, ell_min)
      swsh_arr = winger.sYlm(S_MODE, quat_arr)

      # Sanity check shapes
      if swsh_arr.shape[1] != mode_data.shape[0]:
          raise ValueError(
              "Mismatch in number of modes between computed SWSH array and provided mode_data."
              f" swsh_arr.shape={swsh_arr.shape}, mode_data.shape={mode_data.shape}"
          )

      flat_result = swsh_arr @ mode_data
      result_3d = flat_result.reshape((n_azi, n_colat, n_times)) # Reshape to fit the expected output dimensions

      if status_messages:
          print("SWSH summation complete.")

      return result_3d

  def initialize_tvtk_grid(num_azi: int, num_radius: int, num_colat: int) -> Tuple:
      """
      Set initial parameters for the mesh generation module and return
      a circular, polar mesh with manipulation objects to write and save data.

      :param num_azi: Number of azimuthal points on the mesh.
      :param num_radius: Number of radial points on the mesh.
      :param num_colat: Number of colatitude points on the mesh.
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
          name="Strain", number_of_components=1, number_of_tuples=num_azi * num_radius * num_colat
      )

      # Precompute next_i for all azimuthal points
      next_i_list = [(i + 1) % num_azi for i in range(num_azi)]

      # Precompute cell connectivity data
      cells_data = []
      points_per_layer = num_azi * num_radius

      colat_iterate = range(num_colat - 1) if num_colat > 1 else range(num_colat)

      # Iterate through polar, radial, and azimuthal layers
      for k in colat_iterate:
          if num_colat > 1:
              this_k = k * points_per_layer
              next_k = (k + 1) * points_per_layer
          else:
              this_k = None
              next_k = None

          for j in range(num_radius - 1):
              this_j = j * num_azi
              next_j = (j + 1) * num_azi

              for i in range(num_azi):
                  ni = next_i_list[i]

                  # VTK Hexahedron Node Ordering:
                  # Bottom face (k) -> ordered counter-clockwise
                  p0 = i + this_j + (this_k if num_colat > 1 else 0)
                  p1 = ni + this_j + (this_k if num_colat > 1 else 0)
                  p2 = ni + next_j + (this_k if num_colat > 1 else 0)
                  p3 = i + next_j + (this_k if num_colat > 1 else 0)

                  if num_colat > 1:
                      # Top face (k+1) -> ordered identically to bottom face
                      p4 = i + this_j + next_k
                      p5 = ni + this_j + next_k
                      p6 = ni + next_j + next_k
                      p7 = i + next_j + next_k

                      # 8 indicates the number of points in this cell
                      cells_data.extend([8, p0, p1, p2, p3, p4, p5, p6, p7])

                  else:
                      # If only one polar layer, we create quads instead of hexahedrons
                      cells_data.extend([4, p0, p1, p2, p3])

      # Convert to VTK cell array
      cell_data = np.array(cells_data, dtype=np.int64)
      id_array = tvtk.IdTypeArray()
      id_array.from_array(cell_data)

      num_cells = (num_radius - 1) * num_azi * (num_colat - 1) if num_colat > 1 else (num_radius - 1) * num_azi
      cell_array = tvtk.CellArray()
      cell_array.set_cells(num_cells, id_array)

      # Configure grid
      cell_type = tvtk.Hexahedron().cell_type if num_colat > 1 else tvtk.Quad().cell_type
      grid.set_cells(cell_type, cell_array)

      return strain_array, grid, points

  def create_gw(
      engine: Any, # Mayavi Engine instance expected
      grid: tvtk.UnstructuredGrid,
      color,
      display_radius: int,
      wireframe: bool = False,
      COLORMAP: bool = False
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

      if COLORMAP:
          # Change the colormap
          surface.module_manager.scalar_lut_manager.lut_mode = 'plasma'

      else:
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
          contour.actor.property.color = (1.0, 1.0, 1.0)
          contour.actor.mapper.scalar_visibility = False

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

      return gw

  def create_isosurfaces(
      engine: Any,
      grid: tvtk.UnstructuredGrid,
      mode_names: [str],
      contours: Union[int, List[float]] = 4,
      opacity: float = 0.3,
      palette: Tuple[Tuple[float, float, float], ...] = ((0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2))
  ) -> dict:
      """
      Create and display multiple isosurfaces from a single grid, one for each l,m mode.

      :param engine: Mayavi engine for visualization.
      :param grid: tvtk.UnstructuredGrid representing the strain data containing named arrays.
      :param mode_names: List of string names corresponding to the arrays in the grid.
      :param contours: Integer specifying number of contours, or list of explicit contour values.
      :param opacity: Float (0.0 to 1.0) defining the transparency of the isosurfaces.
      :param palette: Tuple of RGB tuples to color each mode distinctly.

      :return: Dictionary mapping mode names to their IsoSurface module objects.
      """

      # Get the current scene efficiently
      scene = getattr(engine, "current_scene", engine.scenes[0])

      # Create and configure the base data source
      base_source = VTKDataSource(data=grid)
      engine.add_source(base_source, scene)

      isosurfaces = {}

      for i, mode_name in enumerate(mode_names):
          # 1. Branch the pipeline: Tell Mayavi to look at this specific array
          active_attr = SetActiveAttribute()
          active_attr.point_scalars_name = mode_name
          engine.add_filter(active_attr, base_source)

          # 2. Add the IsoSurface module to this specific branch
          iso = IsoSurface()
          engine.add_module(iso, active_attr)

          # 3. Configure the contours
          if isinstance(contours, int):
              iso.contour.number_of_contours = contours
          else:
              iso.contour.contours = contours

          # 4. Configure visual properties
          iso.actor.property.opacity = opacity
          iso.actor.mapper.scalar_visibility = False # Disable colormap to use solid colors
          iso.actor.property.color = palette[i % len(palette)]

          isosurfaces[mode_name] = iso

      # Return the dictionary in case you need to interact with the isosurfaces later
      return isosurfaces

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

> def compute_strain_to_mesh(
      strain_azi_colat: NDArray[np.float32],
      equal_times: NDArray[np.float32], # Not used directly, but lerp_times depends on it
      radius_values: NDArray[np.float32], # Not used directly, but lerp_times depends on it
      colat_values: NDArray[np.float32], # Not used directly, but typically defines the polar dimension of the mesh
      lerp_times: NDArray[np.float32],  # Precomputed 2D array of interpolation points
      time_array: NDArray[np.float32],
      dropoff_2D_flat: NDArray[np.float32],
      use_symlog: bool,
      mmap_filename: str,
      status_messages=STATUS_MESSAGES,
      three_dimensional_vis=THREE_DIMENSIONAL_VIS
  ) -> np.memmap:
      """
      Interpolates strain data onto a spherical mesh grid over specific time points.

      This function takes precomputed strain data, originally defined over a set
      of time points (`time_array`) for various azimuths and colatitudes (`strain_azi`), and
      interpolates it onto a target spatio-temporal grid. The spatial grid
      is implicitly polar (radius and azimuth), and the target time points
      vary with radius, as defined by `lerp_times`.

      The interpolation applies a radial dropoff factor (`dropoff_2D_flat`) and
      optionally uses a symmetric logarithmic scaling (`symlog`) on the strain
      values before interpolation.

      To manage memory, especially for a large number of azimuth points, the
      interpolation is performed in chunks along the azimuth dimension.


      :param strain_azi_colat: Precomputed strain values, summed over modes. Shape: (n_azi_pts, n_original_times).
      :param equal_times: Array of equally spaced times for the final desired mesh state. Shape: (n_times,). Note:
                          Not used directly in calculations, but typically used to generate `lerp_times` and defines the
                          size of the time dimension in the output.
      :param radius_values: Array of radius values for the mesh grid. Shape: (n_rad_pts,). Note: Not used directly in
                            calculations, but typically used to generate `lerp_times` and defines the size of the radius
                            dimension in the output.
      :param colat_values: Array of colatitude values for the mesh grid. Shape: (n_colat_pts,). Note: Not used directly in
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
      n_azi_pts = strain_azi_colat.shape[0]
      n_colat_pts = strain_azi_colat.shape[1]
      n_times = len(equal_times)

      memory_budget = min(0.05 * psutil.virtual_memory().available, 512_000_000)

      # Calculate bytes per element: n_times * 4 bytes
      bytes_per_slice = n_times * 4

      # Calculate max chunk sizes.
      chunk_azi = max(1, int(np.sqrt(memory_budget / bytes_per_slice)))
      chunk_rad = max(1, int(np.sqrt(memory_budget / bytes_per_slice)))
      chunk_colat = max(1, int(np.sqrt(memory_budget / bytes_per_slice)))

      if status_messages:
           print(f"Memory budget: {memory_budget / 1_000_000:.2f} MB")
           print(f"Using chunk size: azi={chunk_azi}, colat={chunk_colat}, rad={chunk_rad} for interpolation.")

      strain_to_mesh = np.memmap(mmap_filename, dtype=np.float32, mode='w+', shape=(n_rad_pts, n_azi_pts, n_colat_pts, n_times))

      if status_messages:
          n_points = n_azi_pts * n_rad_pts * n_colat_pts
          point = 0

      for azi_start in range(0, n_azi_pts, chunk_azi):
          azi_end = min(azi_start + chunk_azi, n_azi_pts)

          # apply symlog locally to chunk
          strain_azi_chunk = strain_azi_colat[azi_start:azi_end, :, :]
          if use_symlog:
              strain_azi_chunk = np.sign(strain_azi_chunk) * np.log1p(np.abs(strain_azi_chunk))

          for colat_start in range(0, n_colat_pts, chunk_colat):
              colat_end = min(colat_start + chunk_colat, n_colat_pts)
              strain_colat_chunk = strain_azi_chunk[:, colat_start:colat_end, :]

              for r_start in range(0, n_rad_pts, chunk_rad):
                  r_end = min(r_start + chunk_rad, n_rad_pts)

                  chunk_buffer = np.zeros((r_end - r_start, azi_end - azi_start, colat_end - colat_start, n_times),
dtype=np.float32)

                  for i, r_idx in enumerate(range(r_start, r_end)):
                      for j, azi_idx in enumerate(range(azi_start, azi_end)):
                          for k, colat_idx in enumerate(range(colat_start, colat_end)):
                              chunk_buffer[i, j, k, :] = np.interp(
                                  lerp_times[r_idx, :],
                                  time_array,
                                  strain_azi_chunk[j, k, :]
                              ) * (dropoff_2D_flat[r_idx] if three_dimensional_vis else 1.0)

                              # Update status
                              if status_messages:
                                  point += 1
                                  progress = point / n_points * 100
                                  print(f"\rProgress: {progress:.1f}% completed", end="", flush=True)

                  strain_to_mesh[r_start:r_end, azi_start:azi_end, colat_start:colat_end, :] = chunk_buffer

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
          factor = z_max_azi / np.max(np.abs(swsh_array[idx, 0, :]))

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

  def create_sound_file(
      mode_data: np.ndarray,
      time_array: np.ndarray,
      ell_min: int,
      ell_max: int,
      movie_times: np.ndarray,
      movie_path_name: str,
      fps=24,
      target_rate=48000,
  ) -> Tuple[str, str, int]:

      n_times = len(movie_times)

      num_audio_samples = int(n_times * target_rate)
      audio_movie_times = np.linspace(movie_times[0], movie_times[-1], num_audio_samples)

      shifted_mode_data = np.zeros((len(mode_data), len(audio_movie_times)), dtype=np.complex128)

      # --- Interpolation of amplitude and phase ---
      for i, mode in enumerate(mode_data):
          # Directly use complex properties
          amp = np.abs(mode)
          phase = np.unwrap(np.angle(mode))

          mapped_time_array = np.interp(time_array, [time_array[0], time_array[-1]], [0, movie_times[-1]])

          # Interpolate amp and phase to high-res audio times
          amp_interp = np.interp(audio_movie_times, mapped_time_array, amp)
          phase_interp = np.interp(audio_movie_times, mapped_time_array, phase)

          # Synthesize pitch-shifted mode
          shifted_mode_data[i] = amp_interp * np.exp(1j * phase_interp)

      # Now evaluate SWSH at the single detector point for all modes, on the audio timeline
      audio_colat = np.pi / 2 - DELTA
      audio_azi = np.array([ALPHA]) # Single point

      print("Computing detector strain for audio...")
      complex_strain_audio = swsh_summation_angles(
          [audio_colat],
          audio_azi,
          shifted_mode_data,
          ell_min,
          ell_max,
          status_messages=False
      )[0] # Extract the single azimuth row

      # Apply antenna pattern logic
      final_audio_strain = compute_detector_strain(complex_strain_audio, ALPHA, DELTA, PSI)

      # Fade in and out to prevent popping
      fade_len = int(0.05 * target_rate) # 50ms fade
      if len(final_audio_strain) > 2 * fade_len:
          fade_in = np.linspace(0, 1, fade_len)
          fade_out = np.linspace(1, 0, fade_len)
          final_audio_strain[:fade_len] *= fade_in
          final_audio_strain[-fade_len:] *= fade_out

      # Global amplitude normalization
      max_signal = np.max(np.abs(final_audio_strain))
      if max_signal > 0:
          final_audio_strain = final_audio_strain / max_signal

      # Convert to int16
      audio_int16 = (final_audio_strain * 32767 * 0.95).astype(np.int16)

      # Setup file names
      audio_path = f"{movie_path_name[:-4]}_audio.wav"
      movie_with_audio = f"{movie_path_name[:-4]}_sound.mp4"

      try:
          write_wav(audio_path, target_rate, audio_int16)
          print("Audio generation complete.")
      except Exception as e:
          print(f"\nAn error occurred during audio generation: {e}")
          print("Skipping audio merging.")
          return "", ""

      return audio_path, movie_with_audio

  def extract_extrad_and_modes(psi4_dir: str, strain_dir: str, user_ext_rad: float = None) -> Tuple[np.float64, int, int, bool,
str]:
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
          raise FileNotFoundError(f"No psi4 or strain data found in the directories")

      bh_files = strain_files + psi4_files # Concatenate psi4 and strain files into general files

      extraction_radii = np.empty(0)
      for b in bh_files:
          # Attempt to convert the part of the file name that is supposed to be the extraction radius into a float
          try:
              r_val = float(b[-10:-4])
          except (ValueError, IndexError):
              try:
                  r_val = float(b[-7:-4]) # Handle the case where it might be infinity
              except (ValueError, IndexError):
                  continue
          extraction_radii = np.unique(np.append(extraction_radii, r_val))

      if user_ext_rad is not None:
          radius_extraction = user_ext_rad
          if radius_extraction not in extraction_radii:
              # Format the string of available radii
              radii_string = ""
              for r in extraction_radii:
                  radii_string += str(r) + ", "

              raise FileNotFoundError(f"Extraction radius doesn't exist. Available radii: {radii_string[:-2]}")
      else:
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

      if STATUS_MESSAGES:
          print(f"Using extraction radius {ext_rad if ext_rad >= 0 else radius_extraction} for {'strain' if strain_exists else
'psi_4'} data")

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

      appropriate_bounds = (-10, -4) if ext_rad != float('inf') else (-7, -4) # Appropriate bounds for which to look for extraction
radius
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
          raise RuntimeError(f"A gap was detected in the l modes: Minimum l is {ell_min}, maximum l is {ell_max}, but no
l={gaps[0][0] + ell_min + 1} file was found")

      if STATUS_MESSAGES:
          print(f"Using minimum mode l={ell_min} and maximum mode l={ell_max} for {'strain' if strain_exists else 'psi_4'} data")

      return ext_rad, ell_min, ell_max, r_ext_in_strain, str_ext_rad

  def create_movie_directory(movie_dir: str, alpha: float, delta: float, psi: float) -> Tuple[str, str]:
      # Determine prefix
      if math.isclose(alpha, 0) and math.isclose(delta, 0) and math.isclose(psi, 0, abs_tol=1e-5):
          prefix = "real"
      elif math.isclose(alpha, 0) and math.isclose(delta, 0) and math.isclose(psi, np.pi/4, abs_tol=1e-5):
          prefix = "imag"
      else:
          prefix = f"({alpha:.2f},{delta:.2f},{psi:.2f})"

      # A while loop to figure out where to save the simulation
      movie_number = 1
      while True:
          movie_dir_name = f"{prefix}_movie{movie_number}"
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

      # Full path + name of the movie + extension
      movie_path_name = os.path.join(movie_file_path, movie_dir_name) + ".mp4" # This is the physics-accurate movie

      return movie_file_path, movie_path_name

  def extract_existing_strain_data(psi4_output_dir: str, ell_min: int, ell_max: int, str_ext_rad: str) ->
Tuple[NDArray[np.float64], NDArray[np.complex128]]:

      # Setup variables
      time_array_set = False
      mode_data_list = []

      # Iterate through each l mode and extract the appropriate strain data
      files_processed = 0
      for l in range(ell_min, ell_max + 1):

          # Construct filename and file path safely
          filename = psi4strain.STRAIN_FILE_FMT + f"_l{l}-r{str_ext_rad}.txt"
          file_path = os.path.join(psi4_output_dir, filename)

          # Determine expected number of columns based on l
          max_col_idx = 2 * l + 1
          cols_to_use = range(0, max_col_idx + 1)

          # Try to load the text files
          try:
              num_skip_rows = 2 * l + 2
              data_all = np.loadtxt(file_path, dtype=np.complex128, skiprows=num_skip_rows, usecols=cols_to_use)
              files_processed += 1
          except FileNotFoundError:
              print(f"Warning: File not found {file_path}, skipping l={l}.")
              continue
          except (ValueError, IndexError) as e:
              print(f"Warning: Error loading {file_path}: {e}. Skipping l={l}.")
              continue

          # For the first l mode, set the time array
          if not time_array_set:
              time_array = data_all[:, 0].real
              time_array_set = True

          # Append mode data to list for efficient stacking
          mode_data_list.append(data_all[:, 1:].T)

      # If no files were processed, end the program
      if files_processed == 0:
          raise FileNotFoundError(f"No valid strain files found in directory {psi4_output_dir} for l={ell_min} to {ell_max}.")

      # Perform a single, efficient vstack after collecting all data
      mode_data = np.vstack(mode_data_list)

      return time_array, mode_data

  def compute_max_bh_separation(bh_azis: NDArray[np.float64], magnitudes: NDArray[np.float64]) -> float:
      orbit_start = find_idx(np.array(bh_azis), bh_azis[0] + (np.pi / 2 if bh_azis[0] < bh_azis[1] else np.pi / -2))

      # If the array is 0 size, or if it only contains edge cases, set equal to last azi
      if orbit_start.size == 0 or orbit_start[0] == bh_azis.size - 1:
          orbit_start_idx = len(bh_azis) - 2
      else:
          orbit_start_idx = orbit_start[0]

      orbit_end = find_idx(np.array(bh_azis), bh_azis[0] + (5 * np.pi / 2 if bh_azis[0] < bh_azis[1] else 5 * np.pi / -2))

      if orbit_end.size == 0 or orbit_end[0] == bh_azis.size - 1:
          orbit_end_idx = len(bh_azis) - 1
      else:
          orbit_end_idx = orbit_end[0]

      return np.max(magnitudes[orbit_start_idx:orbit_end_idx])

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

      mlab.close(all=True) # Close any existing Mayavi figures/engines

      # Allow modification of global variables based on args
      global BH_DIR, \
          MOVIE_DIR, \
          EXT_RAD, \
          USE_SYS_ARGS, \
          STATUS_MESSAGES, \
          TRAJECTORY_LINES, \
          PIP_VIEW, \
          FREQ_SOUND, \
          APPARENT_HORIZONS, \
          SPIN_VECTORS, \
          COLORMAP, \
          ALPHA, \
          DELTA, \
          PSI, \
          MASS, \
          MOVIE_LENGTH, \
          THREE_DIMENSIONAL_VIS, \
          USE_SYMLOG

      bh1_rel_mass: float = 1.0 # Default mass
      bh2_rel_mass: float = 1.24 # Default mass ratio for GW150914
      use_symlog: bool = False # Default scale

      import argparse

      def str2bool(v):
          if isinstance(v, bool):
              return v
          if v.lower() in ('yes', 'true', 't', 'y', '1'):
              return True
          elif v.lower() in ('no', 'false', 'f', 'n', '0'):
              return False
          else:
              raise RuntimeError("Boolean value expected.")

      usage_str = f"""Usage: python3 {os.path.basename(__file__)} <simulation_name> [options]\n\n
  Example: python3 {os.path.basename(__file__)} GW150914_data/r100 --use-symlog\n\n
  Arguments:
  \t<simulation_name> Simulation name with a directory in ../data containing merger data and converted strain.
  \t--help: Optional. Display this help message and exit.
  \t--list: Optional. List all available data directories.
  \t--three-dimensional-vis: Optional. Boolean. Toggle three-dimensional visualization. Defaults to {THREE_DIMENSIONAL_VIS}.
  \t--r<value>: Optional. Extraction radius to use, e.g., --r100 or --rinf. Bypasses automatic detection.
  \t--use-default-args: Optional. Boolean. Use default settings instead of parsed arguments. Defaults to {not USE_SYS_ARGS}.
  \t--status-messages: Optional. Boolean. Toggle terminal progress readouts. Defaults to {STATUS_MESSAGES}.
  \t--trajectory-lines: Optional. Boolean. Toggle black hole trajectory tracking lines. Defaults to {TRAJECTORY_LINES}.
  \t--pip-view: Optional. Boolean. Toggle picture-in-picture view showing close-up of black holes. Defaults to {PIP_VIEW}.
  \t--freq-sound: Optional. Boolean. Toggle rendering of background audio based on strain frequency. Defaults to {FREQ_SOUND}.
  \t--apparent-horizons: Optional. Boolean. Toggle accurate horizon rendering (if data available). Defaults to {APPARENT_HORIZONS}.
  \t--spin-vectors: Optional. Boolean. Toggle display of vectors tracking dimensionful spin. Defaults to {SPIN_VECTORS}.
  \t--colormap: Optional. Boolean. Toggle display of colormap corresponding to out-of-phase strain. Defaults to {COLORMAP}.
  \t--detector-angle ALPHA DELTA PSI: Optional. Specify the three detector angles alpha, delta, and psi. Defaults to {ALPHA},
{DELTA}, {PSI}.
  \t--mass: Optional. Specify the total mass in solar masses. Defaults to {MASS}.
  \t--movie-length: Optional. Specify the desired length of a stretched version of the movie in seconds. Defaults to {MOVIE_LENGTH}.
  \t--use-symlog: Optional. Use symmetric log scale for strain. Defaults to {USE_SYMLOG}.
  """

      if "--list" in sys.argv:
          data_path = "../data"
          try:
              available_dirs = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
          except FileNotFoundError:
              available_dirs = []
          dir_str = f"Available directories in {data_path}:\n\n"
          for name in available_dirs:
              dir_str += name + "\n"
          print(f"\n{dir_str}")
          sys.exit(0)

      if len(sys.argv) == 2 and sys.argv[1] == "--help":
          print(usage_str)
          sys.exit(0)

      parser = argparse.ArgumentParser(description="Animate gravitational wave strain.", add_help=False)
      parser.add_argument("path_to_data_folder", nargs="?", default=BH_DIR, help="Path to the directory containing merger data.")
      parser.add_argument("--use-symlog", type=str2bool, nargs='?', const=True, default=USE_SYMLOG)
      parser.add_argument("--use-default-args", type=str2bool, nargs='?', const=True, default=not USE_SYS_ARGS)
      parser.add_argument("--status-messages", type=str2bool, nargs='?', const=True, default=STATUS_MESSAGES)
      parser.add_argument("--trajectory-lines", type=str2bool, nargs='?', const=True, default=TRAJECTORY_LINES)
      parser.add_argument("--pip-view", type=str2bool, nargs='?', const=True, default=PIP_VIEW)
      parser.add_argument("--freq-sound", type=str2bool, nargs='?', const=True, default=FREQ_SOUND)
      parser.add_argument("--apparent-horizons", type=str2bool, nargs='?', const=True, default=APPARENT_HORIZONS)
      parser.add_argument("--spin-vectors", type=str2bool, nargs='?', const=True, default=SPIN_VECTORS)
      parser.add_argument("--colormap", type=str2bool, nargs='?', const=True, default=COLORMAP)
      parser.add_argument("--detector-angle", type=str, nargs='+', help="Alpha, Delta, and Psi detector angles")
      parser.add_argument("--mass", type=float, nargs='?', default=MASS, help="Total Solar Mass")
      parser.add_argument("--movie-length", type=float, nargs='?', default=MOVIE_LENGTH, help="Desired length of stretched movie in
seconds")
      parser.add_argument("--three-dimensional-vis", type=str2bool, nargs='?', const=True, default=THREE_DIMENSIONAL_VIS,
help="Toggle three-dimensional visualization.")

      args, unknown = parser.parse_known_args()

      # Re-assign global-like configurations locally
      STATUS_MESSAGES = args.status_messages
      TRAJECTORY_LINES = args.trajectory_lines
      PIP_VIEW = args.pip_view
      FREQ_SOUND = args.freq_sound
      APPARENT_HORIZONS = args.apparent_horizons
      SPIN_VECTORS = args.spin_vectors
      COLORMAP = args.colormap
      USE_SYMLOG = args.use_symlog
      THREE_DIMENSIONAL_VIS = args.three_dimensional_vis
      MASS = args.mass
      MOVIE_LENGTH = args.movie_length

      if args.detector_angle:
          if len(args.detector_angle) not in [1, 3]:
              raise RuntimeError("At least three angles must be input when using --detector-angle. For + polarization, use
--detector +. For x polarization, use --detector x.")
          elif len(args.detector_angle) == 1:
              if args.detector_angle[0] == "+":
                  ALPHA, DELTA, PSI = (0, 0, 0)
              elif args.detector_angle[0] == "x":
                  ALPHA, DELTA, PSI = (0, 0, np.pi / 4)
          else:
              ALPHA, DELTA, PSI = [float(x) for x in args.detector_angle[:3]]

      if not args.use_default_args:
          if len(sys.argv) == 1:
              raise RuntimeError(usage_str)

          # Change directories and extraction radius based on inputs
          simulation_name = args.path_to_data_folder
          bh_dir = os.path.join("../data", simulation_name)

          # Set psi4_output_dir relative to bh_dir
          psi4_output_dir = os.path.join(bh_dir, "strain")
          movie_dir = os.path.join(bh_dir, "movies")  # Optimized path construction

          # Handle optional symlog argument
          use_symlog = args.use_symlog

          user_ext_rad = None
          invalid_unknowns = []
          for unk in unknown:
              if unk.startswith("--r"):
                  val_str = unk[3:]
                  if val_str == "inf":
                      user_ext_rad = float("inf")
                  else:
                      try:
                          user_ext_rad = float(val_str)
                      except ValueError:
                          invalid_unknowns.append(unk)
              else:
                  invalid_unknowns.append(unk)

          if invalid_unknowns:
              raise RuntimeError(usage_str)

      else: # Use default parameters defined at the top
          bh_dir = BH_DIR
          movie_dir = MOVIE_DIR
          psi4_output_dir = os.path.join(bh_dir, "converted_strain")
          user_ext_rad = None
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

      # Handle the case where the user entered an invalid simulation name. Also gives the directory list
      if not os.path.exists(bh_dir):
          raise FileNotFoundError(f"Data directory not found: {bh_dir}\n\n{dir_str}")

      # --- Movie File Path Handling ---
      bh_file_name = "puncture_posns_vels_regridxyzU.txt" # Name for the black hole position file
      bh_file_path = os.path.join(bh_dir, "puncture", bh_file_name) # Black hole position file path
      # Puncture file MUST EXIST for code to run
      if not os.path.isfile(bh_file_path):
           raise FileNotFoundError(f"Black hole position file not found: {bh_file_path}")

      bh_scaling_factor = 2.0 # Visual scaling of black holes

      movie_file_path, movie_path_name = create_movie_directory(movie_dir, ALPHA, DELTA, PSI)

      # --- Extraction Radius Calculations ---
      bh_file_list = os.listdir(bh_dir) # Extract the files in the black hole directory
      psi4_dir = os.path.join(bh_dir, "psi4")
      strain_dir = os.path.join(bh_dir, "strain")

      ext_rad, ell_min, ell_max, r_ext_in_strain, str_ext_rad = extract_extrad_and_modes(psi4_dir, strain_dir, user_ext_rad)
      ext_rad_num = ext_rad if ext_rad != float('inf') else 0 # A number to use to manipulate data, based on extraction radius

      # --- Simulation & Visualization Parameters ---
      n_rad_pts = 450       # number of points along the radius
      n_azi_pts = 180       # number of points along the azimuth
      n_colat_pts = 180 if THREE_DIMENSIONAL_VIS else 1 # number of points along the polar angle (colatitude)
      colat = np.pi / 2 - DELTA     # colatitude angle aligned with detector declination

      # Cosmetic & camera parameters
      wireframe = True
      save_rate = 10  # Saves every Nth simulation time step
      resolution = (1920, 1090) # Width, Height
      gw_color = (0.28, 0.46, 1.0) # Blueish
      bh_color = (0.1, 0.1, 0.1)   # Dark grey/black
      zoomout_distance = 350 # Max camera distance after zoomout
      initial_elevation_angle = 50 # Initial camera elevation
      final_elevation_angle = 34   # Final camera

      azi_angle = 45 # Default pi/4 azimuth camera angle
      FOV_angle = 30 # Default field of view angle
      pip_xfraction = 0.8 # Fraction of corner of screen taken up by picture in picture view

      time1 = time.time() # End of initial setup

      if STATUS_MESSAGES:
          print( # Horizontal line of asterisks
              f"{'*' * 70}\nConverting psi4 data to strain...")

      # See if strain files exist, and if so, attempt to load them
      if r_ext_in_strain:

          if STATUS_MESSAGES:
              print(f"Using existing strain data files in {strain_dir}")

          time_array, mode_data = extract_existing_strain_data(psi4_output_dir, ell_min, ell_max, str_ext_rad)

      # If there is no strain, use psi4_FFI_to_strain to convert psi 4 into strain
      else:
          # Convert psi4 to strain and load strain data
          try:
              # Pass the directory where the strain files are expected
              time_array, mode_data = psi4strain.psi4_ffi_to_strain(psi4_dir, psi4_output_dir, ell_max, ext_rad)
          except FileNotFoundError as e:
              raise FileNotFoundError(f"Error loading strain data: {e}. Ensure converted files exist in {psi4_output_dir} or check
psi4strain function.")
          except Exception as e:
              raise RuntimeError(f"An unexpected error occurred during psi4 data conversion and strain loading: {e}")

      # --- Time Array Scaling to Total System Mass ---
      time_array *= MASS * GC3 # Convert time array from M to seconds using total mass in solar masses
      n_sec = time_array[-1] - time_array[0] # Number of seconds in the simulation
      n_frames = int(MOVIE_LENGTH * 24) # Ensure there are enough frames to make 24 fps for the stretched movie
      frames_per_second = math.ceil(max(24, n_frames / n_sec)) # Calculate frames per second for the phys movie (at least 24)
      n_frames = int(frames_per_second * n_sec) # Recalculate just in case fps is now 24
      phys_times = np.linspace(time_array[0], time_array[-1], num=n_frames) # Create an array of times corresponding to each frame
      # Extract number of times and throw an error if there are no times
      n_times = len(time_array)
      if n_times == 0:
          raise ValueError(f"Loaded time array is empty. Cannot proceed.")

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
           raise ValueError(f"""Black hole data in {bh_file_path} has unexpected shape or too few rows. Ensure the file meets the
following criteria:
      - 15 rows of header lines
      - At least 5 columns of data formatted as follows:
          column 0: retarded time
          column 1: areal mass of black hole 1
          column 2: areal mass of black hole 2
          column 3: x position of black hole 1
          column 4: y position of black hole 1""")

      merge_idx_bh = np.argmax(np.diff(bh_data[:, 1])) + 1 # Find index where BH mass jumps as index of merge point
      merge_time = bh_data[merge_idx_bh, 0] * MASS * GC3 # Convert merge time from M to seconds using total mass in solar masses

      # Extract BH time array
      bh_time = bh_data[:, 0] * MASS * GC3 # Convert BH time array from M to seconds using total mass in solar masses

      # Mass ratio calculation using slice object up to merger
      pre_merge = slice(None, merge_idx_bh)

      # Calculating the average areal mass of each black hole before the merge point
      bh1_avg_mass = np.mean(bh_data[pre_merge, 1])
      bh2_avg_mass = np.mean(bh_data[pre_merge, 14])

      bh_total_mass = bh1_avg_mass + bh2_avg_mass
      bh1_rel_mass = bh1_avg_mass / bh_total_mass
      bh2_rel_mass = bh2_avg_mass / bh_total_mass

      mass_ratio = bh1_rel_mass / bh2_rel_mass

      # Extract BH coordinates (Check columns: 3=x, 4=y, 5=z)
      bh1_x0, bh1_y0, bh1_z0 = bh_data[:, 3], bh_data[:, 4], bh_data[:, 5]  # Assume motion is in xy-plane

      # Interpolate BH positions to the movie time array (which is the physical time array)
      # Time array optimization: Use the actual strain time array for interpolation basis
      merge_idx = find_idx(phys_times, merge_time)[0] # Find merge index in the output time array

      # Maintain original interpolation call
      bh_interpolator = interp1d(bh_time, np.vstack((bh1_x0, bh1_y0, bh1_z0)), fill_value="extrapolate")
      bh1_x, bh1_y, bh1_z = bh_interpolator(phys_times)

      # Vectorized coordinate calculation using array stacking
      pre_slice = slice(None, merge_idx)
      post_slice = slice(merge_idx, None)

      # Single array operation for all coordinates
      bh2_coords = np.concatenate([
          -1 * mass_ratio * np.column_stack((bh1_x[pre_slice], bh1_y[pre_slice], bh1_z[pre_slice])),
          np.column_stack((bh1_x[post_slice], bh1_y[post_slice], bh1_z[post_slice])) # bh2 initially moves opposite bh1, but
                                                                                     # changes to following bh1 after merge
      ])

      bh2_x, bh2_y, bh2_z = bh2_coords.T  # Transpose and unpack

      less_massive_x = bh1_x if bh1_rel_mass < bh2_rel_mass else bh2_x # Find x and y coordinates of less massive bh
      less_massive_y = bh1_y if bh1_rel_mass < bh2_rel_mass else bh2_y # These will be the most "sweeping"

      # Compute the radius of each black hole to be used in the animation
      bh1_scaled_radius = bh1_rel_mass * bh_scaling_factor
      bh2_scaled_radius = bh2_rel_mass * bh_scaling_factor
      bh_merged_radius = (bh1_scaled_radius + bh2_scaled_radius) / bh2_scaled_radius

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
                  horizon_times = np.unique(np.append(horizon_times, horizon_time)) # Try converting last part of filename to
timestep number
                  if time_string in horizon_names:
                      horizon_names[time_string].append(horizon_filepath) # If it works, append the file name to the horizon_names
                  else:
                      horizon_names[time_string] = [horizon_filepath]
              except (ValueError, IndexError): # If it's out of bounds or doesn't convert to a number, it's not a file we want
                  continue

          position_to_horizon = horizon_merge_time / (merge_time + ext_rad_num * GC3 * MASS) # ext_rad_num's effect was scaled on
the merge time

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
      magnitudes = np.sqrt(less_massive_x**2 + less_massive_y**2) # Get the distance between bh1 and the center (bh1 is less
massive)
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
      max_rads = 350 + int((available_memory) / (n_frames * n_azi_pts * float64size))
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
          rad_delta = max(300/350, resolution_dropoff_factor * shifted_erf(gen_rad) + 2 / 3) # Calculate radius step. But don't let
it get a weirdly high resolution towards the edges.
          gen_rad += rad_delta

      # Ensure the display radius point is also added to rad_vals, for consistent edges
      if display_radius > rad_vals[-1]:
          rad_vals.append(display_radius)

      # Cast the rad_vals as a numpy array and get the length
      radius_values = np.array(rad_vals)
      n_rad_pts = len(radius_values)

      # Initialize a grid with the calculated azimuth and radius points
      strain_array, grid, points = initialize_tvtk_grid(n_azi_pts, n_rad_pts, (n_colat_pts if THREE_DIMENSIONAL_VIS else 1))

      # theta and radius values for the mesh
      azimuth_values = np.linspace(0, 2 * np.pi, n_azi_pts, endpoint=False, dtype=np.float32) # Use float32
      colat_values = (np.linspace(0, np.pi, n_colat_pts, endpoint=False, dtype=np.float32)) if THREE_DIMENSIONAL_VIS else [colat] #
Use float32

      # Create meshgrid (ij indexing gives radius changing fastest)
      rv, az, cv = np.meshgrid(radius_values, azimuth_values, colat_values, indexing="ij")

      # Flatten angle arrays for use in SWSH functions
      flat_azi = az.ravel()
      flat_colat = cv.ravel()

      # Calculate Cartesian coordinates for the flat mesh
      x_values = rv * np.cos(az)
      y_values = rv * np.sin(az)
      z_values = rv * np.cos(cv)

      if STATUS_MESSAGES:
           print("Calculating spin-weighted spherical harmonics (fast)...")

      if not THREE_DIMENSIONAL_VIS:
          # Apply spin-weighted spherical harmonics using the optimized BLAS-backed routine
          strain_raw = swsh_summation_angles(
              colat_values,
              azimuth_values,
              mode_data,
              ell_min,
              ell_max,
              STATUS_MESSAGES)

          strain_azi_colat = compute_detector_strain(strain_raw, ALPHA, DELTA, PSI)

      # Broadcasts phys_times and radius_values together to create a 2D array (n_radii, n_frames) that shows the retarded
      # time at each radius, plus the extraction radius
      lerp_times = phys_times[np.newaxis, :] - (radius_values[:, np.newaxis] - ext_rad_num) * MASS * GC3

      time5=time.time() # End of grid setup

      if STATUS_MESSAGES:
          print( # Horizontal line of asterisks
              f"{'*' * 70}\nCalculating cosmetic data..."
          )

      max_separation = compute_max_bh_separation(bh_azis, magnitudes)
      # Find radius of the center hole in the mesh (based on uninterpolated max separation + BH size)
      # Hole radius = factor * (max_separation + scaled radius of larger BH)
      omitted_radius_length = max_separation + bh_scaling_factor * max(bh1_rel_mass, bh2_rel_mass) + 1
      # Ensure that simulations where black holes are really far apart don't generate massive
      # holes, so long as trajectories aren't being tracked
      if omitted_radius_length > 100 and not TRAJECTORY_LINES:
          omitted_radius_length = 2 * bh_merged_radius + 2
          if STATUS_MESSAGES:
              print("WARNING: Black holes are spaced too far apart to generate proportional hole in mesh. Default hole will be
used.")

      if not THREE_DIMENSIONAL_VIS:
          # Find point at which to taper off gravitational waves
          width = 0.5 * omitted_radius_length # Width of the transition region
          dropoff_radius = width + omitted_radius_length # Radius at which to start tapering beyond the hole

          # Find max amplitude scale factor allowable without impeding view of center hole (based on dropoff radius)
          # Apply amplitude scale factor calculation based on spin-weighted spherical harmonics max strain
          amplitude_scale_factor = get_amplitude_scale_factor(np.sign(strain_azi_colat) * np.log1p(np.abs(strain_azi_colat)) if
use_symlog else strain_azi_colat, azimuth_values, omitted_radius_length, zoomout_distance, fin_elevation_rads,
np.radians(azi_angle))
          # Dropoff factor (smooth transition to zero amplitude near omitted hole), apply amplitude scale factor
          dropoff_2D_flat = (0.5 + 0.5 * erf((radius_values - dropoff_radius)/width)).ravel() * amplitude_scale_factor

          # Report calculated values
          if STATUS_MESSAGES:
              print(f"Amplitude scale factor: {amplitude_scale_factor:.3f}")

      time6 = time.time() # End of cosmetic data calculations

      if SPIN_VECTORS:
          if STATUS_MESSAGES:
              print( # Horizontal line of asterisks
                  f"{'*' * 70}\nInterpolating dimensionful spin data..."
              )
          bh1_spinx0, bh1_spiny0, bh1_spinz0 = bh_data[:, 6], bh_data[:, 7], bh_data[:, 8]
          bh2_spinx0, bh2_spiny0, bh2_spinz0 = bh_data[:, 19], bh_data[:, 20], bh_data[:, 21]

          spin1_interpolator = interp1d(bh_time, np.vstack((bh1_spinx0, bh1_spiny0, bh1_spinz0)), fill_value="extrapolate")
          bh1_spinx, bh1_spiny, bh1_spinz = spin1_interpolator(phys_times)

          spin2_interpolator = interp1d(bh_time, np.vstack((bh2_spinx0, bh2_spiny0, bh2_spinz0)), fill_value="extrapolate")
          bh2_spinx, bh2_spiny, bh2_spinz = spin2_interpolator(phys_times)

          bh1_mags = np.sqrt(bh1_spinx**2 + bh1_spiny**2 + bh1_spinz**2)
          bh2_mags = np.sqrt(bh2_spinx**2 + bh2_spiny**2 + bh2_spinz**2)
          max_spin = np.max([bh1_mags, bh2_mags])

          spin_vector_max_length = omitted_radius_length / 2

          if spin_vector_max_length <= bh_merged_radius:
              spin_vector_max_length = bh_merged_radius * 1.5
              if STATUS_MESSAGES:
                  print("WARNING: Spin vectors would be too small. Default spin vector length will be used.")
          spin_scale_factor = spin_vector_max_length / max_spin

          times = time.time()

      if STATUS_MESSAGES:
          print(f"{'*' * 70}\nConstructing mesh points in 3D...")

      mmap_name = os.path.join(movie_file_path, "datamap")
      # Interpolate the strain to the appropriate points on the mesh grid

      if THREE_DIMENSIONAL_VIS:
          for l in range(ELL_MIN, ELL_MAX + 1):
              for m in range(-l, l + 1):
                  mode_idx = (l - ELL_MIN) * (2*l + 1) + (m + l) # Calculate the index for the mode data
                  mode_strain_azi_colat = swsh_summation_angles(colat_flat, azi_flat, mode_data, l, l) # Just at one l

                  datamap_name = os.path.join(movie_file_path, f"datamap_l{l}_m{m}")
>                 strain_to_mesh[mode_idx] = compute_strain_to_mesh(
                      phys_times.astype(np.float32),
                      radius_values,
                      lerp_times,
                      time_array.astype(np.float32),
                      dropoff_2D_flat,
                      use_symlog,
                      datamap_name,
                      STATUS_MESSAGES
                  )
      else:
>         strain_to_mesh = compute_strain_to_mesh(
              strain_azi_colat,
              phys_times.astype(np.float32),
              radius_values,
              colat_values,
              lerp_times,
              time_array.astype(np.float32),
              dropoff_2D_flat,
              use_symlog,
              mmap_name,
              STATUS_MESSAGES
          )
          print(strain_to_mesh)

      time7=time.time()

      if COLORMAP:
          if STATUS_MESSAGES:
              print(f"{'*' * 70}\nComputing colormap values...")

          scalar_azi = compute_detector_strain(strain_raw, ALPHA, DELTA, PSI + np.pi / 4)

          scalar_mmap_name = os.path.join(movie_file_path, "scalarmap")
          # Interpolate the strain to the appropriate points on the mesh grid
>         strain_to_scalars = compute_strain_to_mesh(
              scalar_azi,
              phys_times.astype(np.float32),
              radius_values,
              colat_values,
              lerp_times,
              time_array.astype(np.float32),
              dropoff_2D_flat,
              use_symlog,
              scalar_mmap_name,
              STATUS_MESSAGES
          )

          time8=time.time()

      if STATUS_MESSAGES:
          print(f"{'*' * 70}\nInitializing animation...")

      # --- Precompute values for animation loop ---

      # Precompute shifted time array, removing extraction radius offset
      shifted_time_array = phys_times + ext_rad_num * GC3 * MASS # Now it starts at 0

      # Precompute geometric data & masks with vectorization
      # Mask for points outside the central hole
      valid_mask = (rv > omitted_radius_length).ravel()
      # Flattened xy coordinates of the mesh
      x_flat, y_flat, z_flat = x_values.ravel(), y_values.ravel(), z_values.ravel()

      # Initialize VTK data structures once
      points = tvtk.Points() # This holds the 3D coordinates
      vtk_array = tvtk.FloatArray() # VTK array to store coordinate data
      vtk_array.number_of_components = 3
      vtk_array.number_of_tuples = len(x_flat)

      if THREE_DIMENSIONAL_VIS:
          mode_names = [f"l{l}_m{m}" for l in range(ELL_MIN, ELL_MAX + 1) for m in range(-l, l + 1)]
          for mode in mode_names:
              mode_idx = (l - ELL_MIN) * (2*l + 1) + (m + l) # Calculate the index for the mode data
              strain_mode = strain_to_mesh[mode_idx]
              mode_vtk_array = tvtk.FloatArray()
              mode_vtk_array.name = mode
              mode_vtk_array.from_array(strain_mode[:, :, :, 0].flatten().astype(np.float32))
              grid.point_data.add_array(mode_vtk_array)
      else:
          points.data = vtk_array # Set the point data to the VTK array we created
          if COLORMAP:
              grid.point_data.scalars = strain_to_scalars[..., 0].ravel()

      # Get a NumPy view for efficient modification
      np_points = vtk_array.to_array().reshape(-1, 3)

      # Set XY coordinates (these don't change)
      np_points[:, 0] = x_flat
      np_points[:, 1] = y_flat

      if THREE_DIMENSIONAL_VIS:
          # In 3D, use actual Z coordinates
          np_points[:, 2] = z_flat
      else:
          # In 2D, punch the hole
          starting_points = strain_to_mesh[:, :, 0, 0].ravel()
          np_points[valid_mask, 2] = starting_points[valid_mask] # Set Z to the strain value for valid points
          np_points[~valid_mask, 2] = np.nan # Pre-fill the hole with NaNs so it is not rendered

      grid.points = points # Attach points to the grid

      # Precompute camera parameters for each output frame
      time_indices = np.arange(n_frames) # Indices 0 to n_frames-1
      # Smoothly decrease elevation angle over time until it hits the target (this means the camera rises)
      elevations = np.maximum(initial_elevation_angle - time_indices * 0.016, final_elevation_angle)

      # Zoom out a quarter of the way through the data
      zoomout_time = phys_times[-1] / 4 # Start zooming out at a quarter of the largest POSITIVE time
      zoomout_idx = find_idx(phys_times, zoomout_time)[0]

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
      percentage_thresholds = np.round(np.linspace(0, n_frames, 101)).astype(int)

      # Precompute frame filenames using f-strings and os.path.join
      frame_filenames = [os.path.join(movie_file_path, f"z_frame_{i:05d}.png") for i in range(n_frames)]

      # Configure engine and rendering upfront
      engine = Engine()
      engine.start()
      fig = mlab.figure(engine=engine, size=resolution) # Default background color
      fig.scene.interactor.disable() # Make it so camera view can't be changed accidentally

      # Initialize visualization objects once
      if THREE_DIMENSIONAL_VIS:
          isosurfaces = create_isosurfaces(engine, grid, mode_names)
      else:
          # 2D Surface setup
          gw = create_gw(engine, grid, gw_color, display_radius, wireframe, COLORMAP)

      # Black Holes
      if APPARENT_HORIZONS:
          # Create source and mesh for black hole horizons
          bh1_source, bh1 = plot_initial_mesh(engine, horizon_names["0.0"][0], bh_scaling_factor)
          bh2_source, bh2 = plot_initial_mesh(engine, horizon_names["0.0"][1], bh_scaling_factor)
      else:
          bh1 = create_sphere(engine, bh1_scaled_radius, bh_color)
          bh2 = create_sphere(engine, bh2_scaled_radius, bh_color)

      # Create trajectory tracking lines if needed
      if TRAJECTORY_LINES:
          bh1_trajectory = mlab.plot3d(bh1_x[0], bh1_y[0], bh1_z[0], figure=fig, tube_radius=0.1)
          bh2_trajectory = mlab.plot3d(bh2_x[0], bh2_y[0], bh2_z[0], figure=fig, tube_radius=0.1)
          bh1_trajectory.actor.property.opacity = 0.5 # Make them semi-transparent so they don't cover up black holes
          bh2_trajectory.actor.property.opacity = 0.5

      if SPIN_VECTORS:
          bh1_spin = mlab.quiver3d(
              bh1_x[0],
              bh1_y[0],
              bh1_z[0],
              bh1_spinx[0],
              bh1_spiny[0],
              bh1_spinz[0],
              mode="arrow",
              scale_factor = spin_scale_factor,
              color=(1, 0, 0)
          )

          bh2_spin = mlab.quiver3d(
              bh2_x[0],
              bh2_y[0],
              bh2_z[0],
              bh2_spinx[0],
              bh2_spiny[0],
              bh2_spinz[0],
              mode="arrow",
              scale_factor = spin_scale_factor,
              color=(1, 0, 0)
          )

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
          if SPIN_VECTORS:
              print(f"Spin Vector Calculations: {times - time6:.3f}")
          print(f"  Mesh Construction: {time7 - (times if SPIN_VECTORS else time6):.3f}")
          if COLORMAP:
              print(f"Colormap Calculations: {time8 - time7:.3f}")
          print(f"  Animation Setup: {start_time - (time8 if COLORMAP else time7):.3f}")
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

          with imageio.get_writer(movie_path_name, fps=24, codec="libx264", quality=8) as writer:
              for idx, time_idx in enumerate(shifted_time_array):
                  # --- Status Update & ETA ---
                  if idx == 10: # Estimate after 10 frames
                      end_time = time.time()
                      eta = (end_time - start_time) * n_frames / 10
                      print(
                          f"""\nCreating {n_frames} frames and saving them to:
  {movie_path_name}\nEstimated time: {dhms_time(eta)}"""  # <-- MODIFIED: Show silent path
                      )

                  # Update progress percent
                  if STATUS_MESSAGES and idx !=0 and current_percent < len(percentage_thresholds) and idx >
percentage_thresholds[current_percent]:
                      eta = ((time.time() - start_time) / idx) * (n_frames - idx)
                      print(f"\r{int(idx * 100 / n_frames)}% done, {dhms_time(eta)} remaining", end="", flush=True)
                      current_percent +=1

                  # --- Update Scene Objects ---
                  if APPARENT_HORIZONS:
                      horizon_idx = find_idx(horizon_times, position_to_horizon * shifted_time_array)[0]
                      update_mesh_data(bh1_source, horizon_names[str(horizon_times[horizon_idx])][0], bh_scaling_factor)
                      try:
                          update_mesh_data(bh2_source, horizon_names[str(horizon_times[horizon_idx])][1], bh_scaling_factor)
                      except IndexError:
                          try:
                              bh2_mesh.visible = False
                          except ValueError:
                              pass
                  else:
                      bh1.actor.actor.position = bh1_x[idx], bh1_y[idx], bh1_z[idx] # Update first bh position
                      bh2.actor.actor.position = bh2_x[idx], bh2_y[idx], bh2_z[idx] # Update second bh position
                      # Rescale bh2 if black holes have merged to represent combined object (at the specific frame index)
                      if idx == merge_idx:
                          # For a sphere, equally scale in all directions
                          bh2.actor.actor.scale = bh_merged_radius, bh_merged_radius, bh_merged_radius

                  if TRAJECTORY_LINES and idx > 0:
                      bh1_trajectory.mlab_source.reset(x=bh1_x[:idx], y=bh1_y[:idx], z=bh1_z[:idx])
                      bh2_trajectory.mlab_source.reset(x=bh2_x[:idx], y=bh2_y[:idx], z=bh2_z[:idx])

                  if SPIN_VECTORS:
                      bh1_spin.mlab_source.reset(
                          x=bh1_x[idx],
                          y=bh1_y[idx],
                          z=bh1_z[idx],
                          u=bh1_spinx[idx],
                          v=bh1_spiny[idx],
                          w=bh1_spinz[idx]
                      )

                      if bh2_spin.visible:
                          bh2_spin.mlab_source.reset(
                              x=bh2_x[idx],
                              y=bh2_y[idx],
                              z=bh2_z[idx],
                              u=bh2_spinx[idx],
                              v=bh2_spiny[idx],
                              w=bh2_spinz[idx]
                          )

                          if idx >= merge_rescale_idx:
                              bh2_spin.visible = False # Remove the second spin vector after merging, so only the combined one
remains

                  # -- Update GW Surface / Isosurfaces --
                  if not THREE_DIMENSIONAL_VIS:
                      # 2D MODE: Warp the Z-coordinates of the mesh
                      strain_slice = strain_to_mesh[..., 0, idx].ravel()
                      np_points[valid_mask, 2] = strain_slice[valid_mask]

                      strain_array.from_array(strain_slice[valid_mask])

                      if COLORMAP:
                          scalar_slice = strain_to_scalars[..., 0, idx].ravel()
                          current_scalars = grid.point_data.scalars.to_array()
                          current_scalars[valid_mask] = scalar_slice[valid_mask]
                          grid.point_data.scalars.modified()

                      gw.data_changed = True # Notify Mayavi that the data has changed so it can update the visualization

                  else:
                      # 3D MODE: Mesh geometry is static. Update the scalar arrays only.
                      for mode in mode_names:
                          # Parse l and m from the mode string (e.g., "l2_m2")
                          parts = mode.split('_')
                          l_val = int(parts[0][1:])
                          m_val = int(parts[1][1:])

                          # Calculate the index using your existing logic
                          mode_idx = (l_val - ELL_MIN) * (2*l_val + 1) + (m_val + l_val)

                          # Extract 1D data slice for this specific mode and time step
                          data_slice = strain_to_mesh[mode_idx][..., idx].ravel()

                          # Fetch the existing VTK array by its name and update the values in place
                          mode_vtk_array = grid.point_data.get_array(mode)
                          mode_vtk_array.from_array(data_slice.astype(np.float32))

                      # CRITICAL: Notify the Mayavi source object that the underlying VTK data changed.
                      # 'base_source' must be the VTKDataSource object you created in the initialization block.
                      base_source.data_changed = True

                  # --- Update Camera ---
                  mlab.view(
                      azimuth=azi_angle,
                      elevation=elevations[idx],
                      distance=distances[idx],
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
          # --- Generate Audio File ---
          actual_movie_times = np.linspace(0, MOVIE_LENGTH, 24 * MOVIE_LENGTH)
          audio_path, movie_with_audio, = create_sound_file(
              mode_data,
              time_array,
              ell_min,
              ell_max,
              actual_movie_times,
              movie_path_name,
              fps=24,
              target_rate=48000
          )

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
               print(f"\\nAn error occurred during merging: {e}")
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
      """# run doctests first
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
              fDoctest in {psi4strain} failed:
  {p4s_results.failed} of {p4s_results.attempted} test(s) passed
          )
          sys.exit(1)
      else:
          # Use f-string for success message
          print(
              fDoctest in {psi4strain} passed:
  All {p4s_results.attempted} test(s) passed
          )

      if results.failed > 0:
          # Use f-string for error message
          print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
          exit(1)
      else:
          # Use f-string for success message
          print(f"Doctest passed: All {results.attempted} test(s) passed")"""

      # Run main function only if all doctests passed
      #try:
      main()
      """except (RuntimeError, FileNotFoundError, ValueError, IndexError) as e:
          # Catch expected errors from main() and print cleanly
          print(f"\nExecution failed: {e}", file=sys.stderr)
          sys.exit(1)
      except Exception as e:
          # Catch unexpected errors
          print(f"\\nAn unexpected error occurred during main execution: {e}", file=sys.stderr)
          # Optionally print traceback for debugging unexpected errors
          traceback.print_exc()
          sys.exit(1)"""
