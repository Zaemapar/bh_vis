import h5py
import numpy as np
import os
import sys
import sxs

def convert_strain_to_ascii(strain, output_dir, R_ext):
    """
    Convert an HDF5 file containing spherical harmonic decompositions (Y_lm)
    into ASCII files, with one file per ell value.

    The output format is specifically:
    - Column 1: t-R_ext (retarded time), as a floating-point number
      formatted to .15e precision.
    - Subsequent columns (one for each Y_lm mode): Y_lm * R_ext.
      Complex numbers are formatted as a string like 'real_part<sign>imag_partj'
      (e.g., '1.234e+00-4.567e-01j'), without parentheses, with .15e precision
      for both real and imaginary parts.
    
    File names are formatted as Rh_l{ell}-r{R_ext:06.1f}.txt.
    (Note: The original function's docstring mentioned Rpsi4_l{ell}-r{R_ext}.txt,
     but its implementation used Rh_... We follow the implementation's Rh_ naming).
    """
    print( # Horizontal line of asterisks
            f"""{'*' * 70}
Converting strain to ASCII documents..."""
         )

    # Create the strain directory if it doesn't exist
    strain_dir = os.path.join(output_dir, 'strain')
    os.makedirs(strain_dir, exist_ok=True)

    ell_min = strain.ell_min
    ell_max = strain.ell_max

    # Calculate retarded time data for the first column
    # Assuming strain.t is a 1D NumPy array of time values.
    retarded_time_data = strain.t - R_ext
    iterations = ell_max**2 + 2 * ell_max - ell_min**2 + 1
    count = 0

    # Iterate over each ell value
    for ell in range(ell_min, ell_max + 1):
        # Format the output file name
        output_file_name = f"Rh_l{ell}-r{R_ext:06.1f}.txt"
        output_path = os.path.join(strain_dir, output_file_name)

        # List to hold column data. First column is raw floats, subsequent are string arrays.
        data_columns_list = [retarded_time_data]

        # List of format specifiers for np.savetxt, one for each column.
        format_list = ['%.15e'] # Format for the first (time) column

        # Initialize header lines for the current file
        header_lines = [f"# column 0: t-R_ext [retarded_time]"]
        current_column_index = 1  # Start column numbering for Y_lm data from 2

        # Iterate over each em mode for the current ell
        for em in range(-ell, ell + 1):
            idx = strain.index(ell, em)
            # Access the spherical harmonic mode data Y_lm(t)
            # Assumed to be a 1D NumPy array of complex numbers.
            ylm_mode_data_at_times = strain[:, idx]
            ylm_array = np.array(ylm_mode_data_at_times)

            # Custom formatting for complex numbers into a 1D array of strings
            # Example: "1.234567890123456e+00-9.876543210987654e-01j"
            # The :+ in {val.imag:+.15e} ensures the sign (+ or -) is always present for the imaginary part.
            list_of_complex_strings_for_this_mode = []

            for time_strain in ylm_array:
                real = time_strain.real
                imag = time_strain.imag

                # 2. Create the formatted string
                formatted_complex_str = f"{real:.15e}{imag:+.15e}j"

                # 3. Append the string to your Python list
                list_of_complex_strings_for_this_mode.append(formatted_complex_str)

            # 4. After the loop has processed all 'time_strain' for the current mode,
            #    convert the list of strings into a NumPy array of objects.
            complex_str_array_for_mode = np.array(list_of_complex_strings_for_this_mode, dtype=object)

            data_columns_list.append(complex_str_array_for_mode)
            format_list.append('%s') # Use '%s' for columns that are already strings

            # Add header for the current Y_lm mode column
            header_lines.append(f"# column {current_column_index}: rh_{{l={ell},m={em}}}/M")
            current_column_index += 1

            count += 1
            progress = (count / iterations) * 100
            print(f"\rProgress: {progress:.1f}% completed. Iterated through (ell, em) = ({ell}, {em})", end="", flush=True)

        # Stack all collected 1D data columns horizontally.
        # The resulting array will have dtype=object because it mixes float arrays and object arrays (of strings).
        output_data_array = np.column_stack(data_columns_list)

        # Join header lines into a single string, with each line separated by a newline
        header_string = "\n".join(header_lines)

        # Save the 2D array to a text file using the specified list of formats.
        # np.savetxt will use '%.15e' for the first column and '%s' for the others.
        # Default delimiter is a space.
        np.savetxt(output_path, output_data_array, header=header_string, comments="", fmt=format_list)

    print(f"\nStrain conversion complete. Saved {ell_max - ell_min + 1} files to {strain_dir}")

def convert_psi4_modes_to_ascii(psi4_path, output_dir, R_ext):
    """
    Convert an HDF5 file containing spherical harmonic decompositions (Y_lm)
    into ASCII files, with one file per ell value containing all em modes.
    The output format is:
    # column 1: t-R_ext = [retarded time]
    # column 2: Re(Y_lm) * R_ext
    # column 3: Im(Y_lm) * R_ext
    # ...
    File names are formatted as Rpsi4_l{ell}-r{R_ext}.txt.
    """

    # Create the strain directory if it doesn't exist
    strain_dir = os.path.join(output_dir, 'strain')
    os.makedirs(psi4dir, exist_ok=True)

    # Open the HDF5 file
    with h5py.File(h5_file, "r") as f:
        # Check if the group containing Y_lm datasets exists
        group = "/rMPsi4_Asymptotic_GeometricUnits_CoM_Mem"
        if group not in f:
            raise ValueError(f"The HDF5 file does not contain the expected group {group}.")

        # Get the list of all Y_lm datasets
        asymptotic_datasets = [name for name in f[group] if name.startswith("Extrapolated_N")]

        # Get a list of numbers from each name
        highest_res = -1
        index = 0
        max_index = index
        for name in asymptotic_datasets:
            number = name.replace("Extrapolated_N", "").replace(".dir", "")
            try:
                res = int(number)
            except ValueError as e:
                index = index + 1
                continue
            if res > highest_res:
               highest_res = res
               max_index = index
            index = index + 1

        if highest_res < 0:
            raise RuntimeError(f"No extrapolated datasets found in group {group}")
        data_highest_res = asymptotic_datasets[max_index]
        highest_res_dir = os.path.join(group, data_highest_res)
        ylm_datasets = [name for name in f[highest_res_dir]]

        dataset = f[os.path.join(highest_res_dir, 'data')]
        void = dataset[()].tobytes()
        parsed_data = np.frombuffer(void, dtype=np.complex64)
        print(parsed_data)
        print(void.dtype)
        # Organize datasets by ell value
        ell_data = {}
        for dataset_name in ylm_datasets:
            # Extract ell and em from the dataset name
            parts = dataset_name.split("_")
            ell = int(parts[1][1:])  # Extract ell value (e.g., "l2" -> 2)
            em = int(parts[2][1:].split(".")[0])  # Extract em value (e.g., "m-2" -> -2)

            # Load the dataset
            dataset = f[os.path.join(highest_res_dir, dataset_name)]
            time = dataset[:, 0]  # Assuming the first column is time
            real_part = dataset[:, 1]  # Real part
            imag_part = dataset[:, 2]  # Imaginary part

            # Initialize the dictionary entry for this ell if it doesn't exist
            if ell not in ell_data:
                ell_data[ell] = {"time": time, "modes": []}

            # Append the em mode data
            ell_data[ell]["modes"].append((em, real_part, imag_part))

        # Save data for each ell to a separate ASCII file
        for ell, data in ell_data.items():

            # Prepare the data columns and header
            time = data["time"]
            retarded_time = time - R_ext  # Compute retarded time
            data_columns = [retarded_time]
            column_labels = ["# column 1: t-R_ext = [retarded time]"]

            # Sort modes by em value
            data["modes"].sort(key=lambda x: x[0])
            # Add real and imaginary parts for each em mode
            col = 2  # Start column numbering from 2
            for em, real_part, imag_part in data["modes"]:
                data_columns.append(real_part * R_ext)  # Scale by R_ext
                data_columns.append(imag_part * R_ext)  # Scale by R_ext
                column_labels.append(f"# column {col}: Re(Y_l{ell}_m{em}) * R_ext")
                column_labels.append(f"# column {col + 1}: Im(Y_l{ell}_m{em}) * R_ext")
                col += 2

            # Combine all columns into a single array
            data_array = np.column_stack(data_columns)

            # Format the output file name
            output_file = os.path.join(output_dir, f"Rpsi4_l{ell}-r{R_ext:06.1f}.txt")
            header = "\n".join(column_labels)
            np.savetxt(output_file, data_array, header=header, comments="", fmt="%.15e")

            print(f"Saved data for ell={ell} to {output_file}")

def extract_black_hole_positions(horizons, output_dir, R_ext):
    """
    Extract the center-of-mass positions of black holes from an HDF5 file
    and save them to a text file named 'black_hole_positions.txt'.
    Format: 27 columns with time in column 0 and various parameters for two black holes.
    """

    print( # Horizontal line of asterisks
            f"""\n{'*' * 70}
Extracting black hole horizon data..."""
         )

    time = np.concatenate((horizons.A.time, horizons.C.time), axis=None) - R_ext
    amA = np.concatenate((horizons.A.areal_mass, horizons.C.areal_mass), axis=None)
    cmA = np.concatenate((horizons.A.christodoulou_mass, horizons.C.christodoulou_mass), axis=None)
    xA = np.concatenate((horizons.A.coord_center_inertial[:, 0], horizons.C.coord_center_inertial[:, 0]), axis=None)
    yA = np.concatenate((horizons.A.coord_center_inertial[:, 1], horizons.C.coord_center_inertial[:, 1]), axis=None)
    zA = np.concatenate((horizons.A.coord_center_inertial[:, 2], horizons.C.coord_center_inertial[:, 2]), axis=None)
    disxA = np.concatenate((horizons.A.dimensionful_inertial_spin[:, 0], horizons.C.dimensionful_inertial_spin[:, 0]), axis=None)
    disyA = np.concatenate((horizons.A.dimensionful_inertial_spin[:, 1], horizons.C.dimensionful_inertial_spin[:, 1]), axis=None)
    diszA = np.concatenate((horizons.A.dimensionful_inertial_spin[:, 2], horizons.C.dimensionful_inertial_spin[:, 2]), axis=None)
    dismA = np.concatenate((horizons.A.dimensionful_inertial_spin_mag, horizons.C.dimensionful_inertial_spin_mag), axis=None)
    xixA = np.concatenate((horizons.A.chi_inertial[:, 0], horizons.C.chi_inertial[:, 0]), axis=None)
    xiyA = np.concatenate((horizons.A.chi_inertial[:, 1], horizons.C.chi_inertial[:, 1]), axis=None)
    xizA = np.concatenate((horizons.A.chi_inertial[:, 2], horizons.C.chi_inertial[:, 2]), axis=None)
    ximA = np.concatenate((horizons.A.chi_mag_inertial, horizons.C.chi_mag_inertial), axis=None)
    amB = np.concatenate((horizons.B.areal_mass, horizons.C.areal_mass), axis=None)
    cmB = np.concatenate((horizons.B.christodoulou_mass, horizons.C.christodoulou_mass), axis=None)
    xB = np.concatenate((horizons.B.coord_center_inertial[:, 0], horizons.C.coord_center_inertial[:, 0]), axis=None)
    yB = np.concatenate((horizons.B.coord_center_inertial[:, 1], horizons.C.coord_center_inertial[:, 1]), axis=None)
    zB = np.concatenate((horizons.B.coord_center_inertial[:, 2], horizons.C.coord_center_inertial[:, 2]), axis=None)
    disxB = np.concatenate((horizons.B.dimensionful_inertial_spin[:, 0], horizons.C.dimensionful_inertial_spin[:, 0]), axis=None)
    disyB = np.concatenate((horizons.B.dimensionful_inertial_spin[:, 1], horizons.C.dimensionful_inertial_spin[:, 1]), axis=None)
    diszB = np.concatenate((horizons.B.dimensionful_inertial_spin[:, 2], horizons.C.dimensionful_inertial_spin[:, 2]), axis=None)
    dismB = np.concatenate((horizons.B.dimensionful_inertial_spin_mag, horizons.C.dimensionful_inertial_spin_mag), axis=None)
    xixB = np.concatenate((horizons.B.chi_inertial[:, 0], horizons.C.chi_inertial[:, 0]), axis=None)
    xiyB = np.concatenate((horizons.B.chi_inertial[:, 1], horizons.C.chi_inertial[:, 1]), axis=None)
    xizB = np.concatenate((horizons.B.chi_inertial[:, 2], horizons.C.chi_inertial[:, 2]), axis=None)
    ximB = np.concatenate((horizons.B.chi_mag_inertial, horizons.C.chi_mag_inertial), axis=None)

    data = np.vstack((time, amA, cmA, xA, yA, zA, disxA, disyA, diszA, dismA, xixA, xiyA, xizA, ximA,
                      amB, cmB, xB, yB, zB, disxB, disyB, diszB, dismB, xixB, xiyB, xizB, ximB)).T

    # Generate header
    header = [
        "# column 0: time = [time]",
        "# column 1 (bh_1), 14 (bh_2): am = [areal_mass]",
        "# column 2 (bh_1), 15 (bh_2): cm = [christodoulou_mass]",
        "# column 3 (bh_1), 16 (bh_2): x = [positions_x]",
        "# column 4 (bh_1), 17 (bh_2): y = [positions_y]",
        "# column 5 (bh_1), 18 (bh_2): z = [positions_z]",
        "# column 6 (bh_1), 19 (bh_2): disx = [dimensionful_inertial_spin_x]",
        "# column 7 (bh_1), 20 (bh_2): disy = [dimensionful_inertial_spin_y]",
        "# column 8 (bh_1), 21 (bh_2): disz = [dimensionful_inertial_spin_z]",
        "# column 9 (bh_1), 22 (bh_2): dism = [dimensionful_inertial_spin_mag]",
        "# column 10 (bh_1), 23 (bh_2): xix = [chi_inertial_x]",
        "# column 11 (bh_1), 24 (bh_2): xiy = [chi_inertial_y]",
        "# column 12 (bh_1), 25 (bh_2): xiz = [chi_inertial_z]",
        "# column 13 (bh_1), 26 (bh_2): xim = [chi_mag_inertial]"]

    # Save the output
    output_file = os.path.join(output_dir, "puncture_posns_vels_regridxyzU.txt")
    np.savetxt(output_file, data, header="\n".join(header), comments="", fmt="%.15e")
    print(f"Saved black hole positions to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 h5_to_ascii.py <simulation_name> <extraction_radius> <optional: version>")
        sys.exit(1)

    simulation_name = sys.argv[1]
    R_ext = float(sys.argv[2])
    version = sys.argv[3]

    try:
        sim_path = os.path.join(simulation_name + "v2.0")
        sim = sxs.load(os.path.join(sim_path, version), extrapolation="N4")
        horizons = sim.horizons
        strain = sim.h
    except RuntimeError:
        print("SXS simulations are named in the format SXS:<BBH or BHNS or NSNS>:####")

    # Create the output directory if it doesn't exist
    os.makedirs(simulation_name, exist_ok=True)

    # Convert the HDF5 file to ASCII
    convert_strain_to_ascii(strain, simulation_name, R_ext)

    # Extract black hole positions and save to ASCII
    extract_black_hole_positions(horizons, simulation_name, R_ext)

    print("\nConversion and extraction complete.")
