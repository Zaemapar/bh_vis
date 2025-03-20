import h5py
import numpy as np
import os
import sys

def convert_h5_to_ascii(h5_file, output_dir, R_ext):
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
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the HDF5 file
    with h5py.File(h5_file, "r") as f:
        # Check if the group containing Y_lm datasets exists
        if "/R0100.dir" not in f:
            raise ValueError("The HDF5 file does not contain the expected group '/R0100.dir'.")

        # Get the list of all Y_lm datasets
        ylm_datasets = [name for name in f["/R0100.dir"] if name.startswith("Y_l")]

        # Organize datasets by ell value
        ell_data = {}
        for dataset_name in ylm_datasets:
            # Extract ell and em from the dataset name
            parts = dataset_name.split("_")
            ell = int(parts[1][1:])  # Extract ell value (e.g., "l2" -> 2)
            em = int(parts[2][1:].split(".")[0])  # Extract em value (e.g., "m-2" -> -2)

            # Load the dataset
            dataset = f[f"/R0100.dir/{dataset_name}"]
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

def extract_black_hole_positions(pos_file, output_dir):
    """
    Extract the center-of-mass positions of black holes from an HDF5 file
    and save them to a text file named 'black_hole_positions.txt'.
    Format: 27 columns with time in column 0 and various parameters for two black holes.
    """
    with h5py.File(pos_file, "r") as f:
        # Read datasets for each black hole group
        datasets = {}
        prefixes = ['AhA.dir', 'AhB.dir', 'AhC.dir']
        for prefix in prefixes:
            group = {}
            group['positions'] = f[f"{prefix}/CoordCenterInertial.dat"][:]
            group['arealmass'] = f[f"{prefix}/ArealMass.dat"][:]
            group['christodouloumass'] = f[f"{prefix}/ChristodoulouMass.dat"][:]
            dis_data = f[f"{prefix}/DimensionfulInertialSpin.dat"][:]
            group['dimensionfulinertialspin'] = dis_data
            group['dimensionfulinertialspinmag'] = dis_data
            group['chiinertial'] = f[f"{prefix}/chiInertial.dat"][:]
            group['chimaginertial'] = f[f"{prefix}/chiMagInertial.dat"][:]
            datasets[prefix] = group

        # Extract and concatenate time
        ahA_time = datasets['AhA.dir']['positions'][:, 0]
        merged_time = datasets['AhC.dir']['positions'][:, 0]
        time = np.concatenate((ahA_time, merged_time))
        n_rows = len(ahA_time) + len(merged_time)
        output_dat = np.zeros((n_rows, 27))
        output_dat[:, 0] = time

        # Define column mappings for bh1 (AhA + AhC) and bh2 (AhB + AhC)
        columns_bh1 = [
            (1, 'arealmass', 1),
            (2, 'christodouloumass', 1),
            (3, 'positions', 1),
            (4, 'positions', 2),
            (5, 'positions', 3),
            (6, 'dimensionfulinertialspin', 1),
            (7, 'dimensionfulinertialspin', 2),
            (8, 'dimensionfulinertialspin', 3),
            (9, 'dimensionfulinertialspinmag', 1),
            (10, 'chiinertial', 1),
            (11, 'chiinertial', 2),
            (12, 'chiinertial', 3),
            (13, 'chimaginertial', 1),
        ]
        columns_bh2 = [
            (14, 'arealmass', 1),
            (15, 'christodouloumass', 1),
            (16, 'positions', 1),
            (17, 'positions', 2),
            (18, 'positions', 3),
            (19, 'dimensionfulinertialspin', 1),
            (20, 'dimensionfulinertialspin', 2),
            (21, 'dimensionfulinertialspin', 3),
            (22, 'dimensionfulinertialspinmag', 1),
            (23, 'chiinertial', 1),
            (24, 'chiinertial', 2),
            (25, 'chiinertial', 3),
            (26, 'chimaginertial', 1),
        ]

        # Populate bh1 columns (AhA and AhC)
        for col, param, idx in columns_bh1:
            data = np.concatenate([
                datasets['AhA.dir'][param][:, idx],
                datasets['AhC.dir'][param][:, idx]
            ])
            output_dat[:, col] = data

        # Populate bh2 columns (AhB and AhC)
        for col, param, idx in columns_bh2:
            data = np.concatenate([
                datasets['AhB.dir'][param][:, idx],
                datasets['AhC.dir'][param][:, idx]
            ])
            output_dat[:, col] = data

        # Generate header
        header_lines = [
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
            "# column 13 (bh_1), 26 (bh_2): xim = [chi_mag_inertial]",
            f"# Merger time:",
        ]
        header_rows = []
        for line in header_lines:
            header_row = [line] + [''] * (output_dat.shape[1] - 1)
            header_rows.append(header_row)
        header = np.array(header_rows)

        merge_row = [merged_time[0]]
        for row in range(output_dat.shape[1] - 1):
            merge_row.append(0)

        output_data = np.vstack((header, merge_row, output_dat))

        # Save the output
        output_file = os.path.join(output_dir, "puncture_posns_vels_regridxyzU.txt")
        np.savetxt(output_file, output_data, fmt="%s", delimiter=" ")
        print(f"Saved black hole positions to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 convert_h5_to_ascii.py <psi4_h5_file> <horizons_h5_file> <R_ext> <output_dir>")
        sys.exit(1)

    psi4_file = sys.argv[1] # psi 4 data file
    horizon_file = sys.argv[2] # bh horizon data file
    R_ext = float(sys.argv[3])  # Extraction radius
    output_dir = sys.argv[4]

    # Convert the HDF5 file to ASCII
    convert_h5_to_ascii(psi4_file, output_dir, R_ext)

    # Extract black hole positions and save to ASCII
    extract_black_hole_positions(horizon_file, output_dir)

    print("Conversion and extraction complete.")
