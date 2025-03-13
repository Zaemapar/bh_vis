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
    Format: 8 columns, with time in column 1, AhA's x in column 6, and y in column 8 (others zero).
    """
    # Open the HDF5 file containing black hole positions
    with h5py.File(pos_file, "r") as f:
        # Extract AhA's positions (shape: [n_timesteps, 4])
        AhA_positions = f["AhA.dir/CoordCenterInertial.dat"][:]
        AhA_arealmass = f["AhA.dir/ArealMass.dat"][:]
        AhA_christodouloumass = f["AhA.dir/ChristodoulouMass.dat"][:]
        AhA_dimensionfulinertialspin = f["AhA.dir/DimensionfulInertialSpin.dat"][:]
        AhA_dimensionfulinertialspinmag = f["AhA.dir/DimensionfulInertialSpin.dat"][:]
        AhA_chiinertial = f["AhA.dir/chiInertial.dat"][:]
        AhA_chimaginertial = f["AhA.dir/chiMagInertial.dat"][:]
        AhC_positions = f["AhC.dir/CoordCenterInertial.dat"][:]
        AhC_arealmass = f["AhC.dir/ArealMass.dat"][:]
        AhC_christodouloumass = f["AhC.dir/ChristodoulouMass.dat"][:]
        AhC_dimensionfulinertialspin = f["AhC.dir/DimensionfulInertialSpin.dat"][:]
        AhC_dimensionfulinertialspinmag = f["AhC.dir/DimensionfulInertialSpin.dat"][:]
        AhC_chiinertial = f["AhC.dir/chiInertial.dat"][:]
        AhC_chimaginertial = f["AhC.dir/chiMagInertial.dat"][:]

        # Create an array of zeros with 8 columns
        n_rows = AhA_positions.shape[0] + AhC_positions.shape[0]
        print(n_rows)
        output_dat = np.zeros((n_rows, 22))
        
        # Extract time (column 0), x (column 1), and y (column 2) from AhA's data
        time = AhA_positions[:, 0]  # Assumes columns: [time, x, y, z]
        time = np.concatenate((time, AhC_positions[:, 0]))
        print(time)
        x = -1*AhA_positions[:, 1]
        x = np.concatenate((x, -1*AhC_positions[:, 1]))
        y = AhA_positions[:, 2]
        y = np.concatenate((y, AhC_positions[:, 2]))
        z = AhA_positions[:, 3]
        z = np.concatenate((z, AhC_positions[:, 3]))
        am = AhA_arealmass[:, 1]
        am = np.concatenate((am, AhC_arealmass[:, 1]))
        cm = AhA_christodouloumass[:, 1]
        cm = np.concatenate((cm, AhC_christodouloumass[:, 1]))
        disx = AhA_dimensionfulinertialspin[:, 1]
        disx = np.concatenate((disx, AhC_dimensionfulinertialspin[:, 1]))
        disy = AhA_dimensionfulinertialspin[:, 2]
        disy = np.concatenate((disy, AhC_dimensionfulinertialspin[:, 2]))
        disz = AhA_dimensionfulinertialspin[:, 3]
        disz = np.concatenate((disz, AhC_dimensionfulinertialspin[:, 3]))
        dism = AhA_dimensionfulinertialspinmag[:, 1]
        dism = np.concatenate((dism, AhC_dimensionfulinertialspinmag[:, 1]))
        xix = AhA_chiinertial[:, 1]
        xix = np.concatenate((xix, AhC_chiinertial[:, 1]))
        xiy = AhA_chiinertial[:, 2]
        xiy = np.concatenate((xiy, AhC_chiinertial[:, 2]))
        xiz = AhA_chiinertial[:, 3]
        xiz = np.concatenate((xiz, AhC_chiinertial[:, 3]))
        xim = AhA_chimaginertial[:, 1]
        xim = np.concatenate((xim, AhC_chimaginertial[:, 1]))
 
        # Assign time to column 1 (index 0), x to column 6 (index 5), and y to column 8 (index 7)
        
        output_dat[:, 0] = time
        output_dat[:, 1] = am
        output_dat[:, 2] = cm
        output_dat[:, 3] = x
        output_dat[:, 4] = y
        output_dat[:, 5] = z
        output_dat[:, 6] = disx
        output_dat[:, 7] = disy
        output_dat[:, 8] = disz
        output_dat[:, 9] = dism
        output_dat[:, 10] = xix
        output_dat[:, 11] = xiy
        output_dat[:, 12] = xiz
        output_dat[:, 13] = xim

        header = np.array([["# column 0: time = [time]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 1: am = [areal_mass]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 2: cm = [christodoulou_mass]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 3: x = [positions_x] * -1", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 4: y = [positions_y]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 5: z = [positions_z]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 6: disx = [dimensionful_inertial_spin_x]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 7: disy = [dimensionful_inertial_spin_y]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 8: disz = [dimensionful_inertial_spin_z]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 9: dism = [dimensionful_inertial_spin_mag]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 10: xix = [chi_inertial_x]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 11: xiy = [chi_inertial_y]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 12: xiz = [chi_inertial_z]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                  ["# column 13: xim = [chi_mag_inertial]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]])
        
        output_data = np.vstack((header, output_dat))

        # Define the output file path
        output_file = os.path.join(output_dir, "puncture_posns_vels_regridxyzU.txt")
        
        # Save data using np.savetxt
        np.savetxt(output_file, output_data, fmt="%s", delimiter=" ")
        
        print(f"Saved black hole positions to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 convert_h5_to_ascii.py <input_h5_file> <output_dir> <R_ext> <pos_h5_file>")
        sys.exit(1)

    h5_file = sys.argv[1]
    output_dir = sys.argv[2]
    R_ext = float(sys.argv[3])  # Extraction radius
    pos_file = sys.argv[4]  # Black hole position data file

    # Convert the HDF5 file to ASCII
    convert_h5_to_ascii(h5_file, output_dir, R_ext)

    # Extract black hole positions and save to ASCII
    extract_black_hole_positions(pos_file, output_dir)

    print("Conversion and extraction complete.")
