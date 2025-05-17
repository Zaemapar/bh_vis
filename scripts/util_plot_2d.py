'''
This script allows the user to plot 2 dimensional psi4
or strain data in matplotlib. System arguments let you plot
the mode or modes of choice as well as the superimposed waveform.
'''

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import psi4_FFI_to_strain as util

input_type = "" # strain, psi4
input_dir = ""
plot_full = False

modes_to_plot: list[tuple] = []
if len(sys.argv) < 3:
    raise RuntimeError(
        """Please include the following system arguments:
        python
        <name of this file>
        <directory for folder holding data>
        <input type of data (strain or psi4)>
        <mode index in (l,m) format (e.g. (2,2))>
        <as many modes as you want, "full" to get
        superimposed waveform, or "all" to get all modes at once>"""
    )
else:
    input_dir = sys.argv[1]
    input_type = str(sys.argv[2]).lower()
    if input_type != "strain" and input_type != "psi4":
        raise RuntimeError(
            "Invalid input type: please enter 'strain' or 'psi4'")

    # --- Minimum and Maximum Ell Mode Calculations ---
    ells = np.empty(0)
    file_list = os.listdir(input_dir)
    bh_files = [f for f in file_list if os.path.isfile(os.path.join(input_dir, f))] # List the names of these files
    for b in bh_files:
        # Attempt to convert the part of the file name that is supposed to be the mode into an integer
        try:
            ell = float(b[-13])
        except ValueError:
            continue # Skip over files that fail or don't have a mode
        if ell not in ells:
            ells = np.append(ells, ell) # Save the ell mode if unique

    ells = np.sort(ells.astype(int)) # Put the ell modes in order
    if len(ells) == 0:
        # If no modes are found, the files are probably incorrectly named
        raise RuntimeError("No l modes found. Ensure files are formatted as such: {filename}_l#-r####.#")
    # Extract the min and max ells
    ell_min = ells[0]
    ell_max = ells[-1]

    #print(sys.argv[3])
    plot_info = sys.argv[3:]
    for i, mode_input in enumerate(plot_info):
        if str(mode_input).lower() == "full":
            plot_full = True
            plot_info.remove("full")
        elif str(mode_input).lower() == "all":
            plot_all = True
            plot_info.remove("all")
        else:
            plot_info[i] = int(plot_info[i])
        #mode_type_check = ast.literal_eval(mode_input)
        #if isinstance(mode_type_check, tuple):
        #    modes_to_plot.append(mode_input)
    modes_to_plot = [
        (plot_info[i], plot_info[i+1]) for i in range(0, len(plot_info), 2)
    ]

print("modes to be plotted:", modes_to_plot)
print("Plot full waveform:", plot_full)
util.INPUT_DIR = input_dir
if (input_type == "strain"):
    util.FILE_PATTERN = "_l[MODE=L]_conv_to_strain"
print(util.INPUT_DIR)
# finds and stores data to plot, whether it's psi4 or strain
time_data, modes_data = util.read_psi4_dir(input_dir, ell_max)

# create plots for inputted modes
for current_mode in modes_to_plot:
    current_l = current_mode[0]
    current_m = current_mode[1]
    y_plot = modes_data[util.get_index_from_modes(current_l, current_m, ell_min)]
    color_choice = random.choice(plt.cm.tab10.colors)
    plt.plot(
        time_data, 
        y_plot.real, 
        color=color_choice, 
        alpha=0.5, 
        label=current_mode
    )
if plot_full:
    y_plot_sum = np.zeros_like(time_data)
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            y_plot_sum += modes_data[util.get_index_from_modes(ell, m, ell_min)].real
    color_choice = random.choice(plt.cm.tab10.colors)
    plt.plot(
        time_data, 
        y_plot_sum.real, 
        color=color_choice, 
        alpha=0.5, 
        label="full waveform"
    )
if plot_all:
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            data = modes_data[util.get_index_from_modes(ell, m, ell_min)].real
            color_choice = random.choice(plt.cm.tab10.colors)
            plt.plot(
                time_data,
                data.real,
                color=color_choice,
                alpha=0.5,
                label=f"l={ell},m={m}"
    )

plt.title(f"{input_type} vs. time")
plt.ylabel(f"{input_type}")
plt.xlabel("time")
plt.legend()
plt.grid(True)
plt.show()
