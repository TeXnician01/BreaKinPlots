# python 3.6.4 
# BreaKinPlots 2.3

# DESCRIPTION
# batch plot breakdown-curve data or kinetic plots from input files in a given folder

# AUTHORS
# finnkraft 
# svenwilke  
# with the friendly help of TheImportanceOfBeingErnest and Thomas Auth

# DISCLAIMER
# The script is made available to the user without any guarantee that it is error-free


# --------------------------------------------------------
#                         ToDo
# --------------------------------------------------------

# ALL TODOS ARE FLAGGED WITH "WIP" (no quotes)

# Figure export with standard deviation; At the moment the option "calc_stdev_standard" must be set to "N" in the config file "BreaKinPlots-config.txt" to enable figure output

# can calc_peakvalue function be deleted?
# should be of no use due to new functions


# --------------------------------------------------------
#                        Preamble
# --------------------------------------------------------

from pickle import FALSE, TRUE
import numpy as np
import math
#from lmfit.models import GaussianModel

import matplotlib as mpl # type: ignore
from matplotlib import rcParams, cycler, mathtext  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import plot, show  # type: ignore
from matplotlib.ticker import AutoMinorLocator, FixedLocator  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

import os
from shutil import move
import sys


# --------------------------------------------------------
#              Plot setup - breakdown curves
# --------------------------------------------------------

'''
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 16
mathtext.FontConstantsBase.sup1 = 0.6
mathtext.FontConstantsBase.sub1 = 0.5
mathtext.FontConstantsBase.sub2 = 0.5

rcParams['axes.linewidth'] = 1.0
# labepad wird bei Erstellen des Plots manipuliert - siehe unten
# rcParams['axes.labepad'] = 10.0
plot_color_cycle = cycler('color', ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b', 
                                    'e377c2', '7f7f7f', 'bcbd22', '17becf'])
rcParams['axes.prop_cycle'] = plot_color_cycle
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
                                                         # normal (origin)   wide (legend outside)
rcParams.update({"figure.figsize"        : (6.4,4.8),    # was: (6.4,4.8)    (13,4.8)
                 "figure.subplot.left"   : 0.14,        # was: 0.14         0.09
                 "figure.subplot.right"  : 0.946,       # was:  0.6         0.464
                 "figure.subplot.bottom" : 0.156,
                 "figure.subplot.top"    : 0.965,
                 "axes.autolimit_mode"   : "round_numbers",
                 "xtick.major.size"      : 7,
                 "xtick.minor.size"      : 3.5,
                 "xtick.major.width"     : 1.1,
                 "xtick.minor.width"     : 1.1,
                 "xtick.major.pad"       : 5,
                 "xtick.minor.visible"   : True,
                 "ytick.major.size"      : 7,
                 "ytick.minor.size"      : 3.5,
                 "ytick.major.width"     : 1.1,
                 "ytick.minor.width"     : 1.1,
                 "ytick.major.pad"       : 5,
                 "ytick.minor.visible"   : False,
                 "ytick.minor.left"      : False,
                 "lines.markersize"      : 7.5,
                 "legend.fontsize"       : 16,})
# "legend.handlelength": 2,
# "lines.markerfacecolor" : "none", "lines.markeredgewidth"  : 0.8
'''


# --------------------------------------------------------
#                       Functions
# --------------------------------------------------------

# standard function for finding the peak maximum (also for instruments with low resolution)
def calc_peakvalue_max(mi: int, ma: int, x: np.ndarray):
    # search for the maximum of x in all rows from indices mi to ma in column 1 of an array
    # return the maximum (peak) and the corresponding row number for indexing
    try:
        max_peak_range = mi+np.argmax(x[mi:ma, 1])
    except ValueError:
        max_peak_range = int(round(((mi+ma)/2), 0))
    try:
        max_intensity = np.max(x[mi:ma, 1])
    except ValueError:
        max_intensity = x[mi, 1]
    max_peak = x[max_peak_range, 0]

    return max_peak, max_intensity, max_peak_range

# standard function for calculating 
def calc_peakvalue(mi: int, ma: int, interval: float, x: np.ndarray):
    max_peak, max_intensity, max_peak_range = calc_peakvalue_max(mi, ma, x)
    
    # find the peak interval recursively
    line_number_min = max_peak_range
    line_number_max = max_peak_range
    peak_interval_min = max_peak-interval

    while x[line_number_min, 0] > peak_interval_min:
        line_number_min -= 1
    while x[line_number_max, 0] < peak_interval_max:
        line_number_max += 1
    
    # slice array x in the peak interval in the x (m/z vaule) and y (intensity) direction
    mz_x = x[line_number_min:line_number_max, 0]
    mz_y = x[line_number_min:line_number_max, 1]

    return mz_x, mz_y, max_peak, max_intensity

"""
# function for trapezoid integration of peak intensities
def calc_peakvalue_integral(mi: int, ma: int, interval: float, x: np.ndarray):
    # get the maximum and its location from the array
    mz_x, mz_y, max_peak, max_intensity = calc_peakvalue(mi, ma, interval, x)

    # peak integration
    # start at the row of the maximum
    # integrate the peak in the given interval from trapezoids between peaks
    peak_integral = np.trapz(mz_y, mz_x)

    return max_peak, max_intensity, peak_integral
"""

# function for summation of peak intensities 
def calc_peakvalue_sum(mi: int, ma: int, interval: float, x: np.ndarray):
    # get the maximum and its location from the array
    mz_x, mz_y, max_peak, max_intensity = calc_peakvalue(mi, ma, interval, x)
    
    # peak summation
    # start at the row of the maximum
    # sum the peak maxima in the given interval
    peak_sum = float(np.sum(mz_y))

    return max_peak, max_intensity, peak_sum

# Insert given string as a new line at the beginning of a file
# https://thispointer.com/python-how-to-insert-lines-at-the-top-of-a-file/
# author: Varun
# date:   Jan 26, 2020
def prepend_line(file_name, line):
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    # os.remove(file_name)
    # Rename dummy file
    move(dummy_file, ("%s_legend.txt" % (os.path.splitext(file_name)[0])))

def parse_config_file(file_path):
    config = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            
            if ' ' in line:
                var, value = line.split(' ', 1)
                value = value.strip('"')
                if var == "path_standard":
                    value = value.replace(';', os.sep)
                config[var] = value
                
    return config


# --------------------------------------------------------
#     User Variables from BreaKinPlots-settings.txt
# --------------------------------------------------------

# Read the settings from config_file_path
config_file_path = "BreaKinPlots-config.txt"

# Store settings in config dictionary using the parse_config_file function
config = parse_config_file(config_file_path)

# Read and convert the standard settings from the config dictionary
try:
    if "path_standard" in config:
        config["path_standard"] = os.path.abspath(config["path_standard"])

    if "evaluation_standard" in config:
        evaluation = config["evaluation_standard"]

    if config["calc_stdev_standard"] == "Y":
        calc_stdev = True
        stddev_or_not = "_StdDev"
    elif config["calc_stdev_standard"] == "N":
        calc_stdev = False
        stddev_or_not = "_noStdDev"

    if config["report_stdev_standard"] == "Y":
        report_stdev_standard = True
    elif config["report_stdev_standard"] == "N":
        report_stdev_standard = False

    if config["export_with_legend_standard"] == "Y":
        export_with_legend_standard = True
    elif config["export_with_legend_standard"] == "N":
        export_with_legend_standard = False

    if "calc_method_standard" in config:
        calc_method_standard = config["calc_method_standard"]
    
    if "x_unit_standard" in config:
        x_unit_standard = config["x_unit_standard"]

    if config["figure_export_standard"] == "Y":
        figure_export = True
    elif config["figure_export_standard"] == "N":
        figure_export = False

    if "dpi_standard" in config:
        dpi_standard = int(config["dpi_standard"])

except Exception as exc:
    print(f"An error occurred: Possibly the {exc} was not defined properly.")
    sys.exit()


# --------------------------------------------------------
#                         Code
# --------------------------------------------------------

# Clear the terminal window
os.system('cls' if os.name == 'nt' else 'clear')
print("Welcome to \033[1mBreaKinPlots\033[0m, a for automated evaluation of CID spectra and generation of breakdown curves or ion-molecule reaction plots.\nThe programme can be interrupted at any time by pressing the key combination Ctrl+C.\n")

print("The default variables set in the config file are: ")
# Print each config variable and its value as a sentence
for key, value in config.items():
    print(f"\033[1m{key}\033[0m: {value}")
inp_continue = input("\nContinue? Valid options are \033[04mY\033[0m or N. Input: ")
if inp_continue == "N":
    sys.exit()

input_f = []
input_p = []

for file in os.listdir(config["path_standard"]):
        if file.endswith("files.txt"):
            input_f.append(file)
        
        elif file.endswith("peaks.txt"):
            input_p.append(file)

# Note: Error "index out of range ..." at line_number_min is due to incorrect assignment of an input_f file to the input_p file
input_f = sorted(input_f)
input_p = sorted(input_p)

print("Found (input_files):")
filenum = 1
for i in input_f:
    print(filenum,":",i)
    filenum += 1

while True:
    choice_standard = "A"
    choice = input("\nWhich spectra should be read in? \n  \033[04m%s\033[0m: Alle in the folder / All files separated by commas (e.g. 1,2,4) " % (choice_standard)) or "A"

    if choice == "A" or choice == "a":
        print("All spectra are read in.")
        break

    else:
        entry = []
        input_f_new = []
        input_p_new = []
        for i in choice.split(","):
            entry.append(int(i)-1)

        # append file by reading the filename
        # add the energies from filenames to a list and link filenames to list of energies 
        # also make the user choose which files to read before proceeding (saves time)
        for i in range(len(entry)):
            input_f_new.append(input_f[entry[i]])
            input_p_new.append(input_p[entry[i]])
        input_f = sorted(input_f_new)
        input_p = sorted(input_p_new)
        print("Only the spectra from the following files are read in:")
        for file_item in input_f:
            print(file_item, end=" ")
        break

for i in range(len(input_f)):
    input_filename = input_f[i]
    # print(input_filename)
    input_peakfilename = input_p[i]
    # print(input_peakfilename)

    # Read in data - path.join inserts relative path and separator depending on the operating system
    file_list = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_filename)), dtype='U', delimiter=' ', skip_header=1)

    # Split the filename extension for export of the output_array as a text file
    filename_split = (os.path.splitext("%s" % (input_filename))[0])

    peak_list = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_peakfilename)), dtype='float', delimiter=' ', skip_header=1, usecols=(0,1,2,3,4))

    # Read list of species (ions) from input file
    ion_labels = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_peakfilename)), dtype="U", skip_header=1, usecols=(0))

    # Read ion type from input file
    # ion_types = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_peakfilename)), dtype="U", skip_header=1, usecols=(7))

    # List of colors for species (ions)
    try:
        color_list = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_peakfilename)), dtype="U", delimiter=" ", skip_header=1, usecols=(5))
    except ValueError:
        print("\nEs wurden keine Farben in der input-Datei angegeben. Die Standard-Farben werden verwendet.\n")
        color_list = ['000000', '0000FE', 'FE0000', '008001', 'CF3BC1', 'FD8000', 'e377c2', '7f7f7f', 'bcbd22', '17becf']

    # List of markers for species (ions)
    try:
        marker_list = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_peakfilename)), dtype="U", delimiter=" ", skip_header=1, usecols=(6))
    except ValueError:
        print("\nNo markers were specified in the input file. The default markers are used.\n")
        marker_list = ["D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D"]

    # List of marker fill styles for species (ions)
    try:
        fill_list = np.genfromtxt(os.path.join(config["path_standard"], "%s" % (input_peakfilename)), dtype="U", delimiter=" ", skip_header=1, usecols=(7))
    except ValueError:
        print("\nNo fill styles for markers have been specified in the input file. The standard fill styles are used.\n")
        fill_list = ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none"]

    # Generate output array
    if calc_stdev == True:
        output_array_size_x = 1+2*np.size(peak_list, 0)
    
    if calc_stdev == False:
        output_array_size_x = 1+np.size(peak_list, 0)

    output_array_size_y = np.size(file_list, 0)

    output_array = np.zeros((output_array_size_y, output_array_size_x), dtype=float)

    # Loop over all spectra and all peaks
    # Normalisation and output of the final output file
    # --------------------------------------------------
    # Global pointer for the file position and the output array
    file_index = 0

    for file in file_list:
        mz_list = np.genfromtxt(os.path.join(config["path_standard"], file[0]), dtype=float, encoding=None)

        # Iteration over all peaks
        # -------------------------------------------------
        peak_list_size_y = np.size(peak_list, 0)

        for peak in range(0, peak_list_size_y):
            # Iteration ueber alle Isotopologe
            number_of_peaks = peak_list[peak, 4]

            counter = 0

            if calc_stdev == False:
            # Second column for integrals in the output array
                output_array_isotope = 1+peak

            if calc_stdev == True:
            # Every second columns - redefined
                output_array_isotope = 2*peak+1
            
            while counter < number_of_peaks:
                # Line: Start of the peak range
                line_number_min = 0

                # Line: End of the peak range   
                line_number_max = 0

                # Peak interval from input_peaks.txt, interval = on both sides equally to this value
                peak_interval = peak_list[peak, 2]

                # Peak interval minimum and maximum are defined and shifted
                # in the first step counter = 0 -> shift does not take place
                peak_interval_min = peak_list[peak, 1]-peak_interval+(peak_list[peak, 3]*counter)
                peak_interval_max = peak_list[peak, 1]+peak_interval+(peak_list[peak, 3]*counter)

                # Search for the line of the peak and its maximum in the peak interval
                while mz_list[line_number_min, 0] < peak_interval_min:
                    line_number_min += 1

                line_number_max = line_number_min

                while mz_list[line_number_max, 0] < peak_interval_max:
                    line_number_max += 1
                
                # Line number maximum = last value of the valid range
                line_number_max -= 1

                # Finde das Peak-Maximum
                # peak_max = calc_peakvalue(zeilen_zahl_min, zeilen_zahl_max, mz_liste)

                if config["calc_method_standard"] == "SUM":
                    # Find the peak maximum and sum in the peak interval
                    peak_max_sum = calc_peakvalue_sum(line_number_min, line_number_max, peak_interval, mz_list)
                
                    # Add the sum to the value for the peak in the output_array
                    output_array[file_index, output_array_isotope] += peak_max_sum[2]
                
                elif config["calc_method_standard"] == "MAX":
                    # Find and return the peak maximum
                    peak_max_max = calc_peakvalue_max(line_number_min, line_number_max, mz_list)
                
                    # Add the sum to the value for the peak in the output_array
                    output_array[file_index, output_array_isotope] += peak_max_max[1]
                
                counter += 1

        # Form sum of all integrals or sums of all peaks for one file (one voltage or time) for normalization
        peak_sum = np.sum(output_array[file_index, 1:])

        # Version a: no standard deviation
        # Normalize all integrals or sums of all peaks to the sum of the integrals
        if calc_stdev == False:
            for species in range(1, peak_list_size_y+1):
                output_array[file_index, species] /= peak_sum

        # Version a: with standard deviation
        if calc_stdev == True:
            for species in range(0, peak_list_size_y):
                output_array[file_index, (2*species+1)] /= peak_sum

        # Extend the output_array by the x-values (= voltage/time)
        output_array[file_index, 0] = float(file[1])

        file_index += 1

    # Output of the normalized intensities in percent
    output_array[:, 1:] = output_array[:, 1:]*100

    # Sort output_array along the y-axis based on the excitation energy
    output_array = output_array[np.argsort(output_array[:, 0])]

    # For each species and for all energy pairs:
    # Calculate the (pairwise) standard deviation and insert it into the output_array
    if calc_stdev == True:
        for species in range(0, peak_list_size_y):
            # Definition:
            # output_array_size_y-1
            for energy in range(0, output_array_size_y-1):
                if output_array[energy, 0] == output_array[energy+1, 0]:
                    measureA = output_array[energy, 2*species+1]
                    measureB = output_array[energy+1, 2*species+1]
                    medianAB = 1/2*(measureA + measureB)
                    # insert standard deviation (1 sigma) into the output_array
                    output_array[energy, 2*species+2] = math.sqrt((measureA - medianAB)**2 + (measureB - medianAB)**2)
                    # Replace output_array y-value = intensity of ion with the averaged intensity of this ion (medianAB)
                    output_array[energy, 2*species+1] = medianAB

                else:
                    continue

        # Delete every second line via numpy array slice
        output_array = np.array(output_array)[::2]

        # If the median should be calculated and given but the standard deviation should not be reported, slice the output array
        if report_stdev_standard == False:
            # Extract the desired columns: 0, 1, and every second column starting from column 3
            columns = [0, 1] + list(range(3, output_array.shape[1], 2))
            # Modify the output_array with the selected columns
            output_array = output_array[:, columns]

        # Save the output_array in a text file with proper suffix
        np.savetxt(os.path.join(config["path_standard"], "%s_output_%s%s.txt" % (filename_split, config["calc_method_standard"], stddev_or_not)), output_array, delimiter=' ')

    elif calc_stdev == False:
        # Save the output_array in a text file with proper suffix
        np.savetxt(os.path.join(config["path_standard"], "%s_output_%s%s.txt" % (filename_split, config["calc_method_standard"], stddev_or_not)), output_array, delimiter=' ')

    if export_with_legend_standard == True:
        # List of elements for legend to export the output_array with legend
        legend = []

        # Append X-axis label to legend list
        if evaluation == "BDC":
            legend.append("E_Lab (%s)" % config["x_unit_standard"])
        elif evaluation == "IMR":
            legend.append("t (ms)")
        
        if calc_stdev == True:
            for ion in ion_labels:
                legend.append(ion)
                legend.append("std_dev")
        elif calc_stdev == False:
            # Expand legend list to include all ion labels
            for ion in ion_labels:
                legend.append(ion)
        
        # Save the new output file and keep the original file
        # WIP option: To delete the original file, use os.remove(file_name) in prepend_line()

        legend = " ".join(legend)
        prepend_line(os.path.join(config["path_standard"], "%s_output_%s%s.txt" % (filename_split, config["calc_method_standard"], stddev_or_not)), legend)
    
    if figure_export == True:
        
        # Plot x and y values for each ion
        for ion in range(1, output_array_size_x):
            # fill styles
            try:
                if (fill_list[ion-1]) == "full":
                    mfcolor = "#%s" % (color_list[ion-1])
            except IndexError:
                mfcolor = "none"

        # Plotting starts here
        if calc_stdev == True:
            # Extract n values
            n_values = output_array[:, 0]

            # Extract values and errors, assuming pairs of value and y-error
            values = output_array[:, 1::2]
            y_errors = output_array[:, 2::2]

            # Plotting
            for i in range(values.shape[1]):
                plt.errorbar(n_values, values[:, i], yerr=y_errors[:, i], capsize=5, fmt='o')

        if calc_stdev == False:
            # Extract n values
            n_values = output_array[:, 0]

            # Extract values
            values = output_array[:, 1:]
 
            # Plotting
            for i in range(values.shape[1]):
                plt.plot(n_values, values[:, i], 'o')

        # x values
        print("\nAusgewerteter Datensatz: %s" % (filename_split.strip("_input_files")))

        # Check and assign x- and y-limits
        try:
            xtick_max = float(config["figure_x_standard"])
        except ValueError:
            if config["figure_x_standard"] == "MAX":
                xtick_max = np.max(output_array[:, 0])
            else: 
                raise ValueError
        ytick_max = float(config["figure_y_standard"])

        plt.ylim(0, ytick_max)
        plt.ylabel("Normalized signal intensity / %", labelpad=-3)

        plt.xlim(0, xtick_max)
        if evaluation == "BDC":
            # x axis label
            plt.xlabel(r'$E_\mathregular{Lab}$ / %s' % x_unit_standard, labelpad=10)
            
        elif evaluation == "IMR":
            plt.xlabel("$t$ / %s" % x_unit_standard, labelpad=10)

        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))

        majors = (0, ytick_max)
        plt.gca().yaxis.set_major_locator(FixedLocator(majors))

        plt.savefig(os.path.join(config["path_standard"], "%s_output_%s%s.png" % (filename_split, config["calc_method_standard"], stddev_or_not)), transparent=True, dpi=dpi_standard)

        # Empty image cache, otherwise all previous images will also appear in the next one
        plt.clf()

print("Plot and export of %s successful!" % (filename_split.strip("_input_files")))