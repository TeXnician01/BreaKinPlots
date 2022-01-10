#
# python 3.6.4 
# BreaKinPlots - any spectrometer version
# batch plot breakdown-curve data or kinetic plots from input files in a given folder
# 
# finnkraft 
# svenwilke  
# with the friendly help of TheImportanceOfBeingErnest on StackOverflow
# 
# v001 - 2019-11-28
# v010 - 2020-01-31
# v011 - 2020-07-13
# 


# --------------------------------------------------------
#                        Preamble
# --------------------------------------------------------

from timeit import default_timer as timer
import numpy as np
import math

import matplotlib as mpl
from matplotlib import rcParams, cycler, mathtext
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from matplotlib.lines import Line2D

import os
from shutil import move
import sys


# --------------------------------------------------------
#              Plot setup - breakdown curves
# --------------------------------------------------------

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


# --------------------------------------------------------
#                       Functions
# --------------------------------------------------------

def calc_peakvalue(mi: int, ma: int, x: np.ndarray):
    # Suche Maximum von x in allen Zeilen von mi bis ma in Spalte 1
    # Gib' Peak-Maximum und die Zeile des Maximums zurueck
    max_intensity = np.max(x[mi:ma, 1])
    max_peak = x[mi+np.argmax(x[mi:ma, 1]), 0]
    return max_peak, max_intensity

def calc_peakvalue_integral(mi: int, ma: int, interval: float, x: np.ndarray):
    # Suche Maximum von x in allen Zeilen von mi bis ma in Spalte 1
    # Gib' Peak-Maximum, die Zeile des Maximums und das Peak-Integral zurueck
    max_peak_range = mi+np.argmax(x[mi:ma, 1])
    max_intensity = np.max(x[mi:ma, 1])
    max_peak = x[max_peak_range, 0]
    
    # Finde das Peak-Intervall
    line_number_min = max_peak_range
    line_number_max = max_peak_range
    peak_interval_min = max_peak-interval

    while x[line_number_min, 0] > peak_interval_min:
        line_number_min -= 1

    while x[line_number_max, 0] < peak_interval_max:
        line_number_max += 1

    # Integration
    # Beginne in der Zeile des Maximums
    # im Intervall nach unten und oben integrieren
    mz_x = x[line_number_min:line_number_max, 0]
    mz_y = x[line_number_min:line_number_max, 1]

    peak_integral = np.trapz(mz_y, mz_x)

    return max_peak, max_intensity, peak_integral

def calc_peakvalue_sum(mi: int, ma: int, interval: float, x: np.ndarray):
    # Suche Maximum von x in allen Zeilen von mi bis ma in Spalte 1
    # Gib' Peak-Maximum, die Zeile des Maximums und das Peak-Integral zurueck
    # war: max_peak_range = mi+np.argmax(x[mi:ma, 1])
    # war: max_intensity = np.max(x[mi:ma, 1])
    try:
        max_peak_range = mi+np.argmax(x[mi:ma, 1])
    except ValueError:
        max_peak_range = int(round(((mi+ma)/2), 0))
    try:
        max_intensity = np.max(x[mi:ma, 1])
    except ValueError:
        max_intensity = x[mi, 1]
        # locate the array that raises the error:
        # print(mi, ma, x[mi:ma, 1])
    max_peak = x[max_peak_range, 0]
    
    # Finde das Peak-Intervall
    line_number_min = max_peak_range
    line_number_max = max_peak_range
    peak_interval_min = max_peak-interval

    while x[line_number_min, 0] > peak_interval_min:
        line_number_min -= 1

    while x[line_number_max, 0] < peak_interval_max:
        line_number_max += 1

    # Summation
    # Beginne in der Zeile des Maximums
    # im Intervall nach unten und oben summieren
    mz_y = x[line_number_min:line_number_max, 1]

    peak_sum = float(np.sum(mz_y))

    return max_peak, max_intensity, peak_sum

def calc_peakvalue_max(mi: int, ma: int, x: np.ndarray):
    # Suche dasMaximum von x in allen Zeilen von mi bis ma in Spalte 1
    # Gib' das Peak-Maximum, die Zeile des Maximums und die maximale Peak-Intensitaet zurueck
    try:
        max_peak_range = mi+np.argmax(x[mi:ma, 1])
    except ValueError:
        max_peak_range = int(round(((mi+ma)/2), 0))
    try:
        max_intensity = np.max(x[mi:ma, 1])
    except ValueError:
        max_intensity = x[mi, 1]
        # locate the array that raises the error:
        # print(mi, ma, x[mi:ma, 1])
    max_peak = x[max_peak_range, 0]

    return max_peak, max_intensity

# https://thispointer.com/python-how-to-insert-lines-at-the-top-of-a-file/, Autor: Varun, Datum: Jan 26, 2020
def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
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
    # Rename dummy file as the original file
    move(dummy_file, ("%s_legend.txt" % (os.path.splitext(file_name)[0])))



# --------------------------------------------------------
#                    User Variables
# --------------------------------------------------------

# Arbeitsverzeichnis
file_path = os.path.join("Promotion", "veranstaltungen", "2021_2022", "MeMO", "Auswertung Gruppe 1", "F2-EtOH", "MS-Spektren")

# Ordner (Experiment)
the_path = ""

# Ordner (Grafiken)
graph_path = os.path.join("Promotion", "veranstaltungen", "2021_2022", "MeMO", "Auswertung Gruppe 1", "F2-EtOH", "MS-Spektren")



# --------------------------------------------------------
#                         Code
# --------------------------------------------------------

# Clear the terminal window
os.system('cls' if os.name == 'nt' else 'clear')

print("Willkommen zu BreaKinPlots, einem kleinen Python-Tool zur automatisierten Auswertung von CID-Spektren und Erzeugung von Breakdown Curves.\nIn den folgenden Nutzereingaben sind Standardoptionen \033[04munterstrichen\033[0m. Diese Option wird mit der \033[1mEingabe\033[0m (Enter) Taste bestaetigt.\n")

calc_stdev_standard = "J"
print("Soll im Rahmen der Auswertung die Standardabweichung der Messungen berechnet und ausgegeben werden?\nIn diesem Fall wird der Mittelwert aller Messwerte mit der Standardabweichung ausgegeben.")
calc_stdev = input("Valide Eingaben sind \033[04m%s\033[0m oder N. Eingabe:\n " % (calc_stdev_standard) ) or calc_stdev_standard

# Analyse von BDC oder IMR
evaluation_standard = "bdc"
print("Sollen Daten fuer eine Breakdown-Analyse oder Ionenmolekuel-Reaktion ausgewertet werden?")
evaluation = input("Valide Eingaben sind \033[04m%s\033[0m oder imr. Eingabe:\n " % (evaluation_standard) ) or evaluation_standard

# Spektrometer
spectrometer_standard = "tof"
print("Experimente wurden an folgendem Spektrometer durchgefuehrt (Benennung der output-Ordner und Dateien).")
spectrometer = input("Valide Eingaben sind: \033[04m%s\033[0m oder hct oder amazon. Eingabe:\n " % (spectrometer_standard)) or spectrometer_standard

# Setze den Arbeitspfad
exp_path = input("Relativer Ordnerpfad der zu bearbeitenden Spektren (%s):\n " % ('\033[04m'+the_path+'\033[0m')) or the_path
rel_path = os.path.join(file_path, exp_path)

export_with_legend_standard = "J"
export_with_legend = input("Soll die Legende nur in der Output-Datei gespeichert werden (Die Legende wird nicht im Plot aufgetragen)? (\033[04m%s\033[0m/N):\n " % (export_with_legend_standard)) or export_with_legend_standard

input_f = []
input_p = []

for file in os.listdir(rel_path):
        if file.endswith("files.txt"):
            input_f.append(file)
        
        elif file.endswith("peaks.txt"):
            input_p.append(file)

# Error "index out of range ..." bei line_number_min liegt an falscher Zuordnung einer input_f Datei zur input_p Datei
input_f = sorted(input_f)
input_p = sorted(input_p)

print("Gefunden (input_files):")
filenum = 1
for i in input_f:
    print(filenum,":",i)
    filenum += 1

while True:
    choice_standard = "A"
    choice = input("\nWelche Spektren sollen eingelesen werden? \n  \033[04m%s\033[0m: Alle im Ordner / Durch Kommata getrennte Zahlen (z.B. 1,2,4) " % (choice_standard)) or "A"

    if choice == "A" or choice == "a":
        print("Alle Spektren werden eingelesen.")
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
        print("Es werden nur die Spektren aus folgenden Dateien eingelesen:")
        for file_item in input_f:
            print(file_item, end=" ")
        break

# integral_or_sum Benutzer-Eingabe
integral_or_sum_standard = "SUM"

# q-TOF
if spectrometer == "tof":
    print("\nSummation der Peakintensitaeten wird fuer die Auswertung verwendet.")
    integral_or_sum = "SUM"
# HCT QIT
if spectrometer == "hct":
    # Nachfrage, ob integriert, summiert oder das Maximum verwendet werden soll
    print("\nSollen Peaks integriert oder aufsummiert werden? Alternativ kann ausschliesslich das Maximum verwendet werden.")
    
    integral_or_sum = input("Zum Summieren 'SUM' eingeben, Zum Nutzen des Maximums 'MAX' eingeben (Standard: %s). Zum Abbrechen 'ABR' eingeben: " % (integral_or_sum_standard)) or integral_or_sum_standard

    while True:
        if (integral_or_sum == "ABR" or integral_or_sum == "SUM" or integral_or_sum == "MAX"):
            break
        integral_or_sum = input("Falsche Eingabe! Bitte 'ABR' oder 'SUM' eingeben: ")

    if integral_or_sum == "ABR":
        print("Programm durch Benutzereingabe unterbrochen.")
        sys.exit()
# amaZon QIT
if spectrometer == "amazon":
    # Nachfrage, ob integriert, summiert oder das Maximum verwendet werden soll
    print("\nSollen Peaks integriert oder aufsummiert werden? Alternativ kann ausschliesslich das Maximum verwendet werden.")
    print("Eingabe bitte ohne Anfuerhrungszeichen. Erfolgt keine Eingabe, wird summiert.\n")
        
    integral_or_sum = input("Zum Integrieren 'INT', zum Summieren 'SUM' eingeben, Zum Nutzen des Maximums 'MAX' eingeben (Standard: %s). Zum Abbrechen 'ABR' eingeben: " % (integral_or_sum_standard)) or integral_or_sum_standard

    while True:
        if (integral_or_sum == "ABR" or integral_or_sum == "INT" or integral_or_sum == "SUM" or integral_or_sum == "MAX"):
            break
        integral_or_sum = input("Falsche Eingabe! Bitte 'ABR', 'INT' oder 'SUM' eingeben: ")

    if integral_or_sum == "ABR":
        print("Programm durch Benutzereingabe unterbrochen.")
        sys.exit()

try:
    os.mkdir(os.path.join(graph_path, exp_path))
    print("Der Zielordner wurde erstellt.")
except FileExistsError:
    print("Der Zielordner existiert bereits und wurde daher nicht erstellt.")
        
figure_suffix = input("\nSoll die Grafik im eps- oder png- oder svg-Format oder kombiniert (.eps und .png) exportiert werden? Bitte eps, png, \033[04mepspng\033[0m oder svg eingeben: ") or "epspng"

#start = timer()

for i in range(len(input_f)):
    # input-Dateien
    # input_filename = input("Name der Spektren-input-Datei (Dateiliste, z.B.: input_files.txt): ") or "026_977-_input_files.txt"
    # input_peakfilename = input("Name der Peak-input-Datei (Peakliste, z.B.: input_peaks.txt): ") or "026_977-_input_peaks.txt"
    input_filename = input_f[i]
    # print(input_filename)
    input_peakfilename = input_p[i]
    # print(input_peakfilename)

    # Lese Daten ein - path.join fuegt rel. Pfad und Trennzeichen je nach Betriebssystem ein
    file_list = np.genfromtxt(os.path.join(rel_path, "%s" % (input_filename)), dtype='U', delimiter=' ', skip_header=1)

    peak_list = np.genfromtxt(os.path.join(rel_path, "%s" % (input_peakfilename)), dtype='float', delimiter=' ', skip_header=1, usecols=(0,1,2,3,4))

    # Liste der Spezies (Ionen) aus der Input-Datei lesen
    ion_labels = np.genfromtxt(os.path.join(rel_path, "%s" % (input_peakfilename)), dtype="U", skip_header=1, usecols=(0))

    # Liste des Ion-Typs aus der Input-Datei lesen
    # ion_types = np.genfromtxt(os.path.join(rel_path, "%s" % (input_peakfilename)), dtype="U", skip_header=1, usecols=(7))

    # Liste der Farben fuer Spezies im Plot aus input-Datei
    try:
        color_list = np.genfromtxt(os.path.join(rel_path, "%s" % (input_peakfilename)), dtype="U", delimiter=" ", skip_header=1, usecols=(5))
    except ValueError:
        print("\nEs wurden keine Farben in der input-Datei angegeben. Die Standard-Farben werden verwendet.\n")
        color_list = ['000000', '0000FE', 'FE0000', '008001', 'CF3BC1', 'FD8000', 'e377c2', '7f7f7f', 'bcbd22', '17becf']

    # Liste der Marker fuer Spezies im Plot aus input-Datei
    try:
        marker_list = np.genfromtxt(os.path.join(rel_path, "%s" % (input_peakfilename)), dtype="U", delimiter=" ", skip_header=1, usecols=(6))
    except ValueError:
        print("\nEs wurden keine Marker in der input-Datei angegeben. Die Standard-Marker werden verwendet.\n")
        marker_list = ["D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D"]

    # Liste der Marker fuer Spezies im Plot aus input-Datei
    try:
        fill_list = np.genfromtxt(os.path.join(rel_path, "%s" % (input_peakfilename)), dtype="U", delimiter=" ", skip_header=1, usecols=(7))
    except ValueError:
        print("\nEs wurden keine Fuellstile fuer Marker in der input-Datei angegeben. Die Standard-Fuellstile werden verwendet.\n")
        fill_list = ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none"]

    # Erzeuge output-array
    #   Modifizierung: 2*np.size(peak_list, 0) statt np.size(peak_list, 0),
    #   um direkt die Standardabweichung einfuegen zu koennen
    if calc_stdev == "J":
        output_array_size_x = 1+2*np.size(peak_list, 0)
    
    if calc_stdev == "N":
        output_array_size_x = 1+np.size(peak_list, 0)

    output_array_size_y = np.size(file_list, 0)

    output_array = np.zeros((output_array_size_y, output_array_size_x), dtype=float)

    # Schleife ueber alle Spektren und alle Peaks
    # Normierung und Ausgabe der finalen Output-Datei
    # --------------------------------------------------
    # Globaler Pointer fuer die Dateiposition und den Output-Array
    file_index = 0

    for file in file_list:
        mz_list = np.genfromtxt(os.path.join(rel_path, file[0]), dtype=float, encoding=None)

        # Iteration ueber alle Peaks
        # -------------------------------------------------
        peak_list_size_y = np.size(peak_list, 0)

        for peak in range(0, peak_list_size_y):
            # Iteration ueber alle Isotopologe
            number_of_peaks = peak_list[peak, 4]

            counter = 0

            if calc_stdev == "N":
            # 2. Spalte fuer Integrale im Output-Array
                output_array_isotope = 1+peak

            if calc_stdev == "J":
            # Jede 2. Spalte - neue Definition:  
                output_array_isotope = 2*peak+1
            
            while counter < number_of_peaks:
                # Zeile: Beginn der Peak-Range
                line_number_min = 0

                # Zeile: Ende der Peak-Range      
                line_number_max = 0

                # Peak-Interval aus input_peaks.txt, Intervall = bedseitig gleich diesem Wert
                peak_interval = peak_list[peak, 2]

                # Peak-Intervall Minimum und Maximum werden definiert und verschoben
                # im ersten Schritt counter = 0 -> Verschiebung findet nicht statt
                peak_interval_min = peak_list[peak, 1]-peak_interval+(peak_list[peak, 3]*counter)
                peak_interval_max = peak_list[peak, 1]+peak_interval+(peak_list[peak, 3]*counter)

                # Suche Zeile des Peaks und dessen Maximum im Peakintervall
                while mz_list[line_number_min, 0] < peak_interval_min:
                    line_number_min += 1

                line_number_max = line_number_min

                while mz_list[line_number_max, 0] < peak_interval_max:
                    line_number_max += 1
                
                # Zeilennummer-Maximum = letzter Wert des gueltigen Bereichs
                line_number_max -= 1

                # Finde das Peak-Maximum
                # peak_max = calc_peakvalue(line_number_min, line_number_max, mz_list)

                # Fuege dem output_array je nach verwendeter Methode die relative Intensitaet hinzu
                if integral_or_sum == "INT":
                    # Finde das Peak-Maximum und integriere im Peak-Intervall
                    peak_max_integral = calc_peakvalue_integral(line_number_min, line_number_max, peak_interval, mz_list)
                
                    # Addiere das Integral zum Wert fuer den Peak im output_array
                    output_array[file_index, output_array_isotope] += peak_max_integral[2]

                elif integral_or_sum == "SUM":
                    # Finde das Peak-Maximum und summiere im Peak-Intervall
                    peak_max_sum = calc_peakvalue_sum(line_number_min, line_number_max, peak_interval, mz_list)
                
                    # Addiere die Summe zum Wert fuer den Peak im output_array
                    #print(output_array_isotope, peak)
                    output_array[file_index, output_array_isotope] += peak_max_sum[2]
                
                elif integral_or_sum == "MAX":
                    # Finde das Peak-Maximum und gib' es zurueck
                    peak_max_max = calc_peakvalue_max(line_number_min, line_number_max, mz_list)
                
                    # Addiere die Summe zum Wert fuer den Peak im output_array
                    output_array[file_index, output_array_isotope] += peak_max_max[1]
                
                counter += 1

        # Bilde Summe alle Integrale oder Summen aller Peaks fuer eine Datei (eine Spannung oder Zeit) zum Normieren
        peak_sum = np.sum(output_array[file_index, 1:])

        # Version a: keine Standardabweichung
        # Normiere alle Integrale oder Summen aller Peaks auf die Summe der Integrale
        if calc_stdev == "N":
            for species in range(1, peak_list_size_y+1):
                output_array[file_index, species] /= peak_sum

        #Version b: mit Standardabweichung
        if calc_stdev == "J":
            for species in range(0, peak_list_size_y):
                output_array[file_index, (2*species+1)] /= peak_sum

        # Erweitere den output_array um die x-Werte (= Spannung/ Zeit)
        output_array[file_index, 0] = float(file[1])

        file_index += 1

    #print(output_array)

    # Ausgabe der relativen Intensitaeten in Prozent
    output_array[:, 1:] = output_array[:, 1:]*100

    # output_array entlang der y-Achse an Hand der Anregungsenergie sortieren
    output_array = output_array[np.argsort(output_array[:, 0])]
    
    #print(output_array)
    
    # Fuer jede Spezies und fuer alle Energiepaare:
    # Errechne die (paarweise) Standardabweichung und fuege sie in den output_array ein
    if calc_stdev == "J":
        for species in range(0, peak_list_size_y):
            # Definition:
            # output_array_size_y-1
            # ueberpruefen!!!
            for energy in range(0, output_array_size_y-1):
                if output_array[energy, 0] == output_array[energy+1, 0]:
                    #print(energy, energy+1)
                    measureA = output_array[energy, 2*species+1]
                    measureB = output_array[energy+1, 2*species+1]
                    medianAB = 1/2*(measureA + measureB)
                    # fuege Standardabweichung in den output_array ein
                    output_array[energy, 2*species+2] = math.sqrt((measureA - medianAB)**2 + (measureB - medianAB)**2)

                else:
                    continue

        # Loeschen jeder zweiten Zeile 
        #output_array = np.delete(output_array, list(range(0, output_array.shape[0], 2)), axis=0)
        output_array = np.array(output_array)[::2]

    #if calc_stdev == "N":
    #    break

    #print(output_array)
    #sys.exit()

    # Speichere das output_array in einer Textdatei
    filename_split = os.path.splitext("%s" % (input_filename))[0]
    np.savetxt(os.path.join(rel_path, "%s_output_%s_%s.txt" % (filename_split, spectrometer, integral_or_sum)), output_array, delimiter=' ')

    if export_with_legend == "J":
        # Liste der Elemente fuer Legende zum Export des output_array mit Legende
        legend = []

        # Vom Spektrometer abhaengige x-Achsenbeschriftung an Legende anfuegen
        if evaluation == "bdc":
            if spectrometer == "tof":
                legend.append("E_lab(TOF)[eV]")
            if spectrometer == "hct":
                legend.append("V_exc(HCT-QIT)[V]")
            if spectrometer == "amazon":
                legend.append("V_exc(amaZon-QIT)[V]")
        
        elif evaluation == "imr":
            legend.append("t [ms]")

        if calc_stdev == "N":
            # Erweitere Legenden-Liste um alle Ionen-Labels
            for ion in ion_labels:
                legend.append(ion)
        
        if calc_stdev == "J":
            for ion in ion_labels:
                legend.append(ion)
                legend.append("std_dev")
        
        # Speichere die neue Output-Datei und behalte die Originaldatei
        # Zum Loeschen der Originaldatei, os.remove(file_name) in prepend_line() nutzen
        legend = " ".join(legend)
        prepend_line(os.path.join(rel_path, "%s_output_%s_%s.txt" % (filename_split, spectrometer, integral_or_sum)), legend)
    
    #print("Auswertung und Export erfolgreich!")

    # Option fuer Auftragung bei calc_stdev == "J" hinzufuegen
    if calc_stdev == "N":
        # fuer jedes ion
        # trage x- und y-Werte auf
        for ion in range(1, output_array_size_x):
            # automatische Ermittlung der mfc an Hand des fillstyles
            if (fill_list[ion-1]) == "full":
                mfcolor = "#%s" % (color_list[ion-1])
            else:
                mfcolor = "none"
            # Auftragung erstellen

            plt.plot(output_array[:,0], output_array[:,ion], clip_on=True, marker="%s" % (marker_list[ion-1]), mfc=mfcolor, mec="#%s" % (color_list[ion-1]), mew=0.85, ls="none", fillstyle="%s" % (fill_list[ion-1]), markerfacecoloralt="#%s" % (color_list[ion-1]), label=r"$\mathregular{%s}$" % (ion_labels[ion-1])) 

            if export_with_legend == "N":
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

        # Ausgabe der x-Werte
        print("\nAusgewerteter Datensatz: %s" % (filename_split.strip("_input_files")))

        print("\nBis zu welchem x-Wert soll aufgetragen werden? Standardmaessig wird bis zum \033[04mmaximalen x-Wert\033[0m (%.2f) aufgetragen." % np.max(output_array[:, 0]))
        #for xval in range(len(output_array[:, 0])):
        #    print(xval, end=" ")

        xtick_max = float(input("x-Wert: ") or np.max(output_array[:, 0]))
        #ytick_max = np.max(output_array[:, 1:])
        ytick_max = input("\nWelches y-Limit soll gewaehlt werden? Bitte eine ganze Zahl zwischen 0 und \033[04m100\033[0m eingeben: ") or 100

        plt.ylim(0, ytick_max)
        plt.ylabel("Relative signal intensity / %", labelpad=-3)

        plt.xlim(0, xtick_max)
        if evaluation == "bdc":
            # Vom Spektrometer abhaengige x-Achsenbeschriftung
            if spectrometer == "tof":
                plt.xlabel(r'$E_\mathregular{lab}\,\mathregular{(TOF)}$ [eV]', labelpad=10)    # Normal CID
                #plt.xlabel(r'$Collision RF\,\mathregular{(TOF)}$ [Vpp]', labelpad=10)           # Collision RF CID
            if spectrometer == "hct":
                plt.xlabel(r"$V_\mathregular{exc}\,\mathregular{(HCT-QIT)}$ [V]", labelpad=10)
            if spectrometer == "amazon":
                plt.xlabel(r"$V_\mathregular{exc}\,\mathregular{(amaZon-QIT)}$ [V]", labelpad=10)
        
        elif evaluation == "imr":
                plt.xlabel("$t$ [ms]", labelpad=10) # put in [ms] for IMR or [h] for time-dependent measurement

        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))

        # was: majors = (np.min(output_array[:, 1:]), np.max(output_array[:, 1:]))
        majors = (0, 100)
        # was: plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
        plt.gca().yaxis.set_major_locator(FixedLocator(majors))

        if export_with_legend == "N":
            if figure_suffix == "epspng":
                plt.savefig(os.path.join(graph_path, exp_path, "%s_output_%s_%s_legend.eps" % (filename_split, spectrometer, integral_or_sum)), transparent=True, dpi=800)
                plt.savefig(os.path.join(graph_path, exp_path, "%s_output_%s_%s_legend.png" % (filename_split, spectrometer, integral_or_sum)), transparent=True, dpi=800)
            
            else:
                plt.savefig(os.path.join(graph_path, exp_path, "%s_output_%s_%s_legend.%s" % (filename_split, spectrometer, integral_or_sum, figure_suffix)), transparent=True, dpi=800)
                
        elif export_with_legend == "J":
            if figure_suffix == "epspng":
                plt.savefig(os.path.join(graph_path, exp_path, "%s_output_%s_%s.eps" % (filename_split, spectrometer, integral_or_sum)), transparent=True, dpi=800)
                plt.savefig(os.path.join(graph_path, exp_path, "%s_output_%s_%s.png" % (filename_split, spectrometer, integral_or_sum)), transparent=True, dpi=800)

            else:
                plt.savefig(os.path.join(graph_path, exp_path, "%s_output_%s_%s.%s" % (filename_split, spectrometer, integral_or_sum, figure_suffix)), transparent=True, dpi=800)

        # Bild-Cache leeren, sonst tauchen alle vorigen ebenfalls im naechsten auf
        plt.clf()

#end = timer()
#print(end - start)

print("Auftragung und Export von %s erfolgreich!" % (filename_split.strip("_input_files")))
