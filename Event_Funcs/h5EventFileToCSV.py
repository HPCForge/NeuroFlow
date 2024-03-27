import h5py
import numpy as np
import pandas as pd
import sys
import os

# Comment Out if Hardcoding Files ####################################
#if len(sys.argv) < 3:
#    file = input("Enter Input File: ")
#    file = os.path.expanduser(file)
#    file = os.path.relpath(file)
#    outputDir = input("Enter Output Directory (include end slash): ")
#    outputDir = os.path.expanduser(outputDir)
#    outputDir = os.path.relpath(outputDir)
#else:
#    file = sys.argv[1]
#    outputDir = sys.argv[2]
######################################################################

file = '../H5Files/Mass_flux_255_heat_flux_8.h5'
outputDir = '../FullCSVFiles/'

dataset = h5py.File(file, 'r')

h5_events = dataset['events']
print("Loaded Dataset " + file)

np_events_array = np.array(h5_events)
print("Converted " + file + " to NumPy Array")

events_df = pd.DataFrame(np_events_array, columns=['t', 'x', 'y', 'p'])

events_df.to_csv(outputDir + "/" + os.path.basename(file).removesuffix(".h5") + "_events.csv", sep=',', index=False)
print("Converted to csv: " + outputDir + os.path.basename(file).removesuffix(".h5") + "_events.csv")
