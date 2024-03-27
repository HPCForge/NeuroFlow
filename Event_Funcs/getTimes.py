import pandas as pd
import numpy as np
import os

file = '/pub/sarani/KDD/Dataset/Unprocessed/FullCSVFiles/V55.csv'

iterator = 50000
#events_df = pd.read_csv(file)

##########################################################
colnames = ['x','y','p','t']
events_df = pd.read_csv(file, names=colnames, header=None)
########################################################


startTime = events_df.iloc[0]['t']
endTime = events_df.iloc[-1]['t']

print(f"Duration of Original CSV (s): {(endTime - startTime) / 1e6}")
print(f"Iterator Value: {iterator}")

output_df = pd.DataFrame()

print(len(events_df))

list_df = [events_df[i:i+iterator] for i in range(0,len(events_df),iterator)]

print(list_df[0].size)
print(len(list_df))
print(len(list_df[0]))

print(list_df[0])

for i in range(len(list_df)):
    list_df[i].to_csv("./V55CSV/" + str(i*iterator) + "out(55).csv", sep=",", index=False)
