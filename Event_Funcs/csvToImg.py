from PIL import Image
import numpy as np
import pandas as pd
import os

def csvToFrame(filename):
    events_df = pd.read_csv(filename)

    #img_grid = np.zeros((768,1024), dtype=int)
    ############################################
    img_grid = np.zeros((720,1280), dtype=int)
    ############################################

    for index, row in events_df.iterrows():
        img_grid[int(row['y'])][int(row['x'])] = 1

    data = Image.fromarray(np.invert((img_grid * 255).astype(np.uint8)))

    data.save('../V55imgs/' + os.path.basename(filename).split('.')[0] + '_image.png')

path = './V55CSV/'

i = 0
l = len(os.listdir(path))

for filename in os.listdir(path):
    i = i + 1
    print(f"{i} / {l}")
    csvToFrame(path + filename)
