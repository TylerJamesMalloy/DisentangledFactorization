import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd 
import imageio

import cv2
import imageio

from os import listdir
from os.path import isfile, join



SETS = ['set0', 'set1', 'set2', 'set3', 'set4', 'set5']

for SET in SETS:
    all_stimuli = [f for f in listdir('./'+ SET +'/') if isfile(join('./'+ SET +'/', f))]

    all_data = []
    for stimuli_id in range(100):
        #data = np.load(join('./'+ SET +'/', stimuli), allow_pickle=True)
        im = imageio.imread('./'+ SET +'/stimulus' + str(stimuli_id) + '.png')
        im = cv2.resize(im, (64, 64))
        data = np.asarray(im)
        all_data.append(data)

    all_data = np.asarray(all_data)
    np.save(join('./'+ SET +'/', "stimuli.npy"), all_data)