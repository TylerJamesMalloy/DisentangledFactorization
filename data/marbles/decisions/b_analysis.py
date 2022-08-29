from cmath import nan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd 
import numpy as np 
import seaborn as sns
import scipy.stats as stat
import matplotlib.pyplot as plt 
import glob
import ast 
import math 
import scipy 

import matplotlib.pyplot as plt

data_paths = glob.glob("both/*")

response_accuracy = []
for participant_id, data_path in enumerate(data_paths):
    #print(data_path)
    data = pd.read_csv(data_path)
    experiment_data = data.tail(200).to_numpy()
    total_reward = 0

    for response in experiment_data:
        if np.array([value != value for value in response]).any(): continue 
        total_reward += response[2]

    response_accuracy.append(total_reward) 

response_accuracy = np.array(response_accuracy)

decent_response_accuracy = response_accuracy[(response_accuracy > 425)]
print("number of participants with more than 68 percent change detection: ", len(decent_response_accuracy), " out of ", len(response_accuracy), " total participants")

fig, ax = plt.subplots(figsize=(8,3))
_, bins, _ = plt.hist(decent_response_accuracy, 10, histtype = 'bar', density=True)


mu, sigma = scipy.stats.norm.fit(decent_response_accuracy)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
#print(mu, sigma, best_fit_line)
plt.plot(bins, best_fit_line, axes=ax)

#plt.xticks([0, 30, 68, 120, 150])
#plt.xticks([0.64, 0.75, 0.9])
plt.title("Participants Total Points")
plt.show()

