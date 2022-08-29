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

data_paths = glob.glob("data/*")
all_marble_colors = pd.read_csv("../source/colors.csv")
all_marble_colors = all_marble_colors['colors']



stimuli_mean_utilities = []
stimuli_deviations = []
stimuli_marble_values = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))
    stimuli_marble_values.append(marble_values)

num_strictly_better = 0
num_subjectively_better = 0 

"""
for stim_1_marble_values in stimuli_marble_values:
    for stim_2_marble_values in stimuli_marble_values:
        stim_1_3s = np.sum(stim_1_marble_values == 3)
        stim_2_3s = np.sum(stim_2_marble_values == 3)

        stim_1_4s = np.sum(stim_1_marble_values == 4)
        stim_2_4s = np.sum(stim_2_marble_values == 4)

        strictly_better = 0 

        if((stim_1_4s >= stim_2_4s and stim_1_3s > stim_2_3s) or (stim_1_4s > stim_2_4s and stim_1_3s >= stim_2_3s)): # 
            # stim one strictly better 
            strictly_better = 1
        if((stim_2_4s >= stim_1_4s and stim_2_3s > stim_1_3s) or (stim_2_4s > stim_1_4s and stim_2_3s >= stim_1_3s)): # 
            # stim two strictly better 
            if(strictly_better == 1): assert(False) #shouldn't happen
            strictly_better = 2
        
        if(strictly_better != 0):
            num_strictly_better += 1
        else:
            num_subjectively_better += 1
"""

response_accuracy = []
response_inaccuracy = []

response_speed = []

utility_diffs = []
bad_participants = []
good_participants = []

total_points = []

for participant_id, data_path in enumerate(data_paths):
    #print(data_path)
    data = pd.read_csv(data_path)
    paricipant_correct_responses = 0
    paricipant_incorrect_responses = 0
    cumulative_regret = 0

    experiment_data = data.tail(180).to_numpy()
    participant_points = 0

    change_detection = []

    
    for response in experiment_data:
        """if np.array([value != value for value in response]).any(): 
            cumulative_regret += 1
            continue """

        reward = response[2]
        if(reward == reward): participant_points += reward
        
        stim_1 = int(response[6])
        stim_2 = int(response[7])
        key_press = response[1]
        correct = None 
        if(key_press == 'd' or key_press == 'f'):
            stim_1_mean_utility = stimuli_mean_utilities[stim_1]
            stim_2_mean_utility = stimuli_mean_utilities[stim_2]

            stim_1_3s = np.sum(stimuli_marble_values[stim_1] == 3)
            stim_2_3s = np.sum(stimuli_marble_values[stim_2] == 3)

            stim_1_4s = np.sum(stimuli_marble_values[stim_1] == 4)
            stim_2_4s = np.sum(stimuli_marble_values[stim_2] == 4)

            strictly_better = 0 

            if((stim_1_4s >= stim_2_4s and stim_1_3s > stim_2_3s) or (stim_1_4s > stim_2_4s and stim_1_3s >= stim_2_3s)):
                # stim one strictly better 
                strictly_better = 1
            if((stim_2_4s >= stim_1_4s and stim_2_3s > stim_1_3s) or (stim_2_4s > stim_1_4s and stim_2_3s >= stim_1_3s)):
                # stim two strictly better 
                if(strictly_better == 1): assert(False) #shouldn't happen
                strictly_better = 2

            # check if strictly better 
            if(strictly_better != 0):
                correct = 1 if ((strictly_better == 1 and key_press == 'd') or (strictly_better == 2 and key_press == 'f')) else 0
            
            if(key_press == 'd'):
                utility_diff = stim_1_mean_utility - stim_2_mean_utility
            
            if(key_press == 'f'):
                utility_diff = stim_1_mean_utility - stim_2_mean_utility 

            #correct = 1 if ((stim_1_mean_utility >= stim_2_mean_utility and key_press == 'd') or (stim_1_mean_utility <= stim_2_mean_utility and key_press == 'f')) else 0

            #if(utility_diff < 0.25 and utility_diff > -0.25):
            #    correct = 1 
            
            if not correct:
                cumulative_regret += np.abs(utility_diff)

            if(strictly_better != 0):
                if(correct == 1):
                    paricipant_correct_responses += 1
                else:
                    paricipant_incorrect_responses += 1
            #else:
            #    paricipant_correct_responses += 1
        
        #if(response[3] == 1.0 and response[1] != response[1]):
        #    paricipant_incorrect_responses += 1

    #if((paricipant_correct_responses + paricipant_incorrect_responses) == 0 or paricipant_correct_responses / (paricipant_correct_responses + paricipant_incorrect_responses) < 0.61):
    #response_accuracy.append(paricipant_correct_responses) 
    #response_accuracy.append(participant_points) 
    if((paricipant_correct_responses + paricipant_incorrect_responses) > 0):
        response_accuracy.append(paricipant_correct_responses / (paricipant_correct_responses + paricipant_incorrect_responses)) 
        #response_accuracy.append(cumulative_regret)
        if(paricipant_correct_responses / (paricipant_correct_responses + paricipant_incorrect_responses) <= 0.6):
            bad_participants.append(data_path)
        else:
            good_participants.append(data_path)
    else:
        response_accuracy.append(0)
        #response_accuracy.append(999)


print(good_participants)
response_accuracy = np.array(response_accuracy)
#response_accuracy = response_accuracy[(response_accuracy < 999)]
#response_accuracy = response_accuracy[(response_accuracy < 200)]

#print("number of participants with low cumulative regret: ", len(response_accuracy))

response_accuracy = np.array(response_accuracy)

decent_response_accuracy = response_accuracy[(response_accuracy > 0.6)]
print("number of participants with more than 68 percent change detection: ", len(decent_response_accuracy), " out of ", len(response_accuracy), " total participants")

fig, ax = plt.subplots(figsize=(8,3))
_, bins, _ = plt.hist(decent_response_accuracy, 10, histtype = 'bar', density=True)


mu, sigma = scipy.stats.norm.fit(decent_response_accuracy)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
#print(mu, sigma, best_fit_line)
plt.plot(bins, best_fit_line, axes=ax)

#plt.xticks([0, 30, 68, 120, 150])
#plt.xticks([0.64, 0.75, 0.9])
plt.title("Percent Strictly Correct Utility Choices")
plt.show()

