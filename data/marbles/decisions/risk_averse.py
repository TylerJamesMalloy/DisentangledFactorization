
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

data_paths = glob.glob("data/*")
all_marble_colors = pd.read_csv("../source/colors.csv")
all_marble_colors = all_marble_colors['colors']

stimuli_mean_utilities = []
stimuli_deviations = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))


bad_participants = ['1206472879_20220607.csv', '1481432285_20220607.csv', '1624730318_20220623.csv', '169075273_20220607.csv', '1951250462_20220623.csv', '282458648_20220622.csv', '2843103640_20220602.csv', '302208162_20220607.csv', '3130983426_20220602.csv', '3310926888_20220622.csv', '3525719084_20220607.csv', '3597899139_20220622.csv', '4148888000_20220622.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4512223036_20220602.csv', '4717805082_20220622.csv', '4737559307_20220623.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5250442910_20220623.csv', '5878990705_20220607.csv', '6174180168_20220602.csv', '6247410167_20220607.csv', '6969137467_20220622.csv', '7056217438_20220622.csv', '748797646_20220518.csv', '7729591288_20220607.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8805493182_20220622.csv', '8894686670_20220622.csv', '9177013872_20220518.csv', '9796178986_20220623.csv', '9824877929_20220518.csv']


good_participants = []
riskAversion = pd.DataFrame()
allDataFrame = pd.DataFrame()
for participant_id, data_path in enumerate(data_paths):
    #print(participant_id, data_path)
    dataFrame = pd.DataFrame()
    data = pd.read_csv(data_path) 
    data['pid'] = data_path.split("\\")[1].split("_")[0]
    dataFrame = dataFrame.append(data, ignore_index=True)
    allDataFrame = allDataFrame.append(dataFrame, ignore_index=True)

    filename = data_path.split("\\")[1]
    if(filename not in bad_participants):
        good_participants.append(participant_id)
    
    #print(np.array(data['rt']) == np.NaN)
    #print("NaN values is: ", np.sum(np.isnan(np.array(data['rt']))))
    # type 1: utility prediction, type 0: change detection 
    dataFrame = dataFrame[dataFrame['trial_num'] > 40]
    change_detection = dataFrame[dataFrame['type'] == 0]
    utility_prediction = dataFrame[dataFrame['type'] == 1]

    last5trials = dataFrame[dataFrame['trial_num'] > 140]
    #if(np.sum(np.isnan(np.array(last5trials['rt']))) <= 20):
    #    good_participants.append(participant_id)

    stims_1 = utility_prediction['stim_1']
    stims_2 = utility_prediction['stim_2']
    key_presses = np.array(utility_prediction['key_press'])

    negative_bias = []
    for trial_num, (stim_1, stim_2) in enumerate(zip(stims_1, stims_2)):
        if(np.isnan(stim_1)): continue 

        stim_1_marbles = np.array(ast.literal_eval(all_marble_colors[int(stim_1)]))
        stim_1_zeros = np.count_nonzero(stim_1_marbles == 0)
        stim_1_ones = np.count_nonzero(stim_1_marbles == 1)
        stim_1_twos = np.count_nonzero(stim_1_marbles == 2)

        stim_2_marbles = np.array(ast.literal_eval(all_marble_colors[int(stim_2)]))
        stim_2_zeros = np.count_nonzero(stim_2_marbles == 0)
        stim_2_ones = np.count_nonzero(stim_2_marbles == 1)
        stim_2_twos = np.count_nonzero(stim_2_marbles == 2)

        stim_1_mean_util = ((stim_1_zeros * 4) + (stim_1_ones * 3) + (stim_1_twos * 2)) / 9
        stim_2_mean_util = ((stim_2_zeros * 4) + (stim_2_ones * 3) + (stim_2_twos * 2)) / 9

        stim_2_marbles[stim_2_marbles == 0] = 4
        stim_2_marbles[stim_2_marbles == 1] = 3
        stim_2_marbles[stim_2_marbles == 2] = 2

        stim_1_marbles[stim_1_marbles == 0] = 4
        stim_1_marbles[stim_1_marbles == 1] = 3
        stim_1_marbles[stim_1_marbles == 2] = 2
        
        #print(trial_num)
        choice = key_presses[trial_num] # d = stim_1, f = stim_2 
        if(choice == 'd'):
            chosen_stim = stim_1_mean_util
            unchosen_stim = stim_2_mean_util
            chosen_std = np.std(stim_1_marbles)
            unchosen_std = np.std(stim_2_marbles)
        else:
            chosen_stim = stim_2_mean_util
            unchosen_stim = stim_1_mean_util
            chosen_std = np.std(stim_2_marbles)
            unchosen_std = np.std(stim_1_marbles)
        
        riskData = {"Participant Data Path": data_path, "Participant ID": participant_id, "STD Difference": chosen_std - unchosen_std, "Utility Difference": chosen_stim - unchosen_stim, "Chosen Utility":chosen_stim, "Unchosen Utility":unchosen_stim, "Chosen STD":chosen_std, "Unchosen STD":unchosen_std}
        riskAversion = riskAversion.append(riskData, ignore_index=True)
        #print("Chose stimulus with value: ", chosen_stim, " over value ", unchosen_stim)

x = riskAversion["STD Difference"]
y = riskAversion["Utility Difference"]
res = stat.linregress(x, y)
print(res)

#riskAversion = riskAversion.loc[riskAversion['STD Difference'] > -0.4 ]
#riskAversion = riskAversion.loc[riskAversion['STD Difference'] < 0.4 ]

good_res = []
for i in good_participants:
    participant_data = riskAversion.loc[riskAversion['Participant ID'] == i]

    x = participant_data["STD Difference"]
    y = participant_data["Utility Difference"]
    res = stat.linregress(x, y)

    good_res.append([res[0] ,res[1]])

print(good_res)
riskAversion.to_csv("./RiskAverseData")

# only look at good players 
riskAversion = riskAversion.loc[riskAversion['Participant ID'].isin(good_participants)]

#riskAversion = riskAversion[(riskAversion["Utility Difference"] > -0.5) and (riskAversion["Utility Difference"] < 0.5)]
#print("min is", np.min(riskAversion['Utility Difference']))

x = riskAversion["STD Difference"]
y = riskAversion["Utility Difference"]
res = stat.linregress(x, y)
print(res)
sns.set_style("darkgrid")
p = sns.lineplot(data=riskAversion, x="STD Difference", y="Utility Difference", label='Marble Jar Decisions')

p.set_xlabel("Utility Standard Deviation Difference (C-U)", fontsize = 20)
p.set_ylabel("Utility Mean Difference (C-U)", fontsize = 20)

plt.annotate("r = 0.7110, p < 0.0005", (0, 1))

plt.plot(x, res.intercept + res.slope*x, 'r', label='Linear Fit')

p.set_title("Marble Jar Risk Aversion Effect", fontsize = 24)
plt.show()