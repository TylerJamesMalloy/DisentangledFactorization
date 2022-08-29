import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
 
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stat
import matplotlib.pyplot as plt
import glob
import ast
 
 
"""
 
"""
 
data_paths = glob.glob("new/*")
all_marble_colors = pd.read_csv("../source/colors.csv")
all_marble_colors = all_marble_colors['colors']
 
riskAversion = pd.DataFrame()
 
all_bonuses = []
all_close = []
 
 
allDataFrame = pd.DataFrame()
for participant_id, data_path in enumerate(data_paths):
    dataFrame = pd.DataFrame()
    data = pd.read_csv(data_path)
    data['pid'] = data_path.split("\\")[0].split("_")[0]
    dataFrame = dataFrame.append(data, ignore_index=True)
    allDataFrame = allDataFrame.append(dataFrame, ignore_index=True)

    id = data_path.split("\\")[0].split("_")[0]
    # type 1: utility prediction, type 0: change detection
    trialBlocks = dataFrame.tail(200) #[dataFrame['trial_num'] > 40]

    rts = trialBlocks['rt'].to_numpy()
    rts = rts[~np.isnan(rts)]
    mean_rt = -1

    try:
        mean_rt = np.mean(rts)
    except:
        mean_rt = -1

    block_points = 0
    participant_bonus = 0
    participant_close = 0
    for trial_num, reward in enumerate(trialBlocks['reward'].to_numpy()):
        if(reward != -1): block_points += (reward / 50) # negative means no response but they don't actually lose points
        if(trial_num == 199 or (trial_num % 20 == 0 and trial_num > 0)):
            if(block_points >= 1.0): participant_bonus += 0.25
            if(block_points >= 0.8): participant_close += 1/10
            block_points = 0
            
    print(  " with path: ", data_path,
            " gets a bonus of: ", participant_bonus,
            " got close on ", participant_close)
    all_bonuses.append(participant_bonus)
    all_close.append(participant_close)
 
# data2/6857915029_20220607.csv
# data2/2616009388_20220607.csv
 
 
#print(all_close)
print(all_bonuses)
print(data_paths)
print(np.mean(all_bonuses))
 
 
 

