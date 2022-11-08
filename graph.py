import warnings
warnings.filterwarnings('error')
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch 
from torch import optim
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 
from scipy.optimize import curve_fit

data = pd.read_pickle("./representationAnalysis.pkl")

print(data)


fig, axes = plt.subplots(1, 3)

data["Representation Similarity to Neither"] = data["Representation Similarity to Neither"] / 1e8

ax = sns.lineplot(x="Training Block", y="Representation Similarity to Neither", hue="Type", data=data)
axes[2].set_xlabel('Bandit Training Block')
axes[2].set_ylabel('Representation Difference (1e8)') 
axes[2].set_title('Representation Difference by Training Block')

colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]

folders = ["btcvae_celeba_ld5_b256", "btcvae_celeba_ld10_b256", "btcvae_celeba_ld25_b256", "btcvae_celeba_ld50_b256", "btcvae_celeba_ld100_b256"]
labels = ["5", "10", "25", "50", "100"]
for idx, folder in enumerate(folders):
    file = "./results/" + folder + "/train_losses.log"
    data = pd.read_csv(file)

    data = data[data.Loss == "recon_loss"] 

    data.Value = data.Value / 1000

    ax = sns.lineplot(x="Epoch", y="Value", data=data, label=str(labels[idx]), color=colors[idx], ax=axes[0])

axes[0].set_xlabel('Pre-Training Epoch')
axes[0].set_ylabel('Model Reconstruction Loss (1K)') 
axes[0].set_title('Model Reconstruction Loss by Pre-Training Epoch')
legend = axes[0].legend()
legend.remove()


def power_law(x, a, b):
    #return a / np.power(x, -b) # inverse power
    return a / np.exp(b/x) # inverse power
    #return a * np.power(x,b)
    #return a * (np.log(x) / np.log(b))

data = pd.read_pickle("./bandit_results_a0p5e100.pkl")
#data = pd.read_pickle("./bandit_results_a0p5.pkl")
data = data[data.Episode <= 25] # First update has a bias, removed 

latents = data['Latent Size'].unique()


for idx, latent in enumerate(latents):
    latent_data = data[data["Latent Size"] == latent]
    x = latent_data['Episode'].unique()
    y = np.array([])
    e = np.array([])

    for val in x:
        if(val == 1):
            #y = np.append(y, 0.6)
            episode_data_0 = latent_data.loc[latent_data['Episode'] == 0]
            y_0 =  episode_data_0['Predictive Accuracy'].mean()
            episode_data_2 = latent_data.loc[latent_data['Episode'] == 2]
            y_2 =  episode_data_2['Predictive Accuracy'].mean()
            y = np.append(y, (y_0 + y_2)/2)
        else: 
            episode_data = latent_data.loc[latent_data['Episode'] == val]
            y = np.append(y, episode_data['Predictive Accuracy'].mean())
            e = np.append(e, episode_data['Predictive Accuracy'].var())
    
    x += 1

    
    popt, pcov = curve_fit(power_law, x, y, p0=[1, 1], bounds=[[1e-8, 1e-8], [1e20, 1e20]])

    axes[1].plot(x, power_law(x, *popt), label=int(latent), c =colors[idx])
    axes[1].scatter(x, y, c =colors[idx], alpha=0.5)
    #plt.plot(x, y, c =colors[idx], alpha=0.5)


axes[1].set_xlabel('Contextual Bandit Training Episode')
axes[1].set_ylabel('Model Predictive Accuracy') 
axes[1].set_title('Model Predictive Accuracy by Latent Space Size')

axes[1].legend(title='Laten Dims') 
plt.show()


"""

assert(False)

data = pd.read_pickle("./bandit_results_new.pkl")
print(data)
ax = sns.lineplot(x="Episode", y="Utility Prediction Error", hue="Latent Size", data=data)
plt.title("Model Percent of Correct")
plt.show()

assert(False)

"""