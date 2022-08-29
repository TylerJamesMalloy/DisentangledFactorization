import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 
import math 
import scipy.stats as stats
import os 


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

plt.yticks([])

mu = 0
variance = 1.25
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax[0].plot(x, stats.norm.pdf(x, mu, sigma),linewidth=4.0, color="orange")
ax[0].axvline(x=mu, color='orange', linestyle='--')

mu = 1
variance = 1.25
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax[0].plot(x, stats.norm.pdf(x, mu, sigma),linewidth=4.0, color="blue")
ax[0].axvline(x=mu, color='blue', linestyle='--')

mu = -1
variance = 1.25
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax[0].plot(x, stats.norm.pdf(x, mu, sigma),linewidth=4.0, color="green")
ax[0].axvline(x=mu, color='green', linestyle='--')

# Get the two lines from the axes to generate shading
l1 = ax[0].lines[0]
l2 = ax[0].lines[2]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
ax[0].fill_between(x, y, color="red", alpha=0.3)

print(ax[0].lines)
# Get the two lines from the axes to generate shading
l1 = ax[0].lines[0]
l2 = ax[0].lines[4]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
ax[0].fill_between(x, y, color="red", alpha=0.3)

#mu = 0.5
mu = 2.45
variance = 0.5
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y4 = stats.norm.pdf(x, mu, sigma)
ax[1].plot(x, y4, linewidth=4.0, color="blue")
ax[1].axvline(x=mu, color='blue', linestyle='--')

#mu = 2.5
mu = -2.45
variance = 0.5
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y5 = stats.norm.pdf(x, mu, sigma)
ax[1].plot(x, y5,linewidth=4.0, color="green")
ax[1].axvline(x=mu, color='green', linestyle='--')

#mu = -4
mu = 0
variance = 0.5
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y6 = stats.norm.pdf(x, mu, sigma)
ax[1].plot(x, y6, linewidth=4.0, color="orange")
ax[1].axvline(x=mu, color='orange', linestyle='--')

"""# Get the two lines from the axes to generate shading
l1 = ax[1].lines[2]
l2 = ax[1].lines[4]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
ax[1].fill_between(x, y, color="red", alpha=0.3)

print(ax[1].lines)
# Get the two lines from the axes to generate shading
l1 = ax[1].lines[0]
l2 = ax[1].lines[4]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
ax[1].fill_between(x, y, color="red", alpha=0.3)
"""
ax[0].set_title("Overlapping Latent Representations", fontsize=18)
ax[1].set_title("Partitioned Latent Representations", fontsize=18)

ax[0].set_ylim([0, 0.6])
ax[0].set_xticks([-4,4])


plt.show()



assert(False)

# results/bvae_b0_u100_m2/ResponseAccuracy.pkl

data = pd.read_pickle("./results/bvae/ResponseAccuracy_me2_u10000.pkl" )
data = data.loc[data['Model Type'] == "BVAE"]
print(np.mean(data['Accuracy']))

data = pd.read_pickle("./results/cnn/ResponseAccuracy_me2_u10000.pkl" )
data = data.loc[data['Model Type'] == "BVAE"]
print(np.mean(data['Accuracy']))

data = pd.read_pickle("./results/sparse/ResponseAccuracy_me2_u10000.pkl" )
data = data.loc[data['Model Type'] == "BVAE"]
print(np.mean(data['Accuracy']))

data = pd.read_pickle("./results/vae/ResponseAccuracy_me5_u10.0.pkl" )
data = data.loc[data['Model Type'] == "BVAE"]
print(np.mean(data['Accuracy']))

data = pd.read_pickle("./results/vae/ResponseAccuracy_me5_u10.0.pkl" )
data = data.loc[data['Model Type'] == "FRL"]
print(np.mean(data['Accuracy']))

assert(False)
filename = "ResponseAccuracy_me5_u10.pkl"

data = pd.DataFrame()
if(os.path.exists("./results/sparse/" + filename)):
    sparse = pd.read_pickle("./results/sparse/" + filename)
    sparse = sparse.loc[sparse['Model Type'] == "BVAE"]
    sparse["Model Type"] = "Sparse"
    data = data.append(sparse, ignore_index=True)


if(os.path.exists("./results/cnn_64/" + filename)):
    cnn = pd.read_pickle("./results/cnn_64/" + filename)
    cnn = cnn.loc[cnn['Model Type'] == "BVAE"]
    cnn["Model Type"] = "CNN"
    data = data.append(cnn, ignore_index=True)


if(os.path.exists("./results/bvae/" + filename)):
    print("got bvae")
    bvae = pd.read_pickle("./results/bvae/" + filename)
    bvae = bvae.loc[bvae['Model Type'] == "BVAE"]
    bvae["Model Type"] = "BVAE BVAE"
    data = data.append(bvae, ignore_index=True)

    frl = pd.read_pickle("./results/bvae/" + filename)
    frl = frl.loc[frl['Model Type'] == "FRL"]
    frl["Model Type"] = "FRL"
    data = data.append(frl, ignore_index=True)

data = data.loc[data['EpisodeTrial'] < 25]

ax = sns.lineplot(data=data, x="EpisodeTrial", y="Accuracy", hue="Model Type")
ax.set_title('Predictive Accuracy by Episode Trial', fontsize=20)
ax.set_ylabel('Predictive Accuracy', fontsize=16)
ax.set_xlabel('Episode Trial', fontsize=16)
plt.legend(fontsize=14)
plt.show()


data = pd.DataFrame()
vae = pd.read_pickle("./results/bvae/RepresentationOverlap_me10_u10.pkl")
data = data.append(vae, ignore_index=True)
data = data.loc[data['EpisodeTrial'] < 25]

#ax = sns.lineplot(data=data, x="EpisodeTrial", y="Similar KLD", color="orange", label="Similar")
#ax = sns.lineplot(data=data, x="EpisodeTrial", y="Disimilar KLD", color="blue", label="Disimilar")
ax = sns.lineplot(data=data, x="EpisodeTrial", y="Similar Overlap", color="orange", label="Similar Utility Stimuli")
ax1 = sns.lineplot(data=data, x="EpisodeTrial", y="Disimilar Overlap", color="blue", label="Disimilar Utility Stimuli")
#ax = sns.lineplot(data=data, x="EpisodeTrial", y="High Utility Overlap", color="orange", label="High Utility")
#ax = sns.lineplot(data=data, x="EpisodeTrial", y="Low Utility Overlap", color="blue", label="Low Utility")
#plt.legend(title='Overlaps', loc='upper left', labels=['Disimilar Utility', '', 'Similar Utility'])
ax.set_title('Representation Overlap By Episode Trial', fontsize=20)
ax.set_ylabel('Representation Overlap', fontsize=16)
ax.set_xlabel('Episode Trial', fontsize=16)
plt.legend(fontsize=14)
plt.show()


assert(False)