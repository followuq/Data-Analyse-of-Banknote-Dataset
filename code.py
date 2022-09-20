import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from sklearn.cluster import KMeans

dataset = pd.read_csv('Banknote Dataset.csv')
xyz = np.array(dataset)

mean = np.mean(xyz, 0)
mode = stats.mode(xyz)
percentile = np.percentile(xyz, 1)
rng = np.ptp(xyz)
standart_dvtn = np.std(xyz, 0)
stndrt_variation = np.var(xyz)
xy = np.column_stack((xyz[:, 0], xyz[:, 1]))

km_res = KMeans(n_clusters=2).fit(xy)
clusters = km_res.cluster_centers_

ellipse = patches.Ellipse((mean[0], mean[1]), standart_dvtn[0] * 2, standart_dvtn[1] * 2, edgecolor='firebrick', linestyle='--', fill=False)

fig, graph = plt.subplots()

graph.scatter(xyz[:, 0], xyz[:, 1], marker='|', alpha=0.25, color=[*['red']*686, *['green']*686])
graph.scatter(mean[0], mean[1], alpha=0.50)
graph.scatter(mode[0], mode[1], alpha=0.50)
graph.scatter(clusters[:, 0], clusters[:,1], c='purple', alpha=0.50)
graph.add_patch(ellipse)
