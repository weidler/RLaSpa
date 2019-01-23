import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


episode = []
avg = []
with open('JanusPixel_Race_1m_tunnel.txt','r') as f:
    data = f.readlines()
    for row in data:
        split_row = row.split(" ")
        episode.append(int(split_row[2]))
        avg.append(float(split_row[12]))
episode_tunnel = np.array(episode) - 1000000
avg_tunnel = np.array(avg)

episode = []
avg = []
with open('JanusPixel_Race_735k_scroll.txt','r') as f:
    data = f.readlines()
    for row in data:
        split_row = row.split(" ")
        episode.append(int(split_row[2]))
        avg.append(float(split_row[12]))
episode_scrollers = np.array(episode) - 735000
avg_scrollers = np.array(avg)


intervals = np.arange(0,100,5)*500
plt.plot(episode_tunnel, savgol_filter(avg_tunnel, 13, 1), label='Trained on tunnel')
plt.plot(episode_scrollers, savgol_filter(avg_scrollers, 13, 1), label='Trained on scrollers')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Reward over episodes')
plt.legend()
plt.show()