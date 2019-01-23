import matplotlib.pyplot as plt
import csv

episode = []
avg = []
with open('example.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        episode.append(int(row[2]))
        avg.append(float(row[12]))
plt.plot(episode,avg, label='Reward over episode')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Reward over episodes')
plt.legend()
plt.show()