import pandas as pd
import matplotlib.pyplot as plt
import json
from sys import argv


with open(argv[1], 'r') as f:
   reward = json.load(f)

reward = reward['reward']

avg = 10
window = int(len(reward)/avg)
fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=[9,9]);
rolling_mean = pd.Series(reward).rolling(window).mean()
std = pd.Series(reward).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(reward)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
ax1.set_title('Average Reward (last {}-episode window)'.format(avg))
ax1.set_xlabel('Episode'); 
ax1.set_ylabel('Average Reward')
ax2.plot(reward)
ax2.set_title('Average Reward (all)')
ax2.set_xlabel('Episode'); 
ax2.set_ylabel('Average Reward')
fig.tight_layout(pad=2)
plt.show()
