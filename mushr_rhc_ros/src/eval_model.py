import numpy as np
import matplotlib.pyplot as plt

# file_path = '/home/azureuser/hackathon_data/e2e_eval/GPTiros_e2e_8gpu_2022-02-17_v2/info.csv'
file_path = '/home/azureuser/hackathon_data/e2e_eval/model_test/info.csv'
data = np.genfromtxt(file_path, delimiter=',')

distances = data[:,0]
times = data[:,1]

# clear data that crashes immediately
min_time = 3.0
distances = distances[times>min_time]
times = times[times>min_time]

plt.hist(distances, bins=30, facecolor='green', alpha=0.75)
# plt.savefig('/home/azureuser/hackathon_data/e2e_eval/GPTiros_e2e_8gpu_2022-02-17_v2/fig.png')
plt.savefig('/home/azureuser/hackathon_data/e2e_eval/model_test/fig.png')
plt.show()
