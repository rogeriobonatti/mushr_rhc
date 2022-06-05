import numpy as np
import matplotlib.pyplot as plt

# file_path = '/home/azureuser/hackathon_data/e2e_eval/GPTiros_e2e_8gpu_2022-02-17_v2/info.csv'
file_paths = ['/home/azureuser/hackathon_data_premium/e2e_eval/6L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/6L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/6L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/6L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/6L1p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/12L1p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval/24L1p0/info.csv']

for file_path in file_paths:

    data = np.genfromtxt(file_path, delimiter=',')

    distances = data[:,0]
    times = data[:,1]

    # clear data that crashes immediately
    min_time = 0.0
    distances = distances[times>min_time]
    times = times[times>min_time]

    plt.hist(distances, bins=30, facecolor='green', alpha=0.75)
    # plt.savefig('/home/azureuser/hackathon_data/e2e_eval/GPTiros_e2e_8gpu_2022-02-17_v2/fig.png')
    # plt.savefig('/home/azureuser/hackathon_data/e2e_eval/model_test/fig.png')
    plt.show()
    plt.clf()
