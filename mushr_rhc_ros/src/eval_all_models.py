from dis import dis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# file_path = '/home/azureuser/hackathon_data/e2e_eval/GPTiros_e2e_8gpu_2022-02-17_v2/info.csv'
# file_paths = ['/home/azureuser/hackathon_data_premium/e2e_eval/6L0p0/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p0/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p0/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/6L0p01/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p01/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p01/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/6L0p1/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p1/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p1/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/6L0p5/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/12L0p5/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/24L0p5/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/6L1p0/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/12L1p0/info.csv',
#               '/home/azureuser/hackathon_data_premium/e2e_eval/24L1p0/info.csv']

file_paths = ['/home/azureuser/hackathon_data_premium/e2e_eval_models4/3L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/6L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/12L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/24L0p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/3L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/6L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/12L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/24L0p01/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/3L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/6L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/12L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/24L0p1/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/3L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/6L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/12L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/24L0p5/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/3L1p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/6L1p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/12L1p0/info.csv',
              '/home/azureuser/hackathon_data_premium/e2e_eval_models4/24L1p0/info.csv']

all_vals_mean = np.zeros(shape=(4*5))
all_vals_median = np.zeros(shape=(4*5))

for i, file_path in enumerate(file_paths):

    data = np.genfromtxt(file_path, delimiter=',')

    distances = data[:,0]
    times = data[:,1]

    # distances = data[100:,0]
    # times = data[100:,1]

    # clear data that crashes immediately
    min_time = 10.0
    condition = times>min_time
    distances = distances[condition]
    times = times[condition]

    # clear data above episode max
    max_dist = 1000
    condition = distances<max_dist
    distances = distances[condition]
    times = times[condition]

    plt.hist(distances, bins=30, facecolor='green', alpha=0.75)
    # plt.savefig('/home/azureuser/hackathon_data/e2e_eval/GPTiros_e2e_8gpu_2022-02-17_v2/fig.png')
    plt.savefig('/home/azureuser/hackathon_data_premium/e2e_eval_models4/model_test/fig{}.png'.format(str(i)))
    plt.show()
    plt.clf()

    if distances.shape[0]>0:
        all_vals_mean[i] = np.mean(distances)
        all_vals_median[i] = np.median(distances)
    else:
        all_vals_mean[i] = 0.
        all_vals_median[i] = 0.

    # print the stats:
    print("SIZE: {} | AVG: {} | MED: {} | PATH: {}".format(distances.shape[0], np.mean(distances), np.median(distances), file_path))

# 'Num tokens': ['540', '30K', '300K', '1.5M', '3M']

all_vals_mean[:4] = [6.60,6.63,6.15,6.33]
all_vals_median[:4] = [6.60,6.63,6.15,6.33]

data = {'3L':  all_vals_mean[::4],
        '6L':  all_vals_mean[1::4],
        '12L': all_vals_mean[2::4],
        '24L': all_vals_mean[3::4],
        # 'Dataset fraction': [0.0, 0.01, 0.1, 0.5, 1.0],
        'Number of Tokens Processed': ['540', '30K', '300K', '1.5M', '3M']}

# data = {'3L':  all_vals_median[::4],
#         '6L':  all_vals_median[1::4],
#         '12L': all_vals_median[2::4],
#         '24L': all_vals_median[3::4],
#         # 'Dataset fraction': [0.0, 0.01, 0.1, 0.5, 1.0],
#         'Dataset fraction': ['540', '30K', '300K', '1.5M', '3M']}

# sns.lineplot(data=data, x="Dataset fraction", y=['6L', '12L', '24L'])

df = pd.DataFrame(data)
print(df)
dfm = df.melt('Number of Tokens Processed', var_name='cols', value_name='Average meters traveled [m]')
sns.catplot(x="Number of Tokens Processed", y="Average meters traveled [m]", hue='cols', data=dfm, kind='point')
plt.grid()
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('/home/azureuser/hackathon_data_premium/e2e_eval_models4/model_test/all_plots.png')