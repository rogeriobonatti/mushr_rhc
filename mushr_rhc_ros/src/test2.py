# from torchsummary import summary
import sys
import os
import signal
import threading
import random
import numpy as np
from queue import Queue
import time
from collections import OrderedDict
import matplotlib.pyplot as plt

num_samples = 10000
results = np.zeros(shape=(num_samples,))

max_seq_length = 100
for i in range(num_samples):
    episode_path_length = 0.0
    iter = 0
    while iter < max_seq_length:
        result = np.random.random() < 0.99
        if result:
            episode_path_length+= 1.0
        else:
            break
        iter += 1
    results[i] = episode_path_length

plt.hist(results, bins=30, facecolor='green', alpha=0.75)