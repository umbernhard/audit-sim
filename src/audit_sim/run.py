# Code borrowed from Dan wallach's arlo-e2e package
from __future__ import division, print_function
import math
import numpy as np
#import pandas as pd
import time
import sys
import ray
import os

print(os.getcwd())
print(sys.path)
from audit_sim.ray_progress import ProgressBar


ray.init(address="auto")#, _redis_password='5241590000000000')
margin = .2
N = 10**4
reps =  3*10**2
num_workers = int(ray.cluster_resources()['CPU'])
print("Found {} workers".format(num_workers))
reps_per_worker = reps/num_workers


progressbar = (
    ProgressBar(
        {
            "Ballots": N,
            "Iterations": 0,
            "Batch": 0,
        }
    )
)
progressbar_actor = progressbar.actor


def get_kk_estimate(x, margin, N, alpha):
    """
    A function that runs a kaplan-kolmogorov audit on an election specified by vector x.
    """
    g = 50
    t = 1/2
    xp = x + g

    num = 1 - (np.arange(len(xp)))/N
    den = (t + g) - ((np.cumsum(xp) - xp)/N) # subtracting xp here makes it the cumulative sum from 1 to k - 1.

    y = num/den

    # set the first item in z properly:
    np.insert(y, 0, (x[0]+g)/(t+g) if t > 0 else 1)
    z = np.cumprod(xp*y)

    return np.argmax(1/z<=alpha)


@ray.remote
def simulate_audits(seed, reps, margin, N, alpha, progress_bar):
    """
    A function that simulates various audits of <reps> elections with margin
    <margin> and <N> total ballots.
    """
    prng = np.random.RandomState(seed)
    indices = []
    for i in range(reps):
      x = (prng.random(size=N) <= 1/2 + margin/2)
      indices.append(get_kk_estimate(x, margin, N, alpha))

    return indices

prng = np.random.RandomState(1234567890)
seeds = [prng.randint(2**32 - 1) for i in range(num_workers)]

alpha = 0.1
start = time.time()

sample_sizes = ray.get([simulate_audits.remote(seed, reps, margin, N, alpha, progressbar_actor) for seed in seeds])
print(len(sample_sizes))
print(len(sample_sizes[0]))


quantiles = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9]
print(np.quantile(sample_sizes[0], quantiles))
#fraction_table = pd.DataFrame(columns=["Test Statistic"] + quantiles)
#
#raw_table = pd.DataFrame(columns=["Test Statistic"] + quantiles)
#sample_size = {}
#
#for compute_func in compute_funcs:
#  fraction_table.loc[len(fraction_table)] = [compute_func.__name__] + ['{:3.2f}%'.format(100*q/N) for q in np.quantile([i.result() for i in results[compute_func.__name__]], quantiles)]
#  raw_table.loc[len(raw_table)] = [compute_func.__name__] + ['{:,d}'.format(int(q)) for q in np.quantile([i. result() for i in results[compute_func.__name__]], quantiles)]

end = time.time()
print('Took {:0.2f}s'.format(end - start))

#display(fraction_table)
#
#display(raw_table)
