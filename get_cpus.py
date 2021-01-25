from __future__ import division, print_function
import math
import numpy as np
#import pandas as pd
import time
import sys
import ray


ray.init(address="auto")#, _redis_password='5241590000000000')
margin = .2
N = 10**5
reps =  3*10**5
num_workers = int(ray.cluster_resources()['CPU'])
print("Found {} workers".format(num_workers))

