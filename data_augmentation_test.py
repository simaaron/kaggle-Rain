# -*- coding: utf-8 -*-
"""
@author: Aaron Sim
Kaggle competition: How Much Did It Rain II

'Dropin' augmentation of test data
"""
import gc
import numpy as np
import pandas as pd

INPUT_WIDTH = 19 # Any length >= 19, which is the max no. of time obs per hour
NUM_RAND = 61 # No. of augmented samples
COLUMNS = ['Id','minutes_past', 'radardist_km', 'Ref', 'Ref_5x5_10th',
       'Ref_5x5_50th', 'Ref_5x5_90th', 'RefComposite',
       'RefComposite_5x5_10th', 'RefComposite_5x5_50th',
       'RefComposite_5x5_90th', 'RhoHV', 'RhoHV_5x5_10th',
       'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr', 'Zdr_5x5_10th',
       'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp', 'Kdp_5x5_10th',
       'Kdp_5x5_50th', 'Kdp_5x5_90th']
       
       
####### 1. Define 'dropin' augmentation function #######
def extend_series(X, rng, target_len=19):
    """Augment time series to a fixed length by duplicating vectors
    Args:
        X (2D ndarray): Sequence of radar features in a single hour
        rng (numpy RandomState object): random number generator
        target_len (int): fixed target length of the sequence
    Returns:
        the augmented sequence
    """
    curr_len = X.shape[0]
    extra_needed = target_len-curr_len
    if (extra_needed > 0):
        reps = [1]*(curr_len)
        add_ind = rng.randint(0, curr_len, size=extra_needed)
        
        new_reps = [np.sum(add_ind==j) for j in xrange(curr_len)]
        new_reps = np.array(reps) + np.array(new_reps)
        X = np.repeat(X, new_reps, axis=0)
    return X

####### 2. Create random seeds #######
# Any lists would do...
rng_seed_list1 = [234561, 23451, 2341, 231, 21, 678901, 67891, 6781, 671, 16,
                  77177]
rng_seed_list2 = range(9725, 9727+50*7, 7)
rng_seed_list3 = range(9726, 9728+50*7, 7)
rng_seed_list = rng_seed_list1 + rng_seed_list2 + rng_seed_list3
assert len(rng_seed_list) >= NUM_RAND

####### 3. Augment training data #######
data = np.load("./data/processed_test.npy")
obs_ids_all = np.load("./test/obs_ids_test.npy")

data_pd = pd.DataFrame(data=data[:,0:], columns=COLUMNS)
data_pd_ids_all = np.array(data_pd['Id'])
data_pd_ids_selected = np.in1d(data_pd_ids_all, obs_ids_all)
data_pd_filtered = data_pd[data_pd_ids_selected]

data_pd_gp = pd.groupby(data_pd_filtered, "Id")
data_size = len(data_pd_gp)

for jj, rng_seed in enumerate(rng_seed_list[0:NUM_RAND]):
    rng = np.random.RandomState(rng_seed) 
    output = np.empty((data_size, INPUT_WIDTH, 22))
    
    i = 0
    for _, group in data_pd_gp:
        group_array = np.array(group)
        X = extend_series(group_array[:,1:23], rng, target_len=INPUT_WIDTH) 
        output[i,:,:] = X[:,:]
        i += 1
        
    print "X.shape", X.shape
    print "output.shape", output.shape
    
    np.save("./test/data_test_augmented_t%s_rand%s.npy" %
            (INPUT_WIDTH, jj), output)
        
    gc.collect()
    
    

