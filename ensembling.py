# -*- coding: utf-8 -*-
"""
@author: Aaron Sim
Kaggle competition: How Much Did It Rain II

Ensembling predictions from different augmentations of the test dataset
"""
import os
import sys
import numpy as np
import pandas as pd

# default values
dataver = 2
num_rand = 61

## User inputs
for i in range(1,len(sys.argv)):
    if sys.argv[i].startswith('-'):
        option = sys.argv[i][1:]
        if option[0:2] == 'v=':
            dataver = int(option[2:])
        elif option[0:3] == 'nr=':
            num_rand = int(option[3:])

print("num_rand: %d" % (num_rand))
            
output_folder = "output_cv0_v%s" % (dataver)
os.chdir(output_folder)
filename = "submission_v%s_rand0.csv" % (dataver)
ens_output = pd.read_csv(filename)         
   
for rand_ver in xrange(1,num_rand):
    filename = "submission_v%s_rand%s.csv" % (dataver, rand_ver)
    ind_output = pd.read_csv(filename)        
    ens_output = pd.concat([ens_output, ind_output])
    
ens_groups = ens_output.groupby('Id')
#ens = ens_groups.median()
ens = ens_groups.mean()

ens = np.round(ens/0.254)*0.254

#output_filename = "ens_submission_v1_%save_median.csv" % (num_rand)
output_filename = "ens_submission_v%s_%save_mean.csv" % (dataver, num_rand)
ens.to_csv(output_filename, index_label='Id')




