# -*- coding: utf-8 -*-
"""
@author: Aaron Sim
Kaggle competition: How Much Did It Rain II

Data preprocessing: 1. Replaces NaN with zeros, 2. Excludes outliers,
    3. Create validation holdout set
"""
import os
import numpy as np
import pandas as pd
from sklearn import cross_validation

THRESHOLD = 73 
N_FOLDS = 21
RND_SEED = 56

####### 1. Import training data and extract ids #######
train_raw = pd.read_csv("./data/train.csv")
raw_ids_all = train_raw["Id"]
raw_ids = raw_ids_all.unique()

####### 2. Remove ids with only NaNs in the "Ref" column #######
train_raw_tmp = train_raw[~np.isnan(train_raw.Ref)]
raw_ids_tmp = train_raw_tmp["Id"].unique()
train_new = train_raw[np.in1d(raw_ids_all, raw_ids_tmp)]

####### 3. Convert all NaN to zero #######
train_new = train_new.fillna(0.0)
train_new = train_new.reset_index(drop=True)

####### 4. Define and exclude outliers from training set #######
train_new_group = train_new.groupby('Id')
df = pd.DataFrame(train_new_group['Expected'].mean()) # mean, or any value
meaningful_ids = np.array(df[df['Expected'] < THRESHOLD].index)

####### 5. Split off holdout validation subset #######
# Count the no. of observations per hour for each gauge reading
train_new_ids_all = train_new["Id"]
obs_freq = train_new_ids_all.value_counts(ascending=True)
obs_bins = obs_freq.unique()
obs_num = ([(obs_freq==i).sum() for i in obs_bins])
obs_ids = [np.array(obs_freq.index[obs_freq.values==i]) for i in obs_bins]

# Construct stratified c.v. holdout set w.r.t. no. observations per hour
y = np.array(obs_freq)
X = np.concatenate(obs_ids)

rng = np.random.RandomState(RND_SEED)
skf = cross_validation.StratifiedKFold(y, n_folds=N_FOLDS, shuffle=True,
                                       random_state=rng) 

X_train_list = []
X_valid_list = []

cv = 0
for train_index, valid_index in skf:
    X_train, X_valid = X[train_index], X[valid_index]
    print("train.shape before: %s" % (X_train.shape))
    X_train = X_train[np.in1d(X_train, meaningful_ids)]
    
    X_train_list.append(X_train)
    X_valid_list.append(X_valid)
    print("train.shape after: %s" % (X_train.shape))
    print("valid.shape: %s" % (X_valid.shape))
    
    cv += 1
    break # remove if full n-fold cross-validation is desired

np.save("./data/processed_train", np.array(train_new))

####### 5. Save the partitioned IDs into folders #######
if not os.path.exists("train"):
    os.makedirs("train")
if not os.path.exists("valid"):
    os.makedirs("valid")
if not os.path.exists("test"):
    os.makedirs("test")
    
for i, item in enumerate(X_train_list):
    np.save("./train/obs_ids_train_cv%s" % (i), item)

for i, item in enumerate(X_valid_list):
    np.save("./valid/obs_ids_valid_cv%s" % (i), item)

####### 6. Preprocess the test data #######
test_raw = pd.read_csv("./data/test.csv")
test_raw_ids_all = test_raw["Id"]
test_raw_ids = np.array(test_raw_ids_all.unique())

# Convert all NaNs to zero
test_new = test_raw.fillna(0.0)
test_new = test_new.reset_index(drop=True)

np.save("./data/processed_test", np.array(test_new))
np.save("./test/obs_ids_test", test_raw_ids)












