# -*- coding: utf-8 -*-
"""
@author: Aaron Sim
Kaggle competition: How Much Did It Rain II

Regression model v1 predictions
"""
import gc
import os
import sys
import lasagne.layers as LL
import pandas as pd
import numpy as np
import theano
import theano.tensor as T

from NN_architectures import build_1Dregression_v1


############################### Main ################################
def do_prediction(test_batch_multiple=5741, # No. of minibatches per batch
                  test_minibatch_size=125,
                  init_file=None,
                  input_width=19,
                  cross_val=0, # Cross-validation subset label
                  dataver=1): # Label for different runs/architectures/etc
    
    ###################################################
    ################# 0. User inputs ##################
    ###################################################
    rand_ver = 0 
    for i in range(1,len(sys.argv)):
        if sys.argv[i].startswith('-'):
            option = sys.argv[i][1:]
            if option == 'i': init_file = sys.argv[i+1]
            elif option[0:2] == 'v=' : dataver = int(option[2:])
            elif option[0:3] == 'cv=' : cross_val = int(option[3:])
            elif option[0:3] == 'rd=' : rand_ver = int(option[3:])
                                
    print("Running with dataver %s" % (dataver))   
    print("Running with cross_val %s" % (cross_val))   
    
    ###################################################
    ############# 1. Housekeeping values ##############
    ###################################################
    test_batch_size = test_batch_multiple*test_minibatch_size
    submission_file = "submission_v%s_rand%s.csv" % (dataver, rand_ver)  
    
    ###################################################
    ###### 2. Define model and theano variables #######
    ###################################################
    print("Defining variables...")
    index = T.lscalar() # Minibatch index
    x = T.tensor3('x') # Inputs 
    
    print("Defining model...")
    network_0 = build_1Dregression_v1(
                        input_var=x,
                        input_width=input_width,
                        nin_units=12,
                        h_num_units=[64,128,256,128,64],
                        h_grad_clip=1.0,
                        output_width=1
                        )
    
    print("Setting model parametrs...")
    output_folder_filename = "output_cv%s_v%s" % (cross_val, dataver)
    
    if init_file is not None:
        init_model = np.load(init_file)
        init_params = init_model[init_model.files[0]]           
        LL.set_all_param_values([network_0], init_params)
        
        if not os.path.exists(output_folder_filename):
            os.makedirs(output_folder_filename)
        os.chdir(output_folder_filename)
    else:
        os.chdir(output_folder_filename)
        init_model = np.load("model.npz")
        init_params = init_model[init_model.files[0]]           
        LL.set_all_param_values([network_0], init_params)
        
    ###################################################                                
    ################ 3. Import data ###################
    ###################################################
    ## Loading data generation model parameters
    print("Defining shared variables...")
    test_set_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                               borrow=True)
    
    print("Generating test data...")
    chunk_test_data = np.load(
                        "../test/data_test_augmented_t%s_rand%s.npy" 
                        % (input_width, rand_ver)
                        ).astype(theano.config.floatX)
    chunk_test_id = np.load("../test/obs_ids_test.npy")
    assert test_batch_size == chunk_test_data.shape[0]
    
    print("Assigning test data...")
    test_set_x.set_value(chunk_test_data.transpose(0,2,1))
    
    ###################################################                                
    ########### 4. Define prediction model ############
    ###################################################  
    print("Defining prediction expression...")
    test_prediction_0 = LL.get_output(network_0, deterministic=True)
    
    print("Defining theano functions...")
    test_model = theano.function(
        [index],
        test_prediction_0,
        givens={
            x: test_set_x[(index*test_minibatch_size):
                            ((index+1)*test_minibatch_size)],
        }
    )    
    
    ###################################################                                
    ############## 7. Begin predicting  ###############
    ###################################################  
    print("Begin predicting...")
    this_test_prediction= np.concatenate([test_model(i) for i in 
                                            xrange(test_batch_multiple)])

    ###################################################                                
    ################# 8. Save files  ##################
    ###################################################  
    submission = pd.DataFrame(data=this_test_prediction,
                              index=np.array(chunk_test_id),
                              columns=["Expected"])
    submission.to_csv(submission_file, index_label='Id')

    del test_set_x
    gc.collect()
    
    return None
    

if __name__ == '__main__':
    do_prediction()     
            
                
            
            
            











