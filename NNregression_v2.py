# -*- coding: utf-8 -*-
"""
@author: Aaron Sim
Kaggle competition: How Much Did It Rain II

Training regression model v2
"""
import gc
import os
import sys
import time
import lasagne
import lasagne.layers as LL
from lasagne.objectives import aggregate
from lasagne.random import set_rng #, get_rng
import numpy as np
import theano
import theano.tensor as T

from NN_architectures import build_1Dregression_v2


############################### Main ################################
def do_regression(num_epochs=60, # No. of epochs to train
                  init_file=None,  # Saved parameters to initialise training
                  epoch_size=680780,  # Whole dataset size
                  valid_size=34848, # Size of validation holdout set
                  train_batch_multiple=10637,  # No. of minibatches per batch
                  valid_batch_multiple=1089,  # No. of minibatches per batch
                  train_minibatch_size=64,
                  valid_minibatch_size=32,
                  eval_multiple=50,  # No. of minibatches to ave. in report
                  save_model=True,
                  input_width=19,
                  rng_seed=100005,
                  cross_val=0,  # Cross-validation subset label
                  dataver=2,  # Label for different runs/architectures/etc
                  rate_init=1.0,
                  rate_decay=0.999983):

    ###################################################
    ################# 0. User inputs ##################
    ###################################################
    for i in range(1,len(sys.argv)):
        if sys.argv[i].startswith('-'):
            option = sys.argv[i][1:]
            if option == 'i': init_file = sys.argv[i+1]
            elif option[0:2] == 'v=' : dataver = int(option[2:])
            elif option[0:3] == 'cv=' : cross_val = int(option[3:])
            elif option[0:3] == 'rs=' : rng_seed = int(option[3:])
            elif option[0:3] == 'ri=' : rate_init = np.float32(option[3:])
            elif option[0:3] == 'rd=' : rate_decay = np.float32(option[3:])
                                
    print("Running with dataver %s" % (dataver))
    print("Running with cross_val %s" % (cross_val))
    
    
    ###################################################
    ############# 1. Housekeeping values ##############
    ###################################################
    # Batch size is possibly not equal to epoch size due to memory limits
    train_batch_size = train_batch_multiple*train_minibatch_size 
    assert epoch_size >= train_batch_size
    
    # Number of times we expect the training/validation generator to be called
    max_train_gen_calls = (num_epochs*epoch_size)//train_batch_size 

    # Number of evaluations (total minibatches / eval_multiple)
    num_eval = max_train_gen_calls*train_batch_multiple / eval_multiple
    
    
    ###################################################
    ###### 2. Define model and theano variables #######
    ###################################################
    if rng_seed is not None:
        print("Setting RandomState with seed=%i" % (rng_seed))
        rng = np.random.RandomState(rng_seed)
        set_rng(rng)
    
    print("Defining variables...")
    index = T.lscalar() # Minibatch index
    x = T.tensor3('x') # Inputs 
    y = T.fvector('y') # Target
    
    print("Defining model...")
    network_0 = build_1Dregression_v2(
                        input_var=x,
                        input_width=input_width,
                        h_num_units=[120,120,120],
                        h_grad_clip=1.0,
                        output_width=1
                        )
                        
    if init_file is not None:
        print("Loading initial model parametrs...")
        init_model = np.load(init_file)
        init_params = init_model[init_model.files[0]]           
        LL.set_all_param_values([network_0], init_params)
        
    
    ###################################################                                
    ################ 3. Import data ###################
    ###################################################
    # Loading data generation model parameters
    print("Defining shared variables...")
    train_set_y = theano.shared(np.zeros(1, dtype=theano.config.floatX),
                                borrow=True) 
    train_set_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                                borrow=True)
    
    valid_set_y = theano.shared(np.zeros(1, dtype=theano.config.floatX),
                                borrow=True)
    valid_set_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                                borrow=True)
    
    # Validation data (pick a single augmented instance, rand0 here)
    print("Creating validation data...")    
    chunk_valid_data = np.load(
        "./valid/data_valid_augmented_cv%s_t%s_rand0.npy" 
        % (cross_val, input_width)
        ).astype(theano.config.floatX)
    chunk_valid_answers = np.load(
        "./valid/data_valid_expected_cv%s.npy" 
        % (cross_val)
        ).astype(theano.config.floatX)     
    
    print "chunk_valid_answers.shape", chunk_valid_answers.shape
    print("Assigning validation data...")
    valid_set_y.set_value(chunk_valid_answers[:])
    valid_set_x.set_value(chunk_valid_data.transpose(0,2,1))
    
    # Create output directory
    if not os.path.exists("output_cv%s_v%s" % (cross_val, dataver)):
        os.makedirs("output_cv%s_v%s" % (cross_val, dataver))
    
    
    ###################################################                                
    ########### 4. Create Loss expressions ############
    ###################################################
    print("Defining loss expressions...")
    prediction_0 = LL.get_output(network_0) 
    train_loss = aggregate(T.abs_(prediction_0 - y.dimshuffle(0,'x')))
    
    valid_prediction_0 = LL.get_output(network_0, deterministic=True)
    valid_loss = aggregate(T.abs_(valid_prediction_0 - y.dimshuffle(0,'x')))
    
    
    ###################################################                                
    ############ 5. Define update method  #############
    ###################################################
    print("Defining update choices...")
    params = LL.get_all_params(network_0, trainable=True)
    learn_rate = T.scalar('learn_rate', dtype=theano.config.floatX)
    
    updates = lasagne.updates.adadelta(train_loss, params,
                                       learning_rate=learn_rate)
    
    
    ###################################################                                
    ######### 6. Define train/valid functions #########
    ###################################################    
    print("Defining theano functions...")
    train_model = theano.function(
        [index, learn_rate],
        train_loss,
        updates=updates,
        givens={
            x: train_set_x[(index*train_minibatch_size):
                            ((index+1)*train_minibatch_size)],
            y: train_set_y[(index*train_minibatch_size):
                            ((index+1)*train_minibatch_size)]  
        }
    )
    
    validate_model = theano.function(
        [index],
        valid_loss,
        givens={
            x: valid_set_x[index*valid_minibatch_size:
                            (index+1)*valid_minibatch_size],
            y: valid_set_y[index*valid_minibatch_size:
                            (index+1)*valid_minibatch_size]
        }
    )
    
    
    ###################################################                                
    ################ 7. Begin training ################
    ###################################################  
    print("Begin training...")
    sys.stdout.flush()
    
    cum_iterations = 0
    this_train_loss = 0.0
    this_valid_loss = 0.0
    best_valid_loss = np.inf
    best_iter = 0
    
    train_eval_scores = np.empty(num_eval)
    valid_eval_scores = np.empty(num_eval)
    eval_index = 0
    aug_index = 0
    
    for batch in xrange(max_train_gen_calls):
        start_time = time.time()        
        chunk_train_data = np.load(
            "./train/data_train_augmented_cv%s_t%s_rand%s.npy" %
            (cross_val, input_width, aug_index)
            ).astype(theano.config.floatX)
        chunk_train_answers = np.load(
            "./train/data_train_expected_cv%s.npy" % 
            (cross_val)
            ).astype(theano.config.floatX)     
            
        train_set_y.set_value(chunk_train_answers[:])
        train_set_x.set_value(chunk_train_data.transpose(0, 2, 1))
        
        # Iterate over minibatches in each batch
        for mini_index in xrange(train_batch_multiple):
            this_rate = np.float32(rate_init*(rate_decay**cum_iterations))
            this_train_loss += train_model(mini_index, this_rate)
            cum_iterations += 1
            
            # Report loss 
            if (cum_iterations % eval_multiple == 0):
                this_train_loss = this_train_loss / eval_multiple
                this_valid_loss = np.mean([validate_model(i) for
                                    i in xrange(valid_batch_multiple)])
                train_eval_scores[eval_index] = this_train_loss
                valid_eval_scores[eval_index] = this_valid_loss
                
                # Save report every five evaluations
                if ((eval_index+1) % 5 == 0):
                    np.savetxt(
                        "output_cv%s_v%s/training_scores.txt" %
                        (cross_val, dataver),
                         train_eval_scores, fmt="%.5f"
                         )
                    np.savetxt(
                        "output_cv%s_v%s/validation_scores.txt" %
                        (cross_val, dataver),
                         valid_eval_scores, fmt="%.5f"
                         )
                    np.savetxt(
                        "output_cv%s_v%s/last_learn_rate.txt" %
                        (cross_val, dataver),
                        [np.array(this_rate)], fmt="%.5f"
                        )
                
                # Save model if best validation score
                if (this_valid_loss < best_valid_loss):  
                    best_valid_loss = this_valid_loss
                    best_iter = cum_iterations-1
                    
                    if save_model:
                        np.savez("output_cv%s_v%s/model.npz" % 
                                 (cross_val, dataver),
                                 LL.get_all_param_values(network_0))
                    
                # Reset evaluation reports
                eval_index += 1
                this_train_loss = 0.0
                this_valid_loss = 0.0
                
        aug_index += 1
            
        end_time = time.time()
        print("Computing time for batch %d: %f" % (batch, end_time-start_time))
        
    print("Best validation loss %f after %d epochs" %
          (best_valid_loss, (best_iter*train_minibatch_size//epoch_size)))
    
    del train_set_x, train_set_y, valid_set_x, valid_set_y
    gc.collect()
    
    return None


if __name__ == '__main__':
    do_regression()     
            
                
            
            
            











