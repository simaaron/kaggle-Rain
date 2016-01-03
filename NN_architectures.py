# -*- coding: utf-8 -*-
"""
@author: Aaron Sim
Kaggle competition: How Much Did It Rain II

RNN architectures
"""
import lasagne
import lasagne.layers as LL
import theano.tensor as T


def build_1Dregression_v1(input_var=None, input_width=None, nin_units=12,
                            h_num_units=[64,64], h_grad_clip=1.0,
                            output_width=1):
    """
    A stacked bidirectional RNN network for regression, alternating
    with dense layers and merging of the two directions, followed by
    a feature mean pooling in the time direction, with a linear
    dim-reduction layer at the start
    
    Args:
        input_var (theano 3-tensor): minibatch of input sequence vectors
        input_width (int): length of input sequences
        nin_units (list): number of NIN features
        h_num_units (int list): no. of units in hidden layer in each stack
                                from bottom to top
        h_grad_clip (float): gradient clipping maximum value 
        output_width (int): size of output layer (e.g. =1 for 1D regression)
    Returns:
        output layer (Lasagne layer object)
    """
    
    # Non-linearity hyperparameter
    nonlin = lasagne.nonlinearities.LeakyRectify(leakiness=0.15)
    
    # Input layer
    l_in = LL.InputLayer(shape=(None, 22, input_width), 
                            input_var=input_var) 
    batchsize = l_in.input_var.shape[0]
    
    # NIN-layer
    l_in = LL.NINLayer(l_in, num_units=nin_units,
                       nonlinearity=lasagne.nonlinearities.linear)
    
    l_in_1 = LL.DimshuffleLayer(l_in, (0,2,1))
    
    
    # RNN layers
    for h in h_num_units:
        # Forward layers
        l_forward_0 = LL.RecurrentLayer(l_in_1,
                                        nonlinearity=nonlin,
                                        num_units=h,
                                        backwards=False,
                                        learn_init=True,
                                        grad_clipping=h_grad_clip,
                                        unroll_scan=True,
                                        precompute_input=True)
                                    
        l_forward_0a = LL.ReshapeLayer(l_forward_0, (-1, h))
        l_forward_0b = LL.DenseLayer(l_forward_0a, num_units=h,
                                     nonlinearity=nonlin)
        l_forward_0c = LL.ReshapeLayer(l_forward_0b,
                                       (batchsize, input_width, h))
        
        # Backward layers
        l_backward_0 = LL.RecurrentLayer(l_in_1,
                                         nonlinearity=nonlin,
                                         num_units=h,
                                         backwards=True,
                                         learn_init=True,
                                         grad_clipping=h_grad_clip,
                                         unroll_scan=True,
                                         precompute_input=True)
                                        
        l_backward_0a = LL.ReshapeLayer(l_backward_0, (-1, h))
        l_backward_0b = LL.DenseLayer(l_backward_0a, num_units=h,
                                      nonlinearity=nonlin)
        l_backward_0c = LL.ReshapeLayer(l_backward_0b,
                                        (batchsize, input_width, h))
        
        l_in_1 = LL.ElemwiseSumLayer([l_forward_0c, l_backward_0c])                 
                                                                                  
    # Output layers
    network_0a = LL.ReshapeLayer(l_in_1, (-1, h_num_units[-1]))
    network_0b = LL.DenseLayer(network_0a, num_units=output_width,
                               nonlinearity=nonlin)
    network_0c = LL.ReshapeLayer(network_0b,
                                 (batchsize, input_width, output_width))    
    
    output_net_1 = LL.FlattenLayer(network_0c, outdim=2)
    output_net_2 = LL.FeaturePoolLayer(output_net_1, pool_size=input_width,
                                       pool_function=T.mean)
    
    return output_net_2


def build_1Dregression_v2(input_var=None, input_width=None,
                            h_num_units=[64,64], h_grad_clip=1.0,
                            output_width=1):
    """
    A stacked bidirectional RNN network for regression, alternating
    with dense layers and merging of the two directions, followed by
    a feature mean pooling in the time direction
    
    Args:
        input_var (theano 3-tensor): minibatch of input sequence vectors
        input_width (int): length of input sequences
        h_num_units (int list): no. of units in hidden layer in each stack
                                from bottom to top
        h_grad_clip (float): gradient clipping maximum value 
        output_width (int): size of output layer (e.g. =1 for 1D regression)
    Returns:
        output layer (Lasagne layer object)
    """
    
    # Non-linearity hyperparameter
    nonlin = lasagne.nonlinearities.LeakyRectify(leakiness=0.15)
    
    # Input layer
    l_in = LL.InputLayer(shape=(None, 22, input_width), 
                            input_var=input_var) 
    batchsize = l_in.input_var.shape[0]
                            
    l_in_1 = LL.DimshuffleLayer(l_in, (0,2,1)) 
    
    # RNN layers
    for h in h_num_units:
        # Forward layers
        l_forward_0 = LL.RecurrentLayer(l_in_1,
                                        nonlinearity=nonlin,
                                        num_units=h,
                                        backwards=False,
                                        learn_init=True,
                                        grad_clipping=h_grad_clip,
                                        unroll_scan=True,
                                        precompute_input=True)
                                    
        l_forward_0a = LL.ReshapeLayer(l_forward_0, (-1, h))
        l_forward_0b = LL.DenseLayer(l_forward_0a, num_units=h,
                                     nonlinearity=nonlin)
        l_forward_0c = LL.ReshapeLayer(l_forward_0b,
                                       (batchsize, input_width, h))
        
        # Backward layers
        l_backward_0 = LL.RecurrentLayer(l_in_1,
                                         nonlinearity=nonlin,
                                         num_units=h,
                                         backwards=True,
                                         learn_init=True,
                                         grad_clipping=h_grad_clip,
                                         unroll_scan=True,
                                         precompute_input=True)
        
        l_backward_0a = LL.ReshapeLayer(l_backward_0, (-1, h))
        l_backward_0b = LL.DenseLayer(l_backward_0a, num_units=h,
                                      nonlinearity=nonlin)
        l_backward_0c = LL.ReshapeLayer(l_backward_0b,
                                        (batchsize, input_width, h))
        
        l_in_1 = LL.ElemwiseSumLayer([l_forward_0c, l_backward_0c])                 
                                                                                  
    # Output layers
    network_0a = LL.ReshapeLayer(l_in_1, (-1, h_num_units[-1]))
    network_0b = LL.DenseLayer(network_0a, num_units=output_width,
                               nonlinearity=nonlin)
    network_0c = LL.ReshapeLayer(network_0b,
                                 (batchsize, input_width, output_width))    
    
    output_net_1 = LL.FlattenLayer(network_0c, outdim=2)
    output_net_2 = LL.FeaturePoolLayer(output_net_1, pool_size=input_width,
                                       pool_function=T.mean)
    
    return output_net_2








