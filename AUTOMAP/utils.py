## utils file containing functions used in the experiments
import sys
import tensorflow as tf #working with v=1.7
import os
from automap_tools import *
import scipy.io
import numpy as np
model_in_vars = ['W1_fc', 'b1_fc', 'W2_fc', 'b2_fc', 'W1_cnv', 'b1_cnv',
                    'W2_cnv', 'b2_cnv', 'W1_dcv', 'b1_dcv'];

def AUTOMAP_weights(src_weights,robust):
    #read the weights for the AUTOMAP network.
    #src_weights: path where the weights can be found
    #robust: bool, if False load the weights given in the paper. If Ture,
    #used the weights obtained with adversarial training.
    if robust:
        weights = [np.load(os.path.join(src_weights,file+'.npy')) for file in model_in_vars]
        return weights
    else:
        f_weights = read_automap_weights(src_weights)
        W1_cnv = np.asarray(f_weights['W1_cnv']);
        W1_dcv = np.asarray(f_weights['W1_dcv']);
        W1_fc  = np.asarray(f_weights['W1_fc']);
        W2_cnv = np.asarray(f_weights['W2_cnv']);
        W2_fc  = np.asarray(f_weights['W2_fc']);
        b1_cnv = np.asarray(f_weights['b1_cnv']);
        b1_dcv = np.asarray(f_weights['b1_dcv']);
        b1_fc  = np.asarray(f_weights['b1_fc']);
        b2_cnv = np.asarray(f_weights['b2_cnv']);
        b2_fc  = np.asarray(f_weights['b2_fc']);
        return [W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv]

def read_data(path,k_mask_idx1,k_mask_idx2,symbol = False, shuffle = True):
    #read the .mat file and return them as a numpy ndarray (#train/test samples,128,128),
    #path: path of the directory containing two subdir test and train with the corresponding .mat file
    #kmask_idx1,kmask_idx2: the mask containing the indices of the subsampled fourier coeff
    # that the network is trained to recover (input for the sample_image function in automap_tools.py)
    #symbol: if true also load the test set with the heart symbol added
    #shuffle: if true shuffle the np array before returning it
    
    scans_train = os.listdir(path+'/train/')
    scans_test = os.listdir(path+'/test/')
    
    data_train = np.empty(shape = (len(scans_train),128,128))
    
    for i,img in enumerate(os.listdir(path+'train/')):
        #we have to remove from the name img the patient id (first 8 character)
        #the '.','_' and the .mat extension
        x = scipy.io.loadmat(path+'/train/'+img)[img[8:-3].replace('.','').replace('_','')]
        data_train[i,:,:] = x
        
    
    data_test = np.empty(shape = (len(scans_test),128,128))
    
    for i,img in enumerate(os.listdir(path+'test/')):
        #we have to remove from the name img the patient id (first 8 character)
        #the '.','_' and the .mat extension
        x = scipy.io.loadmat(path+'/test/'+img)[img[8:-3].replace('.','').replace('_','')]
        data_test[i,:,:] = x

    if shuffle:
        np.random.shuffle(data_train)
        np.random.shuffle(data_test)
    
    if symbol:
        scans_symbol = os.listdir(path+'test_symbol/')
        data_test_symbol = np.empty(shape=(len(scans_symbol),128,128))
        for i,img in enumerate(os.listdir(path+'test_symbols/')):
                    x = scipy.io.loadmat(path+'/test_symbols/'+img)[img[:-3].replace('.','').replace('_','')+'.mat']
                    data_test_symbol[i,:,:] = x
        if shuffle:
            np.random.shuffle(data_test_symbol)
        return data_train, data_test, data_test_symbol
    else:
        return data_train,  data_test