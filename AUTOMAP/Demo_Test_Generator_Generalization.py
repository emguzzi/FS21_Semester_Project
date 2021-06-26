import matplotlib
matplotlib.use('Agg')
import sys
import tensorflow as tf
from automap_tools import *
import scipy.io
from utils import*
import matplotlib.pyplot as plt
from config import src_weights, folder_path, data_path, saved_models_path
import pandas as pd
import numpy as np
import datetime
from models import *

################################################################################
# Test the generalization ability of the generator network.
################################################################################

epsilon = [0.5,1,2,3,5]

#parameters for the generating network
n_hidden = 5000
lambda1 = -0.1 #weight of the |f(x)-f(x+p)| part of the loss (needs to be negative)
lambda2 = 500 #weight of the |p| part of the loss

#parameters for the training 
learning_rate = 0.0001
batch_size = 64
n_epochs = 300
results_path = folder_path+'Generator_Performances/'


# values for the AUTOMAP network
k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()
[W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights, robust = False)

# AUTOMAP network
g_AUTOMAP,g_AUTOMAP_def,AUTOMAP_net = AUTOMAP_Graph(batch_size = 2*batch_size,trainable = False)
# Generator network
g_gen,gen = Generator_Sample_Graph(g_AUTOMAP_def = g_AUTOMAP_def,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2,learning_rate = learning_rate, trainable_model = True)

#reading of the Dataset
data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2)
data_train = data_train[:1024,:,:]
test_size = batch_size*int(data_test.shape[0]/batch_size)
data_test = data_test[:test_size,:,:]
data_train_sampled = sample_image(data_train,k_mask_idx1, k_mask_idx2)
data_test_sampled = sample_image(data_test,k_mask_idx1, k_mask_idx2)
N = 1+data_train.shape[0]-batch_size
N_test = 1 + test_size - batch_size

#initialize TF variables
sess=tf.Session(graph = g_gen)

initg_AUTOMAP = g_gen.get_operation_by_name('AUTOMAP/recs/init')
initg_gen = g_gen.get_operation_by_name('init')



pert_test_list_no_train = []
pert_test_list_train = []
for eps in epsilon:
    sess.run(initg_AUTOMAP,feed_dict={
                        'AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                        'AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                        'AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                        'AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                        'AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})

    sess.run(initg_gen)
 
#train on the train set and generate perturbation for the test set
    step = 0
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = data_train_sampled[pos:pos+batch_size,:]

            feed_dict = {gen['x_input']: batch_X,gen['epsilon']:eps}
            _, loss = sess.run([gen['opt'], gen['loss']], feed_dict = feed_dict)
            log_file = open(folder_path+'log.txt','w')
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss),file = log_file)
            log_file.close()

            step += 1
            pos += batch_size

    #compute perturbations for the test set after training on the train set
    pos = 0
    pert_test_no_train = np.zeros((test_size,128,128))
    while pos < N_test:
        pert_test_sample = sess.run(gen['y_out'],feed_dict = {gen['x_input']:data_test_sampled[pos:pos+batch_size,:]})
        pert_test_no_train[pos:pos+batch_size,:,:] = adjoint_of_samples(pert_test_sample,k_mask_idx1, k_mask_idx2)
        pos = pos+batch_size
    pert_test_list_no_train.append(pert_test_no_train)
    sess.close()

    sess = tf.Session(graph=g_gen)
    initg_AUTOMAP = g_gen.get_operation_by_name('AUTOMAP/recs/init')
    initg_gen = g_gen.get_operation_by_name('init')
    sess.run(initg_AUTOMAP,feed_dict={
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                               'AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                               'AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})

    sess.run(initg_gen)
    
#for comparison train on the test set and generate perturbation for the test set
    for epoch in range(n_epochs):
        pos = 0
        while pos < N_test:
            batch_X = data_test_sampled[pos:pos+batch_size,:]
        
            feed_dict = {gen['x_input']: batch_X, gen['epsilon']: eps}
            _, loss = sess.run([gen['opt'], gen['loss']], feed_dict = feed_dict)
            log_file = open(folder_path+'log.txt','w')
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss),file = log_file)
            log_file.close()
            
            print('loss:%f'%(loss))
            step += 1
            pos += batch_size
    
    #compute the perturbation for the test set after training on the test set
    pos = 0
    pert_test_train = np.zeros((test_size,128,128))
    while pos < N_test:
        pert_test_sample = sess.run(gen['y_out'],feed_dict = {gen['x_input']:data_test_sampled[pos:pos+batch_size,:]})
        pert_test_train[pos:pos+batch_size,:,:] = adjoint_of_samples(pert_test_sample,k_mask_idx1, k_mask_idx2)
        pos = pos+batch_size
    pert_test_list_train.append(pert_test_train)
sess.close()

#evaluate the perturbations
new_sess = tf.Session()
raw_f, raw_df = compile_network(new_sess, batch_size)

noise_diff = [0]
train_diff = [0]
no_train_diff = [0]
noise_norm = [0]
train_norm = [0]
no_train_norm = [0]

for i,eps in enumerate(epsilon):
    im = np.zeros((test_size,128,128))
    im_pert_trained = np.zeros((test_size,128,128))
    im_pert_no_train = np.zeros((test_size,128,128))
    im_noise = np.zeros((test_size,128,128))
    
    #generate noise of same norm as non trained perturbation
    noise = 2*np.random.rand(test_size,128,128) - 1
    noise = noise/np.linalg.norm(noise,axis=(1,2))[:,None,None]
    noise = np.linalg.norm(pert_test_list_no_train[i],axis=(1,2))[:,None,None]*noise
    
    pos = 0
    while pos < N_test:    
        im[pos:pos+batch_size,:,:] = hand_f(raw_f, data_test[pos:pos+batch_size,:,:], k_mask_idx1, k_mask_idx2)
        im_pert_trained[pos:pos+batch_size,:,:] = hand_f(raw_f,(data_test+pert_test_list_train[i])[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2)
        im_pert_no_train[pos:pos+batch_size,:,:] = hand_f(raw_f,(data_test+pert_test_list_no_train[i])[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2)
        im_noise[pos:pos+batch_size,:,:] = hand_f(raw_f, (data_test+noise)[pos:pos+batch_size,:,:], k_mask_idx1, k_mask_idx2)
        pos = pos+batch_size
    noise_diff.append(np.mean(np.linalg.norm(im-im_noise,axis=(1,2))))
    train_diff.append(np.mean(np.linalg.norm(im-im_pert_trained,axis=(1,2))))
    no_train_diff.append(np.mean(np.linalg.norm(im-im_pert_no_train,axis=(1,2))))
    noise_norm.append(np.mean(np.linalg.norm(noise,axis=(1,2))))
    train_norm.append(np.mean(np.linalg.norm(pert_test_list_train[i],axis=(1,2))))
    no_train_norm.append(np.mean(np.linalg.norm(pert_test_list_no_train[i],axis=(1,2))))

with open(results_path+'Generalization/noise_norm.pkl','wb') as f:
    pickle.dump(noise_norm,f) 
with open(results_path+'Generalization/noise_diff.pkl','wb') as f:
    pickle.dump(noise_diff,f) 
with open(results_path+'Generalization/train_norm.pkl','wb') as f:
    pickle.dump(train_norm,f) 
with open(results_path+'Generalization/train_diff.pkl','wb') as f:
    pickle.dump(train_diff,f) 
with open(results_path+'Generalization/no_train_norm.pkl','wb') as f:
    pickle.dump(no_train_norm,f) 
with open(results_path+'Generalization/no_train_diff.pkl','wb') as f:
    pickle.dump(no_train_diff,f) 

#plot the results
plt.plot(noise_norm,noise_diff,'--',c='0.55',label = 'Uniform noise')
plt.plot(train_norm,train_diff,label = 'Trained Perturbations')
plt.plot(no_train_norm,no_train_diff,label = 'Non Trained Perturbations')
plt.xlabel('Norm of perturbation')
plt.ylabel('Mean difference between clean and perturbed reconstructions')
plt.legend()
plt.savefig(results_path+'Generalization/generalization.pdf')