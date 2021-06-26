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
# Test the effectiveness of adversarial perturbation with independent uniform noise added.
################################################################################

#parameters for the generating network
n_hidden = 5000
epsilon = 3 #set the threshold for the maximal norm of the perturbation
lambda1 = -0.1 #weight of the |f(x)-f(x+p)| part of the loss (needs to be negative)
lambda2 = 250 #weight of the |p| part of the loss

#parameters for the training 
learning_rate = 0.0001
batch_size = 64
n_epochs = 500

results_path = folder_path+'Generator_Noise_Generalization/'

# values for the AUTOMAP network
k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()
[W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights, robust = False)

# AUTOMAP network
g_AUTOMAP,g_AUTOMAP_def,AUTOMAP_net = AUTOMAP_Graph(batch_size = 2*batch_size,trainable = False)
# Generator network
g_gen,gen = Generator_Sample_Graph(g_AUTOMAP_def = g_AUTOMAP_def,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2, learning_rate = learning_rate, trainable_model = True)

#reading of the Dataset
data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2)
data_train = data_train[:1024,:,:]
data_train_sampled = sample_image(data_train,k_mask_idx1, k_mask_idx2)
N = 1+data_train.shape[0]-batch_size

#initialize TF variables
sess=tf.Session(graph = g_gen)

initg_AUTOMAP = g_gen.get_operation_by_name('AUTOMAP/recs/init')
initg_gen = g_gen.get_operation_by_name('init')
sess.run(initg_AUTOMAP,feed_dict={
                       'AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                       'AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                       'AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                       'AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                       'AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})


sess.run(initg_gen)


#train
step = 0
for epoch in range(n_epochs):
    pos = 0
    while pos < N:
        batch_X = data_train_sampled[pos:pos+batch_size,:]
        
        feed_dict = {gen['x_input']: batch_X, gen['epsilon']: epsilon}
        _, loss = sess.run([gen['opt'], gen['loss']], feed_dict = feed_dict)
        log_file = open(folder_path+'log.txt','w')
        print("epoch %d, step %d, loss: %f" % (epoch, step, loss),file = log_file)
        log_file.close()
        
        step += 1
        pos += batch_size
    
  


#generate the perturbations for the sample in data_train 
pos = 0
pert = np.zeros((data_train.shape[0],128,128))
while pos < N:
    pert_sample = sess.run(gen['y_out'],feed_dict = {gen['x_input']:data_train_sampled[pos:pos+batch_size,:]})
    pert[pos:pos+batch_size,:,:] = adjoint_of_samples(pert_sample,k_mask_idx1, k_mask_idx2)
    pos += batch_size
sess.close()


new_sess = tf.Session()
raw_f, raw_df = compile_network(new_sess, batch_size);
x = data_train + pert

pos = 0
im_pert = np.zeros((data_train.shape[0],128,128))
im = np.zeros((data_train.shape[0],128,128))

#reconstruct the origninal and perturbed image
while pos < N:
    im_pert[pos:pos+batch_size,:,:] = np.abs(hand_f(raw_f,x[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2))
    im[pos:pos+batch_size,:,:] = np.abs(hand_f(raw_f,data_train[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2))
    pos = pos + batch_size
    

noise_sizes = [0.5,1,2,5,7,10,15]
noise_list = []
noise_ref_diff_list = []
noise_pert_diff_list = []


for size in noise_sizes:
    noise = 2*np.random.rand(data_train.shape[0],128,128)-1
    noise = size*noise/np.linalg.norm(noise,axis=(1,2))[:,None,None]
    noise_list.append(noise)
    
for noise in noise_list:
    #generate uniform noise of the same size of pert + noise to compare
    ref_noise = 2*np.random.rand(data_train.shape[0],128,128)-1
    ref_noise = np.linalg.norm(pert+ noise,axis=(1,2))[:,None,None]*ref_noise/np.linalg.norm(ref_noise,axis=(1,2))[:,None,None]
    im_pert_noise = np.zeros((data_train.shape[0],128,128))
    im_noise_ref = np.zeros((data_train.shape[0],128,128))
    pos = 0
    print('norm of pert + noise:\n')
    print(np.linalg.norm(pert+noise,axis=(1,2)))
    print('norm of noise_ref:\n')
    print(np.linalg.norm(ref_noise,axis=(1,2)))
    while pos < N:
        im_pert_noise[pos:pos+batch_size,:,:] = np.abs(hand_f(raw_f,(data_train +pert+ noise)[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2))
        im_noise_ref[pos:pos+batch_size,:,:] = np.abs(hand_f(raw_f,(data_train + ref_noise)[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2))
        pos = pos + batch_size
    noise_ref_diff = np.mean(np.linalg.norm(im-im_noise_ref,axis =(1,2)))
    noise_pert_diff = np.mean(np.linalg.norm(im-im_pert_noise,axis=(1,2)))
    noise_ref_diff_list.append(noise_ref_diff)
    noise_pert_diff_list.append(noise_pert_diff)

with open(folder_path+'Generator_Performances/Noise/noise_sizes.pkl','wb') as f:
    pickle.dump(noise_sizes,f) 
with open(folder_path+'Generator_Performances/Noise/noise_pert_diff_list.pkl','wb') as f:
    pickle.dump(noise_pert_diff_list,f) 
with open(folder_path+'Generator_Performances/Noise/noise_ref_diff_list.pkl','wb') as f:
    pickle.dump(noise_ref_diff_list,f) 


plt.plot(noise_sizes,noise_pert_diff_list,label = 'Perturbation of size %f + uniform noise'%(np.mean(np.linalg.norm(pert,axis=(1,2)))))
plt.plot(noise_sizes,noise_ref_diff_list,'--',c='0.55',label = 'Uniform noise')
plt.xlabel('Norm of the added uniform noise')
plt.ylabel('Difference between the perturbed reconstruction and the clean one')
plt.legend()
plt.savefig(folder_path+'Generator_Performances/Noise/added_noise.pdf')

