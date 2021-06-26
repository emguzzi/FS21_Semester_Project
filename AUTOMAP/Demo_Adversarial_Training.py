import matplotlib
matplotlib.use('Agg')
import sys
import tensorflow as tf
from automap_tools import *
import scipy.io
from generator import*
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from config import src_weights, folder_path, data_path, saved_models_path
from utils import *
from models import *
from os import path

################################################################################
# Implement the adversarial training scheme described in the paper for the AUTOMAP network.
################################################################################

#parameters for the training of the generator network
n_hidden = 5000
lambda1 = -0.1 
lambda2 = 250
epsilon = 3
learning_rate = 0.0001
n_epochs_g = 300
#parameters for the training of the reconstruction
lambda1_adv = 1
lambda2_adv = 0.1
precision = 'FP32';
resolution = 128;
in_dim = 19710; # (the resultant size of the k-space data after under sampling)
h_dim = 25000;
out_dim = 16384; #128x128
n_epochs = 20
batch_size = 64
K = 2
dropout_prob = 0


path_to_save = saved_models_path+'18_06_AUTOMAP/'
if not path.exists(path_to_save):
    os.makedirs(path_to_save)

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()

#read the original AUTOMAP weights
[W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights,robust = False)

#graph for the AUTOMAP network non trainable (used to train the generator)
#we use the factor int(batch_size/K) instead of batch_size, because the generator is trained
#separately over K differents parts of every batch
g_AUTOMAP, g_AUTOMAP_def, AUTOMAP_net =  AUTOMAP_Graph(batch_size = 2*int(batch_size/K),trainable = 'False')

#dictionary containing graph for the GAN with the AUTOMAP Network and the generator network
adv_training_dict =  Adversarial_Training_Graph(batch_size,lambda1_adv,lambda2_adv,lambda1,lambda2,epsilon,learning_rate,in_dim,n_hidden,g_AUTOMAP_def)                      
gen = adv_training_dict['Generator']


## read data
data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2)
N_b_train = 1+data_train.shape[0]-batch_size
data_train_sampled = sample_image(data_train,k_mask_idx1, k_mask_idx2)

N_b_eval = 1 + data_test.shape[0]-batch_size
data_test_sampled = sample_image(data_test,k_mask_idx1, k_mask_idx2)

loss_clean_plot_train = np.zeros(n_epochs)
loss_pert_plot_train = np.zeros(n_epochs)

loss_clean_plot_eval = np.zeros(n_epochs)
loss_pert_plot_eval = np.zeros(n_epochs)

## Start Adversarial training
sess = tf.Session(graph = adv_training_dict['graph'])

#initialize the generator from scratch and the AUTOMAP network using the new weights
init_AUTOMAP_Generator = adv_training_dict['graph'].get_operation_by_name('Generator/AUTOMAP/recs/init')

sess.run(init_AUTOMAP_Generator,feed_dict={
                       'Generator/AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'Generator/AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                       'Generator/AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'Generator/AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                       'Generator/AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'Generator/AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                       'Generator/AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'Generator/AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                       'Generator/AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'Generator/AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})

init = adv_training_dict['graph'].get_operation_by_name('init')

sess.run(init,feed_dict={
                       adv_training_dict['AUTOMAP_net']['W1_fc']:  W1_fc,  adv_training_dict['AUTOMAP_net']['b1_fc']:  b1_fc,
                       adv_training_dict['AUTOMAP_net']['W2_fc']:  W2_fc,  adv_training_dict['AUTOMAP_net']['b2_fc']:  b2_fc, 
                       adv_training_dict['AUTOMAP_net']['W1_cnv']: W1_cnv, adv_training_dict['AUTOMAP_net']['b1_cnv']: b1_cnv,
                       adv_training_dict['AUTOMAP_net']['W2_cnv']: W2_cnv, adv_training_dict['AUTOMAP_net']['b2_cnv']: b2_cnv, 
                       adv_training_dict['AUTOMAP_net']['W1_dcv']: W1_dcv, adv_training_dict['AUTOMAP_net']['b1_dcv']: b1_dcv})

for epoch in range(n_epochs):
    
    ## select batch
    pos_b = 0
    step_b = 1

    while pos_b < N_b_train:

        batch_data_train = data_train_sampled[pos_b:pos_b+batch_size,:]

        #initialize array for perturbations with the same shape as batch_data_train
        batch_pert_train = np.zeros(shape = (batch_data_train.shape[0],128,128))

        ## split the batch into K parts
        pos_k = 0
        N_k = batch_data_train.shape[0]-int(batch_size/K)
        start = timer()
        while pos_k < N_k:
            batch_data_k = batch_data_train[pos_k:pos_k+int(batch_size/K),:]

            ## train the generator over the pos_k part of the pos_b Batch for n_epochs_g epochs
            feed_dict = {gen['x_input']: batch_data_k,gen['epsilon']:epsilon}
            for epoch_g in range(n_epochs_g):
                _, loss = sess.run([adv_training_dict['gen_opt'], gen['loss']], feed_dict=feed_dict)
            pos_k = pos_k  + int(batch_size/K)
            batch_pert_train[pos_k:pos_k+int(batch_size/K),:,:] = adjoint_of_samples(sess.run(gen['y_out'],feed_dict = {gen['x_input']:batch_data_k}),k_mask_idx1, k_mask_idx2)


        #merge the perturbation and the original images to train the AUTOMAP net
        batch_merged = np.zeros((2*batch_size,128,128))
        batch_merged[:batch_size,:,:] = data_train[pos_b:pos_b+batch_size,:]
        batch_merged[batch_size:,:,:] = data_train[pos_b:pos_b+batch_size,:] + batch_pert_train
        batch_merged_sampled = sample_image(batch_merged, k_mask_idx1, k_mask_idx2)
        corrupt_prob = np.asarray([0],dtype=np.float32)

        feed_dict = {adv_training_dict['x_train']: batch_merged_sampled,adv_training_dict['y']: batch_merged[:batch_size,:,:],adv_training_dict['corrupt_prob']: corrupt_prob}
        loss_pert_plot_train[epoch] = loss_pert_plot_train[epoch] + 1/(int(data_train.shape[0]/batch_size))*sess.run(adv_training_dict['loss_pert'],feed_dict)
        loss_clean_plot_train[epoch] = loss_pert_plot_train[epoch] + 1/(int(data_train.shape[0]/batch_size))*sess.run(adv_training_dict['loss_clean'],feed_dict)
        _,loss = sess.run([adv_training_dict['train_op'],adv_training_dict['loss_train']],feed_dict = feed_dict)
        
        end = timer()
        print("epoch %d, batch %d,time for batch %f (s), ETA %f (h)" % (epoch, step_b,end-start, (end-start)*((n_epochs-epoch)*int(N_b_train/batch_size)+int(N_b_train/batch_size)-step_b)/3600),file = open('/cluster/home/emguzzi/Code/Invfool/AUTOMAP/log.txt','w'))

        pos_b = pos_b+batch_size
        step_b = step_b + 1
        
        #update the weights of the AUTOMAP network used to train the generator
        #model in vars is the list of vars for the AUTOMAP net (see utils.py)
        for var in model_in_vars:
            tensor_plh = adv_training_dict['graph'].get_tensor_by_name('Generator/AUTOMAP/recs/'+var+':0')
            value = adv_training_dict['graph'].get_tensor_by_name(var+':0').eval(session =sess)
            sess.run(tensor_plh, feed_dict = {tensor_plh: value})
            
##after each epoch evaluate the robustness of the adversariarly trained AUTOMAP network on the test dataset
    pos_b = 0
    while pos_b < N_b_eval:
        batch_data_eval = data_test_sampled[pos_b:pos_b+batch_size,:]
        #initialize array for perturbations with the same shape as batch_X
        batch_pert_eval = np.zeros(shape = (batch_data_eval.shape[0],128,128))
        pos_k = 0
        N_k_eval = batch_data_eval.shape[0]-int(batch_size/K)
        
        while pos_k < N_k_eval:
            batch_data_k_eval = batch_data_eval[pos_k:pos_k+int(batch_size/K),:]

            ## train the generator over the pos_k part of the pos_b Batch for n_epochs_g epochs
            feed_dict = {gen['x_input']: batch_data_k_eval,gen['epsilon']:epsilon}
            for epoch_g in range(n_epochs_g):
                _, loss = sess.run([adv_training_dict['gen_opt'], gen['loss']], feed_dict=feed_dict)
            pos_k = pos_k  + int(batch_size/K)
            batch_pert_eval[pos_k:pos_k+int(batch_size/K),:,:] = adjoint_of_samples(sess.run(gen['y_out'],feed_dict = {gen['x_input']:batch_data_k_eval}),k_mask_idx1, k_mask_idx2)
        
        batch_merged_eval = np.zeros((2*batch_size,128,128))
        batch_merged_eval[:batch_size,:,:] = data_test[pos_b:pos_b+batch_size,:]
        batch_merged_eval[batch_size:,:,:] = data_test[pos_b:pos_b+batch_size,:] + batch_pert_eval
        batch_merged_eval_sampled = sample_image(batch_merged_eval, k_mask_idx1, k_mask_idx2)

        feed_dict = {adv_training_dict['x_train']: batch_merged_eval_sampled,adv_training_dict['y']: batch_merged_eval[:batch_size,:,:],adv_training_dict['corrupt_prob']: corrupt_prob}
        loss_pert_plot_eval[epoch] = loss_pert_plot_eval[epoch] + 1/(int(data_test.shape[0]/batch_size))*sess.run(adv_training_dict['loss_pert'],feed_dict)
        loss_clean_plot_eval[epoch] = loss_pert_plot_eval[epoch] + 1/(int(data_test.shape[0]/batch_size))*sess.run(adv_training_dict['loss_clean'],feed_dict)
        pos_b = pos_b + batch_size

## save the automap weights after adversarial training
for var in model_in_vars:
    weight = adv_training_dict['graph'].get_tensor_by_name(var+':0').eval(session = sess)
    np.save(path_to_save+var,weight)
sess.close()

## save the lists with pickle
with open(folder_path+'Adversarial_Training/loss_pert_plot_train.pkl','wb') as f:
    pickle.dump(loss_pert_plot_train,f)
with open(folder_path+'Adversarial_Training/loss_clean_plot_train.pkl','wb') as f:
    pickle.dump(loss_clean_plot_train,f)
with open(folder_path+'Adversarial_Training/loss_pert_plot_eval.pkl','wb') as f:
    pickle.dump(loss_pert_plot_eval,f)
with open(folder_path+'Adversarial_Training/loss_clean_plot_eval.pkl','wb') as f:
    pickle.dump(loss_clean_plot_eval,f) 
#plot the loss for training and test perormances
x_ticks = range(0,21,5)
plt.plot(range(n_epochs),loss_pert_plot_train)
plt.xlabel('Number of epochs')
plt.ylabel('Perturbed part of the loss')
plt.ylim([0,300])
plt.xticks(x_ticks)
plt.savefig(folder_path+'Adversarial_Training/PerturbedLossTrain.pdf')
plt.clf()

plt.plot(range(n_epochs),loss_clean_plot_train)
plt.xlabel('Number of epochs')
plt.ylabel('Clean part of the loss')
plt.ylim([0,300])
plt.xticks(x_ticks)
plt.savefig(folder_path+'Adversarial_Training/CleanedLossTrain.pdf')
plt.clf()

plt.plot(range(n_epochs),loss_pert_plot_eval)
plt.xlabel('Number of epochs')
plt.ylabel('Perturbed part of the loss')
plt.xticks(x_ticks)
plt.savefig(folder_path+'Adversarial_Training/PerturbedLossEval.pdf')
plt.clf()

plt.plot(range(n_epochs),loss_clean_plot_eval)
plt.xlabel('Number of epochs')
plt.ylabel('Clean part of the loss')
plt.xticks(x_ticks)
plt.savefig(folder_path+'Adversarial_Training/CleanedLossEval.pdf')
plt.clf()
