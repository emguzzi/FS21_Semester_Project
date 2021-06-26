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
# Search for adversarial perturbations using the generator network and save the results.
################################################################################
#parameters for the generating network
n_hidden = 5000
epsilon = 1 #set the threshold for the maximal norm of the perturbation
lambda1 = -0.1 #weight of the |f(x)-f(x+p)| part of the loss (needs to be negative)
lambda2 = 250 #weight of the |p| part of the loss

#parameters for the training 
learning_rate = 0.0001
n_epochs = 500
number_of_samples = 15 #number of samples from the dataset for which we want to compute adversarial perturbations

#plot parameters
plot_loss = True #If True, print the evolution of the loss during the training
print_results = True #If True, save the perturbations and reconstructions as pdf
print_summary = True #If True, save a summary of the norm of the perturbations and reconstrucions
results_path = folder_path+'Adversarial_Pert_Results/'

# values for the AUTOMAP network
k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()
[W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights, robust = False)

# AUTOMAP network
g_AUTOMAP,g_AUTOMAP_def,AUTOMAP_net = AUTOMAP_Graph(batch_size = 2*number_of_samples,trainable = False)

# Generator network
g2,gen = Generator_Sample_Graph(g_AUTOMAP_def = g_AUTOMAP_def,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2, learning_rate = learning_rate, trainable_model = True)

#reading of the Dataset
data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2,shuffle = False)
#choose a small subset of the data set 
data_train = data_train[:number_of_samples,:,:]
data_train_sampled = sample_image(data_train,k_mask_idx1, k_mask_idx2)

#initialize TF variables
sess=tf.Session(graph = g2)

initg_AUTOMAP = g2.get_operation_by_name('AUTOMAP/recs/init')

sess.run(initg_AUTOMAP,feed_dict={
                       'AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                       'AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                       'AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                       'AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                       'AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})

initg2 = g2.get_operation_by_name('init')
sess.run(initg2)

#train
step = 0
epoch_loss = [0]*n_epochs
epoch_loss1 =[0]*n_epochs
epoch_loss2 =[0]*n_epochs
for epoch in range(n_epochs):
    
    feed_dict = {gen['x_input']: data_train_sampled, gen['epsilon']: epsilon}
    _, loss = sess.run([gen['opt'], gen['loss']], feed_dict = feed_dict)
    #use l1 and l2 to monitor the two different components of the loss separately
    loss1,loss2 = sess.run([gen['l1'],gen['l2']],feed_dict = feed_dict)
    log_file = open(folder_path+'log.txt','w')
    print("epoch %d, step %d, loss: %f" % (epoch, step, loss),file = log_file)
    log_file.close()
    
    step += 1

    
    epoch_loss[epoch] = loss
    epoch_loss1[epoch] = loss1
    epoch_loss2[epoch] = loss2
    

if plot_loss:
    # plot the total loss against the epochs
    plt.figure(0)
    plt.plot(range(n_epochs),epoch_loss)
    plt.xlabel('Number of epochs')
    plt.ylabel('Total Loss')
    plt.ylim([-200,200])
    plt.savefig(results_path+'Training_Loss/training_loss.pdf')
    
    # plot the first part of the loss against the epochs
    plt.figure(1)
    plt.plot(range(n_epochs),epoch_loss1)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss 1 (|f(x)-f(x+p)|)')
    plt.savefig(results_path+'Training_Loss/training_loss1.pdf')
    #plt.ylim([0,100])

    # plot the second part of the loss against the epochs
    plt.figure(2)
    plt.plot(range(n_epochs),epoch_loss2)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss 2 (|p|)')
    plt.ylim([0,50])
    plt.savefig(results_path+'Training_Loss/training_loss2.pdf')    

if print_results or print_summary:
    #generate the perturbations for the sample in data_train 
    pert_sample = sess.run(gen['y_out'],feed_dict = {gen['x_input']:data_train_sampled})
    #remove the kernel components
    pert = adjoint_of_samples(pert_sample,k_mask_idx1, k_mask_idx2)
    #print('perturbation shape:'+str(pert.shape))
sess.close()

if print_results or print_summary:
    new_sess = tf.Session()
    raw_f, raw_df = compile_network(new_sess, number_of_samples);
    x = data_train + pert
    im_pert = np.abs(hand_f(raw_f,x,k_mask_idx1, k_mask_idx2))
    im = np.abs(hand_f(raw_f,data_train,k_mask_idx1, k_mask_idx2))
    im_pert_sample = np.abs(raw_f(data_train_sampled+pert_sample))
    
if print_summary:
    summary = pd.DataFrame(columns = ['i','|A^*(p_i)|','|f(x_i)-f(x_i+A^*(p_i))|','|p_i|','|f(x_i)-f(x_i+p_i)|','|AA^*(p_i)-p_i|'])
    for i in range(number_of_samples):
        summary_i = pd.DataFrame([[i,np.linalg.norm(pert[i,:,:]),np.linalg.norm(im_pert[i,:,:]-im[i,:,:]),np.linalg.norm(pert_sample[i,:]),np.linalg.norm(im_pert_sample[i,:,:]-im[i,:,:]),np.linalg.norm(pert_sample[i,:]-sample_image(pert,k_mask_idx1, k_mask_idx2)[i,:])]],columns = ['i','|A^*(p_i)|','|f(x_i)-f(x_i+A^*(p_i))|','|p_i|','|f(x_i)-f(x_i+p_i)|','|AA^*(p_i)-p_i|'])
        summary = pd.concat([summary,summary_i])
    summary.to_csv(results_path+'summary.csv',index = False)

if print_results:
    
    ## save all the images as well as the perturbations
    for i in range(number_of_samples):
        
        #save the reconstruction of the perturbed image
        filename = 'PertRec%d.pdf'%(i)
        vmax = np.amax(im_pert[i,:,:])
        vmin = np.amin(im_pert[i,:,:])
        plt.imsave(results_path+'Reconstructions_Perturbations/'+filename,im_pert[i,:,:],vmin=vmin, vmax=vmax, cmap='gray')
        
        #save the original image
        filename = 'Original%d.pdf'%(i)
        vmax = np.amax(data_train[i,:,:])
        vmin = np.amin(data_train[i,:,:])
        plt.imsave(results_path+'Reconstructions_Perturbations/'+filename,data_train[i,:,:],vmin=vmin, vmax=vmax, cmap='gray')
        
        #save the reconstruction without perturbation
        filename = 'Rec%d.pdf'%(i)
        vmax = np.amax(im[i,:,:])
        vmin = np.amin(im[i,:,:])
        plt.imsave(results_path+'Reconstructions_Perturbations/'+filename,im[i,:,:],vmin=vmin, vmax=vmax, cmap='gray')
        
        #save the differenece between 
        filename = 'Diff%d.pdf'%(i)
        vmax = np.amax(np.abs(im[i,:,:]-im_pert[i,:,:]))
        vmin = np.amin(np.abs(im[i,:,:]-im_pert[i,:,:]))
        plt.imsave(results_path+'Reconstructions_Perturbations/'+filename,np.abs(im[i,:,:]-im_pert[i,:,:]),vmin=vmin, vmax=vmax, cmap='gray')
        
        #Save the perturbation
        filename = 'ViewPert%d.pdf'%(i)
        vmax = np.amax(np.abs(pert[i,:,:]))
        vmin = np.amin(np.abs(pert[i,:,:]))
        plt.imsave(results_path+'Reconstructions_Perturbations/'+filename,np.abs(pert[i,:,:]),vmin=vmin, vmax=vmax, cmap='gray')

        #show the Blurry reconstruction
        filename = 'BlurryRec%d.pdf'%(i)
        Blurry_rec = adjoint_of_samples(data_train_sampled,k_mask_idx1, k_mask_idx2)
        vmax = np.amax(np.abs(Blurry_rec[i,:,:]))
        vmin = np.amin(np.abs(Blurry_rec[i,:,:]))
        plt.imsave(results_path+'Reconstructions_Perturbations/'+filename,np.abs(Blurry_rec[i,:,:]),vmin=vmin, vmax=vmax, cmap='gray')
        
if print_results or print_summary:        
    new_sess.close()