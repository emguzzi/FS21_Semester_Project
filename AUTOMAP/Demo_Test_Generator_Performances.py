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
# test the effectiveness of perturbations computed with generator network trained
# on different amounts of samples.
################################################################################

#parameters for the generating network
n_hidden = 5000
epsilon = 5 #set the threshold for the maximal norm of the perturbation
lambda1 = -0.1 #weight of the |f(x)-f(x+p)| part of the loss (needs to be negative)
lambda2 = 250 #weight of the |p| part of the loss

#parameters for the training 
learning_rate = 0.0001
n_epochs = 500
print_summary = True #If True, save a summary of the norm of the perturbations and difference in the reconstrucions
results_path = folder_path+'Generator_Performances/'


# values for the AUTOMAP network
k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()
[W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights, robust = False)

sample_size = [8,16,32,64,128,256,512,1024]
epsilon = [0.5,1,2,3,5]
mean_effectiveness = np.zeros((len(epsilon),len(sample_size)))
for index_eps,eps in enumerate(epsilon):
    for index_size,size in enumerate(sample_size):
        batch_size = np.min([size,64])
        # AUTOMAP network
        g_AUTOMAP,g_AUTOMAP_def,AUTOMAP_net = AUTOMAP_Graph(batch_size = 2*batch_size,trainable = False)
        # Generator network
        g_gen,gen = Generator_Sample_Graph(g_AUTOMAP_def = g_AUTOMAP_def,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2,learning_rate = learning_rate, trainable_model = True)

        #reading of the Dataset
        data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2)
        data_train = data_train[:size,:,:]
        data_train_sampled = sample_image(data_train,k_mask_idx1, k_mask_idx2)
        N = 1+data_train.shape[0]-batch_size

        #initialize TF variables
        sess=tf.Session(graph = g_gen)

        initg_AUTOMAP = g_gen.get_operation_by_name('AUTOMAP/recs/init')

        sess.run(initg_AUTOMAP,feed_dict={
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                               'AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                               'AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})

        initg_gen = g_gen.get_operation_by_name('init')
        sess.run(initg_gen)

        #train
        step = 0
        epoch_loss = [0]*n_epochs
        epoch_loss1 =[0]*n_epochs
        epoch_loss2 =[0]*n_epochs
        for epoch in range(n_epochs):
            pos = 0
            while pos < N:
                batch_X = data_train_sampled[pos:pos+batch_size,:]
                
                feed_dict = {gen['x_input']: batch_X, gen['epsilon']: eps}
                _, loss = sess.run([gen['opt'], gen['loss']], feed_dict = feed_dict)
                #use l1 and l2 to monitor the two different components of the loss separately
                loss1,loss2 = sess.run([gen['l1'],gen['l2']],feed_dict = feed_dict)
                #log_file = open(folder_path+'log.txt','w')
                print("epoch %d, sample size %d, loss: %f" % (epoch, size, loss),file = open(folder_path+log.txt','w'))
                #log_file.close()
                
                step += 1
                pos += batch_size
            
                epoch_loss[epoch] = epoch_loss[epoch] + loss
                epoch_loss1[epoch] = epoch_loss1[epoch] + loss1
                epoch_loss2[epoch] = epoch_loss2[epoch] + loss2
            

        
        # plot the total loss against the epochs
        plt.figure(0)
        plt.plot(range(n_epochs),np.array(epoch_loss)/size,label = 'Training size %d, Batch size %d'%(size,batch_size))
        plt.xlabel('Number of epochs')
        plt.ylabel('Total Loss')
        plt.ylim([-200,200])
        lgd = plt.legend(loc="center left",bbox_to_anchor=(1.04,0.5))
        plt.savefig(results_path+'Loss/training_loss_'+str(size)+'.pdf',bbox_extra_artists=[lgd], bbox_inches='tight')
        
        # plot the first part of the loss against the epochs
        plt.figure(1)
        plt.plot(range(n_epochs),np.array(epoch_loss1)/size,label = 'Training size %d, Batch size %d'%(size,batch_size))
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss 1 (|f(x)-f(x+p)|)')
        lgd = plt.legend(loc="center left",bbox_to_anchor=(1.04,0.5))
        plt.savefig(results_path+'Loss/training_loss1_'+str(size)+'.pdf',bbox_extra_artists=[lgd], bbox_inches='tight')
        #plt.ylim([0,100])

        # plot the second part of the loss against the epochs
        plt.figure(2)
        plt.plot(range(n_epochs),np.array(epoch_loss2)/size,label = 'Training size %d, Batch size %d'%(size,batch_size))
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss 2 (|p|)')
        plt.ylim([0,50])
        lgd = plt.legend(loc="center left",bbox_to_anchor=(1.04,0.5))
        plt.savefig(results_path+'Loss/training_loss2_'+str(size)+'.pdf',bbox_extra_artists=[lgd], bbox_inches='tight')    

        pos = 0
        pert  = np.zeros((size,128,128))
        while pos < N:
            #generate the perturbations for the sample in data_train 
            pert_sample = sess.run(gen['y_out'],feed_dict = {gen['x_input']:data_train_sampled[pos:pos+batch_size,:]})
            pert[pos:pos+batch_size,:,:] = adjoint_of_samples(pert_sample,k_mask_idx1, k_mask_idx2)
            pos = pos + batch_size
        sess.close()

        if  print_summary:
            new_sess = tf.Session()
            raw_f, raw_df = compile_network(new_sess, batch_size);
            pos = 0
            im_pert  = np.zeros((size,128,128))
            im = np.zeros((size,128,128))
            while pos < N:
                x = data_train[pos:pos+batch_size,:,:] + pert[pos:pos+batch_size,:,:]
                im_pert = np.abs(hand_f(raw_f,x,k_mask_idx1, k_mask_idx2))
                im = np.abs(hand_f(raw_f,data_train[pos:pos+batch_size,:,:],k_mask_idx1, k_mask_idx2))
                pos = pos + batch_size
            mean_effectiveness[index_eps,index_size]=np.mean(np.linalg.norm(im_pert-im,axis=(1,2)))

            summary = pd.DataFrame(columns = ['i','|A^*(p_i)|','|f(x_i)-f(x_i+A^*(p_i))|'])
            for i in range(batch_size):
                summary_i = pd.DataFrame([[i,np.linalg.norm(pert[i,:,:]),np.linalg.norm(im_pert[i,:,:]-im[i,:,:])]],columns = ['i','|A^*(p_i)|','|f(x_i)-f(x_i+A^*(p_i))|'])
                summary = pd.concat([summary,summary_i])
            summary.to_csv(results_path+'Loss/summary_'+str(batch_size)+'.csv',index = False)

            new_sess.close()
            tf.reset_default_graph()

table = pd.DataFrame(data = mean_effectiveness[:,:],index = np.array(epsilon))
table.to_csv(results_path+'Mean_Effectiveness_Table.csv')

