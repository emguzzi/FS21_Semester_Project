import matplotlib
matplotlib.use('Agg')
import sys
from automap_tools import *
import tensorflow as tf 
from generator import*
import matplotlib.pyplot as plt
from config import src_weights, folder_path, data_path, saved_models_path
from utils import*
from models import*

################################################################################
# Estimate the robustness metric rho for the AUTOMAP network.
################################################################################

#generator network parameters
epsilon = [0.5,1,2,3,5]
lambda1 = -0.1
lambda2 = 250
n_hidden = 5000
n_epochs = 300
learning_rate = 0.0001
batch_size = 64

robust_model = False #Wheter to use the robust weights or the weights given in the paper
#If true specify the location of the robust weights
path_automap_weights = ''

add_noise = True # wheter to include the baseline of random uniform noise for adversarial perturbation


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()



#load the weights of the AUTOMAP network
if robust_model:
    [W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = path_automap_weights,robust = robust_model)

else:
    [W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights,robust = robust_model)


g_AUTOMAP, g_AUTOMAP_def, AUTOMAP_net =  AUTOMAP_Graph(batch_size = 2*batch_size,trainable = 'False')

#prepare Data
data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2)
N = 1+data_test.shape[0]-batch_size

data_test_sampled = sample_image(data_test,k_mask_idx1, k_mask_idx2)

# lists with perturbations
pert_list = []
noise_list = []

for eps in epsilon:
        #initiate generator network
        g_gen,gen = Generator_Sample_Graph(g_AUTOMAP_def = g_AUTOMAP_def,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2,learning_rate = learning_rate, trainable_model = True)

        sess=tf.Session(graph = g_gen)
        init2 = g_gen.get_operation_by_name('init')                    
        sess.run(init2)
        initg_AUTOMAP = g_gen.get_operation_by_name('AUTOMAP/recs/init')

        sess.run(initg_AUTOMAP,feed_dict={
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_fc'].name:  W1_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b1_fc'].name:  b1_fc,
                               'AUTOMAP/recs/'+AUTOMAP_net['W2_fc'].name:  W2_fc,  'AUTOMAP/recs/'+AUTOMAP_net['b2_fc'].name:  b2_fc, 
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_cnv'].name: W1_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_cnv'].name: b1_cnv,
                               'AUTOMAP/recs/'+AUTOMAP_net['W2_cnv'].name: W2_cnv, 'AUTOMAP/recs/'+AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                               'AUTOMAP/recs/'+AUTOMAP_net['W1_dcv'].name: W1_dcv, 'AUTOMAP/recs/'+AUTOMAP_net['b1_dcv'].name: b1_dcv})

        #train
        step = 0
        epoch_loss = [0]*n_epochs
        
        for epoch in range(n_epochs):
            pos = 0
            while pos < N:
                batch_data = data_test_sampled[pos:pos+batch_size,:]
                feed_dict = {gen['x_input']: batch_data,gen['epsilon']:eps}
                _, loss = sess.run([gen['opt'], gen['loss']], feed_dict=feed_dict)
                print("epsilon %f, epoch %d, step %d, loss: %f" % (eps, epoch, step, loss),file = open(folder_path+'log.txt','w'))

                step += 1
                pos += batch_size
        
        #compute the perturbations
        pert = np.zeros((data_test_sampled.shape[0],128,128))
        pos = 0
        while pos < N:    
            pert[pos:pos+batch_size,:,:] = adjoint_of_samples(sess.run(gen['y_out'],feed_dict = {gen['x_input']:data_test_sampled[pos:pos+batch_size,:]}),k_mask_idx1, k_mask_idx2)
            pos = pos+batch_size
        pert_list.append(pert)
        sess.close()
        
        if add_noise:
            noise = 2*np.random.rand(data_test_sampled.shape[0],128,128)-1
            noise = np.linalg.norm(pert,axis=(1,2))[:,None,None]*(noise/np.linalg.norm(noise,axis=(1,2))[:,None,None])
            noise_list.append(noise)
    
        
#evaluate the perturbation
new_sess = tf.Session()

raw_f, raw_df = compile_network(new_sess, data_test.shape[0]);

pert_loss = []
#pert_size_sample = []
pert_size = []
noise_loss = []
noise_size = []
for i in range(len(epsilon)):
    x = data_test + pert_list[i]
    im_pert = np.abs(hand_f(raw_f,x,k_mask_idx1, k_mask_idx2))
    im = np.abs(hand_f(raw_f,data_test,k_mask_idx1,k_mask_idx2))
    pert_loss.append(np.mean(np.linalg.norm(im_pert-im,axis=(1,2))))
    pert_size.append(np.mean(np.linalg.norm(pert_list[i],axis=(1,2))))
    if add_noise:
        x_noise = data_test + noise_list[i]
        im_noise = np.abs(hand_f(raw_f,x_noise,k_mask_idx1,k_mask_idx2))
        noise_loss.append(np.mean(np.linalg.norm(im_noise-im,axis=(1,2))))
        noise_size.append(np.mean(np.linalg.norm(noise_list[i],axis = (1,2))))

#save the lists to generate the plot later
if add_noise:
        with open(folder_path+'Estimate_Rho_Results/list_norm_diff_noise.pkl','wb') as f:
            pickle.dump(noise_loss,f) 
        with open(folder_path+'Estimate_Rho_Results/list_norm_noise.pkl','wb') as f:
            pickle.dump(noise_size,f) 
            
with open(folder_path+'Estimate_Rho_Results/list_norm_diff.pkl','wb') as f:
    pickle.dump(pert_loss,f) 
with open(folder_path+'Estimate_Rho_Results/list_norm_r_img.pkl','wb') as f:
    pickle.dump(pert_size,f) 
    

#plot the results
plt.plot(epsilon,pert_loss)
plt.xlabel('Epsilon')
plt.ylabel('Difference between original and perturbed reconstruction')
plt.savefig(folder_path+'Estimate_Rho_Results/Est_rho_ref.png')
plt.clf()

plt.plot(pert_size,pert_loss,label = 'Adversarial Perturbation')
plt.xlabel('Size of the Perturbation in image space')
plt.ylabel('Difference between original and perturbed reconstruction')
if add_noise:
    plt.plot(noise_size,noise_loss,'--',c='0.55',label = 'Uniform Noise')
    plt.legend()
plt.savefig(folder_path+'Estimate_Rho_Results/Est_rho_image.png')
plt.clf()

plt.plot(epsilon,pert_size)
plt.xlabel('Maximal allowed size of perturbatio (epsilon)')
plt.ylabel('Actual size of perturbation')
plt.savefig(folder_path+'Estimate_Rho_Results/Est_rho_size.png')

