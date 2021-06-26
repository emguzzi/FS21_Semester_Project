#This script searches for a perturbation, which makes the network fail.
#The result will be saved in a Runner object. Make sure you have updated the
#automap_config.py file before running this script.

import sys
#add py_adv_tool to python path


import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from config import src_weights, data_path
from automap_tools import *;
from Runner import Runner;
from Automap_Runner import Automap_Runner;
from utils import*


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

#Added line: read data
data_train, data_test = read_data(data_path,k_mask_idx1,k_mask_idx2)
# use data_y_train because data_x_train are the subsampled Fourier coeff and not the imgs
mri_data = data_test
batch_size = mri_data.shape[0]


#data = scipy.io.loadmat(join(src_mri_data, 'HCP_mgh_1033_T2_128_w_symbol.mat'));
#mri_data = data['mr_images_w_symbol'];
#batch_size = mri_data.shape[0];

# Plot parameters
N = 128; # out image shape
bd = 5;  # Boundary between images
plot_dest = './plots_con';
splits = 'splits';

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest);
    split_dest = join(plot_dest, splits);
    if not (os.path.isdir(split_dest)):
        os.mkdir(split_dest);

# Optimization parameters
max_itr = 8; # Update list below. This value is not relevant here
max_r_norm = float('Inf');
max_diff_norm = float('Inf');
la = 0.1; #coeff for the penalty term
warm_start = 'off';
warm_start_factor = 0.0;
perp_start = 'rand';
perp_start_factor = 1e-5;
reference = 'true';
momentum = 0.9;
learning_rate = 0.0001;
verbose=True;

sess = tf.Session();

#raw_f, raw_df = compile_network(sess, batch_size);
raw_f, raw_df = compile_robust_network(sess,batch_size,src_weights)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2);
dQ = lambda x, r, label, la: hand_dQ(raw_df, x, r, label, la,
                                          k_mask_idx1, k_mask_idx2);

runner = Automap_Runner(max_itr, max_r_norm, max_diff_norm,
                         la=la,
                         warm_start=warm_start,
                         warm_start_factor=warm_start_factor,
                         perp_start=perp_start,
                         perp_start_factor=perp_start_factor,
                         reference=reference,
                         momentum=momentum,
                         learning_rate= learning_rate,
                         verbose=verbose,
                         mask= [k_mask_idx1, k_mask_idx2]
                         );

# Update the number of iteration you would like to run
#max_itr_schedule = [12, 4, 4, 4];
#why do we need a list? and not just a scalar?
max_itr_schedule = [5]

for i in range(len(max_itr_schedule)):
    max_itr = max_itr_schedule[i];
    runner.max_itr = max_itr;
    runner.find_adversarial_perturbation(f, dQ, mri_data);

runner_id = runner.save_runner(f);

print('Saving runner as nbr: %d' % runner_id);
runner1 = load_runner(runner_id);
#print(runner1.x0)
#print(len(runner1.x0))
#print(runner1.x0[0].shape)
#print(runner1.r)
#print(len(runner1.r))
#print(runner1.r[0].shape)
#print(runner1.r[1].shape)
mri_data = runner1.x0[0];
im_nbr = 5;
bd = 5;
N = 128;
for i in range(len(runner1.r)):
    rr = runner1.r[i];
    if i == 0:
        rr = np.zeros(rr.shape, dtype=rr.dtype);
    fxr = f(mri_data + rr);
    #uncomment to save particularly effective perturbations
    # for j,norm in enumerate(np.linalg.norm(fxr-f(mri_data),axis=(1,2))):
    #     if norm > 20:
    #         im_extreme = scale_to_01(mri_data[j,:,:])
    #         pert = scale_to_01(np.abs(rr[j,:,:]))
    #         im_extreme_pert = scale_to_01(np.abs(mri_data+10*rr)[j,:,:])
    #         im_extreme_pert_rec = scale_to_01(np.abs(f(mri_data+rr)[j,:,:]))
    #         im_extreme_rec = scale_to_01(np.abs(f(mri_data)[j,:,:]))
    #         plt.imsave('Extreme/original_'+str(j)+'.png',im_extreme,cmap='gray')
    #         plt.imsave('Extreme/original_pert_'+str(j)+'.png',im_extreme_pert,cmap='gray')
    #         plt.imsave('Extreme/rec_'+str(j)+'.png',im_extreme_pert_rec,cmap='gray')
    #         plt.imsave('Extreme/pert_'+str(j)+'.png',pert,cmap='gray')
    #         plt.imsave('Extreme/original_rec'+str(j)+'.png',im_extreme_rec,cmap='gray')

    x = mri_data[im_nbr, :,:];
    r = rr[im_nbr, :,:];
    fxr = fxr[im_nbr, :,:];
    
    im_left  = scale_to_01(abs(x+r));
    im_right = scale_to_01(fxr);
    im_out = np.ones([N, 2*N + bd]);
    im_out[:,:N] = im_left;
    im_out[:,N+bd:] = im_right;
    fname_out = join(plot_dest, \
                     'rec_automap_runner_%d_r_idx_%d.png' % (runner_id, i));
    plt.imsave(fname_out, im_out, cmap='gray');
    fname_out_noisy = join(plot_dest, splits, \
                           'runner_%d_r_idx_%d_noisy.png' % (runner_id, i));
    fname_out_noisy_rec = join(plot_dest, splits, \
                           'runner_%d_r_idx_%d_noisy_rec.png' % (runner_id, i));
    plt.imsave(fname_out_noisy, im_left, cmap='gray');
    plt.imsave(fname_out_noisy_rec, im_right, cmap='gray');

sess.close();
