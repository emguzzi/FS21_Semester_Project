import numpy as np;
import sys;
import pickle as pickle
#added line to use the sample image function from automap_tools
import scipy.io
from automap_config import src_weights, src_mri_data, \
                           src_k_mask, src_automap_runner;
from pyfftw.interfaces import scipy_fftpack as fftw
from os.path import join

def read_automap_k_space_mask(src_k_mask=src_k_mask, fname_k_mask_idx = 'k_mask_idx.mat'):
    """
    Reads the automap k_space indices, and return these.  
    """
    k_mask_idx_data = scipy.io.loadmat(join(src_k_mask, fname_k_mask_idx));
    idx1 = k_mask_idx_data['idx1'];
    idx2 = k_mask_idx_data['idx2'];
    return idx1, idx2;

def sample_image(im, k_mask_idx1, k_mask_idx2):
    """
    Creates the fourier samples the AUTOMAP network is trained to recover.
    
    The parameters k_mask_idx1 and k_mask_idx2 cointains the row and column
    indices, respectively, of the samples the network is trained to recover.  
    It is assumed that these indices have the same ordering of the coefficents,
    as the network is used to recover. 

    :param im: Image, assumed of size [batch_size, height, width]. The intensity 
               values of the image should lie in the range [0, 1]. 
    :param k_maks_idx1: Row indices of the Fourier samples
    :param k_maks_idx2: Column indices of the Fourier samples

    :return: Fourier samples in the format the AUTOMAP network expect
    """
    
    # Scale the image to the right range 
    im1 = 4096*im;
    batch_size = im1.shape[0];
    nbr_samples = k_mask_idx1.shape[0];
    samp_batch = np.zeros([batch_size, 2*nbr_samples], dtype=np.float32);
    
    for i in range(batch_size):
        
        single_im = np.squeeze(im1[i,:,:]);
        fft_im = fftw.fft2(single_im);
        samples = fft_im[k_mask_idx1, k_mask_idx2];
        samples_real = np.real(samples);
        samples_imag = np.imag(np.conj(samples));
        samples_concat = np.squeeze(np.concatenate( (samples_real, samples_imag) ));

        samples_concat = ( 0.0075/(2*4096) )*samples_concat;
        samp_batch[i] = samples_concat;

    return samp_batch;
    
##

l2_norm_of_tensor = lambda x: np.sqrt((abs(x)**2).sum());

def scale_to_01(im):
    """ Scales all array values to the interval [0,1] using an affine map."""
    ma = np.amax(im);
    mi = np.amin(im);
    new_im = im.copy();
    return (new_im-mi)/(ma-mi);


def compute_psnr(rec, ref):
    """
    Computes the PSNR of the recovery `rec` w.r.t. the reference image `ref`. 
    Notice that these two arguments can not be swapped, as it will yield
    different results. 
    
    More precisely PSNR will be computed between the magnitude of the image 
    |rec| and the magnitude of |ref|

    :return: The PSNR value
    """
    mse = np.mean((abs(rec-ref))**2);
    max_I = np.amax(abs(rec));
    return 10*np.log10((max_I*max_I)/mse);


def pertub_SGA(f, dQ, batch, epoch, r0, v0, momentum=0.9, lr=0.01, verbose=True,
               max_r_norm=float('Inf'), max_diff_norm=float('Inf')):
    """  Search for adversarial perturbation using a gradient ascent algorithm.

    For a neural network ``f`` and a ``batch`` of images this function search 
    for an adversarial perturbation for ``f`` using the gradient ascent direction
    with a Nesterov step. Let A be a sampling matrix and define the function
    
    Q(r) = ||f(y + A r) - f(y)||_{2}^{2} - lambda*||r||_{2}^{2}
    
    The function perform ``epoch`` of the following steps 
        v_{k+1} = momentum*v_{k} + lr* Gradient(Q(r))
        r_{k+1} = r_{k} + v_{k+1}

    :param f: Neural Network.
    :param dQ: Gradident of function Q : R^n -> R. 
    :param batch: Batch of images.
    :param epoch: Number of iterations.
    :param r0: Starting perturbation.
    :param v0: Starting direction.
    :param momentum: Momentum.
    :param lr: Learning rate.
    :param verbose: Whether or not to print information.
    :param max_r_norm: Stop iterations if ||r||_{2} > max_r_norm.
    :param max_diff_norm: Stop iterations if ||f(batch+r) -f(batch)||_{2} > max_diff_norm.
    
    :returns: r_final, v_final, str_log_of_iterations.
    """
    if (verbose):
        print('------------------------------------');
        print('Running SGA with paramters:');
        print('Momentum:      %g' % momentum);
        print('Learning rate: %g' % lr);
        print('max_r_norm:    %g' % max_r_norm);
        print('max_diff_norm: %g' % max_diff_norm);
        print('------------------------------------');
    
    
    fx = f(batch);
    i = 1;
    norm_fx_fxr = 0;
    norm_r = 0;
    backlog = '';
    norm_x0 = l2_norm_of_tensor(batch);
    r = r0;
    v = v0;
    # Added lines
    mean_nr_img = np.zeros(epoch+1)
    mean_nr_sample = np.zeros(epoch+1)
    mean_ndiff = np.zeros(epoch+1)
    k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();
    ##
    while (i <= epoch and norm_fx_fxr < max_diff_norm and norm_r < max_r_norm):
        
        dr  = dQ(batch, r);
        fxr = f(batch+r);
        v = momentum*v + lr*dr;
        r = r + v;
        ## added lines
        for im_nbr in range(batch.shape[0]):
            x_i = batch[im_nbr];
            r_i = r[im_nbr];
            fx_i = np.abs(fx[im_nbr]);
            fxr_i = np.abs(fxr[im_nbr]);

            #nx = np.linalg.norm(x_i)
            nr_img = np.linalg.norm(r_i,axis=(0,1))
            n_diff = np.linalg.norm(fx_i - fxr_i,axis=(0,1))

            
            mean_nr_img[i] = mean_nr_img[i]+1/batch.shape[0]*nr_img
            #compute also the norm in sample space in order to get a comparable result
            #mean_nr_sample[i] = mean_nr_sample[i] + 1/batch.shape[0]*nr_sample
            mean_ndiff[i] = mean_ndiff[i] + 1/batch.shape[0]*n_diff
            
        ##
        
        norm_fx_fxr = l2_norm_of_tensor(fx-fxr);
        norm_r = l2_norm_of_tensor(r);
        
        next_str = \
        '%2d: |f(x)-f(x+r)|: %8g, |r|: %8g, |f(x)-f(x+r)|/|r| : %8g, |r|/|x|: %8g' \
        % (i, norm_fx_fxr, norm_r, norm_fx_fxr/norm_r, norm_r/norm_x0);
        
        backlog = backlog + '\n' + next_str;
        if (verbose):
            print(next_str);
        i = i + 1;
    ##added lines
    with open('/cluster/home/emguzzi/Code/storage2/AUTOMAP/runners/data/mean_nr_img.pkl','wb') as file:
        pickle.dump(mean_nr_img,file)
    with open('/cluster/home/emguzzi/Code/storage2/AUTOMAP/runners/data/mean_diff.pkl','wb') as file:
            pickle.dump(mean_ndiff,file)
    #with open('/cluster/home/emguzzi/Code/storage2/AUTOMAP/runners/data/mean_nr_sample.pkl','wb') as file:
    #        pickle.dump(mean_nr_sample,file)    
    return r, v, backlog;


    
        
    
    







