import sys
import tensorflow as tf #working with v=1.7
from automap_tools import *
from utils import*
import numpy as np
from scipy.linalg import dft


#AUTOMAP parameters
precision = 'FP32';
resolution = 128;
in_dim = 19710; #(the resultant size of the k-space data after under sampling)
h_dim = 25000;
out_dim = 16384; #128x128
model_in_vars =  ['W1_fc', 'b1_fc', 'W2_fc', 'b2_fc', 'W1_cnv', 'b1_cnv',
                    'W2_cnv', 'b2_cnv', 'W1_dcv', 'b1_dcv'];

model_in_shape = [(19710, 25000), (25000,), (25000, 16384), (16384,), 
                    (5, 5, 1, 64), (64,), (5, 5, 64, 64), (64,), 
                    (7, 7, 1, 64), (1,)]
                    

def AUTOMAP_Graph(batch_size,trainable):

    with tf.Graph().as_default() as g_AUTOMAP:
        AUTOMAP_net = arch.network(batch_size, precision, resolution, in_dim, h_dim, out_dim,
                           model_in_vars=model_in_vars, 
                           model_in_shapes=model_in_shape, 
                           trainable_model_in=str(trainable));
        

        init1 = tf.global_variables_initializer()
    
    g_AUTOMAP_def = g_AUTOMAP.as_graph_def()
    
    return g_AUTOMAP, g_AUTOMAP_def, AUTOMAP_net
    

def Generator_Sample(g_AUTOMAP_def,in_dim,n_hidden,lambda1,lambda2,learning_rate,trainable_model):
    #model generating the adversarial example for the network specified in the graph_def g1 (AUTOMAP).
    #g_AUTOMAP_def: Graph definition for graph containing the AUTOMAP architecture (need to be created with 2*batch_size)
    #in_dim: dimension of the input vector
    #n_hidden: number of hidden layer
    #lambda1: control the weight of the effectiveness of the adversarial perturbation in the loss fct (has to be negative)
    #lambda2: control the weight of the norm of the adversarial perturbation in the loss function (has to be positive)
    #learning_rate: learning rate
    #trainable_model: bool specifying if the model is trainable 
    
    # input quantites
    with tf.name_scope("placeholders"):
        x_input = tf.placeholder(tf.float32, (None, in_dim),name = 'x_input')
        epsilon = tf.placeholder(tf.float32,shape = (),name = 'epsilon')
    
    #FC layers
    with tf.name_scope("hidden-layer1"):
        with tf.device('/GPU:0'):
            W1 = tf.Variable(tf.random_normal((in_dim, n_hidden), mean=0.0, stddev=0.005),name = 'gen_W1',trainable = trainable_model)
            b1 = tf.Variable(tf.random_normal((n_hidden,), mean=0.0, stddev=0.005),name = 'gen_b1',trainable = trainable_model)
            x_hidden = tf.nn.relu(tf.matmul(x_input, W1) + b1)
    
    with tf.name_scope("hidden-layer2"):
        with tf.device('/GPU:0'):
            W2 = tf.Variable(tf.random_normal((n_hidden, n_hidden), mean=0.0, stddev=0.005),name = 'gen_W2',trainable = trainable_model)
            b2 = tf.Variable(tf.random_normal((n_hidden,), mean=0.0, stddev=0.005),name = 'gen_b2',trainable = trainable_model)
            x_hidden = tf.nn.relu(tf.matmul(x_hidden, W2) + b2)
    
    with tf.name_scope("hidden-layer3"):
        with tf.device('/GPU:0'):
            W3 = tf.Variable(tf.random_normal((n_hidden, n_hidden), mean=0.0, stddev=0.005),name = 'gen_W3',trainable = trainable_model)
            b3 = tf.Variable(tf.random_normal((n_hidden,), mean=0.0, stddev=0.005),name = 'gen_b3',trainable = trainable_model)
            x_hidden = tf.nn.relu(tf.matmul(x_hidden, W3) + b3)
    
    with tf.name_scope("hidden-layer4"):
        with tf.device('/GPU:0'):
            W4 = tf.Variable(tf.random_normal((n_hidden, n_hidden), mean=0.0, stddev=0.005),name = 'gen_W4',trainable = trainable_model)
            b4 = tf.Variable(tf.random_normal((n_hidden,), mean=0.0, stddev=0.005),name = 'gen_b4',trainable = trainable_model)
            x_hidden = tf.nn.relu(tf.matmul(x_hidden, W4) + b4)            
    
    #output no activation
    with tf.name_scope("output"):
        with tf.device('/GPU:0'):
            W_out = tf.Variable(tf.random_normal((n_hidden, in_dim), mean=0.0, stddev=0.005),name='gen_W_out',trainable = trainable_model)
            b_out = tf.Variable(tf.random_normal((in_dim,), mean=0.0, stddev=0.005),name = 'gen_b_out',trainable = trainable_model)
            y_out = tf.matmul(x_hidden, W_out) + b_out
    
    #reconstruct with AUTOMAP
    with tf.name_scope('AUTOMAP'):
        corrupt_prob = np.asarray([0])
        x = tf.concat([x_input,x_input+y_out],axis = 0)
        recs, = tf.import_graph_def(g_AUTOMAP_def,input_map = {'x':x,'corrupt_prob':corrupt_prob},return_elements=['ycrop:0'],name='recs')
        batch_size = tf.cast(tf.shape(recs)[0]/2,dtype =tf.int32)
        rec = recs[:batch_size,:,:]
        rec_pert = recs[batch_size:,:,:]
        
    #objective loss
    with tf.name_scope("loss"):
        #use l1 and l2 to monitor the two different components of the loss separately
        l1 = tf.nn.l2_loss(rec_pert-rec)
        l2 = tf.clip_by_value(tf.nn.l2_loss(tf.norm(y_out,axis = 1)-epsilon),0,tf.float32.max)
        l = tf.scalar_mul(lambda1,l1) + tf.scalar_mul(lambda2,l2)

    #optimizer
    with tf.name_scope("optim"):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam').minimize(l)
        
    
    #initializer for the variables of the generator
    init = tf.global_variables_initializer()
    
    return {'x_input':x_input,'y_out':y_out,'loss':l,'opt':train_op,'l1':l1,'l2':l2,'epsilon':epsilon}

def Generator_Sample_Graph(g_AUTOMAP_def, n_hidden, lambda1, lambda2, learning_rate, trainable_model):
    with tf.Graph().as_default() as g_gen:          

        gen = Generator_Sample(g_AUTOMAP_def = g_AUTOMAP_def,in_dim = in_dim,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2,learning_rate = learning_rate, trainable_model = True)
        
    
    return g_gen,gen

def Adversarial_Training_Graph(batch_size,lambda1_adv,lambda2_adv,lambda1,lambda2,epsilon,learning_rate,in_dim,n_hidden,g_AUTOMAP_def):
    with tf.Graph().as_default() as g_adv_training:
        

        net_train = arch.network(2*batch_size, precision, resolution, in_dim, h_dim, out_dim,
                            model_in_vars=model_in_vars, 
                            model_in_shapes=model_in_shape, 
                            trainable_model_in='True');

        with tf.name_scope('Generator'):
            gen = Generator_Sample(g_AUTOMAP_def = g_AUTOMAP_def,in_dim = in_dim,n_hidden = n_hidden, lambda1 = lambda1, lambda2 = lambda2,learning_rate = learning_rate, trainable_model = True)
        
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'Generator')
        automap_vars = [v for v in vars if not v in generator_vars] 

        generator_minimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam').minimize(gen['loss'],var_list=generator_vars)
        
        x_train = net_train['x']
        y_train = net_train['ycrop']
        corrupt_prob = net_train['corrupt_prob']
        y_clean_sample = y_train[:batch_size,:,:]
        y_perturbed = y_train[batch_size:,:,:]
        
        y = tf.placeholder(tf.float32, (None, 128,128),name = 'original')
        
        #use to plot and visualize training procedure
        loss_clean = tf.nn.l2_loss(y_clean_sample-y)
        loss_pert = tf.nn.l2_loss(y_clean_sample-y_perturbed)
        
        loss_train = tf.scalar_mul(lambda1_adv,tf.nn.l2_loss(y_clean_sample-y)) + tf.scalar_mul(lambda2_adv,tf.nn.l2_loss(y_clean_sample-y_perturbed))

        train_op = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam').minimize(loss_train,var_list=automap_vars)
        init = tf.global_variables_initializer()
        
        
        return {'graph':g_adv_training, 'loss_train':loss_train,
                'loss_clean':loss_clean,'loss_pert':loss_pert,'train_op': train_op,
                'init':init, 'x_train':x_train,'y':y,'corrupt_prob':corrupt_prob,
                'AUTOMAP_net': net_train,'Generator':gen,'vars':vars,'gen_opt':generator_minimizer,
                'AUTOMAP_vars': automap_vars}