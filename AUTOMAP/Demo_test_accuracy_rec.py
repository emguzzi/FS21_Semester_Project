import sys
from automap_tools import *
import tensorflow as tf 
from utils import *
from config import src_weights, folder_path, data_path, saved_models_path
from models import *
import pandas as pd

################################################################################
# Test the quality of the reconstruction with the original or robust weights.
################################################################################

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()
batch_size = 64
robust_model = True
if robust_model:
    path_automap_weights = ''
    [W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = path_automap_weights,robust = robust_model)

else:
    [W1_fc, b1_fc, W2_fc, b2_fc, W1_cnv, b1_cnv, W2_cnv, b2_cnv, W1_dcv, b1_dcv] = AUTOMAP_weights(src_weights = src_weights,robust = robust_model)

data_train, data_test, data_test_symbols = read_data(data_path,k_mask_idx1,k_mask_idx2,symbol = True)


g1, g1_def, AUTOMAP_net =  AUTOMAP_Graph(batch_size = data_test.shape[0],trainable = 'False')
sess = tf.Session(graph = g1)
init = g1.get_operation_by_name('init')
sess.run(init, feed_dict = {
                            AUTOMAP_net['W1_fc'].name:  W1_fc,  AUTOMAP_net['b1_fc'].name:  b1_fc,
                            AUTOMAP_net['W2_fc'].name:  W2_fc,  AUTOMAP_net['b2_fc'].name:  b2_fc, 
                            AUTOMAP_net['W1_cnv'].name: W1_cnv, AUTOMAP_net['b1_cnv'].name: b1_cnv,
                            AUTOMAP_net['W2_cnv'].name: W2_cnv, AUTOMAP_net['b2_cnv'].name: b2_cnv, 
                            AUTOMAP_net['W1_dcv'].name: W1_dcv, AUTOMAP_net['b1_dcv'].name: b1_dcv
                            })
#change the dataset to sample in order to consider images with or without the added symbol
#data_test_sampled = sample_image(data_test, k_mask_idx1, k_mask_idx2)
data_test_sampled = sample_image(data_test_symbols, k_mask_idx1, k_mask_idx2)
data_test = data_test_symbols

rec = sess.run(AUTOMAP_net['ycrop'],feed_dict = {AUTOMAP_net['x']:data_test_sampled,AUTOMAP_net['corrupt_prob']:np.asarray([0],dtype=np.float32)})
diff = np.linalg.norm(data_test-rec,axis = (1,2))
mean = np.mean(diff)
summary = np.zeros((data_test.shape[0],1))
summary[:,0] = diff
df = pd.DataFrame(summary,columns = ['|x_i-f(y_i)|'])
df.to_csv(folder_path+'Test_Accuracy/diff_list.csv')
print("Mean l2 diff: %f" % (mean),file = open(folder_path+'Test_Accuracy/log.txt','w'))
#save 5 images as example
for i in range(5):
    filename = 'Symbol%d.pdf'%(i)
    vmax = np.amax(rec[200+i,:,:])
    vmin = np.amin(rec[200+i,:,:])
    plt.imsave(folder_path+'Test_Accuracy/Reconstructions_Symbols/'+filename,rec[200+i,:,:],vmin=vmin, vmax=vmax, cmap='gray')
