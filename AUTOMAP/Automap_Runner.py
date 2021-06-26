from config import src_automap_runner
from Runner import Runner;
from os.path import join;
from adversarial_tools import l2_norm_of_tensor; 
import numpy as np;
import pickle;



class Automap_Runner(Runner):
    """
    An extension of the Runner class, with a spesicalization to the AUTOMAP 
    network.
    """
    
    def save_runner(self, f, data_path=src_automap_runner):
        """
        Save the current runner object.

        Reads the current runner count and store the runner object at 
        data_path/data/ as 'runner_count.pkl'. In addition the file 
        'runner_description.txt' (located at data_path) is updated with
        statistics from the run. 

        :param f: Neural Network handle.
        :param data_path: Where to store the runner_object. 
                          The data_path directory should contain a subdirectory 
                          called 'data', where the object is stored.

        :returns: Runner count (int).
        """
        count = self._read_count(data_path);
        fname = join('data', 'runner_%d.pkl' % (count));
        fname = join(data_path, fname);
    
        with open(fname, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)
        
        fname = join(data_path, "runner_description.txt");
        
        
        x0   = self.x0[0];
        r    = self.r[-1];
        batch_size = x0.shape[0]; 

        # Evaluate the networks
        fx  = f(x0);
        fxr = f(x0+r);
        
        batch_size = self.x0[0].shape[0];
        network_name = "AUTOMAP";
        with open(fname, 'a') as outfile:

            outfile.write("""
--------    ------------------ %03d: %s --------------------------
            opt: %s, mom: %g, learn_rate: %g, smooth_esp: %g, 
            la: %g, max_r_norm: %g, max_diff_norm: %g,
            max_itr: %d
            ws: %s, wsf: %g, ps: %s psf: %g
----                                                             ----
""" % (
count, network_name, self.optimizer, self.momentum, self.learning_rate,
self.smoothing_eps, self.la, self.max_r_norm, self.max_diff_norm, self.max_itr,
self.warm_start, self.warm_start_factor, self.perp_start, self.perp_start_factor ));        

            for im_nbr in range(batch_size):
                x_i = x0[im_nbr];
                r_i = r[im_nbr];
                fx_i = fx[im_nbr];
                fxr_i = fxr[im_nbr];

                nx = l2_norm_of_tensor(x_i);
                nr = l2_norm_of_tensor(r_i);
                n_diff = l2_norm_of_tensor(fx_i - fxr_i);

                next_str = '%02d: |f(x)-f(x+r)|: %g, |r|: %g, |f(x)-f(x+r)|/|r|: %g |r|/|x|: %g\n' % \
                       (im_nbr, abs(n_diff), nr, abs(n_diff)/nr, nr/nx );
                outfile.write(next_str);

        
        #Added lines: save the norm and the effect of the all the perturbations that we found 
        
        n_r = []
        n_diff = []
        
        for i in range(len(self.r)):
            
            f_xr = f(x0+self.r[i])
            n_r_temp = []
            n_diff_temp = []
            
            for j in range(batch_size):
                n_r_temp.append(l2_norm_of_tensor(self.r[i][j]))
                n_diff_temp.append(l2_norm_of_tensor(f_xr[j]-fx[j]))
            
            n_r.append(n_r_temp)
            n_diff.append(n_diff_temp)
        
        with open(src_automap_runner + 'list_norm_r.pkl','wb') as f:
            pickle.dump(n_r,f)
        
        with open(src_automap_runner + 'list_norm_diff.pkl','wb') as f:
            pickle.dump(n_diff,f)    
            
        
        return count;
