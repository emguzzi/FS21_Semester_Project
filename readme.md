**Adversarial Training for Deep Learning in Image Reconstruction**

This repository contains the code of my semester project 'Adversarial Training for Deep Learning in Image Reconstruction'. The code is based on the work 'On instabilities of deep learning in image reconstruction - Does AI come at a cost?' by V. Antun et al. (see https://github.com/vegarant/Invfool). Hence, some of the file are lightly modified or copied from there.

**Requirements**

The code runs with Python 3.6.4 and Tensorflow 1.7. In order to reproduce the results, the data can be downloaded from https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage2.zip. For our code only the AUTOMAP part of the folder is needed. The folder contains a small dataset, the full dataset has to be downloaded and preprocessed as described in the work. To run the code the paths in the `config.py` file has to be modified to point to the right files and directories. Also, the directory `py_adv_tools` needs to be added to the python path. 
