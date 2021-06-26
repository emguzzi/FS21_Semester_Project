# Update the paths so that they match the underlying data.
folder_path = '' #folder where the code is located
data_path_short = '' #shorter version of the dataset for faster experiments
data_path = ''
saved_models_path = ''#directory where the weights are stored after adversarial training
storage_path = '' #pah of the storage2 folder
src_weights  = storage_path + 'network_weights/AUTOMAP'; 
src_automap_runner = storage_path + 'AUTOMAP/runners';
src_mri_data = storage_path + 'AUTOMAP';
src_k_mask   = storage_path + 'AUTOMAP';