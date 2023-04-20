import numpy as np 
import os
from .loading_data import to_numpy

def saving_files(in_files, out_files, NN_out_files,  NN_name, database,  PATH, realization_k):
    """
    Saving the files in the directory OOD/database/realization_k
    """
    saving_dir = f'{PATH}/{database}/realization_{realization_k}'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    if not isinstance(in_files, np.ndarray):    
        in_files = to_numpy(in_files)
    if not isinstance(out_files, np.ndarray):
        out_files = to_numpy(out_files)
    if not isinstance(NN_out_files, np.ndarray):
        NN_out_files = to_numpy(NN_out_files)

    np.save(os.path.join(saving_dir, f"wavespeed_{database}.npy"), in_files)
    np.save(os.path.join(saving_dir, f"pressure_{database}.npy"), out_files)
    np.save(os.path.join(saving_dir, f"pressure_{NN_name}_{database}.npy"), NN_out_files)
    