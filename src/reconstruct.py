
import numpy as np
import os
from configs import concepts, bottlenecks, target
from align_dim import CAVAutoencoder
# import tensorflow as tf
import torch
import pickle
save_dir = "/p/realai/zhenghao/CAVFusion/analysis/" 
model_to_run = 'GoogleNet'

def save_cavs(concepts, bottleneck, cavs, save_path):
    """Save a dictionary of this CAV to a pickle."""
    save_dict = {
        'concepts': concepts,
        'bottleneck': bottleneck,
        'hparams': None,
        'accuracies': None,
        'cavs': cavs,
        'saved_path': save_path
    }
    with open(save_path, 'wb') as pkl_file:
        pickle.dump(save_dict, pkl_file)
    # if save_path is not None:
    #     with tf.io.gfile.GFile(save_path, 'w') as pkl_file:
    #         pickle.dump(save_dict, pkl_file)
    # else:
    #     tf.compat.v1.logging.info('save_path is None. Not saving anything')


if __name__ == "__main__":
    overwrite = True
    save_dir = "/p/realai/zhenghao/CAVFusion/analysis"
    embed_dim = 4096 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim_align_method="autoencoder"

    fuse_method="mean"
    cavs = np.load(os.path.join(save_dir,model_to_run,"cavs.npy"), allow_pickle=True)

    autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim, device=device, save_dir=os.path.join(save_dir,model_to_run))
    fused_cavs = np.load(os.path.join(save_dir, model_to_run,"fuse_model", dim_align_method, fuse_method, "fused_cavs.npy"), allow_pickle=True)
    reconstructed_save_dir = os.path.join(save_dir, model_to_run, "reconstructed_cavs", dim_align_method, fuse_method)
    os.makedirs(reconstructed_save_dir, exist_ok=True)
    for layer_idx, layer in enumerate(bottlenecks):
        decoder = autoencoders.load_autoencoder(layer_idx).decode
        reconstructed_cavs = []
        for fused_cav in fused_cavs:
            reconstructed = decoder(torch.tensor(fused_cav).to(device)).cpu().detach().numpy()
            reconstructed_cavs.append(reconstructed)
        save_cavs(concepts, layer, reconstructed_cavs, os.path.join(reconstructed_save_dir, f"reconstructed_{layer}_cavs.pkl"))
            


    