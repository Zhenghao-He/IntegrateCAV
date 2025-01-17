from IntegrateCAV import IntegrateCAV
import numpy as np
import os
import torch.autograd
from align_dim import CAVAutoencoder
torch.autograd.set_detect_anomaly(True)
save_dir = "/p/realai/zhenghao/CAVFusion/analysis/" 
model_to_run = 'GoogleNet'



if __name__ == "__main__":
    overwrite = False
    save_dir = "/p/realai/zhenghao/CAVFusion/analysis"
    embed_dim = 4096 #dimension of the embedding space(used for both autoencoder and moco)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_align_method="autoencoder"
    # dim_align_method="zero_padding"
    fuse_method="mean"
    cavs = np.load(os.path.join(save_dir,model_to_run,"cavs.npy"), allow_pickle=True)
    # import pdb; pdb.set_trace()
    autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim, device=device, save_dir=os.path.join(save_dir,model_to_run))
    autoencoders.train_autoencoders(cavs=cavs, overwrite=False, epochs=200) #train autoencoder/ we need decoders to reconstruct the cavs for each layer
    
    integrate_cav = IntegrateCAV(cavs=cavs, device=device, autoencoders=autoencoders,dim_align_method=dim_align_method).to(device)
    # align before fusion
    aligned_cavs = integrate_cav.align_with_moco(queue_size=256, momentum=0.999, temperature=0.07, embed_dim=embed_dim,overwrite=overwrite, epochs=300,save_dir=os.path.join(save_dir,model_to_run))

    fused_cavs = integrate_cav.fuse(fuse_method=fuse_method, overwrite=overwrite, save_dir=os.path.join(save_dir,model_to_run))

    '''
    TODO:
    1. project the fused cavs to each layer and calculate the tcav scores across different layers for one concept
    '''
    import pdb; pdb.set_trace()
