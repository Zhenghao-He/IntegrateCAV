from IntegrateCAV import IntegrateCAV
import numpy as np
import os
import torch.autograd
from align_dim import CAVAutoencoder
from configs import embed_dim, hidden_dims, dropout, dim_align_method, fuse_method, model_to_run, save_dir, num_random_exp,concepts_string, bottlenecks,concepts,target
torch.autograd.set_detect_anomaly(True)




if __name__ == "__main__":
    overwrite = True

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
    cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts_string}.npy"), allow_pickle=True)
    # import pdb; pdb.set_trace()
  
    autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim,hidden_dims=hidden_dims, dropout=dropout, device=device, save_dir=os.path.join(save_dir,model_to_run), overwrite=False)
    autoencoders.train_autoencoders(cavs=cavs, epochs=40, batch_size=32) #train autoencoder/ we need decoders to reconstruct the cavs for each layer
    # raise ValueError("stop here")
    integrate_cav = IntegrateCAV(cavs=cavs, device=device, autoencoders=autoencoders,dim_align_method=dim_align_method,num_random_exp=num_random_exp,save_dir=os.path.join(save_dir,model_to_run)).to(device)
    # align before fusion
    # aligned_cavs = integrate_cav.align_with_moco(queue_size=100, momentum=0.999, temperature=0.07, embed_dim=embed_dim,overwrite=overwrite, epochs=1000)
    aligned_cavs = integrate_cav.train(embed_dim=embed_dim,overwrite=overwrite, epochs=1000, batch_size=32, lr=1e-5)
    import pdb; pdb.set_trace()
    fused_cavs = integrate_cav.fuse(fuse_method=fuse_method,bottlenecks=bottlenecks,concepts=concepts, num_random_exp=num_random_exp, target=target, overwrite=overwrite)

    '''
    TODO:
    1. project the fused cavs to each layer and calculate the tcav scores across different layers for one concept
    '''
    import pdb; pdb.set_trace()
