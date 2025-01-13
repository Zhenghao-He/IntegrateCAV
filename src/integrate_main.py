from IntegrateCAV import IntegrateCAV
import numpy as np
import os
import torch.autograd
torch.autograd.set_detect_anomaly(True)
save_path = "/p/realai/zhenghao/CAVFusion/analysis/" 
model_to_run = 'GoogleNet'
if __name__ == "__main__":
    
    dim_align_method="zero_padding"
    cavs = np.load(os.path.join(save_path,model_to_run,"cavs.npy"), allow_pickle=True)
    intergrate_cav = IntegrateCAV(cavs=cavs)
    intergrate_cav.align(dim_align_method=dim_align_method)
