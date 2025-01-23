# this is a regularizer penalty parameter for linear classifier to get CAVs. 
import torch
alphas = [0.1]   
num_random_exp=3
target = 'zebra'  
# concepts = ["dotted","striped","zigzagged","animal","grass","lakeside","black","white","tiger","horse"]  # @param
concepts = ["dotted","striped","zigzagged"]  # @param
bottlenecks = [ 'mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  # @param 
save_dir = "/p/realai/zhenghao/CAVFusion/analysis/"
model_to_run = 'GoogleNet'
embed_dim = 4096 #dimension of the embedding space(used for both autoencoder and moco)
dim_align_method="autoencoder"
fuse_method="weight"
hidden_dims = []
# hidden_dims = [4096]
dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")