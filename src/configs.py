# this is a regularizer penalty parameter for linear classifier to get CAVs. 
import torch
alphas = [0.1]   
num_random_exp=2
target = 'zebra'  
# concepts = ["dotted","striped","zigzagged","animal","grass","lakeside","black","white","tiger","horse"]  # @param
concepts = ["striped"]  # @param["dotted",,,"honeycombed","zigzagged","striped""chequered"
concepts_string = "_".join(concept.replace(" ", "") for concept in concepts)
bottlenecks = [ 'mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  # @param 
save_dir = "/p/realai/zhenghao/CAVFusion/analysis/"
model_to_run = 'GoogleNet'
embed_dim = 4096 #dimension of the embedding space(used for both autoencoder and moco)
dim_align_method="autoencoder"
fuse_method="transformer"
# hidden_dims = [512]
hidden_dims = [4096]
dropout = 0.5
# fuse_input = "input_cavs"
fuse_input = "aligned_cavs"
concept_map_type = "reconstructed_cavs"
# concept_map_type = "original_cavs"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")