# this is a regularizer penalty parameter for linear classifier to get CAVs. 
import torch
import tcav.utils as utils
import tcav.model  as model
import os
alphas = [0.1]   
num_random_exp=10
target = 'zebra'  
# concepts = ["dotted","striped","zigzagged","animal","grass","lakeside","black","white","tiger","horse"]  # @param
concepts = ["dotted","striped","zigzagged","chequered","honeycombed","scaly"]  # 
concepts_string = "_".join(concept.replace(" ", "") for concept in concepts)
bottlenecks = ['expanded_conv_2', 'expanded_conv_4', 'expanded_conv_5', 'expanded_conv_7', 'expanded_conv_8', 'expanded_conv_9', 'expanded_conv_11', 'expanded_conv_12', 'expanded_conv_14', 'expanded_conv_15']  # @param 
# bottlenecks = [ 'mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  # @param 
save_dir = "/p/realai/zhenghao/CAVFusion/analysis/"
# model_to_run = 'ResNet50V2'
model_to_run = 'MobileNetV2'
# model_to_run = 'GoogleNet'
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
is_attack = False
# is_attack = True
attacked_layer_name = "mixed3a"


source_dir = '/p/realai/zhenghao/CAVFusion/data'
user = 'zhenghao'
# the name of the parent directory that results are stored (only if you want to cache)
project_name = 'tcav_class_test'
working_dir = "/tmp/" + user + '/' + project_name + '/' + model_to_run
# where activations are stored (only if your act_gen_wrapper does so)
sess = utils.create_session()


if not is_attack:
    activation_dir =  working_dir+ '/activations/'
    cav_dir = os.path.join("/p/realai/zhenghao/CAVFusion/analysis/", model_to_run, "original_cavs")
else:
    activation_dir =  working_dir+ '/attacked_activations/'
    cav_dir = os.path.join("/p/realai/zhenghao/CAVFusion/analysis/", model_to_run, "attacked_cavs")

if model_to_run == 'ResNet50V2':
    LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"
    GRAPH_PATH = source_dir + "/resnet50_v2/resnet50v2_frozen.pb"
    mymodel = model.ResNet50V2Wrapper_public(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
elif model_to_run == 'MobileNetV2':
    GRAPH_PATH = source_dir + "/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb"
    LABEL_PATH = source_dir + "/mobilenet_v2_1.0_224/mobilenet_v2_label_strings.txt"
    mymodel = model.MobilenetV2Wrapper_public(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
elif model_to_run == 'GoogleNet':
    GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
    LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"
    mymodel = model.GoogleNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)