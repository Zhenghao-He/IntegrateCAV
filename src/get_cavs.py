from tcav.cav import get_or_train_cav
import pickle
from configs import concepts, bottlenecks, model_to_run, save_dir, concepts_string, num_random_exp
import glob
import numpy as np
import os

def get_cavs(concepts, bottlenecks, cav_dir=None,num_random_exp=10):
    cavs_instances = []
    for bottleneck in bottlenecks:
        cavs_layer = []
        for concept in concepts:
            for i in range(num_random_exp):
                pattern = f"**/{concept}*random500_{i}-*{bottleneck}*"
                search_path = os.path.join(cav_dir, pattern)
                files = glob.glob(search_path, recursive=True)
                for file in files:
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    cav = data['cavs'][0] # use the first CAV
                    # import pdb;pdb.set_trace()
                    cavs_layer.append(cav)
        cavs_instances.append(cavs_layer)
    return cavs_instances

if __name__ == "__main__":
    

    original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")

    cavs = get_cavs(concepts, bottlenecks, cav_dir=original_cavs_path, num_random_exp=num_random_exp)
    os.makedirs(os.path.join(original_cavs_path, "original_cavs"), exist_ok=True)
    cavs_array = np.array(cavs, dtype=object)
    np.save(os.path.join(original_cavs_path, f"cavs_{concepts_string}.npy"), cavs_array)

    import pdb;pdb.set_trace()
