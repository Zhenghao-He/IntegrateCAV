from tcav.cav import get_or_train_cav

def get_cavs(concepts, bottlenecks, activation_generator, cav_dir=None, cav_hparams=None, overwrite=False):
    cavs_instances = []
    for bottleneck in bottlenecks:
        cavs_layer = []
        acts = activation_generator.process_and_load_activations([bottleneck], concepts)
        cav_instances = get_or_train_cav(concepts,bottleneck, acts, cav_dir, cav_hparams, overwrite)
        for concept in concepts:
            cavs_layer.append(cav_instances.get_direction(concept))
        # cavs_layer.append(get_or_train_cav(concepts,bottleneck, acts, cav_dir, cav_hparams, overwrite).get_direction(concept) for concept in concepts)
        cavs_instances.append(cavs_layer)
    return cavs_instances