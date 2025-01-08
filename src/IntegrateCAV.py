from tcav.cav import CAV  
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class IntegrateCAV(CAV):
    def __init__(self, concepts, bottleneck, hparams, save_path=None):
        super().__init__(concepts, bottleneck, hparams, save_path)

    def align( concepts,
                bottleneck,
                acts,
                cav_dir=None,
                cav_hparams=None,
                overwrite=False):
        
        if cav_hparams is None:
            cav_hparams = CAV.default_hparams()

        aligned_cavs = {}

        # Prepare data for alignment
        all_activations = []
        all_labels = []
        for concept in concepts:
            if bottleneck not in acts[concept]:
                raise ValueError(f"Missing activations for concept {concept} at bottleneck {bottleneck}")
            
            # Collect activations and labels
            activations = acts[concept][bottleneck]
            all_activations.append(activations)
            all_labels.extend([concept] * activations.shape[0])

        # Combine all activations into a single matrix
        all_activations = np.vstack(all_activations)
        all_labels = np.array(all_labels)

        # Perform PCA to map all activations to a shared space
        pca = PCA(n_components=min(len(concepts), all_activations.shape[1]))
        pca.fit(all_activations)
        aligned_space = pca.transform(all_activations)

        # Train a linear model for each concept in the shared space
        for concept in concepts:
            concept_mask = (all_labels == concept)
            concept_activations = aligned_space[concept_mask]

            # Use linear regression to align concept-specific CAV
            reg = LinearRegression()
            reg.fit(concept_activations, np.ones(concept_activations.shape[0]))
            aligned_cavs[concept] = reg.coef_

        # Optionally save aligned CAVs
        if cav_dir:
            save_path = os.path.join(cav_dir, f"aligned_cavs_{bottleneck}.pkl")
            if not overwrite and tf.io.gfile.exists(save_path):
                tf.compat.v1.logging.info(f"CAV already exists at {save_path}. Skipping save.")
            else:
                with tf.io.gfile.GFile(save_path, 'wb') as f:
                    pickle.dump(aligned_cavs, f)

        return aligned_cavs

    def fuse(self, cav):
        pass