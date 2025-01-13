from tcav.cav import CAV  
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os
import tensorflow as tf
from tcav import utils
from tcav.cav import get_or_train_cav
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import random
tf.config.run_functions_eagerly(True)
def prepare_batch(cavs):
    """
    
    Args:
        cavs:  [n_layers, n_concepts, cav_dim]
        concept_idx: 
    Returns:
        cav_query: 
        cav_key: 
    """
    n_layers = len(cavs)
    layer_query, layer_key = random.sample(range(n_layers), 2)  #random sample two layers
    cav_query = []
    for cav in cavs[layer_query]:
        cav_query.append(cav)
    cav_query = np.array(cav_query)
    cav_key = []
    for cav in cavs[layer_key]:
        cav_key.append(cav)
    cav_key = np.array(cav_key)
    return cav_query, cav_key



class MoCoCAV(tf.keras.Model):
    def __init__(self, input_dim, embed_dim, cavs, queue_size=4096, momentum=0.999, temperature=0.07):
        """
        MoCo aligns CAVs using a momentum encoder and a dynamic queue.
        Args:
            input_dim: Input dimensionality of CAVs.
            embed_dim: Dimensionality of embedding space.
            cavs: List of CAVs to initialize the queue.
            queue_size: Size of the dynamic queue.
            momentum: Momentum for key encoder update.
            temperature: Temperature for InfoNCE loss.
        """
        super(MoCoCAV, self).__init__()

        # Define encoders
        self.query_encoder = tf.keras.layers.Dense(embed_dim, input_shape=(input_dim,))
        self.key_encoder = tf.keras.layers.Dense(embed_dim, input_shape=(input_dim,))

        # Dynamic queue
        self.queue = tf.Variable(tf.random.normal([queue_size, embed_dim]), trainable=False)
        # self.queue = tf.nn.l2_normalize(self.queue, axis=1)
        self.queue_ptr = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Initialize queue with CAVs
        num_layers = len(cavs)
        num_concepts = len(cavs[0])
        idx = 0
        for i in range(queue_size):
            layer_idx = (i // num_concepts) % num_layers
            concept_idx = i % num_concepts
            cav = tf.convert_to_tensor(cavs[layer_idx][concept_idx], dtype=tf.float32)
            encoded_cav = tf.nn.l2_normalize(self.key_encoder(cav[tf.newaxis, :]), axis=1)
            self.queue[idx].assign(encoded_cav[0])
            idx += 1

        # Parameters
        self.momentum = momentum
        self.temperature = temperature

    # @tf.function
    def _momentum_update_key_encoder(self):
        """Update the key encoder by using momentum."""
        for w_q, w_k in zip(self.query_encoder.trainable_weights, self.key_encoder.trainable_weights):
            w_k.assign(self.momentum * w_k + (1 - self.momentum) * w_q)

    # @tf.function
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with the new keys."""
        batch_size = tf.shape(keys)[0]
        ptr = self.queue_ptr
        # import pdb; pdb.set_trace()
        # assert batch_size <= self.queue.shape[0], "Queue size is too small to hold the current batch."

        self.queue[ptr:ptr + batch_size].assign(keys)
        self.queue_ptr.assign((ptr + batch_size) % self.queue.shape[0])

    def call(self, cav_query, cav_key):
        """
        Args:
            cav_query: Query CAVs (batch).
            cav_key: Key CAVs (batch).
        Returns:
            loss: InfoNCE loss.
        """
        # Normalize the features
        q = tf.nn.l2_normalize(self.query_encoder(cav_query), axis=1)
        # Update the key encoder with momentum (not calculating gradients)
        self._momentum_update_key_encoder()
        k = tf.nn.l2_normalize(self.key_encoder(cav_key), axis=1)
        # Stop gradients here to ensure no backpropagation through `k`
        k = tf.stop_gradient(k)

        # Momentum update for key encoder
        self._momentum_update_key_encoder()

        # Compute logits (InfoNCE loss)
        positive_sim = tf.reduce_sum(q * k, axis=1, keepdims=True)  # Positive similarity
        negative_sim = tf.matmul(q, self.queue, transpose_b=True)  # Negative similarity

        logits = tf.concat([positive_sim, negative_sim], axis=1)  # Concatenate positive and negative logits
        logits /= self.temperature  # Temperature scaling

        labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)  # Positive index is 0
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

        # Update the queue
        self._dequeue_and_enqueue(k)

        return loss

class IntegrateCAV(CAV):
    def __init__(self, concepts, bottlenecks, hparams, save_path=None):
        self.bottlenecks = bottlenecks
        self.cav_instances = []
        self.aligned_cavs = []
        super().__init__(concepts, bottlenecks[0], hparams, save_path)

    def get_cavs(self, activation_generator, cav_dir=None, cav_hparams=None, overwrite=False):
        for bottleneck in self.bottlenecks:
            acts = activation_generator.process_and_load_activations([bottleneck], self.concepts)
            self.cav_instances.append(get_or_train_cav(self.concepts,bottleneck, acts, cav_dir, cav_hparams, overwrite))
        return self.cav_instances
    
    def _align_dimension_by_zero(self):
        """
        Align the dimensions of CAVs by zero padding
        """
        cavs_same_dim = []
        max_dim = max([len(cav.get_direction(concept)) for cav in self.cav_instances for concept in self.concepts])
        for cav in self.cav_instances:
            cavs_layer = []
            for concept in self.concepts:
                cav_array = cav.get_direction(concept)
                if len(cav_array) < max_dim:
                    cav_array = np.pad(cav_array, (0, max_dim - len(cav_array)), 'constant')
                cavs_layer.append(cav_array)
            cavs_same_dim.append(cavs_layer)
        return cavs_same_dim, max_dim
    

    def _align_dimension_by_ae(autoencoder, cavs):
        """
        Args:
            autoencoder: trained CAVAutoencoder model
            cavs: cavs for each layer [n_samples, input_dim]
        Returns:
            aligned_cavs: 
        """
        aligned_cavs = []
        for layer_idx, layer_cavs in enumerate(cavs):
            layer_cavs = tf.convert_to_tensor(layer_cavs, dtype=tf.float32)
            _, embedded = autoencoder(layer_cavs, layer_idx)  # Pass through autoencoder
            aligned_cavs.append(embedded.numpy())
        return aligned_cavs

    def align(self, embed_dim=2048, epochs = 2, learning_rate = 1e-3, dim_align_method="zero_padding"):

        if dim_align_method == "zero_padding":
            input_cavs, input_dim = self._align_dimension_by_zero()
        else:
            raise NotImplementedError(f"Dimension alignment method {dim_align_method} is not implemented.")

        model = MoCoCAV(input_dim=input_dim, embed_dim=embed_dim, cavs=input_cavs, queue_size=256, momentum=0.999, temperature=0.07)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Prepare batch (replace `prepare_batch` with actual batch preparation logic)
            cav_query, cav_key = prepare_batch(input_cavs)
            cav_query, cav_key = tf.convert_to_tensor(cav_query, dtype=tf.float32), tf.convert_to_tensor(cav_key, dtype=tf.float32)

            with tf.GradientTape() as tape:
                loss = model(cav_query, cav_key)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += loss.numpy()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Align CAVs
        aligned_cavs = []
        for cavs_layer in input_cavs:
            aligned_cavs_layer = []
            for cav in cavs_layer:
                cav = tf.convert_to_tensor(cav, dtype=tf.float32)
                aligned_cav = model.query_encoder(cav[tf.newaxis, :])
                aligned_cavs_layer.append(aligned_cav.numpy().squeeze())
            aligned_cavs.append(aligned_cavs_layer)

        return aligned_cavs
