
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import random



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
    cav_query = torch.tensor([cav for cav in cavs[layer_query]]).cuda()
    cav_key = torch.tensor([cav for cav in cavs[layer_key]]).cuda()
    # cav_query = []
    # for cav in cavs[layer_query]:
    #     cav_query.append(cav)
    # cav_query = np.array(cav_query)
    # cav_key = []
    # for cav in cavs[layer_key]:
    #     cav_key.append(cav)
    # cav_key = np.array(cav_key)
    return cav_query, cav_key


class MoCoCAV(nn.Module):
    def __init__(self, input_dim, embed_dim, cavs, queue_size=4096, momentum=0.999, temperature=0.07):
        """
        MoCo aligns CAVs using a momentum encoder and a dynamic queue.
        Args:
            input_dim: CAV 
            embed_dim: 
            cavs: initialize the queue with CAVs
            queue_size: 
            momentum: 
            temperature: 
        """
        super().__init__()
        self.query_encoder = nn.Linear(input_dim, embed_dim).to("cuda")
        self.key_encoder = nn.Linear(input_dim, embed_dim).to("cuda")
        

        # dynamic queue

        self.register_buffer("queue", torch.randn(queue_size, embed_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        num_layers = len(cavs)
        num_concepts = len(cavs[0])
        # import pdb; pdb.set_trace()
        for i in range(queue_size):
            layer_idx = (i // num_concepts) % num_layers  
            concept_idx = i % num_concepts     
            cav = torch.tensor(cavs[layer_idx][concept_idx], dtype=torch.float32).cuda() 
            encoded_cav = F.normalize(self.key_encoder(cav), dim=0).clone() 
            self.queue[i, :] = encoded_cav

        # import pdb; pdb.set_trace()

        self.momentum = momentum
        self.temperature = temperature

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """update the key encoder by using momentum"""
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert batch_size <= self.queue.size(0), "queue size is too small to hold the current batch"

        
        self.queue[ptr:ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.queue.size(0)  # move pointer 
        self.queue_ptr[0] = ptr

    def forward(self, cav_query, cav_key):
        """
        Args:
            cav_query: 
            cav_key: 
        Returns:
            loss: 
        """
        # normalize the features
        # import pdb; pdb.set_trace()
        q = F.normalize(self.query_encoder(cav_query), dim=1).clone()
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k = F.normalize(self.key_encoder(cav_key), dim=1).clone()

        # compute logits InfoNCE loss
        positive_sim = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # positive similarity
        negative_sim = torch.mm(q, self.queue.T)  # negative similarity

        logits = torch.cat([positive_sim, negative_sim], dim=1)  # concatenate positive and negative logits
        logits /= self.temperature  # temperature scaling

        labels = torch.zeros(logits.size(0), dtype=torch.long).cuda()  # positive index is 0
        loss = F.cross_entropy(logits, labels)

        # update the queue
        self._dequeue_and_enqueue(k)

        return loss



class IntegrateCAV(nn.Module):
    def __init__(self, cavs):
        super().__init__()
        self.cavs = cavs
        self.aligned_cavs = []
        
    
    def _align_dimension_by_zero(self):
        """
        Align the dimensions of CAVs by zero padding
        """
        cavs_same_dim = []
        max_dim = max([len(cav) for cavs_layer in self.cavs for cav in cavs_layer])
        for cavs_layer in self.cavs:
            cavs_layer_tmp = []
            for cav in cavs_layer:
                if len(cav) < max_dim:
                    cav = np.pad(cav, (0, max_dim - len(cav)), 'constant')
                cavs_layer_tmp.append(cav)
            cavs_same_dim.append(cavs_layer_tmp)
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
            layer_cavs = torch.tensor(layer_cavs, dtype=torch.float32).cuda()
            _, embedded = autoencoder(layer_cavs, layer_idx)  
            aligned_cavs.append(embedded.cpu().detach().numpy())
        return aligned_cavs

    def align(self, embed_dim=2048, epochs = 2, dim_align_method="zero_padding"):

        if dim_align_method == "zero_padding":
            input_cavs, input_dim = self._align_dimension_by_zero()
        else:
            raise NotImplementedError(f"Dimension alignment method {dim_align_method} is not implemented.")

        model = MoCoCAV(input_dim=input_dim, embed_dim=embed_dim,cavs=input_cavs, queue_size=256, momentum=0.999, temperature=0.07).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        
        epochs = 2
        for epoch in range(epochs):
            epoch_loss = 0

            cav_query, cav_key = prepare_batch(input_cavs)
            # cav_query, cav_key = torch.tensor(cav_query).cuda(), torch.tensor(cav_key).cuda()

            loss = model(cav_query, cav_key)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        model.query_encoder.eval()
        with torch.no_grad():
            for cavs_layer in input_cavs:
                aligned_cavs_layer = []
                for cav in cavs_layer:
                    cav = torch.tensor(cav).cuda()
                    cav = model.query_encoder(cav)
                    aligned_cavs_layer.append(cav.cpu().numpy())
                self.aligned_cavs.append(aligned_cavs_layer)

