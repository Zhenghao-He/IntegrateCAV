
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from align_dim import augment_cavs


def prepare_batch(cavs,device,num_random_exp=3,batch_size=2):    
    """
    
    Args:
        cavs:  [n_layers, n_concepts, cav_dim]
        concept_idx: 
    Returns:
        cav_query: Tensor
        cav_key: Tensor
    """
    n_layers = len(cavs)
    concept_idx = random.randint(0, len(cavs[0])//num_random_exp-1)  #random sample one concept
    concept_indices = [concept_idx*num_random_exp + i for i in range(num_random_exp)]
    concepts_tensors = [cav[i] for cav in cavs for i in concept_indices]
    # concepts_tensors = [cav[concept_indices] for cav in cavs]
    # import pdb; pdb.set_trace()
    concepts_arr = [cav.cpu().numpy() for cav in concepts_tensors]

    half_len = len(concepts_arr) // 2
    if batch_size > half_len:
        raise ValueError("Batch size is larger than the number of concepts.")
    cav_query = random.sample(concepts_arr, batch_size)
    # import pdb; pdb.set_trace()
    cav_key = [x for x in concepts_arr if not any(np.array_equal(x, y) for y in cav_query)]
    if len(cav_key) > batch_size:
        cav_key = random.sample(cav_key, batch_size)

    aug_query = augment_cavs(cavs=cav_query, num_augments=2, noise_std=0.1) 
    aug_query = np.concatenate((aug_query, cav_query), axis=0)
    aug_query = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_query]
    aug_key = augment_cavs(cavs=cav_key, num_augments=2, noise_std=0.1) 
    aug_key = np.concatenate((aug_key, cav_key), axis=0)
    aug_key = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_key]



    # layer_query, layer_key = random.sample(range(n_layers), 2)  #random sample two layers
    # cav_q_arr = np.array([cav.cpu().numpy() for cav in cavs[layer_query]])
    # cav_k_arr = np.array([cav.cpu().numpy() for cav in cavs[layer_key]])
    # aug_query = augment_cavs(cavs=cav_q_arr, num_augments=1, noise_std=0.1)
    # aug_key = augment_cavs(cavs=cav_k_arr, num_augments=1, noise_std=0.1)
    # # import pdb; pdb.set_trace()
    # aug_query = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_query]
    # aug_key = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_key]

    cav_query = torch.stack(aug_query) # [n_concept_tensors, tensor_dim]
    cav_key = torch.stack(aug_key)  
    return cav_query, cav_key

class OrthogonalLinear(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(OrthogonalLinear, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)

    def orthogonal_regularization(self, weight_decay=1e-3):
        """Apply orthogonal regularization to the weight matrix."""
        W = self.fc.weight
        loss = torch.norm(torch.mm(W, W.T) - torch.eye(W.size(0), device=W.device))
        return weight_decay * loss
'''
loss += query_encoder.orthogonal_regularization()
query_encoder = OrthogonalLinear(input_dim=256, embed_dim=256).cuda()

'''
class MoCoCAV(nn.Module):
    def __init__(self, input_dim, embed_dim, cavs, queue_size=4096, momentum=0.999, temperature=0.07, device = "cuda"):
        super().__init__()
        self.device = device
        # self.query_encoder = nn.Linear(input_dim, embed_dim).to(self.device)
        # self.key_encoder = nn.Linear(input_dim, embed_dim).to(self.device)
        self.query_encoder = OrthogonalLinear(input_dim=input_dim, embed_dim=embed_dim).to(self.device)
        self.key_encoder = OrthogonalLinear(input_dim=input_dim, embed_dim=embed_dim).to(self.device)
        # Initialize queue
        self.register_buffer("queue", torch.randn(queue_size, embed_dim))
        self.queue = F.normalize(self.queue, dim=1).detach()
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.cavs = cavs
        # Preprocess cavs
        # cavs_tensor = torch.tensor(cavs, dtype=torch.float32).cuda()
        # cavs_array = np.array(cavs.cpu(), dtype=np.float32)  #
        # cavs_tensor = torch.tensor(cavs_array, dtype=torch.float32).cuda() 

        # Entire tensor for efficiency
        # num_layers = len(cavs)
        # num_concepts = len(cavs[0])
        # # num_layers = cavs_tensor.size(0)
        # # num_concepts = cavs_tensor.size(1)

        # queue_data = []
        # for i in range(queue_size):
        #     layer_idx = (i // num_concepts) % num_layers
        #     concept_idx = i % num_concepts
        #     # cav = cavs_tensor[layer_idx, concept_idx, :]
        #     # import pdb; pdb.set_trace()
        #     cav = cavs[layer_idx][concept_idx].unsqueeze(0)
        #     encoded_cav = F.normalize(self.key_encoder(cav), dim=1).detach()
        #     queue_data.append(encoded_cav.squeeze(0))

        # self.queue = torch.stack(queue_data)

        self.momentum = momentum
        self.temperature = temperature

    @torch.no_grad()
    def init_queue(self):
        num_layers = len(self.cavs)
        num_concepts = len(self.cavs[0])
        # num_layers = cavs_tensor.size(0)
        # num_concepts = cavs_tensor.size(1)

        queue_data = []
        for i in range(self.queue_size):
            layer_idx = (i // num_concepts) % num_layers
            concept_idx = i % num_concepts
            # cav = cavs_tensor[layer_idx, concept_idx, :]
            # import pdb; pdb.set_trace()
            cav = self.cavs[layer_idx][concept_idx].unsqueeze(0)
            encoded_cav = F.normalize(self.key_encoder(cav), dim=1).detach()
            queue_data.append(encoded_cav.squeeze(0))

        self.queue = torch.stack(queue_data)

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """update the key encoder by using momentum"""
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        queue_size = self.queue.size(0)
        assert batch_size <= self.queue.size(0), "queue size is too small to hold the current batch"

        
        # Number of items that can fit at the end of the queue
        remaining_space = queue_size - ptr

        if batch_size <= remaining_space:
            # Case 1: All keys fit in the remaining space
            self.queue[ptr:ptr + batch_size, :] = keys
        else:
            # Case 2: Split the keys into two parts
            self.queue[ptr:, :] = keys[:remaining_space, :]  # Fill the end of the queue
            self.queue[:batch_size - remaining_space, :] = keys[remaining_space:, :]  # Wrap around to the start

        # Update the pointer (modulo to ensure circular behavior)
        ptr = (ptr + batch_size) % queue_size
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
        q = F.normalize(self.query_encoder(cav_query), dim=1)
        k = F.normalize(self.key_encoder(cav_key), dim=1)
        k = k.detach()  # no gradient to key encoder
        # compute logits InfoNCE loss
        positive_sim = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # positive similarity
        negative_sim = torch.mm(q, self.queue.T)  # negative similarity

        logits = torch.cat([positive_sim, negative_sim], dim=1)  # concatenate positive and negative logits
        logits_with_t = logits / self.temperature  # temperature scaling

        labels = torch.zeros(logits_with_t.size(0), dtype=torch.long).to(self.device)  # positive index is 0
        loss = F.cross_entropy(logits_with_t, labels) + self.query_encoder.orthogonal_regularization()

        return loss, k



class IntegrateCAV(nn.Module):
    def __init__(self, cavs, device, autoencoders=None, dim_align_method="zero_padding", num_random_exp=3):
        super().__init__()
        self.cavs = cavs
        self.aligned_cavs = []
        self.fused_cavs = []
        self.device = device
        self.__isAligned = False
        self.autoencoders = autoencoders
        self.dim_align_method = dim_align_method
        self.num_random_exp = num_random_exp

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
                cav = torch.tensor(cav, dtype=torch.float32).to(self.device)
                cavs_layer_tmp.append(cav)
            cavs_same_dim.append(cavs_layer_tmp)
        return cavs_same_dim, max_dim
    

    def _align_dimension_by_ae(self):
        """
        Args:
            autoencoder: trained CAVAutoencoder model
            cavs: cavs for each layer [n_samples, input_dim]
        Returns:
            aligned_cavs: 
        """
        input_cavs = []
        for layer_idx, cavs_layer in enumerate(self.cavs):
            cavs_layer_tmp = []
            layer_autoencoder = self.autoencoders.load_autoencoder(layer_idx)
            for cav in cavs_layer:
                cav = torch.tensor(cav).to(self.device)
                cav = layer_autoencoder.encode(cav).detach()
                cavs_layer_tmp.append(cav)
            input_cavs.append(cavs_layer_tmp)
            del layer_autoencoder # release memory
        return input_cavs

    def align_with_moco(self, queue_size, momentum, temperature, embed_dim=2048, epochs = 2000, overwrite=False,save_dir="./analysis"):
        save_dir = os.path.join(save_dir,"align_model", self.dim_align_method)
        if not overwrite :
            if os.path.exists(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy")):
                print("Aligned CAVs already exist. Loading from saved files.")
                self.aligned_cavs = np.load(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
                print("Aligned CAVs loaded!")
                self.__isAligned = True
                return self.aligned_cavs
            elif os.path.exists(os.path.join(save_dir,f"query_encoder_{self.autoencoders.key_params}.pth")):
                print("Model already exists. Loading from saved files.")
                model = MoCoCAV(input_dim=embed_dim, embed_dim=embed_dim, cavs=self.cavs, queue_size=queue_size, momentum=momentum, temperature=temperature, device = self.device).to(self.device)
                model.query_encoder.load_state_dict(torch.load(os.path.join(save_dir,f"query_encoder_{self.autoencoders.key_params}.pth")))
                model.query_encoder.eval()
                with torch.no_grad():
                    for cavs_layer in self.cavs:
                        aligned_cavs_layer = []
                        for cav in cavs_layer:
                            cav = torch.tensor(cav).to(self.device)
                            cav = cav.unsqueeze(0)
                            cav = model.query_encoder(cav).squeeze(0)
                            aligned_cavs_layer.append(cav.cpu().numpy())
                        self.aligned_cavs.append(aligned_cavs_layer)
                print("CAVs aligned!")
                np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
                print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"))
                self.__isAligned = True
                return self.aligned_cavs
        else:
            print("Overwriting the existing files.")
            self.__isAligned = False
            
        
        if self.__isAligned:
            print("CAVs already aligned!")
            return self.aligned_cavs
        

        if self.autoencoders is None:
            raise ValueError("Autoencoders are not provided. Please provide the trained autoencoders.")
        
        if self.dim_align_method == "zero_padding":
            input_cavs, input_dim = self._align_dimension_by_zero()
        elif self.dim_align_method == "autoencoder":
            input_cavs = self._align_dimension_by_ae()
            input_dim = embed_dim
        else:
            raise NotImplementedError(f"Dimension alignment method {self.dim_align_method} is not implemented.")

        # for cavs_layer in input_cavs:
        #     cav_aligned_layer = []
        #     for cav in cavs_layer:
        #         cav_aligned_layer.append(cav.cpu().numpy())
        #     self.aligned_cavs.append(cav_aligned_layer)
        # print("CAVs aligned!")
        # np.save(os.path.join(save_dir,f"aligned_cavs(original)_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
        # print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs(original)_{self.autoencoders.key_params}.npy"))
        # self.__isAligned = True
        # return self.aligned_cavs
        
        model = MoCoCAV(input_dim=input_dim, embed_dim=embed_dim,cavs=input_cavs, queue_size=queue_size, momentum=momentum, temperature=temperature, device = self.device).to(self.device)
        model.init_queue()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            epoch_loss = 0
            # import pdb; pdb.set_trace()
            cav_query, cav_key = prepare_batch(cavs=input_cavs,device=self.device, batch_size=13)
            # cav_query, cav_key = torch.tensor(cav_query).cuda(), torch.tensor(cav_key).cuda()

            loss, k = model(cav_query, cav_key)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.momentum_update_key_encoder()
            model.dequeue_and_enqueue(k)
            epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        os.makedirs(save_dir, exist_ok=True)
        save_path  = os.path.join(save_dir,f"query_encoder_{self.autoencoders.key_params}.pth")
        torch.save(model.query_encoder.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        model.query_encoder.eval()
        with torch.no_grad():
            for cavs_layer in input_cavs:
                aligned_cavs_layer = []
                for cav in cavs_layer:
                    cav = torch.tensor(cav).to(self.device)
                    # import pdb; pdb.set_trace()
                    cav = cav.unsqueeze(0)
                    cav = model.query_encoder(cav).squeeze(0)
                    aligned_cavs_layer.append(cav.cpu().numpy())
                self.aligned_cavs.append(aligned_cavs_layer)
        print("CAVs aligned!")
        del model # release memory
        np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
        print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"))
        self.__isAligned = True
        return self.aligned_cavs

    def fuse(self,fuse_method="mean", overwrite=False, save_dir="./analysis"):
        """
        Fuse the aligned CAVs
        """
        if not self.__isAligned:
            raise ValueError("CAVs are not aligned. Please align the CAVs first.")

        save_dir = os.path.join(save_dir,"fuse_model", self.dim_align_method, fuse_method)
        if not overwrite:
            if os.path.exists(os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}.npy")):
                print("Fused CAVs already exist. Loading from saved files.")
                self.fused_cavs = np.load(os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
                print("Fused CAVs loaded!")
                return self.fused_cavs
        
            
        if fuse_method == "mean":
            fused_cavs = np.mean(np.array(self.aligned_cavs), axis=0)
        elif fuse_method == "attention":
            attention_fusion = AttentionFusion(embed_dim=len(self.aligned_cavs[0][0]), num_layers=len(self.aligned_cavs)).to(self.device)
            aligned_cavs = torch.tensor(self.aligned_cavs, dtype=torch.float32).to(self.device)
            fused_cavs = attention_fusion(aligned_cavs).cpu().detach().numpy()
        elif fuse_method == "weight":
            # import pdb; pdb.set_trace()
            # variances = [np.var(embed) for embed in self.aligned_cavs]
            fused_cavs = []
            planes = [self.aligned_cavs[:, i, :] for i in range(len(self.aligned_cavs[0]))]
            planes = np.array(planes)
            for i in range(len(planes)):
                variance = [np.var(embed) for embed in planes[i]]
                weight = [var / sum(variance) for var in variance]
                fused_cav = np.sum([weight * embed for weight, embed in zip(weight, self.aligned_cavs[i])], axis=0)
                fused_cavs.append(fused_cav)
            
            fused_cavs = np.array(fused_cavs)
            
        else:
            raise NotImplementedError(f"Fuse method {fuse_method} is not implemented.")
        self.fused_cavs = fused_cavs # [num_concepts, cav_dim]
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}.npy"), fused_cavs)
        print("Fused CAVs saved at", os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}.npy"))
        return self.fused_cavs

    
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super(AttentionFusion, self).__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))  # Learnable query vector
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, aligned_cavs):
        """
        Args:
            aligned_cavs: Tensor of shape [num_layers, batch_size, embed_dim]
        Returns:
            fused_cav: Tensor of shape [batch_size, embed_dim]
        """
        # Compute keys and values
        keys = self.key(aligned_cavs)        # Shape: [num_layers, batch_size, embed_dim]
        values = self.value(aligned_cavs)   # Shape: [num_layers, batch_size, embed_dim]

        # Compute attention scores
        query = self.query.expand(aligned_cavs.size(1), -1)  # Shape: [batch_size, embed_dim]
        scores = torch.einsum("bd,lbd ->lb", query, keys)  # Shape: [num_layers, batch_size]
        weights = self.softmax(scores)  # Shape: [num_layers, batch_size]

        # Weighted sum of values
        fused_cav = torch.einsum("lb,lbd->bd", weights, values)  # Shape: [batch_size, embed_dim]
        return fused_cav



