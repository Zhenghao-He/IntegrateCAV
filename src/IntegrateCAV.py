
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from align_dim import augment_cavs

def normalize_tensor(x, eps=1e-8):
    """
    标准化单个张量。
    
    :param x: 输入张量，形状为 [embedding_dim]
    :param eps: 小常数，避免除零
    :return: 标准化后的张量
    """
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)

def normalize_cav(cavs, eps=1e-8):
    """
    标准化CAV张量。
    
    :param cavs: 输入CAV张量形状为 [num_layers, num_cavs, embedding_dim] 或 [num_cavs, embedding_dim]
    :param eps: 小常数，避免除零
    :return: 标准化后的CAV张量
    """
    result = []
    for layer_cavs in cavs:
        tmp = []
        for cav in layer_cavs:
            cav1 = normalize_tensor(cav, eps)
            tmp.append(cav1)
        result.append(tmp)
    return result



def cosine_similarity(a, b):
    dot_product = np.dot(a, b)  
    norm_a = np.linalg.norm(a)  
    norm_b = np.linalg.norm(b)  
    return dot_product / (norm_a * norm_b)

def prepare_batch(cavs,device,num_random_exp=3,batch_size=2):    
    """
    
    Args:
        cavs:  [n_layers, n_concepts, cav_dim]
        concept_idx: 
    Returns:
        cav_query: Tensor
        cav_key: Tensor
    """
    
    # concept_idx = random.randint(0, len(cavs[0])//num_random_exp-1)  #random sample one concept
    # concept_indices = [concept_idx*num_random_exp + i for i in range(num_random_exp)]
    # concepts_tensors = [cav[i] for cav in cavs for i in concept_indices]
    # # concepts_tensors = [cav[concept_indices] for cav in cavs]
    # # import pdb; pdb.set_trace()
    # concepts_arr = [cav.cpu().numpy() for cav in concepts_tensors]

    # half_len = len(concepts_arr) // 2
    # if batch_size > half_len:
    #     raise ValueError("Batch size is larger than the number of concepts.")
    # cav_query = random.sample(concepts_arr, batch_size)
    # # import pdb; pdb.set_trace()
    # cav_key = [x for x in concepts_arr if not any(np.array_equal(x, y) for y in cav_query)]
    # if len(cav_key) > batch_size:
    #     cav_key = random.sample(cav_key, batch_size)

    # aug_query = augment_cavs(cavs=cav_query, num_augments=2, noise_std=0.1) 
    # aug_query = np.concatenate((aug_query, cav_query), axis=0)
    # aug_query = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_query]
    # aug_key = augment_cavs(cavs=cav_key, num_augments=2, noise_std=0.1) 
    # aug_key = np.concatenate((aug_key, cav_key), axis=0)
    # aug_key = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_key]


    n_layers = len(cavs)
    layer_query, layer_key = random.sample(range(n_layers), 2)  #random sample two layers
    cav_q_arr = np.array([cav.cpu().numpy() for cav in cavs[layer_query]])
    cav_k_arr = np.array([cav.cpu().numpy() for cav in cavs[layer_key]])
    aug_query = augment_cavs(cavs=cav_q_arr, num_augments=1, noise_std=0.1)
    aug_key = augment_cavs(cavs=cav_k_arr, num_augments=1, noise_std=0.1)
    # import pdb; pdb.set_trace()
    aug_query = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_query]
    aug_key = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_key]

    cav_query = torch.stack(aug_query) # [n_concept_tensors, tensor_dim]
    cav_key = torch.stack(aug_key)  
    return cav_query, cav_key



# class OrthogonalTransformer(nn.Module):
#     def __init__(self, input_dim, embed_dim, num_heads=4, ff_dim=256, dropout=0.1):
#         """
#         Replaces the fully connected layer with a simplified Transformer block.
#         - input_dim: Dimension of input features.
#         - embed_dim: Dimension of the output embedding.
#         - num_heads: Number of attention heads.
#         - ff_dim: Dimension of the feed-forward layer.
#         - dropout: Dropout probability.
#         """
#         super(OrthogonalTransformer, self).__init__()
#         self.embed_dim = embed_dim

#         # Multi-Head Self-Attention
#         self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
#         # Feed-Forward Network
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, ff_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_dim, embed_dim)
#         )

#         # Normalization layers
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#         # Positional encoding (optional for transformer-like structure)
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 1, embed_dim))

#     def forward(self, x):
#         # Expand input to match the embedding dimension
#         if x.size(-1) != self.embed_dim:
#             x = nn.Linear(x.size(-1), self.embed_dim)(x)
        
#         # Add positional encoding
#         x = x + self.positional_encoding
        
#         # Self-Attention layer with residual connection
#         attn_output, _ = self.self_attention(x, x, x)
#         x = self.norm1(x + attn_output)
        
#         # Feed-Forward Network with residual connection
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + ffn_output)

#         # Normalize the output embeddings
#         return F.normalize(x, dim=-1)

#     def orthogonal_regularization(self, weight_decay=1e-1):
#         """
#         Apply orthogonal regularization to the weight matrices of the attention and FFN.
#         """
#         loss = 0.0

#         # Regularize the projection matrices of Multi-Head Attention
#         for weight in [self.self_attention.in_proj_weight, 
#                        self.self_attention.out_proj.weight]:
#             W = weight
#             loss += torch.norm(torch.mm(W, W.T) - torch.eye(W.size(0), device=W.device))

#         # Regularize the feed-forward network weights
#         for layer in self.ffn:
#             if isinstance(layer, nn.Linear):
#                 W = layer.weight
#                 loss += torch.norm(torch.mm(W, W.T) - torch.eye(W.size(0), device=W.device))

#         return weight_decay * loss


class TransformerQueryEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=2048, dropout=0.1):
        """
        Transformer-based query encoder for MoCo.
        Args:
            embed_dim: Dimension of the input and output embeddings.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads in the Transformer.
            ff_dim: Dimension of the feed-forward network.
            dropout: Dropout rate for regularization.
        """
        super(TransformerQueryEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="relu",
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim].
        Returns:
            Transformed query embeddings of shape [batch_size, embed_dim].
        """
        # Apply Transformer
        x = self.transformer(x)
        return x[:, 0, :]  # Return the first token embedding as the query

class OrthogonalLinear(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(OrthogonalLinear, self).__init__()
        # self.fc = nn.Linear(input_dim, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)

    def orthogonal_regularization(self, weight_decay=1e-1):
        """Apply orthogonal regularization to the weight matrix."""
        W = self.fc.weight
        loss = torch.norm(torch.mm(W, W.T) - torch.eye(W.size(0), device=W.device))
        return weight_decay * loss
'''
loss += query_encoder.orthogonal_regularization()
query_encoder = OrthogonalLinear(input_dim=256, embed_dim=256).cuda()

'''

# class CAVAlignmentModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),  # 替换BatchNorm为LayerNorm
#             nn.GELU(),                 # 更平滑的激活函数
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
    
#     def forward(self, x):
#         return F.normalize(self.projection(x), dim=-1)
    
class CAVAlignmentModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 多层投影网络
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *self._make_layers(hidden_dim, num_layers, dropout),  # 添加多个残差块
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 多头注意力机制（用于跨模态交互）
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, dropout=dropout)
        
        # 最终的归一化层
        self.final_norm = nn.LayerNorm(output_dim)
        
        # Dropout 正则化
        self.dropout = nn.Dropout(dropout)
    
    def _make_layers(self, hidden_dim, num_layers, dropout):
        layers = []
        for _ in range(num_layers):
            # 残差块
            layers.append(ResidualBlock(hidden_dim, dropout))
        return layers
    def forward(self, x):
        # 输入x形状: [batch_size, input_dim]
        # 自注意力要求序列维度，增加一个虚拟序列长度（此处为1）
        x = x.unsqueeze(1)  # 变为 [batch_size, seq_len=1, input_dim]
        
        # 自注意力计算（query=key=value=x）
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output  # 残差连接
        
        # 投影到低维空间
        x = self.proj(x.squeeze(1))  # 移除序列维度
        return F.normalize(x, dim=-1)
    # def forward(self, x):
    #     # 投影到低维空间
    #     x = self.projection(x)
        
    #     # 多头注意力机制（假设有两个模态的特征，x1 和 x2）
    #     # 这里假设 x 是拼接后的特征，可以拆分为 x1 和 x2
    #     x1, x2 = x.chunk(2, dim=1)  # 将特征拆分为两部分
    #     x1 = x1.unsqueeze(0)  # 增加序列维度
    #     x2 = x2.unsqueeze(0)
        
    #     # 跨模态注意力
    #     attn_output, _ = self.cross_attention(x1, x2, x2)
    #     x = x + self.dropout(attn_output.squeeze(0))  # 残差连接
        
    #     # 最终归一化
    #     x = self.final_norm(x)
        
    #     # 归一化输出
    #     return F.normalize(x, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)  # 残差连接

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=256):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.net(x) + x
    
# class ResidualBlock(nn.Module):
#     def __init__(self, dim):
#         super(ResidualBlock, self).__init__()
#         self.fc = nn.Linear(dim, dim)
#         self.bn = nn.BatchNorm1d(dim)
#         # self.act = nn.GELU()
#         self.act = nn.ReLU()
    
#     def forward(self, x):
#         identity = x
#         x = self.fc(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x + identity  # 残差连接

# class ResidualQueryModel(nn.Module):
#     def __init__(self, input_dim, embed_dim, hidden_dim=2048, num_blocks=10):
#         super(ResidualQueryModel, self).__init__()
#         self.initial_fc = nn.Linear(input_dim, hidden_dim)
#         self.blocks = nn.Sequential(
#             *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
#         )
#         self.final_fc = nn.Linear(hidden_dim, embed_dim)
    
#     def forward(self, x):
#         x = self.initial_fc(x)
#         x = self.blocks(x)
#         x = self.final_fc(x)
#         return x
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # 输入x形状: (batch_size, seq_len, embed_dim)
        # 若为单向量，需添加序列维度
        x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.norm(attn_output + x)
        return attn_output.squeeze(1)

class AttnQueryModel(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(AttnQueryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.attn = SelfAttention(embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.attn(x)
        x = self.fc2(x)
        return x
class MoCoCAV(nn.Module):
    def __init__(self, input_dim, embed_dim, cavs, queue_size=4096, momentum=0.999, temperature=0.07, device = "cuda"):
        super().__init__()
        self.device = device
        # self.query_encoder = AttnQueryModel(input_dim, embed_dim).to(self.device)
        # self.key_encoder = AttnQueryModel(input_dim, embed_dim).to(self.device)
        self.query_encoder = ResidualQueryModel(input_dim, embed_dim).to(self.device)
        self.key_encoder = ResidualQueryModel(input_dim, embed_dim).to(self.device)
        # self.query_encoder = OrthogonalLinear(input_dim, embed_dim).to(self.device)
        # self.key_encoder = OrthogonalLinear(input_dim, embed_dim).to(self.device)
        # self.query_encoder = OrthogonalTransformer(input_dim=input_dim, embed_dim=embed_dim).to(self.device)
        # self.key_encoder = OrthogonalTransformer(input_dim=input_dim, embed_dim=embed_dim).to(self.device)
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
        self.key_encoder.eval()
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
        self.key_encoder.train()

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
        # loss = F.cross_entropy(logits_with_t, labels)
    

        # 原InfoNCE损失
        ce_loss = F.cross_entropy(logits_with_t, labels)

        # 新增方向对齐损失：最大化正样本对的余弦相似度
        direction_loss = 1 - positive_sim.mean()

        # 总损失 = InfoNCE + λ * 方向损失
        loss = ce_loss + 0.1 * direction_loss


        return loss, k



class IntegrateCAV(nn.Module):
    def __init__(self, cavs, device, autoencoders=None, dim_align_method="zero_padding", num_random_exp=3, save_dir="./analysis"):
        super().__init__()
        self.cavs = cavs
        self.aligned_cavs = []
        self.fused_cavs = []
        self.device = device
        self.__isAligned = False
        self.autoencoders = autoencoders
        self.dim_align_method = dim_align_method
        self.num_random_exp = num_random_exp
        self.save_dir = save_dir
        self.num_concepts = len(cavs[0])//num_random_exp

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
    

    def _align_dimension_by_ae(self, type="align"):
        """
        Args:
            autoencoder: trained CAVAutoencoder model
            cavs: cavs for each layer [n_samples, input_dim]
        Returns:
            aligned_cavs: 
        """

        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method)
        if not self.autoencoders.overwrite and os.path.exists(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy")):
            print("Input CAVs already exist. Loading from saved files.")
            input_cavs = np.load(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
            if type == "fuse":
                return input_cavs
            tmp=[]
            for layer_cavs in input_cavs:
                layer_cavs_tmp = []
                for cav in layer_cavs:
                    cav = torch.tensor(cav).to(self.device)
                    layer_cavs_tmp.append(cav)
                tmp.append(layer_cavs_tmp)
            print("Aligned CAVs loaded!")
            return tmp
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

        save_cavs = []
        for layer_cavs in input_cavs:
            save_layer_cavs = []
            for cav in layer_cavs:
                save_layer_cavs.append(cav.cpu().numpy())
            save_cavs.append(save_layer_cavs)
        save_cavs = np.array(save_cavs)
        # import pdb; pdb.set_trace()
        np.save(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"),  save_cavs)
        print("Aligned CAVs saved at", os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"))
        if type == "fuse":
            return save_cavs
        return input_cavs

    def align_with_moco(self, queue_size, momentum, temperature, embed_dim=2048, epochs = 2000, overwrite=False):
        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method)
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

   

    def prepare_pairs(self, input_cavs):
        """生成正负样本对：同一概念跨层为正，不同概念为负"""
        positive_pairs = []
        negative_pairs = []
        
        # 遍历所有概念
        for concept_id in range(self.num_concepts):
            # 收集该概念在所有层的CAV
            concept_cavs = [layer_cavs[concept_id*self.num_random_exp + i] for layer_cavs in input_cavs for i in range(self.num_random_exp)]
            
            # 生成同一概念的正样本对（跨层）
            for i in range(len(concept_cavs)):
                for j in range(i+1, len(concept_cavs)):
                    positive_pairs.append((concept_cavs[i], concept_cavs[j]))
            
            # 生成负样本（随机选择不同概念的CAV）
            for _ in range(len(concept_cavs)):
                neg_concept = np.random.choice([c for c in range(self.num_concepts) if c != concept_id])
                neg_layer = np.random.randint(0, len(input_cavs))
                neg_concept_random = np.random.randint(0, self.num_random_exp)
                negative_pairs.append((
                    concept_cavs[np.random.randint(0, len(concept_cavs))],
                    input_cavs[neg_layer][neg_concept*self.num_random_exp + neg_concept_random]
                ))
        
        return positive_pairs, negative_pairs
    
    def contrastive_loss(self, z1, z2, temperature=0.1):
        """改进的对比损失，显式处理正负样本"""
        batch_size = z1.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.T) / temperature  # [B, B]
        
        # 正样本对位于对角线
        positive_sim = torch.diag(sim_matrix)
        
        # 负样本对为所有非对角线元素
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
        negative_sim = sim_matrix[mask].view(batch_size, -1)
        
        # 构造logits
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        
        # 计算交叉熵损失
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)
    
    def train(self,embed_dim=2048, epochs=1000, batch_size=256, lr=1e-3, overwrite=False):#  embed_dim=2048, epochs = 2000):
        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method)
        if not overwrite :
            if os.path.exists(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy")):
                print("Aligned CAVs already exist. Loading from saved files.")
                self.aligned_cavs = np.load(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
                print("Aligned CAVs loaded!")
                self.__isAligned = True
                return self.aligned_cavs
        model = CAVAlignmentModel(input_dim=embed_dim, hidden_dim=4096, output_dim=embed_dim, num_layers=4, num_heads=8, dropout=0.1).to(self.device) # , input_dim=2048, hidden_dim=1024, output_dim=256
        
        # 使用AdamW优化器 + 学习率预热
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda epoch: min(epoch / 100, 1.0)  # 前100个epoch线性预热
        # )
        input_cavs = self._align_dimension_by_ae()
        input_cavs = normalize_cav(input_cavs)
        # import pdb; pdb.set_trace()
        # 准备所有样本对
        pos_pairs, neg_pairs = self.prepare_pairs(input_cavs=input_cavs)
        all_pairs = pos_pairs + neg_pairs
        labels = torch.cat([
            torch.ones(len(pos_pairs)),   # 正样本标签1
            torch.zeros(len(neg_pairs))   # 负样本标签0
        ]).to(self.device)
        # contrastive_loss = ContrastiveLoss()
        for epoch in range(epochs):
            # 随机采样批次
            indices = torch.randperm(len(all_pairs))[:batch_size]
            batch_pairs = [all_pairs[i] for i in indices]
            batch_labels = labels[indices]
            
            # 转换为tensor
            batch_tensor = torch.stack([
                # torch.cat([torch.tensor(p[0]), torch.tensor(p[1])]) 
                torch.cat([p[0], p[1]]) 
                for p in batch_pairs
            ]).float().to(self.device)
            
            # 分割为两个视图
            z1 = model(batch_tensor[:, :embed_dim])    # 视图1
            z2 = model(batch_tensor[:, embed_dim:])    # 视图2
            
            # 计算损失
            loss = self.contrastive_loss(z1, z2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            # scheduler.step()
            # optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        aligned_cavs=[]
        model.eval()
        with torch.no_grad():
            for cavs_layer in input_cavs:
                aligned_cavs_layer = []
                for cav in cavs_layer:
                    cav = torch.tensor(cav).to(self.device)
                    # import pdb; pdb.set_trace()
                    cav = cav.unsqueeze(0)
                    cav = model(cav).squeeze(0)
                    aligned_cavs_layer.append(cav.cpu().numpy())
                aligned_cavs.append(aligned_cavs_layer)
        print("CAVs aligned!")
        self.aligned_cavs = aligned_cavs
        del model # release memory
        np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
        print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"))
        self.__isAligned = True
        return self.aligned_cavs
        # # 保存对齐后的CAV
        # aligned_cavs = []
        # with torch.no_grad():
        #     for layer in self.layers_cavs:
        #         layer_tensor = torch.tensor(layer).float().to(self.device)
        #         aligned_layer = model(layer_tensor).cpu().numpy()
        #         aligned_cavs.append(aligned_layer)
        
        # np.save(os.path.join(self.save_dir, 'aligned_cavs.npy'), aligned_cavs)
        # torch.save(model.state_dict(), os.path.join(self.save_dir, 'alignment_model.pth'))
        # return aligned_cavs
    
    def fuse(self,fuse_method="mean", overwrite=False):
        """
        Fuse the aligned CAVs
        """
        if not self.__isAligned:
            raise ValueError("CAVs are not aligned. Please align the CAVs first.")

        save_dir = os.path.join(self.save_dir,"fuse_model", self.dim_align_method, fuse_method)
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
                fused_cav = np.sum([weight * embed for weight, embed in zip(weight, planes[i])], axis=0)
                fused_cavs.append(fused_cav)
            
            fused_cavs = np.array(fused_cavs)
        elif fuse_method == "cosine":
            input_cavs = self._align_dimension_by_ae(type="fuse")
            fused_cavs = []
            for concept_idx in range(len(input_cavs[0])):
                cos_sim = []
                for layer_idx in range(len(input_cavs)):
                    cos_sim.append(cosine_similarity(input_cavs[layer_idx][concept_idx],self.aligned_cavs[layer_idx][concept_idx])+1)# add 1 to avoid negative value
                weight = [cos/sum(cos_sim) for cos in cos_sim]
                fused_cav = np.sum([weight * embed for weight, embed in zip(weight, self.aligned_cavs[:, concept_idx,:])], axis=0)
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

class ContrastiveLoss(nn.Module):
    def __init__(self, init_temp=0.1, learnable=True):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp)) if learnable else init_temp
    
    def forward(self, z1, z2):
        sim_matrix = torch.mm(z1, z2.T) / self.temperature
        positive_sim = torch.diag(sim_matrix)
        negative_sim = sim_matrix[~torch.eye(z1.size(0), dtype=torch.bool, device=z1.device)].view(z1.size(0), -1)
        
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        labels = torch.zeros(z1.size(0), dtype=torch.long, device=z1.device)
        return F.cross_entropy(logits, labels)

