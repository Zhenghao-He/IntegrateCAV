import torch
import torch.nn as nn
import torch.nn.functional as F

class CAVAutoencoder(nn.Module):
    def __init__(self, input_dims, embed_dim):
        """
        aligns CAVs using an autoencoder.
        Args:
            input_dims: input dimensions for each layer
            embed_dim: 
        """
        super(CAVAutoencoder, self).__init__()
        
        # encoder for each layer
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, embed_dim) for input_dim in input_dims
        ])
        
        # decoder shared by all layers
        self.decoder = nn.Linear(embed_dim, max(input_dims))  # decode to the max dim

    def forward(self, cav, layer_idx):
        """
        
        Args:
            cav:  [batch_size, input_dim]
            layer_idx: choose the encoder for the layer
        Returns:
            reconstructed:
            embedded: 
        """
        embedded = F.relu(self.encoders[layer_idx](cav))  
        reconstructed = self.decoder(embedded)  
        return reconstructed, embedded

def train_autoencoder(autoencoder, cavs, epochs=50, lr=1e-3):
    """
    训练 Autoencoder
    Args:
        autoencoder: CAVAutoencoder 模型
        cavs: 每层的 CAV 数据列表，每个元素为 [n_samples, input_dim]
        epochs: 训练轮数
        lr: 学习率
    Returns:
        autoencoder: 训练好的模型
    """
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # 使用均方误差作为重建损失

    for epoch in range(epochs):
        total_loss = 0
        for layer_idx, layer_cavs in enumerate(cavs):
            layer_cavs = torch.tensor(layer_cavs, dtype=torch.float32).cuda()
            
            # 前向传播
            reconstructed, _ = autoencoder(layer_cavs, layer_idx)
            
            # 计算损失
            loss = loss_fn(reconstructed, layer_cavs)
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return autoencoder
