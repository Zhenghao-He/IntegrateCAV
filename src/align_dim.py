import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from keys in the state_dict
    to avoid issues when loading a model with nn.DataParallel.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict




class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, dim=1):
        super(CosineSimilarityLoss, self).__init__()
        self.dim = dim  

    def forward(self, output, target):
        cosine_sim = F.cosine_similarity(output, target, dim=self.dim)
        loss = 1 - cosine_sim.mean()  
        return loss

class SingleLayerAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dims=[4096, 4096], dropout=0.1):
        """
        Autoencoder for a single layer's CAVs with MLP encoder and decoder.
        Args:
            input_dim: dimension of the input CAVs for this layer
            embed_dim: dimension of the encoded embedding
            hidden_dims: list of hidden dimensions for encoder/decoder
            dropout: dropout rate for regularization
        """
        super(SingleLayerAutoencoder, self).__init__()

        # Build encoder and decoder
        self.encoder = self.build_encoder(input_dim, embed_dim, hidden_dims, dropout)
        self.decoder = self.build_decoder(embed_dim, input_dim, hidden_dims, dropout)

    def build_encoder(self, input_dim, embed_dim, hidden_dims, dropout):
        layers = []
        dims = [input_dim] + hidden_dims + [embed_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def build_decoder(self, embed_dim, output_dim, hidden_dims, dropout):
        layers = []
        dims = [embed_dim] + hidden_dims[::-1] + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, cav):
        """
        Args:
            cav: [batch_size, input_dim]
        Returns:
            reconstructed: [batch_size, input_dim]
            embedded: [batch_size, embed_dim]
        """
        embedded = self.encoder(cav)
        reconstructed = self.decoder(embedded)
        return reconstructed, embedded

    def encode(self, cav):
        return self.encoder(cav)

    def decode(self, embedded):
        return self.decoder(embedded)

class CAVAutoencoder:
    def __init__(self, input_dims, embed_dim, device, save_dir):
        """
        Args:
            input_dims: list of input dimensions for each layer
            embed_dim: shared embedding dimension for all layers
            device: computation device (e.g., "cuda" or "cpu")
        """
        self.device = device
        self.embed_dim = embed_dim
        self.input_dims = input_dims
        self.autoencoders = [None] * len(input_dims)  # Placeholder for Autoencoders
        self.__isTrained = [False] * len(input_dims)  # Track training status
        self.save_dir = os.path.join(save_dir, "autoencoders")

    def get_autoencoder(self, layer_idx):
        """
        Get or initialize the Autoencoder for a specific layer.
        """
        if self.autoencoders[layer_idx] is None:
            autoencoder = SingleLayerAutoencoder(
                input_dim=self.input_dims[layer_idx],
                embed_dim=self.embed_dim
            ).to(self.device)
  
        if torch.cuda.device_count() > 1:
            autoencoder = nn.DataParallel(autoencoder)
            
        return autoencoder
        

    def train_autoencoders(self, cavs, overwrite, epochs=50, lr=1e-3):
        """
        Train each layer's Autoencoder.
        Args:
            cavs: list of CAVs for each layer (each element: [n_samples, input_dim])
            epochs: number of training epochs
            lr: learning rate
        """
        if not overwrite and all(os.path.exists(os.path.join(self.save_dir, f"autoencoder_layer_{i}.pth")) for i in range(len(self.autoencoders))):
            print("Autoencoders already trained.")
            for layer_idx, autoencoder in enumerate(self.autoencoders):
                self.__isTrained[layer_idx] = True
            return

        os.makedirs(self.save_dir, exist_ok=True)
        # loss_fn = nn.MSELoss()
        loss_fn = CosineSimilarityLoss()

        for layer_idx, (autoencoder, layer_cavs) in enumerate(zip(self.autoencoders, cavs)):
            
            print(f"Training Autoencoder for Layer {layer_idx + 1}")
            autoencoder = self.get_autoencoder(layer_idx)
            # Initialize optimizer for the current Autoencoder
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
            
            # Convert CAVs to tensor
            layer_cavs = np.stack(layer_cavs)
            layer_cavs = torch.tensor(layer_cavs, dtype=torch.float32).cuda()
            
            # Train the Autoencoder for this layer
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, _ = autoencoder(layer_cavs)
                
                # Compute loss
                loss = loss_fn(reconstructed, layer_cavs)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                print(f"Layer {layer_idx + 1}, Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            
            print(f"Training complete for Layer {layer_idx + 1}!")
            autoencoder.eval()
            torch.save(autoencoder.state_dict(), os.path.join(self.save_dir, f"autoencoder_layer_{layer_idx}.pth")) 
            print(f"Autoencoders saved to {self.save_dir}.")
            del autoencoder # Clear memory
            torch.cuda.empty_cache()
            self.__isTrained[layer_idx] = True

    def load_autoencoder(self, layer_idx):
        """
        Load a trained Autoencoder from a file.
        """
        autoencoder = SingleLayerAutoencoder(
                input_dim=self.input_dims[layer_idx],
                embed_dim=self.embed_dim
            ).to(self.device)
        # Load the checkpoint and remove 'module.' prefix
        checkpoint = torch.load(os.path.join(self.save_dir, f"autoencoder_layer_{layer_idx}.pth"))
        # import pdb; pdb.set_trace()
        checkpoint = remove_module_prefix(checkpoint)
        # autoencoder = nn.DataParallel(autoencoder)
        autoencoder.load_state_dict(checkpoint)
        autoencoder.eval()
        return autoencoder


        


