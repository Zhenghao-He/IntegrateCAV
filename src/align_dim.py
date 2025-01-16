import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
class SingleLayerAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dims=[256, 128], dropout=0.1):
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


class CAVAutoencoder(nn.Module):
    def __init__(self, input_dims, embed_dim,device):
        """
        Manages an Autoencoder for each layer's CAVs.
        Args:
            input_dims: list of input dimensions for each layer
            embed_dim: shared embedding dimension for all layers
        """
        super(CAVAutoencoder, self).__init__()
        self.device = device
        self.__isTrained = False
        self.autoencoders = [
            SingleLayerAutoencoder(input_dim, embed_dim).to(self.device)
            for input_dim in input_dims
        ]

    def train_autoencoders(self, cavs, save_dir, overwrite, epochs=50, lr=1e-3):
        """
        Train each layer's Autoencoder.
        Args:
            cavs: list of CAVs for each layer (each element: [n_samples, input_dim])
            epochs: number of training epochs
            lr: learning rate
        """
        save_dir = os.path.join(save_dir, "autoencoders")
        if not overwrite and all(os.path.exists(os.path.join(save_dir, f"autoencoder_layer_{i}.pth")) for i in range(len(self.autoencoders))):
            print("Autoencoders already trained. Loading from saved files.")
            for layer_idx, autoencoder in enumerate(self.autoencoders):
                autoencoder.load_state_dict(torch.load(f"{save_dir}/autoencoder_layer_{layer_idx}.pth"))
                autoencoder.eval()
            self.__isTrained = True
            return

        os.makedirs(save_dir, exist_ok=True)
        optimizers = [
            torch.optim.Adam(autoencoder.parameters(), lr=lr)
            for autoencoder in self.autoencoders
        ]
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for layer_idx, (autoencoder, optimizer, layer_cavs) in enumerate(zip(self.autoencoders, optimizers, cavs)):
                # Convert CAVs to tensor
                layer_cavs = np.stack(layer_cavs)
                layer_cavs = torch.tensor(layer_cavs, dtype=torch.float32).cuda()

                # Forward pass
                reconstructed, _ = autoencoder(layer_cavs)

                # Compute loss
                loss = loss_fn(reconstructed, layer_cavs)
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}")

        print("Training complete!")
            # Save each trained Autoencoder

        for layer_idx, autoencoder in enumerate(self.autoencoders):
            autoencoder.eval()
            torch.save(autoencoder.state_dict(), os.path.join(save_dir, f"autoencoder_layer_{layer_idx}.pth"))
        print(f"Autoencoders saved to {save_dir}.")
        self.__isTrained = True

    def load_autoencoder(self, layer_idx):
        if not self.__isTrained:
            raise ValueError("Autoencoders have not been trained yet.")
        return self.autoencoders[layer_idx]
        


