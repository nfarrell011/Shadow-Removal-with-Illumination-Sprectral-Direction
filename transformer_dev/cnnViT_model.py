"""
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer 

This module contains the a vanilla VisionTransformer class - which can be implemented with
either a patch embedding or a ResNet50 embedding input. 
"""


# Packages
import torch
import math
import logging

from torch import nn, Tensor
from einops import rearrange, repeat
from torchvision.models import resnet50, ResNet50_Weights

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO or WARNING in production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console logging
    ]
)

##############################################################################################################
## Utilities
##############################################################################################################
class RearrangeLayer(nn.Module):
    def __init__(self, pattern, **dims):
        """
        Custom layer for einops.rearrange.
        
        Args:
            pattern (str): Rearrangement pattern.
        """
        super().__init__()
        self.pattern = pattern
        self.dims = dims
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x):
        self.logger.debug(f"Rearranging tensor with pattern {self.pattern} and dimensions {self.dims}")
        return rearrange(x, self.pattern, **self.dims)
    
##############################################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        """
        Sinusoidal Positional Encoding Module.

        Args:
            emb_size (int): The size of the embedding dimension.
            max_len (int): The maximum length of the sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing PositionalEncoding with emb_size={emb_size}, max_len={max_len}")

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, emb_size).

        Returns:
            Tensor: Positional encoded tensor of the same shape as input.
        """
        seq_len = x.size(1)
        self.logger.debug(f"Adding positional encoding to tensor of shape {x.shape}")
        return x + self.positional_encoding[:, :seq_len, :]
    
##############################################################################################################

class PatchEmbedding(nn.Module):
    """
    Uses a COnv2 layer with stride=patch size to create the patch embeddings
    
    Args:
        in_channels (int): Number of channels in the input image.
        patch_size (int): The size of the patch to extract.
        embed_size (int): The size of the embedding dimension.
        img_size (int): SIze of the input iamge.
    """
    def __init__(self, in_channels: int, patch_size: int, embed_size: int, img_size: int):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing PatchEmbedding with in_channels={in_channels}, patch_size={patch_size}, embed_size={embed_size}, img_size={img_size}")
        
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=embed_size, 
                      kernel_size=patch_size, 
                      stride=patch_size),
            RearrangeLayer('b e h w -> b (h w) e'),
        )
        self.pos_embed = PositionalEncoding(embed_size, (img_size // patch_size)**2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates patch embedding of input iamge

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channel, image_size, image_size).

        Returns:
            Tensor: Conv2d with positional embedding output of shape ().
        """
        self.logger.debug(f"Input tensor shape: {x.shape}")
        x = self.embed(x)
        self.logger.debug(f"Conv2d output shape: {x.shape}")
        x = self.pos_embed(x)
        self.logger.debug(f"Output with positional encoding shape: {x.shape}")
        return x

##############################################################################################################

class CNNFeatureEmbedder(nn.Module):
    """
    CNN-based embedding module for hybrid architectures.

    Args:
        cnn_backbone (nn.Module): CNN model for feature extraction. Default: ResNet-50.
        embed_dim (int): Target embedding dimension for the Transformer.
        position_embedding (bool): Whether to add positional embeddings to the sequence.
    """
    def __init__(self, cnn_backbone=None, embed_dim=768, position_embedding=True):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing CNNFeatureEmbedder with embed_dim={embed_dim}, position_embedding={position_embedding}")
        
        if cnn_backbone is None:
            cnn_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(cnn_backbone.children())[:-2])
        else:
            self.feature_extractor = cnn_backbone

        self.patch_embed = nn.Linear(cnn_backbone.fc.in_features, embed_dim)
        self.position_embedding = position_embedding
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the CNN-based embedding module.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Sequence of patch embeddings of shape (batch_size, num_patches, embed_dim).
        """
        self.logger.debug(f"Input image tensor shape: {x.shape}")
        features = self.feature_extractor(x)
        self.logger.debug(f"Features shape: {features.shape}")
        
        batch_size, C, H, W = features.shape
        self.logger.debug(f"Feature map dims: {H}x{W}")
        num_patches = H * W
        self.logger.debug(f"Number of patches: {num_patches}")
        
        features = features.permute(0, 2, 3, 1).reshape(batch_size, num_patches, C)
        self.logger.debug(f"Flattened features shape: {features.shape}")

        embeddings = self.patch_embed(features)
        self.logger.debug(f"Embedding shape after projection: {embeddings.shape}")
        
        if self.position_embedding:
            pos_embed = PositionalEncoding(self.embed_dim, max_len=num_patches)
            embeddings = pos_embed(embeddings)
        return embeddings
    

##############################################################################################################

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels=768):
        """
        Upsample ResNet feature map back to input size using transpose convolutions.

        Output Size = (Input Size - 1) x (Stride - 2) x (Padding + Kernel Size)

        Args:
            in_channels (int): Number of input channels from ResNet feature map.
            target_size (tuple): Target spatial size (height, width) to upsample to.
        """
        super().__init__()

        # Set up logger for the class
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SimpleDecoder.")

        # Transpose convolution layers
        self.upsample = nn.Sequential(
            RearrangeLayer('b (h w) c -> b c h w', h=7, w=7),  # Reshape (batch, size, embed_dim) -> (batch, channel, height, width)
            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),          # 14 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),          # 28 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),           # 56 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),             # 112 -> 224
        )
        self.logger.info("SimpleDecoder initialized.")

    def forward(self, x):
        """
        Forward pass to upsample feature map to input size.
        
        Args:
            x (Tensor): Input feature map from ResNet of shape (batch_size, in_channels, height, width).
        
        Returns:
            Tensor: Upsampled feature map of shape (batch_size, 3, target_size[0], target_size[1]).
        """
        self.logger.debug(f"Forward pass started with input shape: {x.shape}")
        output = self.upsample(x)
        self.logger.debug(f"Forward pass completed. Output shape: {output.shape}")
        return output

##############################################################################################################

class TransformerEncoderBlock(nn.Module):
    """
    Vanilla ViT Block - based on https://arxiv.org/pdf/2010.11929 
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing TransformerEncoderBlock with embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}")

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Takes tensor input an runs through the following:
            * layer norm
            * Multihead Attention
            * Residula Connection
            * layer norm
            * multilayer perceptron
            * residual connection

        Args:
            x (Tensor): Input [batch, num_patches, embed_dim]

        Returns:
            x (Tensor): Output [batch, num_patches, embed_dim]
        """

        self.logger.debug(f"TransformerBlock")
        self.logger.debug(f"Input shape: {x.shape}")
        residual1 = x # first residual
        x = self.layer_norm(x) # first layer norm
        self.logger.debug(f'mlp input shape: {x.shape}')
        x, _ = self.mha(x, x, x) # MultiHeaded Attention
        self.logger.debug(f'self attention output shape: {x.shape}')
        x = self.dropout(x) + residual1 # First dropout layer w/ skip connection

        residual2 = x
        x = self.layer_norm(x) # Second Layernorm
        self.logger.debug(f'mlp input shape: {x.shape}')
        x = self.feed_forward(x) # MLP with GeLU
        self.logger.debug(f'mlp input shape: {x.shape}')
        x = self.dropout(x) + residual2 # Second Dropout w/ skip connection  
        self.logger.debug(f"Output shape: {x.shape}")

        return x

##############################################################################################################

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.

    This model implements a Vision Transformer with options for patch embedding 
    using either a convolutional backbone (ResNet50) or a simpler patch extraction mechanism. 
    The model also supports transformer-based encoding and an output reconstruction layer.

    Args:
        num_layers (int): Number of Transformer encoder layers.
        img_size (int): Size of the input image (assumed to be square: img_size x img_size).
        embed_dim (int): Dimensionality of the embedding vectors.
        patch_size (int): Size of each patch for embedding.
        num_head (int): Number of attention heads in the Multihead Attention module.
        cnn_embedding (bool, optional): Whether to use a CNN-based feature extractor as the embedding layer. 
                                        If False, a simple patch-based embedding is used. Default is True.
    """
    def __init__(self, num_layers, img_size, embed_dim, patch_size, num_head, cnn_embedding=True):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing VisionTransformer with num_layers={num_layers}, img_size={img_size}, embed_dim={embed_dim}, patch_size={patch_size}, num_head={num_head}, cnn_embedding={cnn_embedding}")

        if cnn_embedding:
            self.patch_emb = CNNFeatureEmbedder(embed_dim=embed_dim)
            self.output_layer = SimpleDecoder(in_channels=embed_dim)
        else:
            self.patch_emb = PatchEmbedding(in_channels=3, patch_size=patch_size, img_size=img_size, embed_size=embed_dim)
            self.output_layer = nn.Sequential(RearrangeLayer('b (h w) (patch_c ph pw) -> b patch_c (h ph) (w pw)',
                                                             h=14, w=14, patch_c=3, ph=16, pw=16))

        self.trans_encoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim, num_head) for _ in range(num_layers)])

    def forward(self, x):
        """
        Forward pass for the Vision Transformer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Reconstructed output image tensor of shape (batch_size, 3, img_size, img_size).
        """
        self.logger.debug(f"Input shape: {x.shape}")
        patch_embeddings = self.patch_emb(x)
        self.logger.debug(f"Patch embeddings shape: {patch_embeddings.shape}")
        x = self.trans_encoder(patch_embeddings)
        self.logger.debug(f"Transformer output shape: {x.shape}")
        output_img = self.output_layer(x)
        return output_img
    
def main():

    #Params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = 2
    embed_dim = 768
    num_head = 12
    patch_size=16
    img_size=224
    cnn_embedding = True
    model = VisionTransformer( num_layers=num_layers,
                                img_size=img_size,
                                embed_dim=embed_dim,
                                patch_size=patch_size,
                                num_head=num_head,
                                cnn_embedding=cnn_embedding
                                ).to(device)
    # Test input
    batch_size = 4
    test_input = torch.rand(batch_size, 3, img_size, img_size).to(device)

    # Forward pass
    try:
        output = model(test_input)
        print("\nModel Output shape:", output.shape)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")

if __name__ == "__main__":
    main()