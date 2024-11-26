""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file contains an implementation of the CvT model with added convolutional decoder
"""
import torch
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from einops import repeat, rearrange

import numpy as np
import cv2

from pathlib import Path 
import os
import sys

######################################################## CvT Embedding Class #############################################################
class CVTEmbedding(nn.Module):
  """ 
  Convolutional Embedding Class
  """
  def __init__(self, in_ch, embed_dim, patch_size, stride):
    """ 
    Generates an embedding with conv2 and down sampling
    """
    super().__init__()
    print(f"Entering embedding: Patch Size: {patch_size} -- Stride: {stride} -- Embedding Dims {embed_dim}")
    
	# Generate embedding with Conv2d and Norm
    self.embed = nn.Sequential(nn.Conv2d(in_ch, embed_dim, kernel_size = patch_size, stride = stride))
    self.norm = nn.LayerNorm(embed_dim)
	
  def forward(self, x):
    """ 
	Generates the embedding.
  	"""
    #print(f"Shape X IN: {x.shape}")
    x = self.embed(x)
    
	# Rearrange to batch_size, positional_embeddings, channels
    x = rearrange(x, 'b c h w -> b (h w) c') # i.e. x: B T(h w) C
    x = self.norm(x)
    #print(f"Shape X OUT: {x.shape}")
    return x

######################################################## MHA Class #############################################################
class MultiHeadAttention(nn.Module):
  """ 
  This class will replaces the linear projection of Q, K, V with conv2d
  """
  def __init__(self, in_dim, num_heads, kernel_size = 3, with_cls_token = False):
    super().__init__()
    """ 
    Initializer for MHA

    Parameters:
      * in_dim: (int)           - Embedding size
      * num_heads: (int)        - The number of attention heads
      * kernel_size: (int)      - The kernel size in the conv2d
      * with_cls_token: (bool)  - Used to perform classification
    """
    # Padding to prevent downsampling
    padding = (kernel_size - 1)//2

    # The number of attention heads
    self.num_heads = num_heads

    # Used for classification; we will not use this, but could useful later
    self.with_cls_token = with_cls_token

    # Projection operations; Conv2d, BatchNorm, and Rearrange
    self.conv = nn.Sequential(
        nn.Conv2d(in_dim, in_dim, kernel_size = kernel_size, padding = padding, stride = 1),
        nn.BatchNorm2d(in_dim),
        Rearrange('b c h w -> b (h w) c') # FROM (batch_size, height, width, channels) TO (batch_size, positional_embeddings, embedding_size)
    )

    # Set attention drop out rate
    self.att_drop = nn.Dropout(0.1)

  def forward_conv(self, x):
    """ 
    Performs convolution on x three times

    Parameters:
        * x: (Tensor) - batch_size, positional_embeddings, embedding_size
    
    Returns:
        * Q, K, V
    """
    # Used for classification
    B, hw, C = x.shape
    if self.with_cls_token:
      cls_token, x = torch.split(x, [1, hw-1], 1)
    
    # Height and width equal the number of tokens; num positional embeddings
    H = W = int(x.shape[1]**0.5)

    # Rearrange x
    # FROM (batch_size, positional_embeddings, embedding_size) TO (batch_size, height, width, channels)
    # Brings back to image dimensions
    x = rearrange(x, 'b (h w) c -> b c h w', h = H, w = W)
    
    # Convolution for Q, K, V
    q = self.conv(x)
    k = self.conv(x)
    v = self.conv(x)

    if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)


    # Return Q, K, V
    return q, k, v

  def forward(self, x):
    """ 
    Forward pass

    Parameters:
        * x: (Tensor) - (batch_size, positional_embeddings, embedding_size) or (b, t, c), where:
                b: batch size
                t: Number of positional embeddings or sequence length
                c: (d * H) - Total embedding size (split into d for each head and H heads)
                H: Number of heads
                d: c/H - Dims of each head's embedding
        Note: c % H = 0 MUST BE TRUE
    """
    # Performs the convolutions on x
    q, k, v = self.forward_conv(x)
    #print(f"Shape of q k v: {q.shape}")

    # Rearrange the output from the convolutions for attention computation
    # FROM (batch_size, positional_embeddings, (size of the head embedding, number of heads))
    # TO (batch_size, number of heads, positional_embeddings, head embedding)
    # b: batch_size
    # t: positional embeddings
    # d: head embedding size
    # H: number of heads
    q = rearrange(x, 'b t (d H) -> b H t d', H = self.num_heads)
    k = rearrange(x, 'b t (d H) -> b H t d', H = self.num_heads)
    v = rearrange(x, 'b t (d H) -> b H t d', H = self.num_heads)

    # Compute attention
    # Q: (b, H, t, d)
    # K^T: (b, H, d, T) -- Transposed along last 2 dimensions
    # Allows matrix multiplication
    # OUT: att_score: (b, H, t , t)
    # Each attention head computes scores for all the positional embeddings; closer vectors get higher scores
    att_score = q@k.transpose(2, 3) / self.num_heads**0.5 # This division stabilizes the gradient

    # Perform soft along the last dimension
    att_score = F.softmax(att_score, dim = -1)

    # Apply dropout
    att_score = self.att_drop(att_score)

    # Compute attention scores
    x = att_score@v
    x = rearrange(x, 'b H t d -> b t (H d)')

    return x

######################################################## MLP Class #############################################################
class MLP(nn.Module):
  """ 
  Creats an MLP
  """
  def __init__(self, dim):
    super().__init__()
    self.ff = nn.Sequential(
                            nn.Linear(dim, 4*dim),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(4 * dim, dim),
                            nn.Dropout(0.1)
                          )

  def forward(self, x):
     return self.ff(x)

######################################################## Attention Block Class #############################################################
class Block(nn.Module):
  """ 
  Attention Block Class

  This performs the entire attention pipe
  """
  def __init__(self, embed_dim, num_heads, with_cls_token):
    super().__init__()

    # MHA
    self.mhsa = MultiHeadAttention(embed_dim, num_heads, with_cls_token=with_cls_token)

    # MLP
    self.ff = MLP(embed_dim)

    # Norms and regs
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    """
    Forward pass 
    """
    # Performs MHA
    x = x + self.dropout(self.mhsa(self.norm1(x)))

    # Performs MLP
    x = x + self.dropout(self.ff(self.norm2(x)))
    return x

######################################################## Attention Block Class #############################################################
class VisionTransformer(nn.Module):
  """ 
  Vision Transformer class.

  Generates embedding with a convolution layer (down samples) and performs MHA.

  Params:
    * depth: (int)      -- Controls the number of MHA blocks
    * stride: (int)     -- Controls image down sampling for the embedding.
    * patch_size: (int) -- Kernel size for the embedding conv
    * in_ch: (int)      -- Number of channels in image before embedding.
    * embed_dim: (int)  -- Number of dimensions in the embedding.
    * num_heads: (int)  -- The nu
  """
  def __init__(self, depth:int, embed_dim:int, num_heads:int, patch_size:int, stride:int, in_ch:int = 3, cls_token = False):
    super().__init__()

    # Embedding stride -- This downsamples the image
    self.stride = stride

    # For later use, classification
    self.cls_token = cls_token
    
    # Generates the embedding for the transformer
    self.embedding = CVTEmbedding(in_ch, embed_dim, patch_size, stride)

    # Unpacks the list of Blocks controled by depth param. The is the number of attention blocks in the ViT layer.
    self.layers = nn.Sequential(*[Block(embed_dim, num_heads, cls_token) for _ in range(depth)])

    # Used for classification -- Not used here! 
    if self.cls_token:
       self.cls_token_embed = nn.Parameter(torch.randn(1, 1, 384))

  def forward(self, x, ch_out =False):
    """ 
    Forward pass
    """
    # Get the dims of x for rearranging 
    B, C, H, W = x.shape

    # Get embedding 
    x = self.embedding(x)

    # Used for classification
    if self.cls_token:
      cls_token = repeat(self.cls_token_embed, ' () s e -> b s e', b=B)
      x = torch.cat([cls_token, x], dim=1)

    # Execute attention layers.
    x = self.layers(x)

    # Reshape to image shape
    if not ch_out:
       x = rearrange(x, 'b (h w) c -> b c h w', h = (H - 1) // self.stride, w = (W - 1)// self.stride)
    return x

######################################################## Decoder Class #############################################################
class Decoder(nn.Module):
    """ 
    Decoder Layers

    This will bring the image back to original dims
    """
    def __init__(self):
        super().__init__()

        # Stage 3 to Stage 2
        self.decode_stage3 = nn.Sequential(
                nn.ConvTranspose2d(384, 192, kernel_size = 4, stride = 2, padding = 1, output_padding = 1),  # 13x13 -> 27x27
                nn.BatchNorm2d(192),
                nn.ReLU())

        # Stage 2 to Stage 1
        self.decode_stage2 = nn.Sequential(
                nn.ConvTranspose2d(192, 64, kernel_size = 4, stride = 2, padding = 1, output_padding = 1), # 27x27 -> 55x55
                nn.BatchNorm2d(64),
                nn.ReLU())

        # Stage 1 to Original Dims
        self.decode_stage1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size = 9, stride = 4, padding = 1, output_padding = 1)) # 55x55 -> 224x224

    def forward(self, x):
        """ 
        Forward pass
        """
        # Stage 3 -> Stage 2
        x = self.decode_stage3(x)
        #print(f"After Stage 3 -> 2: {x.shape}")  

        # Stage 2 -> Stage 1
        x = self.decode_stage2(x)
        #print(f"After Stage 2 -> 1: {x.shape}")  

        # Stage 1 -> Original Image
        x = self.decode_stage1(x)
        #print(f"After Stage 1 -> Original: {x.shape}")  

        return x

######################################################## CvT Full Model #############################################################
class CvT(nn.Module):
  """ 
  Convolutional Vision Transformer

  Executes a 3-stage vision transformer with a decoder to get back to image dims
  """
  def __init__(self, embed_dim, cls_token:bool = False, num_class:int = 10):
    super().__init__()
    self.cls_token = cls_token

    self.stage1 = VisionTransformer(depth = 1,
                                     embed_dim = 64,
                                     num_heads = 1,
                                     patch_size = 7,
                                     stride = 4,
                                     )
    
    self.stage2 = VisionTransformer(depth = 2,
                                     embed_dim = 192,
                                     num_heads = 3,
                                     patch_size = 3,
                                     stride = 2,
                                     in_ch = 64
                                     )
    
    self.stage3 = VisionTransformer(depth = 10,
                                     embed_dim = 384,
                                     num_heads = 6,
                                     patch_size = 3,
                                     stride = 2,
                                     in_ch = 192,
                                     cls_token = self.cls_token
                                     )
    
	# For classifiction - Not used! 
    self.ff = nn.Sequential(
        nn.Linear(6 * embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, num_class)
    )
  def forward(self, x):
    """ 
    Forward Pass
    """
    # Stage 1
    x = self.stage1(x)
    #print(f"Stage 1: Shape x: {x.shape}")

    # Stage 2
    x = self.stage2(x)
    #print(f"Stage 2: Shape x: {x.shape}")

    # Stage 3
    x = self.stage3(x, ch_out = False)
    #print(f"Stage 3: Shape x: {x.shape}")

    # For classifiction; no decoder
    if self.cls_token:
      x = x[:, 1, :]
      x = self.ff(x)
    
    # Perform decoding
    else:
      decoder = Decoder()
      x = decoder(x)
    return x 
  
#############################################################################################################################

if __name__ == "__main__":
   pass