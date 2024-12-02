""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file contains a class that will compute the angular distance between two vectors.

Designed to be used in at loss function in Torch network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineDistanceMetric(nn.Module):
    """
    Computes the cosine distance for a single image. Can support mean reduction when computing per pixel
    distance.
    """
    def __init__(self, reduce:bool = True):
        """
        Params:
            * reduce: (bool) - Indicates if reduction is needed. Set to True when computing per pixel dist. Default = True.
    
        Returns
            * cosine_distance: (float) - The angular distance between to true and predicted values.
        """
        super(CosineDistanceMetric, self).__init__()
        self.reduce = reduce

    def forward(self, actual_vectors:torch.Tensor, predicted_vectors:torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine distance for a single image.

        Args:
            actual_vectors (torch.Tensor): Ground truth tensor of shape (batch, channel, height, width).
            predicted_vectors (torch.Tensor): Predicted tensor of shape (batch, channel, height, width).

        Returns:
            torch.Tensor: A single scalar (mean or sum) or the full distance map (height x width).
        """
        if self.reduce:

            # Compute cosine similarity across the channel dimension, one value for each pixel
            cosine_similarity_map = F.cosine_similarity(actual_vectors, predicted_vectors, dim = 1)  # Shape: (batch, height, width)

            # Compute cosine distance
            cosine_distance_map = 1 - cosine_similarity_map  # Shape: (batch, height, width)

            # Apply reduction
            return cosine_distance_map.mean() # This will now be the mean of all cosine distances.

        else:
            # Compute the overall cosine sim using the mean value as the guidance
            cosine_similarity = F.cosine_similarity(actual_vectors, predicted_vectors, dim = 1)

            # Compute and return cosine distance
            cosine_dist = 1 - cosine_similarity
            return cosine_dist
        
if __name__ == "__main__":
    pass


