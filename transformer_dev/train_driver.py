""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file can be used to train ViT.
"""

import os
import yaml
import glob
import torch
from torch import nn
import logging

from transformer_dev import VisionTransformer, TrainViT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting training...")

def load_optimizer_states(optimizer, config):
    checkpoint_path = config['pretrained']['checkpoint_path']
    if config['pretrained']['load_optim_states'] and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning("Optimizer state dictionaries not found in checkpoint.")
    return optimizer

# Load YAML configuration
try:
    with open("config.yaml", "r") as file:
        params = yaml.safe_load(file)
except FileNotFoundError:
    raise FileNotFoundError("The config.yaml file was not found. Please provide a valid path.")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing YAML file: {e}")

# Validate required keys
required_keys = ["data_dir", "num_images", "batch_size", "epochs", "lr", "beta1", "beta2", "run", "save_dir"]
for key in required_keys:
    if key not in params:
        raise ValueError(f"Missing required configuration key: {key}")

# Extract parameters
data_dir = params["data_dir"]
paths = glob.glob(os.path.join(data_dir, "*.tif"))
num_images = params["num_images"]
size = params["size"]
batch_size = params["batch_size"]
epochs = params["epochs"]
lr = params["lr"]
beta1 = params["beta1"]
beta2 = params["beta2"]
run = params["run"]
save_dir = os.path.abspath(params["save_dir"])
start_epoch = params.get("start_epoch", 0)
l1_loss = nn.CosineSimilarity()

# Model setup
model = VisionTransformer(num_layers=12, img_size=size, embed_dim=768, patch_size=16, num_head=8, cnn_embedding=True)
model_params = model.parameters()

# Initialize training class
vit_trainer = TrainViT(size, batch_size, epochs, lr, beta1, beta2, l1_loss, run, start_epoch)
vit_trainer.set_train_and_val_paths(paths, num_images)
vit_trainer.set_data_loaders()
vit_trainer.set_model(model=model)
vit_trainer.set_optimizer(model_params=model_params)

# Load checkpoint if required
if params["pretrained"]["load_model_state"]:
    vit_trainer.load_state(params["pretrained"]["checkpoint_path"])

# Save configuration and start training
vit_trainer.save_config(config=params)
logger.info(f"Starting Training")
results_dict  = vit_trainer.train_model()

logger.info(f"Final training loss: {results_dict['train_loss_history'][-1]}")
logger.info(f"Final validation loss: {results_dict['train_loss_history'][-1]}")