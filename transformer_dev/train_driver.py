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

import VisionTransformer, TrainViT
from vision_transformers.CvT_model import CvT


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
run = params["run"]
image_dir =  params["train_image_dir"]
isd_map_dir = params["isd_map_dir"]
save_dir = os.path.abspath(params["save_dir"])
start_epoch = params.get("start_epoch", 0)
loss = nn.CosineSimilarity()

# optim params
optimizer_config = params.get("optimizer", {})
optimizer_type = optimizer_config.pop("type", "Adam")  # Default to "Adam" if not specified

# schedule params
use_scheduler = params.get("use_scheduler", False)
if use_scheduler:
    scheduler_config = params.get("scheduler", {})
    scheduler_type = optimizer_config.pop("type")  # Default to "Adam" if not specified

# Model setup
# if else for model selection
if params["model"] == 'ccnViT':
    model = VisionTransformer(num_layers=12, img_size=size, embed_dim=768, patch_size=16, num_head=8, cnn_embedding=True)
    model_params = model.parameters()
if params["model"] == 'CvT':
    model = CvT()
    model_params = model.parameters()

# Initialize training class
vit_trainer = TrainViT(size, batch_size, epochs, loss, run, save_dir, start_epoch)

# Sets
vit_trainer.set_data_loaders(image_dir=image_dir, isd_map_dir=isd_map_dir, perform_checks=True) # add arguments to yaml
vit_trainer.set_model(model=model)
vit_trainer.set_optimizer(model_params=model_params, optimizer_type=optimizer_type, **optimizer_config)
if use_scheduler:
    vit_trainer.set_scheduler(scheduler_type=scheduler_type, **scheduler_config)

# Load checkpoint if required
if params["pretrained"]["load_model_state"]:
    vit_trainer.load_state(params["pretrained"]["checkpoint_path"])

# Save configuration and start training
vit_trainer.save_config(config=params)
logger.info(f"Starting Training")
results_dict  = vit_trainer.train_model()

logger.info(f"Final training loss: {results_dict['train_loss_history'][-1]}")
logger.info(f"Final validation loss: {results_dict['train_loss_history'][-1]}")