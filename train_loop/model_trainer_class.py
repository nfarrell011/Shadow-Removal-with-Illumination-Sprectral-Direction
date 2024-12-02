""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file contains a training class (TrainViT).
"""
# Packages
import os
import logging
import torch
import yaml
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

import sys

# Add the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Modules
from utils.dataset_generator_class import ImageDatasetGenerator

########################################## ViT Trainer #########################################################
class TrainViT:
    """
    Class to train ViT.
    """

    def __init__(self,
                 image_size=224,
                 batch_size=32,
                 epochs=100,
                 loss=nn.MSELoss(),
                 run="training_run",
                 save_dir='/results',
                 start_epoch=0):
        """
        Initializes TrainViT class with default or user-provided values.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.image_size = image_size
        self.batch_size = batch_size
        self.loss = loss
        self.run = run
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.start_epoch = start_epoch
        self.epochs = epochs
        self.avg_loss = 0

        self.train_ds = None
        self.val_ds = None
        self.train_dl = None
        self.val_dl = None

        self.train_loss = []
        self.val_loss = []
        self.best_val_loss = float('inf')

        self.val_paths = None
        self.train_paths = None

    def set_model(self, model: callable = None) -> None:
        """
        Sets the generator model.
        """
        self.model = model.to(self.device)
        self.logger.info("Model initialized.")

    def load_state(self, path_to_checkpoint: str) -> None:
        """
        Loads a previous model state.
        """
        try:
            checkpoint = torch.load(path_to_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.avg_loss = checkpoint['loss']
            self.logger.info("Model state loaded successfully.")
        except FileNotFoundError as e:
            self.logger.error(f"Error loading checkpoint: {e}")
        except KeyError as e:
            self.logger.error(f"Checkpoint is missing a key: {e}")

    def set_data_loaders(self, image_dir: str, isd_map_dir: str, perform_checks: bool = True, use_mean: bool = False) -> None:
        """
        Sets up the dataloaders.

        Args:
            image_dir (str): Directory of training images.
            isd_map_dir (str): Directory of isd maps corresponding to training images. 
            performs_check (bool): If True (default), will check that dataloader output.
            use_mean (bool): Sets the output for guidance image to mean isd if True or pixel-wise if False. Optional, default = False
        Returns:
            None
        """
        # Transforms -- We can add augmentations here
        transform_images = transforms.Compose([ 
                                transforms.ToTensor()       
                                #transforms.Normalize(mean = mean, std = std)
                                            ])

        transform_guidance = transforms.Compose([ 
                                transforms.ToTensor()
                                            ])
        
        # Create train split
        self.train_ds = ImageDatasetGenerator(image_dir, 
                                            isd_map_dir, 
                                            split = "train", 
                                            val_size = 0.2, 
                                            random_seed = 42, 
                                            transform_images = transform_images, 
                                            transform_guidance = transform_guidance,
                                            use_mean = use_mean)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Create val split
        self.val_ds = ImageDatasetGenerator(image_dir, 
                                            isd_map_dir, 
                                            split = "val", 
                                            val_size = 0.2, 
                                            random_seed = 42, 
                                            transform_images = transform_images, 
                                            transform_guidance = transform_guidance,
                                            use_mean = use_mean)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        if perform_checks:
            if not len(self.train_ds):
                self.logger.warning("The dataloader is empty. No batches to inspect.")
                return

            for idx, (images, isds) in enumerate(self.train_ds):
                self.logger.info(f"Inspecting Batch {idx + 1}")
                self.logger.info(f"Images shape: {images.shape}")
                self.logger.info(f"ISDs shape: {isds.shape}")
                if idx == 0:
                    break

    def set_optimizer(self, model_params, optimizer_type="Adam", **kwargs) -> None:
        """
        Method to set up the optimizer.

        Args:
            model_params (iterable): Parameters of the model to optimize.
            optimizer_type (str): Type of optimizer (e.g., 'Adam', 'SGD').
            **kwargs: Additional parameters for the optimizer.
        
        Raises:
            ValueError: If the specified optimizer type is not supported.
        """
        # Supported optimizer types
        optimizers = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }

        # Get the optimizer class
        optimizer_class = optimizers.get(optimizer_type)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Initialize the optimizer with provided parameters and kwargs
        self.optimizer = optimizer_class(model_params, **kwargs)
        self.logger.info(f"Optimizer {optimizer_type} initialized with parameters: {kwargs}")

    def set_scheduler(self, scheduler_type="StepLR", **kwargs) -> None:
        """
        Method to set up the learning rate scheduler.

        Args:
            scheduler_type (str): Type of learning rate scheduler. Default is 'StepLR'.
            **kwargs: Additional parameters for the scheduler.
        """
        if not self.optimizer:
            raise ValueError("Optimizer must be set before defining the scheduler.")

        if scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=kwargs.get("step_size", 10), 
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode=kwargs.get("mode", "min"), 
                factor=kwargs.get("factor", 0.1), 
                patience=kwargs.get("patience", 10)
            )
        elif scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=kwargs.get("T_max", 50), 
                eta_min=kwargs.get("eta_min", 0)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        logging.info(f"Scheduler set to {scheduler_type} with parameters: {kwargs}")

    def train_loop(self, epoch: int) -> None:
        """
        Performs the training loop for a single epoch.
        """
        # Reset the total loss and num batches
        epoch_train_loss = 0
        num_batches = 0

        # Set model to train mode
        self.model.train()

        # Sets up the progress bar 
        pbar = tqdm(self.train_dl, desc=f"Training Epoch {epoch}/{self.epochs}")

        # Iterate over batches -- Each batch contains img and isd pair
        for i, (imgs, isds) in enumerate(pbar):

            # Load images and isds
            imgs = imgs.to(self.device)
            isds = isds.to(self.device)

            # Zero the gradients for every batch
            self.optimizer.zero_grad()

            # Make predicstions with model
            model_output = self.model(imgs)

            # Compute loss with the outputs and true isds
            loss = self.loss(model_output, isds)

            # Compute the gradients
            loss.backward()

            # Update the learning weights
            self.optimizer.step()

            # Gather results data
            epoch_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(G_loss=loss.item())

        # Track average loss
        avg_train_loss = epoch_train_loss / num_batches
        self.train_loss.append(avg_train_loss)
        self.logger.info(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")
        return avg_train_loss
    
    def val_loop(self, epoch: int) -> None:
        """
        Performs the validation loop for a single epoch.
        """
        with torch.no_grad():
            epoch_val_loss = 0
            num_batches = 0

            self.model.eval()
            pbar = tqdm(self.val_dl, desc=f"Validation Epoch {epoch}/{self.epochs}")
            for i, (imgs, isds) in enumerate(pbar):
                imgs = imgs.to(self.device)
                isds = isds.to(self.device)

                model_output = self.model(imgs)
                loss = self.loss(model_output, isds)

                epoch_val_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(Val_loss=loss.item())

            avg_val_loss = epoch_val_loss / num_batches
            self.val_loss.append(avg_val_loss)
            self.logger.info(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss}")
     
            
            return avg_val_loss

    def save_model_state(self, epoch: int) -> None:
        """
        Saves the current model state to a checkpoint.
        """
        state_save_dir = os.path.join(self.save_dir, "model_states")
        os.makedirs(state_save_dir, exist_ok=True)

        save_path = os.path.join(state_save_dir, f'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss[-1],
            'val_loss': self.val_loss[-1]
        }, save_path)
        self.logger.info(f"Model state saved at epoch {epoch}: {save_path}")

    def save_config(self, config: dict, file_name = "config.yaml") -> None:
        """
        Saves the configuration dictionary to a YAML file.
        """
        save_path = os.path.join(self.save_dir, file_name)
        try:
            with open(save_path, "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

    def plot_losses(self, epoch: int) -> None:
        """
        Plots and saves the loss vs epoch graph.
        """
        figs_save_dir = os.path.join(self.save_dir, "loss_figs")
        os.makedirs(figs_save_dir, exist_ok=True)

        plt.figure()
        plt.plot(range(self.start_epoch, epoch + 1), self.train_loss, label="Train Loss", color="b")
        plt.plot(range(self.start_epoch, epoch + 1), self.val_loss, label="Validation Loss", color="r")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(figs_save_dir, f"loss_epoch_{epoch}.png")
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Loss plot saved: {fig_path}")

    def train_model(self) -> None:
        """
        Trains the model over multiple epochs and returns the final model, optimizer states, and loss history.
        """
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_loop(epoch)
            avg_val_loss = self.val_loop(epoch)
            self.scheduler.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self.plot_losses(epoch)

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model_state(epoch)

        return {
            "train_loss_history": self.train_loss,
            "val_loss_history": self.val_loss
            }