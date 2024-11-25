""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file contains a training class (TrainViT).
"""

import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from dataloader_dev import ImageDatasetGenerator

class TrainViT:
    """
    Class to pretrain the generator network with a Vision Transformer backbone.
    """

    def __init__(self,
                 image_size=224,
                 batch_size=32,
                 epochs=100,
                 lr=0.0002,
                 beta1=0.5,
                 beta2=0.999,
                 weight_decay=0,
                 loss=nn.CosineSimilarity(),
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
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.loss = loss
        self.run = run
        self.save_dir = save_dir
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

        self.val_paths = None
        self.train_paths = None

    def set_train_and_val_paths(self, data_dir: str, num_images: int) -> None:
        """
        Placeholder for setting training and validation paths.
        """
        self.train_paths = os.path.join(data_dir, "train")
        self.val_paths = os.path.join(data_dir, "val")
        self.logger.info(f"Train and val paths set: {self.train_paths}, {self.val_paths}")

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

    def set_data_loaders(self, perform_checks: bool = True) -> None:
        """
        Sets up the dataloaders.
        """
        self.train_ds = ImageDatasetGenerator(self.batch_size, paths=self.train_paths, split="train")
        self.val_ds = ImageDatasetGenerator(self.batch_size, paths=self.val_paths, split="val")
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size)

        if perform_checks:
            data = next(iter(self.train_dl))
            imgs, isds = data['imgs'], data['isds']
            assert imgs.shape == isds.shape, "Mismatched shapes between inputs and targets!"
            self.logger.info(f"Train data: {imgs.shape}, Validation data: {len(self.val_dl)} batches")

    def set_optimizer(self, model_params, optimizer_type="Adam", **kwargs) -> None:
        """
        Method to set up the optimizer.

        Args:
            model_params (iterable): Parameters of the model to optimize.
            optimizer_type (str): Type of optimizer. Default is 'Adam'.
            **kwargs: Additional parameters for the optimizer.
        """
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                model_params, 
                lr=self.lr, 
                betas=(self.beta1, self.beta2), 
                weight_decay=self.weight_decay,
                **kwargs
            )
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                model_params, 
                lr=self.lr, 
                momentum=kwargs.get("momentum", 0.9), 
                weight_decay=self.weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        logging.info(f"Optimizer set to {optimizer_type} with lr={self.lr}, weight_decay={self.weight_decay}")

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
        epoch_train_loss = 0
        num_batches = 0

        self.model.train()
        pbar = tqdm(self.train_dl, desc=f"Training Epoch {epoch}/{self.epochs}")
        for i, data in enumerate(pbar):
            imgs, isds = data["imgs"].to(self.device), data["isds"].to(self.device)

            self.optimizer.zero_grad()
            model_output = self.model(imgs)
            loss = self.loss(model_output, isds)
            loss.backward()
            self.optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(G_loss=loss.item())

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
            for i, data in enumerate(pbar):
                imgs, isds = data["imgs"].to(self.device), data["isds"].to(self.device)

                model_output = self.model(imgs)
                loss = self.loss(model_output, isds)

                epoch_val_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(Val_loss=loss.item())

        avg_val_loss = epoch_val_loss / num_batches
        self.val_loss.append(avg_val_loss)
        self.logger.info(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss}")

    def save_model_state(self, epoch: int) -> None:
        """
        Saves the current model state to a checkpoint.
        """
        state_save_dir = os.path.join(self.save_dir, "model_states")
        os.makedirs(state_save_dir, exist_ok=True)

        save_path = os.path.join(state_save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_loss[-1],
        }, save_path)
        self.logger.info(f"Model state saved at epoch {epoch}: {save_path}")

    def save_config(self, config: dict, file_name="config.yaml") -> None:
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
            val_loss = self.val_loop(epoch)

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self.plot_losses(epoch)
                self.save_model_state(epoch)

        return {
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "train_loss_history": self.train_loss,
        "val_loss_history": self.val_loss
        }