import torch
from models.model import ModifiedCellSwin
from utils.losses import CombinedLoss
from train.train import train_model
from data.data_loader import get_data_loaders

def main():
    # Hyperparameters
    num_epochs = 100
    learning_rate = 1