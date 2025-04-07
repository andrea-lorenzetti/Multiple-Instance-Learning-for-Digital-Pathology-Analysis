import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from config import *

from data.dataset import MILDataset
from data.dataloaders import ls_mil_collate_fn
from models.encoders import InstanceEncoder, InstanceEncoder2
from models.classifiers import Classifier
from models.mil_models import MaxPoolInstanceMIL, AttentionMIL
from training.train import train_epoch, evaluate

def run_experiment(model_type, instance_encoder, classifier, learning_rate, epochs, criterion):
    # Setup
    os.system(f'unzip -q {PATH_TO_ZIP} -d {ROOT}')
    
    # Data
    dataset_train = MILDataset(root=TRAIN_DIR)
    dataset_eval = MILDataset(root=VALID_DIR)
    train_loader = DataLoader(dataset_train, batch_size=DEFAULT_BATCH_SIZE, collate_fn=ls_mil_collate_fn)
    eval_loader = DataLoader(dataset_eval, batch_size=DEFAULT_BATCH_SIZE, collate_fn=ls_mil_collate_fn)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_type(instance_encoder, classifier).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}] for model {model_type.__name__} with LR={learning_rate}")
        train_loss, *_ = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, *_ = evaluate(model, eval_loader, criterion, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    
    return train_losses, valid_losses

def main():
    # Load class counts for computing weighted loss
    labels_df = pd.read_csv(os.path.join(TRAIN_DIR, "labels.csv"))
    class_counts = labels_df['label'].value_counts()
    weight = torch.tensor(class_counts[0] / class_counts[1], device='cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment configurations
    experiments = [
        {
            "model_type": AttentionMIL,
            "instance_encoder": InstanceEncoder(),
            "classifier": Classifier(input_size=128, hidden_size=128),
            "learning_rate": 1e-6,
            "epochs": DEFAULT_EPOCHS,
            "criterion": nn.BCELoss()  # Normal BCE Loss
        },
        {
            "model_type": AttentionMIL,
            "instance_encoder": InstanceEncoder(),
            "classifier": Classifier(input_size=128, hidden_size=128),
            "learning_rate": 1e-4,
            "epochs": DEFAULT_EPOCHS,
            "criterion": nn.BCELoss()  # Normal BCE Loss
        },
        {
            "model_type": AttentionMIL,
            "instance_encoder": InstanceEncoder2(),
            "classifier": Classifier(input_size=128, hidden_size=128),
            "learning_rate": 1e-6,
            "epochs": DEFAULT_EPOCHS,
            "criterion": nn.BCELoss()  # Normal BCE Loss
        },
        {
            "model_type": AttentionMIL,
            "instance_encoder": InstanceEncoder2(),
            "classifier": Classifier(input_size=128, hidden_size=128),
            "learning_rate": 1e-4,
            "epochs": DEFAULT_EPOCHS,
            "criterion": nn.BCELoss()  # Normal BCE Loss
        },
        {
            "model_type": MaxPoolInstanceMIL,
            "instance_encoder": InstanceEncoder(),
            "classifier": Classifier(input_size=128, hidden_size=128),
            "learning_rate": 1e-6,
            "epochs": DEFAULT_EPOCHS,
            "criterion": nn.BCELoss()  # Normal BCE Loss
        },
        {
            "model_type": MaxPoolInstanceMIL,
            "instance_encoder": InstanceEncoder2(),
            "classifier": Classifier(input_size=128, hidden_size=128),
            "learning_rate": 1e-4,
            "epochs": DEFAULT_EPOCHS,
            "criterion": nn.BCELoss()  # Normal BCE Loss
        },
        # Uncomment below to test with Weighted BCE Loss
        # {
        #     "model_type": AttentionMIL,
        #     "instance_encoder": InstanceEncoder(),
        #     "classifier": Classifier(input_size=128, hidden_size=128),
        #     "learning_rate": 1e-6,
        #     "epochs": DEFAULT_EPOCHS,
        #     "criterion": nn.BCELoss(weight=weight)  # Weighted BCE Loss
        # },
        # {
        #     "model_type": AttentionMIL,
        #     "instance_encoder": InstanceEncoder2(),
        #     "classifier": Classifier(input_size=128, hidden_size=128),
        #     "learning_rate": 1e-4,
        #     "epochs": DEFAULT_EPOCHS,
        #     "criterion": nn.BCELoss(weight=weight)  # Weighted BCE Loss
        # },
        # {
        #     "model_type": MaxPoolInstanceMIL,
        #     "instance_encoder": InstanceEncoder(),
        #     "classifier": Classifier(input_size=128, hidden_size=128),
        #     "learning_rate": 1e-6,
        #     "epochs": DEFAULT_EPOCHS,
        #     "criterion": nn.BCELoss(weight=weight)  # Weighted BCE Loss
        # },
        # {
        #     "model_type": MaxPoolInstanceMIL,
        #     "instance_encoder": InstanceEncoder2(),
        #     "classifier": Classifier(input_size=128, hidden_size=128),
        #     "learning_rate": 1e-4,
        #     "epochs": DEFAULT_EPOCHS,
        #     "criterion": nn.BCELoss(weight=weight)  # Weighted BCE Loss
        # }
    ]

    all_train_losses = []
    all_valid_losses = []
    
    for config in experiments:
        print(f"\nRunning experiment with {config['model_type'].__name__}...")
        train_losses, valid_losses = run_experiment(
            config["model_type"],
            config["instance_encoder"],
            config["classifier"],
            config["learning_rate"],
            config["epochs"],
            config["criterion"]
        )
        all_train_losses.append(train_losses)
        all_valid_losses.append(valid_losses)

        # Plotting results for each experiment
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.title(f"{config['model_type'].__name__} Training and Validation Loss (LR={config['learning_rate']})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
