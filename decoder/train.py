"""
Author : Victor Fairon
Date : 2026-09-09

Train ONE MLP decoder for a given hyperparameter config on a given dataset.

Arguments (CLI):
    --dataset     path to a .npz produced by generate_datasets.py
    --config      path to a .json hyperparameter config (optional, defaults to model.DEFAULT_CONFIG)
    --output-dir  directory to write config.json / model.pth / metrics.json into
    --epochs      number of training epochs (default: 50)
    --seed        random seed, for replicability (default: 42)
    --verbose

Outputted files : 
    config.json  : the hyperparameter config + provenance (dataset path, seed, epochs)
    model.pth    : the best model checkpoint (best validation accuracy so far)
    metrics.json : training/validation loss curves + best validation accuracy/epoch
"""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from model import build_model, DEFAULT_CONFIG


def get_device():
    """Use the GPU whenever available, exactly like the stim_me.ipynb prototype."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyndromeDataset(Dataset):
    """Wraps (features, labels) tensors loaded from a generate_datasets.py .npz."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(dataset_path):
    """Load features/labels from a .npz and convert them to torch tensors."""
    data = np.load(dataset_path)
    features = torch.from_numpy(data["features"].astype(np.float32))
    labels = torch.from_numpy(data["labels"].astype(np.float32))
    return SyndromeDataset(features, labels)


def make_loaders(dataset, batch_size, seed, test_size=0.2):

    """Stratified (to conserve class balance) train/val split on the logical-flip label"""
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.labels.numpy(),
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def train(config, dataset_path, output_dir, epochs=50, seed=42, verbose=False):
    """
    Train one model for `epochs` epochs on `dataset_path`, using `config` for
    architecture/optimizer hyperparameters. Saves config.json, model.pth
    (best validation accuracy so far) and metrics.json into output_dir.

    @returns:
        dict: {"best_val_accuracy": float, "best_epoch": int,
               "train_losses": [...], "val_losses": [...]}
    """
    # Seed everything so this run is replicable given the same config/dataset/seed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    if verbose:
        print(f"Using device: {device}")

    dataset = load_dataset(dataset_path)
    input_size = dataset.features.shape[1]

    #bigger batches on GPU, smaller on CPU.
    batch_size = 1024 if device.type == "cuda" else 128
    train_loader, val_loader = make_loaders(dataset, batch_size, seed)

    model = build_model(config, input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()

    #default learning rate is 0.003 (if cant find in config dict)
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.003))

    #check that the output dictionary works
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "model.pth")

    train_losses, val_losses = [], []
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1, 1).float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.view(-1, 1).float().to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                preds = (logits > 0.0).float()
                correct += (preds == y).sum().item()
                total += x.size(0)
        val_loss /= len(val_loader)
        accuracy = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={accuracy:.4f}")

    metrics = {
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    # Save config + provenance (which dataset/seed produced this checkpoint)
    config_record = {
        **config,
        "dataset_path": os.path.abspath(dataset_path),
        "seed": seed,
        "epochs": epochs,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"Best val accuracy: {best_accuracy:.4f} (epoch {best_epoch}) -> {checkpoint_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train one MLP decoder configuration.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a .npz produced by generate_datasets.py")
    parser.add_argument("--config", type=str, default=None, help="Path to a .json config (default: model.DEFAULT_CONFIG)")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write config.json/model.pth/metrics.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed. Defaults to the seed recorded in --config if it has one "
                              "(so retraining a saved config reproduces it exactly), otherwise 42.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG

    # An explicit --seed always wins. Otherwise, reuse the seed recorded in
    # --config (e.g. a tune.py trial's config.json) so retraining it via
    # --config reproduces that exact run instead of silently defaulting to 42.
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config.get("seed", 42)

    train(config, args.dataset, args.output_dir, epochs=args.epochs, seed=seed, verbose=args.verbose)