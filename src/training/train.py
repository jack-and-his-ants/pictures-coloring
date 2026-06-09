from torch.utils.data import DataLoader

from src.models.colorization_network import ColorizationNetwork
from src.datasets.colorization_dataset import ColorizationDataset
from src.training.trainer import train_model
from src.utils.device import get_device


def main():

    device = get_device()

    train_dataset = ColorizationDataset(
        "dataset/L_luminance",
        "dataset/class_labels",
        "splits/train_split.txt"
    )

    val_dataset = ColorizationDataset(
        "dataset/L_luminance",
        "dataset/class_labels",
        "splits/validation_split.txt"
    )
    from collections import Counter
    import numpy as np
    import torch

    counter = Counter()

    for _, y in train_dataset:

        values, counts = np.unique(
            y.numpy(),
            return_counts=True
        )

        for v, c in zip(values, counts):
            counter[int(v)] += int(c)

    weights = []

    for i in range(32):
        freq = counter[i]

        weights.append(
            1.0 / np.sqrt(freq)
        )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(device)

    weights /= weights.mean()

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    model = ColorizationNetwork(
        n_classes=32
    ).to(device)

    train_model(
        weights,
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3
    )


if __name__ == "__main__":
    main()