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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        lr=1e-3,
        checkpoint_path="checkpoints/best_model.pth"
    )


if __name__ == "__main__":
    main()