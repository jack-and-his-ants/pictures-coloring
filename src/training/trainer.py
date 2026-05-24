import os
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    model.train()

    running_loss = 0.0

    for x, y in tqdm(loader, leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        loss_sum += loss.item()

        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return (
        loss_sum / len(loader),
        correct / total
    )


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    checkpoint_path
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_loss = float("inf")

    os.makedirs(
        os.path.dirname(checkpoint_path),
        exist_ok=True
    )

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f} "
            f"acc={val_acc:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print("Saved best model")