from matplotlib.artist import get
from torch import nn, cuda, optim
import torch
from torchvision.models import resnet50, ResNet50_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s, vit_b_16, ViT_B_16_Weights

from torchvision import transforms
from torch.utils.data import DataLoader

from data_loader import FerDataset
from preprocessing import build_image_transform
from model_loader import get_model

class WeightedKLDiv(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.register_buffer("w", w)
    def forward(self, log_probs, target):
        kl = torch.xlogy(target, target) - target * log_probs
        return (kl * self.w).sum(1).mean()


def training_loop(model, optimizer, train_criterion, val_criterion, train_loader, val_loader, epochs, checkpoint_name):

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for batch_nr, (image_batch, label_batch) in enumerate(train_loader):

            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                output_batch = model(image_batch)

                log_probs = torch.log_softmax(output_batch, dim=1)

                loss = train_criterion(log_probs, label_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * image_batch.size(0)
            if batch_nr % 100 == 0:
                print(
                    f"Batch {batch_nr}: "
                    f"Loss = {loss.item():.4f}"
                )

            if batch_nr % 100 == 0:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                }, f"checkpoints/intermediate/intermediate_{checkpoint_name}.pth")

        train_loss /= len(train_loader.dataset)


        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for image_batch, label_batch in val_loader:

                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                output_batch = model(image_batch)

                log_probs = torch.log_softmax(output_batch, dim=1)

                loss = val_criterion(log_probs, label_batch)

                val_loss += loss.item() * image_batch.size(0)

        val_loss /= len(val_loader.dataset)


        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, f"checkpoints/best_{checkpoint_name}.pth")


def set_trainable_layers(model, trainable_layers):
    for param in model.parameters():
        param.requires_grad = False

    for layer_name in trainable_layers:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True


def make_optimizer(model, layer_lrs, weight_decay=1e-4):
    return torch.optim.AdamW(
        [
            {
                "params": getattr(model, layer_name).parameters(),
                "lr": lr,
            }
            for layer_name, lr in layer_lrs.items()
        ],
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    device = "cuda" if cuda.is_available() else "cpu"
    print(device)


    model_name = "resnet18"
    model, transform = get_model(model_name, 8, load_checkpoint=False, device=device)

    size = 112

    train_dataset = FerDataset(
        "data/train/",
        build_image_transform(size=size, channels=3, augment=True, mean=transform.mean, std=transform.std),
    )

    val_dataset = FerDataset(
        "data/valid/",
        build_image_transform(size=size, channels=3, augment=False, mean=transform.mean, std=transform.std),
    )
    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    train_labels = torch.tensor(train_dataset._labels)
    prior = train_labels.mean(0)

    w = prior.clamp_min(1e-6) ** -0.5
    w = w / (w * prior).sum()

    train_criterion = WeightedKLDiv(w.to(device))
    val_criterion = nn.KLDivLoss(reduction="batchmean")

    set_trainable_layers(model, ["fc"])

    optimizer = make_optimizer(
        model,
        {
            "fc": 1e-3,
        },
    )

    training_loop(
        model, optimizer,
        train_criterion, val_criterion,
        train_loader, val_loader,
        epochs=4, checkpoint_name=model_name
    )

    set_trainable_layers(model, ["layer4", "fc"])

    optimizer = make_optimizer(
        model,
        {
            "layer4": 1e-4,
            "fc": 5e-4,
        },
    )

    training_loop(
        model, optimizer,
        train_criterion, val_criterion,
        train_loader, val_loader,
        epochs=6, checkpoint_name=model_name
    )

    set_trainable_layers(model, ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"])

    optimizer = make_optimizer(
        model,
        {
            "conv1": 1e-6,
            "bn1": 1e-6,
            "layer1":1e-6,
            "layer2": 2e-6,
            "layer3": 5e-6,
            "layer4": 1e-5,
            "fc": 5e-5,
        },
    )

    training_loop(
        model, optimizer,
        train_criterion, val_criterion,
        train_loader, val_loader,
        epochs=8, checkpoint_name=model_name
    )