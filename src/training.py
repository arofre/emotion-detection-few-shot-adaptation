from torch import nn, cuda, optim
import torch
from torchvision.models import resnet18, ResNet18_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision import transforms
from torch.utils.data import DataLoader

from data_loader import FerDataset
from preprocessing import build_image_transform

def training_loop(model, optimizer, criterion, train_loader, val_loader, epochs):

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for batch_nr, (image_batch, label_batch) in enumerate(train_loader):

            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            output_batch = model(image_batch)

            log_probs = torch.log_softmax(output_batch, dim=1)

            loss = criterion(log_probs, label_batch)

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
                }, "checkpoints/intermediate_resnet18.pth")

        train_loss /= len(train_loader.dataset)


        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for image_batch, label_batch in val_loader:

                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                output_batch = model(image_batch)

                log_probs = torch.log_softmax(output_batch, dim=1)

                loss = criterion(log_probs, label_batch)

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
            }, "checkpoints/best_emotion_resnet18.pth")


device = "cuda" if cuda.is_available() else "cpu"
print(device)

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

num_classes = 10

''' #classifier istället för fc för vissa modeller
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
'''
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

#Used to load a previously trained model and continue training it
if False:
    checkpoint = torch.load(
        "checkpoints/intermediate_resnet18.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])


train_dataset = FerDataset(
    "data/train/",
    build_image_transform(size=224, channels=3, augment=True, mean=weights.transforms().mean, std=weights.transforms().std),
)

val_dataset = FerDataset(
    "data/valid/",
    build_image_transform(size=224, channels=3, augment=False, mean=weights.transforms().mean, std=weights.transforms().std),
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

optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-4, weight_decay=1e-4)

criterion = nn.KLDivLoss(reduction="batchmean")

#Only train on classifier
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True


training_loop(model, optimizer, criterion, train_loader, val_loader, epochs = 3)

for param in model.layer4.parameters():
    param.requires_grad = True

training_loop(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs = 8
)
optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-5, weight_decay=1e-4)

#Train on all layers lower lr
for param in model.parameters():
    param.requires_grad = True

training_loop(model, optimizer, criterion, train_loader, val_loader, epochs = 10)