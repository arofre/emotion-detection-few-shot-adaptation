import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




def evaluate_model(model, dataloader, device="cpu", loss_fn=None):
    """Return accuracy, optional loss, predictions and targets.

    Expects batches (images, targets), with integer targets [batch_size]
    and model outputs [batch_size, num_classes]. Raw FER+ vote counts are
    not accepted. Use deterministic evaluation transforms.

    The caller places the model on device. If supplied, loss_fn must return
    a scalar batch mean (e.g. standard CrossEntropyLoss, without weights).
    Predictions and targets are returned as CPU tensors; accuracy is 0..1.
    """
    was_training = model.training
    correct = total = 0
    loss_sum = 0.0
    predictions, targets = [], []
    model.eval()

    try:
        with torch.inference_mode():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                predicted = torch.argmax(outputs, dim=1)
                target_indices = torch.argmax(labels, dim=1)

                batch_size = labels.size(0)

                correct += (predicted == target_indices).sum().item()
                #print("Images:", images.shape)
                #print("Labels:", labels.shape, labels.dtype)
                #print("Labels sample:", labels[:5])
                #print("Outputs:", outputs.shape)
                total += batch_size
                predictions.append(predicted.cpu())
                targets.append(target_indices.cpu())
                if loss_fn is not None:
                    loss_sum += loss_fn(outputs, labels).item() * batch_size

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        model.train(was_training)

    return {
        "accuracy": correct / total,
        "loss": loss_sum / total if loss_fn is not None else None,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
    }


if __name__ == "__main__":
    from torchvision.models import resnet18, ResNet18_Weights
    from torch.utils.data import DataLoader
    from preprocessing import build_image_transform
    from data_loader import FerDataset

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)
    num_classes = 10
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )

    checkpoint = torch.load(
            f"checkpoints/best_emotion_{model.__class__.__name__.lower()}.pth",
            map_location=device,
            weights_only=True

        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_dataset = FerDataset(
        root="data/test/",
        transform=build_image_transform(
            size=112,
            channels=3,
            augment=False,
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )
    results = evaluate_model(model, test_loader, device=device)
    print(f"Test Accuracy: {results['accuracy']:.4f}")