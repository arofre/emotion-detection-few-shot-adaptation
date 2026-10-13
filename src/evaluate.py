import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet18, ResNet18_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s, resnet50, vit_b_16, ViT_B_16_Weights

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def plot_confusion_matrix(targets, predictions, class_names, save_path="confusion_matrix.png"):
    labels = list(range(len(class_names)))
    y_true, y_pred = targets.numpy(), predictions.numpy()

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ConfusionMatrixDisplay(cm_counts, display_labels=class_names).plot(
        ax=axes[0], cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d"
    )
    axes[0].set_title("Confusion matrix (counts)")

    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
        ax=axes[1], cmap="Blues", xticks_rotation=45, colorbar=False, values_format=".2f"
    )
    axes[1].set_title("Confusion matrix (row-normalized)")

    plt.show()

    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))


def evaluate_model(model, dataloader, device="cpu", loss_fn=None):
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

    weights = ResNet50_Weights.IMAGENET1K_V2

    #weights = ViT_B_16_Weights.DEFAULT
    model = resnet50(weights=weights)
    #model = vit_b_16(weights=weights)
    num_classes = 8
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    #model.heads.head = nn.Sequential(
    #    nn.Dropout(p=0.4),
    #    nn.Linear(model.heads.head.in_features, num_classes)
    #)


    checkpoint = torch.load(
            f"checkpoints/best_{model.__class__.__name__.lower()}.pth",
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

    class_names = [
        "neutral", "happiness", "surprise", "sadness", "anger",
        "disgust", "fear", "contempt",
    ]
    plot_confusion_matrix(results["targets"], results["predictions"], class_names)