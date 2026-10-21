import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet18, ResNet18_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s, resnet50, vit_b_16, ViT_B_16_Weights

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import os

from model_loader import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluation_report(targets, predictions, class_names, save_name=None, show=True):
    labels = list(range(len(class_names)))
    y_true, y_pred = targets.numpy(), predictions.numpy()

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ConfusionMatrixDisplay(cm_counts, display_labels=class_names).plot(
        ax=axes[0], cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    axes[0].set_title("Confusion matrix (counts)")
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
        ax=axes[1], cmap="Blues", xticks_rotation=45, colorbar=False, values_format=".2f")
    axes[1].set_title("Confusion matrix (row-normalized)")

    if save_name:
        os.makedirs(f"results/{save_name}", exist_ok=True)
        fig.savefig(f"results/{save_name}/confusion.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=class_names, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=class_names, zero_division=0))
    if save_name:
        os.makedirs(f"results/{save_name}", exist_ok=True)
        pd.DataFrame(report).T.to_csv(f"results/{save_name}/classification_report.csv")
    return report



def evaluate_model(model, dataloader, device="cpu", loss_fn=None, target_map=None, map_predictions=False):
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

                if target_map is not None: 
                    target_map = target_map.to(device)
                    target_indices = target_map[target_indices]
                    if map_predictions:
                        predicted = target_map[predicted]

                batch_size = labels.size(0)

                correct += (predicted == target_indices).sum().item()

                total += batch_size
                predictions.append(predicted.cpu())
                targets.append(target_indices.cpu())
                if loss_fn is not None:
                    loss_sum += loss_fn(outputs, labels).item() * batch_size

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        # this ensures that model is back to training state to prevent some errors
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


    model, transform = get_model("resnet50", 8, load_checkpoint=True, device=device)

    test_dataset = FerDataset(
        root="data/test/",
        transform=build_image_transform(
            size=112,
            channels=3,
            augment=False,
            mean=transform.mean,
            std=transform.std,
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
    evaluation_report(results["targets"], results["predictions"], class_names)