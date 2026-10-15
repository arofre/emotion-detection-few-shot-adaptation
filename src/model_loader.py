
import torch
from torchvision.models import ViT_B_16_Weights, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s, vit_b_16, ViT_B_32_Weights, vit_b_32

def get_model(model_name, n_classes, load_checkpoint, device):
    if load_checkpoint:
        checkpoint = torch.load(
            f"checkpoints/best_{model_name}.pth",
            map_location=device,
            weights_only=True
        )

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(model.fc.in_features, n_classes)
        )
    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(model.fc.in_features, n_classes)
        )

    if model_name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V2
            model = vit_b_16(weights=weights)
            model.heads.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(model.heads.head.in_features, n_classes)
        )

    if model_name == "efficientnet":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(model.classifier[1].in_features, n_classes)
        )

    if load_checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    return model, weights.transforms()

