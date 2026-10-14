"""Explicit model recipes for FER+ deployment comparisons (proposal).

Each initializer documents its head and input recipe. ImageNet weights are
only a starting point; FER+ checkpoints can instead be evaluated directly.
"""

from torch import nn
from torchvision import models

MODEL_NAMES = ('resnet18', 'resnet50', 'mobilenet_v3_small', 'squeezenet1_1', 'small_cnn', 'balanced_cnn')
SCRATCH_MODELS = ('small_cnn', 'balanced_cnn')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SmallCNN(nn.Module):
    """Small spatial inputs and global pooling avoid a large dense classifier."""

    def __init__(self, num_classes, dropout):
        super().__init__()
        layers = []
        channels = 3
        for width in (16, 32, 64):
            layers.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ])
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class BalancedCNN(nn.Module):
    """More capacity than SmallCNN while retaining small inputs and global pooling."""

    def __init__(self, num_classes, dropout):
        super().__init__()

        # Two 3x3 convolutions per block learn richer features before downsampling.
        layers = []
        channels = 3
        for width in (32, 64, 128):
            layers.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            channels = width
        self.features = nn.Sequential(*layers)

        # Keep the classifier small instead of flattening every spatial activation.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def init_resnet18(num_classes, pretrained, dropout):
    # Familiar baseline; retain the backbone and replace only the class head.
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def init_resnet50(num_classes, pretrained, dropout):
    # Matches training.py's architecture and Dropout/Linear head, including Aron.
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def init_mobilenet(num_classes, pretrained, dropout):
    # Compact mobile backbone with a FER+ classification head.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[2] = nn.Dropout(dropout)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def init_squeezenet(num_classes, pretrained, dropout):
    # Convolutional head; remove output ReLU to allow unrestricted FER+ logits.
    weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
    model = models.squeezenet1_1(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1), nn.AdaptiveAvgPool2d(1),
    )
    model.num_classes = num_classes
    return model


# Recipes are starting hypotheses, not claims of each model's optimal input size.
# 112 follows Aron's training input; SmallCNN retains the original 48x48 detail.
RECIPES = {
    'resnet18': dict(builder=init_resnet18, size=112, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    'resnet50': dict(builder=init_resnet50, size=112, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    'mobilenet_v3_small': dict(builder=init_mobilenet, size=112, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    'squeezenet1_1': dict(builder=init_squeezenet, size=112, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    'small_cnn': dict(builder=lambda n, p, d: SmallCNN(n, d), size=48, mean=None, std=None),
    'balanced_cnn': dict(builder=lambda n, p, d: BalancedCNN(n, d), size=48, mean=None, std=None),
}


def build_model(name, num_classes=8, pretrained=None, dropout=0.4):
    # Default to ImageNet weights where available; never silently fall back.
    if pretrained is None:
        pretrained = name not in SCRATCH_MODELS
    if name not in RECIPES or num_classes < 2:
        raise ValueError('Choose a known architecture and at least two classes')
    if name in SCRATCH_MODELS and pretrained:
        raise ValueError(f'{name} has no pretrained weights')
    return RECIPES[name]['builder'](num_classes, pretrained, dropout)


def training_stages(name, model, epochs=18):
    """Aron's 4/6/8 schedule, with architecture-specific layer paths.

    Custom CNNs train fully from scratch. Other budgets retain approximate
    stage proportions; pretrained models require at least three epochs.
    """
    if name in SCRATCH_MODELS:
        return [('full', epochs, {'': 1e-3})]
    if epochs < 3:
        raise ValueError('Staged pretrained training needs at least 3 total epochs')
    head_count = max(1, epochs * 4 // 18)
    block_count = max(1, epochs * 6 // 18)
    full_count = epochs - head_count - block_count
    if name.startswith('resnet'):
        head = 'fc'
        last = {'layer4': 1e-4, head: 5e-4}
        full = {'conv1': 1e-6, 'bn1': 1e-6, 'layer1': 1e-6,
                'layer2': 2e-6, 'layer3': 5e-6, 'layer4': 1e-5, head: 5e-5}
    else:
        head = 'classifier'
        # MobileNet: last inverted residual + final convolution; SqueezeNet: last Fire.
        last_indices = (len(model.features) - 2, len(model.features) - 1) if name == 'mobilenet_v3_small' else (12,)
        last = {f'features.{i}': 1e-4 for i in last_indices}
        last[head] = 5e-4
        full = {f'features.{i}': 1e-5 if i in last_indices else 1e-6
                for i in range(len(model.features))}
        full[head] = 5e-5
    return [('head', head_count, {head: 1e-3}),
            ('last_block', block_count, last), ('full', full_count, full)]
