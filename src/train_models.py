"""Select a model and pass it to Aron's original training loop."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader import EMOTIONS, FerDataset
from models import MODEL_NAMES, SCRATCH_MODELS, RECIPES, build_model, training_stages
from preprocessing import build_image_transform
from training import WeightedKLDiv, training_loop


def train_model(name, epochs=18, device=None, data=Path('data'),
                root=Path('checkpoints/modelspeedtest'), batch_size=32, seed=42):
    """Reuse staged policy for pretrained models; full training for scratch."""
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    recipe = RECIPES[name]
    pretrained = name not in SCRATCH_MODELS
    model = build_model(name, len(EMOTIONS), pretrained=pretrained).to(device)
    transform = dict(size=recipe['size'], channels=3, mean=recipe['mean'], std=recipe['std'])

    # Same data loading, augmentation and class weighting as the original script.
    train = FerDataset(Path(data) / 'train', build_image_transform(**transform, augment=True))
    valid = FerDataset(Path(data) / 'valid', build_image_transform(**transform))
    if not len(train) or not len(valid):
        raise ValueError('Training and validation splits must be nonempty')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    prior = torch.tensor(train._labels).mean(0)
    weights = prior.clamp_min(1e-6).pow(-0.5)
    weights = weights / (weights * prior).sum()
    train_criterion = WeightedKLDiv(weights.to(device))
    val_criterion = nn.KLDivLoss(reduction='batchmean')

    stages = training_stages(name, model, epochs)
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(architecture=name, classes=EMOTIONS, preprocessing=transform,
                  pretrained=pretrained, training_description='Head/block/full' if pretrained else 'Full / scratch',
                  epochs=epochs, stages=stages, batch_size=batch_size, weight_decay=1e-4,
                  augment=True, seed=seed, device=str(device), status='training')
    config_path = folder / 'config.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    hardware = torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU'
    print(f'Device: {device} ({hardware})\n{name}: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    print(f'Latest run replaces files in: {folder.resolve()}', flush=True)

    # Same original loop for each stage; only trainable layers and learning rates vary.
    best_path = folder / f'best_{name}.pth'
    best_loss = float('inf')
    best_checkpoint = None
    for stage, count, layer_lrs in stages:
        for parameter in model.parameters():
            parameter.requires_grad = False
        groups = []
        for layer, lr in layer_lrs.items():
            parameters = list(model.get_submodule(layer).parameters())
            for parameter in parameters:
                parameter.requires_grad = True
            if parameters:
                groups.append({'params': parameters, 'lr': lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        print(f'Stage: {stage}, epochs={count}', flush=True)
        training_loop(model, optimizer, train_criterion, val_criterion, train_loader,
                      val_loader, count, device=device, best_path=best_path,
                      intermediate_path=folder / f'intermediate_{name}.pth')
        # Original loop selects within a stage. Preserve the best across all stages.
        saved = torch.load(best_path, map_location='cpu', weights_only=False)
        saved['stage'] = stage
        if saved['val_loss'] < best_loss:
            best_loss, best_checkpoint = saved['val_loss'], saved
    torch.save(best_checkpoint, best_path)
    config['status'] = 'complete'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return folder / f'best_{name}.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', metavar='MODEL', help=f'Choose from {MODEL_NAMES}')
    parser.add_argument('--epochs', type=int, default=18, help='Total budget; default 4+6+8 for pretrained models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    args = parser.parse_args()
    args.models = args.models or ['resnet18', 'mobilenet_v3_small', 'squeezenet1_1', 'small_cnn', 'balanced_cnn']
    if any(name not in MODEL_NAMES for name in args.models) or len(set(args.models)) != len(args.models):
        parser.error('Choose unique known model names')
    for name in args.models:
        print('Saved:', train_model(name, args.epochs, args.device))


if __name__ == '__main__':
    main()
