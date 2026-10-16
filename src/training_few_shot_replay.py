import copy
import numpy as np 
import pandas as pd
from torch import nn, cuda, optim
import torch
from torchvision.models import resnet50, ResNet50_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s, vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F 

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

from data_loader import FerDataset, EMOTIONS
from preprocessing import build_image_transform
from model_loader import get_model
from evaluate import evaluation_report, evaluate_model

device = "cuda" if cuda.is_available() else "cpu"
print(device)

SEED = 0 
AMPLIFY_REPS = 10 

class WeightedKLDiv(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.register_buffer("w", w)
    def forward(self, log_probs, target):
        kl = torch.xlogy(target, target) - target * log_probs
        return (kl * self.w).sum(1).mean()        


@torch.no_grad()
def compute_prototype(model, loader, device):
    model.eval()
    features = []

    for images, _ in loader:
        images = images.to(device)

        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        features.append(x)

    features = torch.cat(features, dim=0)

    features = nn.functional.normalize(features, p=2, dim=1)
    prototype = features.mean(dim=0)
    prototype = nn.functional.normalize(prototype, p=2, dim=0)

    return prototype


@torch.no_grad()
def expand_head(model, new_idx, prototype, n_total=8):
    old = model.fc[1]

    new = nn.Linear(
        old.in_features, n_total,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )

    kept = [i for i in range(n_total) if i != new_idx]

    new.weight[kept] = old.weight
    new.bias[kept] = old.bias
    new.weight[new_idx] = prototype.to(
        device=new.weight.device,
        dtype=new.weight.dtype,
    )
    new.bias[new_idx] = 0.0
    model.fc[1] = new

def set_adaptation_mode(model, keep_dropout=True):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
        elif isinstance(m, nn.Dropout) and not keep_dropout:
            m.eval()

def load_base(model, name):
    ckpt = torch.load(f"checkpoints/few_shot/{name}.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

def training_loop(model, optimizer, train_criterion, val_criterion, train_loader, val_loader, epochs, checkpoint_name, freeze_bn=False):

    for epoch in range(epochs):
        if freeze_bn:
            set_adaptation_mode(model)
        else:
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



    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, f"checkpoints/few_shot/best_{checkpoint_name}.pth")


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


def make_support(tf, k, novel, others):
    np.random.seed(SEED)
    return FerDataset("data/train/", tf, shots={**{e: 0 for e in others}, novel: k})


if __name__ == "__main__":
    model_name = "resnet50"
    model, transform = get_model("resnet50", 7, load_checkpoint=False, device=device)

    size = 112

    excluded = ["happiness"]

    train_dataset_excluded = FerDataset(
        "data/train/",
        build_image_transform(size=size, channels=3, augment=True, mean=transform.mean, std=transform.std),
        excluded_emotions=excluded
    )

    val_dataset_excluded = FerDataset(
        "data/valid/",
        build_image_transform(size=size, channels=3, augment=False, mean=transform.mean, std=transform.std),
        excluded_emotions=excluded
    )
    batch_size = 32

    train_loader_excluded = DataLoader(
        train_dataset_excluded,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader_excluded = DataLoader(
        val_dataset_excluded,
        batch_size=batch_size,
        shuffle=False
    )

    train_labels = torch.tensor(train_dataset_excluded._labels)
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
        train_loader_excluded, val_loader_excluded,
        epochs=4, checkpoint_name=f'{model_name}_stage11'
    )

    load_base(model, f'best_{model_name}_stage11')
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
        train_loader_excluded, val_loader_excluded,
        epochs=6, checkpoint_name=f'{model_name}_stage12'
    )

    load_base(model, f'best_{model_name}_stage12')
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
        train_loader_excluded, val_loader_excluded,
        epochs=8, checkpoint_name=f'{model_name}_stage13'
    )

    load_base(model, f'best_{model_name}_stage13')

    test_dataset_excluded = FerDataset(
        root="data/test/",
        transform=build_image_transform(
            size=112,
            channels=3,
            augment=False,
            mean=transform.mean,
            std=transform.std,
        ),
        excluded_emotions=excluded
    )

    test_loader_excluded = DataLoader(
        test_dataset_excluded,
        batch_size=32,
        shuffle=False,
    )

    new_idx = EMOTIONS.index(excluded[0]) 
    target_map = torch.tensor([i for i in range(8) if i != new_idx])
    results = evaluate_model(model, test_loader_excluded, device=device, target_map=target_map, map_predictions=True)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    ref_base_acc = results["accuracy"] 

    report_before = evaluation_report(results["targets"], results["predictions"], EMOTIONS, f"{model_name}_stage1", show=False)

    others = [e for e in EMOTIONS if e != excluded[0]]
    eval_transform = build_image_transform(size=size, channels=3, augment=False, mean=transform.mean, std=transform.std)
    strong_transform = build_image_transform(size=size, channels=3, augment=True, mean=transform.mean, std=transform.std, strong=True)

    test_dataset_full = FerDataset(root="data/test/", transform=eval_transform)
    test_loader_full = DataLoader(test_dataset_full, batch_size=32, shuffle=False)

    val_dataset_full = FerDataset("data/valid/", eval_transform)
    val_loader_full = DataLoader(val_dataset_full, batch_size=batch_size, shuffle=False)

    base_state = copy.deepcopy(model.state_dict())
    rows = []  

    for k in [20, 50, 100, 200, 500]:
        for replay_multiplier in [0.5, 1, 2, 5, 10]:
            model.fc[1] = nn.Linear(base_state["fc.1.weight"].size(1), 7).to(device)
            model.load_state_dict(base_state)

            torch.manual_seed(SEED) 

            replay_size = int(k * replay_multiplier)

            np.random.seed(SEED)
            old_dataset = FerDataset(
                "data/train/",
                transform=strong_transform,
                shots={**{e: replay_size for e in others}, excluded[0]: 0},
            )
            support_proto = make_support(eval_transform, k, excluded[0], others)
            support_train = make_support(strong_transform, k, excluded[0], others)

            train_dataset_full = ConcatDataset([old_dataset] + [support_train] * AMPLIFY_REPS)

            train_loader_full = DataLoader(
                train_dataset_full,
                batch_size=batch_size,
                shuffle=True,
            )

            prototype_loader = DataLoader(
                support_proto, 
                batch_size=batch_size,
                shuffle=False
            )

            train_labels = torch.tensor(np.concatenate([old_dataset._labels] + [support_train._labels] * AMPLIFY_REPS))
            prior = train_labels.mean(0)

            w = prior.clamp_min(1e-6) ** -0.5
            w = w / (w * prior).sum()

            train_criterion = WeightedKLDiv(w.to(device))
            val_criterion = nn.KLDivLoss(reduction="batchmean")

            prototype = compute_prototype(model, prototype_loader, device=device)
            expand_head(model, new_idx=EMOTIONS.index(excluded[0]), prototype=prototype, n_total=8)

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
                train_loader_full, val_loader_full,
                epochs=6, checkpoint_name=f'{model_name}_stage2',
                freeze_bn=True
            )

            load_base(model, f'best_{model_name}_stage2')

            results_old = evaluate_model(model, test_loader_excluded, device=device, target_map=target_map)
            report_stage2_excluded = evaluation_report(results_old["targets"], results_old["predictions"], EMOTIONS, f"replay_{excluded[0]}/{model_name}_k{k}_r{replay_multiplier}_stage2_excluded", show=False)

            results = evaluate_model(model, test_loader_full, device=device)
            report_after = evaluation_report(results["targets"], results["predictions"], EMOTIONS, f"replay_{excluded[0]}/{model_name}_k{k}_r{replay_multiplier}_stage2", show=False)