"""Compare saved FER+ models only: test metrics, latency and resources."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import EMOTIONS, FerDataset
from models import MODEL_NAMES, RECIPES, build_model
from preprocessing import build_image_transform
from profiling import profile_resources


def classification_metrics(targets, predictions, class_names):
    """Per-class and macro metrics, including absent classes with zero scores."""

    # Confusion matrix: rows are targets, columns are predictions.
    n = len(class_names)
    indices = torch.tensor(targets) * n + torch.tensor(predictions)
    matrix = torch.bincount(indices, minlength=n * n).reshape(n, n)

    # Calculate per-class scores; undefined scores become zero.
    counts = matrix.float()
    tp = counts.diag()
    precision = tp / counts.sum(0).clamp_min(1)
    recall = tp / counts.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    # Package per-class scores and their unweighted macro averages.
    report = {name: {"precision": precision[i].item(), "recall": recall[i].item(),
                     "f1-score": f1[i].item(), "support": int(matrix[i].sum())}
              for i, name in enumerate(class_names)}
    report["macro avg"] = {"precision": precision.mean().item(),
                           "recall": recall.mean().item(), "f1-score": f1.mean().item()}
    return report, matrix.tolist()


def evaluate_saved_model(model, loader, device):
    """Evaluate FER+ soft targets without an optimizer or training side effects."""
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = (torch.xlogy(labels, labels) - labels * outputs.log_softmax(1)).sum(1).mean()
            predicted, target = outputs.argmax(1), labels.argmax(1)
            n = images.size(0)
            total += n
            correct += (predicted == target).sum().item()
            loss_sum += loss.item() * n
            targets.extend(target.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
    if not total:
        raise ValueError('Test split contains no usable samples')
    return dict(loss=loss_sum / total, accuracy=correct / total,
                targets=targets, predictions=predictions)


def benchmark(model, sample, device, warmup=10, repeats=50):
    """Batch-one classifier latency; excludes transforms, transfers and detection."""
    model.eval()
    sample = sample.to(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)


    # Warm up before measuring; synchronize CUDA to time completed work.
    timings = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize()

        # Time batch-one inference only.
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            model(sample)
            synchronize()
            timings.append((time.perf_counter() - start) * 1000)

    return {"latency_median_ms": float(np.median(timings)),
            "latency_p95_ms": float(np.percentile(timings, 95))}


def print_summary(rows):
    """Print aligned comparison columns; classification scores shown as percentages."""
    columns = [
        ("Model", "model", lambda value: str(value)),
        ("Role", "role", lambda value: str(value)),
        ("Training", "training", lambda value: str(value)),
        ("Epochs", "epochs_display", lambda value: str(value)),
        ("Input", "size", lambda value: str(value)),
        ("Parameters", "parameters", lambda value: f"{value:,}"),
        ("Accuracy %", "accuracy", lambda value: f"{value * 100:.2f}"),
        ("KL loss", "test_kl", lambda value: f"{value:.4f}"),
        ("Precision %", "macro_precision", lambda value: f"{value * 100:.2f}"),
        ("Recall %", "macro_recall", lambda value: f"{value * 100:.2f}"),
        ("F1 %", "macro_f1", lambda value: f"{value * 100:.2f}"),
        ("Median ms", "latency_median_ms", lambda value: f"{value:.3f}"),
        ("P95 ms", "latency_p95_ms", lambda value: f"{value:.3f}"),
    ]

    # Align columns using both header and formatted value widths.
    values = [[formatter(row[key]) for _, key, formatter in columns] for row in rows]
    widths = [max(len(title), *(len(row[i]) for row in values))
              for i, (title, _, _) in enumerate(columns)]

    def line(cells):
        return "  ".join(cell.ljust(width) if i == 0 else cell.rjust(width)
                         for i, (cell, width) in enumerate(zip(cells, widths)))


    # Show the complete comparison after all models finish.
    print("\nTest results (precision/recall/F1: macro averages)", flush=True)
    print(line([title for title, _, _ in columns]), flush=True)
    print("  ".join("-" * width for width in widths), flush=True)
    for row in values:
        print(line(row), flush=True)
    if any(row.get('epochs_source') == 'reference recipe' for row in rows):
        print('* 18 epochs from the original 4+6+8 recipe; not recorded as a total in the checkpoint.', flush=True)

    # Keep resource measurements separate so both tables fit typical terminals.
    print("\nClassifier resources (one face crop)", flush=True)
    print(f"{'Model':<20} {'Conv/FC MMAC':>13} {'State MiB':>11} {'Extra GPU MiB':>15}", flush=True)
    for row in rows:
        peak = row["inference_extra_cuda_mib"]
        peak_text = "N/A" if peak is None else f"{peak:.2f}"
        print(f"{row['model']:<20} {row['conv_linear_macs'] / 1e6:>13.2f} "
              f"{row['model_state_mib']:>11.2f} {peak_text:>15}", flush=True)
    print("MAC: Conv/Linear only; State: weights + buffers; Extra GPU: forward-pass allocations.", flush=True)
    print("CPU peak memory and end-to-end webcam latency: not measured.", flush=True)


def main():
    # Comparison only: never train or download pretrained weights here.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=None)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='MODEL=PATH')
    parser.add_argument('--root', type=Path, default=Path('checkpoints/modelspeedtest'))
    parser.add_argument('--reference', type=Path, default=Path('checkpoints/best_resnet.pth'))
    parser.add_argument('--no-reference', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    if min(args.batch_size, args.threads) <= 0:
        parser.error('batch-size and threads must be positive')
    explicit = {}
    for value in args.checkpoint:
        name, separator, filename = value.partition('=')
        if not separator or name not in MODEL_NAMES or name in explicit:
            parser.error('Use a unique known MODEL=PATH')
        explicit[name] = Path(filename)
        if not explicit[name].is_file():
            parser.error(f'Checkpoint not found: {filename}')

    # Prefer each model's current folder. Older comparison runs remain usable.
    selected = []
    requested = args.models if args.models is not None else list(MODEL_NAMES)
    requested = list(dict.fromkeys([*requested, *explicit]))
    for name in requested:
        path = explicit.get(name, args.root / name / f'best_{name}.pth')
        if not path.is_file() and name not in explicit:
            older = list(args.root.glob(f'run-*/best_{name}.pth'))
            older += list(args.root.glob(f'run-*/{name}/best.pth'))
            if older:
                path = max(older, key=lambda p: p.stat().st_mtime)
        if path.is_file():
            selected.append((name, path, 'Candidate'))
        elif args.models is not None:
            parser.error(f'No saved {name}; run train_models.py first or use --checkpoint')
    if not args.no_reference:
        if args.reference.is_file():
            # Do not duplicate an explicitly selected reference checkpoint.
            selected = [item for item in selected if item[1].resolve() != args.reference.resolve()]
            selected.insert(0, ('resnet50', args.reference, 'Reference'))
        else:
            print(f'Reference not found: {args.reference}', flush=True)
    if not selected:
        parser.error('No saved models found; run train_models.py first')

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    hardware = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f'Device: {device} ({hardware})', flush=True)
    result_folder = args.root / 'comparison'
    result_folder.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, path, role in selected:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        metadata = checkpoint.get('config', {})
        # Latest-run status is stored alongside the model; avoid mixed interrupted runs.
        config_path = path.parent / 'config.json'
        if config_path.is_file():
            current = json.loads(config_path.read_text(encoding='utf-8-sig'))
            if current.get('status') == 'training':
                raise ValueError(f'{name}: latest training is incomplete; finish it before comparison')
            metadata = current
        if metadata.get('classes', EMOTIONS) != EMOTIONS or metadata.get('architecture', name) != name:
            raise ValueError(f'{path}: incompatible architecture or class order')
        recipe = RECIPES[name]
        transform = metadata.get('preprocessing', dict(size=recipe['size'], channels=3,
                                                       mean=recipe['mean'], std=recipe['std']))
        if not metadata:
            print(f'{name}: legacy checkpoint; assuming recipe preprocessing and EMOTIONS order', flush=True)
        model = build_model(name, len(EMOTIONS), pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        dataset = FerDataset(args.data / 'test', build_image_transform(**transform))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        evaluated = evaluate_saved_model(model, loader, device)
        report, confusion = classification_metrics(evaluated['targets'], evaluated['predictions'], EMOTIONS)
        # Total training budget differs from the epoch of the selected checkpoint.
        stages = metadata.get('stages', [])
        epochs = sum(stage[1] for stage in stages) if stages else metadata.get('epochs')
        epochs_source = 'metadata' if epochs is not None else 'unknown'
        description = metadata.get('training_description', 'Checkpoint (unknown)')
        is_aron = role == 'Reference' and path.resolve() == Path('checkpoints/best_resnet.pth').resolve()
        if is_aron:
            description = 'Head/block/full (Aron)'
            if epochs is None:
                epochs, epochs_source = 18, 'reference recipe'
        epochs_display = '?' if epochs is None else str(epochs)
        if epochs_source == 'reference recipe':
            epochs_display += '*'
        row = dict(model=name, role=role, training=description, size=transform['size'],
                   epochs=epochs, epochs_display=epochs_display, epochs_source=epochs_source,
                   best_checkpoint_epoch=checkpoint.get('epoch'),
                   parameters=sum(p.numel() for p in model.parameters()),
                   accuracy=evaluated['accuracy'], test_kl=evaluated['loss'],
                   macro_precision=report['macro avg']['precision'],
                   macro_recall=report['macro avg']['recall'], macro_f1=report['macro avg']['f1-score'],
                   checkpoint=str(path), device=str(device), hardware=hardware,
                   **benchmark(model, dataset[0][0].unsqueeze(0), device),
                   **profile_resources(model, dataset[0][0].unsqueeze(0), device))
        identifier = f'reference_{name}' if role == 'Reference' else name
        (result_folder / f'{identifier}.json').write_text(json.dumps(dict(summary=row,
            per_class=report, confusion_matrix=confusion, preprocessing=transform), indent=2), encoding='utf-8')
        rows.append(row)
        del model, checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    with (args.root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f'\nResults: {(args.root / "summary.csv").resolve()}', flush=True)


if __name__ == '__main__':
    main()

