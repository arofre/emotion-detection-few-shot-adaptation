"""[Förslag] Draft evaluation utilities based on submitted HA notebooks.

Status: proposed implementation for review; not an agreed pipeline component.
Checked with small synthetic inputs; not integrated with project training.

Sources in SSY340-group-92 (cells counted from 1, including Markdown):
- HA2/HA2.ipynb, cell 61: evaluate_model (eval/no_grad, argmax, accuracy).
- HA3/HA3.ipynb, cell 55: validate (validation loss and accuracy).
- HA4/HA4-Part2.ipynb, cell 50: predict_epoch (collect CPU predictions).

Project adaptations: multiclass targets, sample-weighted mean loss, counting
actually processed samples, and restoring the previous training mode.
Precision/recall/F1 and FER+ label processing are not implemented here.
"""

import torch


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
        with torch.no_grad():
            for images, labels in dataloader:
                if labels.ndim != 1 or labels.dtype != torch.long:
                    raise ValueError(
                        "Expected int64 class targets [batch_size]; "
                        "process FER+ vote counts before evaluation."
                    )
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if outputs.ndim != 2 or outputs.shape[0] != labels.shape[0]:
                    raise ValueError("Expected model outputs [batch_size, num_classes]")

                predicted = outputs.argmax(dim=1)
                batch_size = labels.numel()
                correct += (predicted == labels).sum().item()
                total += batch_size
                predictions.append(predicted.cpu())
                targets.append(labels.cpu())

                if loss_fn is not None:
                    loss_sum += loss_fn(outputs, labels).item() * batch_size
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Evaluation requires at least one sample")

    return {
        "accuracy": correct / total,
        "loss": loss_sum / total if loss_fn is not None else None,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
    }
