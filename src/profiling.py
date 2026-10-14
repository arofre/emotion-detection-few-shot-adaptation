"""Classifier resource measurements without additional dependencies.

MAC estimate counts Conv2d and Linear only, not normalization, activations,
pooling or other operations. One MAC is often reported as two FLOPs.
Model-state size includes parameters and buffers, not runtime activations.
CUDA peak is additional allocated tensor memory during one forward pass,
not reserved memory, process memory or complete webcam-pipeline memory.
"""

import torch
from torch import nn


def profile_resources(model, sample, device):
    """Measure one face crop; return None for unavailable CPU peak memory."""
    sample = sample.to(device)
    was_training = model.training
    model.eval()
    macs = 0
    handles = []

    # Count output positions times multiplications per Conv/Linear output.
    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs += output.numel() * (module.in_channels // module.groups) * kh * kw
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(count))
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    state_bytes = sum(t.numel() * t.element_size()
                      for t in (*model.parameters(), *model.buffers()))

    # Measure transient allocations separately from MAC counting and timing.
    extra_peak = None
    if device.type == "cuda":
        model.eval()
        try:
            # Initialize kernels/workspaces before measuring transient allocations.
            with torch.inference_mode():
                for _ in range(3):
                    model(sample)
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                output = model(sample)
            torch.cuda.synchronize(device)
            extra_peak = max(0, torch.cuda.max_memory_allocated(device) - baseline) / 2**20
            del output
        finally:
            model.train(was_training)

    return {
        "conv_linear_macs": macs,
        "model_state_mib": state_bytes / 2**20,
        "inference_extra_cuda_mib": extra_peak,
        "end_to_end_latency_ms": None,
    }
