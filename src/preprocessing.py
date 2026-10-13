"""[Förslag] Draft image preprocessing based on the submitted HA3.ipynb.

Status: proposed implementation for review; not an agreed pipeline component.
Checked with FerDataset samples; not integrated with project training.

Source: SSY340-group-92/HA3/HA3.ipynb.
Cells 30, 41 and 92 (counted from 1, including Markdown):
Compose/ToTensor, separate training and validation transforms, Resize,
Normalize, and training augmentation.

FER+ adaptations: default to 48x48 grayscale, make normalization optional,
and disable augmentation by default. Label processing is not included.
"""

from torchvision import transforms as T


def build_image_transform(
    size=48,
    channels=1,
    mean=None,
    std=None,
    augment=False,
):
    """Convert a PIL image to a float tensor [channels, size, size].

    ToTensor scales pixels to [0, 1] before optional normalization.
    Supply one mean/std value per channel; use training-only statistics
    or values appropriate for the chosen pretrained weights.
    Enable augmentation only for training. Standard ResNet expects 3 channels.
    """
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std")

    steps = [
        T.Grayscale(num_output_channels=channels),
        T.Resize((size, size), antialias=True),
    ]
    if augment:
        steps.append(T.RandomHorizontalFlip())

    steps.append(T.ToTensor())

    if mean is not None:
        steps.append(T.Normalize(mean, std))

    return T.Compose(steps)
