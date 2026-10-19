from torchvision import transforms as T


def build_image_transform(size=224, channels=1, mean=None, std=None, augment=False, strong=False):
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std")

    steps = [
        T.Grayscale(num_output_channels=channels),
        T.Resize((size, size), antialias=True),
    ]
    if augment:
        deg, scale = (15, (0.8, 1.0)) if strong else (7, (0.9, 1.0))
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomRotation(deg),
            T.RandomResizedCrop(size=size, scale=scale, ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]

    steps.append(T.ToTensor())

    if mean is not None:
        steps.append(T.Normalize(mean, std))

    return T.Compose(steps)


