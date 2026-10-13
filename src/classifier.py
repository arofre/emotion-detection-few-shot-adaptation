import torch
from torch import nn
from PIL import Image
from data_loader import CLASSES
import cv2
import numpy as np

def classify_face(images, model, preprocess, device):
    processed_images = []

    for image in images:
        if image is None or image.size == 0:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        image = Image.fromarray(image)
        image = preprocess(image)
        processed_images.append(image)

    if not processed_images:
        return [], np.empty((0, 10))

    images = torch.stack(processed_images).to(device)

    with torch.inference_mode():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)

    predicted_classes = [
        CLASSES[i] for i in predicted_indices.cpu().numpy()
    ]

    return predicted_classes, probabilities.cpu().numpy()

