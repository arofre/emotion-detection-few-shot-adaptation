import threading
import time

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from collections import Counter, deque
from torchvision.models import ResNet50_Weights, resnet18, ResNet18_Weights, resnet50
from torch import nn
import numpy as np

from classifier import classify_face
from preprocessing import build_image_transform
from model_loader import get_model


COLOR = (60, 60, 220)


def show_image(path, detector, model, preprocess, device):
    image = Image.open(path).convert("RGB")
    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    h, w = frame.shape[:2]
    face_images = []
    valid_boxes = []

    start_time = time.time()

    detected_boxes, probabilities = detector.detect(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    for box in detected_boxes if detected_boxes is not None else []:
        x1, y1, x2, y2 = box

        x1 = int(max(0, min(x1, w - 1)))
        x2 = int(max(0, min(x2, w)))
        y1 = int(max(0, min(y1, h - 1)))
        y2 = int(max(0, min(y2, h)))

        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_images.append(face)
        valid_boxes.append((x1, y1, x2, y2))

    classified_boxes = []

    if face_images:
        emotions, emotion_probabilities = classify_face(
            face_images, model, preprocess, device
        )

        classified_boxes = [
            (*box, emotion, probability)
            for box, emotion, probability in zip(
                valid_boxes,
                emotions,
                emotion_probabilities.max(axis=1)
            )
        ]

    elapsed_time = time.time() - start_time

    for x1, y1, x2, y2, emotion, probability in classified_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)

        label = f"{emotion} {probability:.3f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            COLOR,
            2
        )

    cv2.putText(
        frame,
        f"{elapsed_time * 1000:.0f}ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR,
        2
    )

    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def run_webcam(detector, model, preprocess, device):
    smoothing_factor = 0.5 #mellan 0 och 1
    smooth = []   
    history = [] 

    cap = cv2.VideoCapture(0)
    lock = threading.Lock()
    state = {"frame": None, "boxes": None, "running": True, "latest_time": 0.0}

    def worker():
        while state["running"]:
            
            with lock:
                frame = state["frame"]
            if frame is None:
                time.sleep(0.01)
                continue

            face_images = []
            valid_boxes = []

            h, w = frame.shape[:2]


            classified_boxes = []
            start_time = time.time()
            detected_boxes, probabilities = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
            for box in detected_boxes if detected_boxes is not None else []: 
                x1, y1, x2, y2 = box

                x1 = int(max(0, min(x1, w - 1)))
                x2 = int(max(0, min(x2, w)))
                y1 = int(max(0, min(y1, h - 1)))
                y2 = int(max(0, min(y2, h)))

                if x2 <= x1 or y2 <= y1:
                    continue

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_images.append(face)
                valid_boxes.append((x1, y1, x2, y2))


            if face_images:
                emotions, emotion_probabilities = classify_face(face_images, model, preprocess, device)

                classified_boxes = [
                    (*box, emotion, probability)
                    for box, emotion, probability in zip(
                        valid_boxes,
                        emotions,
                        emotion_probabilities.max(axis=1)
                    )
                ]
                
            latest_time = time.time() - start_time
            with lock:
                state["latest_time"] = latest_time
                state["boxes"] = classified_boxes

    thread = threading.Thread(target=worker, daemon=True)
    thread.start() 

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        with lock:
            state["frame"] = frame.copy()
            boxes = state["boxes"]
            latest_time = state["latest_time"]

        if boxes is not None:
            if len(boxes) != len(smooth):
                smooth = [list(b[:4]) for b in boxes]
                history = [deque(maxlen=30) for _ in boxes]

            for i, (x1, y1, x2, y2, emotion, probability) in enumerate(boxes):
                smooth[i] = [s + smoothing_factor * (n - s) for s, n in zip(smooth[i], (x1, y1, x2, y2))]
                history[i].append(emotion)

                label = f'{Counter(history[i]).most_common(1)[0][0]} {probability:.3f}'
                sx1, sy1, sx2, sy2 = map(int, smooth[i])
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), COLOR, 2)
                cv2.putText(frame, label, (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR, 2)
                
        cv2.putText(frame, f'{latest_time * 1000:.0f}ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    state["running"] = False
    thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(keep_all=True, device=device)

    model, model_transform = get_model("resnet50", 8, load_checkpoint=True, device=device)

    model.eval()
    transform = build_image_transform(size=112, channels=3, augment=False, mean=model_transform.mean, std=model_transform.std)
    show_image("src/photo.jpg", detector, model, transform, device)
    run_webcam(detector, model, transform, device)
