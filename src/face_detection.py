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

from classifier import classify_face


COLOR = (60, 60, 220)

def show_image(path, detector, device):
    image = Image.open(path).convert("RGB")

    start = time.time()
    boxes, probs = detector.detect(image)
    print(time.time() - start)

    fig, ax = plt.subplots()
    ax.imshow(image)
    if boxes is not None:
        for (x1, y1, x2, y2), p in zip(boxes, probs):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=2, edgecolor="red", facecolor="none"))
            ax.text(x1, y1, f"{p:.2f}", color="red", fontsize=10)
    ax.axis("off")
    plt.show()



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
                #Ganska ful lösing för att fixa när ett ansikte är i kanten av bilden
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
    thread.start() #Jag ska vara helt ärlig och säga att jag inte riktigt vet
                   #hur threading fungerar

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(keep_all=True, device=device)
    #show_image("src/photo.jpg", detector, device)
    weights = ResNet50_Weights.IMAGENET1K_V2

    #weights = ViT_B_16_Weights.DEFAULT
    model = resnet50(weights=weights)
    preprocess = weights.transforms()

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, 8)
    )

    checkpoint = torch.load(
        "checkpoints/best_resnet.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()



    run_webcam(detector, model, preprocess, device)
