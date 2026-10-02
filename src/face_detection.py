import threading
import time

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from collections import Counter, deque

from classifier import classify_face


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


def run_webcam(detector, device):
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

            boxes = []
            start_time = time.time()
            detected_boxes, _ = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
            for box in detected_boxes if detected_boxes is not None else []: 
                #Här kan vi överväga att köra batch
                x1, y1, x2, y2 = box
                face_image = frame[int(y1):int(y2), int(x1):int(x2)]
                emotion = classify_face(face_image, device)
                boxes.append((x1, y1, x2, y2, emotion))

                
            latest_time = time.time() - start_time
            with lock:
                state["latest_time"] = latest_time
                state["boxes"] = boxes

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

            for i, (x1, y1, x2, y2, emotion) in enumerate(boxes):
                smooth[i] = [s + smoothing_factor * (n - s) for s, n in zip(smooth[i], (x1, y1, x2, y2))]
                history[i].append(emotion)

                label = Counter(history[i]).most_common(1)[0][0]
                sx1, sy1, sx2, sy2 = map(int, smooth[i])
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                cv2.putText(frame, label, (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        cv2.putText(frame, f'{latest_time:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    # show_image("src/photo.jpg", detector)
    run_webcam(detector, device)