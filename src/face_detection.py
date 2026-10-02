from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def bounding_box_finder(image_path, detector):
    boxes, probabilities = detector.detect(image)

    return boxes, probabilities


def visualize_boxes(image, boxes, probabilities):
    fig, ax = plt.subplots()
    ax.imshow(image)

    if boxes is not None:
        for box, probability in zip(boxes, probabilities):
            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1,
                f"{probability:.2f}",
                color="red",
                fontsize=10
            )

    ax.axis("off")
    plt.show()


import cv2
import threading
import time

cap = cv2.VideoCapture(0)

latest_frame = None
latest_result = None

frame_lock = threading.Lock()
result_lock = threading.Lock()

running = True


def detector_worker():
    global latest_frame, latest_result, running

    while running:
        # Get the newest frame
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.001)
            continue

        # Your detector here
        # This takes ~0.2 seconds
        faces = detect_faces(frame)

        # Store the newest result
        with result_lock:
            latest_result = faces


def detect_faces(frame):
    # Replace with your actual detector
    time.sleep(0.2)
    return []


thread = threading.Thread(target=detector_worker, daemon=True)
thread.start()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Give detector the newest frame
    with frame_lock:
        latest_frame = frame.copy()

    # Get the most recent detection result
    with result_lock:
        faces = latest_result

    # Draw the most recent results
    if faces is not None:
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
thread.join(timeout=1)

cap.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = MTCNN(
        keep_all=True,
        device="cpu"
    )
    image = Image.open("src/photo.jpg").convert("RGB")

    time1 = time.time()
    boxes, probabilities = bounding_box_finder(
        image,
        detector
    )
    print(time.time() - time1)

    visualize_boxes(image, boxes, probabilities)
