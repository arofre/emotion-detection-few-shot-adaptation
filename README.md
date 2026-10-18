# Real-Time Facial Emotion Detection with Few-Shot Class-Incremental Learning

Project within **SSY340 Deep Machine Learning** ·

## Why?
Standard FER models need large amounts of labeled datasets and a fixed set of classes. In reality, new classes can appear and labeled data is expensive. Naively fine-tuning on a new class causes catastrophic forgetting. This project compares two ways of adding a class:

| Method | Uses old data? | Gradient steps in the new session? |
|---|---|---|
| **Cosine classifier + prototype imprinting** | No | None |
| **Linear head + exemplar replay** | Yes | Yes (head only) |

## Method
ResNet50 (ImageNet weights) is trained on 7 of the 8 FER+ emotions using KL-divergence loss in multiple stages of depth. 

**Incremental with cosine classifier**. The backbone is frozen.
1. Features are centred on the base-class mean and L2-normalised.
2. Each base class row is rebuilt as its class prototype.
3. The k support images of the new class are averaged into a prototype and inserted as a new row.
4. Prediction = highest cosine similarity.

**Replay baseline.** Same prototype initialisation, then the linear head is fine-tuned on the
k support images plus N replayed images per old class.

##Respository structure
```
.
├── src/
│   ├── data_loader.py              # FerDataset: soft labels, k-shot sampling, class exclusion
│   ├── preprocessing.py            # image transforms (resize, augmentation, normalisation)
│   ├── model_loader.py             # ResNet / ViT / EfficientNet with replaceable heads
│   ├── evaluate.py                 # evaluate_model, confusion matrices, classification report
│   ├── training_general.py         # original linear-head pipeline
│   ├── training_few_shot_cosine.py # cosine head + imprinting
│   ├── training_few_shot_replay.py # linear head + replay grid
│   ├── classifier.py               # inference on face crops
│   ├── face_detection.py           # live webcam pipeline
│   └── metrics.py                  # catastrophic forgetting metric, new class recall
├── data/                           # FER2013 images and FER+ labels
├── checkpoints/                    # saved checkpoints from training
└── results/                        # confusion matrices and CSV reports
```

## Setup and usage
```bash
uv sync
uv run src/training_few_shot_cosine.py
uv run src/training_few_shot_replay.py
uv run src/face_detection.py
```
