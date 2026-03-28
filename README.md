# Facial Emotion Recognition

A real-time facial emotion recognition system using DeepFace and OpenCV.

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — run webcam or batch mode |
| `face_detection.py` | Face detection using OpenCV Haar Cascade |
| `preprocessing.py` | Image preprocessing (resize, grayscale, CLAHE) |
| `webcam_mode.py` | Real-time webcam with FPS counter |
| `classifier.py` | Emotion classifier using DeepFace (7 emotions) |
| `visualizer.py` | Draws bounding boxes, labels, and bar charts |
| `batch_analysis.py` | Batch accuracy analysis and confusion matrix |

## Requirements

- Python 3.10+
- Webcam

## Installation
```bash
pip install deepface opencv-python matplotlib pandas scikit-learn tf-keras
```

## Usage

### Webcam mode (real-time)
```bash
python main.py --mode webcam
```
- Press `q` to quit
- Press `s` to save screenshot

### Batch analysis
```bash
python batch_analysis.py
```

Prepare a `dataset/` folder with images named by emotion:
```
dataset/
├── happy_1.jpg
├── sad_1.jpg
├── angry_1.jpg
├── fear_1.jpg
├── surprise_1.jpg
├── neutral_1.jpg
└── disgust_1.jpg
```

## Emotions Detected

`angry` `disgust` `fear` `happy` `neutral` `sad` `surprise`

## Output

- Real-time emotion label + confidence % on webcam feed
- Color-coded bounding boxes per emotion
- Mini bar chart per face
- `results.csv` — batch prediction results
- `confusion_matrix.png` — per-emotion accuracy visualization

## Partner Contributions

| Partner | Responsibilities |
|---------|-----------------|
| Partner A | Preprocessing, face detection, webcam mode |
| Partner B | Emotion classifier, visualization, batch analysis |