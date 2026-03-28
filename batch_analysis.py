import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classifier import analyze_emotions

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DATASET_PATH = "/Users/nagymsemsedin/Desktop/dataset"

def run_batch_analysis():
    results = []

    image_files = [f for f in os.listdir(DATASET_PATH)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images, analyzing...\n")

    for filename in image_files:
        true_emotion = filename.split('_')[0].split('-')[0].lower()

        if true_emotion not in EMOTIONS:
            print(f"[skip] {filename} — unknown emotion '{true_emotion}'")
            continue

        image_path = os.path.join(DATASET_PATH, filename)
        faces = analyze_emotions(image_path)

        if not faces:
            print(f"[skip] {filename} — no face detected")
            continue

        predicted_emotion = faces[0]['dominant']
        correct = (predicted_emotion == true_emotion)

        results.append({
            'filename': filename,
            'true': true_emotion,
            'predicted': predicted_emotion,
            'correct': correct
        })

        status = "OK" if correct else "WRONG"
        print(f"[{status}] {filename}: true={true_emotion}, predicted={predicted_emotion}")

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print(f"\nSaved results.csv ({len(df)} rows)")

    print("\nAccuracy per emotion:")
    for emotion in EMOTIONS:
        subset = df[df['true'] == emotion]
        if len(subset) == 0:
            continue
        acc = subset['correct'].mean() * 100
        print(f"  {emotion:10s}: {acc:.0f}%  ({subset['correct'].sum()}/{len(subset)})")

    overall = df['correct'].mean() * 100
    print(f"\nOverall accuracy: {overall:.1f}%")

    cm = confusion_matrix(df['true'], df['predicted'], labels=EMOTIONS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)

    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix — Emotion Recognition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    run_batch_analysis()