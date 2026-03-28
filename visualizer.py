import cv2
import numpy as np
import matplotlib.pyplot as plt

EMOTION_COLORS = {
    'angry':    (0, 0, 255),
    'disgust':  (0, 140, 255),
    'fear':     (128, 0, 128),
    'happy':    (0, 255, 0),
    'sad':      (255, 0, 0),
    'surprise': (0, 255, 255),
    'neutral':  (200, 200, 200),
}

def draw_results(image, faces: list) -> np.ndarray:

    img = image.copy()
    h_img, w_img = img.shape[:2]

    for face in faces:
        dominant = face['dominant']
        scores = face['scores']
        region = face['region']

        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        color = EMOTION_COLORS.get(dominant, (255, 255, 255))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        confidence = float(scores[dominant])
        label = f"{dominant}: {confidence:.1f}%"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        bar_x = x + w + 10
        bar_y = y
        bar_w = 80
        bar_h = 10
        gap = 4

        for i, (emotion, value) in enumerate(scores.items()):
            ec = EMOTION_COLORS.get(emotion, (200, 200, 200))
            by = bar_y + i * (bar_h + gap)

            if by + bar_h > h_img or bar_x + bar_w + 45 > w_img:
                continue

            cv2.rectangle(img, (bar_x + 40, by),
                          (bar_x + 40 + bar_w, by + bar_h), (50, 50, 50), -1)
            filled = int((float(value) / 100.0) * bar_w)
            if filled > 0:
                cv2.rectangle(img, (bar_x + 40, by),
                              (bar_x + 40 + filled, by + bar_h), ec, -1)
            cv2.putText(img, emotion[:3], (bar_x, by + bar_h - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, ec, 1)

    return img


def show_matplotlib_summary(image, faces: list, save_path: str = None):

    annotated = draw_results(image, faces)

    fig, axes = plt.subplots(1, len(faces) + 1,
                             figsize=(6 * (len(faces) + 1), 5))

    if len(faces) + 1 == 1:
        axes = [axes]
    elif not hasattr(axes, '__len__'):
        axes = [axes]

    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Detected Faces")
    axes[0].axis('off')

    for i, face in enumerate(faces):
        scores = face['scores']
        emotions = list(scores.keys())
        values = [float(v) for v in scores.values()]
        colors = [tuple(c / 255 for c in EMOTION_COLORS[e][::-1]) for e in emotions]

        ax = axes[i + 1]
        bars = ax.barh(emotions, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_title(f"Face {i+1}: {face['dominant']}")
        ax.set_xlabel("Confidence %")

        for bar, val in zip(bars, values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[visualizer] Saved: {save_path}")

    plt.show()