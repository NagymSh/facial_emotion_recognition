import cv2
import os
import time
from face_detection import detect_faces
from preprocessing import preprocess_face
from classifier import analyze_emotions
from visualizer import draw_results

def run_webcam(frame_skip=3, save_dir="screenshots"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    frame_count = 0
    prev_time = time.time()
    fps = 0.0
    last_faces_raw = []
    last_emotion_faces = []

    print("Press 'q' to quit")
    print("Press 's' to save screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            try:
                last_faces_raw = detect_faces(frame, scaleFactor=1.1, minNeighbors=5)
            except Exception as e:
                print("Detection error:", e)
                last_faces_raw = []

            try:
                last_emotion_faces = analyze_emotions(frame)
            except Exception as e:
                print("Emotion error:", e)
                last_emotion_faces = []

        if last_emotion_faces:
            frame = draw_results(frame, last_emotion_faces)
        else:
            for i, (x, y, w, h) in enumerate(last_faces_raw):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i+1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Facial Emotion Recognition - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            filename = os.path.join(save_dir, f"screenshot_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
