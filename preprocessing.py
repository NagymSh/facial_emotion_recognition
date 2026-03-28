import cv2

def preprocess_face(face_img, target_size=(48, 48)):
    """
    Preprocess face image for emotion recognition:
    1) resize
    2) grayscale
    3) CLAHE contrast enhancement

    Args:
        face_img: input face image (BGR)
        target_size: tuple, desired output size

    Returns:
        processed_img: preprocessed grayscale image
    """
    if face_img is None or face_img.size == 0:
        raise ValueError("Empty face image provided to preprocess_face")

    resized = cv2.resize(face_img, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced