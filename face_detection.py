import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """
    Detect faces in an image using Haar Cascade.

    Args:
        image: input BGR image
        scaleFactor: pyramid scale
        minNeighbors: detection strictness
        minSize: minimum face size

    Returns:
        faces: list of bounding boxes (x, y, w, h)
    """
    if image is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )

    return faces