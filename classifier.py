from deepface import DeepFace


def analyze_emotions(image_path: str) -> dict:

    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        if not isinstance(results, list):
            results = [results]

        output = []
        for face in results:
            output.append({
                'dominant': face['dominant_emotion'],
                'scores': face['emotion'],
                'region': face['region']
            })

        return output

    except Exception as e:
        print(f"[classifier] Ошибка: {e}")
        return []