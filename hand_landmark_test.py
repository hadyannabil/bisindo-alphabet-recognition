
from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np


MODEL_PATH = Path("models/hand_landmarker.task")

# Koneksi antara 21 landmark tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)


def process_frame(frame):
    if frame is None:
        return None

    # Gradio menghasilkan gambar RGB
    frame = np.asarray(frame, dtype=np.uint8)

    # Batasi resolusi untuk mengurangi beban proses
    height, width = frame.shape[:2]

    if width > 640:
        new_height = int(height * 640 / width)
        frame = cv2.resize(frame, (640, new_height))

    frame = np.ascontiguousarray(frame)
    output = frame.copy()

    # Konversi ke format MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    # Deteksi landmark tangan
    result = landmarker.detect(mp_image)

    h, w = output.shape[:2]

    for hand in result.hand_landmarks:
        points = []

        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))

        # Gambar garis skeleton
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                output,
                points[start],
                points[end],
                (50, 220, 100),
                2
            )

        # Gambar titik landmark
        for x, y in points:
            cv2.circle(
                output,
                (x, y),
                4,
                (255, 60, 60),
                -1
            )

    # Tampilkan status deteksi
    hand_count = len(result.hand_landmarks)

    cv2.putText(
        output,
        f"Hands detected: {hand_count}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2
    )

    return output


with gr.Blocks(title="BISINDO Hand Tracking") as demo:
    gr.Markdown("# BISINDO Hand Landmark Detection")
    gr.Markdown(
        "Tahap 3 - Deteksi landmark tangan dengan MediaPipe"
    )

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="Input Webcam"
        )

        output = gr.Image(
            type="numpy",
            label="MediaPipe Hand Landmarks"
        )

    webcam.stream(
        fn=process_frame,
        inputs=webcam,
        outputs=output,
        stream_every=0.15,
        time_limit=120,
        show_progress="hidden",
        concurrency_limit=1
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
