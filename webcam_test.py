
import cv2
import gradio as gr


def process_frame(frame):
    if frame is None:
        return None

    # Salin frame agar input asli tidak berubah
    result = frame.copy()

    # Tambahkan teks sebagai bukti frame diproses Python
    cv2.putText(
        result,
        "BISINDO - Webcam Connected",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return result


with gr.Blocks(title="BISINDO Webcam Test") as demo:
    gr.Markdown("# BISINDO Alphabet Recognition")
    gr.Markdown("Tahap 2 - Pengujian webcam melalui Codespaces")

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="Input Webcam"
        )

        output = gr.Image(
            type="numpy",
            label="Processed Frame"
        )

        webcam.stream(
            fn=process_frame,
            inputs=webcam,
            outputs=output,
            stream_every=0.15,
            time_limit=120,
            show_progress="hidden"
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
