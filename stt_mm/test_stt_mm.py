import gradio as gr
from transformers import pipeline

# Load Whisper Myanmar model on CPU
pipe = pipeline(
    "automatic-speech-recognition",
    model="chuuhtetnaing/whisper-small-myanmar",
    device=-1  # CPU
)

def transcribe_live(audio_path):
    if audio_path is None:
        return ""

    result = pipe(
        audio_path,
        generate_kwargs={
            "task": "transcribe",   # speech → text (same language)
            "language": "my",       # FORCE MYANMAR
            "num_beams": 5
        }
    )
    return result["text"]

def main():
    iface = gr.Interface(
        fn=transcribe_live,
        inputs=gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="မြန်မာလို ပြောပါ"
        ),
        outputs=gr.Textbox(label="Myanmar Transcription"),
        title="Myanmar Speech-to-Text",
        description="Whisper Myanmar-only transcription (CPU)"
    )

    iface.launch()

if __name__ == "__main__":
    main()
