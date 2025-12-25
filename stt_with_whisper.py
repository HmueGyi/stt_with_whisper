import gradio as gr
import whisper

# Load the base model on CPU
model = whisper.load_model("base", device="cpu")

def transcribe_live(audio):
    if audio is None:
        return ""
    # Force English transcription
    result = model.transcribe(audio, language="en", beam_size=5, temperature=0)
    return result["text"]

def main():
    # Gradio interface
    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Now")
    output_text = gr.Textbox(label="Transcription")

    iface = gr.Interface(
        fn=transcribe_live,
        inputs=audio_input,
        outputs=output_text,
        title="Live Speech-to-Text",
        description="Speak into your microphone and get English-only transcription (base model, CPU)."
    )
    
    iface.launch()

if __name__ == "__main__":
    main()
