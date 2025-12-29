# stt_mm

A Speech-to-Text (STT) tool using OpenAI's Whisper model for Myanmar language transcription.

## Features
- Transcribes Myanmar audio to text using Whisper
- Python-based, easy to use
- Web-based interface powered by Gradio

## Requirements
- Python 3.10+
- Dependencies:
  - sounddevice
  - numpy
  - gradio
  - torch
  - transformers

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd stt_mm
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv stt_mm_env
   source stt_mm_env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```bash
python test_stt_mm.py
```

This will launch a web interface where you can record audio in Myanmar and get the transcription.

## Model
This project uses the Whisper model from Hugging Face. The specific model used is `chuuhtetnaing/whisper-small-myanmar`, which is optimized for Myanmar language transcription.

## License
MIT License