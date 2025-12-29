# stt_eng

A Speech-to-Text (STT) tool using OpenAI's Whisper model for English language.

## Features
- Transcribes English audio to text using Whisper
- Python-based, easy to use
- Web-based interface powered by Gradio

## Requirements
- Python 3.10+
- Dependencies:
  - sounddevice
  - numpy
  - gradio
  - openai-whisper

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd stt_eng
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv stt_eng_env
   source stt_eng_env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python stt_with_whisper.py
```

## License
MIT License