# stt_with_whisper

A collection of Speech-to-Text (STT) tools using OpenAI's Whisper model. This repository contains two main projects:
- `stt_eng`: Focused on English language transcription.
- `stt_mm`: Focused on Myanmar language transcription.

## Features
- Transcribes audio to text using Whisper
- Python-based, easy to use
- Separate modules for English and Myanmar support

## Requirements
- Python 3.10+
- See `stt_eng/requirements.txt` and `stt_mm/requirements.txt` for dependencies specific to each module.

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd stt_with_whisper
   ```
2. Navigate to the desired module (`stt_eng` or `stt_mm`):
   ```bash
   cd stt_eng  # or cd stt_mm
   ```
3. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv env_name
   source env_name/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- For English transcription, navigate to `stt_eng` and run:
  ```bash
  python stt_with_whisper.py
  ```
- For multilingual transcription, navigate to `stt_mm` and run:
  ```bash
  python test_stt_mm.py
  ```

## License
MIT License

---

*Edit this README to add more details about usage or configuration as needed.*