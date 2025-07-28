# DEEPGRAM_SPEECHTOTEXT

A simple pipeline for **real‑time microphone transcription** using Deepgram’s WebSocket API, performance logging (latency & CPU), and **post‑processing** to label each segment with speaker identities via SpeechBrain embeddings.

---

## Description

- **Live transcription**: Captures your microphone audio, streams it to Deepgram in real time, and writes out both plain‑text and JSONL transcripts, along with raw audio and performance metrics.
- **Speaker identification**: After recording, processes the transcript and audio to compute embeddings for each segment and assigns speaker labels based on pre‑enrolled reference embeddings.
- **Medical entity extraction**: Detects relevant medical notes (symptoms, medications, conditions, etc.) from the transcript using Open AI and exports them to structured JSON
This toolset is ideal for building demos, research prototypes, or simple logging systems that need quick transcription and speaker labeling without bulky infrastructure.

---

## How to Run

1. **Clone & install dependencies**
   ```bash
   git clone https://your.git.repo/DEEPGRAM_SPEECHTOTEXT.git
   cd DEEPGRAM_SPEECHTOTEXT
   pip install deepgram-sdk pyaudio psutil torch torchaudio speechbrain numpy

2. **Set your Deepgram API key**
    export DEEPGRAM_API_KEY="YOUR_DEEPGRAM_KEY"

3. **Run live transcription then dairization**
    python3 main.py
    python3 merge_transcripts.py

## Project Structure
embeddings/                # Pre‑computed speaker embeddings (pickle files)

pretrained_models/         # SpeechBrain model checkpoints

main.py                    # Live transcription & logging

generate_embeddings.py     # build embeddings from WAV samples

transcript.txt             # Plain text transcript

transcript.jsonl           # JSONL transcript from Deepgram

recording.wav              # Recorded mic audio

merged_transcript.txt      # Speaker‑labeled transcript

README.md                  # This file


DEEPGRAM_SPEECHTOTEXT/

embeddings/                # Pre‑computed speaker embeddings (pickle files)

	nabeela.pkl
	peter.pkl
	liz.pkl
 
pretrained_models/         # SpeechBrain model checkpoints

	spkrec/
	spkrec-ecapa-voxceleb/
 
samples/                   # Example audio files for embedding generation

	nabeela.wav
	peter.wav
	liz.wav
 
test_scripts/              # Additional test or helper scripts

venv/                      # Python virtual environment 

.gitignore

record_sample.py           # Record short mic sample for speaker enrollment

generate_embeddings.py     # Generate and save embeddings from WAV samples

main.py                    # Live transcription: mic → Deepgram → transcript.jsonl, recording.wav

merge_transcripts.py       # (Optional) Merge JSONL transcript into a human‑readable text

run_pipeline.py            # End‑to‑end: transcription + speaker ID → merged_transcript.txt

transcript.txt             # Plain‑text live transcript (logs)

transcript.jsonl           # JSONL output from Deepgram (one object per final segment)

recording.wav              # Raw microphone audio saved after transcription

merged_transcript.txt      # Final speaker‑labeled transcript

README.md                  # This file
