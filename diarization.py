import os
import torch
import librosa
import json
import soundfile as sf
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cosine

# =============== CONFIG =================
HF_TOKEN = os.getenv("HF_TOKEN")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=HUGGINGFACE_TOKEN
)
RECORDING = "samples/nabeela.wav"
SAMPLES = {
    "Nabeela": "samples/nabeela.wav",
    #"Alice": "samples/alice.wav"
}
SIM_THRESHOLD = 0.75  # Adjust if needed
# ========================================

# 1. Load pyannote diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
diarization = pipeline(RECORDING)

# 2. Speaker embedding model
embedder = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cpu"))
audio_loader = Audio(sample_rate=16000)

def get_embedding(file_path):
    waveform, _ = audio_loader(file_path)
    return embedder(waveform[None])[0]

# 3. Enroll known speakers
print("📦 Enrolling known speakers...")
known_embeddings = {name: get_embedding(path) for name, path in SAMPLES.items()}

# 4. Process each diarized segment
print("\n🎯 Identifying speakers...\n")
segment_idx = 0
segments = []  # <--- Move this here

for turn, _, speaker_label in diarization.itertracks(yield_label=True):
    start = turn.start
    end = turn.end
    duration = end - start
    y, sr = librosa.load(RECORDING, sr=16000, offset=start, duration=duration)
    segment_file = f"temp_segment_{segment_idx}.wav"
    sf.write(segment_file, y, sr)

    # Embed segment
    segment_embedding = get_embedding(segment_file)

    # Compare to known speakers
    best_match = "Unknown"
    best_score = 0
    for name, emb in known_embeddings.items():
        sim = 1 - cosine(segment_embedding, emb)
        if sim > best_score and sim >= SIM_THRESHOLD:
            best_match = name
            best_score = sim

    print(f"🗣️ {best_match} speaks from {start:.2f}s to {end:.2f}s")

    # Add this to your JSON segment list
    segments.append({
        "start": float(start),
        "end": float(end),
        "speaker": best_match
    })

    os.remove(segment_file)
    segment_idx += 1

# Save the segments AFTER the loop
with open("pyannote_segments.json", "w") as f:
    json.dump(segments, f, indent=2)
