import json
from datetime import timedelta
import pickle
import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load known speaker embeddings
with open("embeddings/nabeela.pkl", "rb") as f:
    nabeela_emb = pickle.load(f)
with open("embeddings/peter.pkl", "rb") as f:
    peter_emb = pickle.load(f)
with open("embeddings/liz.pkl", "rb") as f:
    liz_emb = pickle.load(f)


# Convert known embeddings to torch tensors once
known_speakers = {
    "Nabeela": nabeela_emb.detach().clone().squeeze(),
    "Peter": peter_emb.detach().clone().squeeze(),
    "Liz": liz_emb.detach().clone().squeeze()

}


# Load Deepgram transcript
with open("transcript.jsonl", "r") as f:
    deepgram_segments = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            deepgram_segments.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping bad line: {line[:30]}... Error: {e}")

deepgram_segments.sort(key=lambda x: x["start"])

# Load full audio
waveform, sr = torchaudio.load("recording.wav")

# Resample if needed
if sr != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

# Load speaker recognition model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Identify speaker per segment
for segment in deepgram_segments:
    start_sample = int(segment["start"] * 16000)
    end_sample = int(segment["end"] * 16000)
    segment_audio = waveform[:, start_sample:end_sample]

    duration = segment["end"] - segment["start"]
    if duration < 0.5:
        print(f"⏱ Skipping segment {segment['start']:.2f}-{segment['end']:.2f} (too short: {duration:.2f}s)")
        segment["speaker"] = "Unknown"
        continue

    print(f"🔊 Analyzing segment {segment['start']:.2f}-{segment['end']:.2f} | Duration: {duration:.2f}s")

    # Get embedding of segment
    try:
       segment_emb = verification.encode_batch(segment_audio, normalize=True).squeeze()
       # Ensure audio is long enough for model input
       if segment_audio.shape[1] < 320:  # empirically ~0.02 sec of audio
        print(f"⚠️ Segment too short for embedding (samples: {segment_audio.shape[1]})")
        segment["speaker"] = "Unknown"
        continue


    except Exception as e:
        print(f"⚠️ Error embedding segment {segment['start']}-{segment['end']}: {e}")
        segment["speaker"] = "Unknown"
        continue

    scores = {}
    for name, ref_emb in known_speakers.items():
        try:
            
            score = torch.nn.functional.cosine_similarity(
                segment_emb,  # shape: [1, 192]
                ref_emb,                # shape: [1, 192]
                dim=0
            ).item()

            scores[name] = score

        except Exception as e:
            print(f"⚠️ Error comparing with {name}: {e}")
            scores[name] = -1.0

    print(f"📊 Scores: {scores}")

    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]
    segment["speaker"] = best_match if best_score > 0.1 else "Unknown"

    extraction_prompt = f"""
    From the following transcript, extract relevant medical notes and return a valid JSON array of 
    dictionaries.

    Examples of relevant medical info include patient being dehydrated, eating/drinking, blood pressure  

    Each dictionary must include:
    - person: "{segment['speaker']}"
    - note: short medical description
    - time: null
    - type: one of ["symptom", "condition", "medication", "activity", "quantity", "other"]

    Ignore irrelevant or non-medical content, this includes social activities.
    If a field is missing just have the field as null. 

    Transcript:
    {segment['text']}

    Return only the JSON array. Do not include markdown formatting or explanation.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant that extracts structured medical notes in JSON format."},
            {"role": "user", "content": extraction_prompt}
        ]
    )

    content = response.choices[0].message.content.strip()

    # Clean markdown-style wrapping if needed
    if content.startswith("```"):
        import re
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        entities = json.loads(content)
        segment["entities"] = entities
        print(f"✅ Extracted {len(entities)} medical notes.")
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse GPT output as JSON: {e}")
        segment["entities"] = []

# Write merged transcript
with open("merged_transcript.txt", "w") as out:
    for segment in deepgram_segments:
        start_time = str(timedelta(seconds=segment["start"]))[:11]
        end_time = str(timedelta(seconds=segment["end"]))[:11]
        speaker = segment["speaker"]
        text = segment["text"]
        out.write(f"[{start_time} - {end_time}] {speaker}: {text}\n")

print("✅ Merged transcript saved to merged_transcript.txt")

# After all segments processed
all_entities = []
for segment in deepgram_segments:
    all_entities.extend(segment.get("entities", []))

with open("extracted_medical_notes.json", "w") as f:
    json.dump(all_entities, f, indent=2)

print("📝 Medical notes saved to extracted_medical_notes.json")
