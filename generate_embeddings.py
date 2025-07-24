from speechbrain.pretrained import SpeakerRecognition
import torchaudio
import torch
import os
import pickle

# Load pretrained speaker embedding model from speechbrain
embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def extract_embedding(wav_path):
    signal, fs = torchaudio.load(wav_path)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
    embedding = embedding_model.encode_batch(signal)
    return embedding.squeeze().detach()

# Create output folder
os.makedirs("embeddings", exist_ok=True)

# Extract and save
for name in ["nabeela", "peter"]:
    path = f"samples/{name}.wav"
    emb = extract_embedding(path)
    with open(f"embeddings/{name}.pkl", "wb") as f:
        pickle.dump(emb, f)
    print(f"✅ Saved: embeddings/{name}.pkl")
