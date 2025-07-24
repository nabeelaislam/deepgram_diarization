import pyaudio
import wave
import os

# ============ CONFIG ==============
SAMPLE_DIR = "samples"
DURATION_SECONDS = 5     # Recording time per speaker
CHANNELS = 1
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
# ==================================

os.makedirs(SAMPLE_DIR, exist_ok=True)

def record_sample(speaker_name):
    filename = f"{SAMPLE_DIR}/{speaker_name.lower()}.wav"

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print(f"\n🎙️ Recording for speaker: {speaker_name}")
    print(f"⏳ Speak clearly for {DURATION_SECONDS} seconds...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * DURATION_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ Done recording!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"📁 Saved as: {filename}\n")

if __name__ == "__main__":
    print("🔊 Known Speaker Voice Sample Recorder\n")
    while True:
        name = input("Enter speaker name (or press ENTER to quit): ").strip()
        if not name:
            break
        record_sample(name)
    print("👋 Done.")
