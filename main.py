import pyaudio
import logging
import threading
import time
import psutil
import json
import wave
from collections import deque
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

# --- Configuration ---
API_KEY = "fcaa3f9d8604f7b0d08660b7483f4643704c3f35"  # Replace with your real key or use env
TRANSCRIPT_FILE = "transcript.txt"
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Logging setup
logging.basicConfig(level=logging.INFO)

# Tracking performance
send_timestamps = deque(maxlen=100)
performance_samples = []
process = psutil.Process()
process.cpu_percent(interval=None)


def main():
    try:
        transcript_file = open(TRANSCRIPT_FILE, "w", encoding="utf-8")
        last_transcript = ""

        # ✅ Setup Deepgram connection
        deepgram = DeepgramClient(API_KEY)
        dg_connection = deepgram.listen.websocket.v("1")


        # ✅ Transcript event callback
        def on_message(self, result):
            nonlocal last_transcript
            
            if result is None:
                print("⚠️ No result received")
                return

            data = result.to_dict()
            print("🧪 Raw message received:", json.dumps(data, indent=2))

        
            if not data.get("is_final", False):
                return

            alt = data["channel"]["alternatives"][0]
            speaker = alt.get("speaker", "Unknown")
            sentence = alt.get("transcript", "")

            if not sentence or sentence == last_transcript:
                return

            last_transcript = sentence
            now = time.time()
            latency = now - send_timestamps[0] if send_timestamps else 0
            cpu = process.cpu_percent(interval=None)

            

            performance_samples.append({
                "latency": round(latency, 2),
                "cpu": round(cpu, 2),
            })

            print(f"\n🎙️ Speaker {speaker}: {sentence}")
            transcript_file.write(f"Speaker {speaker}: {sentence}\n")
            transcript_file.flush()

            # ✅ Save to JSON
            with open("transcript.json", "a", encoding="utf-8") as jf:
                json.dump({
                    "start": alt.get("start", 0),
                    "end": alt.get("end", 0),
                    "speaker": speaker,
                    "text": sentence
                }, jf)
                jf.write("\n")

        # ✅ Register handler
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # ✅ Live transcription options
        options = LiveOptions(
            model="nova-3",
            punctuate=True,
            language="en",
            encoding="linear16",
            channels=1,
            sample_rate=RATE,
            diarize=True,
        )

        if not dg_connection.start(options):
            print("❌ Failed to start Deepgram websocket")
            return

        print("\n🎤 Speak now. Press Ctrl+C to stop...\n")

        audio = pyaudio.PyAudio()
        audio_frames = []

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        stop_flag = False

        def mic_loop():
            while not stop_flag:
                data = stream.read(CHUNK, exception_on_overflow=False)
                dg_connection.send(data)
                audio_frames.append(data)
                send_timestamps.append(time.time())

        thread = threading.Thread(target=mic_loop)
        thread.start()

        try:
            while thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")

        stop_flag = True
        thread.join()
        dg_connection.finish()

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # ✅ Save metrics
        if performance_samples:
            avg_latency = sum(p["latency"] for p in performance_samples) / len(performance_samples)
            avg_cpu = sum(p["cpu"] for p in performance_samples) / len(performance_samples)
            transcript_file.write(json.dumps({"latency": avg_latency, "cpu": avg_cpu}) + "\n")

        transcript_file.close()
        print("✅ Transcription and metrics saved.")

        # ✅ Save audio
        with wave.open("recording.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(audio_frames))
        print("📁 Audio saved to recording.wav")

    except Exception as e:
        print(f"❗Error: {e}")


if __name__ == "__main__":
    main()
