import os

# Step 1: Record live (already saved as `recording.wav` and `transcript.txt`)
print("🎤 Step 1: Recording complete.")

# Step 2: Run diarization
print("🔍 Step 2: Running diarization...")
os.system("python3 main.py")

# Step 3: Merge
print("🔗 Step 3: Merging transcript...")
os.system("python3 merge_transcripts.py")

print("✅ Done. Final transcript at merge_transcript.txt")
