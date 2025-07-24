import os

# Step 1: Record live (already saved as `recording.wav` and `transcript.txt`)
print("🎤 Step 1: Recording complete.")

# Step 2: Run diarization
print("🔍 Step 2: Running diarization...")
os.system("python diarization.py")

# Step 3: Merge
print("🔗 Step 3: Merging transcript...")
os.system("python merge_transcripts.py")

print("✅ Done. Final transcript at final_transcript.txt")
