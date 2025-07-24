from deepgram import DeepgramClient

deepgram = DeepgramClient("test_key")
print(dir(deepgram.listen))
