import sounddevice as sd
import numpy as np
import torch
import librosa
import time
from cnn import AudioCNN
from training import class_map

model = AudioCNN(num_classes=len(class_map))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def mel_from_audio(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    eps = 1e-6
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + eps)
    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


# --------------------------
# Real-time microphone stream
# --------------------------
sr = 22050
chunk_seconds = 3
buffer_size = sr * chunk_seconds

audio_buffer = np.zeros(buffer_size, dtype=np.float32)

idx_to_class = {v: k for k, v in class_map.items()}


def audio_callback(indata, frames, time_, status):
    global audio_buffer

    if status:
        print("Status:", status)

    # Flatten microphone frames
    samples = indata[:, 0]

    # Roll buffer and insert new samples
    audio_buffer = np.roll(audio_buffer, -len(samples))
    audio_buffer[-len(samples) :] = samples


# Start non-blocking microphone stream
stream = sd.InputStream(
    samplerate=sr,
    channels=1,
    callback=audio_callback,
    blocksize=1024,
    # device=3,  # for example
)

print("Listening... Press Ctrl+C to stop.")

try:
    with stream:
        while True:
            # Process latest 3-second window
            audio = audio_buffer.copy()
            mel_tensor = mel_from_audio(audio)

            with torch.no_grad():
                logits = model(mel_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()

            predicted_label = idx_to_class[pred_idx]
            confidence = probs[pred_idx].item()

            print(f"{predicted_label}  ({confidence:.2f})")

            time.sleep(0.3)  # small pause to keep things breathable

except KeyboardInterrupt:
    print("Stopped.")
