from cnn import AudioCNN
from training import class_map
import torch

# from dataset import chunked_mels_from_file
import librosa
import numpy as np


model = AudioCNN(num_classes=len(class_map))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


def audio_to_mel_tensor(
    filepath, chunk_seconds=3, sr=22050, n_mels=128, n_fft=2048, hop_length=512
):
    audio, _ = librosa.load(filepath, sr=sr)
    chunk_size = int(chunk_seconds * sr)

    # Take the first chunk (or pad if short)
    audio = audio[:chunk_size]
    if len(audio) < chunk_size:
        raise "Too short"
        # audio = np.pad(audio, (0, chunk_size - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    # shape: (1, 1, n_mels, time_frames)
    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor


def infer(filepath, model, class_map):
    x = audio_to_mel_tensor(filepath)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    idx_to_class = {v: k for k, v in class_map.items()}
    predicted_label = idx_to_class[pred_idx]
    confidence = probs[0, pred_idx].item()

    print("Prediction:", predicted_label, "confidence:", confidence)


infer("./data/bosozoku/1.mp3", model, class_map)
