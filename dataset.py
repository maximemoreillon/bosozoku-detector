import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import os


def chunked_mels_from_file(
    filepath, chunk_seconds=3, sr=22050, n_mels=128, n_fft=2048, hop_length=512
):
    audio, _ = librosa.load(filepath, sr=sr)
    total_samples = len(audio)
    chunk_size = int(chunk_seconds * sr)
    mel_chunks = []

    for start in range(0, total_samples, chunk_size):
        end = start + chunk_size
        chunk = audio[start:end]

        if len(chunk) < chunk_size:
            continue
            # chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        mel = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        mel_chunks.append(mel_norm.astype(np.float32))

    return np.stack(mel_chunks)


class MelChunksDataset(Dataset):
    def __init__(self, root_dir, class_map):
        self.samples = []
        self.class_map = class_map

        for class_name, label in class_map.items():
            folder = os.path.join(root_dir, class_name)
            for file in os.listdir(folder):
                # if file.lower().endswith(".wav"):
                self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        mel_chunks = chunked_mels_from_file(filepath)

        # Random chunk pick for training diversity
        rnd = np.random.randint(0, mel_chunks.shape[0])
        mel = mel_chunks[rnd]  # shape: (n_mels, time_frames)

        mel = torch.tensor(mel).unsqueeze(0)  # add channel dim

        return mel, torch.tensor(label)
