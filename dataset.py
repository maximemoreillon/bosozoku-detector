import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import os
from config import sr, chunk_duration, n_mels, n_fft, hop_length


def mel_from_audio(audio):
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
    return mel_norm


def chunked_mels_from_file(filepath):
    audio, _ = librosa.load(filepath, sr=sr)
    total_samples = len(audio)
    chunk_size = int(chunk_duration * sr)
    mel_chunks = []

    for start in range(0, total_samples, chunk_size):
        end = start + chunk_size
        chunk = audio[start:end]

        if len(chunk) < chunk_size:
            continue
            # chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        mel_norm = mel_from_audio(chunk)

        mel_chunks.append(mel_norm.astype(np.float32))

    return np.stack(mel_chunks)


class MelChunksDataset(Dataset):
    def __init__(self, root_dir, class_map):
        self.samples = []
        self.class_map = class_map

        for class_name, label in class_map.items():
            folder = os.path.join(root_dir, class_name)
            for file in os.listdir(folder):
                self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Is it really wise to process files each time?
        mel_chunks = chunked_mels_from_file(filepath)

        # Random chunk pick for training diversity
        rnd = np.random.randint(0, mel_chunks.shape[0])
        mel = mel_chunks[rnd]

        mel = torch.tensor(mel).unsqueeze(0)

        return mel, torch.tensor(label)
