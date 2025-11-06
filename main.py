import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def process_file(filepath):

    y, sr = librosa.load(filepath, sr=None)

    chunk_size = int(3.0 * sr)

    chunks = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i : i + chunk_size]

        # Skip last chunk if it's shorter than desired
        if len(chunk) < chunk_size:
            continue
        # Create Mel spectrogram

        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr // 2)

        # Convert to decibels for visibility
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Gen NP arrays
        # Optional normalization
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        npArray = mel_db.astype(np.float32)

        # TODO: have as NP array instead of a list
        chunks.append(npArray)

        # Gen images
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None)
        plt.axis("off")
        plt.savefig(
            f"./data/images/bosozoku/{Path(filepath).stem}-{i}",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    return chunks


if __name__ == "__main__":

    dataFolderPath = Path("./data/sound")
    classFolderPaths = [item for item in dataFolderPath.iterdir() if item.is_dir()]
    for classFolderPath in classFolderPaths:
        files = [item for item in classFolderPath.iterdir() if item.is_file()]
        for file in files:
            process_file(file.absolute())
