from cnn import AudioCNN
from config import class_map
import torch

from dataset import chunked_mels_from_file
import numpy as np


model = AudioCNN(num_classes=len(class_map))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


def infer(filepath, model, class_map):

    chunks = chunked_mels_from_file(filepath)

    print(f"file {filepath} has {len(chunks)} chunks")
    for chunk in chunks:
        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)  # Prediction here
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        idx_to_class = {v: k for k, v in class_map.items()}
        predicted_label = idx_to_class[pred_idx]
        confidence = probs[0, pred_idx].item()

        print("Prediction:", predicted_label, "confidence:", confidence)


infer("./data/normal/road.mp3", model, class_map)
infer("./data/bosozoku/1.mp3", model, class_map)
