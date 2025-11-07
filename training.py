from torch.utils.data import DataLoader
import torch
from dataset import MelChunksDataset
from cnn import AudioCNN
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class_map = {"bosozoku": 0, "normal": 1}
dataset = MelChunksDataset("data", class_map)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AudioCNN(num_classes=len(class_map)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train():
    for epoch in range(100):
        total_loss = 0

        for mel, label in loader:
            mel = mel.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(mel)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch", epoch, "loss", total_loss / len(loader))

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)


if __name__ == "__main__":
    train()
