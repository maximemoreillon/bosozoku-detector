from cnn import AudioCNN
from config import class_map
import torch

model = AudioCNN(num_classes=len(class_map))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
model.cpu()


# TODO: get those from config
dummy_input = torch.randn(1, 1, 128, 130)

torch.onnx.export(
    model,  # The PyTorch model
    dummy_input,  # A tuple or single tensor representing the model's input
    "model/model.onnx",  # Path to save the ONNX model
    input_names=["mel"],  # Names for the model's input(s)
    output_names=["logits"],  # Names for the model's output(s)
)
