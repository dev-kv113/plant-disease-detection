import torch
from torchvision import transforms
from PIL import Image
import sys
import os

checkpoint = torch.load('saved_model/plant_disease_model.pth', map_location='cpu')
class_names = checkpoint['class_names']

# Same architecture as training
from torch import nn
IMG_SIZE = 128
num_classes = len(class_names)
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 128), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

if len(sys.argv) < 2:
    print("Usage: python predict_pytorch.py <image_path>")
    sys.exit()

img_path = sys.argv[1]
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    pred_idx = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

print(f"✅ Predicted: {class_names[pred_idx]} (confidence {confidence:.2f})")
