import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from torchvision import models, transforms
from PIL import Image

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent


CLASSES = [
    'no_damage',
    'lost_parts',
    'torn',
    'dent',
    'paint_scratch',
    'hole',
    'broken_glass',
    'broken_lamp'
]

repair_cost = {
    'no_damage': 0,
    'lost_parts': 12000,
    'torn': 6000,
    'dent': 4000,
    'paint_scratch': 2500,
    'hole': 9000,
    'broken_glass': 15000,
    'broken_lamp': 7000
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- MODEL ----------------
class DamageNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.base = models.resnet50(weights=None)

        in_features = self.base.fc.in_features

        self.base.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base(x)


# ---------------- LOAD MODEL ----------------
model = DamageNet(len(CLASSES)).to(device)

checkpoint = torch.load(
    BASE_DIR / "best_damagenet_model.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- SEVERITY ----------------
def get_severity(confidence):
    if confidence < 0.60:
        return "Minor", 1.0
    elif confidence < 0.85:
        return "Moderate", 1.5
    else:
        return "Severe", 2.2


# ---------------- IMAGE CONVERSION ----------------
def preprocess_input(image):

    # Accept PIL or numpy or file path
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    else:
        raise ValueError("Unsupported image type")

    return image


# ---------------- PREDICT ----------------
def predict_damage(image):

    image = preprocess_input(image)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    damage_type = CLASSES[predicted.item()]
    confidence_score = confidence.item()

    severity, multiplier = get_severity(confidence_score)

    estimated_cost = repair_cost[damage_type] * multiplier

    return {
        "Damage_Type": damage_type,
        "Confidence": round(confidence_score * 100, 2),
        "Severity": severity,
        "Estimated_Cost": round(estimated_cost, 2)
    }