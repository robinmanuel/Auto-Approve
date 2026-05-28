import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image
from ultralytics import YOLO


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


device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)


class DamageNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.base = models.resnet50(
            weights=None
        )

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


model = DamageNet(
    len(CLASSES)
).to(device)


checkpoint = torch.load(
    'best_damagenet_model.pth',
    map_location=device,
    weights_only=False
)


model.load_state_dict(
    checkpoint['model_state_dict']
)


model.eval()


# Load YOLO 11m model for ensemble
yolo_model = YOLO('trained.pt')
yolo_model.to(device)


transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_severity(confidence):

    if confidence < 0.50:
        return 'Minor', 1.0

    elif confidence < 0.70:
        return 'Moderate', 1.5

    else:
        return 'Severe', 2.2


def predict_with_yolo(image_file):
    """Get damage type and confidence from YOLO 11m model"""
    
    # Mapping YOLO classes to DamageNet classes
    yolo_to_damagenet = {
        'shattered_glass': 'broken_glass',
        'flat_tire': 'hole',
        'broken_lamp': 'broken_lamp',
        'dent': 'dent',
        'scratch': 'paint_scratch',
        'crack': 'torn'
    }
    
    results = yolo_model.predict(
        source=image_file,
        device=device,
        verbose=False
    )
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            # Get the most confident detection
            confidences = result.boxes.conf
            max_idx = confidences.argmax().item()
            max_conf = confidences[max_idx].item()
            
            # Get class name (damage type) from YOLO
            class_id = int(result.boxes.cls[max_idx].item())
            yolo_damage_type = result.names[class_id]
            
            # Map YOLO class to DamageNet class
            damage_type = yolo_to_damagenet.get(
                yolo_damage_type,
                'no_damage'
            )
            
            return damage_type, max_conf
    
    return 'no_damage', 0.0


def predict_damage_ensemble(image_file, use_yolo_confidence=False):
    """Ensemble prediction - run both models and use the one with higher confidence"""
    
    image = Image.open(
        image_file
    ).convert('RGB')

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    # Get DamageNet prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)
        confidence, predicted = torch.max(
            probs,
            1
        )

    damagenet_confidence = confidence.item()
    damagenet_damage_type = CLASSES[
        predicted.item()
    ]

    # Get YOLO prediction
    yolo_damage_type, yolo_confidence = predict_with_yolo(image_file)

    # Choose the model with higher confidence
    if yolo_confidence > damagenet_confidence:
        confidence_score = yolo_confidence
        damage_type = yolo_damage_type
        use_yolo_confidence = True
    else:
        confidence_score = damagenet_confidence
        damage_type = damagenet_damage_type
        use_yolo_confidence = False

    severity, multiplier = get_severity(
        confidence_score
    )

    estimated_cost = (
        repair_cost[damage_type]
        * multiplier
    )

    return {
        'Damage_Type': damage_type,
        'Confidence': round(
            confidence_score * 100,
            2
        ),
        'Severity': severity,
        'Estimated_Cost': round(
            estimated_cost,
            2
        ),
        'Model_Used': 'YOLO' if use_yolo_confidence else 'DamageNet'
    }


def predict_damage(image_file):
    """Main prediction function using ensemble approach"""
    return predict_damage_ensemble(image_file)