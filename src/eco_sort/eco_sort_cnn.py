import torch
from torch import nn
from torchvision import models, transforms


class EcoSortCNN:
    def __init__(
        self,
        model_path: str,
        class_names=[
            "background",
            "glass",
            "metal",
            "organic",
            "paper",
            "plastic",
            "styrofoam",
            "textiles",
        ],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier[3] = nn.Linear(
            in_features=self.model.classifier[3].in_features,
            out_features=len(class_names),
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.recyclable_class = ["glass", "metal", "plastic", "textiles"]
        self.organic_class = ["organic", "paper"]

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def predict(self, image_pil):
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            label = self.class_names[pred_idx] if confidence > 0.6 else "Tidak yakin"
        return label, confidence

    def decide_recyclability(self, label: str):
        return True if label in self.recyclable_class else False
