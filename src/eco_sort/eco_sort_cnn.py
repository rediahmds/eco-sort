import torch
from torch import nn
from torchvision import models, transforms


class EcoSortCNN:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        class_names: list = [
            "background",
            "glass",
            "metal",
            "organic",
            "paper",
            "plastic",
            "textiles",
        ],
    ):
        """
        Inisialisasi object untuk prediksi menggunakan model yang sudah dilatih.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        self.model = self._load_deployment_model(
            model_path=model_path,
            model_name=model_name,
            num_classes=len(self.class_names),
        )

        self.recyclable_class = ["glass", "metal", "plastic", "textiles"]
        self.organic_class = ["organic", "paper"]

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # Ukuran standar untuk fine-tuning
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_pil, confidence_threshold=0.6):
        """
        Memprediksi kelas dari sebuah gambar PIL.
        """
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, pred_idx = torch.max(probs, 0)
            confidence = confidence.item()
            pred_idx = pred_idx.item()

            label = (
                self.class_names[pred_idx]
                if confidence >= confidence_threshold
                else "Tidak Yakin"
            )

        return label, confidence

    def decide_recyclability(self, label: str):
        return True if label in self.recyclable_class else False

    def _create_model(
        self, model_name: str, num_classes: int, feature_extract: bool = True
    ):
        """
        Memuat pre-trained model dari torchvision menggunakan pendekatan hardcode yang andal
        dan menyesuaikannya untuk transfer learning. ✅

        Args:
            model_name (str): Nama model yang didukung (contoh: 'resnet50', 'mobilenet_v3_small').
            num_classes (int): Jumlah kelas output untuk dataset baru.
            feature_extract (bool): Jika True, bekukan bobot kecuali layer terakhir.
                                Jika False, seluruh model akan dilatih (fine-tuning).

        Returns:
            torch.nn.Module: Model yang sudah disesuaikan dan siap pakai.
        """

        supported_models = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "mobilenet_v3_small": (
                models.mobilenet_v3_small,
                models.MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "mobilenet_v3_large": (
                models.mobilenet_v3_large,
                models.MobileNet_V3_Large_Weights.DEFAULT,
            ),
            "efficientnet_b0": (
                models.efficientnet_b0,
                models.EfficientNet_B0_Weights.DEFAULT,
            ),
            "efficientnet_b7": (
                models.efficientnet_b7,
                models.EfficientNet_B7_Weights.DEFAULT,
            ),
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
            "swin_t": (models.swin_t, models.Swin_T_Weights.DEFAULT),
        }

        if model_name not in supported_models:
            raise ValueError(
                f"Model '{model_name}' tidak didukung.\n"
                f"Model yang tersedia: {list(supported_models.keys())}"
            )

        model_constructor, weights = supported_models[model_name]

        model = model_constructor(weights=weights)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        if hasattr(model, "fc"):  # Untuk ResNet, dll.
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

        elif hasattr(model, "classifier"):
            if isinstance(
                model.classifier, nn.Sequential
            ):  # Untuk VGG, MobileNet, EfficientNet
                last_layer = model.classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    num_ftrs = last_layer.in_features
                    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                else:
                    raise TypeError(
                        f"Layer terakhir dari classifier ({type(last_layer)}) bukan nn.Linear."
                    )
            elif isinstance(model.classifier, nn.Linear):  # Untuk DenseNet
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            else:
                raise TypeError(
                    f"Tipe classifier ({type(model.classifier)}) tidak didukung."
                )

        elif hasattr(model, "head"):  # Untuk Vision Transformer, Swin Transformer
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)

        else:
            raise NameError(f"Layer klasifikasi untuk '{model_name}' tidak ditemukan.")

        return model

    def _load_deployment_model(
        self, model_path: str, model_name: str, num_classes: int
    ):
        """
        Memuat model yang sudah dilatih untuk deployment atau prediksi.
        Args:
            model_path (str): Path ke file bobot model (.pt atau .pth).
            model_name (str): Nama arsitektur model (contoh: 'resnet50').
            num_classes (int): Jumlah kelas output yang digunakan saat training.
        Returns:
            torch.nn.Module: Model yang siap digunakan untuk prediksi.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._create_model(model_name=model_name, num_classes=num_classes)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)

        print(
            f"✅ Model '{model_name}' dari '{model_path}' berhasil dimuat di device '{device}' dan siap untuk prediksi."
        )

        return model
