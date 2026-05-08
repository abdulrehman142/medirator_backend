import io
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


class XrayASService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_path = Path(settings.xrayas_model_path)
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parents[2] / self.model_path
        self.device = torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.labels: list[str] = []
        self.ready = False
        self.load_error: str | None = None

    def load_model(self) -> None:
        if self.ready and self.model is not None:
            return

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"XRayAS model file not found: {self.model_path}")

            with open(self.model_path, "rb") as handle:
                package = CPU_Unpickler(handle).load()

            self.labels = list(package.get("classes", []))
            if not self.labels:
                raise ValueError("XRayAS model package is missing class labels.")

            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(1024, len(self.labels))
            model.load_state_dict(package["state_dict"])
            model.to(self.device)
            model.eval()

            self.model = model
            self.ready = True
            self.load_error = None
            logger.info("XRayAS model loaded successfully from %s", self.model_path)

        except Exception as e:
            self.model = None
            self.ready = False
            self.load_error = str(e)
            logger.error("Failed to load XRayAS model: %s", e)
            raise

    def analyze(self, image_path: str, text: str | None = None) -> dict[str, Any]:
        if self.model is None or not self.ready:
            self.load_model()

        if not image_path:
            raise ValueError("An X-ray image path is required for XRayAS analysis.")

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        findings = [
            {"condition": label, "confidence": float(score)}
            for label, score in zip(self.labels, probs)
        ]
        findings.sort(key=lambda item: item["confidence"], reverse=True)

        threshold = 0.25
        significant = [
            item for item in findings if item["confidence"] >= threshold
        ]

        if significant:
            top_labels = ", ".join(
                f"{item['condition']} ({item['confidence']*100:.1f}%" + ")"
                for item in significant[:5]
            )
            answer = (
                f"XRayAS detected the following chest x-ray findings: {top_labels}. "
                "Interpret this as a screening prediction and consult a radiologist for confirmation."
            )
        else:
            answer = (
                "XRayAS analysis did not find any chest x-ray findings above 25% confidence. "
                "If you believe this is a clinical case, please upload a clearer x-ray or consult a radiologist."
            )

        if text:
            answer = f"Question: {text}\n{answer}"

        return {
            "ok": True,
            "answer": answer,
            "confidence": float(findings[0]["confidence"] if findings else 0.0),
            "details": findings[:7],
            "model": "XRayAS",
            "model_name": "XRayAS",
        }


xrayas_service_instance = XrayASService()
