import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import io

# ── 1. DEFINE DISEASE LABELS (Required for the chart) ──
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ── 2. CONFIG ──
IMAGE_PATH = r"C:\Users\Hassan Raza\Pictures\Screenshots\Screenshot 2026-05-07 234112.png" 
# Ensure this path matches where you downloaded the file
MODEL_PKL_PATH = r'C:\Users\Hassan Raza\Downloads\chest_xray_model_full.pkl'

device = torch.device('cpu') # Force CPU for your laptop

# ── 3. CUSTOM LOADER FOR CPU (To prevent the CUDA error) ──
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

# ── 4. LOAD MODEL FUNCTION ──
def load_inference_model(path):
    # Rebuild architecture
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, len(DISEASES))
    
    # Load from the .pkl file
    with open(path, 'rb') as f:
        reloaded_package = CPU_Unpickler(f).load()
    
    model.load_state_dict(reloaded_package['state_dict'])
    model.to(device)
    model.eval()
    return model

# ── 5. PREDICTION FUNCTION ──
def predict_xray(img_path):
    print("Loading model and image...")
    model = load_inference_model(MODEL_PKL_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0].numpy() # Apply sigmoid for percentages

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Input X-Ray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(DISEASES))
    plt.barh(y_pos, probs * 100, color='skyblue')
    plt.yticks(y_pos, DISEASES)
    plt.xlabel('Confidence (%)')
    plt.title('Prediction Scores')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()

    print("\n--- DETECTED FINDINGS (Threshold > 25%) ---")
    found = False
    for i, disease in enumerate(DISEASES):
        if probs[i] > 0.25:
            print(f"⚠️ {disease}: {probs[i]*100:.2f}%")
            found = True
    if not found:
        print("✅ No significant findings detected.")

# Run it!
if __name__ == "__main__":
    predict_xray(IMAGE_PATH)