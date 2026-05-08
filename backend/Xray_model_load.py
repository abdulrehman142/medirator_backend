import pickle
import torch
import torch.nn as nn
from torchvision import models
import io

# 1. FIXED PATH (Raw string for Windows)
PKL_FILENAME = r'C:\Users\Hassan Raza\Downloads\chest_xray_model_full.pkl'

# 2. CUSTOM UNPICKLER FOR CPU MAPPING
# This tells torch to ignore the GPU requirement from the Kaggle session
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

# 3. Load the package safely on CPU
with open(PKL_FILENAME, 'rb') as f:
    reloaded_package = CPU_Unpickler(f).load()

saved_state_dict = reloaded_package['state_dict']
saved_classes = reloaded_package['classes']

# 4. Setup Model (Force CPU)
device = torch.device('cpu') 
reloaded_model = models.densenet121(weights=None)
reloaded_model.classifier = nn.Linear(1024, len(saved_classes))

# 5. Load weights
reloaded_model.load_state_dict(saved_state_dict)
reloaded_model.to(device)
reloaded_model.eval()

print("✅ Model successfully reloaded on your CPU!")
print(f"Diseases this model can detect: {saved_classes}")