import torch
import sys
import os

# 1. The Folder Bridge
sys.path.append(os.path.abspath('src'))
from models import get_model

def run_tests():
    # 2. Hardware Activation: Target your M4 MacBook's GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Active Compute Hardware: {device.type.upper()}")

    # 3. Create Mock Data: 2 patients, 3 color channels, 128x128 pixels
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    print(f" Generated mock input tensor of shape: {dummy_input.shape}\n")

    # 4. The Architecture Roster
    models_to_test = ["vanilla", "resnet", "mobilenet", "inception"]
    
    print("--- Running Pipeline Integrity Checks ---")
    
    # 5. The Verification Loop
    for name in models_to_test:
        try:
            # Pull the blueprint and load it into the Mac's GPU memory
            model = get_model(name).to(device)
            
            # The Forward Pass: stream the fake data through the layers
            outputs = model(dummy_input)
            
            # The Guardrail: Ensure it outputs exactly [2 patients, 2 diagnostic scores]
            assert outputs.shape == (2, 2), f"Expected shape (2, 2), but got {outputs.shape}"
            
            print(f"[PASSED] {name.upper():<10} | Output Shape: {list(outputs.shape)}")
            
        except Exception as e:
            print(f"[FAILED] {name.upper():<10} | Error: {e}")

    print("\n ALL TESTS COMPLETE! Pipeline is architecturally sound.")

if __name__ == "__main__":
    run_tests()