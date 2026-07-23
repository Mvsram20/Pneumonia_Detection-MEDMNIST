import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)

# Add src folder to workspace path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dataset import get_transforms
from models import get_model

def evaluate_single_model(model_name, test_loader, device):
    """
    Evaluates one model on the test dataset and returns metrics + raw prediction probabilities.
    """
    # Checkpoint path fallbacks
    checkpoint_path = f"trained/{model_name}_best.pth"
    if not os.path.exists(checkpoint_path):
        if os.path.exists(f"best_model_{model_name}.pth"):
            checkpoint_path = f"best_model_{model_name}.pth"
        elif model_name == "vanilla" and os.path.exists("best_model.pth"):
            checkpoint_path = "best_model.pth"
        else:
            print(f"⚠️ Checkpoint for {model_name.upper()} not found at '{checkpoint_path}'... Skipping.")
            return None, None, None

    # Load model and inject parameters
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long()

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability for Pneumonia (Class 1)

    # Compute metric scores
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc_val = roc_auc_score(all_targets, all_probs)

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "Specificity": spec, "F1-Score": f1, "AUC-ROC": auc_val
    }

    return metrics, np.array(all_targets), np.array(all_probs)


def plot_colored_performance_graph(model_data):
    """
    Plots a single ROC Performance graph comparing all models with distinct line colors.
    """
    plt.figure(figsize=(9, 6))

    # Distinct color and line style configuration for each model
    color_palette = {
        "vanilla":   {"label": "Vanilla CNN",  "color": "#3371A3", "style": "-"},  # Blue
        "resnet":    {"label": "ResNet-18",   "color": "#E06666", "style": "-"},  # Red
        "inception": {"label": "Inception",    "color": "#38A149", "style": "-"},  # Green
        "mobilenet": {"label": "MobileNet-V2", "color": "#674EA7", "style": "-"}   # Purple
    }

    print("\n📈 Generating Multi-Model Performance Comparison Graph...")

    for model_name, (targets, probs) in model_data.items():
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        
        cfg = color_palette.get(model_name.lower(), {"label": model_name.upper(), "color": "orange", "style": "-"})

        # Draw the distinct colored line for this model
        plt.plot(
            fpr, tpr, 
            color=cfg["color"], 
            linestyle=cfg["style"],
            linewidth=2.2, 
            label=f'{cfg["label"]} (AUC = {roc_auc:.4f})'
        )

    # Reference baseline line for random guessing
    plt.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.5, label='Random Baseline (AUC = 0.50)')

    # Chart Styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    plt.title('Multi-Model Clinical Performance Comparison Graph', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor='#F8F9FA', edgecolor='#D3D3D3')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Export graph image
    plt.tight_layout()
    output_filename = "multi_model_performance_graph.png"
    plt.savefig(output_filename, dpi=300)
    print(f"💾 SUCCESS: Colored performance graph saved as '{output_filename}'")
    plt.show()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running evaluation engine on hardware: {device.type.upper()}")

    # Load test dataset via get_transforms()
    print("Loading test dataset...")
    _, val_transform = get_transforms()
    test_dataset = PneumoniaMNIST(split='test', transform=val_transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    models_list = ["vanilla", "resnet", "inception", "mobilenet"]
    results = {}
    graph_data = {}

    for model_name in models_list:
        print(f"Analyzing {model_name.upper()}...")
        metrics, targets, probs = evaluate_single_model(model_name, test_loader, device)
        if metrics is not None:
            results[model_name] = metrics
            graph_data[model_name] = (targets, probs)

    if not results:
        print("\n❌ Error: No trained model checkpoint files (.pth) found in directory.")
        return

    # Print Live Metrics Table
    print("\n" + "="*86)
    print(f"{'Model':15s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'Specificity':11s} | {'F1-Score':8s} | {'AUC-ROC':8s}")
    print("="*86)
    for model_name, m in results.items():
        print(f"{model_name.upper():15s} | "
              f"{m['Accuracy']*100:8.2f}% | "
              f"{m['Precision']*100:8.2f}% | "
              f"{m['Recall']*100:8.2f}% | "
              f"{m['Specificity']*100:10.2f}% | "
              f"{m['F1-Score']*100:7.2f}% | "
              f"{m['AUC-ROC']*100:7.2f}%")
    print("="*86)

    # Draw colored lines graph
    plot_colored_performance_graph(graph_data)

if __name__ == "__main__":
    main()