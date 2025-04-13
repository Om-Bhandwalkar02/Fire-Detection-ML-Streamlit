import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
    accuracy_score, ConfusionMatrixDisplay
)
import os

from models.cnn import FireCNN
from models.logistic_regression import LogisticRegressionModel
from models.resnet import ResNet18Model
from utils.preprocess import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_predictions(model, test_loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_metrics(labels, preds, probs, model_name):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = auc(*roc_curve(labels, probs)[:2])

    print(f"\n📊 {model_name.upper()} METRICS:")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")
    print("=" * 40)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': roc_auc
    }


def plot_confusion_matrix(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Fire", "Fire"])
    disp.plot(cmap="OrRd")
    plt.title(f"Confusion Matrix - {title}")
    plt.savefig(f"outputs/confusion_matrix_{title.lower().replace(' ', '_')}.png")
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)
    _, test_loader, _ = load_data()

    models_info = {
        "CNN": (FireCNN, "models/cnn.pth"),
        "Logistic Regression": (LogisticRegressionModel, "models/logistic_regression.pth"),
        "ResNet18": (ResNet18Model, "models/resnet18.pth")
    }

    metrics_data = {}
    plt.figure(figsize=(8, 6))

    for name, (model_class, path) in models_info.items():
        print(f"\n🔍 Analyzing {name}...")
        model = load_model(model_class, path)
        preds, labels, probs = get_predictions(model, test_loader)

        metrics = print_metrics(labels, preds, probs, name)
        metrics_data[name] = metrics
        plot_confusion_matrix(labels, preds, name)

        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig("outputs/roc_curves.png")
    plt.close()

    # Metrics comparison plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(
            x=list(metrics_data.keys()),
            y=[m[metric] for m in metrics_data.values()]
        )
        plt.title(metric.replace('_', ' ').title())
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/metrics_comparison.png")
    plt.close()

    print("\n✅ Analysis complete! Check the 'outputs' folder for visualizations.")


if __name__ == "__main__":
    main()