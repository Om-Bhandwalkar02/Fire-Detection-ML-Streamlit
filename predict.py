import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

from models.cnn import FireCNN
from models.logistic_regression import LogisticRegressionModel
from models.resnet import ResNet18Model
from utils.preprocess import load_data

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


def predict_single_image(model, image_tensor):
    """Make prediction for a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[0]
    return predicted.item(), probabilities.cpu().numpy()


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100


def load_trained_model(model_class, model_path):
    """Load a trained model from file"""
    model = model_class().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        print(f"❌ Model file not found: {model_path}")
        return None


def print_prediction_result(model_name, prediction, confidence, accuracy):
    """Print formatted prediction results"""
    class_map = {0: "🔥 FIRE DETECTED", 1: "🟢 NO FIRE"}
    print(f"\n{'=' * 50}")
    print(f"{model_name.upper()} MODEL RESULTS")
    print(f"{'=' * 50}")
    print(f"Prediction: {class_map[prediction]}")
    print(f"Confidence: {confidence[prediction] * 100:.2f}%")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'-' * 50}")
    print(f"Class Probabilities:")
    print(f"Fire: {confidence[0] * 100:.2f}% | No Fire: {confidence[1] * 100:.2f}%")


def main(image_path):
    """Main prediction function"""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # Load test dataloader for accuracy evaluation
    _, test_loader, _ = load_data(batch_size=32)

    # Load models
    models = {
        "CNN": (FireCNN, "models/cnn.pth"),
        "Logistic Regression": (LogisticRegressionModel, "models/logistic_regression.pth"),
        "ResNet18": (ResNet18Model, "models/resnet18.pth")
    }

    # Load and process image
    image_tensor = load_image(image_path)
    if image_tensor is None:
        return

    print(f"\n🔍 Predicting for image: {image_path}\n")

    # Get predictions from all models
    for name, (model_class, model_path) in models.items():
        model = load_trained_model(model_class, model_path)
        if model:
            # Evaluate on test set
            accuracy = evaluate_model(model, test_loader)

            # Predict single image
            prediction, confidence = predict_single_image(model, image_tensor)

            # Print results
            print_prediction_result(name, prediction, confidence, accuracy)


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Fire Detection Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # Run prediction
    main(args.image)