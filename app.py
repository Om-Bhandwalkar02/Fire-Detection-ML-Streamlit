import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Import your models and utils
from models.cnn import FireCNN
from models.logistic_regression import LogisticRegressionModel
from models.resnet import ResNet18Model
from utils.preprocess import load_data

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)

st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide"
)


def load_trained_model(model_class, model_path):
    """Load a trained model from file"""
    model = model_class().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return None


def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(uploaded_image).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_image(model, image_tensor):
    """Make prediction for image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[0]
    return predicted.item(), probabilities.cpu().numpy()


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
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


def load_outputs_images():
    """Load visualization images from the outputs folder if they exist"""
    images = {}
    output_files = [
        "confusion_matrix_cnn.png",
        "confusion_matrix_logistic_regression.png",
        "confusion_matrix_resnet18.png",
        "roc_curves.png",
        "metrics_comparison.png"
    ]

    for filename in output_files:
        path = os.path.join("outputs", filename)
        if os.path.exists(path):
            images[filename] = path

    return images


def generate_analysis_images():
    """Generate analysis images using the analyze_results.py logic"""
    try:
        from analyze_results import main as analyze_main
        analyze_main()
        st.success("Analysis images generated successfully!")
    except Exception as e:
        st.error(f"Failed to generate analysis images: {e}")


def main():
    st.title("🔥 Fire Detection System")

    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "Model Analysis"])

    with tab1:
        st.header("Fire Detection Prediction")

        # File uploader for prediction
        uploaded_file = st.file_uploader("Upload an image for fire detection", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.write("Making predictions...")
                # Process image and make predictions
                try:
                    image_tensor = preprocess_image(uploaded_file)

                    # Load test dataloader for accuracy evaluation
                    _, test_loader, _ = load_data(batch_size=32)

                    # Load all models and make predictions
                    models = {
                        "CNN": (FireCNN, "models/cnn.pth"),
                        "Logistic Regression": (LogisticRegressionModel, "models/logistic_regression.pth"),
                        "ResNet18": (ResNet18Model, "models/resnet18.pth")
                    }

                    for name, (model_class, model_path) in models.items():
                        model = load_trained_model(model_class, model_path)

                        if model:
                            # Evaluate on test set
                            with st.spinner(f"Evaluating {name} model..."):
                                accuracy = evaluate_model(model, test_loader)

                            # Predict for single image
                            prediction, confidence = predict_image(model, image_tensor)

                            # Display results
                            st.subheader(f"{name} Model Results")

                            # Create a progress bar for confidence
                            if prediction == 0:  # Fire
                                st.error(f"🔥 FIRE DETECTED (Confidence: {confidence[0] * 100:.2f}%)")
                                st.progress(float(confidence[0]))
                            else:  # No Fire
                                st.success(f"🟢 NO FIRE (Confidence: {confidence[1] * 100:.2f}%)")
                                st.progress(float(confidence[1]))

                            st.write(f"Test Accuracy: {accuracy:.2f}%")
                            st.write("---")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    with tab2:
        st.header("Model Analysis Results")

        # Button to generate/refresh analysis
        if st.button("Generate/Refresh Analysis"):
            with st.spinner("Generating analysis..."):
                generate_analysis_images()

        # Load and display output images
        output_images = load_outputs_images()

        if not output_images:
            st.warning("No analysis images found. Please click 'Generate/Refresh Analysis' to create them.")
        else:
            # Display confusion matrices
            st.subheader("Confusion Matrices")
            cols = st.columns(3)

            # Show confusion matrices in columns
            if "confusion_matrix_cnn.png" in output_images:
                cols[0].image(output_images["confusion_matrix_cnn.png"], caption="CNN Confusion Matrix")

            if "confusion_matrix_logistic_regression.png" in output_images:
                cols[1].image(output_images["confusion_matrix_logistic_regression.png"],
                              caption="Logistic Regression Confusion Matrix")

            if "confusion_matrix_resnet18.png" in output_images:
                cols[2].image(output_images["confusion_matrix_resnet18.png"], caption="ResNet18 Confusion Matrix")

            # Show ROC curves
            st.subheader("ROC Curves")
            if "roc_curves.png" in output_images:
                st.image(output_images["roc_curves.png"], caption="ROC Curves Comparison")

            # Show metrics comparison
            st.subheader("Model Metrics Comparison")
            if "metrics_comparison.png" in output_images:
                st.image(output_images["metrics_comparison.png"], caption="Metrics Comparison")


if __name__ == "__main__":
    main()