import torch
import torch.nn as nn
import torch.optim as optim
import os

from models.cnn import FireCNN
from models.logistic_regression import LogisticRegressionModel
from models.resnet import ResNet18Model
from utils.preprocess import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
    return model


def evaluate_model(model, test_loader):
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
    return correct / total


def main():
    os.makedirs("models", exist_ok=True)
    train_loader, test_loader, _ = load_data()

    models = {
        "cnn": FireCNN(),
        "logistic_regression": LogisticRegressionModel(),
        "resnet18": ResNet18Model()
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_model = train_model(model, train_loader)
        accuracy = evaluate_model(trained_model, test_loader)
        torch.save(trained_model.state_dict(), f"models/{name}.pth")
        print(f"{name} accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()