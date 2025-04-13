from utils.preprocess import load_data

if __name__ == "__main__":
    train_loader, test_loader, classes = load_data()
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")