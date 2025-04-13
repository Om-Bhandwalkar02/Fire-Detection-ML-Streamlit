from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_data(data_dir='dataset', batch_size=32, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Class Mapping: {dataset.class_to_idx}")

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.classes