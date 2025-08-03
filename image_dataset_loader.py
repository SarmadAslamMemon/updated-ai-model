import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from collections import Counter
import json

# Define the species mapping
species_mapping = {
    "dog": 0,
    "cat": 1,
    "fish": 2
}

# Extract all disease classes dynamically
def get_disease_classes(root_dir):
    disease_classes = set()
    for species in os.listdir(root_dir):
        species_path = os.path.join(root_dir, species)
        if os.path.isdir(species_path):
            for disease in os.listdir(species_path):
                disease_path = os.path.join(species_path, disease)
                if os.path.isdir(disease_path):
                    disease_classes.add(disease)

    disease_classes = sorted(disease_classes)  # Ensure consistent ordering
    return {disease: idx for idx, disease in enumerate(disease_classes)}

# Get class distribution for imbalance handling
def get_class_distribution(dataset):
    disease_counts = Counter([disease_label for _, _, disease_label in dataset])
    return {disease: count for disease, count in sorted(disease_counts.items())}

# Define MultiPetDataset
class MultiPetDataset(Dataset):
    def __init__(self, root_dir, transform=None, disease_classes=None):
        self.data = []
        self.transform = transform
        self.disease_classes = disease_classes
        self.invalid_images = 0  # Track corrupted images

        for species in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species)
            if os.path.isdir(species_path):
                for disease in os.listdir(species_path):
                    disease_path = os.path.join(species_path, disease)
                    if os.path.isdir(disease_path):
                        for image_name in os.listdir(disease_path):
                            image_path = os.path.join(disease_path, image_name)
                            self.data.append((image_path, species, disease))

        print(f"📢 Loaded {len(self.data)} images from {root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, species_name, disease_name = self.data[idx]

        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            print(f"⚠️ Skipping corrupted/missing image: {image_path}")
            self.invalid_images += 1
            return self.__getitem__((idx + 1) % len(self))  # Load next valid image

        if self.transform:
            image = self.transform(image)

        species_label = species_mapping.get(species_name.lower(), -1)
        disease_label = self.disease_classes.get(disease_name, -1)

        if species_label == -1 or disease_label == -1:
            print(f"⚠️ Skipping invalid labels: {species_name} - {disease_name}")
            return self.__getitem__((idx + 1) % len(self))  # Load next valid image

        return image, species_label, disease_label

# Improved Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to create dataset loaders
def get_data_loaders(data_dir, batch_size=32):
    disease_classes = get_disease_classes(os.path.join(data_dir, "train"))

    train_dataset = MultiPetDataset(os.path.join(data_dir, "train"), transform=transform, disease_classes=disease_classes)
    val_dataset = MultiPetDataset(os.path.join(data_dir, "val"), transform=transform, disease_classes=disease_classes)
    test_dataset = MultiPetDataset(os.path.join(data_dir, "test"), transform=transform, disease_classes=disease_classes)

    print("🔍 Disease Class Distribution:", json.dumps(get_class_distribution(train_dataset), indent=4))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, list(species_mapping.keys()), list(disease_classes.keys()), train_dataset
