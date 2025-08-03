import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageEnhance, ImageFilter
from collections import Counter
import json
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Define the species mapping
species_mapping = {
    "dog": 0,
    "cat": 1,
    "fish": 2
}

class AdvancedImageAugmentation:
    """Advanced image augmentation techniques"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # Albumentations for better augmentation
        self.albumentations = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.Affine(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),  # Use Affine instead of ShiftScaleRotate
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.albumentations(image=image)['image']

class MixupAugmentation:
    """Mixup augmentation for better generalization"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_labels = labels
        
        return mixed_images, mixed_labels, lam

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
    disease_counts = Counter([disease_label.item() if torch.is_tensor(disease_label) else disease_label for _, _, disease_label in dataset])
    return {disease: count for disease, count in sorted(disease_counts.items())}

class ImprovedMultiPetDataset(Dataset):
    def __init__(self, root_dir, transform=None, disease_classes=None, augment=True):
        self.data = []
        self.transform = transform
        self.disease_classes = disease_classes
        self.augment = augment
        self.invalid_images = 0
        self.advanced_augment = AdvancedImageAugmentation()

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
            # Return a valid item instead of recursive call to avoid infinite loops
            return self.__getitem__((idx + 1) % len(self))

        # Apply transformations
        if self.augment:
            # Use advanced augmentation for training
            image = self.advanced_augment(image)
        elif self.transform:
            # Use basic transformation for validation/test
            image = self.transform(image)
        else:
            # Convert to tensor if no transform specified
            image = transforms.ToTensor()(image)

        species_label = species_mapping.get(species_name.lower(), -1)
        disease_label = self.disease_classes.get(disease_name, -1)

        if species_label == -1 or disease_label == -1:
            print(f"⚠️ Skipping invalid labels: {species_name} - {disease_name}")
            # Return a valid item instead of recursive call to avoid infinite loops
            return self.__getitem__((idx + 1) % len(self))

        # Ensure labels are tensors
        species_label = torch.tensor(species_label, dtype=torch.long)
        disease_label = torch.tensor(disease_label, dtype=torch.long)

        return image, species_label, disease_label

# Basic transformation for validation/test
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to create improved dataset loaders
def get_improved_data_loaders(data_dir, batch_size=32, num_workers=4):
    disease_classes = get_disease_classes(os.path.join(data_dir, "train"))

    # Create datasets with different augmentation strategies
    train_dataset = ImprovedMultiPetDataset(
        os.path.join(data_dir, "train"), 
        transform=None,  # Will use advanced augmentation
        disease_classes=disease_classes, 
        augment=True
    )
    
    val_dataset = ImprovedMultiPetDataset(
        os.path.join(data_dir, "val"), 
        transform=basic_transform, 
        disease_classes=disease_classes, 
        augment=False
    )
    
    test_dataset = ImprovedMultiPetDataset(
        os.path.join(data_dir, "test"), 
        transform=basic_transform, 
        disease_classes=disease_classes, 
        augment=False
    )

    print("🔍 Disease Class Distribution:", json.dumps(get_class_distribution(train_dataset), indent=4))

    # Create data loaders with improved settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory for CPU
        drop_last=True  # Drop incomplete batches for better training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for CPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for CPU
    )

    return train_loader, val_loader, test_loader, list(species_mapping.keys()), list(disease_classes.keys()), train_dataset

class BalancedSampler:
    """Balanced sampler for handling class imbalance"""
    
    def __init__(self, dataset, num_samples=None):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples else len(dataset)
        
        # Get class labels
        self.labels = [disease_label for _, _, disease_label in dataset]
        
        # Calculate class weights
        class_counts = Counter(self.labels)
        self.class_weights = {cls: len(self.labels) / count for cls, count in class_counts.items()}
        
        # Create balanced indices
        self.balanced_indices = []
        for cls in class_counts.keys():
            cls_indices = [i for i, label in enumerate(self.labels) if label == cls]
            # Oversample minority classes
            if len(cls_indices) < self.num_samples // len(class_counts):
                cls_indices = cls_indices * (self.num_samples // len(class_counts) // len(cls_indices) + 1)
            self.balanced_indices.extend(cls_indices[:self.num_samples // len(class_counts)])
        
        # Shuffle indices
        random.shuffle(self.balanced_indices)
    
    def __iter__(self):
        return iter(self.balanced_indices)
    
    def __len__(self):
        return len(self.balanced_indices) 