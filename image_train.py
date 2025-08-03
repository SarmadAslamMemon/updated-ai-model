import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
from collections import Counter
from multi_pet_model import MultiPetDiseaseModel
from image_dataset_loader import get_data_loaders

def get_class_weights(dataset):
    labels = [disease_label for _, _, disease_label in dataset]
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())

    weights = {cls: total_samples/class_counts[cls] for cls in class_counts}
    return torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "dataset"

    train_loader, val_loader, test_loader, species_classes, disease_classes, train_dataset = get_data_loaders(data_dir)

    disease_weights = get_class_weights(train_dataset).to(device)

    model = MultiPetDiseaseModel(num_diseases=len(disease_classes)).to(device)

    species_criterion = nn.CrossEntropyLoss()
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.3)

    best_model_path = "best_multi_pet_disease_model.pth"
    best_val_loss = float("inf")

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct_species, correct_disease = 0, 0
        total_samples = 0

        for images, species_labels, disease_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
            images, species_labels, disease_labels = images.to(device), species_labels.to(device), disease_labels.to(device).long()

            optimizer.zero_grad()
            species_outputs, disease_outputs = model(images)

            species_loss = species_criterion(species_outputs, species_labels)
            disease_loss = disease_criterion(disease_outputs, disease_labels)
            loss = species_loss + disease_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, species_predicted = species_outputs.max(1)
            _, disease_predicted = disease_outputs.max(1)

            correct_species += species_predicted.eq(species_labels).sum().item()
            correct_disease += disease_predicted.eq(disease_labels).sum().item()
            total_samples += species_labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        species_accuracy = 100 * correct_species / total_samples
        disease_accuracy = 100 * correct_disease / total_samples

        print(f"🟢 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Species Accuracy: {species_accuracy:.2f}%, Disease Accuracy: {disease_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_species, correct_disease = 0, 0
        total_samples = 0

        with torch.no_grad():
            for images, species_labels, disease_labels in val_loader:
                images, species_labels, disease_labels = images.to(device), species_labels.to(device).long(), disease_labels.to(device).long()
                species_outputs, disease_outputs = model(images)

                species_loss = species_criterion(species_outputs, species_labels)
                disease_loss = disease_criterion(disease_outputs, disease_labels)
                loss = species_loss + disease_loss

                val_loss += loss.item()

                _, species_predicted = species_outputs.max(1)
                _, disease_predicted = disease_outputs.max(1)

                correct_species += species_predicted.eq(species_labels).sum().item()
                correct_disease += disease_predicted.eq(disease_labels).sum().item()
                total_samples += species_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_species_accuracy = 100 * correct_species / total_samples
        val_disease_accuracy = 100 * correct_disease / total_samples

        print(f"🔵 Validation Loss: {avg_val_loss:.4f}, Species Accuracy: {val_species_accuracy:.2f}%, Disease Accuracy: {val_disease_accuracy:.2f}%")

        # Adjust Learning Rate
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best Model Updated: {best_model_path}")
