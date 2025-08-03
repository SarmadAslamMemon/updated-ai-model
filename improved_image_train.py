import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
from collections import Counter
from improved_image_model import SimpleMultiPetDiseaseModel, LabelSmoothingLoss
from improved_image_dataset_loader import get_improved_data_loaders, BalancedSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def get_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    labels = [disease_label for _, _, disease_label in dataset]
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())

    weights = {cls: total_samples/class_counts[cls] for cls in class_counts}
    return torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float)

def calculate_metrics(outputs, targets):
    """Calculate comprehensive metrics"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    
    # Convert to numpy for sklearn metrics
    pred_np = predicted.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    f1 = f1_score(target_np, pred_np, average='weighted')
    
    return accuracy, f1, pred_np, target_np

def evaluate_model(model, data_loader, criterion_species, criterion_disease):
    """Evaluate model on given data loader"""
    model.eval()
    total_loss = 0
    all_species_preds = []
    all_species_targets = []
    all_disease_preds = []
    all_disease_targets = []
    
    with torch.no_grad():
        for images, species_labels, disease_labels in data_loader:
            images, species_labels, disease_labels = images.to(device), species_labels.to(device), disease_labels.to(device).long()
            
            species_outputs, disease_outputs = model(images)

            species_loss = criterion_species(species_outputs, species_labels)
            disease_loss = criterion_disease(disease_outputs, disease_labels)
            loss = species_loss + disease_loss

            total_loss += loss.item()

            # Calculate metrics
            species_acc, species_f1, species_pred, species_target = calculate_metrics(species_outputs, species_labels)
            disease_acc, disease_f1, disease_pred, disease_target = calculate_metrics(disease_outputs, disease_labels)
            
            all_species_preds.extend(species_pred)
            all_species_targets.extend(species_target)
            all_disease_preds.extend(disease_pred)
            all_disease_targets.extend(disease_target)
    
    avg_loss = total_loss / len(data_loader)
    avg_species_acc = sum(all_species_preds[i] == all_species_targets[i] for i in range(len(all_species_preds))) / len(all_species_preds)
    avg_disease_acc = sum(all_disease_preds[i] == all_disease_targets[i] for i in range(len(all_disease_preds))) / len(all_disease_preds)
    
    avg_species_f1 = f1_score(all_species_targets, all_species_preds, average='weighted')
    avg_disease_f1 = f1_score(all_disease_targets, all_disease_preds, average='weighted')
    
    return avg_loss, avg_species_acc, avg_disease_acc, avg_species_f1, avg_disease_f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    data_dir = "dataset"

    # Get improved data loaders
    train_loader, val_loader, test_loader, species_classes, disease_classes, train_dataset = get_improved_data_loaders(data_dir, batch_size=16)

    # Calculate class weights for imbalanced dataset
    disease_weights = get_class_weights(train_dataset).to(device)

    # Initialize improved model
    model = SimpleMultiPetDiseaseModel(num_diseases=len(disease_classes)).to(device)

    # Improved loss functions
    species_criterion = LabelSmoothingLoss(classes=3, smoothing=0.1)  # Label smoothing for better generalization
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weights)  # Weighted loss for imbalanced classes

    # Improved optimizer with better learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_model_path = "improved_multi_pet_disease_model.pth"
    best_val_f1 = 0.0
    patience = 7
    patience_counter = 0

    # Metrics tracking
    train_metrics = {'loss': [], 'species_acc': [], 'disease_acc': [], 'f1_species': [], 'f1_disease': []}
    val_metrics = {'loss': [], 'species_acc': [], 'disease_acc': [], 'f1_species': [], 'f1_disease': []}

    print("🚀 Starting improved image training...")
    
    for epoch in range(30):  # Increased epochs for better convergence
        model.train()
        running_loss = 0.0
        all_species_preds = []
        all_species_targets = []
        all_disease_preds = []
        all_disease_targets = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")
        
        for images, species_labels, disease_labels in progress_bar:
            images, species_labels, disease_labels = images.to(device), species_labels.to(device), disease_labels.to(device).long()

            optimizer.zero_grad()
            species_outputs, disease_outputs = model(images)

            species_loss = species_criterion(species_outputs, species_labels)
            disease_loss = disease_criterion(disease_outputs, disease_labels)
            loss = species_loss + disease_loss

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics
            species_acc, species_f1, species_pred, species_target = calculate_metrics(species_outputs, species_labels)
            disease_acc, disease_f1, disease_pred, disease_target = calculate_metrics(disease_outputs, disease_labels)
            
            all_species_preds.extend(species_pred)
            all_species_targets.extend(species_target)
            all_disease_preds.extend(disease_pred)
            all_disease_targets.extend(disease_target)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Species Acc': f'{species_acc:.3f}',
                'Disease Acc': f'{disease_acc:.3f}'
            })

        # Update learning rate
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        avg_species_acc = sum(all_species_preds[i] == all_species_targets[i] for i in range(len(all_species_preds))) / len(all_species_preds)
        avg_disease_acc = sum(all_disease_preds[i] == all_disease_targets[i] for i in range(len(all_disease_preds))) / len(all_disease_preds)
        avg_species_f1 = f1_score(all_species_targets, all_species_preds, average='weighted')
        avg_disease_f1 = f1_score(all_disease_targets, all_disease_preds, average='weighted')
        
        # Store training metrics
        train_metrics['loss'].append(avg_train_loss)
        train_metrics['species_acc'].append(avg_species_acc)
        train_metrics['disease_acc'].append(avg_disease_acc)
        train_metrics['f1_species'].append(avg_species_f1)
        train_metrics['f1_disease'].append(avg_disease_f1)

        print(f"🟢 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Species Acc: {avg_species_acc:.3f}, Disease Acc: {avg_disease_acc:.3f}")
        print(f"🟢 F1 Scores - Species: {avg_species_f1:.3f}, Disease: {avg_disease_f1:.3f}")

        # Validation phase
        val_loss, val_species_acc, val_disease_acc, val_species_f1, val_disease_f1 = evaluate_model(
            model, val_loader, species_criterion, disease_criterion
        )
        
        # Store validation metrics
        val_metrics['loss'].append(val_loss)
        val_metrics['species_acc'].append(val_species_acc)
        val_metrics['disease_acc'].append(val_disease_acc)
        val_metrics['f1_species'].append(val_species_f1)
        val_metrics['f1_disease'].append(val_disease_f1)

        print(f"🔵 Validation - Loss: {val_loss:.4f}, Species Acc: {val_species_acc:.3f}, Disease Acc: {val_disease_acc:.3f}")
        print(f"🔵 F1 Scores - Species: {val_species_f1:.3f}, Disease: {val_disease_f1:.3f}")

        # Early stopping based on F1 score
        current_f1 = (val_species_f1 + val_disease_f1) / 2
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_f1': best_val_f1,
                'species_classes': species_classes,
                'disease_classes': disease_classes
            }, best_model_path)
            print(f"✅ Best model saved with F1: {best_val_f1:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping after {patience} epochs without improvement")
                break

    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on test set
    print("\n🧪 Final Evaluation on Test Set")
    test_loss, test_species_acc, test_disease_acc, test_species_f1, test_disease_f1 = evaluate_model(
        model, test_loader, species_criterion, disease_criterion
    )

    print(f"🎯 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Species Accuracy: {test_species_acc:.3f}")
    print(f"   Disease Accuracy: {test_disease_acc:.3f}")
    print(f"   Species F1: {test_species_f1:.3f}")
    print(f"   Disease F1: {test_disease_f1:.3f}")

    # Save training history
    training_history = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': {
            'loss': test_loss,
            'species_acc': test_species_acc,
            'disease_acc': test_disease_acc,
            'species_f1': test_species_f1,
            'disease_f1': test_disease_f1
        },
        'species_classes': species_classes,
        'disease_classes': disease_classes
    }

    with open('improved_image_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)

    print("✅ Improved image model training complete!")
    print("📊 Training history saved to improved_image_training_history.json") 