import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from transformers import DistilBertTokenizer
from torchvision import transforms
from PIL import Image

# Import original models
from text_model import PetDiseaseTextClassifier
from multi_pet_model import MultiPetDiseaseModel

# Import improved models
from improved_text_model import ImprovedPetDiseaseTextClassifier
from improved_image_model import ImprovedMultiPetDiseaseModel

# Import dataset loaders
from text_dataset_loader import load_dataset
from improved_text_dataset_loader import load_improved_dataset
from image_dataset_loader import get_data_loaders
from improved_image_dataset_loader import get_improved_data_loaders

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def evaluate_text_model(self, model, data_loader, model_name):
        """Evaluate text model performance"""
        model.eval()
        all_species_preds = []
        all_species_targets = []
        all_disease_preds = []
        all_disease_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                species_labels = batch["species_label"].to(self.device)
                disease_labels = batch["disease_label"].to(self.device)

                species_output, disease_output = model(input_ids, attention_mask)
                
                # Get predictions
                _, species_preds = torch.max(species_output, 1)
                _, disease_preds = torch.max(disease_output, 1)
                
                all_species_preds.extend(species_preds.cpu().numpy())
                all_species_targets.extend(species_labels.cpu().numpy())
                all_disease_preds.extend(disease_preds.cpu().numpy())
                all_disease_targets.extend(disease_labels.cpu().numpy())
        
        # Calculate metrics
        species_accuracy = sum(p == t for p, t in zip(all_species_preds, all_species_targets)) / len(all_species_targets)
        disease_accuracy = sum(p == t for p, t in zip(all_disease_preds, all_disease_targets)) / len(all_disease_targets)
        
        species_f1 = f1_score(all_species_targets, all_species_preds, average='weighted')
        disease_f1 = f1_score(all_disease_targets, all_disease_preds, average='weighted')
        
        precision_species, recall_species, _, _ = precision_recall_fscore_support(all_species_targets, all_species_preds, average='weighted')
        precision_disease, recall_disease, _, _ = precision_recall_fscore_support(all_disease_targets, all_disease_preds, average='weighted')
        
        return {
            'species_accuracy': species_accuracy,
            'disease_accuracy': disease_accuracy,
            'species_f1': species_f1,
            'disease_f1': disease_f1,
            'species_precision': precision_species,
            'species_recall': recall_species,
            'disease_precision': precision_disease,
            'disease_recall': recall_disease,
            'species_predictions': all_species_preds,
            'species_targets': all_species_targets,
            'disease_predictions': all_disease_preds,
            'disease_targets': all_disease_targets
        }
    
    def evaluate_image_model(self, model, data_loader, model_name):
        """Evaluate image model performance"""
        model.eval()
        all_species_preds = []
        all_species_targets = []
        all_disease_preds = []
        all_disease_targets = []
        
        with torch.no_grad():
            for images, species_labels, disease_labels in data_loader:
                images = images.to(self.device)
                species_labels = species_labels.to(self.device)
                disease_labels = disease_labels.to(self.device)
                
                species_output, disease_output = model(images)
                
                # Get predictions
                _, species_preds = torch.max(species_output, 1)
                _, disease_preds = torch.max(disease_output, 1)
                
                all_species_preds.extend(species_preds.cpu().numpy())
                all_species_targets.extend(species_labels.cpu().numpy())
                all_disease_preds.extend(disease_preds.cpu().numpy())
                all_disease_targets.extend(disease_labels.cpu().numpy())
        
        # Calculate metrics
        species_accuracy = sum(p == t for p, t in zip(all_species_preds, all_species_targets)) / len(all_species_targets)
        disease_accuracy = sum(p == t for p, t in zip(all_disease_preds, all_disease_targets)) / len(all_disease_targets)
        
        species_f1 = f1_score(all_species_targets, all_species_preds, average='weighted')
        disease_f1 = f1_score(all_disease_targets, all_disease_preds, average='weighted')
        
        precision_species, recall_species, _, _ = precision_recall_fscore_support(all_species_targets, all_species_preds, average='weighted')
        precision_disease, recall_disease, _, _ = precision_recall_fscore_support(all_disease_targets, all_disease_preds, average='weighted')
        
        return {
            'species_accuracy': species_accuracy,
            'disease_accuracy': disease_accuracy,
            'species_f1': species_f1,
            'disease_f1': disease_f1,
            'species_precision': precision_species,
            'species_recall': recall_species,
            'disease_precision': precision_disease,
            'disease_recall': recall_disease,
            'species_predictions': all_species_preds,
            'species_targets': all_species_targets,
            'disease_predictions': all_disease_preds,
            'disease_targets': all_disease_targets
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, title, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        report = {
            'summary': {},
            'detailed_metrics': {},
            'improvements': {}
        }
        
        # Calculate improvements
        for model_type in ['text', 'image']:
            if f'original_{model_type}' in self.results and f'improved_{model_type}' in self.results:
                original = self.results[f'original_{model_type}']
                improved = self.results[f'improved_{model_type}']
                
                improvements = {}
                for metric in ['species_accuracy', 'disease_accuracy', 'species_f1', 'disease_f1']:
                    if metric in original and metric in improved:
                        improvement = ((improved[metric] - original[metric]) / original[metric]) * 100
                        improvements[metric] = round(improvement, 2)
                
                report['improvements'][model_type] = improvements
        
        return report
    
    def save_results(self, output_dir='evaluation_results'):
        """Save all evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Save comparison report
        report = self.generate_comparison_report()
        with open(os.path.join(output_dir, 'comparison_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate summary table
        self.generate_summary_table(output_dir)
        
        print(f"✅ Evaluation results saved to {output_dir}/")
    
    def generate_summary_table(self, output_dir):
        """Generate summary table of results"""
        data = []
        
        for model_name, results in self.results.items():
            if 'species_accuracy' in results:
                data.append({
                    'Model': model_name,
                    'Species Accuracy': f"{results['species_accuracy']:.3f}",
                    'Disease Accuracy': f"{results['disease_accuracy']:.3f}",
                    'Species F1': f"{results['species_f1']:.3f}",
                    'Disease F1': f"{results['disease_f1']:.3f}",
                    'Species Precision': f"{results['species_precision']:.3f}",
                    'Species Recall': f"{results['species_recall']:.3f}",
                    'Disease Precision': f"{results['disease_precision']:.3f}",
                    'Disease Recall': f"{results['disease_recall']:.3f}"
                })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
        
        # Print summary
        print("\n📊 Model Performance Summary:")
        print(df.to_string(index=False))

def main():
    """Main evaluation function"""
    print("🔍 Starting comprehensive model evaluation...")
    
    evaluator = ModelEvaluator()
    
    # Load datasets
    print("📂 Loading datasets...")
    
    # Text datasets
    try:
        _, test_dataset, _, _ = load_dataset("pet.csv")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Load original text model
        print("🔄 Loading original text model...")
        original_text_model = PetDiseaseTextClassifier(num_species=3, num_diseases=16)
        original_text_model.load_state_dict(torch.load("text_disease_model.pth", map_location='cpu'))
        original_text_model.to(evaluator.device)
        
        # Evaluate original text model
        print("📊 Evaluating original text model...")
        evaluator.results['original_text'] = evaluator.evaluate_text_model(
            original_text_model, test_loader, "original_text"
        )
        
    except Exception as e:
        print(f"⚠️ Could not evaluate original text model: {e}")
    
    try:
        # Load improved text model
        print("🔄 Loading improved text model...")
        improved_text_model = ImprovedPetDiseaseTextClassifier(num_species=3, num_diseases=16)
        checkpoint = torch.load("improved_text_disease_model.pth", map_location='cpu')
        improved_text_model.load_state_dict(checkpoint['model_state_dict'])
        improved_text_model.to(evaluator.device)
        
        # Evaluate improved text model
        print("📊 Evaluating improved text model...")
        evaluator.results['improved_text'] = evaluator.evaluate_text_model(
            improved_text_model, test_loader, "improved_text"
        )
        
    except Exception as e:
        print(f"⚠️ Could not evaluate improved text model: {e}")
    
    # Image datasets
    try:
        print("🔄 Loading image datasets...")
        _, _, test_loader, _, _, _ = get_data_loaders("dataset")
        
        # Load original image model
        print("🔄 Loading original image model...")
        original_image_model = MultiPetDiseaseModel(num_diseases=16)
        original_image_model.load_state_dict(torch.load("best_multi_pet_disease_model.pth", map_location='cpu'))
        original_image_model.to(evaluator.device)
        
        # Evaluate original image model
        print("📊 Evaluating original image model...")
        evaluator.results['original_image'] = evaluator.evaluate_image_model(
            original_image_model, test_loader, "original_image"
        )
        
    except Exception as e:
        print(f"⚠️ Could not evaluate original image model: {e}")
    
    try:
        # Load improved image model
        print("🔄 Loading improved image model...")
        improved_image_model = ImprovedMultiPetDiseaseModel(num_diseases=16)
        checkpoint = torch.load("improved_multi_pet_disease_model.pth", map_location='cpu')
        improved_image_model.load_state_dict(checkpoint['model_state_dict'])
        improved_image_model.to(evaluator.device)
        
        # Evaluate improved image model
        print("📊 Evaluating improved image model...")
        evaluator.results['improved_image'] = evaluator.evaluate_image_model(
            improved_image_model, test_loader, "improved_image"
        )
        
    except Exception as e:
        print(f"⚠️ Could not evaluate improved image model: {e}")
    
    # Generate confusion matrices
    print("📈 Generating confusion matrices...")
    for model_name, results in evaluator.results.items():
        if 'species_predictions' in results:
            evaluator.plot_confusion_matrix(
                results['species_targets'],
                results['species_predictions'],
                f'{model_name} - Species Classification',
                f'evaluation_results/{model_name}_species_cm.png'
            )
            
            evaluator.plot_confusion_matrix(
                results['disease_targets'],
                results['disease_predictions'],
                f'{model_name} - Disease Classification',
                f'evaluation_results/{model_name}_disease_cm.png'
            )
    
    # Save results
    evaluator.save_results()
    
    # Print improvements
    report = evaluator.generate_comparison_report()
    if 'improvements' in report:
        print("\n🚀 Performance Improvements:")
        for model_type, improvements in report['improvements'].items():
            print(f"\n{model_type.upper()} Model:")
            for metric, improvement in improvements.items():
                print(f"  {metric}: {improvement:+.2f}%")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main() 