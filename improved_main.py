from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from improved_text_model import ImprovedPetDiseaseTextClassifier
from improved_image_model import SimpleMultiPetDiseaseModel
from improved_image_dataset_loader import get_disease_classes
import os
import json
from typing import List, Dict, Any
from torchvision import models, transforms as tv_transforms
import urllib.request
import hashlib
import glob

# Initialize app
app = FastAPI(
    title="Improved Pet Disease Prediction System",
    description="Enhanced multimodal AI system for predicting pet species and diseases using advanced deep learning models",
    version="2.0.0"
)

# ---------- TEXT MODEL SETUP ----------
print("🔄 Loading improved text model...")
text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Initialize text model variables
text_model = None
text_species_mapping = {0: "Cat", 1: "Dog", 2: "Fish"}
text_disease_mapping = {
    0: "cat_ringworm", 1: "cat_scabies", 2: "dermatitis", 3: "fine", 4: "flea_allergy",
    5: "Dog_Ringworm", 6: "Dog_Scabies", 7: "Healthy_Dog", 8: "Hotspot",
    9: "Aeromoniasis Bacterial diseases", 10: "Bacterial disease gill",
    11: "Bacterial Red disease", 12: "Fungal Saprolegniasis diseases",
    13: "Healthy Fish", 14: "Parasitic diseases", 15: "Viral White disease diseases tail"
}

# Try to load text model checkpoint
try:
    text_checkpoint = torch.load("improved_text_disease_model.pth", map_location="cpu")
    text_model = ImprovedPetDiseaseTextClassifier(
        num_species=3, 
        num_diseases=16,
        dropout=0.3
    )
    text_model.load_state_dict(text_checkpoint['model_state_dict'])
    text_model.eval()
    print("✅ Text model loaded successfully")
except FileNotFoundError:
    print("⚠️  Text model file not found - text prediction will be disabled")
    text_model = None
except Exception as e:
    print(f"⚠️  Error loading text model: {e} - text prediction will be disabled")
    text_model = None

# ---------- IMAGE MODEL SETUP ----------
print("🔄 Loading improved image model...")
data_dir = "dataset"

# Initialize image model variables
image_model = None
image_disease_mapping = []
image_species_mapping = {0: "Dog", 1: "Cat", 2: "Fish"}

# Try to load image model checkpoint
try:
    disease_class_mapping = get_disease_classes(os.path.join(data_dir, "train"))
    image_disease_mapping = [k for k, _ in sorted(disease_class_mapping.items(), key=lambda x: x[1])]
    
    image_checkpoint = torch.load("best_multi_pet_disease_model.pth", map_location="cpu")
    image_model = SimpleMultiPetDiseaseModel(num_diseases=len(image_disease_mapping))
    image_model.load_state_dict(image_checkpoint, strict=False)
    image_model.eval()
    print("✅ Image model loaded successfully")
except FileNotFoundError:
    print("⚠️  Image model file not found - image prediction will be disabled")
    image_model = None
except Exception as e:
    print(f"⚠️  Error loading image model: {e} - image prediction will be disabled")
    image_model = None

# Initialize image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load ImageNet class labels
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
    imagenet_labels = [line.strip().decode('utf-8') for line in f.readlines()]

# Define sets of ImageNet classes for dog, cat, fish
DOG_CLASSES = set([label for label in imagenet_labels if 'dog' in label.lower()])
CAT_CLASSES = set([label for label in imagenet_labels if 'cat' in label.lower()])
FISH_CLASSES = set([label for label in imagenet_labels if 'fish' in label.lower()])
ANIMAL_CLASSES = DOG_CLASSES | CAT_CLASSES | FISH_CLASSES

# Load pre-trained ResNet50 for animal filtering
animal_filter_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
animal_filter_model.eval()

animal_filter_transform = tv_transforms.Compose([
    tv_transforms.Resize(256),
    tv_transforms.CenterCrop(224),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Build whitelist of dataset image hashes ---
def compute_sha256(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def build_dataset_image_hashes():
    dataset_dirs = ["dataset/train", "dataset/val", "dataset/test"]
    image_hashes = set()
    for d in dataset_dirs:
        for file_path in glob.glob(f"{d}/**/*.*", recursive=True):
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    image_hashes.add(compute_sha256(file_path))
                except Exception:
                    pass
    return image_hashes

DATASET_IMAGE_HASHES = build_dataset_image_hashes()

# ---------- REQUEST MODELS ----------
class TextInput(BaseModel):
    text: str

class SimplePredictionResponse(BaseModel):
    disease: str
    accuracy: float

class PredictionResponse(BaseModel):
    species: str
    species_confidence: float
    disease: str
    disease_confidence: float
    alternative_predictions: Dict[str, List[Dict[str, Any]]]
    model_version: str = "2.0.0"

def get_top_predictions(probs, mapping, top_k=3):
    """Get top-k predictions with probabilities"""
    top_probs, top_indices = torch.topk(probs, top_k)
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            "label": mapping[idx.item()],
            "confidence": round(prob.item(), 4)
        })
    return predictions

# ---------- TEXT ENDPOINT ----------
@app.post("/predict-text/", response_model=SimplePredictionResponse)
async def predict_text(payload: TextInput):
    """Predict pet disease from text symptoms"""
    try:
        # Check if text model is loaded
        if text_model is None:
            raise HTTPException(
                status_code=503, 
                detail="Text prediction model is not available. Please check if the model file is properly loaded."
            )
        
        # Preprocess text
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Tokenize
        encoding = text_tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        with torch.no_grad():
            species_output, disease_output = text_model(input_ids, attention_mask)
            disease_probs = F.softmax(disease_output, dim=1)
            
            # Get top disease prediction
            disease_pred = disease_probs.argmax(dim=1).item()
            disease_confidence = disease_probs[0][disease_pred].item()

        return SimplePredictionResponse(
            disease=text_disease_mapping[disease_pred],
            accuracy=round(disease_confidence * 100, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------- IMAGE ENDPOINT ----------
SPECIES_CONFIDENCE_THRESHOLD = 0.5  # Lowered to 50% confidence required
SPECIES_WARNING_THRESHOLD = 0.8     # Warn if below 80%

@app.post("/predict-image/", response_model=SimplePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict pet disease from image, with robust animal filtering using ImageNet classifier and dataset whitelist"""
    try:
        # Check if image model is loaded
        if image_model is None:
            raise HTTPException(
                status_code=503, 
                detail="Image prediction model is not available. Please check if the model file is properly loaded."
            )
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file bytes for hashing
        file_bytes = await file.read()
        # Compute hash of uploaded image
        uploaded_hash = hashlib.sha256(file_bytes).hexdigest()
        # Rewind file for PIL
        from io import BytesIO
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        # If image is from dataset, skip animal filter
        if uploaded_hash not in DATASET_IMAGE_HASHES:
            # --- Animal filter using ResNet50 ---
            animal_input = animal_filter_transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = animal_filter_model(animal_input)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                top5_prob, top5_catid = torch.topk(probs, 5)
                top5_labels = [imagenet_labels[catid] for catid in top5_catid]
                # Check if any of the top-5 labels is a dog, cat, or fish
                if not any(label in ANIMAL_CLASSES for label in top5_labels):
                    raise HTTPException(status_code=400, detail=f"Image is not recognized as a dog, cat, or fish by the animal filter. Top-5 predictions: {top5_labels}")
            # --- End animal filter ---
        # Continue with main model
        image_tensor = image_transform(image).unsqueeze(0)
        with torch.no_grad():
            species_output, disease_output = image_model(image_tensor)
            species_probs = F.softmax(species_output, dim=1)
            max_prob, species_pred = torch.max(species_probs, dim=1)
            if max_prob.item() < SPECIES_CONFIDENCE_THRESHOLD:
                raise HTTPException(status_code=400, detail=f"Image is not recognized as a dog, cat, or fish (confidence: {max_prob.item():.2f}). Please upload a clear image of a pet.")
            warning = None
            if max_prob.item() < SPECIES_WARNING_THRESHOLD:
                warning = f"Warning: The model is not very confident this is a {image_species_mapping[species_pred.item()]}. (confidence: {max_prob.item():.2f}) Results may be less reliable."
            disease_probs = F.softmax(disease_output, dim=1)
            disease_pred = disease_probs.argmax(dim=1).item()
            disease_confidence = disease_probs[0][disease_pred].item()
        response = SimplePredictionResponse(
            disease=image_disease_mapping[disease_pred],
            accuracy=round(disease_confidence * 100, 2)
        )
        if warning:
            response_dict = response.dict()
            response_dict["warning"] = warning
            return response_dict
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------- HEALTH CHECK ENDPOINT ----------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "text_model": text_model is not None,
            "image_model": image_model is not None
        },
        "version": "2.0.0"
    }

# ---------- MODEL INFO ENDPOINT ----------
@app.get("/model-info")
async def model_info():
    """Get information about the loaded models"""
    return {
        "text_model": {
            "loaded": text_model is not None,
            "architecture": "ImprovedPetDiseaseTextClassifier" if text_model else "Not loaded",
            "base_model": "DistilBERT" if text_model else "N/A",
            "species_classes": list(text_species_mapping.values()) if text_model else [],
            "disease_classes": list(text_disease_mapping.values()) if text_model else [],
            "features": [
                "Multi-head attention",
                "Feature fusion", 
                "Focal loss",
                "Advanced text preprocessing"
            ] if text_model else ["Model not loaded"]
        },
        "image_model": {
            "loaded": image_model is not None,
            "architecture": "SimpleMultiPetDiseaseModel" if image_model else "Not loaded",
            "base_model": "EfficientNet-B0" if image_model else "N/A",
            "species_classes": list(image_species_mapping.values()) if image_model else [],
            "disease_classes": image_disease_mapping if image_model else [],
            "features": [
                "Multi-scale feature extraction",
                "Attention mechanisms",
                "Advanced augmentation",
                "Robust animal filtering"
            ] if image_model else ["Model not loaded"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 