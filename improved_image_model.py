import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleMultiPetDiseaseModel(nn.Module):
    def __init__(self, num_diseases, dropout=0.5, feature_dim=1280):
        super(SimpleMultiPetDiseaseModel, self).__init__()
        # Load EfficientNet-B0 backbone (same as original working model)
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove classifier
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = feature_dim
        
        # Species classifier head - using Sequential to match checkpoint
        self.species_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        # Disease classifier head - using Sequential to match checkpoint
        self.disease_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 16)
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)  # (batch, 1280, 7, 7)
        features = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)  # (batch, 1280)
        features = self.dropout(features)
        species_output = self.species_classifier(features)
        disease_output = self.disease_classifier(features)
        return species_output, disease_output

# Keep LabelSmoothingLoss as is
class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization"""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) 