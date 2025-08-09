import torch
import torch.nn as nn
from transformers import DistilBertModel

class ImprovedPetDiseaseTextClassifier(nn.Module):
    def __init__(self, num_species=3, num_diseases=16, dropout=0.3):
        super(ImprovedPetDiseaseTextClassifier, self).__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Freeze BERT layers for fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Classification heads
        bert_output_dim = self.bert.config.hidden_size
        
        # Species classification head
        self.species_classifier = nn.Sequential(
            nn.Linear(bert_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_species)
        )
        
        # Disease classification head
        self.disease_classifier = nn.Sequential(
            nn.Linear(bert_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_diseases)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation for classification
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        
        # Get predictions
        species_logits = self.species_classifier(cls_output)
        disease_logits = self.disease_classifier(cls_output)
        
        return species_logits, disease_logits
