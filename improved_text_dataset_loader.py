import pandas as pd
import torch
import re
import random
import numpy as np
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class TextAugmentation:
    """Text augmentation techniques for better training"""
    
    @staticmethod
    def synonym_replacement(text, n=1):
        """Replace words with synonyms"""
        synonyms = {
            'cat': ['feline', 'kitty', 'kitten'],
            'dog': ['canine', 'puppy', 'hound'],
            'fish': ['aquatic', 'fin', 'scaly'],
            'itching': ['scratching', 'irritation', 'discomfort'],
            'red': ['inflamed', 'irritated', 'sore'],
            'spots': ['patches', 'lesions', 'marks'],
            'hair': ['fur', 'coat', 'pelage'],
            'loss': ['falling', 'shedding', 'balding'],
            'skin': ['dermis', 'epidermis', 'coat'],
            'disease': ['condition', 'illness', 'infection'],
            'healthy': ['fine', 'well', 'normal'],
            'symptoms': ['signs', 'indicators', 'manifestations']
        }
        
        words = text.split()
        n = min(n, len(words))
        new_words = words.copy()
        
        for _ in range(n):
            for i, word in enumerate(new_words):
                if word.lower() in synonyms:
                    synonym = random.choice(synonyms[word.lower()])
                    new_words[i] = synonym
                    break
        
        return ' '.join(new_words)
    
    @staticmethod
    def random_insertion(text, n=1):
        """Insert random words"""
        words = text.split()
        n = min(n, len(words))
        
        for _ in range(n):
            insert_words = ['very', 'really', 'quite', 'extremely', 'slightly']
            random_word = random.choice(insert_words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text, p=0.1):
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return words[rand_int]
        
        return ' '.join(new_words)

def clean_text(text):
    """Clean and preprocess text - simplified version without NLTK dependencies"""
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Simple word splitting (no NLTK dependency)
    words = text.split()
    
    # Basic lemmatization using simple rules
    lemmatized_words = []
    for word in words:
        # Simple lemmatization rules
        if word.endswith('ing'):
            word = word[:-3]
        elif word.endswith('ed'):
            word = word[:-2]
        elif word.endswith('s'):
            word = word[:-1]
        lemmatized_words.append(word)
    
    text = ' '.join(lemmatized_words)
    
    return text.strip()

class ImprovedPetDiseaseDataset(Dataset):
    def __init__(self, texts, species_labels, disease_labels, augment=False, max_length=128):
        self.texts = texts
        self.species_labels = torch.tensor(species_labels, dtype=torch.long)
        self.disease_labels = torch.tensor(disease_labels, dtype=torch.long)
        self.augment = augment
        self.max_length = max_length
        
        # Tokenize all texts
        self.encodings = tokenizer(
            [clean_text(text) for text in texts],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.species_labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["species_label"] = self.species_labels[idx]
        item["disease_label"] = self.disease_labels[idx]
        return item

def load_improved_dataset(csv_path, test_size=0.2, val_size=0.1, augment_train=True, random_state=42):
    """Load and preprocess dataset with improved techniques"""
    df = pd.read_csv(csv_path)
    
    # Ensure all text entries are strings and clean them
    df["Symptoms"] = df["Symptoms"].astype(str).fillna("No description available")
    df["Symptoms"] = df["Symptoms"].apply(clean_text)
    
    # Convert species and disease labels to categorical values
    species_mapping = {species: idx for idx, species in enumerate(df["Species"].unique())}
    disease_mapping = {disease: idx for idx, disease in enumerate(df["Disease"].unique())}

    print("🔍 Species Mapping:", species_mapping)
    print("🔍 Disease Mapping:", disease_mapping)
    print(f"📊 Total samples: {len(df)}")
    
    df["species_label"] = df["Species"].map(species_mapping)
    df["disease_label"] = df["Disease"].map(disease_mapping)
    
    # Split dataset into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state, stratify=df['species_label'])
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=temp_df['species_label'])
    
    print(f"📊 Train samples: {len(train_df)}")
    print(f"📊 Validation samples: {len(val_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    
    # Data augmentation for training set
    if augment_train:
        augmented_texts = []
        augmented_species = []
        augmented_diseases = []
        
        for _, row in train_df.iterrows():
            text = row["Symptoms"]
            species = row["species_label"]
            disease = row["disease_label"]
            
            # Original sample
            augmented_texts.append(text)
            augmented_species.append(species)
            augmented_diseases.append(disease)
            
            # Augmented samples (for minority classes)
            if random.random() < 0.3:  # 30% chance of augmentation
                # Synonym replacement
                aug_text = TextAugmentation.synonym_replacement(text, n=1)
                augmented_texts.append(aug_text)
                augmented_species.append(species)
                augmented_diseases.append(disease)
                
                # Random insertion
                aug_text = TextAugmentation.random_insertion(text, n=1)
                augmented_texts.append(aug_text)
                augmented_species.append(species)
                augmented_diseases.append(disease)
        
        train_dataset = ImprovedPetDiseaseDataset(
            augmented_texts, augmented_species, augmented_diseases, augment=False
        )
    else:
        train_dataset = ImprovedPetDiseaseDataset(
            train_df["Symptoms"].tolist(),
            train_df["species_label"].tolist(),
            train_df["disease_label"].tolist(),
            augment=False
        )
    
    val_dataset = ImprovedPetDiseaseDataset(
        val_df["Symptoms"].tolist(),
        val_df["species_label"].tolist(),
        val_df["disease_label"].tolist(),
        augment=False
    )
    
    test_dataset = ImprovedPetDiseaseDataset(
        test_df["Symptoms"].tolist(),
        test_df["species_label"].tolist(),
        test_df["disease_label"].tolist(),
        augment=False
    )
    
    return train_dataset, val_dataset, test_dataset, species_mapping, disease_mapping 