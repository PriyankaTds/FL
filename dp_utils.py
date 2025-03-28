from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from opacus.validators import ModuleValidator
from typing import Dict, List, Union

class IMDataset(Dataset):
    """Custom dataset class to ensure consistent batch formatting"""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function to ensure proper batching"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

class FixedDistilBert(DistilBertForSequenceClassification):
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Ensure batch-first inputs and correct dimensions
        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.size(0) != labels.size(0):
                raise ValueError(f"Batch size mismatch: input_ids {input_ids.shape}, labels {labels.shape}")
                
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        if labels is not None:
            if input_ids.size(0) != labels.size(0):
                raise ValueError(f"Batch size mismatch: input_ids {input_ids.shape}, labels {labels.shape}")
                
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Ensure batch-first output
        if outputs.logits.dim() == 1:
            outputs.logits = outputs.logits.unsqueeze(0)
            
        return outputs

def load_imdb_data():
    dataset = load_dataset('imdb')
    train_size = 7000
    test_size = 2000
    dataset['train'] = dataset['train'].shuffle().select(range(train_size))
    dataset['test'] = dataset['test'].shuffle().select(range(test_size))
    return dataset

def split_data(dataset, client_id, num_clients=3):
    data_size = len(dataset)
    split_size = data_size // num_clients
    start = client_id * split_size
    end = (client_id + 1) * split_size
    indices = list(range(start, end))
    return torch.utils.data.Subset(dataset, indices)

def preprocess_data(dataset, tokenizer):
    def tokenize(examples):
        tokenized = tokenizer(
            examples['text'], 
            padding="max_length", 
            truncation=True, 
            max_length=200,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': examples['label']
        }
    
    tokenized_train = dataset['train'].map(tokenize, batched=True, remove_columns=['text'])
    tokenized_test = dataset['test'].map(tokenize, batched=True, remove_columns=['text'])
    
    # Convert to tensors with explicit types
    train_data = {
        'input_ids': torch.tensor(tokenized_train['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(tokenized_train['attention_mask'], dtype=torch.long),
        'labels': torch.tensor(tokenized_train['labels'], dtype=torch.long)
    }
    
    test_data = {
        'input_ids': torch.tensor(tokenized_test['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(tokenized_test['attention_mask'], dtype=torch.long),
        'labels': torch.tensor(tokenized_test['labels'], dtype=torch.long)
    }
    
    return IMDataset(train_data), IMDataset(test_data)

def create_model():
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Load model and immediately set to train mode
    model = FixedDistilBert.from_pretrained(model_name, num_labels=2)
    model.train()  # Critical for Opacus
    
    # Validate and fix model
    print("Initial validation warnings:", ModuleValidator.validate(model, strict=False))
    model = ModuleValidator.fix(model)
    
    # Special handling for embedding layers
    if not ModuleValidator.is_valid(model.distilbert.embeddings):
        model.distilbert.embeddings = ModuleValidator.fix(model.distilbert.embeddings)
        model.distilbert.embeddings.train()
    
    # Final strict validation
    errors = ModuleValidator.validate(model, strict=True)
    if errors:
        raise RuntimeError(f"Model validation failed: {errors}")
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    
    return model, tokenizer, optimizer, loss_fn