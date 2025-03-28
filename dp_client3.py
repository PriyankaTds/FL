import flwr as fl
from dp_utils import load_imdb_data, preprocess_data, create_model, split_data, custom_collate_fn, IMDataset
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message=".*grad_sample.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def validate_grad_samples(model, batch_size):
    """Validate all gradient samples have correct batch dimension"""
    for name, param in model.named_parameters():
        if hasattr(param, 'grad_sample'):
            if param.grad_sample.dim() < 2 or param.grad_sample.size(0) != batch_size:
                print(f"Fixing grad_sample shape for {name}: {param.grad_sample.shape}")
                param.grad_sample = param.grad_sample.unsqueeze(0).expand(
                    batch_size, *[-1]*(param.grad_sample.dim())
                )

def main():
    # Load data and model
    dataset = load_imdb_data()
    model, tokenizer, optimizer, loss_fn = create_model()
    model.to(device)
    model.train()

    # Preprocess and split data
    train_dataset, test_dataset = preprocess_data(dataset, tokenizer)
    train_subset = split_data(train_dataset, client_id=2)
    test_subset = split_data(test_dataset, client_id=2)

    # Create DataLoaders with custom collate
    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    # Initialize Privacy Engine with robust settings
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.5,
    max_grad_norm=1.0,
    poisson_sampling=False,
    clipping="flat",
    grad_sample_mode="hooks"  # This goes in make_private() now
)

    class IMDBClient(fl.client.NumPyClient):
        def __init__(self, model, train_loader, test_loader, optimizer, loss_fn, privacy_engine):
            self.model = model
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.privacy_engine = privacy_engine

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def fit(self, parameters, config):
            # Set parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

            # Train with progress bar
            self.model.train()
            total_loss, total_accuracy = 0, 0
            num_samples = 0

            with tqdm(self.train_loader, desc="Training", unit="batch") as progress:
                for batch in progress:
                    self.optimizer.zero_grad()
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    # Verify batch dimensions
                    assert input_ids.size(0) == labels.size(0), \
                        f"Batch size mismatch: {input_ids.size(0)} vs {labels.size(0)}"
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    loss.backward()
                    
                    # Validate and fix gradient samples
                    validate_grad_samples(self.model, input_ids.size(0))
                    
                    self.optimizer.step()

                    # Metrics
                    predictions = torch.argmax(outputs.logits, dim=1)
                    batch_accuracy = (predictions == labels).sum().item()
                    total_accuracy += batch_accuracy
                    total_loss += loss.item()
                    num_samples += len(labels)

                    # Update progress bar
                    progress.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{batch_accuracy/len(labels):.2%}',
                        'ε': f'{self.privacy_engine.get_epsilon(delta=1e-3):.2f}'
                    })

            avg_loss = total_loss / len(self.train_loader)
            avg_accuracy = total_accuracy / num_samples
            epsilon = self.privacy_engine.get_epsilon(delta=1e-3)
            
            print(f"\nTraining Summary - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, ε: {epsilon:.2f}")
            return self.get_parameters(config), num_samples, {
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "epsilon": epsilon,
            "num_examples": num_samples  # THIS IS CRUCIAL
            }

        def evaluate(self, parameters, config):
            # Set parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

            self.model.eval()
            total_loss, total_accuracy = 0, 0
            num_samples = 0

            with tqdm(self.test_loader, desc="Evaluating", unit="batch") as progress:
                with torch.no_grad():
                    for batch in progress:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels = labels
                        )
                        loss = self.loss_fn(outputs.logits, labels)
                        total_loss += loss.item()
                        
                        predictions = torch.argmax(outputs.logits, dim=1)
                        batch_accuracy = (predictions == labels).sum().item()
                        total_accuracy += batch_accuracy
                        num_samples += len(labels)

                        progress.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{batch_accuracy/len(labels):.2%}'
                        })

            avg_loss = total_loss / len(self.test_loader)
            avg_accuracy = total_accuracy / num_samples
            epsilon = self.privacy_engine.get_epsilon(delta=1e-3)
            print(f"\nEvaluation Summary - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            return avg_loss, num_samples, {
            "accuracy": avg_accuracy,
            "num_examples": num_samples,
              "epsilon": epsilon  # THIS IS CRUCIAL
            }
            

    # Create and start client
    client = IMDBClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        privacy_engine=privacy_engine
    )

    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=client
    )

if __name__ == "__main__":
    main()