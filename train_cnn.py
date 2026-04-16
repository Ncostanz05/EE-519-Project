import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from tqdm import tqdm

from cnn_dataset import CommonVoiceSpectrogramDataset
from cnn_model import AgeAudioCNN

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Initializing dataset...")
    full_dataset = CommonVoiceSpectrogramDataset(
        metadata_path="cv_au_subset/metadata.csv",
        audio_dir="cv_au_subset/audio"
    )

    # 2. Speaker-Independent Split (Train/Val/Test)
    # First split: 70% Train, 30% Temp (Val + Test)
    splitter_main = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    client_ids = full_dataset.df["client_id"].values
    
    dummy_X = np.zeros(len(client_ids))
    train_idx, temp_idx = next(splitter_main.split(dummy_X, groups=client_ids))
    
    # Second split: 50% of the Temp portion (15% total) for Val, 50% for Test
    splitter_temp = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    temp_client_ids = client_ids[temp_idx]
    dummy_X_temp = np.zeros(len(temp_client_ids))
    
    val_rel_idx, test_rel_idx = next(splitter_temp.split(dummy_X_temp, groups=temp_client_ids))
    
    # Map relative indices back to original dataset indices
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"Dataset split: {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Test")

    # 3. Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model, Loss, Optimizer
    model = AgeAudioCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 5. Training Loop
    num_epochs = 20
    best_val_acc = 0.0
    patience = 3
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Wrap the loader in tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            mel_spec, labels, _ = batch
            # DataLoaders naturally batch the 1-channel mel specs: (Batch, 1, Mels, Time)
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': running_loss/total_train*batch_size, 'acc': correct_train/total_train})
            
        # 6. Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec, labels, _ = batch
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)
                
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = correct_val / total_val
        val_loss_avg = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary - Train Acc: {correct_train/total_train:.3f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss_avg:.3f}")

        # 7. Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "cnn_age_model.pth")
            print("  --> Best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.3f}")

    # 8. Evaluate on Untouched Test Set
    print("\nLoading best model for Final Test Evaluation...")
    best_model = AgeAudioCNN(num_classes=7).to(device)
    best_model.load_state_dict(torch.load("cnn_age_model.pth"))
    best_model.eval()

    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch in test_loader:
            mel_spec, labels, _ = batch
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            
            outputs = best_model(mel_spec)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    test_acc = correct_test / total_test
    print(f"=====================================")
    print(f"FINAL OFFICIAL TEST ACCURACY: {test_acc:.3f}")
    print(f"=====================================")

if __name__ == "__main__":
    train()
