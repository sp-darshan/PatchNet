import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torchvision.transforms as transforms
import pandas as pd
from utils.utils import read_cfg
from Dataset.Dataset import SkinCancerDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.patchnet import PatchNet

cfg = read_cfg(cfg_file='D:\\skin\\config\\config.yaml')

# We resize the images to 224x224 so that our patch extraction works cleanly.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize using ImageNet means and stds (adjust if necessary)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset from CSV file
train_set_path = cfg['train_set']  # This is just a file path
train_df = pd.read_csv(train_set_path)  # Read the CSV file

# Count occurrences of each class in 'dx' column
class_counts = Counter(train_df['dx'])
total_samples = sum(class_counts.values())

# Compute weights (inverse frequency)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

# Convert class weights into a tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float32).to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, patience=5, save_path=cfg['output_dir']):
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = save_path

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)  # Save best model
            print(f"Best model saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break  # Stop training

    print(f"Training complete. Best model saved at {best_model_path}")

def main():
    csv_file = cfg['train_set']
    val_file = cfg['val_set']
    img_dir = cfg['img_dir']

    train_dataset = SkinCancerDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    val_dataset = SkinCancerDataset(csv_file=val_file, img_dir=img_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchNet(patch_size=32, num_classes=7)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15, patience=5, save_path=cfg['output_dir'])

if __name__ == '__main__':
    main()
