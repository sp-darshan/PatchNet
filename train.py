import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from utils.utils import read_cfg
from Dataset.Dataset import SkinCancerDataset
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

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, save_path = cfg['output_dir']):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    csv_file = cfg['train_set']
    img_dir = cfg['img_dir']

    dataset = SkinCancerDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchNet(patch_size=56, num_classes=7)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=20, save_path = cfg['output_dir'])

if __name__ == '__main__':
    main()
