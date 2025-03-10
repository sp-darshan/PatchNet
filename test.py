import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Dataset.Dataset import SkinCancerDataset
from models.patchnet import PatchNet
from utils.utils import read_cfg
import pandas as pd

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, dataloader, device, criterion):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")

def main():
    cfg = read_cfg(cfg_file='D:\\skin\\config\\config.yaml')
    csv_file = cfg['test_set']
    df = pd.read_csv(csv_file)
    #print(df.shape)
    img_dir = cfg['img_dir']

    dataset = SkinCancerDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchNet(patch_size=56, num_classes=7)
    model.load_state_dict(torch.load("patchnet_model.pth", map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    
    evaluate_model(model, dataloader, device, criterion)

if __name__ == '__main__':
    main()
