from PIL import Image
import pandas as pd
import os
from utils.utils import read_cfg
from torch.utils.data import Dataset 

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Define a mapping from string labels to numerical indices.
        self.label_map = {
            "nv": 0,
            "mel": 1,
            "bkl": 2,
            "bcc": 3,
            "akiec": 4,
            "vasc": 5,
            "df": 6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row corresponding to the idx
        row = self.data.iloc[idx]
        img_id = row['image_id']
        # Construct the full image path (adjust extension if needed)
        '''
        if img_id.endswith('.jpg'):
            img_path = os.path.join(self.img_dir, img_id)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                img_path = os.path.join(self.img_dir, "augmented", img_id)
                image = Image.open(img_path).convert('RGB')
        else:
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                img_path = os.path.join(self.img_dir, "augmented", f"{img_id}.jpg")
                image = Image.open(img_path).convert('RGB')
        '''
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Get the target label and map it to an integer
        target_str = row['dx']
        label = self.label_map[target_str]
        
        return image, label