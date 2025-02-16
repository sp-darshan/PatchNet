import torch
import torch.nn as nn
import torchvision.models as models

class PatchNet(nn.Module):
    def __init__(self, patch_size=56, num_classes=4):
        """
        Args:
            patch_size (int): The height and width of each patch.
                              (For a 224x224 image and patch_size=56, we get 16 patches.)
            num_classes (int): Number of target classes.
        """
        super(PatchNet, self).__init__()
        self.patch_size = patch_size
        
        # Define a simple CNN as a feature extractor for each patch.
        # All patches share these weights.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # output: 32 x patch_size x patch_size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # output: 64 x (patch_size/2) x (patch_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions again
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 x (patch_size/4) x (patch_size/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # global pooling to get a 128-dim vector
        )
        
        # Load Pretrained ResNet50 Model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 128)  # Modify final layer

        # A final classifier that maps the aggregated feature to the target classes.
        self.classifier = nn.Linear(256, num_classes)  # Updated for concatenated features

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        B, C, H, W = x.size()
        
        # Use nn.Unfold to extract non-overlapping patches.
        # Here, we assume that H and W are divisible by patch_size.
        # For a 224x224 image with patch_size=56, there will be (224/56)^2 = 16 patches.
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold(x)  # shape: (B, C * patch_size * patch_size, num_patches)
        num_patches = patches.shape[-1]
        
        # Rearrange patches to shape (B * num_patches, C, patch_size, patch_size)
        patches = patches.transpose(1, 2).contiguous().view(B * num_patches, C, self.patch_size, self.patch_size)
        
        # Pass each patch through the feature extractor
        patch_features = self.feature_extractor(patches)  # shape: (B * num_patches, 128, 1, 1)
        patch_features = patch_features.view(B, num_patches, -1)  # shape: (B, num_patches, 128)
        
        # Aggregate the patch features using mean pooling across the patches
        aggregated_features = patch_features.mean(dim=1)  # shape: (B, 128)
        
        # Extract global features using ResNet50
        global_features = self.resnet50(x)  # shape: (B, 128)
        
        # Concatenate patch-based features and global features
        combined_features = torch.cat((aggregated_features, global_features), dim=1)  # shape: (B, 256)
        
        # Final classification
        logits = self.classifier(combined_features)  # shape: (B, num_classes)
        return logits
