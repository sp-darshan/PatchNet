import torch
import torch.nn as nn
import torchvision.models as models

class PatchNet(nn.Module):
    def __init__(self, patch_size=56, num_classes=4):
        """
        Args:
            patch_size (int): The height and width of each patch.
            num_classes (int): Number of target classes.
        """
        super(PatchNet, self).__init__()
        self.patch_size = patch_size

        # Feature extractor for each patch with Dropout
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Dropout(p=0.5)  # Dropout to reduce overfitting
        )

        # Load Pretrained ResNet50 with reduced output size
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 64)  # Reduce complexity

        # Final classifier (Reduced input from 256 → 128)
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),  # Reduce input size
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout before final layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # Extract patches
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold(x)  # shape: (B, C * patch_size * patch_size, num_patches)
        num_patches = patches.shape[-1]

        # Reshape patches
        patches = patches.transpose(1, 2).contiguous().view(B * num_patches, C, self.patch_size, self.patch_size)

        # Patch-based features
        patch_features = self.feature_extractor(patches)
        patch_features = patch_features.view(B, num_patches, -1)
        aggregated_features = patch_features.mean(dim=1)  # shape: (B, 128)

        # Global features from ResNet50
        global_features = self.resnet50(x)  # shape: (B, 64)

        # Combine both features
        combined_features = torch.cat((aggregated_features, global_features), dim=1)  # shape: (B, 192)

        # Classification
        logits = self.classifier(combined_features)
        return logits
