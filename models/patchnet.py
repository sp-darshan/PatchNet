import torch
import torch.nn as nn
import torchvision.models as models

class PatchNet(nn.Module):
    def __init__(self, patch_size=56, num_classes=4):
        super(PatchNet, self).__init__()
        self.patch_size = patch_size

        # Feature extractor for each patch with Layer Normalization
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output shape (B, 128, 1, 1)
            nn.Flatten(),  # Convert to (B, 128)
            nn.GroupNorm(8, 128),  # Normalize features
            nn.Dropout(p=0.5)  # Regularization
        )

        # Pretrained ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 64)  # Reduce to 64-dim

        # Feature Reduction Layers
        self.patch_fc = nn.Linear(128, 64)  # Reduce patch features
        self.global_fc = nn.Linear(64, 64)  # Reduce ResNet features

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # Extract overlapping patches
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size // 2)  # Overlapping patches
        patches = unfold(x)  # Shape: (B, C * patch_size * patch_size, num_patches)
        num_patches = patches.shape[-1]

        # Reshape patches for CNN processing
        patches = patches.transpose(1, 2).contiguous().view(B * num_patches, C, self.patch_size, self.patch_size)

        # Patch feature extraction
        patch_features = self.feature_extractor(patches)  # (B*num_patches, 128)
        patch_features = patch_features.view(B, num_patches, -1).mean(dim=1)  # Aggregate patch features
        patch_features = self.patch_fc(patch_features)  # (B, 64)

        # Global ResNet50 feature extraction
        global_features = self.resnet50(x)  # (B, 64)
        global_features = self.global_fc(global_features)  # (B, 64)

        # Combine both feature vectors
        combined_features = torch.cat((patch_features, global_features), dim=1)  # (B, 128)

        # Classification
        logits = self.classifier(combined_features)
        return logits
