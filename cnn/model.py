# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class MultiHeadRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Head for x, y coordinates
        self.coord_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # x1, y1
        )

        # Head for size (height)
        self.size_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # height
        )

        # Head for aspect ratio
        self.ratio_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # ratio
        )

    def forward(self, x):
        x = self.shared(x)
        coords = self.coord_head(x)
        size = self.size_head(x)
        ratio = self.ratio_head(x)
        # concatenate: [x1, y1, height, ratio]
        return torch.cat([coords, size, ratio], dim=1)


def create_model():
    # Load pretrained backbone
    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()

    # Attach multi-head regressor
    reg_head = MultiHeadRegressor(in_features)
    return nn.Sequential(backbone, reg_head)
