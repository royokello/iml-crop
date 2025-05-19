import torch
import torch
import torch.nn as nn
from transformers import ViTModel
import size_conversion

class ViTCropper(nn.Module):
    """
    ViT-based model for image cropping.
    Uses a pre-trained Vision Transformer as the backbone and adds separate heads for:
      - Normalized crop coordinates (x, y)
      - Crop shape classification (predicting the shape class)
    """
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224', num_shape_classes: int = 12):
        super(ViTCropper, self).__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        hidden_size = self.vit.config.hidden_size

        # Shared MLP before heads
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Head for x, y coordinates (normalized between 0 and 1)
        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

        # Head for crop shape classification
        # This predicts which of the predefined crop shapes to use
        self.shape_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_shape_classes)  # logits for shape classes
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through ViT and separate heads.

        Args:
            x: Tensor of shape (batch, 3, 224, 224), already normalized.
        Returns:
            Tuple of two tensors:
              coords: (batch, 2) tensor of [x, y] normalized coordinates
              shape_logits: (batch, num_shape_classes) tensor of logits for shape classification
        """
        x = x.to(next(self.parameters()).device)
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]

        shared_feats = self.shared(cls_token)
        coords = self.coord_head(shared_feats)  # x, y coordinates
        shape_logits = self.shape_head(shared_feats)  # shape class logits

        return coords, shape_logits
    
    def predict(self, x: torch.Tensor):
        """
        Make predictions with the model and convert to usable format.
        
        Args:
            x: Tensor of shape (batch, 3, 224, 224), already normalized.
        Returns:
            List of tuples, each containing:
              (x, y, shape_class) where shape_class is an integer
        """
        coords, shape_logits = self.forward(x)
        shape_preds = torch.argmax(shape_logits, dim=1) + 1  # Add 1 because classes start at 1
        
        # Convert to list of tuples
        results = []
        for i in range(coords.shape[0]):
            x, y = coords[i].cpu().numpy().tolist()
            shape_class = shape_preds[i].item()
            results.append((x, y, shape_class))
        
        return results

