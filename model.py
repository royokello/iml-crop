import torch
import torch.nn as nn
from transformers import ViTModel

class IMLCropModel(nn.Module):
    """
    ViT-based model for image cropping.
    Uses a pre-trained Vision Transformer as the backbone and adds separate heads for:
      - Normalized crop coordinates (x, y)
      - Normalized width (w)
      - Aspect ratio classification (dynamic number of classes)
    """
    def __init__(self, 
                 pretrained_model_name='google/vit-base-patch16-384', 
                 num_ratio_classes: int = 1):
        super(IMLCropModel, self).__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        hidden_size = self.vit.config.hidden_size
        
        # Number of aspect ratio classes (dynamic)
        self.num_ratio_classes = num_ratio_classes

        # Shared MLP before heads
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Head for x, y coordinates and width (normalized between 0 and 1)
        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # x, y, width
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

        # Head for aspect ratio classification (dynamic output size)
        self.ratio_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_ratio_classes)  # logits for ratio classes
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through ViT and separate heads.

        Args:
            x: Tensor of shape (batch, 3, 224, 224), already normalized.
        Returns:
            Tuple of two tensors:
              coords: (batch, 3) tensor of [x, y, width] normalized coordinates
              ratio_logits: (batch, num_ratio_classes) tensor of logits for ratio classification
        """
        x = x.to(next(self.parameters()).device)
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]

        shared_feats = self.shared(cls_token)
        coords = self.coord_head(shared_feats)  # x, y, width
        ratio_logits = self.ratio_head(shared_feats)  # ratio class logits

        return coords, ratio_logits
    
    def predict(self, x: torch.Tensor):
        """
        Make predictions with the model and convert to usable format.
        
        Args:
            x: Tensor of shape (batch, 3, 224, 224), already normalized.
        Returns:
            List of tuples, each containing:
              (x, y, width, ratio_class) where ratio_class is an integer 0 to (num_ratio_classes-1)
        """
        coords, ratio_logits = self.forward(x)
        ratio_preds = torch.argmax(ratio_logits, dim=1)
        
        # Convert to list of tuples
        results = []
        for i in range(coords.shape[0]):
            x_val, y_val, width_val = coords[i].cpu().numpy().tolist()
            ratio_class = ratio_preds[i].item()
            results.append((x_val, y_val, width_val, ratio_class))
        
        return results

