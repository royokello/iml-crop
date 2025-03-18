import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor

class ViTCropper(nn.Module):
    """
    ViT-based model for image cropping.
    Uses a pre-trained Vision Transformer as the backbone and adds a dual head for regression
    and classification to predict:
    - Normalized crop coordinates and height (x1, y1, height) 
    - Aspect ratio class (0 for 1:1, 1 for 2:3, 2 for 3:2)
    """
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTCropper, self).__init__()
        
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        
        # Get hidden size from the ViT config
        hidden_size = self.vit.config.hidden_size  # 768 for base model
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Regression head for coordinates and height (x1, y1, height)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
            nn.Sigmoid()  # Output normalized coordinates [0,1]
        )
        
        # Classification head for aspect ratio (0=1:1, 1=2:3, 2=3:2)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # 3 class outputs (no activation, will use softmax in loss)
        )
        
        # Pre-trained model feature extractor for preprocessing
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name)
    
    def preprocess(self, images):
        """
        Preprocess images to match ViT input requirements.
        For batched processing during training.
        """
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # Convert batched tensor to list of PIL Images for feature extractor
            # Assuming images is [batch_size, channels, height, width] and normalized
            images = [img.permute(1, 2, 0).cpu().numpy() for img in images]
            
        # Use the feature extractor to prepare inputs
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        return inputs
        
    def forward(self, x):
        """
        Forward pass through the ViT model and dual heads.
        
        Args:
            x: Input tensor or preprocessed inputs
        
        Returns:
            In training mode: Tuple of (coords, ratio_logits) 
                - coords: Tensor of shape (batch_size, 3) with normalized crop coordinates (x1, y1, height)
                - ratio_logits: Tensor of shape (batch_size, 3) with raw logits for aspect ratio classification
            In inference mode: Tensor of shape (batch_size, 4) with combined [x1, y1, height, ratio]
        """
        # Check if input is already preprocessed
        if isinstance(x, dict) and 'pixel_values' in x:
            inputs = x
        else:
            # Preprocess the input
            inputs = self.preprocess(x)
            
        # Move inputs to the same device as the model
        pixel_values = inputs['pixel_values'].to(next(self.parameters()).device)
        
        # ViT forward pass
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use the [CLS] token representation for prediction
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Shared feature extraction
        shared_features = self.shared(cls_output)
        
        # Predict normalized coordinates and height
        coords = self.regressor(shared_features)
        
        # Predict aspect ratio class
        ratio_logits = self.classifier(shared_features)
        
        # During inference mode, combine outputs
        if not self.training:
            # Get the predicted ratio class (0, 1, or 2)
            ratio_class = torch.argmax(ratio_logits, dim=1).unsqueeze(1).float()
            
            # Combine coordinates and ratio class into a single tensor
            combined_output = torch.cat([coords, ratio_class], dim=1)
            return combined_output
        
        # In training mode, always return a tuple
        return coords, ratio_logits
