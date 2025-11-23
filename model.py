import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class StegoCNN(nn.Module):
    def __init__(self, pretrained=True, freeze_resnet=False):
        super(StegoCNN, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final fully connected layer
        # ResNet-50 has 2048 features before the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze ResNet backbone if requested
        if freeze_resnet:
            for param in self.features.parameters():
                param.requires_grad = False

        # Add custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
