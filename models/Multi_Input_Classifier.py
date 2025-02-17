import torch
import torch.nn as nn
import torch.nn.functional as F

# THIS IS AN EXAMPLE USING LATE-FUSION. NOT TO BE USED!!!!

class MultiInputClassifier(nn.Module):
    def __init__(self, num_stat_features, num_classes, image_size):
        super(MultiInputClassifier, self).__init__()
        
        # CNN Branch for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv Layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),  # Flatten for fully connected layers
        )
        
        # Fully connected branch for global statistics
        self.stats_fc = nn.Sequential(
            nn.Linear(num_stat_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined fully connected layers
        self.combined_fc = nn.Sequential(
            nn.Linear(64 * (image_size // 4) * (image_size // 4) + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, stats):
        # Forward pass for image through CNN
        image_features = self.cnn(image)
        
        # Forward pass for statistics
        stats_features = self.stats_fc(stats)
        
        # Concatenate features
        combined_features = torch.cat((image_features, stats_features), dim=1)
        
        # Final prediction
        output = self.combined_fc(combined_features)
        return output
