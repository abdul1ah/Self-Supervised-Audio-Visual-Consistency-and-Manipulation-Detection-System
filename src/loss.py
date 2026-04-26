import torch
import torch.nn as nn

class AudioVisualLoss(nn.Module):
    def __init__(self):
        super(AudioVisualLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predictions, labels):
        """
        Args:
            predictions: The raw logit output from the Fusion Module. Shape: [Batch, 1]
            labels: The ground truth labels from the DataLoader. Shape: [Batch]
        """
        labels = labels.view(-1, 1).float()
  
        loss = self.criterion(predictions, labels)
        
        return loss