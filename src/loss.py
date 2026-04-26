import torch
import torch.nn as nn

class AudioVisualLoss(nn.Module):
    def __init__(self):
        super(AudioVisualLoss, self).__init__()
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def forward(self, predictions, labels):
        labels = labels.view(-1, 1).float()
        loss = self.criterion(predictions, labels)
        return loss