import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video_models

class AudioVisualFusion(nn.Module):
    def __init__(self):
        super(AudioVisualFusion, self).__init__()

        self.visual_encoder = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        self.visual_encoder.fc = nn.Identity()

        for name, param in self.visual_encoder.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.audio_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        old_conv_weight = self.audio_encoder.conv1.weight.clone()
        self.audio_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.audio_encoder.conv1.weight = nn.Parameter(old_conv_weight.mean(dim=1, keepdim=True))

        self.audio_encoder.fc = nn.Identity()

        for name, param in self.audio_encoder.named_parameters():
            if "layer4" in name or "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1)
        )

    def forward(self, visual, audio):
        v_features = self.visual_encoder(visual)
        a_features = self.audio_encoder(audio)
        
        combined = torch.cat((v_features, a_features), dim=1)
        logits = self.fusion_head(combined)
        return logits