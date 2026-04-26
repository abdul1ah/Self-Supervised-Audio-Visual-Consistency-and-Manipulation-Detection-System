import torch
import torch.nn as nn
import torchvision.models as models


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(VisualEncoder, self).__init__()
        self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioEncoder, self).__init__()
        self.backbone = models.resnet18(weights=None) 
        
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):

        return self.backbone(x)


class AudioVisualFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioVisualFusion, self).__init__()
        
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        
        fusion_dim = embed_dim * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, visual_input, audio_input):
        v_features = self.visual_encoder(visual_input)
        a_features = self.audio_encoder(audio_input)
        
        diff_features = torch.abs(v_features - a_features)
        
        fused_features = torch.cat((v_features, a_features, diff_features), dim=1)

        logits = self.classifier(fused_features)
        
        return logits