import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import math

# get current device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'using device: {device}')


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s  # Scaling factor
        self.m = m  # Margin

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)


        output = (label * phi) + ((1.0 - label) * cosine)
        output *= self.s

        return output


class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(ArcFaceModel, self).__init__()
        # Load a pre-trained ResNet model and remove the last fully connected layer
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)     
           
        # Initialize the ArcFace layer
        self.arcface = ArcFace(in_features=feature_dim, out_features=num_classes)

    def forward(self, x, labels=None):
        x = self.backbone(x)

        if labels is not None:
            x = self.arcface(x, labels)

        return x