import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import math

from arcface import ArcFaceModel
from dataset import DorsalDataset


# get current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'using device: {device}')
    

images = []
labels = []

imdir = './dorsal_db_v1/'


for folder in os.listdir(imdir):
    for file in os.listdir(imdir + folder):
        img = cv2.imread(imdir + folder + '/' + file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(int(folder) - 1)
    

images = torch.from_numpy(np.stack(images))
labels = torch.from_numpy(np.array(labels))

images = torch.stack([images, images, images], dim=1).to(device)
labels_onehot = F.one_hot(labels.long()).to(device)


dataset = DorsalDataset(images, labels_onehot)
trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

dataloader = DataLoader(trainset, batch_size=32, shuffle=True)


# 276 set of dorsal hands, each with seven images
num_classes = 276


# evaluation
# using test set
from sklearn.neighbors import KNeighborsClassifier


trainloader = DataLoader(trainset, batch_size=32)


model = ArcFaceModel(num_classes=num_classes)
model.load_state_dict(torch.load('./50_epochs_v1.pth', map_location=device))


feature_extractor = model.backbone
feature_extractor.eval()

features = []
true_label = []

with torch.no_grad():
    for images, labels in trainloader:
        outs = feature_extractor(images.float()).cpu()
        for i in range(outs.size(0)):
            features.append(outs[i].numpy())
            true_label.append(labels[i].cpu().numpy())


features = np.asarray(features)
true_label = np.asarray(true_label)


# KNearestNeighbors
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(features, true_label)


testloader = DataLoader(testset, batch_size=32)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outs = feature_extractor(images.float()).cpu().numpy()
        pred = knn.predict(outs)
        
        converted_pred = np.argmax(pred, axis=1)
        converted_labels = np.argmax(labels.detach().cpu().numpy(), axis=1)

        correct += (converted_pred == converted_labels).sum()
        total += labels.shape[0]
    
    
print(correct)
print(total)
acc = correct / total * 100
print(f'accuracy: {acc}%')