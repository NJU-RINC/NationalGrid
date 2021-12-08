import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ghostnet import ghostnet, GhostNet


class sim_ghost(nn.Module):
    def __init__(self, classid_list, pretrain=True):
        super().__init__()
        self.R = 1.0
        self.num_classes = len(classid_list)
        self.latent_dim = 960
        self.prototypes = nn.parameter.Parameter(
            torch.zeros(self.num_classes, self.latent_dim))
        self.ghostnet: GhostNet = ghostnet(num_classes=self.num_classes)

        if pretrain:
            model_dict = self.ghostnet.state_dict()
            pretrained_dict = torch.load('./model/pretrained/ghostnet_1x-f97d70db.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier' not in k}
            model_dict.update(pretrained_dict)
            self.ghostnet.load_state_dict(model_dict)
        nn.init.normal_(self.prototypes)
        self.classid_list = classid_list

    def get_feature(self, x):
        x = self.ghostnet.features(x)
        x = self.ghostnet.squeeze(x)
        x = x.view(x.size(0), -1)

        return x

    def get_output(self, features):
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        features = F.normalize(features, p=2, dim=1)

        dists = torch.cdist(features, prototypes, p=2) ** 2

        logits = 10 * (self.R - dists)
        return logits, dists

    def forward(self, x):
        features = self.get_feature(x)
        logits, dists = self.get_output(features)
        return logits, dists

    def get_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='sum')
        return bce
    
    def estimate_threshold(self, dists, labels, std_coeff=1):
        self.classwise_thresholds = []
        classwise_dists = []
        for i in range(self.num_classes):
            classwise_dists.append([])

        for i, val in enumerate(dists):
            if self.classid_list.count(labels[i]) > 0:
                id_index = self.classid_list.index(labels[i])
                mindist = np.min(dists[i])
                if dists[i, id_index] == mindist:
                    classwise_dists[id_index].append(dists[i, id_index])

        for dists in classwise_dists:
            if len(dists) == 0:
                self.classwise_thresholds.append(self.R)
            else:
                mean = np.mean(dists)
                std = np.std(dists)
                self.classwise_thresholds.append(mean + std_coeff * std)
        return self.classwise_thresholds
    
    def predict(self, x):
        features = self.get_feature(x)
        outs, dists = self.get_output(features)
        thresholds = torch.from_numpy(np.array(self.classwise_thresholds)).cuda()

        probs = torch.sigmoid(10 * (thresholds - dists))
        # probs = torch.sigmoid(thresholds - dists)

        minDists, minIndexes = torch.min(dists, 1)
        prediction = torch.zeros([minIndexes.shape[0]], requires_grad=False).cuda()

        for i in range(minIndexes.shape[0]):
            prediction[i] = self.classid_list[minIndexes[i]]
            if minDists[i] >= self.classwise_thresholds[minIndexes[i]]:
                prediction[i] = -1

        return prediction.long(), probs.detach().cpu().numpy(), dists.detach().cpu().numpy()


# if __name__ == '__main__':
#     import sys, os
#     sys.path.append(os.getcwd())
#     from torchvision import transforms
#     from utils.data import NationalGridDataset
#     from torch.utils.data import DataLoader
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     )
#     train_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ColorJitter(hue=.05, saturation=.05),
#         transforms.ToTensor(),
#         normalize,
#     ])

#     train_set = NationalGridDataset("./data/drawback_labels.txt")
#     train_set.transform = train_transform

#     train_loader = DataLoader(train_set, batch_size=2)
#     model = sim_ghost()
#     model.eval()
#     for idx, batch in enumerate(train_loader):
#         input = batch['image']
#         outputs, dists = model(input)
#         break