import torch
from torch import nn
from .ghostnet import ghostnet, GhostNet


def get_model(num_classes):
    model: GhostNet = ghostnet(num_classes=num_classes)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('./model/pretrained/ghostnet_1x-f97d70db.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


if __name__ == "__main__":
    net = ghostnet()
    net.load_state_dict(torch.load('./model/pretrained/ghostnet_1x-f97d70db.pth'))