import torch
import torch.nn as nn
from torchvision.models import resnet152

def load_torch_model():

    model = resnet152(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features,2),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(torch.load("pneumonia_resnet152.pth",map_location="cpu"))

    model.eval()

    return model