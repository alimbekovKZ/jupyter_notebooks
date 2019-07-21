import torch
import torch.nn as nn
import torchvision


class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

    def forward(self, x):
        return x


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel, self).__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = Empty()
        self.conv1d = nn.Conv1d(
            in_channels=5,
            out_channels=1,
            kernel_size=(3),
            stride=(2),
            padding=(1))
        self.fc = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        vectors = []
        for i in range(0, x.shape[1]):
            v = self.encoder(x[:, i])
            v = v.reshape(v.size(0), -1)
            vectors.append(v)
        vectors = torch.stack(vectors)
        vectors = vectors.permute((1, 0, 2))
        vectors = self.conv1d(vectors)
        x = self.fc(vectors)
        return x
