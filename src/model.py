import torch
import torch.nn as nn


class CNN3d(nn.Module):
    def __init__(self):
        super(CNN3d, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=6, padding=0),
            nn.ELU(alpha=1.0),
            nn.AvgPool3d(2),
            # nn.Dropout(p=0.2),

            nn.Conv3d(32, 32, kernel_size=4, padding=0),
            nn.ELU(alpha=1.0),
            nn.AvgPool3d(2),
            # nn.Dropout(p=0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7 * 7, 128),
            nn.Sigmoid(),
            # nn.Tanh(),
            # nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            # elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            #     print('Initialized', m, 'constant')
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shape_embedding = self.cnn1(x)
        shape_embedding = shape_embedding.view(shape_embedding.size()[0], -1)
        return self.fc1(shape_embedding)


class ReachabilityPredictor(nn.Module):
    def __init__(self):
        super(ReachabilityPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(6),
            nn.Linear(6, 64),
            nn.Tanh(),
            # nn.LeakyReLU(0.3),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)
