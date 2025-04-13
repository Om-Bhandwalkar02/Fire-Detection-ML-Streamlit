import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 128 * 128, 2)
        )

    def forward(self, x):
        return self.model(x)