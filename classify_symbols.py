import torch
import torch.nn as nn
import torch.nn.functional as f


class SymbolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=2592, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv1(x)
        x = f.relu(x)
        x = f.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.dropout(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(batchsz, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = f.relu(x)
        return x


def classify(images):
    device = torch.device("cuda" if images.is_cuda else "cpu")
    model = SymbolClassifier()
    model = model.to(device)
    model.load_state_dict(torch.load("weights.pkl", map_location=torch.device(device)))
    model.eval()
    with torch.no_grad():
        output = model(images)
    predicted_classes = torch.argmax(output, 1)
    return predicted_classes
