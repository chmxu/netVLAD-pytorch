import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, feature_size, num_classes, hidden_size=512):
        super(QNet, self).__init__()
        self.input_size = feature_size * 2
        self.output_size = num_classes + 1
        self.fc1 = nn.Linear(self.input_size, hidden_size)
	self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.output_size)

    def forward(self, model_input):
        fc1 = F.relu(self.fc1(model_input))
	fc2 = F.relu(self.fc2(fc1))
        return self.fc3(fc2)


