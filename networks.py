import torch
import torch.nn as nn

class Fea_encoder(nn.Module):
    def __init__(self, out_dim):  # 接收输出维度参数
        super(Fea_encoder, self).__init__()
        self.Net1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),  # Batch Normalization
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=out_dim),
            nn.Dropout(0.4)
        )

    def forward(self, input):
        input = input.unsqueeze(1)
        x = self.Net1(input)
        return x

class Attribute_encoder(nn.Module):
    def __init__(self, in_dim):
        super(Attribute_encoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)  # 批归一化
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 13)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, dimension_fea):
        super(SiameseNetwork, self).__init__()
        self.Fea_encoder = Fea_encoder(out_dim=dimension_fea)
        self.Attribute_encoder = Attribute_encoder(in_dim=dimension_fea)

    def forward_once(self, x):
        output = self.Fea_encoder(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        output_attri = self.Attribute_encoder(output1)
        return output1, output2, output3, output_attri